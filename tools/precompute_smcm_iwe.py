#!/usr/bin/env python3
"""Precompute detector-facing SMCM IWEs per DSEC frame.

This is the offline, sequence-ordered alternative to running frozen streaming
SMCM inside shuffled detector batches. The output is one HDF5 file per
sequence, each containing the exact detector-facing IWE tensor consumed by EDUM
and the event backbone.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

os.environ.setdefault("BLOSC_NTHREADS", "1")
os.environ.setdefault("OPENCV_OPENCL_RUNTIME", "disabled")

from smcfnet.config.config import Config, load_config_from_yaml, update_config_from_dict
from smcfnet.dataset.dsec import DSECLightweightMapper, build_dsec_sequence_index_datasets
from smcfnet.model.build import (
    _build_mcm,
    _is_orig_rectified_events_resolution,
    _normalize_rectified_events_resolution,
    _resolve_mcm_event_resolution,
    _resolve_rectified_events_filename,
    _should_use_rectified_events,
)
from smcfnet.model.mcm.weight_utils import load_mcm_weights_into_module
from smcfnet.utils.event_utils import events_to_cuadrado_histogram_torch, events_to_voxel_grid_torch


logger = logging.getLogger("precompute_smcm_iwe")
OVERWRITE_CHOICES = ("never", "if-compatible", "always")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_path(path_raw: str | Path) -> Path:
    path = Path(path_raw)
    if path.exists():
        return path
    if not path.is_absolute():
        project_root = Path(__file__).resolve().parents[1]
        candidate = project_root / path
        if candidate.exists():
            return candidate
    return path


def _resolve_device(device_raw: str) -> torch.device:
    requested = str(device_raw or "cuda").strip().lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        logger.warning("CUDA requested but unavailable; falling back to CPU")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("%s requested but CUDA is unavailable; falling back to CPU", requested)
        return torch.device("cpu")
    return device


def _renderer_version(cfg: Config) -> str:
    output = getattr(cfg.model.mcm, "output", None)
    payload = {
        "renderer": "LiuRGBIWEEventOutput",
        "output_representation": str(getattr(cfg.model.mcm, "output_representation", "auto")),
        "type": str(getattr(output, "type", "")),
        "t_ref": float(getattr(output, "t_ref", 1.0)),
        "iwe_channels": str(getattr(output, "iwe_channels", "liu_rgb")),
        "iwe_normalize": str(getattr(output, "iwe_normalize", "p99")),
        "iwe_scale": float(getattr(output, "iwe_scale", 1.0)),
        "iwe_splat": str(getattr(output, "iwe_splat", "bilinear")),
        "iwe_align_to_rgb": str(getattr(output, "iwe_align_to_rgb", "auto")),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _load_cfg(path: Path) -> Config:
    cfg = Config()
    update_config_from_dict(cfg, load_config_from_yaml(str(path)))
    return cfg


def _sequence_filter(
    seqs: Sequence,
    *,
    include: Optional[Sequence[str]],
    exclude: Optional[Sequence[str]],
) -> List:
    include_set = {str(s) for s in include} if include else None
    exclude_set = {str(s) for s in exclude} if exclude else set()
    out = []
    for ds in seqs:
        name = str(getattr(ds, "sequence"))
        if include_set is not None and name not in include_set:
            continue
        if name in exclude_set:
            continue
        out.append(ds)
    return out


def _build_sequence_datasets(cfg: Config, *, split: str, sequences: Optional[Sequence[str]]):
    dsec_yaml = _resolve_path(getattr(cfg.data, "dataset_config"))
    image_height = int(getattr(cfg.data, "image_height", 1080))
    image_width = int(getattr(cfg.data, "image_width", 1440))
    prefer_distorted = bool(getattr(cfg.data, "mapper_prefer_distorted", False))
    rectified_event_resolution = _normalize_rectified_events_resolution(cfg)
    edum_enabled = bool(getattr(getattr(cfg.model, "edum", None), "enabled", False))
    use_rectified_events = _should_use_rectified_events(
        mapper_load_events=True,
        prefer_distorted=prefer_distorted,
        image_height=image_height,
        image_width=image_width,
        rectified_event_resolution=rectified_event_resolution,
    )
    rectified_events_filename = "rectified_events.h5"
    if use_rectified_events:
        rectified_events_filename = _resolve_rectified_events_filename(
            rectified_event_resolution=rectified_event_resolution,
            edum_enabled=edum_enabled,
        )
    if not use_rectified_events and not prefer_distorted and _is_orig_rectified_events_resolution(rectified_event_resolution):
        use_rectified_events = True
        rectified_events_filename = "rectified_events_orig_res.h5"

    return build_dsec_sequence_index_datasets(
        dsec_yaml_path=dsec_yaml,
        split=str(split),
        sequences=sequences,
        dt_ms=float(cfg.data.event_window_ms),
        use_rectified_events=use_rectified_events,
        use_rectified_tracks=False,
        rectified_events_filename=rectified_events_filename,
    )


def _read_coordinate_frame(path: Path) -> str:
    try:
        import h5py
        with h5py.File(str(path), "r") as fh:
            value = fh.attrs.get("coordinate_frame", None)
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            if value is not None:
                return str(value)
    except Exception:
        pass
    if path.name == "rectified_events_orig_res.h5":
        return "rectified_rgb_orig_res"
    if path.name == "rectified_events.h5":
        return "rectified_rgb_full"
    return "event_camera_distorted"


def _should_skip(path: Path, *, checkpoint_sha: str, renderer_version: str, frame_count: int, policy: str) -> bool:
    if policy == "always" or not path.exists():
        return False
    if policy == "never":
        return True
    try:
        import h5py
        with h5py.File(str(path), "r") as fh:
            sha = fh.attrs.get("checkpoint_sha", None)
            renderer = fh.attrs.get("renderer_version", None)
            if isinstance(sha, bytes):
                sha = sha.decode("utf-8")
            if isinstance(renderer, bytes):
                renderer = renderer.decode("utf-8")
            n = int(fh.attrs.get("frame_count", -1))
            dset_n = int(fh["iwe"].shape[0]) if "iwe" in fh else -1
    except Exception:
        return False
    return str(sha) == checkpoint_sha and str(renderer) == renderer_version and n == frame_count and dset_n == frame_count


def _encode_iwe(tensor: torch.Tensor, dtype_name: str) -> np.ndarray:
    arr = tensor.detach().cpu().float().numpy()
    dtype_name = str(dtype_name).lower()
    if dtype_name == "fp16":
        return arr.astype(np.float16)
    if dtype_name == "float32":
        return arr.astype(np.float32)
    if dtype_name == "uint8":
        return np.rint(np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    raise ValueError(f"Unsupported cache dtype {dtype_name!r}; expected fp16|float32|uint8")


def _time_window_in_event_units(
    events: torch.Tensor,
    t_start_us: torch.Tensor,
    t_end_us: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if events.numel() == 0:
        return None, None
    t0_us = t_start_us.to(device=events.device, dtype=torch.float32)
    t1_us = t_end_us.to(device=events.device, dtype=torch.float32)
    ts = events[:, 2].to(torch.float32)
    window_scale = max(float(t0_us.abs()), float(t1_us.abs()))
    if window_scale > 0.0 and float(ts.abs().median()) < window_scale * 0.1:
        return t0_us / 1000.0, t1_us / 1000.0
    return t0_us, t1_us


def _build_mcm_inputs(
    *,
    events: torch.Tensor,
    cfg: Config,
    event_h: int,
    event_w: int,
    t_start_us: torch.Tensor,
    t_end_us: torch.Tensor,
) -> tuple[torch.Tensor | None, Dict[str, torch.Tensor]]:
    estimator_representation = str(getattr(cfg.model.mcm, "estimator_representation", "voxel")).lower()
    if estimator_representation == "streaming_events":
        return None, {}

    bins = int(getattr(cfg.data, "event_bins", 5))
    voxel = events_to_voxel_grid_torch(
        events,
        num_bins=bins,
        width=int(event_w),
        height=int(event_h),
        source_width=int(event_w),
        source_height=int(event_h),
    ).unsqueeze(0)

    kwargs: Dict[str, torch.Tensor] = {}
    if estimator_representation == "cuadrado_histogram":
        t0, t1 = _time_window_in_event_units(events, t_start_us[0], t_end_us[0])
        histogram = events_to_cuadrado_histogram_torch(
            events,
            num_bins=bins,
            width=int(event_w),
            height=int(event_h),
            t_start=t0,
            t_end=t1,
            source_width=int(event_w),
            source_height=int(event_h),
        ).unsqueeze(0)
        kwargs["estimator_input"] = histogram
    elif estimator_representation not in {"voxel", "auto", ""}:
        raise ValueError(
            f"Unsupported non-streaming MCM estimator_representation={estimator_representation!r} "
            "for IWE precompute"
        )
    return voxel, kwargs


def _write_sequence(
    *,
    mcm: torch.nn.Module,
    mapper: DSECLightweightMapper,
    seq_ds,
    cfg: Config,
    device: torch.device,
    out_path: Path,
    dtype_name: str,
    checkpoint_sha: str,
    renderer_version: str,
    dataset_yaml_sha: str,
) -> Dict[str, object]:
    import h5py

    seq_name = str(seq_ds.sequence)
    num_frames = len(seq_ds)
    event_h, event_w = _resolve_mcm_event_resolution(cfg)
    channels = int(getattr(mcm, "output_channels", 3))
    np_dtype = np.uint8 if dtype_name == "uint8" else (np.float16 if dtype_name == "fp16" else np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    meta = seq_ds.meta
    coordinate_frame = _read_coordinate_frame(meta.events_h5_path)

    mcm.reset_state()
    with h5py.File(str(tmp_path), "w") as fh:
        dset = fh.create_dataset(
            "iwe",
            shape=(num_frames, channels, event_h, event_w),
            dtype=np_dtype,
            chunks=(1, channels, event_h, event_w),
        )
        if dtype_name == "uint8":
            fh.attrs["scale"] = 255.0
        else:
            fh.attrs["scale"] = 1.0

        for frame_index in range(num_frames):
            index_item = dict(seq_ds[frame_index])
            index_item["seq_idx"] = 0
            sample = mapper(index_item)
            events = torch.as_tensor(sample["events"], device=device).contiguous()
            offsets = torch.tensor([0, int(events.shape[0])], dtype=torch.int64, device=device)
            t_start_us = torch.tensor([int(sample["t_start_us"])], dtype=torch.int64, device=device)
            t_end_us = torch.tensor([int(sample["t_end_us"])], dtype=torch.int64, device=device)
            t_ref_us = torch.tensor([int(sample["t_ref_us"])], dtype=torch.int64, device=device)
            seq_idx = torch.tensor([0], dtype=torch.int64, device=device)
            frame_idx_t = torch.tensor([int(frame_index)], dtype=torch.int64, device=device)
            sequence_reset = torch.tensor([frame_index == 0], dtype=torch.bool, device=device)
            voxel, mcm_extra_kwargs = _build_mcm_inputs(
                events=events,
                cfg=cfg,
                event_h=int(event_h),
                event_w=int(event_w),
                t_start_us=t_start_us,
                t_end_us=t_end_us,
            )

            with torch.inference_mode(), torch.amp.autocast("cuda", enabled=False):
                corrected, _ = mcm(
                    voxel,
                    events=events,
                    events_offsets=offsets,
                    t_start_us=t_start_us,
                    t_end_us=t_end_us,
                    t_ref_us=t_ref_us,
                    seq_idx=seq_idx,
                    frame_index=frame_idx_t,
                    sequence_reset=sequence_reset,
                    **mcm_extra_kwargs,
                )
            if corrected.ndim != 4 or int(corrected.shape[0]) != 1:
                raise RuntimeError(f"Unexpected SMCM output shape for {seq_name}:{frame_index}: {tuple(corrected.shape)}")
            if tuple(corrected.shape[-2:]) != (event_h, event_w):
                raise RuntimeError(
                    f"SMCM output spatial shape {tuple(corrected.shape[-2:])} "
                    f"!= expected {(event_h, event_w)}"
                )
            dset[frame_index] = _encode_iwe(corrected[0], dtype_name)

            if (frame_index + 1) % 250 == 0:
                logger.info("  %s: %d/%d frames", seq_name, frame_index + 1, num_frames)

        fh.attrs["checkpoint_sha"] = checkpoint_sha
        fh.attrs["architecture"] = str(getattr(cfg.model.mcm, "architecture", ""))
        fh.attrs["renderer_version"] = renderer_version
        fh.attrs["dsec_sequence"] = seq_name
        fh.attrs["frame_count"] = int(num_frames)
        fh.attrs["source_events_file"] = str(meta.events_h5_path)
        fh.attrs["coordinate_frame"] = coordinate_frame
        fh.attrs["event_height"] = int(event_h)
        fh.attrs["event_width"] = int(event_w)
        fh.attrs["image_height"] = int(getattr(cfg.data, "image_height", 0))
        fh.attrs["image_width"] = int(getattr(cfg.data, "image_width", 0))
        fh.attrs["dtype"] = str(dtype_name)
        fh.attrs["dataset_yaml_sha"] = dataset_yaml_sha
        fh.attrs["written_at"] = _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"

    tmp_path.replace(out_path)
    return {
        "sequence": seq_name,
        "frame_count": int(num_frames),
        "path": str(out_path),
        "source_events_file": str(meta.events_h5_path),
        "coordinate_frame": coordinate_frame,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Precompute detector-facing SMCM IWEs into per-sequence HDF5 files.")
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--variant", default=None)
    ap.add_argument("--split", default="train", choices=("train", "val", "test"))
    ap.add_argument("--sequences", nargs="*", default=None)
    ap.add_argument("--exclude-sequences", nargs="*", default=None)
    ap.add_argument("--dtype", default="fp16", choices=("fp16", "float32", "uint8"))
    ap.add_argument("--overwrite", choices=OVERWRITE_CHOICES, default="if-compatible")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")

    cfg_path = _resolve_path(args.config)
    cfg = _load_cfg(cfg_path)
    checkpoint = args.checkpoint or getattr(cfg.model.mcm, "pretrained_weights", None)
    if checkpoint is None:
        raise ValueError("--checkpoint or model.mcm.pretrained_weights is required")
    checkpoint = _resolve_path(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"SMCM checkpoint not found: {checkpoint}")

    checkpoint_sha = _sha256_file(checkpoint)
    dataset_yaml = _resolve_path(getattr(cfg.data, "dataset_config"))
    dataset_yaml_sha = _sha256_file(dataset_yaml)
    renderer_version = _renderer_version(cfg)

    variant = args.variant or str(getattr(cfg.model.mcm, "architecture", "smcm"))
    checkpoint_id = checkpoint_sha[:16]
    output_dir = args.output_dir or Path("outputs/cache/smcm_iwe") / variant / checkpoint_id
    output_dir = _resolve_path(output_dir)

    device = _resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    cfg.model.mcm.enabled = True
    mcm = _build_mcm(cfg).to(device)
    load_mcm_weights_into_module(mcm, str(checkpoint), strict=False)
    mcm.eval()

    sequence_datasets = _build_sequence_datasets(cfg, split=args.split, sequences=args.sequences)
    sequence_datasets = _sequence_filter(
        sequence_datasets,
        include=args.sequences,
        exclude=args.exclude_sequences,
    )
    if not sequence_datasets:
        raise RuntimeError("No sequences selected for SMCM IWE precompute")

    logger.info(
        "Precomputing SMCM IWE: split=%s sequences=%d dtype=%s output=%s checkpoint_sha=%s renderer=%s",
        args.split, len(sequence_datasets), args.dtype, output_dir, checkpoint_sha[:12], renderer_version[:12],
    )

    summaries: List[Dict[str, object]] = []
    for seq_ds in sequence_datasets:
        seq_name = str(seq_ds.sequence)
        out_path = output_dir / f"{seq_name}.h5"
        if _should_skip(
            out_path,
            checkpoint_sha=checkpoint_sha,
            renderer_version=renderer_version,
            frame_count=len(seq_ds),
            policy=args.overwrite,
        ):
            logger.info("Skipping compatible existing cache: %s", out_path)
            summaries.append({"sequence": seq_name, "frame_count": len(seq_ds), "path": str(out_path), "skipped": True})
            continue

        mapper = DSECLightweightMapper(
            [seq_ds],
            load_events=True,
            load_tracks=False,
            load_images=False,
            build_targets=False,
            image_height=int(getattr(cfg.data, "image_height", 480)),
            image_width=int(getattr(cfg.data, "image_width", 640)),
            prefer_distorted=bool(getattr(cfg.data, "mapper_prefer_distorted", True)),
            event_timestamp_unit="us",
        )
        logger.info("Writing %s (%d frames)", seq_name, len(seq_ds))
        summaries.append(
            _write_sequence(
                mcm=mcm,
                mapper=mapper,
                seq_ds=seq_ds,
                cfg=cfg,
                device=device,
                out_path=out_path,
                dtype_name=args.dtype,
                checkpoint_sha=checkpoint_sha,
                renderer_version=renderer_version,
                dataset_yaml_sha=dataset_yaml_sha,
            )
        )

    manifest = {
        "variant": variant,
        "checkpoint": str(checkpoint),
        "checkpoint_sha": checkpoint_sha,
        "architecture": str(getattr(cfg.model.mcm, "architecture", "")),
        "renderer_version": renderer_version,
        "dataset_yaml": str(dataset_yaml),
        "dataset_yaml_sha": dataset_yaml_sha,
        "split": str(args.split),
        "dtype": str(args.dtype),
        "sequences": summaries,
        "written_at": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    logger.info("Wrote manifest: %s", output_dir / "manifest.json")


if __name__ == "__main__":
    main()
