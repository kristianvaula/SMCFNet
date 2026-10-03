#!/usr/bin/env python3
"""Precompute Cuadrado MCM flow per frame for DSEC-Det sequences.

Why a precompute path instead of live in-graph MCM: Cuadrado's checkpoint was
pretrained at 21 event bins, while the R3 detector backbone we pair STTP with
expects 5-bin voxels at the event stream stem. We can't change
``data.event_bins`` at runtime without breaking either Cuadrado's input conv
or the R3 stem. The clean fix is to run Cuadrado offline at its native bin
count and cache its 2-channel flow at 480x640, then feed it to STTP via
``data.precomputed_flow_dir`` (the same path E-RAFT and Kristian baselines
use). The detector backbone keeps its 5-bin contract; flow is sourced from
the cache and overrides ``mcm._last_flow`` at yolox.py:152.

Output is a per-sequence HDF5 file under --output-dir, each containing a
``flow`` dataset of shape ``(num_frames, 2, H, W)`` in fp16 pixels/frame
(flow_scaling=640 already applied). Frame 0 is zeros since Cuadrado consumes
the event window ending at frame ``t`` and there is no prior pair for the
first frame. Files carry the ``checkpoint_sha`` / ``frame_count`` attrs that
``PrecomputedFlowProvider`` requires.

Invocation via ``python -m scripts.precompute_cuadrado_flow`` (SLURM-friendly).
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from smcfnet.config.config import Config, load_config_from_yaml, update_config_from_dict
from smcfnet.dataset.dsec.dsec_lightweight_mapper import DSECLightweightMapper
from smcfnet.dataset.dsec.dsec_sequence_index import (
    DSECSequenceIndexDataset,
    build_dsec_sequence_index_datasets,
)
from smcfnet.model.ofe import create_flow_estimator
from smcfnet.utils.event_utils import events_to_voxel_grid_torch

logger = logging.getLogger(__name__)

OVERWRITE_CHOICES = ("never", "if-sha-mismatch", "always")

# DSEC event sensor native resolution (distorted events; rectified live in a
# different coordinate system but the mapper returns whatever the underlying
# h5 stores in raw pixels).
DSEC_SENSOR_H = 480
DSEC_SENSOR_W = 640


def _checkpoint_fingerprint(path: Path) -> str:
    """SHA256 over the first and last MB of a checkpoint — same scheme as the
    E-RAFT cache, so downstream consumers can detect a stale checkpoint swap."""
    sz = path.stat().st_size
    chunk = 1024 * 1024
    h = hashlib.sha256()
    with path.open("rb") as f:
        h.update(f.read(min(chunk, sz)))
        if sz > chunk:
            f.seek(max(0, sz - chunk))
            h.update(f.read(chunk))
    return h.hexdigest()


def _build_estimator(cfg: Config, device: torch.device):
    """Construct the Cuadrado estimator from the run config and load weights."""
    mcm = cfg.model.mcm
    if mcm.architecture != "cuadrado_unet":
        raise ValueError(
            f"This script targets architecture='cuadrado_unet'; got {mcm.architecture!r}. "
            f"Use scripts/precompute_eraft_flow.py for E-RAFT."
        )

    neuron_kwargs = dict(mcm.neuron_kwargs or {})
    neuron_kwargs.setdefault("alpha", mcm.lif_alpha)
    neuron_kwargs.setdefault("theta", mcm.lif_theta)

    estimator = create_flow_estimator(
        mcm.architecture,
        num_bins=int(cfg.data.event_bins),
        encoder_channels=list(mcm.encoder_channels),
        num_res_blocks=int(mcm.num_res_blocks),
        alpha=float(mcm.lif_alpha),
        theta=float(mcm.lif_theta),
        mode=str(mcm.mode),
        neuron_type=str(mcm.neuron_type),
        neuron_kwargs=neuron_kwargs,
    )

    ckpt_path = Path(mcm.pretrained_weights)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Cuadrado checkpoint not found: {ckpt_path}")
    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sd = ck.get("mcm", ck)
    estimator_sd = {
        k.replace("estimator.", "", 1): v
        for k, v in sd.items()
        if k.startswith("estimator.")
    }
    missing, unexpected = estimator.load_state_dict(estimator_sd, strict=False)
    if missing or unexpected:
        logger.warning(
            "Cuadrado state_dict loaded with missing=%d unexpected=%d",
            len(missing), len(unexpected),
        )
        if missing[:5]:
            logger.warning("  sample missing: %s", missing[:5])
        if unexpected[:5]:
            logger.warning("  sample unexpected: %s", unexpected[:5])

    estimator.eval()
    estimator.to(device)
    return estimator, ckpt_path


def _build_mapper(cfg: Config, per_sequence, fullres: bool) -> DSECLightweightMapper:
    return DSECLightweightMapper(
        per_sequence,
        load_events=True,
        load_tracks=False,
        load_images=False,
        build_targets=False,
        build_future_targets=False,
        image_height=int(cfg.data.image_height),
        image_width=int(cfg.data.image_width),
        prefer_distorted=not fullres,
    )


def _events_to_voxel(events: torch.Tensor, num_bins: int, h: int, w: int,
                     device: torch.device) -> torch.Tensor:
    """Voxelize the per-frame event window for Cuadrado's expected input."""
    events = events.to(device, non_blocking=True)
    return events_to_voxel_grid_torch(
        events,
        num_bins=num_bins,
        width=w,
        height=h,
        source_width=DSEC_SENSOR_W,
        source_height=DSEC_SENSOR_H,
    )


def _should_skip(
    out_path: Path,
    *,
    expected_sha: str,
    expected_frames: int,
    policy: str,
) -> bool:
    if not out_path.exists():
        return False
    if policy == "always":
        return False
    try:
        import h5py
        with h5py.File(str(out_path), "r") as f:
            sha = f.attrs.get("checkpoint_sha", None)
            if isinstance(sha, bytes):
                sha = sha.decode("utf-8")
            fc = int(f.attrs.get("frame_count", -1))
            dset_frames = int(f["flow"].shape[0]) if "flow" in f else -1
    except Exception as exc:
        logger.warning("Could not inspect existing %s: %s -- will overwrite", out_path, exc)
        return False
    if policy == "never":
        return True
    return sha == expected_sha and fc == expected_frames and dset_frames == expected_frames


def _write_sequence(
    *,
    estimator,
    mapper: DSECLightweightMapper,
    seq_ds: DSECSequenceIndexDataset,
    cfg: Config,
    device: torch.device,
    out_path: Path,
    checkpoint_sha: str,
) -> None:
    import h5py

    seq_name = seq_ds.sequence
    num_frames = len(seq_ds)
    img_h = int(cfg.data.image_height)
    img_w = int(cfg.data.image_width)
    num_bins = int(cfg.data.event_bins)
    flow_scaling = float(cfg.model.mcm.flow_scaling)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    t0 = time.time()
    first_flow_logged = False
    with h5py.File(str(tmp_path), "w") as f:
        dset = f.create_dataset(
            "flow",
            shape=(num_frames, 2, img_h, img_w),
            dtype=np.float32,
            chunks=(1, 2, img_h, img_w),
        )
        # Frame 0: no prior pair, write zeros (consistent with E-RAFT cache).
        dset[0] = np.zeros((2, img_h, img_w), dtype=np.float32)

        for frame_idx in range(num_frames):
            if frame_idx == 0:
                continue

            estimator.reset_state()
            sample = mapper(seq_ds[frame_idx])
            events = sample["events"]
            if not isinstance(events, torch.Tensor):
                events = torch.as_tensor(events)
            voxel = _events_to_voxel(events, num_bins, img_h, img_w, device)

            with torch.inference_mode():
                flow = estimator(voxel.unsqueeze(0))
            if isinstance(flow, tuple):
                flow = flow[0]
            # Apply the same scaling the runtime MCM applies before warping, so
            # the cached flow is directly in pixel units at frame dt.
            flow = (flow * flow_scaling)[0].detach().cpu().float().numpy()

            if flow.shape[-2:] != (img_h, img_w):
                raise RuntimeError(
                    f"Cuadrado flow shape {flow.shape[-2:]} != ({img_h}, {img_w}). "
                    f"Check config image_height/image_width."
                )

            if not first_flow_logged:
                mag = np.linalg.norm(flow, axis=0)
                logger.info(
                    "  %s first-flow |flow| mean=%.4f p90=%.4f max=%.4f",
                    seq_name, float(mag.mean()), float(np.percentile(mag, 90)), float(mag.max()),
                )
                first_flow_logged = True

            dset[frame_idx] = flow.astype(np.float32)

        f.attrs["checkpoint_sha"] = checkpoint_sha
        f.attrs["dsec_sequence"] = seq_name
        f.attrs["frame_count"] = int(num_frames)
        f.attrs["num_bins"] = int(num_bins)
        f.attrs["flow_scaling"] = float(flow_scaling)
        f.attrs["resolution"] = f"{img_h}x{img_w}"
        f.attrs["architecture"] = "cuadrado_unet"
        f.attrs["written_at"] = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"

    tmp_path.replace(out_path)

    elapsed = time.time() - t0
    fps = num_frames / elapsed if elapsed > 0 else 0.0
    logger.info(
        "  %s: wrote %d frames to %s in %.1fs (%.2f fps)",
        seq_name, num_frames, out_path, elapsed, fps,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Precompute Cuadrado MCM optical flow per DSEC-Det frame -> HDF5 cache.",
    )
    ap.add_argument("--config", required=True,
                    help=("YAML supplying data.event_bins (must equal Cuadrado's pretrained "
                          "bins, e.g. 21) and data.image_height/width (e.g. 480x640). "
                          "Also reads model.mcm.{architecture,pretrained_weights,flow_scaling,...}."))
    ap.add_argument("--output-dir", default="outputs/cache/cuadrado_flow_480x640",
                    help="Where to write <seq>.h5 files.")
    ap.add_argument("--splits", nargs="+", default=["train", "test"],
                    help="DSEC splits to precompute. Default: train test.")
    ap.add_argument("--sequences", nargs="*", default=None,
                    help="Restrict to these sequence names within each split.")
    ap.add_argument("--overwrite", choices=OVERWRITE_CHOICES, default="if-sha-mismatch",
                    help="Resume policy when <seq>.h5 already exists.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
        force=True,  # override any handler an imported module already installed
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = Config()
    update_config_from_dict(cfg, load_config_from_yaml(args.config))

    img_h = int(cfg.data.image_height)
    img_w = int(cfg.data.image_width)
    if (img_h, img_w) != (480, 640):
        logger.warning(
            "Config resolution is %dx%d, not 480x640. STTP currently expects 480x640.",
            img_h, img_w,
        )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info("Device: %s", device)
    logger.info("Architecture: %s | bins: %d | flow_scaling: %.3f",
                cfg.model.mcm.architecture, cfg.data.event_bins, cfg.model.mcm.flow_scaling)

    estimator, ckpt_path = _build_estimator(cfg, device)
    checkpoint_sha = _checkpoint_fingerprint(ckpt_path)
    logger.info("Cuadrado checkpoint: %s (sha256 head+tail=%s)", ckpt_path, checkpoint_sha)

    out_root = Path(args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    logger.info("Output dir: %s", out_root)

    run_meta = {
        "checkpoint": str(ckpt_path),
        "checkpoint_sha": checkpoint_sha,
        "config": args.config,
        "architecture": cfg.model.mcm.architecture,
        "num_bins": int(cfg.data.event_bins),
        "image_height": img_h,
        "image_width": img_w,
        "flow_scaling": float(cfg.model.mcm.flow_scaling),
        "event_window_ms": float(cfg.data.event_window_ms),
        "splits": list(args.splits),
        "sequences_filter": list(args.sequences) if args.sequences else None,
        "written_at": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
    }
    with (out_root / "run_meta.json").open("w") as f:
        json.dump(run_meta, f, indent=2)

    dsec_cfg = Path(str(getattr(cfg.data, "dataset_config", "")))
    if not dsec_cfg.is_absolute() and not dsec_cfg.exists():
        candidate = (Path(__file__).resolve().parents[1] / dsec_cfg).resolve()
        if candidate.exists():
            dsec_cfg = candidate
    fullres = (img_h, img_w) == (1080, 1440)

    total_sequences = 0
    skipped_sequences = 0
    processed_sequences = 0
    for split in args.splits:
        logger.info("=== Split: %s ===", split)
        try:
            per_sequence = build_dsec_sequence_index_datasets(
                dsec_yaml_path=dsec_cfg,
                split=split,
                dt_ms=float(cfg.data.event_window_ms),
                sequences=list(args.sequences) if args.sequences else None,
                use_rectified_events=fullres,
                use_rectified_tracks=fullres,
            )
        except Exception as exc:
            logger.error("Split %s failed to build: %s", split, exc)
            continue
        if not per_sequence:
            logger.warning("No sequences in split '%s'", split)
            continue
        mapper = _build_mapper(cfg, per_sequence, fullres=fullres)

        for seq_ds in per_sequence:
            total_sequences += 1
            out_path = out_root / f"{seq_ds.sequence}.h5"

            if _should_skip(
                out_path,
                expected_sha=checkpoint_sha,
                expected_frames=len(seq_ds),
                policy=args.overwrite,
            ):
                skipped_sequences += 1
                logger.info(
                    "  %s: up-to-date at %s, skipping (policy=%s)",
                    seq_ds.sequence, out_path, args.overwrite,
                )
                continue

            logger.info("Processing %s (%d frames) -> %s",
                        seq_ds.sequence, len(seq_ds), out_path)
            _write_sequence(
                estimator=estimator,
                mapper=mapper,
                seq_ds=seq_ds,
                cfg=cfg,
                device=device,
                out_path=out_path,
                checkpoint_sha=checkpoint_sha,
            )
            processed_sequences += 1

    logger.info(
        "Done. processed=%d skipped=%d total=%d output=%s",
        processed_sequences, skipped_sequences, total_sequences, out_root,
    )


if __name__ == "__main__":
    main()
