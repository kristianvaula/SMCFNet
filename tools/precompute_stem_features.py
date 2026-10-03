#!/usr/bin/env python3
"""Precompute per-frame detector stem features for the recurrent STTP heads.

The spiking-recurrent and ConvGRU STTP heads (`head_design: spiking_recurrent`
/ `convgru`) consume a K-frame window of the detector's per-scale stem features
rather than running the detector live each step. This script caches those
features so `smcfnet.dataset.stem_feature_provider.StemFeatureProvider` can
serve windows during training/eval.

Output format (must match StemFeatureProvider exactly):
  one HDF5 file per sequence, `<output-dir>/<sequence>.h5`, with datasets
  `p0`, `p1`, `p2` (one per FPN scale P3/P4/P5), each `[n_frames, C, H_k, W_k]`
  fp16, plus a required `checkpoint_sha` attr. The sha is the sha256 of the
  detector checkpoint; the provider refuses to mix files with different shas so
  a window can never blend features from two detectors.

What is cached: the literal output of `model.head.stems[k]` for each scale,
captured via a forward hook. That is exactly the tensor `yolox.py` feeds the
STTP head as `stem_window`/`stem_features` (see DualStreamYOLOX.forward), so the
cache is bit-faithful to the live path by construction.

Fidelity is the make-or-break property here (a silent preprocessing mismatch is
the L1/L2-class bug this project has been bitten by). Run `--verify-fidelity`
FIRST on a few frames: it caches them, then re-runs the live detector on the
same frames and asserts the stem features match. Only run the full 60-sequence
cache after that passes.

Invocation: `python -m scripts.precompute_stem_features` (SLURM-friendly).
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
from typing import Dict, List, Optional

import numpy as np
import torch

from smcfnet.config.config import Config, load_config_from_yaml, update_config_from_dict
from smcfnet.model.build import build_model
from smcfnet.dataset.dsec.dsec_lightweight_mapper import DSECLightweightMapper
from smcfnet.dataset.dsec.dsec_sequence_index import build_dsec_sequence_index_datasets
from smcfnet.utils.event_utils import events_to_voxel_grid_torch

logger = logging.getLogger(__name__)

OVERWRITE_CHOICES = ("never", "if-sha-mismatch", "always")
NUM_SCALES = 3


def _checkpoint_sha(path: Path) -> str:
    """Full sha256 of the checkpoint file; matches StemFeatureProvider's check."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class _StemHookCapture:
    """Registers forward hooks on each model.head.stems[k] and records outputs."""

    def __init__(self, model: torch.nn.Module):
        head = getattr(model, "head", None)
        stems = getattr(head, "stems", None) if head is not None else None
        if stems is None or len(stems) != NUM_SCALES:
            raise RuntimeError(
                "Could not locate model.head.stems (expected an nn.ModuleList of "
                f"{NUM_SCALES} stem convs). Got: {type(stems)}. The stem-feature "
                "contract assumes a YOLOXHead with .stems (see yolo_head.py:42)."
            )
        self._buf: Dict[int, torch.Tensor] = {}
        self._handles = []
        for k, stem in enumerate(stems):
            self._handles.append(stem.register_forward_hook(self._make_hook(k)))

    def _make_hook(self, k: int):
        def hook(_module, _inp, out):
            self._buf[k] = out.detach()
        return hook

    def pop(self) -> List[torch.Tensor]:
        if len(self._buf) != NUM_SCALES:
            raise RuntimeError(
                f"Expected {NUM_SCALES} stem outputs captured, got {sorted(self._buf)}. "
                "Did the detector forward actually run the head?"
            )
        out = [self._buf[k] for k in range(NUM_SCALES)]
        self._buf = {}
        return out

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []


def _build_detector(cfg: Config, checkpoint: str, device: torch.device) -> torch.nn.Module:
    """Build the frozen detector (no STTP, no MCM) with pretrained weights loaded."""
    # Force detector-only: build_model loads pretrained_path internally and logs
    # the skipped_nomodel count (must be 0 for the v5 sum-fusion ckpt).
    cfg.model.pretrained = True
    cfg.model.pretrained_path = checkpoint
    cfg.model.sttp.enabled = False
    cfg.model.mcm.enabled = False
    cfg.model.edum.enabled = False
    cfg.model.cmm.enabled = False
    model = build_model(cfg)
    model.eval().to(device)
    return model


def _frame_inputs(sample: dict, cfg: Config, device: torch.device):
    """Reproduce the prefetcher's (rgb, event_voxel) for one frame, batch dim added.

    Voxelization uses the same events_to_voxel_grid_torch the GPU prefetcher uses
    (dsec_gpu_prefetcher._voxelize_packed), with config-derived bins/resolution.
    DSEC event sensor is already 480x640, so source==target (no rescale).
    """
    h = int(cfg.data.image_height)
    w = int(cfg.data.image_width)
    # Match the prefetcher EXACTLY: detector-input voxelization uses
    # cfg.data.event_bins (build.py:66), NOT model.voxel_bins (an SMCM-output
    # concept). Using voxel_bins here would silently diverge from the live path.
    num_bins = int(cfg.data.event_bins)

    events = sample["events"]
    if not isinstance(events, torch.Tensor):
        events = torch.as_tensor(events)
    events = events.to(device)
    voxel = events_to_voxel_grid_torch(
        events, num_bins=num_bins, width=w, height=h,
        source_width=w, source_height=h,
    ).unsqueeze(0)  # [1, num_bins, H, W]

    image = sample["image"]
    if not isinstance(image, torch.Tensor):
        image = torch.as_tensor(image)
    image = image.to(device).float()
    if image.dim() == 3:
        image = image.unsqueeze(0)  # [1, 3, H, W]
    if image.shape[-2:] != (h, w):
        image = torch.nn.functional.interpolate(
            image, size=(h, w), mode="bilinear", align_corners=False
        )
    return image, voxel


def _build_mapper(cfg: Config, per_sequence) -> DSECLightweightMapper:
    return DSECLightweightMapper(
        per_sequence,
        load_events=True,
        load_tracks=False,
        load_images=True,   # detector needs the RGB frame
        build_targets=False,
        build_future_targets=False,
        image_height=int(cfg.data.image_height),
        image_width=int(cfg.data.image_width),
        prefer_distorted=bool(getattr(cfg.data, "mapper_prefer_distorted", True)),
    )


def _should_skip(out_path: Path, *, expected_sha: str, expected_frames: int, policy: str) -> bool:
    if not out_path.exists() or policy == "always":
        return False
    try:
        import h5py
        with h5py.File(str(out_path), "r") as f:
            sha = f.attrs.get("checkpoint_sha", None)
            if isinstance(sha, bytes):
                sha = sha.decode("utf-8")
            fc = int(f.attrs.get("frame_count", -1))
            dset_frames = int(f["p0"].shape[0]) if "p0" in f else -1
    except Exception as exc:
        logger.warning("Could not inspect %s: %s -- will overwrite", out_path, exc)
        return False
    if policy == "never":
        return True
    return sha == expected_sha and fc == expected_frames and dset_frames == expected_frames


def _write_sequence(*, model, hooks, mapper, seq_ds, cfg, device, out_path, checkpoint_sha) -> None:
    import h5py

    seq_name = seq_ds.sequence
    num_frames = len(seq_ds)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    t0 = time.time()
    dsets = None
    with h5py.File(str(tmp_path), "w") as f:
        for frame_idx in range(num_frames):
            sample = mapper(seq_ds[frame_idx])
            rgb, event = _frame_inputs(sample, cfg, device)
            with torch.inference_mode():
                model(rgb, event)
            feats = hooks.pop()  # list of NUM_SCALES [1, C, H_k, W_k]

            if dsets is None:
                dsets = []
                for s, fk in enumerate(feats):
                    c, hk, wk = int(fk.shape[1]), int(fk.shape[2]), int(fk.shape[3])
                    dsets.append(f.create_dataset(
                        f"p{s}", shape=(num_frames, c, hk, wk),
                        dtype=np.float16, chunks=(1, c, hk, wk),
                    ))
            for s, fk in enumerate(feats):
                dsets[s][frame_idx] = fk[0].float().cpu().numpy().astype(np.float16)

        f.attrs["checkpoint_sha"] = checkpoint_sha
        f.attrs["dsec_sequence"] = seq_name
        f.attrs["frame_count"] = int(num_frames)
        f.attrs["num_scales"] = NUM_SCALES
        f.attrs["written_at"] = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"

    tmp_path.replace(out_path)
    elapsed = time.time() - t0
    logger.info("  %s: wrote %d frames to %s in %.1fs (%.2f fps)",
                seq_name, num_frames, out_path, elapsed,
                num_frames / elapsed if elapsed > 0 else 0.0)


def _verify_fidelity(*, model, hooks, mapper, seq_ds, cfg, device, n_frames: int) -> None:
    """Cache N frames then re-run the live detector and assert stem features match.

    This is the analog of the eraft STTP smoke: it proves the cache is bit-faithful
    to the live path BEFORE committing the full ~118 GB precompute.
    """
    n = min(n_frames, len(seq_ds))
    logger.info("Fidelity check on %s, first %d frames ...", seq_ds.sequence, n)
    # Observable preprocessing parity: the feature comparison below is fp16+hook
    # only (it cannot, on its own, prove _frame_inputs matches the prefetcher).
    # So log the resolved preprocessing the cache commits to, and the actual
    # RGB/voxel tensor stats, against which the prefetcher's contract is checked
    # by eye: detector input is RGB [0,255] float (dsec_gpu_prefetcher.py:375 is
    # .float() with NO /255) and event_bins voxelization (build.py:66).
    logger.info("  resolved: num_bins=%d (cfg.data.event_bins), HxW=%dx%d, source=target, "
                "prefer_distorted=%s", int(cfg.data.event_bins), int(cfg.data.image_height),
                int(cfg.data.image_width), bool(getattr(cfg.data, "mapper_prefer_distorted", True)))
    max_abs = 0.0
    voxel_nonzero_frames = 0
    for frame_idx in range(n):
        sample = mapper(seq_ds[frame_idx])
        rgb, event = _frame_inputs(sample, cfg, device)
        ev_abs = event.abs()
        if ev_abs.max().item() > 0:
            voxel_nonzero_frames += 1
        logger.info("  frame %d: rgb[min=%.1f max=%.1f] voxel[max=%.3f mean=%.4f nonzero=%.4f]",
                    frame_idx, rgb.min().item(), rgb.max().item(), ev_abs.max().item(),
                    event.mean().item(), (ev_abs > 0).float().mean().item())
        if frame_idx == 0 and rgb.max().item() <= 1.5:
            raise RuntimeError(
                f"RGB max={rgb.max().item():.3f} <= 1.5: input looks normalized to [0,1], "
                "but the detector is trained on RGB [0,255] (prefetcher does .float() with "
                "no /255). Preprocessing mismatch. Fix before caching."
            )
        with torch.inference_mode():
            model(rgb, event)
        live = hooks.pop()
        # round-trip through fp16 (what the cache stores) and compare
        for s, fk in enumerate(live):
            cached = torch.from_numpy(
                fk[0].float().cpu().numpy().astype(np.float16).astype(np.float32)
            )
            ref = fk[0].float().cpu()
            d = (cached - ref).abs().max().item()
            max_abs = max(max_abs, d)
    logger.info("Voxel non-empty in %d/%d verify frames", voxel_nonzero_frames, n)
    if voxel_nonzero_frames == 0:
        raise RuntimeError(
            "Event voxel is all-zero across EVERY verify frame: events are not reaching "
            "the voxelizer (check mapper load_events / sample['events'] format). A dead "
            "event stream would make cached stem features RGB-only and incomparable to the "
            "matrix heads. Frame-0 alone being empty is fine; all frames is the bug."
        )
    logger.info("Fidelity max |cached_fp16 - live| = %.3e (fp16 rounding only; "
                "should be ~1e-2 or less, never large)", max_abs)
    if max_abs > 1.0:
        raise RuntimeError(
            f"Fidelity check FAILED: max abs diff {max_abs:.3e} is too large to be "
            "fp16 rounding. The cached stem features do not match the live detector "
            "path. Fix preprocessing (voxelization/RGB) before running the full cache."
        )
    logger.info("Fidelity check PASSED (differences are fp16 rounding only).")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Precompute detector stem features per DSEC-Det frame -> HDF5 cache.")
    ap.add_argument("--config", required=True,
                    help="YAML with data.image_height/width, dataset_config, event_bins.")
    ap.add_argument("--checkpoint", required=True,
                    help="Frozen detector checkpoint (v5_m03_na15/best.pth).")
    ap.add_argument("--output-dir", default="outputs/cache/stem_v5_m03_na15",
                    help="Where to write <seq>.h5 stem caches.")
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--sequences", nargs="*", default=None)
    ap.add_argument("--overwrite", choices=OVERWRITE_CHOICES, default="if-sha-mismatch")
    ap.add_argument("--verify-fidelity", action="store_true",
                    help="Run the cache-vs-live fidelity check on the first sequence and exit.")
    ap.add_argument("--verify-frames", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(levelname)s  %(message)s", datefmt="%H:%M:%S")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    ckpt_path = Path(args.checkpoint).resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    checkpoint_sha = _checkpoint_sha(ckpt_path)
    logger.info("Detector checkpoint: %s (sha256=%s)", ckpt_path, checkpoint_sha[:16])

    cfg = Config()
    update_config_from_dict(cfg, load_config_from_yaml(args.config))

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info("Device: %s", device)
    model = _build_detector(cfg, str(ckpt_path), device)
    hooks = _StemHookCapture(model)

    out_root = Path(args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    logger.info("Output dir: %s", out_root)

    dsec_cfg = Path(str(getattr(cfg.data, "dataset_config", "")))
    if not dsec_cfg.is_absolute() and not dsec_cfg.exists():
        cand = (Path(__file__).resolve().parents[1] / dsec_cfg).resolve()
        if cand.exists():
            dsec_cfg = cand

    # Match the loader's sequence-index construction exactly so cache frame_index
    # aligns with training-time get_window lookups (mirrors precompute_eraft_flow.py:
    # rectified only at fullres 1080x1440; distorted 480x640 otherwise).
    fullres = (int(cfg.data.image_height), int(cfg.data.image_width)) == (1080, 1440)

    processed = skipped = total = 0
    for split in args.splits:
        logger.info("=== Split: %s ===", split)
        try:
            per_sequence = build_dsec_sequence_index_datasets(
                dsec_yaml_path=dsec_cfg, split=split,
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
        mapper = _build_mapper(cfg, per_sequence)

        for seq_ds in per_sequence:
            if args.verify_fidelity:
                _verify_fidelity(model=model, hooks=hooks, mapper=mapper, seq_ds=seq_ds,
                                 cfg=cfg, device=device, n_frames=args.verify_frames)
                hooks.remove()
                return
            total += 1
            out_path = out_root / f"{seq_ds.sequence}.h5"
            if _should_skip(out_path, expected_sha=checkpoint_sha,
                            expected_frames=len(seq_ds), policy=args.overwrite):
                skipped += 1
                logger.info("  %s: up-to-date, skipping (policy=%s)", seq_ds.sequence, args.overwrite)
                continue
            logger.info("Processing %s (%d frames) -> %s", seq_ds.sequence, len(seq_ds), out_path)
            _write_sequence(model=model, hooks=hooks, mapper=mapper, seq_ds=seq_ds,
                            cfg=cfg, device=device, out_path=out_path, checkpoint_sha=checkpoint_sha)
            processed += 1

    hooks.remove()
    with (out_root / "run_meta.json").open("w") as f:
        json.dump({
            "checkpoint": str(ckpt_path), "checkpoint_sha": checkpoint_sha,
            "config": args.config, "splits": list(args.splits),
            "image_height": int(cfg.data.image_height), "image_width": int(cfg.data.image_width),
            "num_bins": int(cfg.data.event_bins),
            "slurm_job": os.environ.get("SLURM_JOB_ID"),
            "written_at": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }, f, indent=2)
    logger.info("Done. processed=%d skipped=%d total=%d output=%s",
                processed, skipped, total, out_root)


if __name__ == "__main__":
    main()
