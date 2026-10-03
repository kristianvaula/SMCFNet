#!/usr/bin/env python3
"""
Evaluate a trained checkpoint: detection mAP + module-specific metrics.

Produces a clean summary table that works for debug and production runs.
Optionally writes a JSON results file for automated comparisons.

Usage:
  # Single checkpoint
  python tools/eval_checkpoint.py \
    --config src/smcfnet/config/runs/mcm-simple/ecm_s_debug.yaml \
    --checkpoint outputs/mcm-simple/ecm_s_debug/epoch_5.pth

  # Compare multiple runs (prints side-by-side table)
  python tools/eval_checkpoint.py \
    --config src/smcfnet/config/runs/mcm-simple/ecm_s_debug.yaml \
    --checkpoint outputs/mcm-simple/ecm_s_debug/epoch_5.pth \
    --config src/smcfnet/config/runs/edum/edum_s_debug.yaml \
    --checkpoint outputs/edum/edum_s_debug/epoch_5.pth
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Ensure project is importable
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from smcfnet.config.config import Config, load_config_from_yaml, update_config_from_dict
from smcfnet.model.build import build_model, build_dsec_loader_cfg
from smcfnet.utils.validation import postprocess_predictions, postprocess_detr_predictions, calculate_map
from smcfnet.dataset.dsec.dsec_gpu_prefetcher import DSECGPUPrefetcher

logger = logging.getLogger(__name__)

# DSEC-Det class names (must match class ID ordering in tracks.npy)
DSEC_CLASSES = (
    "pedestrian", "rider", "car", "bus",
    "truck", "bicycle", "motorcycle", "train",
)


# ── MCM metrics ──────────────────────────────────────────────────────
def compute_mcm_metrics(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    num_bins: int,
    height: int,
    width: int,
    max_batches: int = 0,
) -> Dict[str, float]:
    """Compute MCM-specific metrics over the validation set.

    Returns:
        contrast_before: Mean IWE variance before correction
        contrast_after:  Mean IWE variance after correction
        contrast_ratio:  after / before  (>1 means MCM helps)
        flow_mag_mean:   Mean flow magnitude (pixels)
        flow_mag_std:    Std of flow magnitude
        flow_mag_max:    Max flow magnitude
    """
    backbone = model.backbone if hasattr(model, "backbone") else None
    if backbone is None or not getattr(backbone, "use_mcm", False):
        return {}

    mcm = backbone.mcm
    prefetch = DSECGPUPrefetcher(
        loader, device=device, num_bins=num_bins,
        height=height, width=width, require_targets=False,
    )

    vars_before: List[float] = []
    vars_after: List[float] = []
    flow_mags: List[torch.Tensor] = []

    batch_idx = 0
    with torch.no_grad():
        while True:
            try:
                batch = prefetch.next()
            except StopIteration:
                break

            voxel = batch["events_voxel"]  # (B, bins, H, W)

            # IWE variance BEFORE correction
            iwe_before = voxel.sum(dim=1)  # (B, H, W)
            var_before = iwe_before.reshape(iwe_before.shape[0], -1).var(dim=1)
            vars_before.append(var_before.cpu())

            # Run MCM
            corrected, _ = mcm(voxel)

            # IWE variance AFTER correction
            iwe_after = corrected.sum(dim=1)
            var_after = iwe_after.reshape(iwe_after.shape[0], -1).var(dim=1)
            vars_after.append(var_after.cpu())

            # Flow statistics
            flow = mcm._last_flow  # (B, 2, H, W)
            mag = flow.norm(dim=1)  # (B, H, W)
            flow_mags.append(mag.cpu())

            batch_idx += 1
            if max_batches > 0 and batch_idx >= max_batches:
                break

    if len(vars_before) == 0:
        return {}

    vars_before_t = torch.cat(vars_before)
    vars_after_t = torch.cat(vars_after)
    flow_mags_t = torch.cat(flow_mags)

    mean_before = float(vars_before_t.mean())
    mean_after = float(vars_after_t.mean())

    return {
        "mcm/contrast_before": mean_before,
        "mcm/contrast_after": mean_after,
        "mcm/contrast_ratio": mean_after / max(mean_before, 1e-8),
        "mcm/flow_mag_mean": float(flow_mags_t.mean()),
        "mcm/flow_mag_std": float(flow_mags_t.std()),
        "mcm/flow_mag_max": float(flow_mags_t.max()),
    }


# ── Detection mAP ─────────────────────────────────────────────────────
def compute_detection_metrics(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    num_bins: int,
    height: int,
    width: int,
    num_classes: int,
    conf_threshold: float = 0.001,
    nms_threshold: float = 0.65,
    head_type: str = "yolox",
    num_select: int = 300,
) -> Dict[str, float]:
    """Run full validation and return mAP metrics."""
    prefetch = DSECGPUPrefetcher(
        loader, device=device, num_bins=num_bins,
        height=height, width=width, require_targets=True,
    )

    all_predictions: List[Optional[torch.Tensor]] = []
    all_targets: List[torch.Tensor] = []

    batch_idx = 0
    with torch.no_grad():
        while True:
            try:
                batch = prefetch.next()
            except StopIteration:
                break

            images = batch["image"]
            voxels = batch["events_voxel"]
            targets = batch.get("targets")

            outputs = model(images, voxels)
            if head_type == "detr":
                preds = postprocess_detr_predictions(
                    outputs,
                    img_h=int(images.shape[-2]),
                    img_w=int(images.shape[-1]),
                    num_select=num_select,
                    conf_thre=conf_threshold,
                )
            else:
                preds = postprocess_predictions(
                    outputs,
                    num_classes=num_classes,
                    conf_thre=conf_threshold,
                    nms_thre=nms_threshold,
                )

            bsz = int(targets.shape[0]) if targets is not None else 0
            for i in range(bsz):
                pred_i = preds[i] if i < len(preds) else None
                tgt_i = _filter_targets(targets[i]) if targets is not None else torch.zeros(0, 5)
                all_predictions.append(pred_i.cpu() if pred_i is not None else None)
                all_targets.append(tgt_i.cpu() if isinstance(tgt_i, torch.Tensor) else tgt_i)

            batch_idx += 1
            if (batch_idx) % 10 == 0:
                logger.info("  eval batch %d/%d", batch_idx, len(loader))

    if not all_targets:
        return {"mAP": 0.0, "mAP50": 0.0, "mAP75": 0.0}

    metrics = calculate_map(
        all_predictions,
        all_targets,
        num_classes=num_classes,
        class_names=DSEC_CLASSES,
    )
    return metrics


def _filter_targets(t: torch.Tensor) -> torch.Tensor:
    """Filter out padding rows from target tensor."""
    if t is None:
        return torch.zeros((0, 5), dtype=torch.float32)
    if not isinstance(t, torch.Tensor):
        t = torch.as_tensor(t)
    if t.numel() == 0:
        return t.reshape(0, 5).to(torch.float32)
    tt = t.to(torch.float32)
    if tt.ndim != 2 or tt.shape[-1] != 5:
        tt = tt.reshape(-1, 5)
    valid = (tt[:, 3] > 0) & (tt[:, 4] > 0)
    valid = valid & (tt[:, 0] >= 0)
    return tt[valid]


# ── Model loading ─────────────────────────────────────────────────────
def load_model(config: Config, checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """Build model from config, load checkpoint weights, set to eval mode."""
    model = build_model(config).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and ("model" in ckpt or "state_dict" in ckpt):
        state = ckpt.get("model", ckpt.get("state_dict"))
    else:
        state = ckpt

    # Strip DDP "module." prefix if present
    if any(k.startswith("module.") for k in state):
        state = {k.removeprefix("module."): v for k, v in state.items()}

    model.load_state_dict(state)
    model.eval()
    return model


# ── Pretty printing ───────────────────────────────────────────────────
def print_single(name: str, metrics: Dict[str, float]) -> None:
    """Print metrics for a single run."""
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")

    # Detection
    det_keys = ["mAP", "mAP50", "mAP75"]
    print("\n  Detection:")
    for k in det_keys:
        if k in metrics:
            print(f"    {k:20s} {metrics[k]:8.4f}")

    # Per-class AP50
    cls_keys = sorted(k for k in metrics if k.startswith("AP50/"))
    if cls_keys:
        print("\n  Per-class AP50:")
        for k in cls_keys:
            print(f"    {k:20s} {metrics[k]:8.4f}")

    # MCM metrics
    mcm_keys = sorted(k for k in metrics if k.startswith("mcm/"))
    if mcm_keys:
        print("\n  MCM module:")
        for k in mcm_keys:
            print(f"    {k:20s} {metrics[k]:8.4f}")

    print()


def print_comparison(runs: List[Tuple[str, Dict[str, float]]]) -> None:
    """Print side-by-side comparison table."""
    if len(runs) == 1:
        print_single(runs[0][0], runs[0][1])
        return

    # Collect all keys
    all_keys: List[str] = []
    seen = set()
    for _, m in runs:
        for k in m:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    # Sort: detection first, then per-class, then module-specific
    def key_sort(k):
        if k in ("mAP", "mAP50", "mAP75"):
            return (0, k)
        if k.startswith("AP50/"):
            return (1, k)
        if k.startswith("mcm/"):
            return (2, k)
        return (3, k)

    all_keys.sort(key=key_sort)

    names = [n for n, _ in runs]
    col_w = max(12, max(len(n) for n in names) + 2)
    label_w = 24

    # Header
    print(f"\n{'=' * (label_w + col_w * len(runs) + 4)}")
    print(f"  {'Metric':<{label_w}}", end="")
    for name in names:
        print(f"{name:>{col_w}}", end="")
    print()
    print(f"  {'-' * label_w}", end="")
    for _ in names:
        print(f"{'':>{col_w - 10}}{'-' * 10}", end="")
    print()

    # Rows
    for k in all_keys:
        print(f"  {k:<{label_w}}", end="")
        for _, m in runs:
            v = m.get(k)
            if v is not None:
                print(f"{v:>{col_w}.4f}", end="")
            else:
                print(f"{'—':>{col_w}}", end="")
        print()

    print(f"{'=' * (label_w + col_w * len(runs) + 4)}\n")


# ── Main ──────────────────────────────────────────────────────────────
def evaluate_single(
    config_path: str,
    checkpoint_path: str,
    device: torch.device,
    max_mcm_batches: int = 0,
) -> Dict[str, float]:
    """Evaluate a single config+checkpoint pair. Returns all metrics."""
    # Build config
    cfg = Config()
    raw = load_config_from_yaml(config_path)
    update_config_from_dict(cfg, raw)

    # Force data loader settings for eval
    cfg.data.mapper_load_images = True
    cfg.data.mapper_load_tracks = True
    cfg.data.mapper_build_targets = True

    num_bins = int(cfg.data.event_bins)
    height = int(cfg.data.image_height)
    width = int(cfg.data.image_width)
    num_classes = int(cfg.model.num_classes)

    # Load model
    logger.info("Loading model from %s", checkpoint_path)
    model = load_model(cfg, checkpoint_path, device)

    eval_split = str(getattr(cfg.eval, "split", "test") or "test")
    eval_sequences = list(getattr(cfg.eval, "sequences", []) or [])
    eval_dataset_config = getattr(cfg.eval, "dataset_config", None)

    # Build validation/test loader from the same eval contract as training.
    logger.info(
        "Building evaluation loader (split=%s, explicit_sequences=%d)...",
        eval_split,
        len(eval_sequences),
    )
    val_loader = build_dsec_loader_cfg(
        cfg,
        split=eval_split,
        is_train=False,
        sequences=eval_sequences,
        dataset_config_override=eval_dataset_config,
    )
    logger.info("Evaluation set: %d batches", len(val_loader))

    all_metrics: Dict[str, float] = {}

    # 1. MCM metrics (if MCM is enabled)
    backbone = model.backbone if hasattr(model, "backbone") else None
    has_mcm = backbone is not None and getattr(backbone, "use_mcm", False)

    if has_mcm:
        logger.info("Computing MCM metrics...")
        val_loader_mcm = build_dsec_loader_cfg(
            cfg,
            split=eval_split,
            is_train=False,
            sequences=eval_sequences,
            dataset_config_override=eval_dataset_config,
        )
        mcm_metrics = compute_mcm_metrics(
            model, val_loader_mcm, device,
            num_bins=num_bins, height=height, width=width,
            max_batches=max_mcm_batches,
        )
        all_metrics.update(mcm_metrics)

    # 2. Detection mAP
    head_type = getattr(cfg.model, "head_type", "yolox")
    num_select = int(getattr(cfg.model.detr_head, "num_select", 300)) if head_type == "detr" else 300
    logger.info("Computing detection mAP (head_type=%s)...", head_type)
    det_metrics = compute_detection_metrics(
        model, val_loader, device,
        num_bins=num_bins, height=height, width=width,
        num_classes=num_classes,
        conf_threshold=cfg.eval.conf_threshold,
        nms_threshold=cfg.eval.nms_threshold,
        head_type=head_type,
        num_select=num_select,
    )
    all_metrics.update(det_metrics)

    return all_metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate checkpoint(s) with detection mAP + module metrics",
    )
    parser.add_argument(
        "--config", action="append", required=True,
        help="Config YAML (repeat for multi-run comparison)",
    )
    parser.add_argument(
        "--checkpoint", action="append", required=True,
        help="Checkpoint path (repeat, must match --config count)",
    )
    parser.add_argument(
        "--name", action="append", default=None,
        help="Display name for each run (optional, defaults to experiment_name)",
    )
    parser.add_argument(
        "--output-json", type=str, default=None,
        help="Write results to JSON file",
    )
    parser.add_argument(
        "--max-mcm-batches", type=int, default=0,
        help="Limit MCM metric computation to N batches (0=all)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device (cuda or cpu)",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
    )

    args = parse_args()

    if len(args.config) != len(args.checkpoint):
        print(f"ERROR: Got {len(args.config)} configs but {len(args.checkpoint)} checkpoints")
        sys.exit(1)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    runs: List[Tuple[str, Dict[str, float]]] = []

    for idx, (cfg_path, ckpt_path) in enumerate(zip(args.config, args.checkpoint)):
        # Determine display name
        if args.name and idx < len(args.name):
            name = args.name[idx]
        else:
            raw = load_config_from_yaml(cfg_path)
            name = raw.get("log", {}).get("experiment_name", Path(ckpt_path).stem)

        logger.info("=" * 60)
        logger.info("Evaluating: %s", name)
        logger.info("  config:     %s", cfg_path)
        logger.info("  checkpoint: %s", ckpt_path)
        logger.info("=" * 60)

        t0 = time.time()
        metrics = evaluate_single(cfg_path, ckpt_path, device, args.max_mcm_batches)
        elapsed = time.time() - t0
        metrics["eval_time_s"] = elapsed
        logger.info("Evaluation took %.1fs", elapsed)

        runs.append((name, metrics))

    # Print results
    print_comparison(runs)

    # Optionally save JSON
    if args.output_json:
        out = {name: m for name, m in runs}
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
