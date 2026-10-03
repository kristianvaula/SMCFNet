#!/usr/bin/env python3
"""Qualitative evaluation of optical flow estimation.

Produces 2×2 panels for randomly sampled training frames:
  ┌──────────────────┬────────────────────────────┐
  │ (a) RGB image    │ (b) Flow (HSV) on RGB      │
  ├──────────────────┼────────────────────────────┤
  │ (c) Raw events   │ (d) Warped events on RGB   │
  │     on RGB       │    (motion-compensated)     │
  └──────────────────┴────────────────────────────┘

Usage:
  python -m tools.visualize_flow_qualitative \
      --config src/smcfnet/config/runs/mcm/idnet/pretrain_idnet.yaml \
      --checkpoint outputs/mcm/pretrain_idnet_debug/epoch_5.pth \
      --output-dir visualizations/flow_qualitative \
      --num-frames 5 \
      --seed 42
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from smcfnet.config.config import Config, load_config_from_yaml, update_config_from_dict
from smcfnet.dataset.dsec.dsec_sequence_index import build_dsec_sequence_index_datasets
from smcfnet.dataset.dsec.dsec_lightweight_mapper import DSECLightweightMapper
from smcfnet.dataset.peod import build_peod_sequence_index_datasets, PEODLightweightMapper
import yaml
from smcfnet.model.build import _build_mcm
from smcfnet.model.mcm.weight_utils import load_mcm_weights_into_module
from smcfnet.utils.event_utils import events_to_voxel_grid_torch
from smcfnet.utils.flow_viz import flow_to_rgb

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Overlay helpers
# ---------------------------------------------------------------------------

def _voxel_to_event_image(voxel: np.ndarray) -> np.ndarray:
    """Convert a (C, H, W) voxel grid to an RGB event image.

    Positive event accumulation → red, negative → blue.
    Returns (H, W, 3) uint8.
    """
    iwe = voxel.sum(axis=0)  # (H, W)
    pos = np.clip(iwe, 0, None)
    neg = np.clip(-iwe, 0, None)

    # Normalise to [0, 1]
    pos_max = pos.max() if pos.max() > 0 else 1.0
    neg_max = neg.max() if neg.max() > 0 else 1.0
    pos_n = pos / pos_max
    neg_n = neg / neg_max

    img = np.zeros((*iwe.shape, 3), dtype=np.float32)
    img[..., 0] = pos_n   # red = positive
    img[..., 2] = neg_n   # blue = negative
    return (img * 255).astype(np.uint8)


def _overlay(base_rgb: np.ndarray, overlay_rgb: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Alpha-blend *overlay_rgb* onto *base_rgb*.  Both (H, W, 3) uint8."""
    blended = base_rgb.astype(np.float32) * (1 - alpha) + overlay_rgb.astype(np.float32) * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def _overlay_events_on_rgb(
    rgb: np.ndarray,
    voxel: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Overlay event image on RGB, only where events are non-zero."""
    event_img = _voxel_to_event_image(voxel)  # (H, W, 3) uint8
    iwe = voxel.sum(axis=0)
    mask = np.abs(iwe) > 1e-6  # where events exist

    out = rgb.copy()
    for c in range(3):
        out[..., c] = np.where(
            mask,
            (rgb[..., c].astype(np.float32) * (1 - alpha) + event_img[..., c].astype(np.float32) * alpha),
            rgb[..., c],
        ).astype(np.uint8)
    return out


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_config(config_path: str) -> Config:
    yaml_dict = load_config_from_yaml(config_path)
    cfg = Config()
    update_config_from_dict(cfg, yaml_dict)
    return cfg


def _load_sample(mapper, index_ds, frame_idx: int, cfg: Config, device: torch.device):
    """Load a single sample and return (rgb_np, voxel_tensor, events, offsets, meta)."""
    index_item = index_ds[frame_idx]
    sample = mapper(index_item)

    events = sample["events"]  # (N, 4)
    events = events.to(device)
    events_offsets = torch.tensor([0, int(events.shape[0])], dtype=torch.long, device=device)
    num_bins = int(cfg.data.event_bins)
    h = int(cfg.data.image_height)
    w = int(cfg.data.image_width)

    # Build voxel grid
    voxel = events_to_voxel_grid_torch(events, num_bins=num_bins, width=w, height=h)  # (C, H, W)

    # Load image
    image = sample.get("image", None)
    if image is not None:
        # (3, H, W) float -> (H, W, 3) uint8
        rgb_np = image.permute(1, 2, 0).clamp(0, 255).byte().numpy()
    else:
        rgb_np = np.zeros((h, w, 3), dtype=np.uint8)

    return rgb_np, voxel.to(device), events, events_offsets, sample


# ---------------------------------------------------------------------------
# Panel construction
# ---------------------------------------------------------------------------

@torch.no_grad()
def make_panel(
    mcm: torch.nn.Module,
    rgb_np: np.ndarray,
    voxel: torch.Tensor,
    events: torch.Tensor,
    events_offsets: torch.Tensor,
    max_flow: Optional[float] = None,
    event_alpha: float = 0.6,
    flow_alpha: float = 0.6,
) -> np.ndarray:
    """Build a 2×2 panel for one frame.

    Args:
        mcm: MotionCompensationModule (on correct device, eval mode).
        rgb_np: (H, W, 3) uint8 RGB image.
        voxel: (C, H, W) voxel grid on device.
        max_flow: Cap for flow magnitude normalisation (None = auto).
        event_alpha: Blend factor for events overlay.
        flow_alpha: Blend factor for flow overlay.

    Returns:
        (2H, 2W, 3) uint8 composite image.
    """
    device = voxel.device
    voxel_batch = voxel.unsqueeze(0)  # (1, C, H, W)

    # Run MCM: estimate flow + warp events
    corrected, _ = mcm(voxel_batch, events=events, events_offsets=events_offsets)
    flow = mcm._last_flow[0]  # (2, H, W)

    # Numpy copies
    voxel_np = voxel.cpu().float().numpy()              # (C, H, W)
    corrected_np = corrected[0].cpu().float().numpy()   # (C, H, W)

    # (a) RGB image
    cell_a = rgb_np.copy()

    # (b) Flow HSV on RGB, with RGB base opacity 0.4.
    flow_rgb = flow_to_rgb(flow, max_flow=max_flow)  # (H, W, 3) uint8
    cell_b = _overlay(rgb_np, flow_rgb, alpha=flow_alpha)

    # (c) Raw events on RGB, with RGB base opacity 0.4.
    cell_c = _overlay_events_on_rgb(rgb_np, voxel_np, alpha=event_alpha)

    # (d) Warped (corrected) events on RGB, with RGB base opacity 0.4.
    cell_d = _overlay_events_on_rgb(rgb_np, corrected_np, alpha=event_alpha)

    # Assemble 2×2
    H, W = rgb_np.shape[:2]
    panel = np.zeros((2 * H, 2 * W, 3), dtype=np.uint8)
    panel[:H, :W] = cell_a
    panel[:H, W:] = cell_b
    panel[H:, :W] = cell_c
    panel[H:, W:] = cell_d

    return panel


def annotate_panel(panel: np.ndarray, labels: tuple[str, ...] = (
    "(a) RGB Image",
    "(b) Optical Flow (HSV) on RGB",
    "(c) Raw Events on RGB",
    "(d) MCM-Warped Events on RGB",
)) -> np.ndarray:
    """Draw cell labels on the 2×2 panel."""
    H = panel.shape[0] // 2
    W = panel.shape[1] // 2
    out = panel.copy()
    positions = [(10, 25), (W + 10, 25), (10, H + 25), (W + 10, H + 25)]
    for (x, y), text in zip(positions, labels):
        # Black outline for readability
        cv2.putText(out, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def visualize_flow_qualitative(
    config_path: str,
    checkpoint_path: str,
    output_dir: str,
    num_frames: int = 5,
    seed: int = 42,
    max_flow: Optional[float] = None,
    event_alpha: float = 0.5,
    flow_alpha: float = 0.5,
    device_str: str = "cuda",
) -> list[Path]:
    """Run qualitative flow visualisation and save panel PNGs.

    Returns list of saved PNG paths.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    cfg = _load_config(config_path)

    # ---- Build MCM and load checkpoint ----
    mcm = _build_mcm(cfg).to(device)
    load_mcm_weights_into_module(mcm, checkpoint_path, strict=False)
    mcm.eval()

    if max_flow is None:
        max_flow = getattr(cfg.model.mcm, "max_flow", None)

    # ---- Build dataset components (events + images) ----
    dataset_cfg_raw = str(getattr(cfg.data, "dataset_config", ""))
    dataset_cfg = Path(dataset_cfg_raw)
    if not dataset_cfg.is_absolute() and not dataset_cfg.exists():
        project_root = Path(__file__).resolve().parents[1]
        candidate = (project_root / dataset_cfg).resolve()
        if candidate.exists():
            dataset_cfg = candidate

    # Detect dataset type from YAML and select appropriate builders/mappers
    with dataset_cfg.open("r") as fh:
        ds_yaml = yaml.safe_load(fh)
    ds_name = str(ds_yaml.get("dataset", {}).get("name", "DSEC")).upper()

    if ds_name == "PEOD":
        per_sequence = build_peod_sequence_index_datasets(
            peod_yaml_path=dataset_cfg,
            split="train",
            dt_ms=float(cfg.data.event_window_ms),
        )
        mapper = PEODLightweightMapper(
            per_sequence,
            load_events=True,
            load_images=True,
            load_tracks=False,
            build_targets=False,
            image_height=int(cfg.data.image_height),
            image_width=int(cfg.data.image_width),
            prefer_distorted=False,
        )
    else:
        per_sequence = build_dsec_sequence_index_datasets(
            dsec_yaml_path=dataset_cfg,
            split="train",
            dt_ms=float(cfg.data.event_window_ms),
        )
        mapper = DSECLightweightMapper(
            per_sequence,
            load_events=True,
            load_images=True,
            load_tracks=False,
            build_targets=False,
            image_height=int(cfg.data.image_height),
            image_width=int(cfg.data.image_width),
            prefer_distorted=True,
        )

    if len(per_sequence) == 0:
        raise RuntimeError("No sequences found in training data")

    # Pick one sequence, then sample frames only from that sequence.
    chosen_seq = random.choice(per_sequence)
    n_seq_frames = len(chosen_seq)
    if n_seq_frames == 0:
        raise RuntimeError(f"Selected sequence {chosen_seq.sequence!r} has no frames")

    num_frames = min(num_frames, n_seq_frames)
    chosen_frame_indices = sorted(random.sample(range(n_seq_frames), num_frames))
    chosen = [(chosen_seq, fi) for fi in chosen_frame_indices]
    logger.info(
        "Selected sequence %s with %d total frames; sampling %d frame(s)",
        chosen_seq.sequence,
        n_seq_frames,
        num_frames,
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    mcm.reset_state()

    for i, (seq_ds, fi) in enumerate(chosen):
        seq_name = seq_ds.sequence
        logger.info("Processing frame %d/%d: %s frame %d", i + 1, num_frames, seq_name, fi)

        rgb_np, voxel, events, events_offsets, sample = _load_sample(mapper, seq_ds, fi, cfg, device)

        # Reset state for each independent frame
        mcm.reset_state()

        panel = make_panel(
            mcm, rgb_np, voxel, events, events_offsets,
            max_flow=max_flow, event_alpha=event_alpha, flow_alpha=flow_alpha,
        )
        panel = annotate_panel(panel)

        fname = out_dir / f"flow_qual_{seq_name}_frame{fi:06d}.png"
        cv2.imwrite(str(fname), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        saved.append(fname)
        logger.info("  Saved %s", fname)

    logger.info("Wrote %d panels to %s", len(saved), out_dir)
    return saved


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Qualitative optical flow visualisation (2×2 panels)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", required=True, help="Path to training config YAML")
    p.add_argument("--checkpoint", required=True, help="Path to MCM checkpoint (.pth)")
    p.add_argument("--output-dir", default="visualizations/flow_qualitative", help="Output directory")
    p.add_argument("--num-frames", type=int, default=5, help="Number of random frames to visualise")
    p.add_argument("--seed", type=int, default=42, help="Random seed for frame selection")
    p.add_argument("--max-flow", type=float, default=None, help="Max flow for HSV normalisation")
    p.add_argument("--event-alpha", type=float, default=0.6, help="Blend factor for events overlay")
    p.add_argument("--flow-alpha", type=float, default=0.6, help="Blend factor for flow overlay")
    p.add_argument("--device", default="cuda", help="Device ('cuda' or 'cpu')")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _build_argparser().parse_args(argv)

    visualize_flow_qualitative(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        num_frames=args.num_frames,
        seed=args.seed,
        max_flow=args.max_flow,
        event_alpha=args.event_alpha,
        flow_alpha=args.flow_alpha,
        device_str=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
