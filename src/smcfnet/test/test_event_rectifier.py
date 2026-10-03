"""Test script for visualizing event camera to rectified RGB alignment.

This script demonstrates:
1. The coordinate space mismatch between raw DSEC events (640×480) and rectified RGB (1440×1080)
2. Naive linear scaling vs proper rectification mapping
3. Target/detection warping from event camera space to rectified RGB space

Visualizations are saved to ./visualizations/
"""
from pathlib import Path
import sys
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
import torch

from smcfnet.utils.event_utils import events_to_voxel_grid_torch
from smcfnet.config.config import Config, load_config_from_yaml, update_config_from_dict
from smcfnet.dataset.dsec import DSECDataset, DSECLightweightMapper, build_dsec_sequence_index_datasets


def find_repo_root(start: Path) -> Path:
    """Find SMCM-MCFNet repo root by walking upwards until `src/smcfnet` exists."""
    start = start.resolve()
    # Check current dir and parents
    for cand in [start, *start.parents]:
        if (cand / "src" / "smcfnet").is_dir():
            return cand
    # Also check for SMCM-MCFNet subdirectory (common when running from parent dir)
    for cand in [start, *start.parents]:
        smcm_subdir = cand / "SMCM-MCFNet"
        if smcm_subdir.is_dir() and (smcm_subdir / "src" / "smcfnet").is_dir():
            return smcm_subdir
    raise FileNotFoundError(
        "Could not find repo root containing `src/smcfnet` starting from: " + str(start)
    )


def chw_to_hwc_uint8(img_chw: torch.Tensor) -> np.ndarray:
    """Convert CHW PyTorch tensor to HWC uint8 numpy array."""
    img = img_chw.detach().cpu().float().numpy()
    img = np.transpose(img, (1, 2, 0))
    if img.max() <= 1.0:
        img = img * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)


def pick_nonempty_event_sample(dataset, mapper, max_tries: int = 50):
    """Pick a random sample with non-empty events."""
    for _ in range(max_tries):
        idx = random.randrange(0, len(dataset))
        index_item = dataset[idx]
        s = mapper(index_item)
        if isinstance(s.get("events"), torch.Tensor) and s["events"].numel() > 0:
            return idx, index_item, s
    # fall back to first item
    idx = 0
    index_item = dataset[idx]
    return idx, index_item, mapper(index_item)


def _warp_xyxy_boxes_via_remap(
    xyxy_ev: np.ndarray,
    remap_ev_to_rect: np.ndarray,
    rgb_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Warp axis-aligned boxes from event/distorted pixel coords to rectified RGB.

    Uses the `rectify_map`-based lookup remap: (x_ev, y_ev) -> (x_rect, y_rect)
    on the four corners, then returns the enclosing axis-aligned rectangle.

    Args:
        xyxy_ev: (N, 4) boxes in event pixel coords [x1,y1,x2,y2].
        remap_ev_to_rect: (H_ev, W_ev, 2) float map giving rectified coords.
        rgb_hw: (H_rgb, W_rgb)

    Returns:
        warped_xyxy: (N, 4) float boxes in rectified coords [x1,y1,x2,y2]
        valid: (N,) bool mask for boxes successfully warped.
    """
    if xyxy_ev.size == 0:
        return xyxy_ev.astype(np.float32), np.zeros((0,), dtype=bool)

    H_ev, W_ev = remap_ev_to_rect.shape[:2]
    H_rgb, W_rgb = rgb_hw

    xyxy_ev = np.asarray(xyxy_ev, dtype=np.float32)
    x1, y1, x2, y2 = [xyxy_ev[:, i] for i in range(4)]

    # 4 corners: TL, TR, BL, BR
    corners = np.stack(
        [
            np.stack([x1, y1], axis=1),
            np.stack([x2, y1], axis=1),
            np.stack([x1, y2], axis=1),
            np.stack([x2, y2], axis=1),
        ],
        axis=1,
    )  # (N, 4, 2)

    corners_int = np.rint(corners).astype(np.int64)
    cx = corners_int[:, :, 0]
    cy = corners_int[:, :, 1]

    in_bounds = (cx >= 0) & (cx < W_ev) & (cy >= 0) & (cy < H_ev)

    # Safe indexing: clamp for lookup, but keep validity via in_bounds.
    cx_clamped = np.clip(cx, 0, W_ev - 1)
    cy_clamped = np.clip(cy, 0, H_ev - 1)

    warped = remap_ev_to_rect[cy_clamped, cx_clamped]  # (N, 4, 2)
    finite = np.isfinite(warped).all(axis=2)

    valid = in_bounds.all(axis=1) & finite.all(axis=1)

    warped_x = warped[:, :, 0]
    warped_y = warped[:, :, 1]

    wx1 = warped_x.min(axis=1)
    wy1 = warped_y.min(axis=1)
    wx2 = warped_x.max(axis=1)
    wy2 = warped_y.max(axis=1)

    warped_xyxy = np.stack([wx1, wy1, wx2, wy2], axis=1).astype(np.float32)

    # Clip to RGB bounds and filter degenerate boxes.
    warped_xyxy[:, 0] = np.clip(warped_xyxy[:, 0], 0, W_rgb - 1)
    warped_xyxy[:, 2] = np.clip(warped_xyxy[:, 2], 0, W_rgb - 1)
    warped_xyxy[:, 1] = np.clip(warped_xyxy[:, 1], 0, H_rgb - 1)
    warped_xyxy[:, 3] = np.clip(warped_xyxy[:, 3], 0, H_rgb - 1)

    nondeg = (warped_xyxy[:, 2] > warped_xyxy[:, 0]) & (warped_xyxy[:, 3] > warped_xyxy[:, 1])
    valid = valid & nondeg

    return warped_xyxy, valid


def main():
    """Main test function for event rectification visualization."""
    # Setup output directory
    vis_dir = Path("./visualizations")
    vis_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving visualizations to: {vis_dir.resolve()}")

    # Constants
    EV_W, EV_H = 640, 480  # Event camera resolution
    
    # Find repo root and load configuration
    repo_root = find_repo_root(Path.cwd())
    run_cfg = repo_root / "src" / "smcfnet" / "config" / "runs" / "rgbe-fullscale" / "scale_test.yaml"
    assert run_cfg.exists(), f"Run config not found: {run_cfg}"

    cfg = Config()
    update_config_from_dict(cfg, load_config_from_yaml(str(run_cfg)))
    
    # Force mapper to return events + tracks + rectified RGB
    cfg.data.mapper_load_events = True
    cfg.data.mapper_load_tracks = True
    cfg.data.mapper_load_images = True
    cfg.data.mapper_prefer_distorted = False  # rectified RGB

    # Resolve DSEC dataset config path
    p = Path(cfg.data.dataset_config)
    if not p.is_absolute():
        p = (repo_root / p).resolve()
    assert p.exists(), f"DSEC dataset YAML not found: {p}"

    RGB_W, RGB_H = int(cfg.data.image_width), int(cfg.data.image_height)

    print("repo_root:", repo_root)
    print("run_cfg:", run_cfg)
    print("dataset_cfg:", p)
    print(f"RGB resolution: {RGB_W}×{RGB_H}")
    print(f"Event resolution: {EV_W}×{EV_H}")

    # Use rectified events/tracks for full-resolution rectified images
    use_rectified = (not cfg.data.mapper_prefer_distorted) and (RGB_H == 1080 and RGB_W == 1440)
    print(f"Using rectified events/tracks: {use_rectified}")

    # Build dataset
    per_sequence = build_dsec_sequence_index_datasets(
        dsec_yaml_path=p,
        split="train",
        dt_ms=float(cfg.data.event_window_ms),
        use_rectified=use_rectified,
    )
    dataset = DSECDataset(per_sequence)

    mapper = DSECLightweightMapper(
        per_sequence,
        load_events=True,
        load_tracks=True,
        mmap_tracks=bool(cfg.data.mapper_mmap_tracks),
        load_images=True,
        image_height=RGB_H,
        image_width=RGB_W,
        prefer_distorted=False,
        build_targets=bool(cfg.data.mapper_build_targets),
    )

    print(f"Loaded {len(per_sequence)} sequences, {len(dataset)} frames total")

    # Pick a sample with events
    idx, index_item, s = pick_nonempty_event_sample(dataset, mapper)
    img = s["image"]
    events = s.get("events", torch.empty((0, 4), dtype=torch.int64))
    targets = s.get("targets", torch.zeros((0, 5), dtype=torch.float32))

    print(f"\nPicked sample idx: {idx}")
    print(f"  image shape: {tuple(img.shape)}")
    print(f"  events shape: {tuple(events.shape)}")
    print(f"  targets shape: {tuple(targets.shape)}")

    bins = int(getattr(cfg.data, "event_bins", 5))

    # ========== VISUALIZATION 1: Event coordinate spaces ==========
    print("\n[1/4] Creating event coordinate space comparison...")
    
    # Voxelize raw events at event camera resolution
    voxel_raw = events_to_voxel_grid_torch(events, num_bins=bins, width=EV_W, height=EV_H)
    heat_raw = voxel_raw.sum(dim=0).cpu().numpy()

    # Naively scaled events
    sx, sy = RGB_W / EV_W, RGB_H / EV_H
    events_scaled = events.clone()
    if events_scaled.numel() > 0:
        events_scaled[:, 0] = torch.clamp((events_scaled[:, 0].float() * sx).round().to(torch.int64), 0, RGB_W - 1)
        events_scaled[:, 1] = torch.clamp((events_scaled[:, 1].float() * sy).round().to(torch.int64), 0, RGB_H - 1)

    voxel_scaled = events_to_voxel_grid_torch(events_scaled, num_bins=bins, width=RGB_W, height=RGB_H)
    heat_scaled = voxel_scaled.sum(dim=0).cpu().numpy()

    print(f"  heat_raw: {heat_raw.shape}, range [{heat_raw.min():.2f}, {heat_raw.max():.2f}], {np.count_nonzero(heat_raw)} nonzero")
    print(f"  heat_scaled: {heat_scaled.shape}, range [{heat_scaled.min():.2f}, {heat_scaled.max():.2f}], {np.count_nonzero(heat_scaled)} nonzero")

    # Prepare RGB image
    img_np = chw_to_hwc_uint8(img)

    # Create 3-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Panel 1: Just the rectified RGB
    axes[0].imshow(img_np)
    axes[0].set_title("Rectified RGB (1440×1080)")
    axes[0].axis("off")

    # Panel 2: Raw events at native resolution
    axes[1].imshow(img_np)
    if heat_raw.max() > 0:
        heat_raw_fullres = np.zeros((RGB_H, RGB_W), dtype=np.float32)
        heat_raw_fullres[:EV_H, :EV_W] = heat_raw
        
        vmax_raw = np.percentile(heat_raw[heat_raw > 0], 95) if np.any(heat_raw > 0) else heat_raw.max()
        norm_raw = Normalize(vmin=0, vmax=vmax_raw, clip=True)
        
        axes[1].imshow(heat_raw_fullres, cmap="hot", alpha=0.6, norm=norm_raw)
        axes[1].set_title(f"Raw event coords (640×480, upper-left)\nmax={heat_raw.max():.1f}")
        
        from matplotlib.patches import Rectangle
        rect = Rectangle((0, 0), EV_W, EV_H, linewidth=2, edgecolor='cyan', facecolor='none', linestyle='--')
        axes[1].add_patch(rect)
    else:
        axes[1].set_title("Raw event coords (no events)")
    axes[1].axis("off")

    # Panel 3: Naively scaled events
    axes[2].imshow(img_np)
    if heat_scaled.max() > 0:
        vmax_scaled = np.percentile(heat_scaled[heat_scaled > 0], 95) if np.any(heat_scaled > 0) else heat_scaled.max()
        norm_scaled = Normalize(vmin=0, vmax=vmax_scaled, clip=True)
        
        axes[2].imshow(heat_scaled, cmap="hot", alpha=0.6, norm=norm_scaled)
        axes[2].set_title(f"Naive scaled events (1440×1080)\nmax={heat_scaled.max():.1f}")
    else:
        axes[2].set_title("Naive scaled events (no events)")
    axes[2].axis("off")

    plt.tight_layout()
    out_path = vis_dir / "01_event_coordinate_spaces.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")

    # ========== VISUALIZATION 2: Warped events ==========
    print("\n[2/4] Creating warped event visualization...")
    
    # Make dsec-det importable
    dsec_det_src = (repo_root.parent / "dsec-det" / "src").resolve()
    if str(dsec_det_src) not in sys.path:
        sys.path.insert(0, str(dsec_det_src))

    from smcfnet.dataset.dsec.io import yaml_file_to_dict, h5_file_to_dict
    from smcfnet.dataset.dsec.remapping import compute_remapping

    seq_idx = int(s["seq_idx"])
    meta = per_sequence[seq_idx].meta

    assert meta.rectify_map_h5_path is not None and Path(meta.rectify_map_h5_path).exists()
    cam_to_cam = Path(meta.seq_path) / "calibration" / "cam_to_cam.yaml"
    assert cam_to_cam.exists()

    calib = yaml_file_to_dict(cam_to_cam)
    rectification_map = h5_file_to_dict(Path(meta.rectify_map_h5_path))
    remap = compute_remapping(calib, rectification_map)

    print(f"  remap shape: {remap.shape}, dtype: {remap.dtype}")
    print(f"  remap x-range: [{float(np.nanmin(remap[..., 0])):.1f}, {float(np.nanmax(remap[..., 0])):.1f}]")
    print(f"  remap y-range: [{float(np.nanmin(remap[..., 1])):.1f}, {float(np.nanmax(remap[..., 1])):.1f}]")

    # Warp events
    max_events = 250_000
    if events.shape[0] > max_events:
        step = int(np.ceil(events.shape[0] / max_events))
        events_sub = events[::step]
    else:
        events_sub = events

    x = events_sub[:, 0].cpu().numpy().astype(np.int64)
    y = events_sub[:, 1].cpu().numpy().astype(np.int64)

    H_ev, W_ev = remap.shape[:2]
    valid = (x >= 0) & (x < W_ev) & (y >= 0) & (y < H_ev)

    xy_rect = remap[y[valid], x[valid]]

    x_rect = np.rint(xy_rect[:, 0]).astype(np.int64)
    y_rect = np.rint(xy_rect[:, 1]).astype(np.int64)
    valid2 = (x_rect >= 0) & (x_rect < RGB_W) & (y_rect >= 0) & (y_rect < RGB_H)

    ev_warp = events_sub[valid].clone()
    ev_warp = ev_warp[torch.from_numpy(valid2)]
    ev_warp[:, 0] = torch.from_numpy(x_rect[valid2]).to(ev_warp.device, dtype=torch.int64)
    ev_warp[:, 1] = torch.from_numpy(y_rect[valid2]).to(ev_warp.device, dtype=torch.int64)

    print(f"  events_sub: {events_sub.shape[0]} -> warped valid: {ev_warp.shape[0]}")

    voxel_warp = events_to_voxel_grid_torch(ev_warp, num_bins=bins, width=RGB_W, height=RGB_H)
    heat_warp = voxel_warp.sum(dim=0).cpu().numpy()

    # Plot warped event heatmap
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.imshow(img_np)

    if heat_warp.max() > 0:
        vmax_warp = np.percentile(heat_warp[heat_warp > 0], 90) if np.any(heat_warp > 0) else heat_warp.max()
        norm_warp = Normalize(vmin=0, vmax=vmax_warp, clip=True)
        ax.imshow(heat_warp, cmap="hot", alpha=0.75, norm=norm_warp)
        ax.set_title(f"Overlay: warped events via rectify_map (max={heat_warp.max():.1f})")
    else:
        ax.set_title("Overlay: warped events via rectify_map (no events)")

    ax.axis("off")
    out_path = vis_dir / "02_warped_events_heatmap.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")

    # Plot warped event scatter
    n_scatter = 20_000
    if ev_warp.shape[0] > 0:
        idxs = torch.randperm(ev_warp.shape[0])[: min(n_scatter, ev_warp.shape[0])]
        pts = ev_warp[idxs][:, :2].cpu().numpy()
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        ax.imshow(img_np)
        ax.scatter(pts[:, 0], pts[:, 1], s=0.5, c="cyan", alpha=0.7)
        ax.set_title(f"Warped event points (subset, n={len(pts)})")
        ax.axis("off")
        out_path = vis_dir / "03_warped_events_scatter.png"
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {out_path}")

    # ========== VISUALIZATION 3: Warped targets ==========
    print("\n[3/4] Creating target warping visualization...")
    
    # Convert targets from [cls, cx, cy, w, h] to [x1, y1, x2, y2]
    if not isinstance(targets, torch.Tensor) or targets.numel() == 0:
        print("  No targets available in this sample.")
        xyxy_ev = np.zeros((0, 4), dtype=np.float32)
        cls = None
        valid_boxes = np.zeros((0,), dtype=bool)
        warped_xyxy = np.zeros((0, 4), dtype=np.float32)
    else:
        targets_np = targets.detach().cpu().numpy()
        
        cls_col = targets_np[:, 0]
        cx, cy, w, h = targets_np[:, 1], targets_np[:, 2], targets_np[:, 3], targets_np[:, 4]
        
        xyxy_ev = np.stack([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], axis=1)
        cls = cls_col

        warped_xyxy, valid_boxes = _warp_xyxy_boxes_via_remap(xyxy_ev, remap, rgb_hw=(RGB_H, RGB_W))

        print(f"  targets: {len(xyxy_ev)} | warped valid: {int(valid_boxes.sum())}")

    # Visualize raw vs warped
    if len(xyxy_ev) > 0:
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.imshow(img_np)
        ax.set_title("Targets: raw event coords (red, scaled) vs warped-to-rectified (lime)")

        # Raw boxes - scale from event coords to RGB coords for visualization
        scale_x, scale_y = RGB_W / EV_W, RGB_H / EV_H
        for i in range(xyxy_ev.shape[0]):
            x1, y1, x2, y2 = xyxy_ev[i]
            # Scale to RGB resolution for display
            x1_scaled, x2_scaled = x1 * scale_x, x2 * scale_x
            y1_scaled, y2_scaled = y1 * scale_y, y2 * scale_y
            rect = patches.Rectangle(
                (x1_scaled, y1_scaled), max(0.0, x2_scaled - x1_scaled), max(0.0, y2_scaled - y1_scaled),
                linewidth=2, edgecolor="red", facecolor="none", alpha=0.7, linestyle="--"
            )
            ax.add_patch(rect)

        # Warped boxes
        for i in np.where(valid_boxes)[0]:
            x1, y1, x2, y2 = warped_xyxy[i]
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor="lime", facecolor="none", alpha=0.9
            )
            ax.add_patch(rect)

            if cls is not None:
                ax.text(
                    x1, max(0.0, y1 - 2), f"{int(cls[i])}",
                    color="lime", fontsize=9,
                    bbox=dict(facecolor="black", alpha=0.4, pad=1, edgecolor="none")
                )

        ax.set_xlim(0, RGB_W)
        ax.set_ylim(RGB_H, 0)
        out_path = vis_dir / "04_warped_targets.png"
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {out_path}")
    else:
        print("  No targets to visualize.")

    # ========== VISUALIZATION 4: Combined view ==========
    print("\n[4/4] Creating combined visualization with warped events overlay...")
    
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(img_np)

    # Add warped event heatmap overlay with better visibility
    if heat_warp.max() > 0:
        vmax_combined = np.percentile(heat_warp[heat_warp > 0], 90) if np.any(heat_warp > 0) else heat_warp.max()
        norm_combined = Normalize(vmin=0, vmax=vmax_combined, clip=True)
        ax.imshow(heat_warp, alpha=0.5, cmap="hot", norm=norm_combined)

    ax.set_title("Combined: warped events + warped targets (lime)")

    # Only show warped boxes (no raw boxes)
    if len(xyxy_ev) > 0:
        for i in np.where(valid_boxes)[0]:
            x1, y1, x2, y2 = warped_xyxy[i]
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor="lime", facecolor="none", alpha=0.95
            )
            ax.add_patch(rect)

            if cls is not None:
                ax.text(
                    x1, max(0.0, y1 - 2), f"{int(cls[i])}",
                    color="lime", fontsize=10,
                    bbox=dict(facecolor="black", alpha=0.6, pad=2, edgecolor="none")
                )

    ax.set_xlim(0, RGB_W)
    ax.set_ylim(RGB_H, 0)
    out_path = vis_dir / "05_combined_overlay.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")

    print(f"\n✓ All visualizations saved to: {vis_dir.resolve()}")


if __name__ == "__main__":
    main()
