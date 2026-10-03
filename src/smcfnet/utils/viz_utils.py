"""
Utility functions for visualization.
Compatible with DSEC dataset format.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import re

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, Union
from smcfnet.utils.event_utils import events_to_voxel_grid

logger = logging.getLogger(__name__)


def _natural_key(p: Path) -> list[object]:
    """Sort paths in human order (e.g. img2 before img10)."""
    parts = re.split(r"(\d+)", p.name)
    key: list[object] = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())
    return key


def _resolve_png_dir(input_path: Path, vis_dir: str = "eval_vis") -> Path:
    """Accept either the PNG directory itself, or an experiment output dir that contains it."""
    if input_path.is_file():
        raise ValueError(f"Expected a directory, got file: {input_path}")

    # If the directory itself contains PNGs, use it.
    if any(input_path.glob("*.png")):
        return input_path

    # Common case: point at outputs/<experiment_name>/ and we pick outputs/<exp>/eval_vis
    candidate = input_path / vis_dir
    if candidate.exists() and candidate.is_dir() and any(candidate.glob("*.png")):
        return candidate

    raise FileNotFoundError(
        f"No PNGs found in {input_path} and none in {candidate}. "
        f"Pass the folder that contains the .png files (e.g. .../eval_vis)."
    )


def _images_to_video(
    png_dir: Path,
    output_path: Path,
    *,
    fps: int = 15,
    pattern: str = "*.png",
    recursive: bool = False,
    max_frames: int = 0,
) -> Path:
    """Create a video (mp4) or gif from PNGs.

    Internal helper (not part of the public API).
    """
    png_dir = png_dir.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if recursive:
        files = [Path(p) for p in png_dir.rglob(pattern)]
    else:
        files = [Path(p) for p in png_dir.glob(pattern)]

    files = [p for p in files if p.is_file()]
    files.sort(key=_natural_key)

    if max_frames and max_frames > 0:
        files = files[: int(max_frames)]

    if len(files) == 0:
        raise FileNotFoundError(f"No files matched {pattern} in {png_dir}")

    suffix = output_path.suffix.lower()
    if suffix not in {".mp4", ".gif"}:
        raise ValueError(f"Unsupported output format: {output_path.suffix} (use .mp4 or .gif)")

    if suffix == ".mp4":
        # Use OpenCV for mp4 writing (usually the most reliable on clusters).
        import cv2

        first = cv2.imread(str(files[0]), cv2.IMREAD_COLOR)
        if first is None:
            raise RuntimeError(f"Failed to read image: {files[0]}")
        h, w = first.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), (w, h))
        if not writer.isOpened():
            raise RuntimeError(
                f"Failed to open VideoWriter for {output_path}. "
                f"Try output .gif, or ensure OpenCV was built with mp4 support."
            )

        try:
            for p in files:
                frame = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if frame is None:
                    logger.warning("Skipping unreadable frame: %s", p)
                    continue
                if frame.shape[0] != h or frame.shape[1] != w:
                    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
                writer.write(frame)
        finally:
            writer.release()

        return output_path

    # GIF
    try:
        import imageio.v3 as iio
    except Exception:  # pragma: no cover
        import imageio as iio  # type: ignore

    frames = []
    for p in files:
        frame = iio.imread(str(p))
        frames.append(frame)
    # imageio expects duration per frame in seconds.
    duration = 1.0 / float(max(int(fps), 1))
    iio.imwrite(str(output_path), frames, duration=duration, loop=0)
    return output_path


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Create an .mp4/.gif from a folder of PNG detection visualizations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input",
        required=True,
        type=str,
        help="Path to the folder containing PNGs (or the experiment output dir that contains eval_vis/)",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (.mp4 or .gif). Default: <png_dir>.mp4",
    )
    p.add_argument("--fps", type=int, default=15, help="Frames per second")
    p.add_argument("--pattern", type=str, default="*.png", help="Glob pattern for input frames")
    p.add_argument("--recursive", action="store_true", help="Search for frames recursively")
    p.add_argument("--max-frames", type=int, default=0, help="If >0, limit number of frames")
    p.add_argument(
        "--vis-dir",
        type=str,
        default="eval_vis",
        help="If --input points at an experiment output dir, use this subfolder name",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)

    input_path = Path(args.input).expanduser()
    png_dir = _resolve_png_dir(input_path, vis_dir=str(args.vis_dir))

    if args.output is None:
        output_path = png_dir.with_suffix(".mp4")
    else:
        output_path = Path(args.output)

    out = _images_to_video(
        png_dir,
        output_path,
        fps=int(args.fps),
        pattern=str(args.pattern),
        recursive=bool(args.recursive),
        max_frames=int(args.max_frames),
    )
    print(f"Wrote video: {out}")
    return 0

def visualize_voxel_grid(
    voxel: torch.Tensor,
    save_path: Optional[str] = None,
    title: str = "Event Voxel Grid",
) -> None:
    """
    Visualize voxel grid representation.
    
    Args:
        voxel: (B, H, W) voxel grid
        save_path: Optional path to save figure
        title: Figure title
    """
    if isinstance(voxel, torch.Tensor):
        voxel = voxel.cpu().numpy()
    
    num_bins = voxel.shape[0]
    
    fig, axes = plt.subplots(1, num_bins, figsize=(num_bins * 3, 3))
    if num_bins == 1:
        axes = [axes]
    
    fig.suptitle(title)
    
    for i, ax in enumerate(axes):
        im = ax.imshow(voxel[i], cmap='RdBu_r', vmin=-5, vmax=5)
        ax.set_title(f'Bin {i}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved voxel visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_event_stream(
    events: Union[Dict, np.ndarray],
    width: int,
    height: int,
    save_path: Optional[str] = None,
    title: str = "Event Stream",
) -> None:
    """
    Visualize event stream as scatter plot.
    
    Args:
        events: Either dict with keys {'x','y','t_ms','p'} or Nx4 array
        width: Image width
        height: Image height
        save_path: Optional path to save figure
        title: Figure title
    """
    # Handle dict format
    if isinstance(events, dict):
        x = events['x'] if isinstance(events['x'], np.ndarray) else events['x'].numpy()
        y = events['y'] if isinstance(events['y'], np.ndarray) else events['y'].numpy()
        p = events['p'] if isinstance(events['p'], np.ndarray) else events['p'].numpy()
    else:
        x = events[:, 0]
        y = events[:, 1]
        p = events[:, 3]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Separate positive and negative events
    pos_mask = p > 0
    neg_mask = p < 0
    
    # Plot with color coding
    if pos_mask.sum() > 0:
        ax.scatter(x[pos_mask], y[pos_mask], 
                  c='red', s=0.5, alpha=0.5, label='Positive')
    
    if neg_mask.sum() > 0:
        ax.scatter(x[neg_mask], y[neg_mask], 
                  c='blue', s=0.5, alpha=0.5, label='Negative')
    
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # Invert y-axis
    ax.set_aspect('equal')
    ax.legend()
    ax.set_title(f'{title} ({len(x)} events)')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved event visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_rgb_event_pair(
    image: torch.Tensor,
    voxel: torch.Tensor,
    save_path: Optional[str] = None,
    title: str = "RGB and Event Voxel",
) -> None:
    """
    Visualize RGB image and corresponding event voxel side by side.
    
    Args:
        image: (3, H, W) RGB image
        voxel: (B, H, W) event voxel
        save_path: Optional path to save figure
        title: Figure title
    """
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if isinstance(voxel, torch.Tensor):
        voxel = voxel.cpu().numpy()
    
    # Convert image from CHW to HWC
    image = np.transpose(image, (1, 2, 0))
    
    # Normalize image to [0, 1]
    if image.max() > 1.0:
        image = image / 255.0
    image = np.clip(image, 0, 1)
    
    # Sum voxel bins for visualization
    voxel_sum = voxel.sum(axis=0)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title)
    
    axes[0].imshow(image)
    axes[0].set_title('RGB Image')
    axes[0].axis('off')
    
    im = axes[1].imshow(voxel_sum, cmap='RdBu_r')
    axes[1].set_title('Event Voxel (summed)')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved RGB-Event pair to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_dsec_sample(
    sample: Dict,
    save_path: Optional[str] = None,
    num_bins: int = 5,
) -> None:
    """
    Visualize a complete DSEC dataset sample.
    
    Args:
        sample: Sample dict from DSECDataset with keys 'rgb', 'events', 'meta'
        save_path: Optional path to save figure
        num_bins: Number of voxel bins
    """
    rgb = sample['rgb']
    events = sample['events']
    meta = sample['meta']
    
    if rgb is None:
        print("No RGB image in sample")
        return
    
    # Get dimensions
    _, height, width = rgb.shape
    
    # Convert events to voxel grid
    voxel = events_to_voxel_grid(
        events=events,
        num_bins=num_bins,
        width=width,
        height=height,
    )
    
    # Create title with metadata
    title = f"Sequence: {meta['sequence']}, Frame: {meta['frame_index']}"
    
    # Visualize
    visualize_rgb_event_pair(rgb, voxel, save_path, title)


# DSEC-Detection class names
DSEC_CLASSES = ('pedestrian', 'rider', 'car', 'bus', 'truck', 'bicycle', 'motorcycle', 'train')


def draw_detection_boxes(
    image: np.ndarray,
    boxes: np.ndarray,
    scores: Optional[np.ndarray] = None,
    classes: Optional[np.ndarray] = None,
    box_format: str = 'xyxy',
    color: tuple = (0, 255, 0),
    thickness: int = 2,
    show_labels: bool = True,
    class_names: tuple = DSEC_CLASSES,
) -> np.ndarray:
    """
    Draw bounding boxes on image.
    
    Args:
        image: (H, W, 3) RGB image as numpy array
        boxes: (N, 4) bounding boxes in format specified by box_format
        scores: (N,) confidence scores (optional)
        classes: (N,) class indices (optional)
        box_format: 'xyxy' (x1,y1,x2,y2) or 'cxcywh' (cx,cy,w,h)
        color: RGB color tuple for boxes
        thickness: Line thickness
        show_labels: Whether to show class labels and scores
        class_names: Tuple of class names
    
    Returns:
        Image with boxes drawn
    """
    import cv2
    
    # Make a copy to avoid modifying original
    img = image.copy()
    
    # Ensure uint8 for cv2
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    
    if len(boxes) == 0:
        return img
    
    # Convert boxes to xyxy format if needed
    if box_format == 'cxcywh':
        boxes_xyxy = np.zeros_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
        boxes = boxes_xyxy
    
    # Draw each box
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.astype(int)
        
        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label if requested
        if show_labels:
            label_parts = []
            if classes is not None:
                class_id = int(classes[i])
                if 0 <= class_id < len(class_names):
                    label_parts.append(class_names[class_id])
            if scores is not None:
                label_parts.append(f'{scores[i]:.2f}')
            
            if label_parts:
                label = ' '.join(label_parts)
                
                # Get text size
                (text_w, text_h), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                
                # Draw background rectangle for text
                cv2.rectangle(
                    img,
                    (x1, y1 - text_h - baseline - 5),
                    (x1 + text_w, y1),
                    color,
                    -1
                )
                
                # Draw text
                cv2.putText(
                    img,
                    label,
                    (x1, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )
    
    return img


def visualize_detections(
    image: torch.Tensor,
    predictions: Optional[torch.Tensor] = None,
    targets: Optional[torch.Tensor] = None,
    save_path: Optional[str] = None,
    title: str = "Detections",
    conf_threshold: float = 0.3,
) -> None:
    """
    Visualize predictions and ground truth on image.
    
    Args:
        image: (3, H, W) RGB image tensor
        predictions: (N, 7) predictions [x1, y1, x2, y2, obj_conf, class_conf, class_id]
        targets: (M, 5) targets [class_id, cx, cy, w, h]
        save_path: Path to save visualization
        title: Figure title
        conf_threshold: Minimum confidence to show predictions
    """
    import cv2
    
    # Convert image to numpy
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    
    # Convert CHW to HWC
    image = np.transpose(image, (1, 2, 0))
    
    # Normalize to [0, 255]
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)

    # OpenCV drawing assumes BGR. Our tensors are RGB, so convert once here.
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(title, fontsize=14)
    
    # Draw ground truth
    img_gt = image_bgr.copy()
    if targets is not None and len(targets) > 0:
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        
        # Filter out padding (class_id == 0 and all zeros)
        valid_mask = targets[:, 0] >= 0
        targets = targets[valid_mask]
        
        if len(targets) > 0:
            gt_boxes = targets[:, 1:5]  # cx, cy, w, h
            gt_classes = targets[:, 0]
            img_gt = draw_detection_boxes(
                img_gt,
                gt_boxes,
                classes=gt_classes,
                box_format='cxcywh',
                color=(0, 255, 0),  # Green for GT
                thickness=2,
                show_labels=True,
            )
    
    axes[0].imshow(cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f'Ground Truth ({len(targets) if targets is not None else 0} boxes)')
    axes[0].axis('off')
    
    # Draw predictions
    img_pred = image_bgr.copy()
    if predictions is not None and len(predictions) > 0:
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        
        # Filter by confidence
        conf_scores = predictions[:, 4] * predictions[:, 5]
        conf_mask = conf_scores >= conf_threshold
        predictions = predictions[conf_mask]
        
        if len(predictions) > 0:
            pred_boxes = predictions[:, :4]  # x1, y1, x2, y2
            pred_scores = predictions[:, 4] * predictions[:, 5]
            pred_classes = predictions[:, 6]
            img_pred = draw_detection_boxes(
                img_pred,
                pred_boxes,
                scores=pred_scores,
                classes=pred_classes,
                box_format='xyxy',
                color=(255, 0, 0),  # Red for predictions
                thickness=2,
                show_labels=True,
            )
    
    axes[1].imshow(cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'Predictions ({len(predictions) if predictions is not None else 0} boxes, conf>{conf_threshold})')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved detection visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two arrays of xyxy boxes. Returns (Na, Nb)."""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def _greedy_match_by_iou(
    det_xyxy: np.ndarray,
    gt_xyxy: np.ndarray,
    *,
    iou_thresh: float = 0.3,
) -> list[tuple[int, int, float]]:
    """Greedy pred↔gt matching on current-frame IoU. Returns (det_idx, gt_idx, iou)."""
    if det_xyxy.size == 0 or gt_xyxy.size == 0:
        return []
    iou = _iou_xyxy(det_xyxy, gt_xyxy)
    matches: list[tuple[int, int, float]] = []
    used_d: set[int] = set()
    used_g: set[int] = set()
    # Walk from highest IoU down.
    flat = [(iou[i, j], i, j) for i in range(iou.shape[0]) for j in range(iou.shape[1])]
    flat.sort(key=lambda t: -t[0])
    for v, i, j in flat:
        if v < iou_thresh:
            break
        if i in used_d or j in used_g:
            continue
        used_d.add(i)
        used_g.add(j)
        matches.append((i, j, float(v)))
    return matches


def _cos_to_bgr(cos: float) -> tuple[int, int, int]:
    """Map cosine similarity in [-1, 1] to a BGR triple on the RdYlGn colormap.

    Used to color predicted-future-box outlines by direction agreement with GT:
    green ≈ same direction, yellow ≈ orthogonal, red ≈ opposite.
    NaN (undefined — e.g. zero-magnitude GT displacement) maps to neutral grey.
    """
    from matplotlib import cm

    if not np.isfinite(cos):
        return (180, 180, 180)
    t = float(np.clip((cos + 1.0) * 0.5, 0.0, 1.0))
    r, g, b, _ = cm.RdYlGn(t)
    return (int(b * 255), int(g * 255), int(r * 255))


def visualize_sttp_trajectories(
    image,
    det_current_xyxy: Optional[np.ndarray] = None,
    det_future_cxcywh: Optional[np.ndarray] = None,
    gt_current_cxcywh: Optional[np.ndarray] = None,
    gt_future_cxcywh: Optional[np.ndarray] = None,
    gt_valid: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    title: str = "STTP trajectories",
    match_iou_thresh: float = 0.3,
    max_table_rows: int = 12,
) -> list[dict]:
    """Overlay current + t+k future boxes for GT and predictions.

    GT boxes are green (solid current, thin future). Unmatched predictions are
    dim red. Matched predictions have their future-box outline colored by
    cosine similarity between the predicted and GT displacement vectors (green
    = same direction, yellow = orthogonal, red = opposite), with a per-box
    text label ``cos=x r=|p|/|g| epe=dpx`` so the viewer can read direction
    agreement at a glance and magnitude agreement as text.

    Returns the per-match records (one dict per matched pred↔GT pair) so the
    caller can persist a metrics JSONL alongside the PNG.
    """
    import cv2

    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)

    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]

    def _cxcywh_to_xyxy(b):
        out = np.zeros_like(b, dtype=np.float32)
        out[..., 0] = b[..., 0] - b[..., 2] / 2
        out[..., 1] = b[..., 1] - b[..., 3] / 2
        out[..., 2] = b[..., 0] + b[..., 2] / 2
        out[..., 3] = b[..., 1] + b[..., 3] / 2
        return out

    def _in_bounds(cx, cy):
        return 0 <= cx < w and 0 <= cy < h

    def _put_label(img, text, anchor, fg, *, bg=(0, 0, 0)):
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.42
        thick = 1
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thick)
        x, y = int(anchor[0]), int(anchor[1])
        x0 = max(0, min(x, img.shape[1] - tw - 4))
        y0 = max(th + 2, min(y, img.shape[0] - 2))
        cv2.rectangle(
            img,
            (x0 - 2, y0 - th - baseline - 1),
            (x0 + tw + 2, y0 + baseline - 1),
            bg,
            -1,
        )
        cv2.putText(img, text, (x0, y0 - 2), font, scale, fg, thick, cv2.LINE_AA)

    # --- Build pred/GT records without drawing yet (we need match results to
    # decide pred outline color) ---------------------------------------------
    gt_records: list[dict] = []
    if gt_current_cxcywh is not None and gt_future_cxcywh is not None:
        valid = gt_valid.astype(bool) if gt_valid is not None else np.ones(len(gt_current_cxcywh), dtype=bool)
        for i in range(len(gt_current_cxcywh)):
            if not valid[i]:
                continue
            cur = gt_current_cxcywh[i]
            fut = gt_future_cxcywh[i]
            if cur[2] <= 0 or cur[3] <= 0 or fut[2] <= 0 or fut[3] <= 0:
                continue
            gt_records.append({
                "cur_cxcywh": cur.astype(np.float32),
                "fut_cxcywh": fut.astype(np.float32),
                "cur_xyxy": _cxcywh_to_xyxy(cur).astype(np.float32),
                "fut_xyxy": _cxcywh_to_xyxy(fut).astype(np.float32),
                "disp": (float(fut[0] - cur[0]), float(fut[1] - cur[1])),
            })

    det_records: list[dict] = []
    if det_current_xyxy is not None and det_future_cxcywh is not None:
        n = min(len(det_current_xyxy), len(det_future_cxcywh))
        for i in range(n):
            cur_xyxy = det_current_xyxy[i, :4].astype(np.float32)
            fut = det_future_cxcywh[i]
            if fut[2] <= 0 or fut[3] <= 0:
                continue
            cur_cx = float(cur_xyxy[0] + cur_xyxy[2]) / 2
            cur_cy = float(cur_xyxy[1] + cur_xyxy[3]) / 2
            det_records.append({
                "cur_xyxy": cur_xyxy,
                "fut_cxcywh": fut.astype(np.float32),
                "fut_xyxy": _cxcywh_to_xyxy(fut).astype(np.float32),
                "cur_center": (cur_cx, cur_cy),
                "disp": (float(fut[0] - cur_cx), float(fut[1] - cur_cy)),
            })

    # --- Greedy pred↔gt matching on current-box IoU --------------------------
    matches: list[dict] = []
    matched_det: dict[int, dict] = {}
    if det_records and gt_records:
        det_arr = np.stack([r["cur_xyxy"] for r in det_records], axis=0)
        gt_arr = np.stack([r["cur_xyxy"] for r in gt_records], axis=0)
        fut_det_arr = np.stack([r["fut_xyxy"] for r in det_records], axis=0)
        fut_gt_arr = np.stack([r["fut_xyxy"] for r in gt_records], axis=0)
        fut_iou_all = _iou_xyxy(fut_det_arr, fut_gt_arr)
        for di, gi, iou_cur in _greedy_match_by_iou(det_arr, gt_arr, iou_thresh=match_iou_thresh):
            ddx, ddy = det_records[di]["disp"]
            gdx, gdy = gt_records[gi]["disp"]
            pred_mag = float(np.hypot(ddx, ddy))
            gt_mag = float(np.hypot(gdx, gdy))
            denom = pred_mag * gt_mag
            cos = float((ddx * gdx + ddy * gdy) / denom) if denom > 1e-6 else float("nan")
            mag_ratio = float(pred_mag / gt_mag) if gt_mag > 1e-6 else float("nan")
            epe = float(np.hypot(ddx - gdx, ddy - gdy))
            iou_tk = float(fut_iou_all[di, gi])
            rec = {
                "det_idx": di,
                "gt_idx": gi,
                "iou_current": float(iou_cur),
                "iou_future": iou_tk,
                "pred_disp": [ddx, ddy],
                "gt_disp": [gdx, gdy],
                "pred_mag": pred_mag,
                "gt_mag": gt_mag,
                "cosine": cos,
                "mag_ratio": mag_ratio,
                "epe": epe,
            }
            matches.append(rec)
            matched_det[di] = rec

    # --- Draw GT boxes --------------------------------------------------------
    for g in gt_records:
        cur_xyxy = g["cur_xyxy"].astype(int)
        fut_xyxy = g["fut_xyxy"].astype(int)
        cur = g["cur_cxcywh"]
        fut = g["fut_cxcywh"]
        cv2.rectangle(img_bgr, tuple(cur_xyxy[:2]), tuple(cur_xyxy[2:]), (0, 200, 0), 2)
        cv2.rectangle(img_bgr, tuple(fut_xyxy[:2]), tuple(fut_xyxy[2:]), (0, 200, 0), 1)
        p0 = (int(cur[0]), int(cur[1]))
        p1 = (int(fut[0]), int(fut[1]))
        if _in_bounds(*p0) and _in_bounds(*p1):
            cv2.arrowedLine(img_bgr, p0, p1, (0, 200, 0), 2, tipLength=0.25)

    # --- Draw pred boxes (colored by cosine when matched) --------------------
    for di, d in enumerate(det_records):
        cur_xyxy = d["cur_xyxy"].astype(int)
        fut_xyxy = d["fut_xyxy"].astype(int)
        cur_cx, cur_cy = d["cur_center"]
        fut_cx, fut_cy = float(d["fut_cxcywh"][0]), float(d["fut_cxcywh"][1])
        m = matched_det.get(di)
        if m is None:
            # Unmatched prediction: dim red, no cosine context available.
            color_cur = (70, 70, 200)
            color_fut = (70, 70, 200)
            label = f"P d=({d['disp'][0]:+.0f},{d['disp'][1]:+.0f})"
        else:
            color_cur = (0, 0, 200)
            color_fut = _cos_to_bgr(m["cosine"])
            cos_s = "NA" if not np.isfinite(m["cosine"]) else f"{m['cosine']:+.2f}"
            r_s = "NA" if not np.isfinite(m["mag_ratio"]) else f"{m['mag_ratio']:.2f}"
            label = f"cos={cos_s} r={r_s} epe={m['epe']:.1f}"
        cv2.rectangle(img_bgr, tuple(cur_xyxy[:2]), tuple(cur_xyxy[2:]), color_cur, 2)
        cv2.rectangle(img_bgr, tuple(fut_xyxy[:2]), tuple(fut_xyxy[2:]), color_fut, 2)
        p0 = (int(cur_cx), int(cur_cy))
        p1 = (int(fut_cx), int(fut_cy))
        if _in_bounds(*p0) and _in_bounds(*p1):
            cv2.arrowedLine(img_bgr, p0, p1, color_fut, 2, tipLength=0.25)
        _put_label(img_bgr, label, (cur_xyxy[0], cur_xyxy[3] + 12), color_fut)

    # --- Legend on image ------------------------------------------------------
    legend_lines = [
        ("GT  (solid cur / thin fut)", (0, 200, 0)),
        ("Pred future: outline hue = cos(pred, gt)  green=same dir, red=opposite", (180, 180, 180)),
        ("Per-box label: cos / r=|pred|/|gt| / epe px", (180, 180, 180)),
    ]
    y = 22
    for text, color in legend_lines:
        cv2.putText(img_bgr, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        y += 18

    # --- Figure with main image + side table ---------------------------------
    if matches:
        fig, axes = plt.subplots(
            1, 2, figsize=(14, 8),
            gridspec_kw={"width_ratios": [3, 1]},
        )
        ax_img, ax_tab = axes
    else:
        fig, ax_img = plt.subplots(1, 1, figsize=(10, 8))
        ax_tab = None

    ax_img.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    ax_img.set_title(title, fontsize=11)
    ax_img.axis("off")

    if ax_tab is not None:
        ax_tab.axis("off")
        # Sort by worst-direction (lowest cos) first so regressions stand out;
        # NaN (zero-mag GT) pushed to the end so real failures rank first.
        def _sort_key(m):
            c = m["cosine"]
            return (1, 0.0) if not np.isfinite(c) else (0, c)
        matches_sorted = sorted(matches, key=_sort_key)[:max_table_rows]
        rows = [["#", "cos", "EPE px", "|p|/|g|", "IoU t+k"]]
        for k, m in enumerate(matches_sorted):
            cos_s = "—" if not np.isfinite(m["cosine"]) else f"{m['cosine']:+.2f}"
            r_s = "—" if not np.isfinite(m["mag_ratio"]) else f"{m['mag_ratio']:.2f}"
            rows.append([
                str(k),
                cos_s,
                f"{m['epe']:.2f}",
                r_s,
                f"{m['iou_future']:.2f}",
            ])
        table = ax_tab.table(
            cellText=rows[1:], colLabels=rows[0],
            cellLoc="center", loc="upper center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.2)
        finite_cos = [m["cosine"] for m in matches if np.isfinite(m["cosine"])]
        mean_epe = float(np.mean([m["epe"] for m in matches]))
        mean_iou_tk = float(np.mean([m["iou_future"] for m in matches]))
        mean_cos_s = f"{np.mean(finite_cos):+.2f}" if finite_cos else "—"
        n_shown = len(matches_sorted)
        extra = len(matches) - n_shown
        footer = (
            f"matched: {len(matches)}   "
            f"mean cos: {mean_cos_s}   "
            f"mean EPE: {mean_epe:.2f} px   "
            f"mean IoU@t+k: {mean_iou_tk:.2f}"
        )
        if extra > 0:
            footer += f"   (+{extra} more)"
        ax_tab.set_title(footer, fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved STTP trajectory visualization to {save_path}")
    else:
        plt.show()
    plt.close()
    return matches


if __name__ == "__main__":
    raise SystemExit(main())