"""
Optical flow and IWE (Image of Warped Events) visualization utilities.

Provides HSV-coded flow field rendering and before/after IWE comparison
panels for evaluating the MCM (Motion Compensation Module).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ── Flow colorization ────────────────────────────────────────────────────

def flow_to_rgb(
    flow: Union[torch.Tensor, np.ndarray],
    max_flow: Optional[float] = None,
) -> np.ndarray:
    """Convert a 2-channel optical flow field to an RGB image (Middlebury convention).

    Args:
        flow: Shape ``(2, H, W)`` or ``(H, W, 2)`` — dx, dy in pixels.
        max_flow: Normalisation ceiling for magnitude.  If *None*, uses the
            per-image maximum.

    Returns:
        ``(H, W, 3)`` uint8 RGB array.
    """
    if isinstance(flow, torch.Tensor):
        flow = flow.detach().cpu().float().numpy()

    # Accept both (2, H, W) and (H, W, 2)
    if flow.ndim == 3 and flow.shape[0] == 2:
        flow = np.transpose(flow, (1, 2, 0))  # -> (H, W, 2)

    dx, dy = flow[..., 0], flow[..., 1]
    mag = np.sqrt(dx ** 2 + dy ** 2)
    ang = np.arctan2(dy, dx)  # radians, [-pi, pi]

    if max_flow is None:
        max_flow = float(mag.max()) + 1e-8

    # HSV: hue = direction, saturation = 1, value = magnitude
    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.float32)
    hsv[..., 0] = (ang + np.pi) / (2 * np.pi)  # [0, 1]
    hsv[..., 1] = 1.0
    hsv[..., 2] = np.clip(mag / max_flow, 0.0, 1.0)

    # HSV → RGB via matplotlib (avoids cv2 dependency for this one function)
    import matplotlib.colors as mcolors
    rgb = mcolors.hsv_to_rgb(hsv)
    return (rgb * 255).astype(np.uint8)


# ── GT flow helpers ──────────────────────────────────────────────────────

def _to_flow_hw2(flow: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Return (H, W, 2) float32 dx/dy and strip optional validity channel."""
    if isinstance(flow, torch.Tensor):
        arr = flow.detach().cpu().float().numpy()
    else:
        arr = np.asarray(flow, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] in (2, 3):
        arr = np.transpose(arr, (1, 2, 0))  # -> (H, W, C)
    if arr.ndim != 3 or arr.shape[-1] < 2:
        raise ValueError(f"Expected flow with 2 or 3 channels, got shape {arr.shape}")
    return arr[..., :2].astype(np.float32)


def _flow_valid_mask(flow_gt: Union[torch.Tensor, np.ndarray]) -> Optional[np.ndarray]:
    """Return (H, W) bool validity mask if a third channel is present, else None."""
    if isinstance(flow_gt, torch.Tensor):
        arr = flow_gt.detach().cpu().float().numpy()
    else:
        arr = np.asarray(flow_gt, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 3:
        return arr[2] > 0.5
    if arr.ndim == 3 and arr.shape[-1] == 3:
        return arr[..., 2] > 0.5
    return None


def _endpoint_error(
    pred: Union[torch.Tensor, np.ndarray],
    gt: Union[torch.Tensor, np.ndarray],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Per-pixel EPE and (optional) validity mask."""
    p = _to_flow_hw2(pred)
    g = _to_flow_hw2(gt)
    if p.shape != g.shape:
        raise ValueError(f"pred/gt flow shape mismatch: {p.shape} vs {g.shape}")
    epe = np.sqrt(((p - g) ** 2).sum(axis=-1))
    return epe, _flow_valid_mask(gt)


# Cache of forward_timestamps.txt contents per sequence path.
_FLOW_INDEX_CACHE: dict = {}


def _load_sequence_flow_index(
    seq_path: Union[str, Path],
    flow_direction: str = "forward",
) -> Optional[Tuple[np.ndarray, list]]:
    """Return ((N, 2) int64 timestamps, list[Path] pngs) or None if absent."""
    key = (str(seq_path), flow_direction)
    if key in _FLOW_INDEX_CACHE:
        return _FLOW_INDEX_CACHE[key]

    flow_dir = Path(seq_path) / "flow"
    ts_path = flow_dir / f"{flow_direction}_timestamps.txt"
    png_dir = flow_dir / flow_direction
    try:
        if not flow_dir.is_dir() or not ts_path.exists() or not png_dir.is_dir():
            _FLOW_INDEX_CACHE[key] = None
            return None
        data = np.loadtxt(str(ts_path), delimiter=",", dtype="int64")
        if data.ndim == 1:
            data = data.reshape(1, -1)
        pngs = sorted(png_dir.glob("*.png"))
        if len(pngs) != data.shape[0]:
            logger.warning(
                "Flow index mismatch for %s: %d pngs vs %d ts entries",
                seq_path, len(pngs), data.shape[0],
            )
            _FLOW_INDEX_CACHE[key] = None
            return None
        _FLOW_INDEX_CACHE[key] = (data[:, :2], pngs)
        return _FLOW_INDEX_CACHE[key]
    except (PermissionError, OSError) as exc:
        logger.warning("Cannot access flow dir %s: %s", flow_dir, exc)
        _FLOW_INDEX_CACHE[key] = None
        return None
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to load flow index for %s: %s", seq_path, exc)
        _FLOW_INDEX_CACHE[key] = None
        return None


def load_gt_flow_for_frame(
    seq_path: Union[str, Path],
    t_ref_us: int,
    *,
    flow_direction: str = "forward",
    tolerance_us: int = 0,
) -> Optional[torch.Tensor]:
    """Lookup DSEC GT optical-flow PNG matching ``t_ref_us``.

    Matches on the ``to_us`` column first (flow end aligns with the frame
    timestamp that STTP reasons about), falling back to ``from_us``.

    Returns a ``(3, H, W)`` float32 tensor (flow_x, flow_y, valid) or None.
    """
    idx = _load_sequence_flow_index(seq_path, flow_direction=flow_direction)
    if idx is None:
        return None
    ts, pngs = idx
    t_ref_us = int(t_ref_us)

    # Match exact to_us first, then from_us, then nearest within tolerance.
    for col in (1, 0):
        hits = np.where(ts[:, col] == t_ref_us)[0]
        if hits.size > 0:
            png_path = pngs[int(hits[0])]
            from smcfnet.dataset.dsec.dsec_lightweight_mapper import DSECLightweightMapper
            return DSECLightweightMapper._load_flow_png(str(png_path))

    if tolerance_us > 0:
        diffs = np.abs(ts[:, 1] - t_ref_us)
        j = int(np.argmin(diffs))
        if diffs[j] <= tolerance_us:
            from smcfnet.dataset.dsec.dsec_lightweight_mapper import DSECLightweightMapper
            return DSECLightweightMapper._load_flow_png(str(pngs[j]))

    return None


# ── Single flow renderer (optionally with GT + EPE panels) ───────────────

def visualize_flow(
    flow: Union[torch.Tensor, np.ndarray],
    save_path: Union[str, Path],
    *,
    flow_gt: Optional[Union[torch.Tensor, np.ndarray]] = None,
    max_flow: Optional[float] = None,
    title: Optional[str] = None,
    dpi: int = 120,
) -> Path:
    """Render a flow field, with an optional GT + EPE comparison.

    When ``flow_gt`` is ``None`` the output is a single HSV-coded panel.
    When provided, the output has three panels: prediction | GT | EPE map
    (masked by the validity channel if present).
    """
    import matplotlib.pyplot as plt

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    pred_hw2 = _to_flow_hw2(flow)
    pred_mag = np.linalg.norm(pred_hw2, axis=-1)

    # Shared magnitude scale across pred/gt so colors are directly comparable.
    shared_max = max_flow
    if shared_max is None:
        if flow_gt is not None:
            gt_hw2 = _to_flow_hw2(flow_gt)
            shared_max = float(max(pred_mag.max(), np.linalg.norm(gt_hw2, axis=-1).max()))
        else:
            shared_max = float(pred_mag.max())
        shared_max = shared_max if shared_max > 1e-8 else 1e-8

    pred_rgb = flow_to_rgb(pred_hw2, max_flow=shared_max)

    if flow_gt is None:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(pred_rgb)
        ax.axis("off")
        caption = f"mag: mean={pred_mag.mean():.2f}, max={pred_mag.max():.2f} px"
        if title:
            caption = f"{title}\n{caption}"
        ax.set_title(caption, fontsize=9)
        fig.tight_layout(pad=0.5)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logger.debug("Saved flow viz to %s", save_path)
        return save_path

    # GT-comparable 3-panel version.
    gt_hw2 = _to_flow_hw2(flow_gt)
    gt_rgb = flow_to_rgb(gt_hw2, max_flow=shared_max)
    gt_mag = np.linalg.norm(gt_hw2, axis=-1)
    epe, valid = _endpoint_error(pred_hw2, flow_gt)
    if valid is not None:
        epe_vis = np.where(valid, epe, 0.0)
        epe_mean = float(epe[valid].mean()) if valid.any() else float("nan")
    else:
        epe_vis = epe
        epe_mean = float(epe.mean())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(pred_rgb)
    axes[0].set_title(f"Predicted\nmean={pred_mag.mean():.2f} max={pred_mag.max():.2f} px", fontsize=9)
    axes[0].axis("off")

    axes[1].imshow(gt_rgb)
    axes[1].set_title(f"GT\nmean={gt_mag.mean():.2f} max={gt_mag.max():.2f} px", fontsize=9)
    axes[1].axis("off")

    im2 = axes[2].imshow(epe_vis, cmap="hot", vmin=0.0)
    axes[2].set_title(f"EPE = |pred - gt|\nmean={epe_mean:.2f} px", fontsize=9)
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], fraction=0.04, pad=0.02)

    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout(pad=0.5)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.debug("Saved flow viz (with GT) to %s", save_path)
    return save_path


# ── IWE visualization ────────────────────────────────────────────────────

def visualize_iwe(
    voxel_before: Union[torch.Tensor, np.ndarray],
    voxel_after: Union[torch.Tensor, np.ndarray],
    save_path: Union[str, Path],
    *,
    title: Optional[str] = None,
    dpi: int = 120,
) -> Path:
    """Side-by-side IWE comparison: uncorrected vs MCM-corrected.

    The Image of Warped Events (IWE) is the sum across temporal bins.
    A sharper (higher-variance) IWE indicates better motion compensation.
    """
    import matplotlib.pyplot as plt

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    def _to_numpy(t):
        if isinstance(t, torch.Tensor):
            return t.detach().cpu().float().numpy()
        return np.asarray(t, dtype=np.float32)

    before = _to_numpy(voxel_before)
    after = _to_numpy(voxel_after)

    iwe_before = before.sum(axis=0)
    iwe_after = after.sum(axis=0)

    var_before = float(iwe_before.var())
    var_after = float(iwe_after.var())

    vmin = min(iwe_before.min(), iwe_after.min())
    vmax = max(iwe_before.max(), iwe_after.max())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].imshow(iwe_before, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Before MCM\nIWE var={var_before:.2f}", fontsize=9)
    axes[0].axis("off")

    im1 = axes[1].imshow(iwe_after, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"After MCM\nIWE var={var_after:.2f}", fontsize=9)
    axes[1].axis("off")

    fig.colorbar(im1, ax=axes, fraction=0.02, pad=0.02)

    suptitle = title or "IWE Comparison"
    delta = var_after - var_before
    sign = "+" if delta >= 0 else ""
    fig.suptitle(f"{suptitle}  ({sign}{delta:.2f} variance)", fontsize=10)

    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    logger.debug("Saved IWE viz to %s", save_path)
    return save_path


# ── Combined MCM debug panel ─────────────────────────────────────────────

def visualize_mcm_debug(
    voxel_before: Union[torch.Tensor, np.ndarray],
    voxel_after: Union[torch.Tensor, np.ndarray],
    flow: Union[torch.Tensor, np.ndarray],
    save_path: Union[str, Path],
    *,
    flow_gt: Optional[Union[torch.Tensor, np.ndarray]] = None,
    title: Optional[str] = None,
    max_flow: Optional[float] = None,
    dpi: int = 120,
) -> Path:
    """Four-panel MCM debug image: IWE-after | MCM flow | GT flow | EPE.

    When ``flow_gt`` is unavailable the GT panel is replaced with a
    placeholder and the EPE panel falls back to per-pixel correction
    magnitude (|IWE_after - IWE_before|) so the frame is still informative.
    """
    import matplotlib.pyplot as plt

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    def _np(t):
        if isinstance(t, torch.Tensor):
            return t.detach().cpu().float().numpy()
        return np.asarray(t, dtype=np.float32)

    before = _np(voxel_before)
    after = _np(voxel_after)
    pred_hw2 = _to_flow_hw2(flow)
    pred_mag = np.linalg.norm(pred_hw2, axis=-1)

    iwe_after = after.sum(axis=0)
    iwe_before = before.sum(axis=0)
    var_after = float(iwe_after.var())

    shared_max = max_flow
    if shared_max is None:
        if flow_gt is not None:
            gt_hw2 = _to_flow_hw2(flow_gt)
            shared_max = float(max(pred_mag.max(), np.linalg.norm(gt_hw2, axis=-1).max()))
        else:
            shared_max = float(pred_mag.max())
        shared_max = shared_max if shared_max > 1e-8 else 1e-8

    flow_rgb = flow_to_rgb(pred_hw2, max_flow=shared_max)

    vmin = float(min(iwe_before.min(), iwe_after.min()))
    vmax = float(max(iwe_before.max(), iwe_after.max()))

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    axes[0].imshow(iwe_after, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"IWE after\nvar={var_after:.2f}", fontsize=9)
    axes[0].axis("off")

    axes[1].imshow(flow_rgb)
    axes[1].set_title(
        f"MCM flow\nmean={pred_mag.mean():.2f} max={pred_mag.max():.2f} px",
        fontsize=9,
    )
    axes[1].axis("off")

    if flow_gt is not None:
        gt_hw2 = _to_flow_hw2(flow_gt)
        gt_mag = np.linalg.norm(gt_hw2, axis=-1)
        gt_rgb = flow_to_rgb(gt_hw2, max_flow=shared_max)
        axes[2].imshow(gt_rgb)
        axes[2].set_title(
            f"GT flow\nmean={gt_mag.mean():.2f} max={gt_mag.max():.2f} px",
            fontsize=9,
        )
        axes[2].axis("off")

        epe, valid = _endpoint_error(pred_hw2, flow_gt)
        if valid is not None:
            epe_vis = np.where(valid, epe, 0.0)
            epe_mean = float(epe[valid].mean()) if valid.any() else float("nan")
        else:
            epe_vis = epe
            epe_mean = float(epe.mean())
        im3 = axes[3].imshow(epe_vis, cmap="hot", vmin=0.0)
        axes[3].set_title(f"EPE (pred - gt)\nmean={epe_mean:.2f} px", fontsize=9)
        axes[3].axis("off")
        fig.colorbar(im3, ax=axes[3], fraction=0.04, pad=0.02)
    else:
        # No GT available: leave a placeholder and show correction magnitude.
        axes[2].imshow(np.zeros_like(iwe_after), cmap="gray", vmin=0, vmax=1)
        axes[2].text(
            0.5, 0.5, "GT flow\nunavailable",
            transform=axes[2].transAxes,
            ha="center", va="center", color="white", fontsize=11,
        )
        axes[2].set_title("GT flow", fontsize=9)
        axes[2].axis("off")

        diff = np.abs(after - before).sum(axis=0)
        im3 = axes[3].imshow(diff, cmap="hot")
        axes[3].set_title("Correction magnitude\n(|after - before|)", fontsize=9)
        axes[3].axis("off")
        fig.colorbar(im3, ax=axes[3], fraction=0.04, pad=0.02)

    if title:
        fig.suptitle(title, fontsize=11)

    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    logger.debug("Saved MCM debug viz to %s", save_path)
    return save_path
