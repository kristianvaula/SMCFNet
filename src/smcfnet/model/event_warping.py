"""Sparse event forward-warping utilities.

Implements the forward-splatting IWE (Image of Warped Events) pipeline
from Zhu et al., CVPR'19, as used in the original MCFNet ECM.

This operates on raw events (not voxel grids), preserving per-event
timing and polarity information for the averaged-timestamp loss.
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)
_POLARITY_DEBUG_CALLS_REMAINING: int | None = None


def _should_log_polarity_debug() -> bool:
    """Gate optional polarity logging via an environment variable.

    ``SMCFNET_DEBUG_EVENT_POLARITY_CALLS=<N>`` enables rank-0 logging for the
    first ``N`` calls to ``prepare_events_for_warping`` in a process.
    """
    global _POLARITY_DEBUG_CALLS_REMAINING

    if _POLARITY_DEBUG_CALLS_REMAINING is None:
        raw_value = os.getenv("SMCFNET_DEBUG_EVENT_POLARITY_CALLS", "").strip()
        if not raw_value:
            _POLARITY_DEBUG_CALLS_REMAINING = 0
        else:
            try:
                _POLARITY_DEBUG_CALLS_REMAINING = max(int(raw_value), 0)
            except ValueError:
                logger.warning(
                    "Ignoring invalid SMCFNET_DEBUG_EVENT_POLARITY_CALLS=%r; "
                    "expected a non-negative integer.",
                    raw_value,
                )
                _POLARITY_DEBUG_CALLS_REMAINING = 0

    if _POLARITY_DEBUG_CALLS_REMAINING <= 0:
        return False

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() != 0:
            return False

    _POLARITY_DEBUG_CALLS_REMAINING -= 1
    return True


def _maybe_log_polarity_debug(
    events_packed: torch.Tensor,
    pol_mask: torch.Tensor,
    valid: torch.Tensor,
) -> None:
    """Emit one lightweight summary of runtime polarity values and masks."""
    if not _should_log_polarity_debug():
        return

    raw_polarities = events_packed[:, 3].detach()
    unique_vals, unique_counts = torch.unique(raw_polarities, return_counts=True)
    unique_summary = list(
        zip(
            unique_vals.to("cpu").tolist(),
            unique_counts.to("cpu").tolist(),
        )
    )

    valid_counts = valid.sum(dim=1).to(torch.int64).to("cpu").tolist()
    pos_counts = pol_mask[:, :, 0].sum(dim=1).to(torch.int64).to("cpu").tolist()
    neg_counts = pol_mask[:, :, 1].sum(dim=1).to(torch.int64).to("cpu").tolist()

    logger.info(
        "Event polarity debug: raw_unique=%s valid_events_per_sample=%s "
        "pos_mask_per_sample=%s neg_mask_per_sample=%s",
        unique_summary,
        valid_counts,
        pos_counts,
        neg_counts,
    )


def purge_unfeasible(
    x: torch.Tensor, res: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero out interpolation weights for events warped outside image bounds.

    Args:
        x: (B, N, 2) warped event locations (y, x).
        res: (H, W) image resolution.

    Returns:
        Masked locations and binary mask (B, N, 1).
    """
    mask = torch.ones((x.shape[0], x.shape[1], 1), device=x.device, dtype=x.dtype)
    mask_y = (x[:, :, 0:1] < 0) | (x[:, :, 0:1] >= res[0])
    mask_x = (x[:, :, 1:2] < 0) | (x[:, :, 1:2] >= res[1])
    mask[mask_y | mask_x] = 0
    return x * mask, mask


def get_interpolation(
    events: torch.Tensor,
    flow: torch.Tensor,
    tref: float,
    res: tuple[int, int],
    flow_scaling: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward-warp events and compute bilinear splatting weights.

    Each event is warped to a new location and distributed to 4 neighboring
    integer pixels via bilinear interpolation weights.

    Args:
        events: (B, N, 4) input events in **(ts, y, x, p)** format.
            ``ts`` must be normalized to [0, 1].
        flow: (B, N, 2) per-event optical flow **(flow_y, flow_x)**.
        tref: Reference time (1.0=forward, 0.0=backward).
        res: (H, W) image resolution.
        flow_scaling: Scalar multiplier for the optical flow.

    Returns:
        idx: (B, 4N, 1) linear pixel indices for the four splatting corners.
        weights: (B, 4N, 1) bilinear interpolation weights.
    """
    # Warped position = (y, x) + (tref - ts) * flow * scaling
    warped_events = events[:, :, 1:3] + (tref - events[:, :, 0:1]) * flow * flow_scaling

    # Four integer corners for bilinear splatting. Keep the same corner order as
    # the legacy implementation: top-left, top-right, bottom-left, bottom-right.
    y0 = torch.floor(warped_events[:, :, 0:1])
    x0 = torch.floor(warped_events[:, :, 1:2])
    y1 = y0 + 1
    x1 = x0 + 1

    fy = warped_events[:, :, 0:1] - y0
    fx = warped_events[:, :, 1:2] - x0
    weights = torch.cat(
        [
            (1.0 - fy) * (1.0 - fx),
            (1.0 - fy) * fx,
            fy * (1.0 - fx),
            fy * fx,
        ],
        dim=1,
    )

    corner_y = torch.cat([y0, y0, y1, y1], dim=1)
    corner_x = torch.cat([x0, x1, x0, x1], dim=1)
    valid = (
        (corner_y >= 0)
        & (corner_y < res[0])
        & (corner_x >= 0)
        & (corner_x < res[1])
    )
    weights = weights * valid.to(dtype=weights.dtype)

    # Convert 2D (y, x) to linear index: y * W + x. Invalid corners receive a
    # zero weight and index 0 to keep scatter_add_ indices in range.
    # Cast to long before arithmetic to avoid float16 overflow at large resolutions
    # (float16 max 65504 < 52 * 1280 = 66560 → +Inf → INT64_MAX OOB scatter).
    idx = corner_y.long() * res[1] + corner_x.long()
    idx = idx.masked_fill(~valid, 0)

    return idx, weights


def interpolate_to_iwe(
    idx: torch.Tensor,
    weights: torch.Tensor,
    res: tuple[int, int],
    polarity_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scatter-accumulate warped events into an image.

    Args:
        idx: (B, N, 1) linear pixel indices (long).
        weights: (B, N, 1) interpolation weights.
        res: (H, W) image resolution.
        polarity_mask: (B, N, 1) optional per-polarity binary mask.

    Returns:
        iwe: (B, 1, H, W) image of warped events.
    """
    if polarity_mask is not None:
        weights = weights * polarity_mask
    iwe = torch.zeros(
        (idx.shape[0], res[0] * res[1], 1), device=idx.device, dtype=weights.dtype,
    )
    iwe = iwe.scatter_add_(1, idx.long(), weights)
    iwe = iwe.view((idx.shape[0], 1, res[0], res[1]))
    return iwe


def interpolate_to_iwe_channels(
    idx: torch.Tensor,
    weights: torch.Tensor,
    res: tuple[int, int],
    channel_weights: torch.Tensor,
) -> torch.Tensor:
    """Scatter-accumulate warped events into a multi-channel image.

    Args:
        idx: (B, N, 1) linear pixel indices.
        weights: (B, N, 1) interpolation weights.
        res: (H, W) image resolution.
        channel_weights: (B, N, C) per-channel event weights or masks.

    Returns:
        iwe: (B, C, H, W) multi-channel image of warped events.
    """
    if channel_weights.ndim != 3:
        raise ValueError(
            f"channel_weights must have shape (B, N, C), got {tuple(channel_weights.shape)}"
        )
    if channel_weights.shape[:2] != weights.shape[:2]:
        raise ValueError(
            "channel_weights and weights must agree on batch/event dimensions, "
            f"got {tuple(channel_weights.shape[:2])} and {tuple(weights.shape[:2])}"
        )

    splat_weights = weights * channel_weights
    iwe = torch.zeros(
        (idx.shape[0], res[0] * res[1], channel_weights.shape[2]),
        device=idx.device,
        dtype=splat_weights.dtype,
    )
    scatter_idx = idx.long().expand(-1, -1, channel_weights.shape[2])
    iwe = iwe.scatter_add_(1, scatter_idx, splat_weights)
    return iwe.permute(0, 2, 1).reshape(
        idx.shape[0],
        channel_weights.shape[2],
        res[0],
        res[1],
    )


def prepare_events_for_warping(
    events_packed: torch.Tensor,
    offsets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert packed events to batched format for the event warping loss.

    Args:
        events_packed: (N_total, 4) with columns **[x, y, t_ms, p]**
            (our DSEC convention).
        offsets: (B+1,) int64 sample boundary offsets.

    Returns:
        events_batch: (B, N_max, 4) in **(ts_norm, y, x, p)** format
            with ``ts_norm`` in [0, 1].  Padded samples use coordinates
            (-1, -1) so they are zeroed by ``purge_unfeasible``.
        pol_mask: (B, N_max, 2) per-polarity binary mask
            ``[:, :, 0]`` = positive (p>0), ``[:, :, 1]`` = negative (p<=0).
            This supports both DSEC's signed ``{-1, +1}`` encoding and older
            synthetic tests that use ``{0, 1}``.
    """
    bsz = int(offsets.numel() - 1)
    device = events_packed.device

    # Find max event count for padding
    counts = offsets[1:] - offsets[:-1]
    n_max = int(counts.max().item()) if bsz > 0 else 0

    if n_max == 0:
        events_batch = torch.zeros((bsz, 1, 4), device=device, dtype=torch.float32)
        events_batch[:, :, 1:3] = -1  # unfeasible coords
        pol_mask = torch.zeros((bsz, 1, 2), device=device, dtype=torch.float32)
        return events_batch, pol_mask

    events_batch = torch.zeros((bsz, n_max, 4), device=device, dtype=torch.float32)
    # Fill padding with unfeasible coords so purge_unfeasible zeros them out
    events_batch[:, :, 1] = -1  # y
    events_batch[:, :, 2] = -1  # x

    for i in range(bsz):
        start = int(offsets[i].item())
        end = int(offsets[i + 1].item())
        n_i = end - start
        if n_i == 0:
            continue
        ev_i = events_packed[start:end].float()  # (n_i, 4) [x, y, t_ms, p]

        # Normalize timestamps to [0, 1]
        t = ev_i[:, 2]
        t_min = t.min()
        t_max = t.max()
        t_range = t_max - t_min
        if t_range > 0:
            ts_norm = (t - t_min) / t_range
        else:
            ts_norm = torch.zeros_like(t)

        # Reorder: (x, y, t_ms, p) -> (ts_norm, y, x, p)
        events_batch[i, :n_i, 0] = ts_norm
        events_batch[i, :n_i, 1] = ev_i[:, 1]  # y
        events_batch[i, :n_i, 2] = ev_i[:, 0]  # x
        events_batch[i, :n_i, 3] = ev_i[:, 3]  # p

    # Polarity masks
    pol_mask = torch.zeros((bsz, n_max, 2), device=device, dtype=torch.float32)
    pol_mask[:, :, 0] = (events_batch[:, :, 3] > 0).float()   # positive
    pol_mask[:, :, 1] = (events_batch[:, :, 3] <= 0).float()  # negative
    # Zero out polarity for padded events (coords are -1, p=0 → neg mask=1)
    # Fix: padded events have y=-1, so they shouldn't have pol_mask set
    valid = (events_batch[:, :, 1] >= 0) & (events_batch[:, :, 2] >= 0)  # (B, N)
    pol_mask[:, :, 0] *= valid.float()
    pol_mask[:, :, 1] *= valid.float()
    _maybe_log_polarity_debug(events_packed, pol_mask, valid)

    return events_batch, pol_mask
