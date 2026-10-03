"""Trajectory prediction metrics and TTC proxy.

Standalone functions for ADE, FDE, and center-displacement TTC proxy.
Used by both the STTP head (training-time logging) and the offline
evaluation script.
"""

from __future__ import annotations

import torch


def compute_ade(
    preds: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Average Displacement Error over valid future steps.

    Args:
        preds: ``[N, T, 2]`` predicted (dx, dy) displacements.
        targets: ``[N, T, 2]`` ground-truth (dx, dy) displacements.
        valid: ``[N, T]`` binary mask (1 = valid step).

    Returns:
        Scalar ADE (mean L2 across all valid steps). Returns 0 if no valid steps.
    """
    n_valid = valid.sum().clamp(min=1)
    l2 = (preds - targets).norm(dim=2)  # [N, T]
    return (l2 * valid).sum() / n_valid


def compute_fde(
    preds: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Final Displacement Error at each object's last valid step.

    Args:
        preds: ``[N, T, 2]`` predicted (dx, dy) displacements.
        targets: ``[N, T, 2]`` ground-truth (dx, dy) displacements.
        valid: ``[N, T]`` binary mask (1 = valid step).

    Returns:
        Scalar FDE (mean L2 at last valid step per object). Returns 0 if no
        valid steps.
    """
    T = valid.shape[1]
    l2 = (preds - targets).norm(dim=2)  # [N, T]
    has_any = valid.sum(dim=1) > 0  # [N]
    if not has_any.any():
        return torch.tensor(0.0, device=preds.device)
    # Last valid index per object via flip+argmax
    last_idx = (T - 1) - valid[has_any].flip(1).argmax(dim=1)
    fde_per_obj = l2[has_any].gather(1, last_idx.unsqueeze(1)).squeeze(1)
    return fde_per_obj.mean()


def compute_ade_fde(
    preds: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute both ADE and FDE in one call (shared L2 computation).

    Args:
        preds: ``[N, T, 2]`` predicted (dx, dy) displacements.
        targets: ``[N, T, 2]`` ground-truth (dx, dy) displacements.
        valid: ``[N, T]`` binary mask (1 = valid step).

    Returns:
        ``(ade, fde)`` — both scalar tensors.
    """
    _zero = torch.tensor(0.0, device=preds.device)
    n_valid = valid.sum().clamp(min=1)
    l2 = (preds - targets).norm(dim=2)  # [N, T]
    ade = (l2 * valid).sum() / n_valid

    T = valid.shape[1]
    has_any = valid.sum(dim=1) > 0
    if not has_any.any():
        return ade, _zero
    last_idx = (T - 1) - valid[has_any].flip(1).argmax(dim=1)
    fde = l2[has_any].gather(1, last_idx.unsqueeze(1)).squeeze(1).mean()
    return ade, fde


def compute_ttc_proxy(
    preds: torch.Tensor,
    current_centers: torch.Tensor,
    valid: torch.Tensor,
    dt: float = 0.05,
    ego_position: torch.Tensor | None = None,
) -> torch.Tensor:
    """Center-displacement TTC proxy from predicted trajectories.

    Approximates time-to-collision from the rate of change of the predicted
    center distance to the ego (image center by default). Objects whose
    distance is increasing (moving away) get TTC = inf.

    Args:
        preds: ``[N, T, 2]`` predicted (dx, dy) displacements from current center.
        current_centers: ``[N, 2]`` current (cx, cy) in pixel coordinates.
        valid: ``[N, T]`` binary mask.
        dt: Time interval between consecutive future steps (seconds).
        ego_position: ``[2]`` ego reference point in pixels. Defaults to
            image center estimated as ``mean(current_centers)``.

    Returns:
        ``[N]`` TTC proxy in seconds. ``inf`` for objects moving away or
        with insufficient valid steps.
    """
    N, T, _ = preds.shape
    inf = torch.tensor(float("inf"), device=preds.device)
    ttc = inf.expand(N).clone()

    if ego_position is None:
        ego_position = current_centers.mean(dim=0)  # [2]

    # Absolute future positions: current_center + cumulative displacement
    # preds are per-step displacements from current center (not cumulative)
    future_abs = current_centers.unsqueeze(1) + preds  # [N, T, 2]

    # Distance to ego at each future step
    dist_to_ego = (future_abs - ego_position.unsqueeze(0).unsqueeze(0)).norm(
        dim=2
    )  # [N, T]

    # Current distance to ego
    dist_current = (current_centers - ego_position.unsqueeze(0)).norm(dim=1)  # [N]

    # For each object, compute closing velocity from valid steps
    for i in range(N):
        valid_mask = valid[i].bool()
        if valid_mask.sum() < 1:
            continue

        # Use last valid step to compute average closing velocity
        valid_indices = valid_mask.nonzero(as_tuple=True)[0]
        last_valid = valid_indices[-1].item()

        d_current = dist_current[i]
        d_future = dist_to_ego[i, last_valid]
        time_elapsed = (last_valid + 1) * dt

        # Closing velocity (positive = approaching)
        closing_vel = (d_current - d_future) / time_elapsed

        if closing_vel > 1e-6:  # approaching
            ttc[i] = d_current / closing_vel

    return ttc


def ttc_center_closing(
    curr_center: torch.Tensor,
    fut_center: torch.Tensor,
    dt_total: float,
    ego: torch.Tensor,
) -> torch.Tensor:
    """Center-closing TTC (seconds) for a single current->future center pair.

    Same formula as ``compute_ttc_proxy`` (distance-to-ego / closing-velocity) but for
    one future box per object at elapsed time ``dt_total`` seconds, with an EXPLICIT ego
    point (the image-center focus-of-expansion, never the per-batch mean, which would be
    degenerate for a single object). ``inf`` for objects not approaching the ego point.

    Args:
        curr_center: ``[N, 2]`` current (cx, cy) pixels.
        fut_center: ``[N, 2]`` predicted future (cx, cy) pixels at t + dt_total.
        dt_total: elapsed seconds between current and future (e.g. k * 0.05).
        ego: ``[2]`` ego reference point (image center) in pixels.

    Returns:
        ``[N]`` TTC seconds, ``inf`` when receding or static.
    """
    d_curr = (curr_center - ego).norm(dim=1)               # [N]
    d_fut = (fut_center - ego).norm(dim=1)                 # [N]
    closing_vel = (d_curr - d_fut) / max(dt_total, 1e-9)   # px/s, positive = approaching
    ttc = torch.full_like(d_curr, float("inf"))
    approaching = closing_vel > 1e-6
    ttc[approaching] = d_curr[approaching] / closing_vel[approaching]
    return ttc


def ttc_looming(
    curr_w: torch.Tensor,
    fut_w: torch.Tensor,
    dt_total: float,
) -> torch.Tensor:
    """Looming (box-scale expansion) TTC in seconds: TTC = w / (dw/dt).

    Monocular, ego-free time-to-contact from the apparent size growth rate. ``inf`` for
    objects whose box is not expanding (dw <= 0), e.g. receding or lateral-crossing
    objects whose width does not change.

    Args:
        curr_w: ``[N]`` current box width (pixels).
        fut_w: ``[N]`` predicted future width (pixels) at t + dt_total.
        dt_total: elapsed seconds between current and future.

    Returns:
        ``[N]`` TTC seconds, ``inf`` when not expanding.
    """
    dw_dt = (fut_w - curr_w) / max(dt_total, 1e-9)   # px/s
    ttc = torch.full_like(curr_w, float("inf"))
    expanding = dw_dt > 1e-6
    ttc[expanding] = curr_w[expanding] / dw_dt[expanding]
    return ttc
