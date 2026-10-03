"""Detector-facing event output renderers for MCM/SMCM.

These modules intentionally render only the event representation forwarded to
EDUM/the event backbone. The self-supervised event-warping loss keeps its own
IWE construction in ``EventWarpingLoss``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from smcfnet.model.event_warping import (
    get_interpolation,
    interpolate_to_iwe_channels,
    prepare_events_for_warping,
)
from smcfnet.utils.event_utils import events_to_voxel_grid_torch


class EventOutputRenderer(nn.Module, ABC):
    """Base class for MCM/SMCM downstream event representations."""

    output_channels: int
    output_type: str

    @abstractmethod
    def forward(
        self,
        *,
        flow: torch.Tensor,
        events: torch.Tensor,
        events_offsets: torch.Tensor,
        voxel: Optional[torch.Tensor] = None,
        t_start_us: Optional[torch.Tensor] = None,
        t_end_us: Optional[torch.Tensor] = None,
        t_ref_us: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError


class _SparseEventOutputBase(EventOutputRenderer):
    """Shared sparse event flow lookup for detector-output renderers."""

    def __init__(
        self,
        *,
        res: Tuple[int, int] = (480, 640),
        flow_scaling: float = 1.0,
        t_ref: float = 1.0,
    ):
        super().__init__()
        self.res = (int(res[0]), int(res[1]))
        self.flow_scaling = float(flow_scaling)
        self.t_ref = float(t_ref)

    def _prepare_events(
        self,
        events: torch.Tensor,
        events_offsets: torch.Tensor,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        events_batch, pol_mask = prepare_events_for_warping(events, events_offsets)
        return events_batch.to(dtype), pol_mask.to(dtype)

    def _lookup_event_flow(self, events_batch: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        b, _, h, w = flow.shape
        if (h, w) != self.res:
            raise ValueError(
                f"Renderer resolution {self.res} does not match flow shape {(h, w)}"
            )

        flow_idx = events_batch[:, :, 1:3].clone()
        flow_idx[:, :, 0] *= self.res[1]
        flow_idx = torch.sum(flow_idx, dim=2).clamp(
            0,
            self.res[0] * self.res[1] - 1,
        ).long()

        flow_flat = flow.view(b, 2, -1)
        event_flowy = torch.gather(flow_flat[:, 1, :], 1, flow_idx)
        event_flowx = torch.gather(flow_flat[:, 0, :], 1, flow_idx)
        return torch.cat([event_flowy.unsqueeze(2), event_flowx.unsqueeze(2)], dim=2)

    def _require_packed_events(
        self,
        events: Optional[torch.Tensor],
        events_offsets: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if events is None or events_offsets is None:
            raise ValueError(
                f"{self.__class__.__name__} requires 'events' and 'events_offsets' "
                "to render sparse event outputs."
            )
        return events, events_offsets


class VoxelGridEventOutput(_SparseEventOutputBase):
    """Warp sparse events and re-voxelize them with triangular time bins."""

    output_type = "voxel_grid"

    def __init__(
        self,
        *,
        num_bins: int = 5,
        res: Tuple[int, int] = (480, 640),
        flow_scaling: float = 1.0,
        t_ref: float = 1.0,
    ):
        super().__init__(res=res, flow_scaling=flow_scaling, t_ref=t_ref)
        self.num_bins = int(num_bins)
        if self.num_bins <= 0:
            raise ValueError(f"num_bins must be > 0, got {self.num_bins}")
        self.output_channels = self.num_bins

    def forward(
        self,
        *,
        flow: torch.Tensor,
        events: torch.Tensor,
        events_offsets: torch.Tensor,
        voxel: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        events, events_offsets = self._require_packed_events(events, events_offsets)
        events_batch, _ = self._prepare_events(events, events_offsets, flow.dtype)
        event_flow = self._lookup_event_flow(events_batch, flow)

        warped_locations = (
            events_batch[:, :, 1:3]
            + (self.t_ref - events_batch[:, :, 0:1]) * event_flow * self.flow_scaling
        )
        warped_events = torch.cat(
            [events_batch[:, :, 0:1], warped_locations, events_batch[:, :, 3:4]],
            dim=2,
        )
        return self._voxelize_batched_events(warped_events, events_offsets, flow.device)

    def _voxelize_batched_events(
        self,
        events_batch: torch.Tensor,
        offsets: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        batch_size = int(offsets.numel() - 1)
        voxels = []

        for batch_index in range(batch_size):
            start = int(offsets[batch_index].item())
            end = int(offsets[batch_index + 1].item())
            n_valid = end - start

            if n_valid == 0:
                voxels.append(
                    torch.zeros(
                        (self.num_bins, self.res[0], self.res[1]),
                        device=device,
                        dtype=torch.float32,
                    )
                )
                continue

            events_b = events_batch[batch_index, :n_valid, :].clone()
            valid_mask = (events_b[:, 1] >= 0) & (events_b[:, 2] >= 0)
            events_b = events_b[valid_mask]

            if events_b.numel() == 0:
                vox_i = torch.zeros(
                    (self.num_bins, self.res[0], self.res[1]),
                    device=device,
                    dtype=torch.float32,
                )
            else:
                events_xytp = torch.stack(
                    [events_b[:, 2], events_b[:, 1], events_b[:, 0], events_b[:, 3]],
                    dim=1,
                )
                vox_i = events_to_voxel_grid_torch(
                    events_xytp,
                    num_bins=self.num_bins,
                    width=self.res[1],
                    height=self.res[0],
                )
            voxels.append(vox_i)

        if not voxels:
            return torch.zeros(
                (0, self.num_bins, self.res[0], self.res[1]),
                device=device,
                dtype=torch.float32,
            )
        return torch.stack(voxels, dim=0)


class LiuRGBIWEEventOutput(_SparseEventOutputBase):
    """Liu-style color-coded IWE: zero red, positive green, negative blue."""

    output_type = "iwe_liu_rgb"
    output_channels = 3

    def __init__(
        self,
        *,
        res: Tuple[int, int] = (480, 640),
        flow_scaling: float = 1.0,
        t_ref: float = 1.0,
        channels: str = "liu_rgb",
        normalize: str = "p99",
        scale: float = 1.0,
        splat: str = "bilinear",
        align_to_rgb: str = "auto",
        eps: float = 1e-6,
    ):
        super().__init__(res=res, flow_scaling=flow_scaling, t_ref=t_ref)
        self.channels = str(channels or "liu_rgb").lower()
        self.normalize = str(normalize or "p99").lower()
        self.scale = float(scale)
        self.splat = str(splat or "bilinear").lower()
        self.align_to_rgb = str(align_to_rgb or "auto").lower()
        self.eps = float(eps)

        if self.channels != "liu_rgb":
            raise ValueError(f"Unsupported IWE channel encoding {channels!r}; expected 'liu_rgb'")
        if self.normalize not in {"p99", "none"}:
            raise ValueError(f"Unsupported IWE normalization {normalize!r}; expected 'p99' or 'none'")
        if self.splat != "bilinear":
            raise ValueError(f"Unsupported IWE splat mode {splat!r}; expected 'bilinear'")
        if self.align_to_rgb not in {"auto", "none", "mapping"}:
            raise ValueError(
                f"Unsupported iwe_align_to_rgb {align_to_rgb!r}; expected 'auto', 'none', or 'mapping'"
            )

    def forward(
        self,
        *,
        flow: torch.Tensor,
        events: torch.Tensor,
        events_offsets: torch.Tensor,
        voxel: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        events, events_offsets = self._require_packed_events(events, events_offsets)
        events_batch, pol_mask = self._prepare_events(events, events_offsets, flow.dtype)
        event_flow = self._lookup_event_flow(events_batch, flow)

        idx, weights = get_interpolation(
            events_batch,
            event_flow,
            self.t_ref,
            self.res,
            self.flow_scaling,
        )
        pol_mask_4 = torch.cat([pol_mask] * 4, dim=1)

        iwe_pol = interpolate_to_iwe_channels(
            idx,
            weights,
            self.res,
            channel_weights=pol_mask_4,
        )

        if self.normalize == "p99":
            iwe_pol = self._normalize_p99(iwe_pol)

        red_zero = torch.zeros_like(iwe_pol[:, 0:1])
        out = torch.cat([red_zero, iwe_pol[:, 0:1], iwe_pol[:, 1:2]], dim=1)
        if self.scale != 1.0:
            out = out * self.scale
        return self._apply_alignment(out, **kwargs)

    def _normalize_p99(self, iwe: torch.Tensor) -> torch.Tensor:
        flat = iwe.flatten(start_dim=2).float()  # quantile() requires float/double
        q = torch.quantile(flat, 0.99, dim=2, keepdim=True).view(iwe.shape[0], iwe.shape[1], 1, 1).to(iwe.dtype)
        denom = q + self.eps
        normalized = torch.clamp(iwe / denom, min=0.0, max=1.0)
        return torch.where(q > self.eps, normalized, torch.zeros_like(iwe))

    def _apply_alignment(self, out: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.align_to_rgb == "none":
            return out

        mapping = (
            kwargs.get("iwe_mapping")
            if kwargs.get("iwe_mapping") is not None
            else kwargs.get("event_to_rgb_mapping", kwargs.get("mapping"))
        )
        if mapping is None:
            if self.align_to_rgb == "mapping":
                raise ValueError(
                    "iwe_align_to_rgb='mapping' requires an 'iwe_mapping', "
                    "'event_to_rgb_mapping', or 'mapping' tensor in the batch."
                )
            return out

        if mapping.ndim != 4 or mapping.shape[-1] != 2:
            raise ValueError(
                "IWE alignment mapping must have shape (B, H, W, 2) in "
                "torch.grid_sample normalized coordinates."
            )
        return F.grid_sample(out, mapping.to(device=out.device, dtype=out.dtype), align_corners=True)


def create_event_output_renderer(
    *,
    output_type: str = "voxel_grid",
    num_bins: int = 5,
    res: Tuple[int, int] = (480, 640),
    t_ref: float = 1.0,
    iwe_channels: str = "liu_rgb",
    iwe_normalize: str = "p99",
    iwe_scale: float = 1.0,
    iwe_splat: str = "bilinear",
    iwe_align_to_rgb: str = "auto",
) -> EventOutputRenderer:
    """Create a detector-facing event output renderer from config values."""

    mode = str(output_type or "voxel_grid").lower()
    if mode == "voxel_grid":
        return VoxelGridEventOutput(
            num_bins=num_bins,
            res=res,
            flow_scaling=1.0,
            t_ref=t_ref,
        )
    if mode == "iwe_liu_rgb":
        return LiuRGBIWEEventOutput(
            res=res,
            flow_scaling=1.0,
            t_ref=t_ref,
            channels=iwe_channels,
            normalize=iwe_normalize,
            scale=iwe_scale,
            splat=iwe_splat,
            align_to_rgb=iwe_align_to_rgb,
        )
    raise ValueError(f"Unsupported MCM output renderer type {output_type!r}")
