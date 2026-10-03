"""
Cross-Modal Mamba Fusion Module (CMM).

Implements cross-modal interlaced scanning fusion at backbone levels
(dark3/dark4/dark5). The module follows MCFNet's reference flow
(``MCFNet/joint/yolox/models/darknet_mamba.py``):

1) Per-stream affine projection (separate gamma, beta for event and RGB).
2) Column-interleaved stacking along width: (B, C, H, 2W) with event
   and RGB columns alternating pixelwise.
3) Global interaction via selective state-space scan (SS2D/Mamba).
4) Split SS2D output back into per-stream tensors.
5) Residual addition to preserve original modality information.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    selective_scan_fn = None


class SS2D(nn.Module):
    """2D selective scan block used by CMM.

    Input/Output shape: (B, H, W, C).
    """

    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=2.0,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.0,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
    ):
        if selective_scan_fn is None:
            raise ImportError(
                "CMM requires mamba_ssm. Install it or disable CMM "
                "(set model.cmm.enabled=False / --no-cmm)."
            )

        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    @staticmethod
    def dt_init(
        dt_rank,
        d_inner,
        dt_scale=1.0,
        dt_init="random",
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        **factory_kwargs,
    ):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4
        x_hwwh = torch.stack(
            [x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
            dim=1,
        ).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)
        out_y = self.selective_scan(
            xs,
            dts,
            As,
            Bs,
            Cs,
            Ds,
            z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor):
        B, H, W, _ = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class CrossModalMambaFusionModule(nn.Module):
    """Cross-modal Mamba Fusion Module.

    This module consumes same-resolution event and RGB features and returns
    refined modality-specific outputs with residual preservation.
    """

    def __init__(
        self,
        channels: int,
        d_state: int = 16,
        d_conv: int = 3,
        expand: float = 2.0,
        dt_rank: str = "auto",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.channels = channels

        # Per-stream affine projection. MCFNet uses one LinearTransformLayer
        # per modality per scale (gamma_h, beta_h of shape (C, 1, 1));
        # mirrored here as four parameters keyed by modality.
        self.gamma_rgb = nn.Parameter(torch.ones(channels, 1, 1))
        self.beta_rgb = nn.Parameter(torch.zeros(channels, 1, 1))
        self.gamma_event = nn.Parameter(torch.ones(channels, 1, 1))
        self.beta_event = nn.Parameter(torch.zeros(channels, 1, 1))

        self.ssm = SS2D(
            d_model=channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_rank=dt_rank,
            dropout=dropout,
        )

    def forward(self, feat_event: torch.Tensor, feat_rgb: torch.Tensor):
        if feat_event.shape != feat_rgb.shape:
            raise ValueError(
                f"CMM expects event/rgb shapes to match, got {feat_event.shape} and {feat_rgb.shape}."
            )

        input_dtype = feat_event.dtype
        device_type = "cuda" if feat_event.is_cuda else "cpu"

        # selective_scan is substantially more fragile than the surrounding
        # YOLOX convolutions under AMP. Keep the whole fusion block in FP32 so
        # overflows do not poison the CUDA stream and surface later as NCCL
        # watchdog failures.
        with torch.amp.autocast(device_type, enabled=False):
            feat_event = feat_event.float()
            feat_rgb = feat_rgb.float()
            feat_event_out, feat_rgb_out = self._forward_fp32(feat_event, feat_rgb)

        return feat_event_out.to(dtype=input_dtype), feat_rgb_out.to(dtype=input_dtype)

    def _forward_fp32(self, feat_event: torch.Tensor, feat_rgb: torch.Tensor):
        B, C, H, W = feat_event.shape

        # 1) Per-stream affine projection.
        z_event = feat_event * self.gamma_event + self.beta_event
        z_rgb = feat_rgb * self.gamma_rgb + self.beta_rgb

        # 2) Column-interleaved stacking. cat on a new last dim gives
        # (B, C, H, W, 2) with event at index 0 and rgb at index 1; the
        # subsequent view(B, C, H, 2W) interleaves modalities along width
        # (event at even columns, rgb at odd columns) so the SS2D scan
        # alternates per column rather than processing two blocks.
        merged = torch.cat([z_event.unsqueeze(4), z_rgb.unsqueeze(4)], dim=4)
        ssm_in = (
            merged.contiguous()
            .view(B, C, H, 2 * W)
            .permute(0, 2, 3, 1)
            .contiguous()
        )

        # 3) Global interaction via SS2D. Output is used directly as the
        # per-stream feature transform (no multiplicative gating, no
        # post-SS2D refinement).
        fusion = self.ssm(ssm_in).permute(0, 3, 1, 2).contiguous()
        out = fusion.view(B, C, H, W, 2)

        # 4) Split per-stream features.
        event_f = out[..., 0]
        rgb_f = out[..., 1]

        # 5) Residual fusion onto the post-affine streams (matches MCFNet
        # darknet_mamba.py:799-800, where x is rebound to the post-affine
        # value before the residual add).
        feat_event_out = z_event + event_f
        feat_rgb_out = z_rgb + rgb_f

        return feat_event_out, feat_rgb_out
