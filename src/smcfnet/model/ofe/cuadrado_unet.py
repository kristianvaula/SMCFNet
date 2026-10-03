"""Cuadrado/OF_EV_SNN-style spiking U-Net flow estimator.

This implementation keeps the original layer naming/layout used in
``OF_EV_SNN/network_3d/poolingNet_cat_1res.py`` so pretrained checkpoints can
be loaded directly into the estimator state dict.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from smcfnet.model.ofe import FlowEstimator, register_estimator


def _resolve_ifnode_class():
    """Resolve IFNode across SpikingJelly API variants."""
    try:
        from spikingjelly.clock_driven import neuron as sj_neuron
    except Exception:
        from spikingjelly.activation_based import neuron as sj_neuron
    return sj_neuron.IFNode


class SwitchableIFNode(nn.Module):
    """IFNode wrapper that supports ann/snn switching without changing keys."""

    def __init__(self, mode: str = "snn", **if_kwargs):
        super().__init__()
        ifnode_cls = _resolve_ifnode_class()
        self.node = ifnode_cls(**if_kwargs)
        self.mode = mode
        self.v = None
        self.compile_friendly_snn = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "ann":
            # ANN fallback keeps tensor continuity and exposes ``v`` like IFNode.
            self.v = x
            return x
        out = self.node(x)
        self.v = getattr(self.node, "v", None)
        if self.compile_friendly_snn and isinstance(self.v, torch.Tensor):
            self.v = self.v.clone()
        return out

    def reset_state(self) -> None:
        self.v = None
        if hasattr(self.node, "reset") and callable(self.node.reset):
            self.node.reset()


class MultiplyBy(nn.Module):
    """Scale activations before IF neurons (matches OF_EV_SNN behavior)."""

    def __init__(self, scale_value: float = 5.0):
        super().__init__()
        self.scale_value = float(scale_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mul(x, self.scale_value)


class PoolingEncoder(nn.Module):
    """Depthwise-separable 3D encoder stage used by OF_EV_SNN."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, multiply_factor: float = 5.0):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(5, kernel_size, kernel_size),
                padding=(0, kernel_size // 2, kernel_size // 2),
                padding_mode="replicate",
                bias=False,
                groups=in_channels,
            ),
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0),
            MultiplyBy(multiply_factor),
            SwitchableIFNode(mode="snn"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv1(x)


class DepthwiseSeparableBottom(nn.Module):
    """Cheaper Cuadrado stem: full temporal-spatial depthwise conv + pointwise mix."""

    def __init__(self, in_channels: int = 2, out_channels: int = 32, multiply_factor: float = 10.0):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(5, 7, 7),
                stride=1,
                padding=(0, 3, 3),
                bias=False,
                padding_mode="replicate",
                groups=in_channels,
            ),
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False),
            MultiplyBy(multiply_factor),
            SwitchableIFNode(mode="snn"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv1(x)


class FactorizedBottom(nn.Module):
    """(2+1)D Cuadrado stem for latency-oriented from-scratch training."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 32,
        spatial_kernel: int = 7,
        multiply_factor: float = 10.0,
    ):
        super().__init__()
        spatial_pad = spatial_kernel // 2
        self.conv1 = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(5, 1, 1),
                padding=(0, 0, 0),
                bias=False,
                groups=in_channels,
            ),
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(1, spatial_kernel, spatial_kernel),
                padding=(0, spatial_pad, spatial_pad),
                padding_mode="replicate",
                bias=False,
                groups=in_channels,
            ),
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False),
            MultiplyBy(multiply_factor),
            SwitchableIFNode(mode="snn"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv1(x)


class FactorizedPoolingEncoder(nn.Module):
    """(2+1)D encoder stage with the same Cuadrado temporal size arithmetic."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, multiply_factor: float = 5.0):
        super().__init__()
        spatial_pad = kernel_size // 2
        self.temporal = nn.Conv3d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=(5, 1, 1),
            padding=(0, 0, 0),
            bias=False,
            groups=in_channels,
        )
        self.spatial = nn.Conv3d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=(1, kernel_size, kernel_size),
            padding=(0, spatial_pad, spatial_pad),
            padding_mode="replicate",
            bias=False,
            groups=in_channels,
        )
        self.pointwise = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False)
        self.scale = MultiplyBy(multiply_factor)
        self.sn = SwitchableIFNode(mode="snn")

    @staticmethod
    def _pool_spatial(x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape
        x2d = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x2d = F.max_pool2d(x2d, kernel_size=2, stride=2)
        h2, w2 = x2d.shape[-2:]
        return x2d.reshape(b, t, c, h2, w2).permute(0, 2, 1, 3, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.pointwise(x)
        x = self._pool_spatial(x)
        x = self.scale(x)
        return self.sn(x)


class SeparableSEWResBlock(nn.Module):
    """Separable spike-element-wise residual block (single residual stage)."""

    def __init__(self, in_channels: int, kernel_size: int = 7, multiply_factor: float = 10.0):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                groups=in_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=pad,
                bias=False,
            ),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, bias=False),
            MultiplyBy(multiply_factor),
        )
        self.sn1 = nn.Sequential(SwitchableIFNode(mode="snn"))

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                groups=in_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=pad,
                bias=False,
            ),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, bias=False),
            MultiplyBy(multiply_factor),
        )
        self.sn2 = nn.Sequential(SwitchableIFNode(mode="snn"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.sn1(out)
        out = self.conv2(out)
        out = self.sn2(out)
        return out + identity


class separable_decoder_block(nn.Module):
    """Nearest-neighbor upsample + separable conv decoder block."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, up_size: tuple, bias: bool = False, multiply_factor: float = 10.0):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.deconv_1 = nn.Sequential(
            nn.UpsamplingNearest2d(size=(up_size[0] + (kernel_size - 1), up_size[1] + (kernel_size - 1))),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, bias=bias, groups=in_channels),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=bias),
            MultiplyBy(multiply_factor),
            SwitchableIFNode(mode="snn"),
        )

    def set_output_size(self, up_size: tuple[int, int]) -> None:
        self.deconv_1[0].size = (int(up_size[0]) + (self.kernel_size - 1), int(up_size[1]) + (self.kernel_size - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.deconv_1(x)


class separable_predictor_block(nn.Module):
    """Prediction head with nearest-neighbor upsample + separable conv."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, up_size: tuple, bias: bool = False):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.pred_1 = nn.Sequential(
            nn.UpsamplingNearest2d(size=(up_size[0] + (kernel_size - 1), up_size[1] + (kernel_size - 1))),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, bias=bias, groups=in_channels),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=bias),
        )

    def set_output_size(self, up_size: tuple[int, int]) -> None:
        self.pred_1[0].size = (int(up_size[0]) + (self.kernel_size - 1), int(up_size[1]) + (self.kernel_size - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pred_1(x)


@register_estimator("cuadrado_unet")
class CuadradoUNet(FlowEstimator):
    """OF_EV_SNN-compatible Cuadrado-style estimator.

    The forward contract accepts standard SMCM voxel tensors ``(B, T, H, W)``
    and internally converts them to the split-polarity shape expected by the
    original model ``(B, 2, T, H, W)``.
    """

    MIN_TEMPORAL_BINS = 21

    def __init__(
        self,
        num_bins: int = 21,
        mode: str = "snn",
        image_height: int = 480,
        image_width: int = 640,
        multiply_factor: float = 10.0,
        kernel_size: int = 7,
        **kwargs,
    ):
        super().__init__()
        del kwargs  # absorbed for compatibility with the shared estimator factory.

        self.num_bins = int(num_bins)
        self.image_height = int(image_height)
        self.image_width = int(image_width)

        if self.image_height <= 0 or self.image_width <= 0:
            raise ValueError(
                f"image_height/image_width must be positive, got "
                f"({self.image_height}, {self.image_width})"
            )

        # Encoder
        self.bottom = nn.Sequential(
            nn.Conv3d(
                in_channels=2,
                out_channels=32,
                kernel_size=(5, 7, 7),
                stride=1,
                padding=(0, 3, 3),
                bias=False,
                padding_mode="replicate",
            ),
            MultiplyBy(multiply_factor),
            SwitchableIFNode(mode="snn"),
        )

        self.conv1 = PoolingEncoder(in_channels=32, out_channels=64, kernel_size=kernel_size, multiply_factor=multiply_factor)
        self.conv2 = PoolingEncoder(in_channels=64, out_channels=128, kernel_size=kernel_size, multiply_factor=multiply_factor)
        self.conv3 = PoolingEncoder(in_channels=128, out_channels=256, kernel_size=kernel_size, multiply_factor=multiply_factor)
        self.conv4 = PoolingEncoder(in_channels=256, out_channels=512, kernel_size=kernel_size, multiply_factor=multiply_factor)

        # Bottleneck
        self.res1 = SeparableSEWResBlock(in_channels=512, multiply_factor=multiply_factor, kernel_size=kernel_size)

        # Decoder + multi-scale predictors.
        self.deconv4 = separable_decoder_block(in_channels=512, out_channels=256, up_size=(self.image_height // 8, self.image_width // 8), kernel_size=kernel_size, multiply_factor=multiply_factor)
        self.deconv3 = separable_decoder_block(in_channels=256 * 2, out_channels=128, up_size=(self.image_height // 4, self.image_width // 4), kernel_size=kernel_size, multiply_factor=multiply_factor)
        self.deconv2 = separable_decoder_block(in_channels=128 * 2, out_channels=64, up_size=(self.image_height // 2, self.image_width // 2), kernel_size=kernel_size, multiply_factor=multiply_factor)
        self.deconv1 = separable_decoder_block(in_channels=64 * 2, out_channels=32, up_size=(self.image_height, self.image_width), kernel_size=kernel_size, multiply_factor=multiply_factor)

        # Since the model only uses pred1 we dont instantiate the other predictors to save memory, but the code is left here for reference.
        #self.pred4 = separable_predictor_block(in_channels=256 * 2, out_channels=2, up_size=(self.image_height, self.image_width), kernel_size=kernel_size)
        #self.pred3 = separable_predictor_block(in_channels=128 * 2, out_channels=2, up_size=(self.image_height, self.image_width), kernel_size=kernel_size)
        #self.pred2 = separable_predictor_block(in_channels=64 * 2, out_channels=2, up_size=(self.image_height, self.image_width), kernel_size=kernel_size)
        self.pred1 = separable_predictor_block(in_channels=32 * 2, out_channels=2, up_size=(self.image_height, self.image_width), kernel_size=kernel_size)

        self.pool = SwitchableIFNode(mode="snn", v_threshold=float("inf"), v_reset=0.0)

        self.set_mode(mode)
        self._refresh_if_nodes()
        self._compiled_forward_flow = None
        self._uses_compiled_forward_flow = False
        self._configure_static_decoder_sizes()

    def _refresh_if_nodes(self) -> None:
        self._if_nodes = [module for module in self.modules() if isinstance(module, SwitchableIFNode)]

    def _configure_static_decoder_sizes(self) -> None:
        self.deconv4.set_output_size((self.image_height // 8, self.image_width // 8))
        self.deconv3.set_output_size((self.image_height // 4, self.image_width // 4))
        self.deconv2.set_output_size((self.image_height // 2, self.image_width // 2))
        self.deconv1.set_output_size((self.image_height, self.image_width))
        self.pred1.set_output_size((self.image_height, self.image_width))

    def set_compile_friendly_snn(self, enabled: bool = True) -> None:
        for node in self._if_nodes:
            node.compile_friendly_snn = bool(enabled)

    def compile_for_inference(self, **kwargs) -> None:
        compile_kwargs = {key: value for key, value in kwargs.items() if value is not None}
        self.set_compile_friendly_snn(True)
        self._compiled_forward_flow = torch.compile(self._forward_flow, **compile_kwargs)
        self._uses_compiled_forward_flow = True

    def _to_ofev_input(self, voxel: torch.Tensor) -> torch.Tensor:
        """Convert SMCM voxel format into OF_EV_SNN split-polarity format."""
        if voxel.ndim == 5:
            if int(voxel.shape[1]) != 2:
                raise ValueError(
                    "Expected 5D OF_EV input as (B,2,T,H,W), got "
                    f"{tuple(voxel.shape)}"
                )
            return voxel

        if voxel.ndim != 4:
            raise ValueError(
                "Expected voxel tensor with shape (B,T,H,W) or (B,2,T,H,W), got "
                f"{tuple(voxel.shape)}"
            )

        bins = int(voxel.shape[1])
        if self.num_bins > 0 and bins == 2 * self.num_bins:
            on = voxel[:, :self.num_bins]
            off = voxel[:, self.num_bins:]
            return torch.stack((on, off), dim=1)

        # Default SMCM representation is signed bins: positive and negative
        # events in one channel per bin. Split it for OF_EV_SNN compatibility.
        on = torch.clamp(voxel, min=0.0)
        off = torch.clamp(-voxel, min=0.0)
        return torch.stack((on, off), dim=1)

    def set_mode(self, mode: str) -> None:
        assert mode in ("ann", "snn"), f"Invalid mode: {mode}"
        self._mode = mode
        for module in self.modules():
            if isinstance(module, SwitchableIFNode):
                module.mode = mode

    def get_mode(self) -> str:
        return self._mode

    def reset_state(self) -> None:
        for module in self.modules():
            if isinstance(module, SwitchableIFNode):
                module.reset_state()

    def _pool_membrane(self, x: torch.Tensor) -> torch.Tensor:
        self.pool(x)
        if self.pool.v is None:
            raise RuntimeError("Cuadrado predictor pool did not expose membrane state")
        return self.pool.v

    def _forward_flow(self, voxel: torch.Tensor) -> torch.Tensor:
        x = self._to_ofev_input(voxel)
        temporal_bins = int(x.shape[2])
        if temporal_bins < self.MIN_TEMPORAL_BINS:
            raise ValueError(
                "cuadrado_unet requires at least "
                f"{self.MIN_TEMPORAL_BINS} temporal bins after polarity split "
                f"to match the OF_EV_SNN encoder depth, but got {temporal_bins}. "
                "Set data.event_bins>=21 for signed-bin voxels or provide "
                "a (B,2,T,H,W) input with T>=21."
            )
        _, _, _, h, w = x.shape

        in_conv1 = self.bottom(x)
        out_conv1 = self.conv1(in_conv1)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)

        in_res1 = torch.squeeze(out_conv4, 2)
        out_res1 = self.res1(in_res1)

        in_deconv4 = out_res1 + out_conv4[:, :, -1]
        dynamic_geometry = (h, w) != (self.image_height, self.image_width)
        if dynamic_geometry:
            self.deconv4.set_output_size((out_conv3.shape[-2], out_conv3.shape[-1]))
        out_deconv4 = self.deconv4(in_deconv4)

        in_deconv3 = torch.cat((out_deconv4, out_conv3[:, :, -1]), dim=1)
        if dynamic_geometry:
            self.deconv3.set_output_size((out_conv2.shape[-2], out_conv2.shape[-1]))
        out_deconv3 = self.deconv3(in_deconv3)

        in_deconv2 = torch.cat((out_deconv3, out_conv2[:, :, -1]), dim=1)
        if dynamic_geometry:
            self.deconv2.set_output_size((out_conv1.shape[-2], out_conv1.shape[-1]))
        out_deconv2 = self.deconv2(in_deconv2)

        in_deconv1 = torch.cat((out_deconv2, out_conv1[:, :, -1]), dim=1)
        if dynamic_geometry:
            self.deconv1.set_output_size((h, w))
        out_deconv1 = self.deconv1(in_deconv1)

        if dynamic_geometry:
            self.pred1.set_output_size((h, w))
        up_1 = self.pred1(torch.cat((out_deconv1, in_conv1[:, :, -1]), dim=1))

        if self._mode == "ann":
          flow = up_1
        else:
          flow = self._pool_membrane(up_1)

        return flow

    def forward(self, voxel: torch.Tensor) -> torch.Tensor:
        if self._compiled_forward_flow is not None:
            marker = getattr(getattr(torch, "compiler", None), "cudagraph_mark_step_begin", None)
            if self._uses_compiled_forward_flow and callable(marker):
                marker()
            flow = self._compiled_forward_flow(voxel)
        else:
            flow = self._forward_flow(voxel)
        self._last_flow = flow.detach()
        return flow


@register_estimator("cuadrado_unet_peod")
class PEODCuadradoUNet(CuadradoUNet):
    """Cuadrado/OF_EV_SNN estimator configured for PEOD 720x1280 events.

    This subclass keeps the exact module hierarchy and parameter names from
    ``CuadradoUNet``. It only fixes the default event geometry and validates
    accidental non-PEOD construction, preserving checkpoint compatibility with
    the base estimator when temporal/kernel settings match.
    """

    def __init__(
        self,
        num_bins: int = 21,
        mode: str = "snn",
        image_height: int = 720,
        image_width: int = 1280,
        multiply_factor: float = 10.0,
        kernel_size: int = 7,
        **kwargs,
    ):
        if int(image_height) != 720 or int(image_width) != 1280:
            raise ValueError(
                "cuadrado_unet_peod expects PEOD event geometry "
                f"(height=720, width=1280), got ({int(image_height)}, {int(image_width)})"
            )
        super().__init__(
            num_bins=num_bins,
            mode=mode,
            image_height=image_height,
            image_width=image_width,
            multiply_factor=multiply_factor,
            kernel_size=kernel_size,
            **kwargs,
        )


@register_estimator("cuadrado_unet_bottom_sep")
class BottomSeparableCuadradoUNet(CuadradoUNet):
    """Cuadrado variant with only the full-resolution stem made separable.

    This is the conservative latency ablation: the original non-separable
    ``bottom`` Conv3d is replaced by depthwise ``(5,7,7)`` filtering over the
    two polarity channels followed by a ``1x1x1`` channel mix. All later encoder,
    bottleneck, decoder, skip, and readout paths remain inherited from
    ``CuadradoUNet``.
    """

    def __init__(
        self,
        num_bins: int = 21,
        mode: str = "snn",
        image_height: int = 480,
        image_width: int = 640,
        multiply_factor: float = 10.0,
        kernel_size: int = 7,
        **kwargs,
    ):
        super().__init__(
            num_bins=num_bins,
            mode=mode,
            image_height=image_height,
            image_width=image_width,
            multiply_factor=multiply_factor,
            kernel_size=kernel_size,
            **kwargs,
        )
        self.bottom = DepthwiseSeparableBottom(
            in_channels=2,
            out_channels=32,
            multiply_factor=multiply_factor,
        )
        self._refresh_if_nodes()
        self.set_mode(mode)


@register_estimator("cuadrado_unet_factorized")
class FactorizedCuadradoUNet(CuadradoUNet):
    """Cuadrado variant with a (2+1)D factorized encoder.

    This is the stronger latency ablation for from-scratch training. It keeps
    Cuadrado's temporal-collapse arithmetic and U-Net data flow, but replaces
    the stem and four pooling encoders with temporal-depthwise plus
    spatial-depthwise stages. It is not checkpoint-compatible with the reference
    Cuadrado estimator.
    """

    def __init__(
        self,
        num_bins: int = 21,
        mode: str = "snn",
        image_height: int = 480,
        image_width: int = 640,
        multiply_factor: float = 10.0,
        kernel_size: int = 7,
        **kwargs,
    ):
        super().__init__(
            num_bins=num_bins,
            mode=mode,
            image_height=image_height,
            image_width=image_width,
            multiply_factor=multiply_factor,
            kernel_size=kernel_size,
            **kwargs,
        )
        self.bottom = FactorizedBottom(
            in_channels=2,
            out_channels=32,
            spatial_kernel=7,
            multiply_factor=multiply_factor,
        )
        self.conv1 = FactorizedPoolingEncoder(
            in_channels=32,
            out_channels=64,
            kernel_size=kernel_size,
            multiply_factor=multiply_factor,
        )
        self.conv2 = FactorizedPoolingEncoder(
            in_channels=64,
            out_channels=128,
            kernel_size=kernel_size,
            multiply_factor=multiply_factor,
        )
        self.conv3 = FactorizedPoolingEncoder(
            in_channels=128,
            out_channels=256,
            kernel_size=kernel_size,
            multiply_factor=multiply_factor,
        )
        self.conv4 = FactorizedPoolingEncoder(
            in_channels=256,
            out_channels=512,
            kernel_size=kernel_size,
            multiply_factor=multiply_factor,
        )
        self._refresh_if_nodes()
        self.set_mode(mode)
