"""PEOD-only Cuadrado U-Net variants.

This module intentionally keeps PEOD Cuadrado registrations separate from the
DSEC-locked ``cuadrado_unet`` implementation.  The v2 estimator currently
preserves the Cuadrado module hierarchy and parameter names, while the v2
training/config path can evolve independently for PEOD.
"""

from __future__ import annotations

from smcfnet.model.ofe import register_estimator
from smcfnet.model.ofe.cuadrado_unet import CuadradoUNet


@register_estimator("cuadrado_unet_peod_v2")
class PEODCuadradoUNetV2(CuadradoUNet):
    """Cuadrado/OF_EV_SNN estimator locked to PEOD 720x1280 geometry."""

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
                "cuadrado_unet_peod_v2 expects PEOD event geometry "
                f"(height=720, width=1280), got ({int(image_height)}, {int(image_width)})"
            )
        if int(num_bins) != 21:
            raise ValueError(f"cuadrado_unet_peod_v2 expects num_bins=21, got {int(num_bins)}")
        if str(mode) != "snn":
            raise ValueError(f"cuadrado_unet_peod_v2 expects mode='snn', got {mode!r}")
        super().__init__(
            num_bins=num_bins,
            mode=mode,
            image_height=image_height,
            image_width=image_width,
            multiply_factor=multiply_factor,
            kernel_size=kernel_size,
            **kwargs,
        )
