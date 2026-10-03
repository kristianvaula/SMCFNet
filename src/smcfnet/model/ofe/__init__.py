"""Flow estimator package — pluggable architectures for MCM (Axis A).

All estimators implement the ``FlowEstimator`` contract:
    forward(voxel) -> flow  (B, C, H, W) -> (B, 2, H, W)

Use ``create_flow_estimator(name, **kwargs)`` to instantiate by config key.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, Type


class FlowEstimator(nn.Module):
    """Abstract base class for all flow estimator architectures.

    Contract: voxel in, dense optical flow out.

    Every subclass must implement:
        - ``forward(voxel)`` -> ``(B, 2, H, W)``
        - ``reset_state()``
        - ``set_mode(mode)``
        - ``get_mode()``
    """

    def forward(self, voxel: torch.Tensor) -> torch.Tensor:
        """Predict dense optical flow from a voxel grid.

        Args:
            voxel: Event voxel grid, shape ``(B, C, H, W)`` where C = num_bins.

        Returns:
            Dense optical flow, shape ``(B, 2, H, W)``.
        """
        raise NotImplementedError

    def reset_state(self) -> None:
        """Clear all temporal state (ConvGRU hidden, LIF membranes, etc.)."""
        raise NotImplementedError

    def set_mode(self, mode: str) -> None:
        """Switch ann/snn for all spiking blocks."""
        raise NotImplementedError

    def get_mode(self) -> str:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_ESTIMATOR_REGISTRY: Dict[str, Type[FlowEstimator]] = {}


def register_estimator(name: str):
    """Decorator to register a FlowEstimator subclass."""
    def wrapper(cls: Type[FlowEstimator]):
        _ESTIMATOR_REGISTRY[name] = cls
        return cls
    return wrapper


def create_flow_estimator(name: str, **kwargs) -> FlowEstimator:
    """Create a flow estimator by registry key.

    Raises ValueError if the name is not registered.
    """
    if name not in _ESTIMATOR_REGISTRY:
        raise ValueError(
            f"Unknown flow estimator '{name}'. "
            f"Available: {list(_ESTIMATOR_REGISTRY.keys())}"
        )
    return _ESTIMATOR_REGISTRY[name](**kwargs)


# Import submodules to trigger registration
from smcfnet.model.ofe.st_flownet import STFlowNet  # noqa: F401, E402
from smcfnet.model.ofe.cuadrado_unet import (  # noqa: F401, E402
    BottomSeparableCuadradoUNet,
    CuadradoUNet,
    FactorizedCuadradoUNet,
    PEODCuadradoUNet,
)
from smcfnet.model.ofe.cuadrado_unet_peod import PEODCuadradoUNetV2  # noqa: F401, E402
from smcfnet.model.ofe.cnn_baseline import CNNFlowNet  # noqa: F401, E402
from smcfnet.model.ofe.ecm_ofe import ECMFlowEstimator, PEODECMFlowEstimator  # noqa: F401, E402
from smcfnet.model.ofe.sofe import StreamingSpikingFlowEstimator  # noqa: F401, E402
from smcfnet.model.ofe.sdformerflow.sdformerflow_snn import SDFormerFlowEstimator  # noqa: F401, E402
