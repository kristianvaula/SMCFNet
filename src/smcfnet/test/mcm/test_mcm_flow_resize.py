from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch

from smcfnet.model.mcm.mcm import MotionCompensationModule
from smcfnet.model.ofe import FlowEstimator


class _CoarseFlowEstimator(FlowEstimator):
    def __init__(self):
        super().__init__()
        self._mode = "ann"

    def forward(self, voxel: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = voxel.shape
        flow = voxel.new_zeros((batch_size, 2, height // 2, width // 2))
        flow[:, 0].fill_(1.0)
        flow[:, 1].fill_(0.5)
        return flow

    def reset_state(self) -> None:
        return None

    def set_mode(self, mode: str) -> None:
        self._mode = mode

    def get_mode(self) -> str:
        return self._mode


def test_motion_compensation_upsamples_coarse_flow_before_warping():
    mcm = MotionCompensationModule(estimator=_CoarseFlowEstimator(), flow_scaling=1.0)
    voxel = torch.randn(2, 5, 8, 10)

    corrected, loss = mcm(voxel)

    assert tuple(corrected.shape) == tuple(voxel.shape)
    assert tuple(mcm._last_flow.shape) == (2, 2, 8, 10)
    assert torch.isfinite(corrected).all()
    assert loss.item() == 0.0
