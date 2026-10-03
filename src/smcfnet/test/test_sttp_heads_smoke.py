"""WS4 gate: every STTP head builds and forwards to [B,4,H,W] x num_scales.

One consolidated smoke over all head_design variants, so a refactor that breaks
any head's forward contract fails here before a cluster submit. Complements the
per-head tests; this one is the "all heads still work on current main" check.

Run: python -m pytest src/smcfnet/test/test_sttp_heads_smoke.py -v
"""

from __future__ import annotations

import torch

from smcfnet.model.sttp import (
    ConvGRUSTTPHead,
    FlowOnlySTTPHead,
    FlowProjectionHead,
    SpikingRecurrentSTTPHead,
    STTPHead,
)

B, C, K = 2, 128, 3                       # batch, stem channels (256*0.5 for 's'), K_ctx
K_TRAJ = 5                                # trajectory length: heads emit 4*K_TRAJ channels
SHAPES = [(8, 8), (4, 4), (2, 2)]         # P3/P4/P5 at strides 8/16/32


def _stem_4d():
    return [torch.randn(B, C, h, w) for (h, w) in SHAPES]


def _stem_5d():
    return [torch.randn(B, K, C, h, w) for (h, w) in SHAPES]


def _flow():
    return torch.randn(B, 2, 64, 64)


def _assert_outputs(outs, k=K_TRAJ):
    assert len(outs) == len(SHAPES), len(outs)
    for out, (h, w) in zip(outs, SHAPES):
        assert out.shape == (B, 4 * k, h, w), out.shape


def test_design_c_feat_plus_flow():
    head = STTPHead(in_channels=C, num_scales=3, flow_channels=2, flow_proj_dim=16, future_steps=K_TRAJ)
    _assert_outputs(head(_stem_4d(), flow=_flow()))


def test_design_b_feat_only():
    head = STTPHead(in_channels=C, num_scales=3, flow_channels=0, future_steps=K_TRAJ)
    _assert_outputs(head(_stem_4d()))


def test_design_c_identity_warp_zero_flow():
    # Exp 3.3 identity-warp arm: Design C head, flow absent -> head substitutes
    # zeros internally. Must forward cleanly (no flow dir at train/eval time).
    head = STTPHead(in_channels=C, num_scales=3, flow_channels=2, flow_proj_dim=16, future_steps=K_TRAJ)
    _assert_outputs(head(_stem_4d(), flow=None))


def test_flow_only_head():
    head = FlowOnlySTTPHead(num_scales=3, flow_channels=2, flow_proj_dim=16, future_steps=K_TRAJ)
    _assert_outputs(head(_stem_4d(), flow=_flow()))


def test_trivial_flow_projection_head():
    head = FlowProjectionHead(future_steps=K_TRAJ, num_scales=3, strides=(8, 16, 32))
    assert sum(p.numel() for p in head.parameters()) == 0  # zero-param baseline
    _assert_outputs(head(_stem_4d(), flow=_flow()))


def test_spiking_recurrent_head():
    head = SpikingRecurrentSTTPHead(in_channels=C, num_scales=3, hidden_dim=64, future_steps=K_TRAJ)
    _assert_outputs(head(_stem_5d()))


def test_convgru_head():
    head = ConvGRUSTTPHead(in_channels=C, num_scales=3, hidden_dim=64, future_steps=K_TRAJ)
    _assert_outputs(head(_stem_5d()))


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
