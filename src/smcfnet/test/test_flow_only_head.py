"""Unit tests for FlowOnlySTTPHead (Design D Stage 2).

Run locally:  python -m pytest src/smcfnet/test/test_flow_only_head.py -v
"""

import pytest

torch = pytest.importorskip("torch")

from smcfnet.model.sttp import FlowOnlySTTPHead, STTPHead


def _stem_features(B=2, C=128, H_full=480, W_full=640, strides=(8, 16, 32)):
    """Mock stem features at three FPN scales for a given full-image (H, W)."""
    return [torch.randn(B, C, H_full // s, W_full // s) for s in strides]


def test_output_shape_with_flow():
    head = FlowOnlySTTPHead(num_scales=3, flow_channels=2, flow_proj_dim=16)
    stems = _stem_features(B=2, C=128, H_full=480, W_full=640)
    flow = torch.randn(2, 2, 480, 640)

    outs = head(stems, flow=flow)

    assert len(outs) == 3
    for k, out in enumerate(outs):
        assert out.shape == (2, 4, stems[k].shape[2], stems[k].shape[3])


def test_no_flow_substitutes_zero_and_keeps_shape():
    head = FlowOnlySTTPHead(num_scales=3, flow_channels=2, flow_proj_dim=16)
    stems = _stem_features(B=1, C=128, H_full=480, W_full=640)

    outs = head(stems, flow=None)

    assert len(outs) == 3
    for k, out in enumerate(outs):
        assert out.shape == (1, 4, stems[k].shape[2], stems[k].shape[3])


def test_param_count_under_20k():
    """Whole point of Design D: tiny head. Total <20k params at default F=16."""
    head = FlowOnlySTTPHead(num_scales=3, flow_channels=2, flow_proj_dim=16)
    n = sum(p.numel() for p in head.parameters())
    assert n < 20_000, f"FlowOnlySTTPHead has {n} params, expected <20k"


def test_pred_layer_zero_init_matches_sttp_head():
    """fut_pred bias zero, weight std=0.01 (mirrors STTPHead init)."""
    head = FlowOnlySTTPHead(num_scales=3, flow_channels=2, flow_proj_dim=16)
    for pred in head.fut_preds:
        assert pred.bias.abs().max().item() < 1e-9
        assert pred.weight.std().item() < 0.05


def test_gradients_flow_through_all_params():
    head = FlowOnlySTTPHead(num_scales=3, flow_channels=2, flow_proj_dim=16)
    stems = _stem_features(B=1, C=128, H_full=480, W_full=640)
    flow = torch.randn(1, 2, 480, 640, requires_grad=True)

    outs = head(stems, flow=flow)
    loss = sum(o.sum() for o in outs)
    loss.backward()

    for name, p in head.named_parameters():
        if not p.requires_grad:
            continue
        assert p.grad is not None, f"{name} has no grad"
        assert p.grad.abs().sum().item() > 0, f"{name} has zero grad"


def test_zero_flow_channels_rejected():
    """Design D requires flow input; refuse misconfigurations early."""
    with pytest.raises(ValueError, match="flow_channels"):
        FlowOnlySTTPHead(num_scales=3, flow_channels=0, flow_proj_dim=16)


def test_drop_in_signature_matches_sttp_head():
    """forward(stems, flow=...) returns same shapes as STTPHead Design C."""
    flowonly = FlowOnlySTTPHead(num_scales=3, flow_channels=2, flow_proj_dim=16)
    designc = STTPHead(in_channels=128, num_scales=3, flow_channels=2, flow_proj_dim=16)

    stems = _stem_features(B=1, C=128, H_full=480, W_full=640)
    flow = torch.randn(1, 2, 480, 640)

    out_flowonly = flowonly(stems, flow=flow)
    out_designc = designc(stems, flow=flow)

    assert len(out_flowonly) == len(out_designc)
    for a, b in zip(out_flowonly, out_designc):
        assert a.shape == b.shape


def test_widening_flow_proj_dim_grows_params():
    """Sanity: F=64 must produce more params than F=16 (architectural lever)."""
    small = FlowOnlySTTPHead(num_scales=3, flow_channels=2, flow_proj_dim=16)
    large = FlowOnlySTTPHead(num_scales=3, flow_channels=2, flow_proj_dim=64)
    n_small = sum(p.numel() for p in small.parameters())
    n_large = sum(p.numel() for p in large.parameters())
    assert n_large > n_small * 4
