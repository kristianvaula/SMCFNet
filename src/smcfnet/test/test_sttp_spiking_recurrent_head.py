import torch
from smcfnet.model.sttp.sttp_head import SpikingRecurrentSTTPHead


def _window(B, K, C, scales):
    # per-scale list of [B, K, C, H, W]
    return [torch.randn(B, K, C, h, w) for (h, w) in scales]


def test_spiking_recurrent_output_shapes():
    B, K, C = 2, 5, 128
    scales = [(60, 80), (30, 40), (15, 20)]
    head = SpikingRecurrentSTTPHead(in_channels=C, num_scales=3, hidden_dim=16)
    out = head(_window(B, K, C, scales))
    assert len(out) == 3
    for o, (h, w) in zip(out, scales):
        assert o.shape == (B, 4, h, w)


def test_membrane_resets_between_forwards():
    # Two identical windows must give identical outputs (no leaked state).
    B, K, C = 1, 4, 128
    scales = [(8, 10)]
    head = SpikingRecurrentSTTPHead(in_channels=C, num_scales=1, hidden_dim=8)
    head.eval()
    w = _window(B, K, C, scales)
    with torch.no_grad():
        a = head(w)[0]
        b = head(w)[0]
    assert torch.allclose(a, b, atol=1e-6)


def test_bptt_gradients_flow_to_all_window_steps():
    # A leaf window must receive gradient at every timestep after backward.
    B, K, C = 1, 5, 128
    head = SpikingRecurrentSTTPHead(in_channels=C, num_scales=1, hidden_dim=8)
    w = [torch.randn(B, K, C, 8, 10, requires_grad=True)]
    out = head(w)[0]
    out.sum().backward()
    g = w[0].grad  # [B, K, C, H, W]
    per_step = g.abs().flatten(2).sum(-1).squeeze(0)  # [K]
    assert (per_step > 0).all(), f"some window steps got zero gradient: {per_step}"


def test_convgru_head_output_shapes():
    from smcfnet.model.sttp.sttp_head import ConvGRUSTTPHead
    B, K, C = 2, 5, 128
    scales = [(60, 80), (30, 40), (15, 20)]
    head = ConvGRUSTTPHead(in_channels=C, num_scales=3, hidden_dim=16)
    out = head(_window(B, K, C, scales))
    assert len(out) == 3
    for o, (h, w) in zip(out, scales):
        assert o.shape == (B, 4, h, w)
