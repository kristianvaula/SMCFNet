import torch
from smcfnet.config.config import Config
from smcfnet.model.build import build_model


def test_inference_forward_consumes_stem_window():
    c = Config()
    c.model.sttp.enabled = True
    c.model.sttp.head_design = "spiking_recurrent"
    c.model.sttp.context_window = 4
    c.model.sttp.recurrent_hidden_dim = 8
    model = build_model(c).eval()

    B = 1
    rgb = torch.randn(B, 3, 480, 640)
    event = torch.randn(B, 5, 480, 640)
    # size_config matches build_model: size 's' -> width_mul=0.5 -> stem_channels=128
    size_config = {'nano': 0.25, 'tiny': 0.375, 's': 0.5, 'm': 0.75, 'l': 1.0, 'x': 1.25}
    C = int(256 * size_config[c.model.size])  # stem channels
    K = 4
    window = [
        torch.randn(B, K, C, 60, 80),
        torch.randn(B, K, C, 30, 40),
        torch.randn(B, K, C, 15, 20),
    ]
    with torch.no_grad():
        detections, fut_preds = model(rgb, event, stem_window=window)
    assert len(fut_preds) == 3
    assert fut_preds[0].shape[1] == 4
