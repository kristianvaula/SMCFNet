
import torch

from smcfnet.utils.event_utils import events_to_cuadrado_histogram_torch


def test_cuadrado_histogram_split_polarity_fixed_bins():
    events = torch.tensor([
        [1, 2, 0.0, 1.0],
        [1, 2, 5.0, -1.0],
        [3, 0, 10.0, 1.0],
    ])

    hist = events_to_cuadrado_histogram_torch(
        events,
        num_bins=3,
        width=4,
        height=3,
        t_start=0.0,
        t_end=10.0,
    )

    assert hist.shape == (2, 3, 3, 4)
    assert hist[0, 0, 2, 1].item() == 1.0
    assert hist[1, 1, 2, 1].item() == 1.0
    assert hist[0, 2, 0, 3].item() == 1.0
    assert hist.sum().item() == 3.0


def test_cuadrado_histogram_scales_source_coordinates():
    events = torch.tensor([[3, 3, 0.0, 1.0]])

    hist = events_to_cuadrado_histogram_torch(
        events,
        num_bins=2,
        width=2,
        height=2,
        source_width=4,
        source_height=4,
    )

    assert hist[0, 0, 1, 1].item() == 1.0
    assert hist.sum().item() == 1.0
