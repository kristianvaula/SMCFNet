import torch
import torch.nn as nn

from smcfnet.model.backbone import CSPDarknetDualStream


class _IntegralOutputMCM(nn.Module):
    def __init__(self):
        super().__init__()
        self._last_flow = None

    def forward(self, voxel, **kwargs):
        b, _, h, w = voxel.shape
        self._last_flow = voxel.new_zeros((b, 2, h, w))
        corrected = torch.ones((b, 1, h, w), device=voxel.device, dtype=torch.uint8)
        return corrected, voxel.new_tensor(0.0)


class _RecordingEDUM(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen_dtype = None

    def forward(self, event_voxel, rgb):
        assert torch.is_floating_point(event_voxel)
        self.seen_dtype = event_voxel.dtype
        return event_voxel


def test_backbone_casts_integral_mcm_output_before_edum():
    model = CSPDarknetDualStream(
        dep_mul=0.33,
        wid_mul=0.125,
        event_in_channels=1,
        use_mcm=True,
        mcm_instance=_IntegralOutputMCM(),
        use_edum=True,
        edum_kwargs={"num_bins": 1, "target_height": 32, "target_width": 32},
    )
    edum = _RecordingEDUM()
    model.edum = edum
    model.eval()

    rgb = torch.randn(1, 3, 32, 32)
    event = torch.zeros(1, 1, 32, 32)

    with torch.no_grad():
        outputs = model(rgb, event)

    assert edum.seen_dtype == torch.float32
    assert set(outputs) >= {"dark3", "dark4", "dark5"}
