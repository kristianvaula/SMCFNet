"""Regression tests for issue #110: SMCM/SOFE streaming path never populated
``_last_profile_events``, so MCM stage timings were silently absent for every
streaming config, and the ``_profile_mcm_stages`` flag leaked into the
estimator/warp/loss calls.

The streaming forward paths now pop ``_profile_mcm_stages`` and set
``_last_profile_events``. A minimal stub estimator (no real SOFE / no dataset)
exercises the contract on CPU; the populated-keys check is CUDA-gated.
"""

import unittest

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from smcfnet.model.mcm.smcm import StreamingMotionCompensationModule


class _StubStreamingEstimator(nn.Module):
    """Minimal estimator implementing forward_streaming with an explicit
    signature (no **kwargs), so a leaked _profile_mcm_stages kwarg would raise."""

    def __init__(self, height: int = 8, width: int = 8):
        super().__init__()
        # A trainable parameter so the module is NOT all-frozen (exercises the
        # non-frozen inline streaming path rather than the frozen path).
        self.weight = nn.Parameter(torch.zeros(1))
        self.bptt_steps = 0
        self._height = height
        self._width = width

    def forward_streaming(
        self,
        *,
        events,
        events_offsets,
        t_start_us,
        t_end_us,
        t_ref_us,
        state=None,
        detach_state=False,
        sequence_reset=None,
        return_debug=False,
    ):
        batch = int(events_offsets.numel() - 1)
        flow = events.new_zeros((batch, 2, self._height, self._width)) + self.weight
        return flow, None


def _make_module():
    return StreamingMotionCompensationModule(
        estimator=_StubStreamingEstimator(),
        warper=None,        # warp_events returns the provided voxel unchanged
        loss_fn=None,       # compute_loss short-circuits to (0, None)
        flow_scaling=1.0,
    )


def _make_batch(device="cpu", height=8, width=8):
    return {
        "events": torch.zeros((0, 4), device=device),
        "events_offsets": torch.tensor([0, 0], dtype=torch.int64, device=device),
        "t_start_us": torch.tensor([0], dtype=torch.int64, device=device),
        "t_end_us": torch.tensor([1000], dtype=torch.int64, device=device),
        "t_ref_us": torch.tensor([500], dtype=torch.int64, device=device),
        "voxel": torch.zeros((1, 1, height, width), device=device),
    }


class TestSMCMProfiling(unittest.TestCase):
    def test_attribute_present_after_construction(self):
        module = _make_module()
        self.assertEqual(module._last_profile_events, {})

    def test_profile_flag_does_not_leak_and_attr_is_dict(self):
        # _profile_mcm_stages must be popped before estimator/warp/loss; if it
        # leaked into the stub's explicit-signature forward_streaming it would
        # raise TypeError. On CPU the timer is disabled so events stay empty.
        module = _make_module()
        batch = _make_batch()
        out = module(flow_gt=None, _profile_mcm_stages=True, **batch)
        self.assertIsInstance(out, tuple)
        self.assertIsInstance(module._last_profile_events, dict)
        self.assertEqual(module._last_profile_events, {})  # CPU: timer disabled

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA for stage timing")
    def test_stage_timings_populated_on_cuda(self):
        module = _make_module().cuda()
        batch = _make_batch(device="cuda")
        module(flow_gt=None, _profile_mcm_stages=True, **batch)
        events = module._last_profile_events
        self.assertIn("smcm_estimator", events)
        self.assertIn("smcm_warp", events)
        self.assertIn("smcm_loss", events)


if __name__ == "__main__":
    unittest.main()
