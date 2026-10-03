"""Tests for the classical detect->track->predict baseline (tools/sttp_baseline_tracker).

Pure CPU, no dataset, no GPU. Covers the three P0 silent-ADE risks:
  P0-1 frame-gap velocity (displacement / frame-gap, not per-observation),
  P0-2 metric wiring parity (same accumulator, identity-anchor mapping),
  P0-3 order/shape alignment (one prediction per detection, in order).

Run: python -m pytest src/smcfnet/test/test_sttp_baseline_tracker.py -v
"""

import math
import os
import sys

import torch

# tools/ is a sibling package at the repo root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from tools.sttp_baseline_tracker import (  # noqa: E402
    FrameGapTracker, KalmanTracker, score_frames, _self_check,
)


def _det7(cx, cy, w, h, cls=0.0):
    return torch.tensor([[cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, 0.9, 0.9, cls]])


def _frame(fi, cx, cy, w, h, fut_cx, fut_cy, k, valid=True, cls=0.0):
    return {
        "seq": "s", "frame_index": fi,
        "dets_xyxy7": _det7(cx, cy, w, h, cls),
        "gt_targets": torch.tensor([[cls, cx, cy, w, h]]),
        "future_targets": {k: {"boxes": torch.tensor([[fut_cx, fut_cy, w, h]]),
                               "validity": torch.tensor([bool(valid)])}},
    }


def test_self_check_passes():
    _self_check()


def test_framegap_velocity_uses_frame_delta():
    # Object observed at frames 0 and 3 (1,2 missed); boxes overlap so association holds.
    tr = FrameGapTracker(iou_threshold=0.3)
    for f, c in [(0, 100.0), (3, 106.0)]:  # +6px over 3 frames -> 2px/frame
        tr.update(torch.tensor([[c - 20, c - 20, c + 20, c + 20]]), f)
    px, py = tr.predict(0, k=5)
    assert math.isclose(px, 116.0, abs_tol=1e-4)  # 106 + 5*2
    assert math.isclose(py, 116.0, abs_tol=1e-4)
    # The naive bug (treating the two observations as one frame apart) gives 136.
    assert abs(px - 136.0) > 1.0


def test_consecutive_velocity():
    tr = FrameGapTracker(iou_threshold=0.3)
    for f, c in [(0, 50.0), (1, 54.0), (2, 58.0)]:  # +4px/frame, 40px box
        tr.update(torch.tensor([[c - 20, c - 20, c + 20, c + 20]]), f)
    px, _ = tr.predict(0, k=2, two_frame=True)
    assert math.isclose(px, 58.0 + 2 * 4.0, abs_tol=1e-4)


def test_cold_start_zero_velocity():
    # A single observation has no velocity -> predicts the current center.
    tr = FrameGapTracker()
    tr.update(torch.tensor([[0.0, 0.0, 20.0, 20.0]]), 0)
    px, py = tr.predict(0, k=10)
    assert (px, py) == (10.0, 10.0)
    assert tr.age(0) == 1


def test_stale_track_expires():
    tr = FrameGapTracker(max_age=2)
    tr.update(torch.tensor([[0.0, 0.0, 20.0, 20.0]]), 0)
    # Re-appear far in the future at the SAME spot: the old track must have expired,
    # so this is a brand-new track (age 1), not a coasting revival.
    tr.update(torch.tensor([[0.0, 0.0, 20.0, 20.0]]), 100)
    # Only one live track id, and it has a single observation.
    assert tr.age(1) == 1


def test_metric_wiring_displaced_gt():
    # Stationary tracker prediction, GT future displaced by (dx,dy): ADE == hypot.
    dx, dy = 7.0, 24.0
    frames = [_frame(fi, 200.0, 150.0, 30.0, 30.0, 200.0 + dx, 150.0 + dy, k=3)
              for fi in range(3)]
    m = score_frames(frames, k=3, variant="full")
    assert math.isclose(m["ADE"], math.hypot(dx, dy), abs_tol=1e-3)
    assert math.isclose(m["FDE"], m["ADE"], abs_tol=1e-9)  # single box at k


def test_constant_velocity_track_low_ade():
    # Object moving +3px/frame; once velocity is established the prediction matches GT.
    k = 5
    frames = []
    for fi in range(6):
        cx = 100.0 + 3.0 * fi
        cy = 100.0 + 3.0 * fi
        frames.append(_frame(fi, cx, cy, 40.0, 40.0, cx + 3.0 * k, cy + 3.0 * k, k=k))
    m = score_frames(frames, k=k, variant="full")
    # Only the cold-start frame (age 1, zero velocity) contributes error.
    cold_err = 3.0 * k * math.sqrt(2)
    expected = cold_err / 6.0
    assert math.isclose(m["ADE"], expected, rel_tol=0.02), f"{m['ADE']} vs {expected}"


def test_order_alignment_multi_detection():
    # Two objects with different velocities in one frame: each detection's future box
    # must be its OWN track's prediction (identity-anchor order preserved).
    k = 4
    frames = []
    for fi in range(3):
        a_cx = 50.0 + 2.0 * fi      # object A: +2/frame
        b_cx = 300.0 - 5.0 * fi     # object B: -5/frame
        dets = torch.tensor([
            [a_cx - 15, 50 - 15, a_cx + 15, 50 + 15, 0.9, 0.9, 0.0],
            [b_cx - 15, 80 - 15, b_cx + 15, 80 + 15, 0.9, 0.9, 1.0],
        ])
        gt = torch.tensor([[0.0, a_cx, 50.0, 30.0, 30.0],
                           [1.0, b_cx, 80.0, 30.0, 30.0]])
        ft = {k: {"boxes": torch.tensor([[a_cx + 2.0 * k, 50.0, 30.0, 30.0],
                                         [b_cx - 5.0 * k, 80.0, 30.0, 30.0]]),
                  "validity": torch.tensor([True, True])}}
        frames.append({"seq": "s", "frame_index": fi, "dets_xyxy7": dets,
                       "gt_targets": gt, "future_targets": ft})
    m = score_frames(frames, k=k, variant="full")
    # If A's and B's future boxes were swapped, ADE would be huge (~250px). Established
    # tracks predict exactly, so ADE is dominated only by the cold-start frame.
    assert m["ADE"] < 20.0, f"order misalignment suspected, ADE={m['ADE']}"


def test_warm_cold_stratification():
    # Object moving +3px/frame; GT future = true CV endpoint. Frame 0 is a cold start
    # (age 1, zero-velocity guess -> error = 3k*sqrt2); frames>=1 are warm (predict
    # exactly -> ~0). So ADE_warm ~ 0 and ADE_cold = 3k*sqrt2.
    k = 5
    frames = []
    for fi in range(4):
        cx = 100.0 + 3.0 * fi
        cy = 100.0 + 3.0 * fi
        frames.append(_frame(fi, cx, cy, 40.0, 40.0, cx + 3.0 * k, cy + 3.0 * k, k=k))
    m = score_frames(frames, k=k, variant="full")
    assert m["ADE_warm"] < 1e-3, f"warm ADE should be ~0, got {m['ADE_warm']}"
    assert math.isclose(m["ADE_cold"], 3.0 * k * math.sqrt(2), rel_tol=1e-3)
    assert math.isclose(m["_frac_warm"], 3.0 / 4.0, abs_tol=1e-6)  # 3 warm of 4 scored


def test_kalman_predicts_constant_velocity():
    # Feed a clean CV track; after a few updates the KF velocity converges and the
    # k-step prediction should land near the true future center.
    tr = KalmanTracker(iou_threshold=0.3)
    for fi in range(6):
        c = 100.0 + 4.0 * fi  # +4px/frame, 40px box
        tr.update(torch.tensor([[c - 20, c - 20, c + 20, c + 20]]), fi)
    px, _ = tr.predict(0, k=5)
    true = (100.0 + 4.0 * 5) + 4.0 * 5  # last center + 5*velocity
    assert abs(px - true) < 3.0, f"kalman CV predict {px} vs {true}"


def test_kalman_coasts_through_gap():
    # Object detected at frames 0,1, MISSED at 2,3 (those frames exist with a far-away
    # decoy so the frame advances), re-detected at 4 where CV says it should be.
    # The KF coasts track 0 through the gap, so the reappearance re-associates and the
    # track stays alive (age >= 3) instead of cold-starting a new track.
    tr = KalmanTracker(iou_threshold=0.2, max_age=10)
    def box(c, s=40):
        return torch.tensor([[c - s / 2, c - s / 2, c + s / 2, c + s / 2]])
    tr.update(box(100.0), 0)   # track 0 born
    tr.update(box(105.0), 1)   # +5/frame
    # frames 2,3: object missing; a decoy elsewhere keeps the frame populated
    tr.update(box(500.0), 2)
    tr.update(box(500.0), 3)
    # frame 4: object reappears where CV predicts (100 + 5*4 = 120)
    assign = tr.update(box(120.0), 4)
    reassoc_tid = assign[0]
    assert tr.age(reassoc_tid) >= 3, "KF should coast + re-associate, not cold-start"


def test_no_detections_frame_is_safe():
    k = 3
    frames = [
        {"seq": "s", "frame_index": 0, "dets_xyxy7": None,
         "gt_targets": torch.zeros((0, 5)), "future_targets": {k: {"boxes": torch.zeros((0, 4)),
                                                                   "validity": torch.zeros((0,), dtype=torch.bool)}}},
        _frame(1, 10.0, 10.0, 20.0, 20.0, 10.0, 10.0, k=k),
    ]
    m = score_frames(frames, k=k, variant="full")  # must not raise
    assert "ADE" in m or m == {} or m.get("_n_pred") is not None
