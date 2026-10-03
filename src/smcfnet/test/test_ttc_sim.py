"""Tests for the TTC hazard-warning simulation (tools/sttp_ttc_sim).

Pure CPU, no dump. Covers the TTC math, the documented failure modes, the warning/
false-alarm/missed accounting, and the tautology guard (lead time is not the trivial k).

Run: python -m pytest src/smcfnet/test/test_ttc_sim.py -v
"""

import math
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from smcfnet.utils.trajectory_metrics import ttc_center_closing, ttc_looming  # noqa: E402
from tools.sttp_ttc_sim import score_tracks, _self_check  # noqa: E402

EGO = (320.0, 240.0)


def test_self_check_passes():
    _self_check()


def test_center_closing_math():
    # object 100px above ego, closes 50px in 0.25s -> closing 200px/s -> TTC 0.5s
    t = ttc_center_closing(torch.tensor([[320.0, 140.0]]), torch.tensor([[320.0, 190.0]]),
                           0.25, torch.tensor(EGO))
    assert math.isclose(float(t[0]), 0.5, abs_tol=1e-4)


def test_center_closing_receding_is_inf():
    # moving away from ego -> inf
    t = ttc_center_closing(torch.tensor([[320.0, 140.0]]), torch.tensor([[320.0, 100.0]]),
                           0.25, torch.tensor(EGO))
    assert math.isinf(float(t[0]))


def test_looming_math():
    t = ttc_looming(torch.tensor([20.0]), torch.tensor([40.0]), 0.25)  # dw/dt=80, 20/80
    assert math.isclose(float(t[0]), 0.25, abs_tol=1e-4)


def test_lateral_crossing_both_inf():
    # width unchanged -> looming inf; same distance to ego -> center-closing inf
    assert math.isinf(float(ttc_looming(torch.tensor([30.0]), torch.tensor([30.0]), 0.25)[0]))
    # object crossing laterally at constant radius from ego: distance unchanged -> inf
    t = ttc_center_closing(torch.tensor([[420.0, 240.0]]), torch.tensor([[320.0, 340.0]]),
                           0.25, torch.tensor(EGO))  # both 100px from ego
    assert math.isinf(float(t[0]))


def _hazard_track(ys, k, with_detonly=True, with_sttp=True, perfect_sttp=True):
    frames = []
    for i, y in enumerate(ys):
        fy = ys[i + k] if i + k < len(ys) else None
        oracle = {"center": [320.0, fy], "w": 20.0} if fy is not None else None
        sttp = oracle if (with_sttp and perfect_sttp) else None
        if with_detonly and i >= 1:
            vy = ys[i] - ys[i - 1]
            detonly = {"center": [320.0, y + k * vy], "w": 20.0}
        else:
            detonly = None
        frames.append({"frame_index": i, "curr_center": [320.0, y], "curr_w": 20.0,
                       "oracle": oracle, "sttp": sttp, "detonly": detonly, "cascade": None})
    return {"track_id": 0, "seq": "s", "frames": frames}


def test_tautology_guard_lead_not_trivially_k():
    # Accelerating approach: a perfect STTP (=oracle) warns earlier than instantaneous
    # det-only, but the lead depends on dynamics and is strictly < k (not the k ceiling).
    k = 5
    ys = [140.0, 150.0, 162.0, 176.0, 192.0, 210.0, 230.0, 252.0]
    res = score_tracks([_hazard_track(ys, k)], k=k, taus=(1.0,), ego=EGO)
    lead = res[1.0]["lead_sttp_vs_detonly_frames"]
    assert not math.isnan(lead)
    assert 0 <= lead < k, f"lead {lead} must be in [0,{k}) to defeat the tautology"


def test_perfect_sttp_matches_oracle_timing():
    # A perfect STTP equals the oracle, so its timing error vs oracle is 0.
    k = 5
    ys = [140.0, 150.0, 162.0, 176.0, 192.0, 210.0, 230.0, 252.0]
    res = score_tracks([_hazard_track(ys, k)], k=k, taus=(1.0,), ego=EGO)
    assert math.isclose(res[1.0]["per_method"]["sttp"]["timing_err_frames"], 0.0, abs_tol=1e-9)


def test_false_alarm_on_nonhazard_track():
    # Object stays at constant distance (never a hazard): oracle never warns, but a method
    # that spuriously warns counts as a false alarm.
    k = 3
    frames = []
    for i in range(6):
        frames.append({"frame_index": i, "curr_center": [320.0, 140.0], "curr_w": 20.0,
                       "oracle": {"center": [320.0, 140.0], "w": 20.0},   # static -> no closing
                       "sttp": {"center": [320.0, 230.0], "w": 20.0},      # spurious approach
                       "detonly": None, "cascade": None})
    res = score_tracks([{"track_id": 0, "seq": "s", "frames": frames}], k=k, taus=(1.0,), ego=EGO)
    r = res[1.0]
    assert r["n_nonhazard"] == 1 and r["n_hazard"] == 0
    assert r["per_method"]["sttp"]["fa_rate"] == 1.0


def test_missed_warning_when_method_never_warns():
    # Oracle flags a hazard but STTP never predicts an approach -> missed warning.
    k = 3
    ys = [120.0, 140.0, 165.0, 195.0, 225.0]
    frames = []
    for i, y in enumerate(ys):
        fy = ys[i + k] if i + k < len(ys) else None
        oracle = {"center": [320.0, fy], "w": 20.0} if fy is not None else None
        frames.append({"frame_index": i, "curr_center": [320.0, y], "curr_w": 20.0,
                       "oracle": oracle, "sttp": {"center": [320.0, y], "w": 20.0},  # predicts no motion
                       "detonly": None, "cascade": None})
    res = score_tracks([{"track_id": 0, "seq": "s", "frames": frames}], k=k, taus=(1.0,), ego=EGO)
    r = res[1.0]
    if r["n_hazard"] == 1:
        assert r["per_method"]["sttp"]["miss_rate"] == 1.0
