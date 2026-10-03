"""Smoke tests for the result visualizers (tools/plot_ttc_sim, tools/plot_sttp_baseline).

Pure CPU, no real dumps. Builds tiny synthetic dumps, runs each plotter's score->plot path
into a tmp dir, and asserts the figures + sidecar JSON appear and are non-empty. Also covers
the degenerate-NaN path: a dump whose looming channel is all-NaN must still render a figure
with no looming lead line (never a 0).

Run: python -m pytest src/smcfnet/test/test_plot_results_smoke.py -v
"""

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import tools.plot_ttc_sim as ttc  # noqa: E402
import tools.plot_sttp_baseline as base  # noqa: E402


def _tiny_ttc_dump(k=5, n_frames=8):
    """One sequence, one object approaching image center, CONSTANT width (looming degenerate)."""
    frames = []
    for i in range(n_frames):
        cy = 140.0 + 12.0 * i  # moving down toward ego center (240)
        det = np.array([[320.0, cy, 20.0, 20.0]], dtype=np.float32)
        # sttp/oracle future = position k frames ahead (same dynamics), width constant -> looming nan
        fy = 140.0 + 12.0 * (i + k)
        fut = np.array([[320.0, fy, 20.0, 20.0]], dtype=np.float32)
        valid = i + k < n_frames + k  # always present in this toy
        frames.append({
            "seq": "toy", "frame_index": i, "k": k,
            "det_cxcywh": torch.tensor(det),
            "sttp_future": torch.tensor(fut),
            "oracle_future": torch.tensor(fut),
            "oracle_valid": torch.tensor([valid]),
        })
    return {"k": k, "frames": frames}


def _tiny_baseline_dump(n_frames=8):
    frames = []
    for i in range(n_frames):
        cy = 140.0 + 12.0 * i
        # dets_xyxy7: x1,y1,x2,y2,obj,cls_conf,cls
        dets = torch.tensor([[310.0, cy - 10, 330.0, cy + 10, 0.9, 0.9, 2.0]])
        gt = torch.tensor([[2.0, 320.0, cy, 20.0, 20.0]])  # class 2 (car)
        ft = {}
        for k in (3, 5, 10):
            fy = 140.0 + 12.0 * (i + k)
            ft[k] = {"boxes": torch.tensor([[320.0, fy, 20.0, 20.0]]),
                     "validity": torch.tensor([True])}
        frames.append({"seq": "toy", "frame_index": i, "dets_xyxy7": dets,
                       "gt_targets": gt, "future_targets": ft})
    return frames


def _assert_outputs(out_dir, names):
    for name in names:
        for ext in ("png", "pdf", "json"):
            p = os.path.join(out_dir, f"{name}.{ext}")
            assert os.path.exists(p) and os.path.getsize(p) > 0, f"missing/empty {p}"
        json.load(open(os.path.join(out_dir, f"{name}.json")))  # valid JSON


def test_ttc_plotters_run_and_looming_is_nan(tmp_path):
    ttc._style()
    dump_dir = tmp_path / "dumps"
    dump_dir.mkdir()
    torch.save(_tiny_ttc_dump(5), dump_dir / "eraft_c_k5.pt")
    torch.save(_tiny_ttc_dump(3), dump_dir / "eraft_c_k3.pt")
    scored = ttc.score_all(str(dump_dir), ttc.TAUS_DEFAULT)
    assert 5 in scored and "ttc_mae" in scored[5]

    out = str(tmp_path / "ttc")
    ttc.fig_lead_vs_tau(scored, ttc.TAUS_DEFAULT, out, "test")
    ttc.fig_ttc_error_vs_k(scored, out, "test")
    ttc.fig_fa_miss(scored, ttc.TAUS_DEFAULT, out, "test")
    ttc.fig_looming_vs_center(scored, ttc.TAUS_DEFAULT, out, "test")
    _assert_outputs(out, ["lead_vs_tau", "ttc_error_vs_k", "fa_miss", "looming_vs_center"])

    # degenerate-NaN guarantee: constant width -> looming lead all-NaN, rendered as no line not 0.
    loom = json.load(open(os.path.join(out, "looming_vs_center.json")))
    assert all(not np.isfinite(x) for x in loom["looming_lead"]), loom["looming_lead"]


def test_baseline_plotters_run(tmp_path):
    base._style()
    frames = _tiny_baseline_dump()
    scored = base.score_all(frames)
    assert "full" in scored and 5 in scored["full"]

    out = str(tmp_path / "baseline")
    base.fig_ade_vs_k(frames, scored, {3: 3.8, 5: 5.41, 10: 10.79}, out, "test")
    base.fig_warm_cold(scored, out, "test")
    base.fig_per_class(frames, scored, out, "test")
    # per_class only writes if per-class ADE exists; ade_vs_k + warm_cold always do.
    _assert_outputs(out, ["ade_vs_k", "warm_cold"])
