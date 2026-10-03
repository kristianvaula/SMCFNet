#!/usr/bin/env python3
"""TTC hazard-warning simulation (thesis RQ1).

Defensible version (per the 9-model design panel): every method estimates a TTC at the
CURRENT frame from what it can see, and all warnings are scored against a GT oracle. This
defeats the tautology where STTP, warning on its predicted t+k box while detection-only
warns on the current box, wins by exactly k*50ms by construction.

Four methods per object per frame, each with a "future box at t+k":
  - detonly : instantaneous projection (current velocity from t-1 -> t, extrapolated k frames)
  - cascade : constant-velocity extrapolation from the cv_tracker (RQ2 same-lookahead foil)
  - sttp    : the learned predicted box at t+k
  - oracle  : the real future-GT box at t+k (ground-truth warning timing)

A method WARNS at the first frame its TTC drops below tau. Per object we compare each
method's warning frame to the oracle's, and report lead time + false-alarm + missed-warning
rates over a pre-registered tau sweep, by horizon k. RQ1 = STTP vs detonly, RQ2 = STTP vs
cascade.

TTC formulations (reused/added in smcfnet.utils.trajectory_metrics):
  - center-closing (primary): distance-to-image-center / closing-velocity
  - looming (secondary): w / (dw/dt), probes STTP's unique size prediction

  # self-check (anywhere, no dump):
  python tools/sttp_ttc_sim.py
  # run on a dump that carries STTP predictions + GT track ids:
  python tools/sttp_ttc_sim.py --dump <ttc_dump.pt> --k 5 --csv ... [--wandb]

This module's scoring core (score_tracks) is pure and unit-tested on synthetic tracks; the
assemble_tracks_from_dump layer is the thin dump-specific glue.
"""

from __future__ import annotations

import argparse
import math
from typing import Dict, List, Optional, Tuple

import torch

from smcfnet.utils.trajectory_metrics import ttc_center_closing, ttc_looming

METHODS = ("detonly", "cascade", "sttp", "oracle")
PRED_METHODS = ("detonly", "cascade", "sttp")
TAUS_DEFAULT = (0.5, 1.0, 1.5, 2.0)
KS_DEFAULT = (3, 5, 10)
FPS = 20.0  # DSEC RGB ~20 Hz -> 50 ms/frame
IMAGE_WH = (640.0, 480.0)


def _ttc_at_frame(curr_center, curr_w, fut, dt_total, ego, use_looming) -> float:
    """TTC (s) for one object at one frame given its method-future box. inf if no future."""
    if fut is None:
        return float("inf")
    if use_looming:
        t = ttc_looming(torch.tensor([curr_w]), torch.tensor([fut["w"]]), dt_total)
    else:
        t = ttc_center_closing(torch.tensor([curr_center]), torch.tensor([fut["center"]]),
                               dt_total, torch.tensor(ego))
    return float(t[0])


def _ttc_series(frames: List[dict], method: str, dt_total: float, ego, use_looming):
    """Per-frame (frame_index, ttc) for one method over a track's frames."""
    out = []
    for fr in frames:
        out.append((fr["frame_index"],
                    _ttc_at_frame(fr["curr_center"], fr["curr_w"], fr.get(method),
                                  dt_total, ego, use_looming)))
    return out


def _first_warning_frame(series, tau: float) -> Optional[int]:
    for frame_index, ttc in series:
        if ttc < tau:
            return frame_index
    return None


def score_tracks(
    tracks: List[dict],
    k: int,
    taus: Tuple[float, ...] = TAUS_DEFAULT,
    ego: Tuple[float, float] = (IMAGE_WH[0] / 2, IMAGE_WH[1] / 2),
    fps: float = FPS,
    use_looming: bool = False,
) -> Dict[Tuple[float], dict]:
    """Score warning timing for every method against the oracle, per tau.

    Each track: {"track_id", "seq", "frames": [{frame_index, curr_center[2], curr_w,
    detonly{center,w}|None, cascade{...}|None, sttp{...}|None, oracle{...}|None}, ...]}.

    Returns {tau: {method: {lead_vs_oracle_mean, miss_rate, fa_rate, n_hazard, n_nonhazard,
    n_warned}, "_lead_sttp_vs_detonly_mean": ..., "_lead_sttp_vs_cascade_mean": ...}}.
    """
    dt_total = k / fps
    out: Dict = {}
    for tau in taus:
        # per-method accumulators
        timing_err = {m: [] for m in PRED_METHODS}   # method_wf - oracle_wf (hazard tracks)
        n_miss = {m: 0 for m in PRED_METHODS}
        n_fa = {m: 0 for m in PRED_METHODS}
        n_hazard = 0
        n_nonhazard = 0
        lead_sttp_detonly = []
        lead_sttp_cascade = []

        for tr in tracks:
            wf = {m: _first_warning_frame(_ttc_series(tr["frames"], m, dt_total, ego, use_looming), tau)
                  for m in METHODS}
            oracle_wf = wf["oracle"]
            if oracle_wf is None:
                n_nonhazard += 1
                for m in PRED_METHODS:
                    if wf[m] is not None:
                        n_fa[m] += 1
                continue
            n_hazard += 1
            for m in PRED_METHODS:
                if wf[m] is None:
                    n_miss[m] += 1
                else:
                    timing_err[m].append(wf[m] - oracle_wf)
            # lead time: how many frames earlier STTP warns than the foils (hazard, both warn)
            if wf["sttp"] is not None and wf["detonly"] is not None:
                lead_sttp_detonly.append(wf["detonly"] - wf["sttp"])
            if wf["sttp"] is not None and wf["cascade"] is not None:
                lead_sttp_cascade.append(wf["cascade"] - wf["sttp"])

        def _mean(xs):
            return float(sum(xs) / len(xs)) if xs else float("nan")

        ms = 1000.0 / fps
        per_method = {}
        for m in PRED_METHODS:
            n_warned = len(timing_err[m])
            per_method[m] = {
                "timing_err_frames": _mean(timing_err[m]),
                "timing_err_ms": _mean(timing_err[m]) * ms,
                "miss_rate": n_miss[m] / max(n_hazard, 1),
                "fa_rate": n_fa[m] / max(n_nonhazard, 1),
                "n_warned_hazard": n_warned,
            }
        out[tau] = {
            "per_method": per_method,
            "n_hazard": n_hazard,
            "n_nonhazard": n_nonhazard,
            "lead_sttp_vs_detonly_frames": _mean(lead_sttp_detonly),
            "lead_sttp_vs_detonly_ms": _mean(lead_sttp_detonly) * ms,
            "lead_sttp_vs_cascade_frames": _mean(lead_sttp_cascade),
            "lead_sttp_vs_cascade_ms": _mean(lead_sttp_cascade) * ms,
        }
    return out


# --------------------------------------------------------------------------- #
# Self-check: synthetic head-on closing track. Oracle warns; det-only (instantaneous,
# lagging) warns no earlier than STTP; the STTP-vs-detonly lead is NOT the trivial k.
# --------------------------------------------------------------------------- #
def _self_check() -> None:
    ego = (320.0, 240.0)
    # TTC math: head-on, object at (320,140) closing to ego over dt=0.25s by 50px.
    t = ttc_center_closing(torch.tensor([[320.0, 140.0]]), torch.tensor([[320.0, 190.0]]),
                           0.25, torch.tensor(ego))
    # d_curr=100, d_fut=50, closing=(100-50)/0.25=200px/s, ttc=100/200=0.5s
    assert abs(float(t[0]) - 0.5) < 1e-4, float(t[0])
    # Looming: w 20 -> 40 over 0.25s, dw/dt=80, ttc=20/80=0.25
    tl = ttc_looming(torch.tensor([20.0]), torch.tensor([40.0]), 0.25)
    assert abs(float(tl[0]) - 0.25) < 1e-4, float(tl[0])
    # Lateral-crossing: width unchanged -> looming inf; center same distance -> closing inf
    assert math.isinf(float(ttc_looming(torch.tensor([20.0]), torch.tensor([20.0]), 0.25)[0]))

    # Tautology guard: a hazard track where the object accelerates toward ego. The oracle
    # (true t+k box) crosses tau at some frame; a PERFECT STTP matches the oracle frame;
    # det-only (instantaneous velocity, lagging an accelerating object) warns later. The
    # STTP-over-detonly lead must be a real, dynamics-dependent number, NOT a constant k.
    k = 5
    taus = (1.0,)
    frames = []
    # object approaches ego center on the y-axis, accelerating: y goes 140,150,162,176,192...
    ys = [140.0, 150.0, 162.0, 176.0, 192.0, 210.0, 230.0]
    for i, y in enumerate(ys):
        cc = [320.0, y]
        # oracle = real future box k frames ahead (use the actual future y if available)
        fy = ys[i + k] if i + k < len(ys) else None
        oracle = {"center": [320.0, fy], "w": 20.0} if fy is not None else None
        # perfect STTP = oracle
        sttp = oracle
        # det-only: instantaneous velocity from previous frame, extrapolated k frames
        if i >= 1:
            vy = ys[i] - ys[i - 1]
            detonly = {"center": [320.0, y + k * vy], "w": 20.0}
        else:
            detonly = None
        frames.append({"frame_index": i, "curr_center": cc, "curr_w": 20.0,
                       "oracle": oracle, "sttp": sttp, "detonly": detonly, "cascade": None})
    res = score_tracks([{"track_id": 0, "seq": "s", "frames": frames}], k=k, taus=taus, ego=ego)
    r = res[1.0]
    assert r["n_hazard"] == 1, r
    lead = r["lead_sttp_vs_detonly_frames"]
    # STTP (=oracle) warns at least as early as det-only, and the lead is finite and < k
    # (det-only lags an accelerating object but not by the full horizon), so NOT tautological.
    assert not math.isnan(lead) and 0 <= lead < k, f"lead {lead} should be in [0,{k})"
    print("ok: TTC math + tautology-guard self-check passed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", default=None, help="TTC dump (.pt) with STTP preds + GT track ids")
    ap.add_argument("--k", type=int, action="append", default=None)
    ap.add_argument("--tau", type=float, action="append", default=None)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--label", default="ttc_sim_v5")
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    _self_check()
    if args.dump:
        from tools._ttc_dump_io import assemble_tracks_from_dump, run_sim  # lazy: glue layer
        ks = tuple(args.k) if args.k else KS_DEFAULT
        taus = tuple(args.tau) if args.tau else TAUS_DEFAULT
        run_sim(args.dump, ks, taus, args.csv, args.label, args.wandb)


if __name__ == "__main__":
    main()
