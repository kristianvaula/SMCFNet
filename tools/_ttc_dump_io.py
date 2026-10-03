"""Glue between a TTC dump (tools/sttp_ttc_dump) and the pure scorer (tools/sttp_ttc_sim).

Assembles per-object tracks from a single-horizon dump by re-tracking the detector's
detections (the cv_tracker the classical baseline uses), attaching each method's future box
per frame:
  oracle  = dumped GT box at t+k (valid rows only)
  sttp    = dumped STTP predicted box at t+k
  cascade = cv_tracker windowed constant-velocity extrapolation to t+k (size kept constant)
  detonly = instantaneous (last-2-observation) velocity extrapolated to t+k (size constant)

Then runs the pure scorer and writes a CSV row per (method, k, tau) plus optional wandb.
"""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import torch

from tools.sttp_baseline_tracker import FrameGapTracker, _xyxy_to_cxcywh  # reuse the tracker
from tools.sttp_ttc_sim import score_tracks, PRED_METHODS, TAUS_DEFAULT, IMAGE_WH, FPS


def _cxcywh_to_xyxy(b: np.ndarray) -> torch.Tensor:
    cx, cy, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return torch.tensor(np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1),
                        dtype=torch.float32)


def assemble_tracks_from_dump(dump: dict) -> List[dict]:
    """Re-track dumped detections per sequence and attach the four method futures."""
    k = int(dump["k"])
    frames = dump["frames"]
    # group by sequence
    by_seq: Dict[str, List[dict]] = {}
    for fr in frames:
        by_seq.setdefault(fr["seq"], []).append(fr)

    tracks: List[dict] = []
    for seq, seq_frames in by_seq.items():
        seq_frames.sort(key=lambda f: f["frame_index"])
        tracker = FrameGapTracker(iou_threshold=0.3)
        track_frames: Dict[int, List[dict]] = {}  # tid -> assembled frame records
        for fr in seq_frames:
            det = fr["det_cxcywh"]
            n = int(det.shape[0])
            if n == 0:
                tracker.update(torch.zeros((0, 4)), int(fr["frame_index"]))
                continue
            assign = tracker.update(_cxcywh_to_xyxy(det), int(fr["frame_index"]))
            sttp = fr["sttp_future"]
            orc = fr["oracle_future"]
            orc_valid = fr["oracle_valid"]
            for d in range(n):
                tid = assign[d]
                curr_center = [float(det[d, 0]), float(det[d, 1])]
                curr_w = float(det[d, 2])
                # oracle: only if present at t+k and finite
                oracle = None
                if bool(orc_valid[d]) and np.isfinite(orc[d, :2]).all():
                    oracle = {"center": [float(orc[d, 0]), float(orc[d, 1])], "w": float(orc[d, 2])}
                sttp_fut = ({"center": [float(sttp[d, 0]), float(sttp[d, 1])], "w": float(sttp[d, 2])}
                            if np.isfinite(sttp[d, :2]).all() else None)
                # cascade (windowed CV) + detonly (2-frame) from the tracker; size kept constant
                cx_c, cy_c = tracker.predict(tid, k, two_frame=False)
                cx_d, cy_d = tracker.predict(tid, k, two_frame=True)
                cascade = {"center": [cx_c, cy_c], "w": curr_w}
                detonly = {"center": [cx_d, cy_d], "w": curr_w}
                track_frames.setdefault(tid, []).append({
                    "frame_index": int(fr["frame_index"]),
                    "curr_center": curr_center, "curr_w": curr_w,
                    "oracle": oracle, "sttp": sttp_fut,
                    "cascade": cascade, "detonly": detonly,
                })
        for tid, fl in track_frames.items():
            tracks.append({"track_id": tid, "seq": seq, "frames": fl})
    return tracks


def _csv_rows(scored: dict, k: int, label: str, dump_path: str) -> List[dict]:
    import datetime as _dt
    ts = _dt.datetime.now().isoformat(timespec="seconds")
    rows = []
    for tau, res in scored.items():
        base = {
            "timestamp": ts, "label": f"{label}_k{k}_tau{tau}", "design": "ttc_sim",
            "flow_source": "eraft", "k": k, "config_path": dump_path,
            "ade5_px": "", "fde_px": "", "ade_per_step": "", "per_class_ade": "",
            "map50": "", "map_full": "",
        }
        # encode the headline lead times + per-method timing/fa/miss into per_class_ade text
        lead_d = res["lead_sttp_vs_detonly_ms"]
        lead_c = res["lead_sttp_vs_cascade_ms"]
        pm = res["per_method"]
        summary = (f"tau={tau} lead_vs_detonly={lead_d:.1f}ms lead_vs_cascade={lead_c:.1f}ms "
                   f"sttp[err={pm['sttp']['timing_err_ms']:.1f}ms,fa={pm['sttp']['fa_rate']:.3f},"
                   f"miss={pm['sttp']['miss_rate']:.3f}] "
                   f"detonly[err={pm['detonly']['timing_err_ms']:.1f}ms,fa={pm['detonly']['fa_rate']:.3f}] "
                   f"n_haz={res['n_hazard']}")
        base["per_class_ade"] = summary
        rows.append(base)
    return rows


def run_sim(dump_path: str, ks, taus, csv_path, label: str, wandb_on: bool) -> None:
    dump = torch.load(dump_path, weights_only=False)
    k = int(dump["k"])
    taus = tuple(taus) if taus else TAUS_DEFAULT
    tracks = assemble_tracks_from_dump(dump)
    print(f"assembled {len(tracks)} tracks (k={k}) from {len(dump['frames'])} frames")

    for use_looming, name in [(False, "center_closing"), (True, "looming")]:
        scored = score_tracks(tracks, k=k, taus=taus, use_looming=use_looming)
        print(f"  --- {name} (k={k}) ---")
        for tau, res in scored.items():
            print(f"  tau={tau}: lead_vs_detonly={res['lead_sttp_vs_detonly_ms']:.1f}ms  "
                  f"lead_vs_cascade={res['lead_sttp_vs_cascade_ms']:.1f}ms  "
                  f"n_hazard={res['n_hazard']}  n_nonhazard={res['n_nonhazard']}  "
                  f"sttp_fa={res['per_method']['sttp']['fa_rate']:.3f}  "
                  f"sttp_miss={res['per_method']['sttp']['miss_rate']:.3f}")
        if csv_path:
            _append_csv(csv_path, _csv_rows(scored, k, f"{label}_{name}", dump_path))
        if wandb_on:
            _wandb_log(scored, k, name, label, dump_path)


def _append_csv(csv_path: str, rows: List[dict]) -> None:
    import csv as _csv
    import os
    from tools.sttp_eval_aggregate import CSV_FIELDS
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in CSV_FIELDS})


def _wandb_log(scored: dict, k: int, formulation: str, label: str, dump_path: str) -> None:
    try:
        import wandb
    except ImportError:
        print("  [wandb] not installed; skipping")
        return
    try:
        run = wandb.init(project="smcfnet_sttp_baseline", entity="smcm-mcfnet",
                         name=f"{label}_ttc_{formulation}_k{k}", job_type="ttc_sim",
                         tags=["ttc", "rq1", formulation], config={"k": k, "dump": dump_path})
    except Exception as e:
        print(f"  [wandb] init failed ({e}); skipping")
        return
    try:
        cols = ["tau", "lead_vs_detonly_ms", "lead_vs_cascade_ms", "n_hazard", "n_nonhazard",
                "sttp_err_ms", "sttp_fa", "sttp_miss", "detonly_err_ms", "cascade_err_ms"]
        table = wandb.Table(columns=cols)
        for tau, res in scored.items():
            pm = res["per_method"]
            table.add_data(tau, res["lead_sttp_vs_detonly_ms"], res["lead_sttp_vs_cascade_ms"],
                           res["n_hazard"], res["n_nonhazard"], pm["sttp"]["timing_err_ms"],
                           pm["sttp"]["fa_rate"], pm["sttp"]["miss_rate"],
                           pm["detonly"]["timing_err_ms"], pm["cascade"]["timing_err_ms"])
            pre = f"k{k}/{formulation}/tau{tau}"
            run.summary[f"{pre}/lead_vs_detonly_ms"] = res["lead_sttp_vs_detonly_ms"]
            run.summary[f"{pre}/lead_vs_cascade_ms"] = res["lead_sttp_vs_cascade_ms"]
            run.summary[f"{pre}/sttp_fa"] = pm["sttp"]["fa_rate"]
            run.summary[f"{pre}/sttp_miss"] = pm["sttp"]["miss_rate"]
        run.log({f"ttc_{formulation}_k{k}": table})
        print(f"  [wandb] logged ttc {formulation} k{k}")
    except Exception as e:
        print(f"  [wandb] logging failed ({e})")
    finally:
        run.finish()
