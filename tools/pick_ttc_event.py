#!/usr/bin/env python3
"""Pick a hazard event for the Exp 3.4 TTC filmstrip figure.

Re-tracks a TTC dump exactly as the RQ1 sim does (assemble_tracks_from_dump), finds
tracks where the integrated STTP head warns earlier than the cascade at the given
tau, and emits per-frame annotation data (TTC per method, warn flags, boxes) for
the chosen track so the filmstrip can be composed offline.

Also prints the dataset-level lead numbers (vs cascade AND vs detonly) from
score_tracks so the figure caption quotes the reproduced value. Candidate
selection can additionally require a visually substantial object at warning
time. This prevents numerically valid TTC crossings on tiny distant boxes from
being presented as intuitive hazards.

Runs on the mac, CPU-only.

  python tools/pick_ttc_event.py --dump outputs/sttp_results/ttc_dump/eraft_c_k5.pt \
      --tau 0.5 --out outputs/sttp_results/thesis_viz/event.json
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from tools._ttc_dump_io import assemble_tracks_from_dump
from tools.sttp_ttc_sim import (
    score_tracks, _ttc_series, _first_warning_frame, METHODS, IMAGE_WH, FPS,
)

EGO = (IMAGE_WH[0] / 2, IMAGE_WH[1] / 2)
IMAGE_AREA = IMAGE_WH[0] * IMAGE_WH[1]


def find_full_box(dump_by_key, seq, frame_index, center) -> list:
    """Recover the full cxcywh det box for a track frame (track records keep only
    center + w) by matching the center back to the dump's det rows."""
    fr = dump_by_key.get((seq, frame_index))
    if fr is None:
        return [float(center[0]), float(center[1]), 0.0, 0.0]
    det = fr["det_cxcywh"]
    if det.shape[0] == 0:
        return [float(center[0]), float(center[1]), 0.0, 0.0]
    d = np.linalg.norm(det[:, :2] - np.asarray(center, dtype=np.float32), axis=1)
    return [float(x) for x in det[int(d.argmin())]]


def visual_hazard_metrics(dump_by_key, track: dict, warn_frames: dict, pad: int) -> dict:
    """Measure box occupancy and growth over the displayed warning interval."""
    by_frame = {int(fr["frame_index"]): fr for fr in track["frames"]}
    warning = by_frame[int(warn_frames["sttp"])]
    end_limit = int(warn_frames["cascade"]) + int(pad)
    end = max(
        (fr for fr in track["frames"] if int(fr["frame_index"]) <= end_limit),
        key=lambda fr: int(fr["frame_index"]),
    )
    warning_box = find_full_box(
        dump_by_key, track["seq"], int(warning["frame_index"]), warning["curr_center"],
    )
    end_box = find_full_box(
        dump_by_key, track["seq"], int(end["frame_index"]), end["curr_center"],
    )

    warning_area_fraction = max(warning_box[2], 0.0) * max(warning_box[3], 0.0) / IMAGE_AREA
    end_area_fraction = max(end_box[2], 0.0) * max(end_box[3], 0.0) / IMAGE_AREA
    area_growth = end_area_fraction / max(warning_area_fraction, 1e-12)
    return {
        "warning_box": warning_box,
        "end_box": end_box,
        "warning_area_fraction": float(warning_area_fraction),
        "end_area_fraction": float(end_area_fraction),
        "area_growth": float(area_growth),
        "end_frame": int(end["frame_index"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--out", required=True, help="event.json output path")
    ap.add_argument("--lead-min", type=int, default=3, help="min sttp-vs-cascade lead (frames)")
    ap.add_argument("--lead-max", type=int, default=5, help="max sttp-vs-cascade lead (frames)")
    ap.add_argument("--pick", type=int, default=0, help="candidate index to emit (default best)")
    ap.add_argument("--pad", type=int, default=2, help="strip frames beyond [sttp_wf, cascade_wf]")
    ap.add_argument(
        "--min-warning-area-frac", type=float, default=0.0,
        help="minimum current-box area divided by image area at the STTP warning",
    )
    ap.add_argument(
        "--min-end-area-frac", type=float, default=0.0,
        help="minimum current-box area divided by image area at the strip end",
    )
    ap.add_argument(
        "--min-area-growth", type=float, default=1.0,
        help="minimum end-area / warning-area ratio across the displayed interval",
    )
    ap.add_argument(
        "--skip-dataset-score", action="store_true",
        help="skip the aggregate score_tracks pass when only selecting visual candidates",
    )
    args = ap.parse_args()

    dump = torch.load(args.dump, weights_only=False)
    k = int(dump["k"])
    dt_total = k / FPS
    dump_by_key = {(fr["seq"], int(fr["frame_index"])): fr for fr in dump["frames"]}
    tracks = assemble_tracks_from_dump(dump)
    print(f"assembled {len(tracks)} tracks (k={k}) from {len(dump['frames'])} frames")

    # Dataset-level anchors for the caption.
    scored = None
    if not args.skip_dataset_score:
        scored = score_tracks(tracks, k=k, taus=(args.tau,))[args.tau]
        print(f"tau={args.tau}: lead_vs_cascade={scored['lead_sttp_vs_cascade_ms']:.1f}ms  "
              f"lead_vs_detonly={scored['lead_sttp_vs_detonly_ms']:.1f}ms  "
              f"n_hazard={scored['n_hazard']}")

    # Candidate tracks: everyone warns, STTP leads cascade by lead-min..lead-max frames.
    candidates = []
    for tr in tracks:
        series = {m: _ttc_series(tr["frames"], m, dt_total, EGO, False) for m in METHODS}
        wf = {m: _first_warning_frame(series[m], args.tau) for m in METHODS}
        if any(wf[m] is None for m in METHODS):
            continue
        lead = wf["cascade"] - wf["sttp"]
        if not (args.lead_min <= lead <= args.lead_max):
            continue
        if wf["sttp"] > wf["oracle"] + 1:  # STTP should be near-oracle-timed, not just less late
            continue
        first_fi = tr["frames"][0]["frame_index"]
        last_fi = tr["frames"][-1]["frame_index"]
        if wf["sttp"] - first_fi < args.pad or last_fi - wf["cascade"] < args.pad:
            continue  # need context frames around the warn window
        visual = visual_hazard_metrics(dump_by_key, tr, wf, args.pad)
        if visual["warning_area_fraction"] < args.min_warning_area_frac:
            continue
        if visual["end_area_fraction"] < args.min_end_area_frac:
            continue
        if visual["area_growth"] < args.min_area_growth:
            continue
        candidates.append({
            "track": tr, "wf": wf, "lead": lead, "series": series, "visual": visual,
        })

    candidates.sort(key=lambda c: (
        -c["visual"]["end_area_fraction"],
        -c["visual"]["warning_area_fraction"],
        abs(c["wf"]["sttp"] - c["wf"]["oracle"]),
        -c["lead"],
    ))
    print(f"{len(candidates)} candidate tracks (lead {args.lead_min}..{args.lead_max} frames)")
    for i, c in enumerate(candidates[:10]):
        tr, wf, vis = c["track"], c["wf"], c["visual"]
        print(f"  [{i}] {tr['seq']} track {tr['track_id']}: "
              f"oracle_wf={wf['oracle']} sttp_wf={wf['sttp']} cascade_wf={wf['cascade']} "
              f"detonly_wf={wf['detonly']} lead={c['lead']}f ({c['lead'] * 1000 / FPS:.0f}ms) "
              f"area={100 * vis['warning_area_fraction']:.2f}%->"
              f"{100 * vis['end_area_fraction']:.2f}% growth={vis['area_growth']:.2f}x")
    if not candidates:
        raise SystemExit("no candidate found; relax --lead-min/--lead-max or --tau")

    c = candidates[min(args.pick, len(candidates) - 1)]
    tr, wf, series = c["track"], c["wf"], c["series"]
    lo = wf["sttp"] - args.pad
    hi = wf["cascade"] + args.pad
    ttc_by_frame = {m: dict(series[m]) for m in METHODS}
    frames = []
    for fr in tr["frames"]:
        fi = fr["frame_index"]
        if not (lo <= fi <= hi):
            continue
        frames.append({
            "frame_index": fi,
            "box_cxcywh": find_full_box(dump_by_key, tr["seq"], fi, fr["curr_center"]),
            "ttc": {m: (None if np.isinf(ttc_by_frame[m][fi]) else round(ttc_by_frame[m][fi], 3))
                    for m in METHODS},
            "warned": {m: fi >= wf[m] for m in METHODS},
        })

    event = {
        "dump": args.dump, "k": k, "tau": args.tau, "fps": FPS,
        "seq": tr["seq"], "track_id": int(tr["track_id"]),
        "warn_frame": wf, "lead_frames": c["lead"],
        "lead_ms": c["lead"] * 1000.0 / FPS,
        "dataset_lead_vs_cascade_ms": (
            None if scored is None else scored["lead_sttp_vs_cascade_ms"]
        ),
        "dataset_lead_vs_detonly_ms": (
            None if scored is None else scored["lead_sttp_vs_detonly_ms"]
        ),
        "visual_hazard": c["visual"],
        "strip_range": [lo, hi],
        "frames": frames,
        "frame_list": f"{tr['seq']}:{','.join(str(f['frame_index']) for f in frames)}",
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(event, f, indent=2)
    print(f"\npicked {tr['seq']} track {tr['track_id']} frames {lo}..{hi} "
          f"(lead {c['lead']}f = {event['lead_ms']:.0f}ms)")
    print(f"frame list: {event['frame_list']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
