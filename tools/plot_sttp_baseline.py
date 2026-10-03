#!/usr/bin/env python3
"""Visualizers for the classical detect-track-predict baseline (thesis Exp 3.3 / RQ2 floor).

Re-scores the baseline dump (outputs/sttp_results/baseline_dump/val.pt) with the SAME tested
`score_frames` that produced the logged numbers (warm/cold + per-class are NOT in phase3.csv,
only re-scoring recovers them), and reads the learned-head reference (Design C eRAFT) from
phase3.csv. Renders publication PNG+PDF.

Figures (outputs/plots/baseline/):
  ade_vs_k    ADE (px) vs horizon for the 3 cv variants + the learned Design C eRAFT reference,
              with a per-sequence bootstrap CI band. Shows the classical floor ~2x above learned.
  warm_cold   warm- vs cold-start ADE per variant at k=5, with n labels. Conditions on track age,
              does not fully isolate motion quality from detection/selection effects.
  per_class   per-class ADE for cv_tracker at k=5 with instance counts (tiny classes flagged).

Run:
  python tools/plot_sttp_baseline.py
  python tools/plot_sttp_baseline.py --dump outputs/sttp_results/baseline_dump/val.pt \
      --csv outputs/sttp_results/phase3.csv --out outputs/plots/baseline
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tools.sttp_baseline_tracker import score_frames  # noqa: E402

KS = (3, 5, 10)
VARIANTS = {"full": ("cv_tracker", "#2196F3"), "twoframe": ("cv_tracker_2frame", "#FF5722"),
            "kalman": ("cv_tracker_kalman", "#4CAF50")}
LEARNED_DESIGN, LEARNED_FLOW, LEARNED_NAME = "C", "eraft", "Design C eRAFT (learned)"
# regression anchor: re-scored cv_tracker k5 ADE must match the logged 11.36 or scoring drifted.
ANCHOR = {"variant": "full", "k": 5, "ade": 11.36, "tol": 0.3}
CLASS_NAMES = {0: "pedestrian", 1: "rider", 2: "car", 3: "bus", 4: "truck",
               5: "bicycle", 6: "motorcycle", 7: "train"}


def _style() -> None:
    style = os.path.join(os.path.dirname(__file__), "..", "experiments", "p0", "paper.mplstyle")
    if os.path.exists(style):
        try:
            plt.style.use(style)
        except Exception:
            pass
    plt.rcParams.update({"figure.dpi": 200, "savefig.dpi": 200, "font.size": 11,
                         "axes.grid": True, "grid.alpha": 0.3,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "text.usetex": False})  # portable mathtext; re-render with LaTeX if wanted


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=os.path.dirname(__file__)).decode().strip()
    except Exception:
        return "unknown"


def _save(fig, out_dir: str, name: str, payload: dict, sha: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    with open(os.path.join(out_dir, f"{name}.json"), "w") as fh:
        json.dump({"figure": name, "git_sha": sha, **payload}, fh, indent=2, default=float)
    print(f"  wrote {out_dir}/{name}.{{png,pdf,json}}")


def _load_frames(path: str) -> List[dict]:
    frames = torch.load(path, weights_only=False)
    assert isinstance(frames, list) and frames, f"baseline dump must be a non-empty list: {path}"
    for key in ("seq", "frame_index", "future_targets"):
        assert key in frames[0], f"baseline dump frame missing '{key}': {path}"
    return frames


def score_all(frames: List[dict]) -> Dict[str, Dict[int, dict]]:
    """{variant: {k: metrics}} from score_frames."""
    out: Dict[str, Dict[int, dict]] = {}
    for variant in VARIANTS:
        out[variant] = {}
        for k in KS:
            out[variant][k] = score_frames(frames, k, variant=variant)
        print(f"  scored {VARIANTS[variant][0]}: "
              f"ADE@5={out[variant][5].get('ADE', float('nan')):.2f}px")
    return out


def _check_anchor(scored) -> None:
    got = scored[ANCHOR["variant"]][ANCHOR["k"]].get("ADE", float("nan"))
    assert abs(got - ANCHOR["ade"]) <= ANCHOR["tol"], (
        f"REGRESSION: re-scored cv_tracker k5 ADE={got:.2f}px, expected ~{ANCHOR['ade']}px. "
        f"Scoring code drifted from the logged run.")
    print(f"  [anchor] cv_tracker k5 ADE={got:.2f}px OK")


def _learned_ade(csv_path: str) -> Dict[int, float]:
    if not os.path.exists(csv_path):
        print(f"  [warn] {csv_path} missing, no learned reference line")
        return {}
    out = {}
    for x in csv.DictReader(open(csv_path)):
        if x.get("design") == LEARNED_DESIGN and x.get("flow_source") == LEARNED_FLOW:
            try:
                out[int(x["k"])] = float(x["ade5_px"])
            except (ValueError, KeyError):
                pass
    return out


def _seq_bootstrap_ade(frames, variant, k, n_boot=2000, seed=0) -> tuple:
    """95% CI on ADE by sequence-clustered bootstrap. (lo, hi) px or (nan, nan).

    Score each sequence ONCE to get (per-seq ADE, n_pairs), then resample sequences in pure
    numpy and recompute the pair-weighted pooled ADE. ~n_seq score passes, not n_boot*n_seq
    (re-scoring all frames per resample, with the slow Kalman filter, OOM/timeouts).
    """
    by_seq = {}
    for f in frames:
        by_seq.setdefault(f["seq"], []).append(f)
    if len(by_seq) < 3:
        return float("nan"), float("nan")
    ades, ns = [], []
    for sf in by_seq.values():
        m = score_frames(sf, k, variant=variant)
        a = m.get("ADE", float("nan"))
        n = m.get("_n_warm", 0.0) + m.get("_n_cold", 0.0)  # = scored pairs in this seq
        if np.isfinite(a) and n > 0:
            ades.append(a)
            ns.append(n)
    if len(ades) < 3:
        return float("nan"), float("nan")
    ades, ns = np.array(ades), np.array(ns)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ades), size=(n_boot, len(ades)))
    pooled = (ades[idx] * ns[idx]).sum(1) / ns[idx].sum(1)  # pair-weighted pooled ADE per resample
    return float(np.percentile(pooled, 2.5)), float(np.percentile(pooled, 97.5))


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_ade_vs_k(frames, scored, learned, out_dir, sha) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    payload = {"ade": {}, "ci": {}, "learned": learned}
    for variant, (name, color) in VARIANTS.items():
        ades = [scored[variant][k].get("ADE", float("nan")) for k in KS]
        ax.plot(KS, ades, "-o", color=color, label=name)
        payload["ade"][name] = ades
        lo = [_seq_bootstrap_ade(frames, variant, k)[0] for k in KS]
        hi = [_seq_bootstrap_ade(frames, variant, k)[1] for k in KS]
        m = np.isfinite(lo) & np.isfinite(hi)
        if m.any():
            ax.fill_between(np.array(KS)[m], np.array(lo)[m], np.array(hi)[m],
                            color=color, alpha=0.15)
        payload["ci"][name] = {"lo": lo, "hi": hi}
    if learned:
        lk = sorted(learned)
        ax.plot(lk, [learned[k] for k in lk], "--D", color="#9C27B0", label=LEARNED_NAME)
    ax.set_xlabel("horizon k (frames)")
    ax.set_ylabel("ADE (px)")
    ax.set_xticks(KS)
    ax.set_title("Classical baseline vs learned head: ADE by horizon")
    ax.legend()
    fig.text(0.5, 0.005, "Classical ADE re-scored from dump; learned ADE from logged CSV "
             "(different code paths). Band = 95% sequence-bootstrap CI.",
             ha="center", fontsize=8, style="italic")
    _save(fig, out_dir, "ade_vs_k", payload, sha)


def fig_warm_cold(scored, out_dir, sha, k=5) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5))
    names = [VARIANTS[v][0] for v in VARIANTS]
    x = np.arange(len(names))
    w = 0.38
    warm = [scored[v][k].get("ADE_warm", float("nan")) for v in VARIANTS]
    cold = [scored[v][k].get("ADE_cold", float("nan")) for v in VARIANTS]
    n_warm = [scored[v][k].get("_n_warm", 0) for v in VARIANTS]
    n_cold = [scored[v][k].get("_n_cold", 0) for v in VARIANTS]

    def _bar(off, vals, ns, color, label):
        for i, (val, n) in enumerate(zip(vals, ns)):
            if not np.isfinite(val):  # hatched empty bar, never a 0
                ax.bar(i + off, 0, w, edgecolor=color, fill=False, hatch="//")
                ax.annotate("n=0", (i + off, 0), ha="center", va="bottom", fontsize=8)
            else:
                ax.bar(i + off, val, w, color=color, label=label if i == 0 else None)
                ax.annotate(f"n={int(n)}", (i + off, val), ha="center", va="bottom", fontsize=7)

    _bar(-w / 2, warm, n_warm, "#2196F3", "warm (age≥2)")
    _bar(w / 2, cold, n_cold, "#FF5722", "cold (age==1)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=10)
    ax.set_ylabel("ADE (px)")
    ax.set_title(f"Warm- vs cold-start ADE at k={k}")
    ax.legend()
    fig.text(0.5, 0.005, "Conditions on track age (≥2 vs ==1); does not fully isolate motion "
             "quality from detection/selection effects.", ha="center", fontsize=8, style="italic")
    _save(fig, out_dir, "warm_cold", {"k": k, "warm": warm, "cold": cold,
                                      "n_warm": n_warm, "n_cold": n_cold}, sha)


def fig_per_class(frames, scored, out_dir, sha, variant="full", k=5) -> None:
    m = scored[variant][k]
    per_class = {int(key[len("ADE_cls"):]): float(v)
                 for key, v in m.items() if key.startswith("ADE_cls")}
    if not per_class:
        print("  [skip] per_class: no per-class ADE in metrics")
        return
    # instance counts per class from the dump GT
    counts = {}
    for fr in frames:
        gt = fr.get("gt_targets")
        if gt is not None and len(gt):
            for cls in gt[:, 0].tolist():
                counts[int(cls)] = counts.get(int(cls), 0) + 1
    classes = sorted(per_class)
    labels = [CLASS_NAMES.get(c, str(c)) for c in classes]
    vals = [per_class[c] for c in classes]
    ns = [counts.get(c, 0) for c in classes]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    bars = ax.bar(range(len(classes)), vals, color="#2196F3")
    for i, (v, n, c) in enumerate(zip(vals, ns, classes)):
        ax.annotate(f"n={n}", (i, v), ha="center", va="bottom", fontsize=7)
        if n < 50:  # tiny class: a high ADE here is not persuasive
            bars[i].set_color("#9E9E9E")
    hardest = classes[int(np.nanargmax(vals))]
    ax.annotate("hardest", (classes.index(hardest), max(vals)), color="#d62728",
                fontsize=9, xytext=(0, 14), textcoords="offset points", ha="center")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel("ADE (px)")
    ax.set_title(f"Per-class ADE, {VARIANTS[variant][0]} k={k} (grey = n<50, not persuasive)")
    _save(fig, out_dir, "per_class", {"variant": variant, "k": k,
                                      "per_class": per_class, "counts": counts}, sha)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="outputs/sttp_results/baseline_dump/val.pt")
    ap.add_argument("--csv", default="outputs/sttp_results/phase3.csv")
    ap.add_argument("--out", default="outputs/plots/baseline")
    ap.add_argument("--no-anchor", action="store_true")
    args = ap.parse_args()

    _style()
    sha = _git_sha()
    print(f"re-scoring baseline dump {args.dump} (git {sha[:8]})")
    frames = _load_frames(args.dump)
    scored = score_all(frames)
    if not args.no_anchor:
        _check_anchor(scored)
    learned = _learned_ade(args.csv)
    fig_ade_vs_k(frames, scored, learned, args.out, sha)
    fig_warm_cold(scored, args.out, sha)
    fig_per_class(frames, scored, args.out, sha)
    print("done")


if __name__ == "__main__":
    main()
