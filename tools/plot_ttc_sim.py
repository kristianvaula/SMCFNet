#!/usr/bin/env python3
"""Visualizers for the TTC hazard-warning simulation (thesis RQ1/RQ2).

Re-scores the per-horizon TTC dumps (outputs/sttp_results/ttc_dump/eraft_c_k*.pt) with the
SAME tested pure functions that produced the logged numbers (`assemble_tracks_from_dump` +
`score_tracks`), then renders publication PNG+PDF. The result CSV is lossy (a summary string),
so we re-score rather than parse it.

Figures (outputs/plots/ttc/):
  lead_vs_tau        headline: STTP warning lead (ms) vs tau, per k, 2 panels (vs detonly|cascade),
                     with a bootstrap CI band at k3/k5 and a COVERAGE row (n warned per method) so
                     the lead is read against its denominator (defeats survivorship bias).
  ttc_error_vs_k     MAE of each method's center-closing TTC vs the oracle TTC, per k. Explains the
                     k10 lead collapse as prediction-quality degradation (the pure RQ1 question).
  fa_miss            center-closing false-alarm/miss operating curve, marker size ~ n_hazard.
  looming_vs_center  honest negative result: looming lead is all-NaN (no line drawn), only FA shown.

Run:
  python tools/plot_ttc_sim.py            # uses default dump dir
  python tools/plot_ttc_sim.py --dump-dir outputs/sttp_results/ttc_dump --out outputs/plots/ttc
"""

from __future__ import annotations

import argparse
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

from smcfnet.utils.trajectory_metrics import ttc_center_closing  # noqa: E402
from tools._ttc_dump_io import assemble_tracks_from_dump  # noqa: E402
from tools.sttp_ttc_sim import score_tracks, TAUS_DEFAULT, FPS, IMAGE_WH  # noqa: E402

KS = (3, 5, 10)
PALETTE = {3: "#2196F3", 5: "#FF5722", 10: "#4CAF50"}
EGO = (IMAGE_WH[0] / 2, IMAGE_WH[1] / 2)
# regression anchor: re-scored value must match the logged number or the scoring code drifted.
ANCHOR = {"k": 5, "tau": 0.5, "lead_vs_cascade_ms": 194.7, "tol": 2.0}


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


def _load_dump(path: str) -> dict:
    d = torch.load(path, weights_only=False)
    assert isinstance(d, dict) and "frames" in d and "k" in d, f"bad TTC dump schema: {path}"
    f0 = d["frames"][0]
    for key in ("seq", "frame_index", "det_cxcywh", "sttp_future", "oracle_future", "oracle_valid"):
        assert key in f0, f"TTC dump frame missing '{key}': {path}"
    return d


def _save(fig, out_dir: str, name: str, payload: dict, sha: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    side = {"figure": name, "git_sha": sha, **payload}
    with open(os.path.join(out_dir, f"{name}.json"), "w") as fh:
        json.dump(side, fh, indent=2, default=float)
    print(f"  wrote {out_dir}/{name}.{{png,pdf,json}}")


# --------------------------------------------------------------------------- #
# Re-scoring
# --------------------------------------------------------------------------- #
def score_all(dump_dir: str, taus) -> Dict[int, Dict[str, dict]]:
    """{k: {center, looming, lead_ci, ttc_mae}} per horizon.

    Track-dependent figures (bootstrap CI, TTC-MAE) are computed inline here so the per-k
    track list can be freed before the next horizon. Holding all three at once OOMs (~40k
    tracks each). Only the small derived dicts survive.
    """
    out: Dict[int, Dict[str, dict]] = {}
    for k in KS:
        path = os.path.join(dump_dir, f"eraft_c_k{k}.pt")
        if not os.path.exists(path):
            print(f"  [skip] missing {path}")
            continue
        dump = _load_dump(path)
        tracks = assemble_tracks_from_dump(dump)
        entry = {
            "center": score_tracks(tracks, k=k, taus=taus, use_looming=False),
            "looming": score_tracks(tracks, k=k, taus=taus, use_looming=True),
            "ttc_mae": _method_ttc_mae(tracks, k),
            "lead_ci": {},
        }
        if k in (3, 5):  # CI band only where n is healthy
            for comp in ("detonly", "cascade"):
                entry["lead_ci"][comp] = {t: _bootstrap_lead(tracks, k, t, comp) for t in taus}
        out[k] = entry
        print(f"  scored k={k}: {len(tracks)} tracks", flush=True)
        del tracks, dump
    return out


def _check_anchor(scored: Dict[int, Dict[str, dict]]) -> None:
    k, tau = ANCHOR["k"], ANCHOR["tau"]
    if k not in scored:
        print("  [anchor] k5 dump absent, skipping regression check")
        return
    got = scored[k]["center"][tau]["lead_sttp_vs_cascade_ms"]
    assert abs(got - ANCHOR["lead_vs_cascade_ms"]) <= ANCHOR["tol"], (
        f"REGRESSION: re-scored k5 tau0.5 lead_vs_cascade={got:.1f}ms, "
        f"expected ~{ANCHOR['lead_vs_cascade_ms']}ms. Scoring code drifted from the logged run.")
    print(f"  [anchor] k5 tau0.5 lead_vs_cascade={got:.1f}ms OK")


def _bootstrap_lead(tracks: List[dict], k: int, tau: float, comparator: str,
                    n_boot: int = 500, seed: int = 0) -> tuple:
    """95% CI on the mean lead by resampling tracks. Returns (lo, hi) ms or (nan, nan)."""
    # per-track lead (frames): comparator_warn - sttp_warn, only where both warn at this tau.
    from tools.sttp_ttc_sim import _first_warning_frame, _ttc_series
    dt = k / FPS
    leads = []
    for tr in tracks:
        wf = {m: _first_warning_frame(_ttc_series(tr["frames"], m, dt, EGO, False), tau)
              for m in ("sttp", comparator, "oracle")}
        if wf["oracle"] is None or wf["sttp"] is None or wf[comparator] is None:
            continue
        leads.append(wf[comparator] - wf["sttp"])
    leads = np.array(leads, dtype=np.float64)
    if leads.size < 5:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = [rng.choice(leads, size=leads.size, replace=True).mean() for _ in range(n_boot)]
    ms = 1000.0 / FPS
    return float(np.percentile(means, 2.5) * ms), float(np.percentile(means, 97.5) * ms)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_lead_vs_tau(scored, taus, out_dir, sha) -> None:
    comps = [("detonly", "vs detection-only (RQ1)"), ("cascade", "vs CV-cascade (RQ2)")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), height_ratios=[2, 1], sharex="col")
    payload = {"taus": list(taus), "leads": {}, "coverage": {}}
    for col, (comp, title) in enumerate(comps):
        ax_lead, ax_cov = axes[0, col], axes[1, col]
        for k in sorted(scored):
            sc = scored[k]["center"]
            leads = np.array([sc[t][f"lead_sttp_vs_{comp}_ms"] for t in taus], dtype=np.float64)
            ml = np.ma.masked_invalid(leads)
            ax_lead.plot(taus, ml, "-o", color=PALETTE[k], label=f"k={k}")
            payload["leads"].setdefault(comp, {})[k] = leads.tolist()
            if comp in scored[k].get("lead_ci", {}):  # CI band only where n is healthy
                ci = scored[k]["lead_ci"][comp]
                lo = np.array([ci[t][0] for t in taus])
                hi = np.array([ci[t][1] for t in taus])
                m = np.isfinite(lo) & np.isfinite(hi)
                if m.any():
                    ax_lead.fill_between(np.array(taus)[m], lo[m], hi[m],
                                         color=PALETTE[k], alpha=0.15)
            # coverage: n warned by sttp vs comparator, per tau
            n_sttp = [sc[t]["per_method"]["sttp"]["n_warned_hazard"] for t in taus]
            n_comp = [sc[t]["per_method"][comp]["n_warned_hazard"] for t in taus]
            payload["coverage"].setdefault(comp, {})[k] = {"sttp": n_sttp, comp: n_comp}
            ax_cov.plot(taus, n_sttp, "-o", color=PALETTE[k], lw=1.2)
            ax_cov.plot(taus, n_comp, "--s", color=PALETTE[k], lw=1.0, alpha=0.7, ms=4)
        ax_lead.axhline(0, color="k", lw=0.8, ls=":")
        ax_lead.set_title(title)
        ax_lead.set_ylabel("warning lead (ms)\n(>0 = STTP earlier)")
        # annotate k10 small-n / zero-hazard taus
        for t in taus:
            nh = scored[max(scored)]["center"][t]["n_hazard"] if 10 in scored else 1
            if 10 in scored and nh == 0:
                ax_lead.annotate("n=0", (t, 0), color=PALETTE[10], fontsize=8,
                                 ha="center", va="bottom")
        ax_cov.set_ylabel("n warned\n(— STTP, -- foil)")
        ax_cov.set_xlabel(r"$\tau$ (s)")
    axes[0, 0].legend(title="horizon", fontsize=9)
    fig.suptitle("STTP warning lead vs alarm threshold (center-closing TTC)", y=0.98)
    fig.text(0.5, 0.005, "Lead averaged over hazards both methods warn; coverage row shows the "
             "denominator (n warned per method).", ha="center", fontsize=8, style="italic")
    _save(fig, out_dir, "lead_vs_tau", payload, sha)


def _method_ttc_mae(tracks, k) -> Dict[str, float]:
    """MAE (s) of each method's center-closing TTC vs the oracle TTC, over frames both finite."""
    dt = k / FPS
    ego = torch.tensor(EGO)
    err = {m: [] for m in ("sttp", "cascade", "detonly")}
    for tr in tracks:
        for fr in tr["frames"]:
            o = fr.get("oracle")
            if o is None:
                continue
            t_o = float(ttc_center_closing(torch.tensor([fr["curr_center"]]),
                                           torch.tensor([o["center"]]), dt, ego)[0])
            if not np.isfinite(t_o):
                continue
            for m in err:
                fut = fr.get(m)
                if fut is None:
                    continue
                t_m = float(ttc_center_closing(torch.tensor([fr["curr_center"]]),
                                               torch.tensor([fut["center"]]), dt, ego)[0])
                if np.isfinite(t_m):
                    err[m].append(abs(t_m - t_o))
    return {m: (float(np.mean(v)) if v else float("nan")) for m, v in err.items()}


def fig_ttc_error_vs_k(scored, out_dir, sha) -> None:
    methods = {"sttp": "#FF5722", "cascade": "#2196F3", "detonly": "#9C27B0"}
    ks = sorted(scored)
    mae = {k: scored[k]["ttc_mae"] for k in ks}
    fig, ax = plt.subplots(figsize=(7, 5))
    for m, c in methods.items():
        ys = [mae[k][m] for k in ks]
        ax.plot(ks, ys, "-o", color=c, label=m)
    ax.set_xlabel("horizon k (frames)")
    ax.set_ylabel("TTC MAE vs oracle (s)")
    ax.set_xticks(ks)
    ax.set_title("TTC prediction error vs horizon (lower = better future-box)")
    ax.legend()
    _save(fig, out_dir, "ttc_error_vs_k", {"mae": mae}, sha)


def fig_fa_miss(scored, taus, out_dir, sha) -> None:
    fig, ax = plt.subplots(figsize=(7, 5.5))
    payload = {}
    for k in sorted(scored):
        sc = scored[k]["center"]
        fa = [sc[t]["per_method"]["sttp"]["fa_rate"] for t in taus]
        miss = [sc[t]["per_method"]["sttp"]["miss_rate"] for t in taus]
        nh = [sc[t]["n_hazard"] for t in taus]
        sizes = [30 + 120 * (n / max(max(nh), 1)) for n in nh]
        ax.plot(fa, miss, "-", color=PALETTE[k], alpha=0.6, label=f"k={k}")
        ax.scatter(fa, miss, s=sizes, color=PALETTE[k], zorder=3)
        for t, x, y in zip(taus, fa, miss):
            ax.annotate(rf"$\tau$={t}", (x, y), fontsize=7, xytext=(3, 3),
                        textcoords="offset points")
        payload[k] = {"fa": fa, "miss": miss, "n_hazard": nh}
    ax.set_xlabel("false-alarm rate (STTP warns on non-hazard)")
    ax.set_ylabel("miss rate (STTP misses real hazard)")
    ax.set_title("STTP operating curve (center-closing, marker size ∝ n_hazard)")
    ax.legend(title="horizon")
    _save(fig, out_dir, "fa_miss", payload, sha)


def fig_looming_vs_center(scored, taus, out_dir, sha, k=5) -> None:
    if k not in scored:
        k = max(scored)
    fig, ax = plt.subplots(figsize=(7, 5))
    cen = scored[k]["center"]
    loo = scored[k]["looming"]
    fa_c = [cen[t]["per_method"]["sttp"]["fa_rate"] for t in taus]
    fa_l = [loo[t]["per_method"]["sttp"]["fa_rate"] for t in taus]
    ax.plot(taus, fa_c, "-o", color="#4CAF50", label="center-closing FA")
    ax.plot(taus, fa_l, "-s", color="#d62728", label="looming FA")
    # looming lead is all-NaN: state it, draw no lead line.
    lead_l = np.array([loo[t]["lead_sttp_vs_cascade_ms"] for t in taus])
    if not np.isfinite(lead_l).any():
        ax.text(0.5, 0.92, "looming TTC degenerate: lead all-NaN (size channel unusable)",
                transform=ax.transAxes, ha="center", fontsize=9, color="#d62728",
                bbox=dict(boxstyle="round", fc="#fff0f0", ec="#d62728"))
    ax.set_xlabel(r"$\tau$ (s)")
    ax.set_ylabel("false-alarm rate")
    ax.set_title(f"Center-closing vs looming TTC at k={k}")
    ax.legend()
    _save(fig, out_dir, "looming_vs_center", {"k": k, "fa_center": fa_c, "fa_looming": fa_l,
                                              "looming_lead": lead_l.tolist()}, sha)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", default="outputs/sttp_results/ttc_dump")
    ap.add_argument("--out", default="outputs/plots/ttc")
    ap.add_argument("--no-anchor", action="store_true", help="skip the regression anchor assert")
    args = ap.parse_args()

    _style()
    taus = TAUS_DEFAULT
    sha = _git_sha()
    print(f"re-scoring TTC dumps in {args.dump_dir} (git {sha[:8]})")
    scored = score_all(args.dump_dir, taus)
    if not scored:
        sys.exit("no TTC dumps found")
    if not args.no_anchor:
        _check_anchor(scored)
    fig_lead_vs_tau(scored, taus, args.out, sha)
    fig_ttc_error_vs_k(scored, args.out, sha)
    fig_fa_miss(scored, taus, args.out, sha)
    fig_looming_vs_center(scored, taus, args.out, sha)
    print("done")


if __name__ == "__main__":
    main()
