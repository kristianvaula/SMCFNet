#!/usr/bin/env python3
"""Plot the scheduled event-warping/supervised loss weights.

Defaults match the 35-epoch schedule requested for the scheduled loss in
``smcfnet.model.losses.ScheduledEventWarpingSupervisedLoss``:

    python tools/plot_scheduled_loss_weights.py

The plotted values are the effective scalar multipliers:

    alpha * event_warping_weight(epoch)
    beta * supervised_weight(epoch)
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
STYLE_FILE = REPO_ROOT / "experiments" / "p0" / "paper.mplstyle"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from smcfnet.model.losses import ScheduledEventWarpingSupervisedLoss  # noqa: E402


def setup_style() -> None:
    """Apply the p0 thesis figure style with a LaTeX fallback."""
    if STYLE_FILE.exists():
        try:
            plt.style.use(str(STYLE_FILE))
        except Exception as exc:
            warnings.warn(f"Could not load {STYLE_FILE}: {exc}")

    if plt.rcParams.get("text.usetex"):
        fig, ax = plt.subplots(figsize=(1, 1))
        try:
            ax.set_xlabel(r"$\alpha$")
            fig.savefig(os.devnull, format="pdf")
        except Exception:
            warnings.warn("LaTeX rendering failed; falling back to matplotlib mathtext.")
            plt.rcParams["text.usetex"] = False
            plt.rcParams["mathtext.fontset"] = "cm"
        finally:
            plt.close(fig)

    plt.rcParams.update(
        {
            "font.size": 12.0,
            "axes.labelsize": 13.0,
            "legend.fontsize": 11.0,
            "xtick.labelsize": 12.0,
            "ytick.labelsize": 12.0,
            "figure.dpi": 200,
            "savefig.dpi": 300,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", "--a", dest="alpha", type=float, default=1.0)
    parser.add_argument("--beta", "--b", dest="beta", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--self-supervised-only-epochs", type=int, default=11)
    parser.add_argument("--schedule-start-epoch", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/plots/scheduled_loss_weights_e35_ssonly11.png"),
        help="Output image path, relative to the current working directory.",
    )
    parser.add_argument(
        "--format",
        choices=("png", "pdf", "svg"),
        default=None,
        help="Override output format. By default matplotlib infers it from --output.",
    )
    return parser.parse_args()


def scheduled_weights(
    alpha: float,
    beta: float,
    epochs: int,
    self_supervised_only_epochs: int,
    schedule_start_epoch: int,
) -> tuple[list[int], list[float], list[float], int]:
    loss = ScheduledEventWarpingSupervisedLoss(
        alpha=alpha,
        beta=beta,
        schedule_total_epochs=epochs,
        self_supervised_only_epochs=self_supervised_only_epochs,
        schedule_start_epoch=schedule_start_epoch,
    )

    epoch_ids = list(range(max(epochs, 1)))
    event_warping = []
    supervised = []
    for epoch in epoch_ids:
        ew_weight, sup_weight = loss.schedule_weights(epoch=epoch, num_epochs=epochs)
        event_warping.append(alpha * ew_weight)
        supervised.append(beta * sup_weight)

    final_selfsup_epochs = min(max(int(self_supervised_only_epochs), 0), max(int(epochs), 1))
    selfsup_start_epoch = max(int(epochs) - final_selfsup_epochs, int(schedule_start_epoch))
    return epoch_ids, event_warping, supervised, selfsup_start_epoch


def plot_weights(
    epoch_ids: list[int],
    event_warping: list[float],
    supervised: list[float],
    selfsup_start_epoch: int,
    args: argparse.Namespace,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.0), layout="constrained")

    ax.plot(
        epoch_ids,
        event_warping,
        label=r"$\alpha$ event-warping term",
        color="#1f77b4",
        linewidth=2.2,
    )
    ax.plot(
        epoch_ids,
        supervised,
        label=r"$\beta$ supervised EPE term",
        color="#d62728",
        linewidth=2.2,
    )

    if 0 <= selfsup_start_epoch <= max(epoch_ids):
        ax.axvspan(
            selfsup_start_epoch,
            max(epoch_ids),
            color="#2ca02c",
            alpha=0.08,
            label="self-supervised only",
        )
        ax.axvline(selfsup_start_epoch, color="#2ca02c", linestyle="--", linewidth=1.2)

    max_weight = max([1.0, *event_warping, *supervised])
    ax.set_ylim(0.0, max(1.2, max_weight * 1.15))
    ax.set_xlim(min(epoch_ids), max(epoch_ids))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss term multiplier")
    ax.legend(loc="center right")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, format=args.format)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    setup_style()
    epoch_ids, event_warping, supervised, selfsup_start_epoch = scheduled_weights(
        alpha=args.alpha,
        beta=args.beta,
        epochs=args.epochs,
        self_supervised_only_epochs=args.self_supervised_only_epochs,
        schedule_start_epoch=args.schedule_start_epoch,
    )
    plot_weights(epoch_ids, event_warping, supervised, selfsup_start_epoch, args)

    print(f"Saved {args.output}")
    print(
        f"Final self-supervised-only region starts at epoch {selfsup_start_epoch} "
        f"for {args.self_supervised_only_epochs} epoch(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
