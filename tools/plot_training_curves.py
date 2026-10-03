#!/usr/bin/env python3
"""
Plot training curves from TensorBoard logs for model comparison.

Produces publication-quality figures comparing training loss, validation mAP,
and per-class AP50 between two runs.

Usage:
  python tools/plot_training_curves.py \
    --tb-dirs outputs/detr_test_yolox_s/tensorboard outputs/detr_test_detr_s/tensorboard \
    --names "YOLOX-S" "RF-DETR-S" \
    --output-dir outputs/comparison/plots/
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

logger = logging.getLogger(__name__)

DSEC_CLASSES = ("pedestrian", "rider", "car", "bus", "truck", "bicycle", "motorcycle", "train")

# Publication style
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def load_scalars(tb_dir: str, tag: str) -> tuple[list[float], list[float]]:
    """Load scalar data from TensorBoard logs. Returns (steps, values)."""
    ea = EventAccumulator(tb_dir)
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return [], []
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values


def plot_metric(ax, runs_data, tag: str, ylabel: str, title: str, use_epoch_x: bool = True):
    """Plot a single metric for multiple runs on the given axes."""
    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]
    for idx, (name, tb_dir) in enumerate(runs_data):
        steps, values = load_scalars(tb_dir, tag)
        if not steps:
            logger.warning("Tag '%s' not found in %s", tag, tb_dir)
            continue
        xlabel = "Epoch" if use_epoch_x else "Step"
        ax.plot(steps, values, label=name, color=colors[idx % len(colors)], linewidth=1.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()


def plot_training_loss(runs_data, output_dir: Path):
    """Plot epoch-averaged total loss comparison."""
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_metric(ax, runs_data, "epoch/total", "Total Loss", "Training Loss (epoch avg)")
    fig.savefig(output_dir / "training_loss.png")
    plt.close(fig)
    logger.info("Saved training_loss.png")


def plot_val_map(runs_data, output_dir: Path):
    """Plot validation mAP, mAP50, mAP75 curves."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = [
        ("val/mAP", "mAP", "Validation mAP"),
        ("val/mAP50", "mAP50", "Validation mAP50"),
        ("val/mAP75", "mAP75", "Validation mAP75"),
    ]
    for ax, (tag, ylabel, title) in zip(axes, metrics):
        plot_metric(ax, runs_data, tag, ylabel, title)
    fig.savefig(output_dir / "validation_map.png")
    plt.close(fig)
    logger.info("Saved validation_map.png")


def plot_lr(runs_data, output_dir: Path):
    """Plot learning rate schedule."""
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_metric(ax, runs_data, "train/lr", "Learning Rate", "Learning Rate Schedule", use_epoch_x=False)
    fig.savefig(output_dir / "learning_rate.png")
    plt.close(fig)
    logger.info("Saved learning_rate.png")


def plot_per_class_bar(runs_data, output_dir: Path):
    """Plot per-class AP50 bar chart from final validation step."""
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#2196F3", "#FF5722"]
    bar_width = 0.35
    x = np.arange(len(DSEC_CLASSES))

    for idx, (name, tb_dir) in enumerate(runs_data):
        ap50_values = []
        for cls_idx, cls_name in enumerate(DSEC_CLASSES):
            # Try both tag formats
            for tag in [f"val/AP50/{cls_name}", f"val/AP50/cls_{cls_idx}"]:
                steps, values = load_scalars(tb_dir, tag)
                if values:
                    ap50_values.append(values[-1])  # Last validation step
                    break
            else:
                ap50_values.append(0.0)

        offset = (idx - 0.5) * bar_width
        bars = ax.bar(x + offset, ap50_values, bar_width, label=name, color=colors[idx % len(colors)])
        # Add value labels on bars
        for bar, val in zip(bars, ap50_values):
            if val > 0.005:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Class")
    ax.set_ylabel("AP50")
    ax.set_title("Per-Class AP50 Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(DSEC_CLASSES, rotation=45, ha="right")
    ax.legend()
    fig.savefig(output_dir / "per_class_ap50.png")
    plt.close(fig)
    logger.info("Saved per_class_ap50.png")


def print_latex_table(runs_data):
    """Print a LaTeX table with final metrics."""
    print("\n% LaTeX table for thesis")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Detection head comparison on DSEC-Det test set (150 epochs).}")
    print("\\label{tab:head-comparison}")
    print("\\begin{tabular}{l" + "c" * len(runs_data) + "}")
    print("\\toprule")

    names = [name for name, _ in runs_data]
    print("Metric & " + " & ".join(names) + " \\\\")
    print("\\midrule")

    # mAP metrics
    for tag, label in [("val/mAP", "mAP"), ("val/mAP50", "mAP$_{50}$"), ("val/mAP75", "mAP$_{75}$")]:
        vals = []
        for _, tb_dir in runs_data:
            _, values = load_scalars(tb_dir, tag)
            vals.append(f"{values[-1]:.4f}" if values else "---")
        print(f"{label} & " + " & ".join(vals) + " \\\\")

    print("\\midrule")

    # Per-class AP50
    for cls_idx, cls_name in enumerate(DSEC_CLASSES):
        vals = []
        for _, tb_dir in runs_data:
            found = False
            for tag in [f"val/AP50/{cls_name}", f"val/AP50/cls_{cls_idx}"]:
                _, values = load_scalars(tb_dir, tag)
                if values:
                    vals.append(f"{values[-1]:.4f}")
                    found = True
                    break
            if not found:
                vals.append("---")
        print(f"AP$_{{50}}$/{cls_name} & " + " & ".join(vals) + " \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}\n")


def main():
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Plot training curves from TensorBoard logs")
    parser.add_argument("--tb-dirs", nargs="+", required=True, help="TensorBoard log directories")
    parser.add_argument("--names", nargs="+", required=True, help="Display names for each run")
    parser.add_argument("--output-dir", default="outputs/comparison/plots/", help="Output directory")
    parser.add_argument("--latex", action="store_true", help="Print LaTeX table")
    args = parser.parse_args()

    if len(args.tb_dirs) != len(args.names):
        print(f"ERROR: Got {len(args.tb_dirs)} dirs but {len(args.names)} names")
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs_data = list(zip(args.names, args.tb_dirs))

    plot_training_loss(runs_data, output_dir)
    plot_val_map(runs_data, output_dir)
    plot_lr(runs_data, output_dir)
    plot_per_class_bar(runs_data, output_dir)

    if args.latex:
        print_latex_table(runs_data)

    logger.info("All plots saved to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
