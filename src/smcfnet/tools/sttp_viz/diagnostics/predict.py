# src/smcfnet/tools/sttp_viz/diagnostics/predict.py
"""Per-frame trajectory-prediction overlay (style A). Frame-level diagnostic."""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from PIL import Image

from ..base import Diagnostic
from ..registry import register_diagnostic
from ..decode import cxcywh_to_xywh

COL_CUR = "#ffffff"
COL_PRED = "#39c5ff"
COL_GT = "#5fe08a"


@register_diagnostic("predict")
class PredictDiagnostic(Diagnostic):
    LEVEL = "frame"

    def __init__(self, out_dir, style="A", fps=5, annotate=False, **kwargs):
        super().__init__(out_dir)
        self.style = style
        self.fps = int(fps)
        self.annotate = bool(annotate)
        self._frames = defaultdict(list)  # seq_name -> [png Path]

    def on_frame(self, bundle) -> None:
        paper = str(self.style).lower() == "paper"
        fig, ax = plt.subplots(figsize=(10, 7.5), dpi=300)
        ax.imshow(bundle.rgb)
        ax.axis("off")
        img_h, img_w = bundle.rgb.shape[:2]
        pred_steps = bundle.pred_future_steps  # [N,k,4] or None (older core)
        gt_steps = bundle.gt_future_steps
        valid_steps = bundle.valid_steps
        for i in range(len(bundle.detections)):
            x, y, w, h = cxcywh_to_xywh(bundle.detections[i, :4])
            ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=COL_CUR, lw=1.5))
            if bool(bundle.valid[i]):
                px, py, pw, ph = cxcywh_to_xywh(bundle.pred_future[i])
                ax.add_patch(Rectangle((px, py), pw, ph, fill=False, edgecolor=COL_PRED, lw=1.5, ls="--"))
                gx, gy, gw, gh = cxcywh_to_xywh(bundle.gt_future[i])
                ax.add_patch(Rectangle((gx, gy), gw, gh, fill=False, edgecolor=COL_GT, lw=1.5))
                ccx, ccy = float(bundle.detections[i, 0]), float(bundle.detections[i, 1])
                if pred_steps is not None:
                    # Full trajectory polyline through box centers t -> t+1 .. t+k.
                    pxs = [ccx] + [float(c) for c in pred_steps[i, :, 0]]
                    pys = [ccy] + [float(c) for c in pred_steps[i, :, 1]]
                    ax.plot(pxs, pys, color=COL_PRED, ls="--", lw=1.2, marker="o", ms=2.5)
                    if gt_steps is not None and valid_steps is not None:
                        vs = valid_steps[i]
                        gxs = [ccx] + [float(gt_steps[i, s, 0]) for s in range(len(vs)) if vs[s]]
                        gys = [ccy] + [float(gt_steps[i, s, 1]) for s in range(len(vs)) if vs[s]]
                        ax.plot(gxs, gys, color=COL_GT, lw=1.2, marker="o", ms=2.5)
                else:
                    ax.annotate("", xy=(float(bundle.pred_future[i, 0]), float(bundle.pred_future[i, 1])),
                                xytext=(ccx, ccy), arrowprops=dict(color=COL_PRED, arrowstyle="->", lw=0.8))
                if self.annotate and not np.isnan(bundle.per_obj_ade[i]):
                    ax.text(px, py - 3, f"{bundle.per_obj_ade[i]:.1f}", color=COL_PRED, fontsize=6)
        # Polylines/boxes reaching outside the frame must not autoscale the axes,
        # otherwise the image shrinks and per-method panels stop being congruent.
        ax.set_xlim(-0.5, img_w - 0.5)
        ax.set_ylim(img_h - 0.5, -0.5)
        if not paper:
            ade = bundle.meta.get("frame_ade", float("nan"))
            ax.set_title(f"{bundle.seq_name}  frame {bundle.frame_index}  "
                         f"k={bundle.meta.get('k', '?')}  ADE {ade:.1f}px", fontsize=9)
            ax.legend(handles=[
                Line2D([0], [0], color=COL_CUR, label="now (t)"),
                Line2D([0], [0], color=COL_PRED, ls="--", label="pred t+k"),
                Line2D([0], [0], color=COL_GT, label="GT t+k"),
            ], loc="lower right", fontsize=7, framealpha=0.6)

        seq_dir = self.out_dir / bundle.seq_name
        seq_dir.mkdir(parents=True, exist_ok=True)
        png = seq_dir / f"frame_{bundle.frame_index:06d}.png"
        fig.savefig(png, bbox_inches="tight", pad_inches=0 if paper else 0.1)
        plt.close(fig)
        self._frames[bundle.seq_name].append(png)

    def finalize(self) -> None:
        for seq, paths in self._frames.items():
            if not paths:
                continue
            # Image.open is lazy; load+copy under a context manager so the file
            # handle closes immediately (avoids FD exhaustion on long runs).
            imgs = []
            for p in sorted(paths):
                with Image.open(p) as im:
                    imgs.append(im.convert("RGB"))
            imgs[0].save(self.out_dir / f"{seq}.gif", save_all=True,
                         append_images=imgs[1:], duration=int(1000 / max(self.fps, 1)), loop=0)
