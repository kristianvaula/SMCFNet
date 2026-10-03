# src/smcfnet/tools/sttp_viz/decode.py
"""Pure geometry / decode / ADE helpers (no torch model dependency)."""
from __future__ import annotations
import numpy as np


def cxcywh_to_xywh(box) -> tuple:
    """(cx,cy,w,h) -> (x_topleft, y_topleft, w, h) for matplotlib Rectangle."""
    cx, cy, w, h = (float(v) for v in box[:4])
    return (cx - w / 2.0, cy - h / 2.0, w, h)


def pair_ade(pred_future: np.ndarray, gt_future: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Per-object center displacement error (px). NaN where invalid."""
    pred = np.asarray(pred_future, dtype=np.float64)
    gt = np.asarray(gt_future, dtype=np.float64)
    d = np.sqrt((pred[:, 0] - gt[:, 0]) ** 2 + (pred[:, 1] - gt[:, 1]) ** 2)
    d[~np.asarray(valid, dtype=bool)] = np.nan
    return d.astype(np.float32)


def decode_future_boxes(fut_preds, expanded_strides, grids):
    """Decode the STTP head's per-anchor future predictions to absolute cxcywh,
    for ALL anchors. Exact replica of STTPLoss._decode_absolute math (sttp_loss.py:52-55)
    so the tool's boxes equal training/eval. fut_preds: list of [B,4,H_k,W_k];
    expanded_strides: [1,n_anchors]; grids: [1,n_anchors,2]. Returns [B,n_anchors,4].
    """
    import torch
    flat = []
    for pred in fut_preds:
        B, C, H, W = pred.shape
        flat.append(pred.permute(0, 2, 3, 1).reshape(B, -1, C))
    flat = torch.cat(flat, dim=1)  # [B, n_anchors_all, 4]
    decoded = flat.clone()
    decoded[..., :2] = (flat[..., :2] + grids) * expanded_strides.unsqueeze(-1)
    decoded[..., 2:4] = torch.exp(flat[..., 2:4].clamp(max=10.0)) * expanded_strides.unsqueeze(-1)
    return decoded
