# src/smcfnet/tools/sttp_viz/bundle.py
"""Per-frame contract handed from the core to frame-level diagnostics."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np


@dataclass
class FrameBundle:
    seq_name: str
    frame_index: int
    rgb: np.ndarray                 # [H, W, 3] uint8
    detections: np.ndarray          # [N, 6] cx, cy, w, h, score, cls (abs px)
    pred_future: np.ndarray         # [M, 4] predicted box @ t+k (cxcywh abs px)
    gt_future: np.ndarray           # [M, 4] GT box @ t+k; NaN row if object left frame
    gt_current: np.ndarray          # [M, 4] GT box @ t
    valid: np.ndarray               # [M] bool — present at t+k
    per_obj_ade: np.ndarray         # [M] displacement error px, NaN where invalid
    pred_future_steps: Optional[np.ndarray] = None  # [M, k, 4] full predicted trajectory
    gt_future_steps: Optional[np.ndarray] = None    # [M, k, 4] GT trajectory, NaN if unmatched
    valid_steps: Optional[np.ndarray] = None        # [M, k] bool per-step GT presence
    flow: Optional[np.ndarray] = None        # [2, H, W]
    stem_features: Optional[List[np.ndarray]] = None
    meta: dict = field(default_factory=dict)  # {k, strides, run_name, frame_ade}
