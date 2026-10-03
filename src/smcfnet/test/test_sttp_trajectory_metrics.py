"""Self-checks for STTP trajectory metrics (WS1).

Covers two things the Phase 3 cards depend on:
  1. The multi-step accumulator emits ADE (mean over valid steps), a genuinely
     distinct FDE (final-step error), per-step ADE_step{j}, and per-class ADE.
  2. compute_ade / compute_fde agree when T==1 and diverge when T>1.

Run: python -m pytest src/smcfnet/test/test_sttp_trajectory_metrics.py -v
"""

from __future__ import annotations

import torch

from smcfnet.utils.trajectory_metrics import compute_ade, compute_fde
from smcfnet.utils.validation import TrajectoryMetricsAccumulator


def _xyxy(cx, cy, w, h):
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def test_accumulator_per_step_and_distinct_fde():
    acc = TrajectoryMetricsAccumulator(iou_threshold=0.5)

    # Two objects, distinct classes, k=2 trajectory. Detections == GT so IoU == 1.
    targets = torch.tensor(
        [[3.0, 100.0, 100.0, 20.0, 20.0],   # class 3 @ (100,100)
         [7.0, 200.0, 200.0, 20.0, 20.0]],  # class 7 @ (200,200)
    )
    dets = torch.tensor(
        [_xyxy(100, 100, 20, 20) + [0.9, 0.0, 3.0],
         _xyxy(200, 200, 20, 20) + [0.9, 0.0, 7.0]],
    )  # [N,7] xyxy + score + obj + cls

    # GT future centers == current at both steps. Predicted offsets:
    #   obj0: step1 (3,4)->L2 5, step2 (6,8)->L2 10
    #   obj1: step1 (5,12)->L2 13, step2 (8,15)->L2 17
    fut_decoded = torch.tensor([[  # [B=1, A=2, k=2, 4]
        [[103.0, 104.0, 20.0, 20.0], [106.0, 108.0, 20.0, 20.0]],
        [[205.0, 212.0, 20.0, 20.0], [208.0, 215.0, 20.0, 20.0]],
    ]])
    future_targets = {
        "boxes": torch.tensor([[  # [1, 2, k=2, 4]
            [[100.0, 100.0, 20.0, 20.0], [100.0, 100.0, 20.0, 20.0]],
            [[200.0, 200.0, 20.0, 20.0], [200.0, 200.0, 20.0, 20.0]],
        ]]),
        "validity": torch.ones(1, 2, 2, dtype=torch.bool),
    }
    det_pre_nms = torch.zeros(1, 2, 4)  # unused when anchor indices are given
    anchor_idx = [torch.tensor([0, 1], dtype=torch.long)]  # det i -> anchor i

    acc.update(
        det_pre_nms,
        fut_decoded,
        [dets],
        [targets],
        future_targets,
        anchor_indices_per_image=anchor_idx,
    )
    out = acc.compute()

    assert abs(out["ADE"] - 11.25) < 1e-6, out         # (5+10+13+17)/4
    assert abs(out["FDE"] - 13.5) < 1e-6, out          # final step (10+17)/2
    assert out["FDE"] != out["ADE"], out               # genuinely distinct now
    assert abs(out["ADE_step1"] - 9.0) < 1e-6, out     # (5+13)/2
    assert abs(out["ADE_step2"] - 13.5) < 1e-6, out    # (10+17)/2
    assert abs(out["ADE_cls3"] - 7.5) < 1e-6, out      # obj0 (5+10)/2
    assert abs(out["ADE_cls7"] - 15.0) < 1e-6, out     # obj1 (13+17)/2


def test_ade_fde_functions_single_vs_multi_step():
    valid = torch.ones(1, 1)
    preds1 = torch.tensor([[[3.0, 4.0]]])   # [N=1, T=1, 2]
    tgt1 = torch.zeros(1, 1, 2)
    ade = compute_ade(preds1, tgt1, valid).item()
    fde = compute_fde(preds1, tgt1, valid).item()
    assert abs(ade - 5.0) < 1e-6 and abs(fde - 5.0) < 1e-6  # T==1 => equal

    # T==2: step errors 5 then 13 -> ADE=9, FDE=13 (distinct).
    preds2 = torch.tensor([[[3.0, 4.0], [5.0, 12.0]]])
    tgt2 = torch.zeros(1, 2, 2)
    valid2 = torch.ones(1, 2)
    assert abs(compute_ade(preds2, tgt2, valid2).item() - 9.0) < 1e-6
    assert abs(compute_fde(preds2, tgt2, valid2).item() - 13.0) < 1e-6


if __name__ == "__main__":
    test_accumulator_per_step_and_distinct_fde()
    test_ade_fde_functions_single_vs_multi_step()
    print("ok")
