#!/usr/bin/env python3
"""Exp 3.4: profile the unified SMCM-MCFNet pipeline vs a lazy sequential cascade.

Two systems, same detector backbone:
  - Unified:  detector + STTP head share the fused pyramid (one forward).
  - Cascade:  detector -> greedy-IoU tracker -> constant-velocity predictor
              (the STTP head is replaced by an external track+predict stage).

Reports params, FLOPs (thop, best-effort), and wall-clock latency/frame. The
cascade's trajectory accuracy is a zero-parameter constant-velocity extrapolation;
its ADE-over-DSEC reuses the eval accumulator and is a follow-up (see plan).

  # full profile on a GPU node:
  python -m tools.profile_sttp_pipeline --config <sttp_yaml> [--resume <ckpt>] --iters 100
  # model-free self-check (runs anywhere):
  python -m tools.profile_sttp_pipeline

# ponytail: naive greedy-IoU assoc + per-track constant velocity. Upgrade to
# Kalman/ByteTrack only if the thesis wants a stronger sequential baseline.
"""

from __future__ import annotations

import argparse
import time
from typing import Dict, List, Optional, Tuple

import torch


# --------------------------------------------------------------------------- #
# Lazy sequential cascade: greedy-IoU tracker + constant-velocity predictor.
# --------------------------------------------------------------------------- #
def _iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """IoU between [N,4] and [M,4] boxes in xyxy. Returns [N,M]."""
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union.clamp(min=1e-9)


class IoUTracker:
    """Minimal greedy-IoU tracker carrying each track's last two centers."""

    def __init__(self, iou_threshold: float = 0.3):
        self.iou_threshold = iou_threshold
        self._tracks: Dict[int, List[Tuple[float, float]]] = {}  # id -> [prev, cur] centers
        self._boxes: Dict[int, torch.Tensor] = {}                 # id -> last xyxy box
        self._next_id = 0

    @staticmethod
    def _center(box: torch.Tensor) -> Tuple[float, float]:
        return (float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2))

    def update(self, boxes_xyxy: torch.Tensor) -> Dict[int, int]:
        """Associate detections to tracks. Returns det_idx -> track_id."""
        assignment: Dict[int, int] = {}
        if boxes_xyxy.numel() == 0:
            return assignment
        track_ids = list(self._boxes.keys())
        if track_ids:
            prev = torch.stack([self._boxes[t] for t in track_ids])
            iou = _iou_xyxy(boxes_xyxy, prev)  # [N_det, N_track]
            taken = set()
            order = iou.max(dim=1).values.argsort(descending=True)
            for d in order.tolist():
                t = int(iou[d].argmax())
                if iou[d, t] < self.iou_threshold or t in taken:
                    continue
                tid = track_ids[t]
                taken.add(t)
                assignment[d] = tid
        for d in range(boxes_xyxy.shape[0]):
            box = boxes_xyxy[d]
            c = self._center(box)
            if d in assignment:
                tid = assignment[d]
                self._tracks[tid] = [self._tracks[tid][-1], c]
            else:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = [c, c]
                assignment[d] = tid
            self._boxes[tid] = box
        return assignment

    def predict(self, track_id: int, k: int) -> Tuple[float, float]:
        """Constant-velocity future center at horizon k (frames)."""
        prev, cur = self._tracks[track_id]
        vx, vy = cur[0] - prev[0], cur[1] - prev[1]
        return cur[0] + k * vx, cur[1] + k * vy


# --------------------------------------------------------------------------- #
# Model profiling (cluster only; needs the built model).
# --------------------------------------------------------------------------- #
def count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def profile_latency(forward_fn, iters: int, device: torch.device, warmup: int = 5) -> float:
    """Median ms/frame over `iters` forwards (CUDA-synced)."""
    for _ in range(warmup):
        forward_fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        forward_fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return times[len(times) // 2]


def try_flops(model: torch.nn.Module, inputs: tuple) -> Optional[float]:
    try:
        import thop  # noqa: WPS433 (optional dep, present on the cluster)
    except ImportError:
        return None
    macs, _ = thop.profile(model, inputs=inputs, verbose=False)
    return float(macs)


class _BackboneNeck(torch.nn.Module):
    """Runs only the shared feature extraction (backbone + fusion + neck), no heads.

    This is the redundant computation a non-shared learned forecaster must repeat,
    and the cost the unified model pays once. Mirrors DualStreamYOLOX.forward up to
    the neck (yolox.py:81-114).
    """

    def __init__(self, backbone: torch.nn.Module, neck: torch.nn.Module):
        super().__init__()
        self.backbone = backbone
        self.neck = neck

    def forward(self, rgb: torch.Tensor, event: torch.Tensor):
        bf = self.backbone(rgb, event)
        for k in list(bf.keys()):
            if str(k).startswith("_"):  # drop MCM/profile aux keys
                bf.pop(k, None)
        return self.neck([bf["dark3"], bf["dark4"], bf["dark5"]])


def profile_model(
    config_path: str,
    resume: Optional[str],
    iters: int,
    unified_ade: Optional[float] = None,
    unified_fde: Optional[float] = None,
    classical_ade: float = 11.36,
) -> None:
    from smcfnet.config.config import Config, load_config_from_yaml, update_config_from_dict
    from smcfnet.model.build import build_model

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def _load(sttp_enabled: bool):
        cfg = Config()
        update_config_from_dict(cfg, load_config_from_yaml(config_path))
        cfg.model.sttp.enabled = sttp_enabled
        cfg.model.pretrained = False  # cost-only: params/FLOP are architecture, not weights
        m = build_model(cfg).to(device).eval()
        if resume:
            ck = torch.load(resume, map_location="cpu", weights_only=False)
            sd = ck.get("model", ck)
            missing, unexpected = m.load_state_dict(sd, strict=False)
            print(f"  loaded {resume}: missing={len(missing)} unexpected={len(unexpected)}")
        return m

    unified = _load(sttp_enabled=True)
    detector = _load(sttp_enabled=False)  # detection stage shared by both cascades
    bneck = _BackboneNeck(detector.backbone, detector.neck).to(device).eval()

    h = int(getattr(unified, "_profile_h", 480))
    w = int(getattr(unified, "_profile_w", 640))
    rgb = torch.randn(1, 3, h, w, device=device)
    event = torch.randn(1, 5, h, w, device=device)
    flow = torch.randn(1, 2, h, w, device=device)

    head = getattr(unified, "sttp_head", None)
    head_params = count_params(head) if head is not None else 0
    bn_params = count_params(bneck)

    with torch.inference_mode():
        uni_lat = profile_latency(lambda: unified(rgb, event, flow=flow), iters, device)
        det_lat = profile_latency(lambda: detector(rgb, event), iters, device)
        bn_lat = profile_latency(lambda: bneck(rgb, event), iters, device)

    uni_flops = try_flops(unified, (rgb, event))
    det_flops = try_flops(detector, (rgb, event))
    bn_flops = try_flops(bneck, (rgb, event))

    # STTP-head marginal = unified - detector (avoids isolating the head's inputs).
    sttp_flops = (uni_flops - det_flops) if (uni_flops and det_flops) else None
    sttp_lat = uni_lat - det_lat

    # Three systems, same detection front end:
    #   unified  = detector + STTP head (one shared forward)
    #   classical= detector + IoU tracker + CV predictor (zero-param, ~0 FLOP)
    #   learned  = detector + a non-shared forecaster that RE-EXTRACTS features
    #              (a second backbone+neck pass) + a trajectory head
    uni_p, uni_f = count_params(unified), uni_flops
    cls_p, cls_f, cls_lat = count_params(detector), det_flops, det_lat
    lrn_p = count_params(detector) + bn_params + head_params
    lrn_f = (det_flops + bn_flops + sttp_flops) if (det_flops and bn_flops and sttp_flops) else None
    lrn_lat = det_lat + bn_lat + sttp_lat

    def _f(x):
        return f"{x:,.0f}" if x else "n/a"

    print("\n=== Exp 3.4: unified vs sequential cascades (RQ2) ===")
    print(f"device: {device}, input: 1x{h}x{w}, iters: {iters}")
    print("NOTE: flow precomputed (SMCM not in forward); SMCM is a shared detection")
    print("      front-end cost identical across all three systems (cancels in the delta).")
    print(f"\n{'system':<26}{'params':>14}{'FLOPs(MACs)':>16}{'latency_ms':>14}")
    print(f"{'unified (det+STTP)':<26}{uni_p:>14,}{_f(uni_f):>16}{uni_lat:>14.3f}")
    print(f"{'classical cascade':<26}{cls_p:>14,}{_f(cls_f):>16}{cls_lat:>14.3f}")
    print(f"{'learned cascade':<26}{lrn_p:>14,}{_f(lrn_f):>16}{lrn_lat:>14.3f}")

    print("\n-- marginal over the shared detector --")
    print(f"  unified   adds STTP head:          {head_params:,} params, {_f(sttp_flops)} FLOP")
    print(f"  classical adds IoU+CV:             ~0 params, ~0 FLOP (Python-loop only)")
    print(f"  learned   adds 2nd backbone+neck + head: {bn_params + head_params:,} params, "
          f"{_f((bn_flops + sttp_flops) if (bn_flops and sttp_flops) else None)} FLOP")
    if uni_f and lrn_f:
        print(f"  unified vs learned saving (backbone computed once): {_f(bn_flops)} FLOP "
              f"({100.0 * bn_flops / lrn_f:.1f}% of the learned cascade)")

    print("\n-- trajectory accuracy (ADE@k px; lower=better) --")
    print(f"  unified C   : ADE={unified_ade if unified_ade else 'TODO(from eval.log)'}"
          f"  FDE={unified_fde if unified_fde else 'TODO'}")
    print(f"  classical cv: ADE={classical_ade}  (cv_tracker, sttp_baseline_tracker)")
    print("  learned cascade: ADE n/a (cost-only; not trained — RQ2 is efficiency).")
    print("  note: multi-step unified ADE = trajectory average; old single-step ADE == new FDE.")

    # self-check: RQ2 direction must hold (learned heavier than unified heavier than classical)
    assert lrn_p > uni_p > cls_p, f"param order broke: learned {lrn_p} uni {uni_p} cls {cls_p}"
    if uni_f and lrn_f and cls_f:
        assert lrn_f > uni_f > cls_f, f"FLOP order broke: learned {lrn_f} uni {uni_f} cls {cls_f}"
    print("\nself-check OK: learned > unified > classical in params/FLOP (RQ2 direction)")


# --------------------------------------------------------------------------- #
# Self-check: CV predictor on a synthetic constant-velocity track -> ~0 ADE.
# --------------------------------------------------------------------------- #
def _self_check() -> None:
    tracker = IoUTracker(iou_threshold=0.3)
    # Object moving +5px/frame in x, +3 in y; 20x20 box. Feed 3 frames.
    centers = [(100, 100), (105, 103), (110, 106)]
    tid = None
    for cx, cy in centers:
        box = torch.tensor([[cx - 10, cy - 10, cx + 10, cy + 10]], dtype=torch.float32)
        assign = tracker.update(box)
        tid = assign[0]
    k = 5
    px, py = tracker.predict(tid, k)
    # True future center at +k from last: (110 + 5*5, 106 + 5*3) = (135, 121)
    ade = ((px - 135) ** 2 + (py - 121) ** 2) ** 0.5
    assert ade < 1e-6, f"CV predictor ADE should be ~0 on a CV track, got {ade}"

    # IoU sanity: identical box -> IoU 1; disjoint -> 0.
    a = torch.tensor([[0.0, 0, 10, 10]])
    assert abs(float(_iou_xyxy(a, a)) - 1.0) < 1e-6
    b = torch.tensor([[100.0, 100, 110, 110]])
    assert float(_iou_xyxy(a, b)) == 0.0
    print("ok: CV predictor + IoU tracker self-check passed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="STTP run YAML (omit for self-check only)")
    ap.add_argument("--resume", default=None, help="checkpoint to load (optional)")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--unified-ade", type=float, default=None, help="unified C ADE@k from eval.log")
    ap.add_argument("--unified-fde", type=float, default=None, help="unified C FDE@k from eval.log")
    ap.add_argument("--classical-ade", type=float, default=11.36, help="cv_tracker ADE@k")
    args = ap.parse_args()

    _self_check()
    if args.config:
        profile_model(args.config, args.resume, args.iters,
                      unified_ade=args.unified_ade, unified_fde=args.unified_fde,
                      classical_ade=args.classical_ade)


if __name__ == "__main__":
    main()
