#!/usr/bin/env python3
"""Classical detect -> track -> predict trajectory baseline (STTP Exp 3.3 floor).

The NON-learned reference the learned STTP head is compared against. Stage 2 of a
two-stage pipeline:

  stage 1 (tools/sttp_baseline_dump.py, GPU+data): run the frozen detector over the
           val set, dump per-frame {post-NMS dets, GT, future-GT per k}.
  stage 2 (this file, pure CPU, no GPU/data): replay the dump per sequence in frame
           order, run a frame-gap constant-velocity tracker, and score with the SAME
           TrajectoryMetricsAccumulator the learned head uses.

Because stage 2 consumes the exact post-NMS detections + GT the learned-head eval
produced and feeds them to the same accumulator, the baseline and the learned head are
scored on byte-identical inputs with identical matching. The tracker is the only new
moving part, and it is unit-testable here without a GPU.

  # self-check (runs anywhere, no dump needed):
  python tools/sttp_baseline_tracker.py
  # score a dump:
  python tools/sttp_baseline_tracker.py --dump outputs/sttp_results/baseline_dump/val.pt \
      --k 3 --k 5 --k 10 --csv outputs/sttp_results/phase3.csv

# ponytail: greedy-IoU association + per-track constant velocity only. Upgrade to a
# Kalman filter / SORT / ByteTrack only if the thesis wants a stronger sequential
# baseline; the metric wiring below is independent of the tracker choice.
"""

from __future__ import annotations

import argparse
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import torch

from tools.profile_sttp_pipeline import _iou_xyxy

KS_DEFAULT: Tuple[int, ...] = (3, 5, 10)
VELOCITY_WINDOW = 5  # full_cv fits velocity over up to this many past observations


def _xyxy_to_cxcywh(b: torch.Tensor) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0, (x2 - x1), (y2 - y1)


def _lsq_slope(ts: List[float], vs: List[float]) -> float:
    """Least-squares slope of v vs t. Zero when the times are degenerate."""
    n = len(ts)
    if n < 2:
        return 0.0
    mt = sum(ts) / n
    mv = sum(vs) / n
    num = sum((t - mt) * (v - mv) for t, v in zip(ts, vs))
    den = sum((t - mt) ** 2 for t in ts)
    return num / den if den > 0 else 0.0


class FrameGapTracker:
    """Greedy-IoU association + constant-velocity prediction, frame-gap aware.

    P0-1 fix over the profiling-only ``IoUTracker``: velocity is per *frame*, not per
    *observation*. After missed detections two consecutive observations span more than
    one frame, so velocity = displacement / frame-gap, never displacement assumed over
    one frame. Tracks unobserved for longer than ``max_age`` frames are expired so a
    stale track cannot be revived and coast with an ancient velocity.
    """

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30,
                 window: int = VELOCITY_WINDOW):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.window = window
        # id -> deque of (frame_index, cx, cy), newest last, bounded to `window`
        self._hist: Dict[int, Deque[Tuple[int, float, float]]] = {}
        self._wh: Dict[int, Tuple[float, float]] = {}      # id -> last (w, h)
        self._box: Dict[int, torch.Tensor] = {}            # id -> last xyxy box
        self._last_frame: Dict[int, int] = {}              # id -> last seen frame
        self._next_id = 0

    def _expire(self, frame_index: int) -> None:
        dead = [tid for tid, f in self._last_frame.items()
                if frame_index - f > self.max_age]
        for tid in dead:
            self._hist.pop(tid, None)
            self._wh.pop(tid, None)
            self._box.pop(tid, None)
            self._last_frame.pop(tid, None)

    def update(self, boxes_xyxy: torch.Tensor, frame_index: int) -> Dict[int, int]:
        """Associate detections to tracks. Returns det_idx -> track_id (one per det)."""
        self._expire(frame_index)
        assignment: Dict[int, int] = {}
        if boxes_xyxy.numel() == 0:
            return assignment

        track_ids = list(self._box.keys())
        if track_ids:
            prev = torch.stack([self._box[t] for t in track_ids])
            iou = _iou_xyxy(boxes_xyxy.float(), prev.float())  # [N_det, N_track]
            taken = set()
            order = iou.max(dim=1).values.argsort(descending=True)
            for d in order.tolist():
                t = int(iou[d].argmax())
                if iou[d, t] < self.iou_threshold or t in taken:
                    continue
                taken.add(t)
                assignment[d] = track_ids[t]

        for d in range(boxes_xyxy.shape[0]):
            box = boxes_xyxy[d]
            cx, cy, w, h = _xyxy_to_cxcywh(box)
            if d in assignment:
                tid = assignment[d]
            else:
                tid = self._next_id
                self._next_id += 1
                self._hist[tid] = deque(maxlen=self.window)
                assignment[d] = tid
            self._hist[tid].append((frame_index, cx, cy))
            self._wh[tid] = (w, h)
            self._box[tid] = box
            self._last_frame[tid] = frame_index
        return assignment

    def velocity(self, track_id: int, two_frame: bool) -> Tuple[float, float]:
        hist = self._hist[track_id]
        if len(hist) < 2:
            return 0.0, 0.0
        if two_frame or len(hist) == 2:
            (f0, x0, y0) = hist[-2]
            (f1, x1, y1) = hist[-1]
            df = max(f1 - f0, 1)
            return (x1 - x0) / df, (y1 - y0) / df
        ts = [float(f) for (f, _, _) in hist]
        return (_lsq_slope(ts, [x for (_, x, _) in hist]),
                _lsq_slope(ts, [y for (_, _, y) in hist]))

    def predict(self, track_id: int, k: int, two_frame: bool = False) -> Tuple[float, float]:
        """Constant-velocity future center at horizon k (frames) from the last observation."""
        _, cx, cy = self._hist[track_id][-1]
        vx, vy = self.velocity(track_id, two_frame)
        return cx + k * vx, cy + k * vy

    def last_wh(self, track_id: int) -> Tuple[float, float]:
        return self._wh[track_id]

    def age(self, track_id: int) -> int:
        return len(self._hist[track_id])


# --------------------------------------------------------------------------- #
# Kalman variant: constant-velocity Kalman filter (SORT-style) that COASTS
# through missed detections, so tracks survive gaps instead of cold-starting.
# A strictly stronger classical baseline than greedy-IoU + finite-difference CV.
# --------------------------------------------------------------------------- #
class _CVKalman:
    """Constant-velocity Kalman filter on the box center. State = [cx, cy, vx, vy]."""

    # ponytail: SORT-ish defaults (R ~ 3px detection-center std, large initial
    # velocity covariance so the first updates trust measurements). Tune Q/R only if
    # the Kalman arm needs to be a tighter floor.
    def __init__(self, cx: float, cy: float):
        import numpy as np
        self.x = np.array([cx, cy, 0.0, 0.0], dtype=np.float64)
        self.P = np.diag([10.0, 10.0, 1e4, 1e4]).astype(np.float64)

    def predict(self) -> None:
        import numpy as np
        F = _CVKalman._F
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + _CVKalman._Q

    def update(self, cx: float, cy: float) -> None:
        import numpy as np
        H, R = _CVKalman._H, _CVKalman._R
        z = np.array([cx, cy], dtype=np.float64)
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P


def _init_kalman_constants() -> None:
    import numpy as np
    _CVKalman._F = np.array([[1, 0, 1, 0], [0, 1, 0, 1],
                             [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
    _CVKalman._H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
    _CVKalman._Q = (np.diag([1.0, 1.0, 10.0, 10.0]) * 0.01).astype(np.float64)
    _CVKalman._R = np.diag([10.0, 10.0]).astype(np.float64)


class KalmanTracker:
    """SORT-style tracker: predict all tracks each frame (coast), greedy-IoU
    associate against the PREDICTED boxes, KF-correct matches. Tracks coast through
    missed-detection frames, so reappearances re-associate instead of cold-starting.
    """

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30):
        _init_kalman_constants()
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self._kf: Dict[int, _CVKalman] = {}
        self._wh: Dict[int, Tuple[float, float]] = {}
        self._n_updates: Dict[int, int] = {}
        self._last_seen: Dict[int, int] = {}       # last frame with a measurement
        self._last_predicted: Dict[int, int] = {}  # last frame the KF was advanced to
        self._next_id = 0

    def _pred_box(self, tid: int) -> torch.Tensor:
        cx, cy = float(self._kf[tid].x[0]), float(self._kf[tid].x[1])
        w, h = self._wh[tid]
        return torch.tensor([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    def update(self, boxes_xyxy: torch.Tensor, frame_index: int) -> Dict[int, int]:
        # 1) advance every live track to this frame (coast through gaps).
        for tid, kf in self._kf.items():
            steps = frame_index - self._last_predicted[tid]
            for _ in range(max(steps, 0)):
                kf.predict()
            self._last_predicted[tid] = frame_index

        assignment: Dict[int, int] = {}
        if boxes_xyxy.numel() == 0:
            self._expire(frame_index)
            return assignment

        # 2) greedy-IoU associate detections to PREDICTED track boxes.
        track_ids = list(self._kf.keys())
        if track_ids:
            pred = torch.stack([self._pred_box(t) for t in track_ids])
            iou = _iou_xyxy(boxes_xyxy.float(), pred.float())
            taken = set()
            for d in iou.max(dim=1).values.argsort(descending=True).tolist():
                t = int(iou[d].argmax())
                if iou[d, t] < self.iou_threshold or t in taken:
                    continue
                taken.add(t)
                assignment[d] = track_ids[t]

        # 3) KF-correct matched tracks; spawn new tracks for unmatched detections.
        for d in range(boxes_xyxy.shape[0]):
            cx, cy, w, h = _xyxy_to_cxcywh(boxes_xyxy[d])
            if d in assignment:
                tid = assignment[d]
                self._kf[tid].update(cx, cy)
                self._n_updates[tid] += 1
            else:
                tid = self._next_id
                self._next_id += 1
                self._kf[tid] = _CVKalman(cx, cy)
                self._n_updates[tid] = 1
                self._last_predicted[tid] = frame_index
                assignment[d] = tid
            self._wh[tid] = (w, h)
            self._last_seen[tid] = frame_index
        self._expire(frame_index)
        return assignment

    def _expire(self, frame_index: int) -> None:
        dead = [t for t, f in self._last_seen.items() if frame_index - f > self.max_age]
        for t in dead:
            for d in (self._kf, self._wh, self._n_updates,
                      self._last_seen, self._last_predicted):
                d.pop(t, None)

    def predict(self, track_id: int, k: int) -> Tuple[float, float]:
        x = self._kf[track_id].x
        return float(x[0] + k * x[2]), float(x[1] + k * x[3])

    def age(self, track_id: int) -> int:
        return self._n_updates[track_id]


def score_frames(
    frames: List[dict],
    k: int,
    variant: str = "full",
    iou_assoc: float = 0.3,
    iou_match: float = 0.5,
) -> Dict[str, float]:
    """Replay one set of dumped frames through tracker + the eval accumulator.

    Each frame dict carries:
      "seq": str, "frame_index": int,
      "dets_xyxy7": [N,7] post-NMS detections (xyxy,obj,cls_conf,cls) or None,
      "gt_targets": [M,5] (class_id, cx, cy, w, h),
      "future_targets": {k: {"boxes":[M,4] cxcywh, "validity":[M] bool}}.

    ``variant`` is "full" (windowed velocity over track history) or "twoframe"
    (last-two-observation velocity, the closest same-information point to the
    stateless flow-conditioned learned head).
    """
    from smcfnet.utils.validation import TrajectoryMetricsAccumulator

    acc = TrajectoryMetricsAccumulator(iou_threshold=iou_match)
    if variant == "kalman":
        tracker = KalmanTracker(iou_threshold=iou_assoc)
    else:
        tracker = FrameGapTracker(iou_threshold=iou_assoc)
    two = variant == "twoframe"
    n_pred = 0
    n_cold = 0

    for fr in sorted(frames, key=lambda f: f["frame_index"]):
        dets = fr["dets_xyxy7"]
        n = 0 if dets is None else int(dets.shape[0])
        boxes = dets[:, :4] if n else torch.zeros((0, 4))
        assign = tracker.update(boxes, int(fr["frame_index"]))

        fut_decoded = torch.zeros((n, 4), dtype=torch.float32)
        det_pre = torch.zeros((1, n, 4), dtype=torch.float32)
        ages = torch.zeros((n,), dtype=torch.long)  # tracker age per detection (>=1)
        for d in range(n):
            cx, cy, w, h = _xyxy_to_cxcywh(dets[d, :4])
            det_pre[0, d] = torch.tensor([cx, cy, w, h])
            tid = assign[d]
            if variant == "kalman":
                px, py = tracker.predict(tid, k)
            else:
                px, py = tracker.predict(tid, k, two_frame=two)
            fut_decoded[d] = torch.tensor([px, py, w, h])  # keep last observed w,h
            ages[d] = tracker.age(tid)
            n_pred += 1
            if tracker.age(tid) < 2:
                n_cold += 1

        # P0-3: predictions must be one-per-detection, in detection order, so the
        # identity-anchor lookup in the accumulator attaches the right future box.
        assert fut_decoded.shape[0] == n, f"fut_decoded {fut_decoded.shape[0]} != N {n}"
        if n == 0:
            continue

        ft = fr["future_targets"][k]
        # The tracker predicts a single box at horizon k; wrap it as a length-1
        # trajectory for the multi-step accumulator: boxes [1,M,1,4], validity [1,M,1],
        # fut_decoded [1,n,1,4]. FDE then coincides with ADE (single step).
        future = {
            "boxes": ft["boxes"].unsqueeze(0).unsqueeze(2),
            "validity": ft["validity"].unsqueeze(0).unsqueeze(2),
        }
        acc.update(
            det_pre,
            fut_decoded.unsqueeze(0).unsqueeze(2),
            [dets],
            [fr["gt_targets"]],
            future,
            anchor_indices_per_image=[torch.arange(n)],
            pred_tag_per_image=[ages],  # tag each scored pair with its track age
        )

    metrics = acc.compute()
    metrics["_n_pred"] = float(n_pred)
    metrics["_cold_start_frac"] = float(n_cold) / max(n_pred, 1)

    # Stratify the SCORED pairs (matched, future-valid) by tracker age, single pass.
    # warm = age>=2 (real constant-velocity), cold = age==1 (zero-velocity guess).
    import numpy as _np
    errs = _np.array(acc._l2_errors, dtype=_np.float64)
    tags = _np.array(acc._tags, dtype=_np.int64)
    if errs.size:
        warm = errs[tags >= 2]
        cold = errs[tags == 1]
        metrics["ADE_warm"] = float(warm.mean()) if warm.size else float("nan")
        metrics["ADE_cold"] = float(cold.mean()) if cold.size else float("nan")
        metrics["_n_warm"] = float(warm.size)
        metrics["_n_cold"] = float(cold.size)
        metrics["_frac_warm"] = float(warm.size) / float(errs.size)
    return metrics


def _format_per_class(metrics: Dict[str, float]) -> str:
    parts = []
    for key in sorted(metrics):
        if key.startswith("ADE_cls"):
            parts.append(f"{key[len('ADE_cls'):]}={metrics[key]:.3f}")
    return ",".join(parts)


def _append_csv_rows(csv_path: str, rows: List[dict]) -> None:
    import csv as _csv
    import os
    from tools.sttp_eval_aggregate import CSV_FIELDS

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def _replay_dump(dump_path: str, ks: Tuple[int, ...], csv_path: Optional[str],
                 label: str, variants: Tuple[str, ...],
                 wandb_args: Optional[dict] = None) -> None:
    import datetime as _dt

    frames = torch.load(dump_path, weights_only=False)
    dumped_ks = set()
    for fr in frames:
        dumped_ks.update(int(x) for x in fr.get("future_targets", {}).keys())
    ts = _dt.datetime.now().isoformat(timespec="seconds")
    results: List[dict] = []  # numeric per (variant, k) for both CSV and wandb

    design_of = {"full": "cv_tracker", "twoframe": "cv_tracker_2frame",
                 "kalman": "cv_tracker_kalman"}
    for variant in variants:
        design = design_of.get(variant, f"cv_tracker_{variant}")
        for k in ks:
            if k not in dumped_ks:
                print(f"  [skip] k={k} not in dump (has {sorted(dumped_ks)})")
                continue
            m = score_frames(frames, k, variant=variant)
            ade = m.get("ADE", float("nan"))
            ade_warm = m.get("ADE_warm", float("nan"))
            ade_cold = m.get("ADE_cold", float("nan"))
            print(f"  {design} k={k}: ADE={ade:.3f}px (warm={ade_warm:.3f} "
                  f"cold={ade_cold:.3f})  FDE={ade:.3f}px  per_step={ade / k:.3f}  "
                  f"n_pred={int(m.get('_n_pred', 0))}  "
                  f"frac_warm={m.get('_frac_warm', 0):.3f}")
            results.append({
                "design": design, "variant": variant, "k": int(k),
                "ade": float(ade), "ade_warm": float(ade_warm), "ade_cold": float(ade_cold),
                "per_step": float(ade / k),
                "cold_start_frac": float(m.get("_cold_start_frac", 0.0)),
                "frac_warm": float(m.get("_frac_warm", 0.0)),
                "n_pred": int(m.get("_n_pred", 0)),
                "per_class": {int(key[len("ADE_cls"):]): float(v)
                              for key, v in m.items() if key.startswith("ADE_cls")},
                "per_class_str": _format_per_class(m),
            })

    if csv_path and results:
        rows = [{
            "timestamp": ts,
            "label": f"{label}_{r['design']}",
            "design": r["design"],
            "flow_source": "none",
            "k": r["k"],
            "config_path": dump_path,
            "ade5_px": f"{r['ade']:.4f}",
            "fde_px": f"{r['ade']:.4f}",
            "ade_per_step": f"{r['per_step']:.4f}",
            "per_class_ade": r["per_class_str"],
            "map50": "",
            "map_full": "",
        } for r in results]
        _append_csv_rows(csv_path, rows)
        print(f"  appended {len(rows)} rows to {csv_path}")

    if wandb_args and wandb_args.get("enabled") and results:
        _wandb_log(results, label, dump_path, wandb_args)


def _wandb_log(results: List[dict], label: str, dump_path: str, wa: dict) -> None:
    """Log baseline results to a dedicated wandb project. Online with offline fallback."""
    try:
        import wandb
    except ImportError:
        print("  [wandb] not installed; skipping (pip install wandb)")
        return

    init_kwargs = dict(
        project=wa["project"], entity=wa.get("entity"),
        name=wa.get("run_name") or label, job_type="baseline_eval",
        tags=["sttp", "baseline", "cv_tracker"],
        config={"dump": dump_path, "label": label,
                "ks": sorted({r["k"] for r in results}),
                "designs": sorted({r["design"] for r in results})},
    )
    try:
        run = wandb.init(**init_kwargs)
    except Exception as e:  # offline fallback mirrors smcfnet.utils.wandb
        print(f"  [wandb] online init failed ({e}); retrying offline")
        try:
            run = wandb.init(mode="offline", **init_kwargs)
        except Exception as e2:
            print(f"  [wandb] offline init also failed ({e2}); skipping")
            return

    # Logging is best-effort: the CSV is already written, so a wandb API hiccup
    # must not fail stage 2. Always finish the run.
    try:
        table = wandb.Table(columns=["design", "k", "ADE", "ADE_warm", "ADE_cold",
                                     "FDE", "ade_per_step", "frac_warm", "n_pred"])
        for r in results:
            table.add_data(r["design"], r["k"], r["ade"], r["ade_warm"], r["ade_cold"],
                           r["ade"], r["per_step"], r["frac_warm"], r["n_pred"])
            prefix = f"{r['design']}/k{r['k']}"
            run.summary[f"{prefix}/ADE"] = r["ade"]
            run.summary[f"{prefix}/ADE_warm"] = r["ade_warm"]
            run.summary[f"{prefix}/ADE_cold"] = r["ade_cold"]
            run.summary[f"{prefix}/FDE"] = r["ade"]
            run.summary[f"{prefix}/ade_per_step"] = r["per_step"]
            run.summary[f"{prefix}/frac_warm"] = r["frac_warm"]
            run.summary[f"{prefix}/n_pred"] = r["n_pred"]
            for cls_id, val in r["per_class"].items():
                run.summary[f"{prefix}/ADE_cls{cls_id}"] = val
        run.log({"results": table})
        # ADE-vs-horizon line per design for an at-a-glance comparison.
        for design in sorted({r["design"] for r in results}):
            pts = sorted((r["k"], r["ade"]) for r in results if r["design"] == design)
            line = wandb.Table(data=[[k, a] for k, a in pts], columns=["k", "ADE"])
            run.log({f"ADE_vs_k/{design}": wandb.plot.line(line, "k", "ADE",
                                                           title=f"{design} ADE vs horizon")})
        print(f"  [wandb] logged {len(results)} rows to "
              f"{wa.get('entity') or '?'}/{wa['project']}")
    except Exception as e:
        print(f"  [wandb] logging failed ({e}); CSV is unaffected")
    finally:
        run.finish()


# --------------------------------------------------------------------------- #
# Self-check: frame-gap velocity + identity-anchor scoring on synthetic frames.
# --------------------------------------------------------------------------- #
def _self_check() -> None:
    # Frame-gap: a slow object missed for frames 1,2 but boxes still overlap, so the
    # tracker re-associates across the gap. 40x40 box, +2px/frame, observed at 0 and 3.
    tr = FrameGapTracker(iou_threshold=0.3)
    for f, c in [(0, 100.0), (3, 106.0)]:
        box = torch.tensor([[c - 20, c - 20, c + 20, c + 20]])
        tr.update(box, f)
    tid = 0
    px, py = tr.predict(tid, k=5)
    # velocity = (106-100)/3 = 2 per frame; predict from last (106): 106 + 5*2 = 116.
    assert abs(px - 116.0) < 1e-4 and abs(py - 116.0) < 1e-4, f"frame-gap predict {px},{py}"
    # A naive gap=1 velocity would give 106 + 5*6 = 136; guard that we are not that.
    assert abs(px - 136.0) > 1.0

    # Metric wiring: stationary tracker prediction, GT future displaced by (dx,dy).
    # ADE must equal hypot(dx,dy) exactly (single matched object, identity anchor).
    dx, dy = 7.0, 24.0
    cx, cy, w, h = 200.0, 150.0, 30.0, 30.0
    frames = []
    for fi in range(3):
        det = torch.tensor([[cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, 0.9, 0.9, 0.0]])
        gt = torch.tensor([[0.0, cx, cy, w, h]])
        ft = {3: {"boxes": torch.tensor([[cx + dx, cy + dy, w, h]]),
                  "validity": torch.tensor([True])}}
        frames.append({"seq": "s", "frame_index": fi, "dets_xyxy7": det,
                       "gt_targets": gt, "future_targets": ft})
    m = score_frames(frames, k=3, variant="full")
    expected = (dx ** 2 + dy ** 2) ** 0.5
    assert abs(m["ADE"] - expected) < 1e-3, f"ADE {m['ADE']} != {expected}"

    # Stationary object whose GT future is also stationary -> ADE ~ 0.
    frames0 = []
    for fi in range(3):
        det = torch.tensor([[cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, 0.9, 0.9, 0.0]])
        gt = torch.tensor([[0.0, cx, cy, w, h]])
        ft = {3: {"boxes": torch.tensor([[cx, cy, w, h]]), "validity": torch.tensor([True])}}
        frames0.append({"seq": "s", "frame_index": fi, "dets_xyxy7": det,
                        "gt_targets": gt, "future_targets": ft})
    m0 = score_frames(frames0, k=3, variant="full")
    assert m0["ADE"] < 1e-4, f"stationary ADE should be ~0, got {m0['ADE']}"
    print("ok: frame-gap velocity + identity-anchor scoring self-check passed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", default=None, help="path to a stage-1 dump (.pt); omit for self-check")
    ap.add_argument("--k", type=int, action="append", default=None, help="horizon(s); repeatable")
    ap.add_argument("--csv", default=None, help="append result rows to this CSV")
    ap.add_argument("--label", default="cv_tracker_v5")
    ap.add_argument("--variants", default="full,twoframe,kalman",
                    help="comma list of {full,twoframe,kalman}")
    ap.add_argument("--wandb", action="store_true", help="log results to wandb")
    ap.add_argument("--wandb-project", default="smcfnet_sttp_baseline")
    ap.add_argument("--wandb-entity", default="smcm-mcfnet")
    ap.add_argument("--wandb-run-name", default=None)
    args = ap.parse_args()

    _self_check()
    if args.dump:
        ks = tuple(args.k) if args.k else KS_DEFAULT
        variants = tuple(v.strip() for v in args.variants.split(",") if v.strip())
        wandb_args = {
            "enabled": args.wandb,
            "project": args.wandb_project,
            "entity": args.wandb_entity,
            "run_name": args.wandb_run_name,
        }
        _replay_dump(args.dump, ks, args.csv, args.label, variants, wandb_args)


if __name__ == "__main__":
    main()
