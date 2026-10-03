# src/smcfnet/tools/sttp_viz/core.py
"""Core: load a run, iterate a sequence, run the frozen forward pass, yield FrameBundle.

Reuses the project's input pipeline (events_to_voxel_grid_torch, the [0,255] RGB
convention), build_model, and STTPLoss._decode_absolute so the visualized
predictions are numerically identical to training/eval.

DIVERGENCE FROM PLAN (deliberate, follows the real code):
  The plan assumed ``model(rgb, voxel)`` returns post-NMS detections shaped [n,7],
  and that ``_decode_future`` would pair future boxes to detections by a
  nearest-anchor-by-center heuristic. In the real codebase the model's eval forward
  (``model/yolox.py`` else-branch) returns the *raw decoded* head outputs
  ``[B, n_anchors, 5+ncls]`` (``decode_in_inference=True``), NOT post-NMS boxes. NMS is
  applied separately by ``utils.validation.postprocess_predictions``, which can also
  return the exact source-anchor index for each surviving detection
  (``return_anchor_indices=True``). The canonical eval (``TrajectoryMetricsAccumulator``)
  uses those exact indices, so we mirror that here instead of the nearest-anchor
  heuristic: this makes the visualized future boxes numerically identical to the
  ADE the trainer reports.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np
import torch

from smcfnet.config.config import Config, load_config_from_yaml, update_config_from_dict
from smcfnet.model.build import build_model
from smcfnet.model.sttp.sttp_loss import STTPLoss
from smcfnet.dataset.dsec.dsec_lightweight_mapper import DSECLightweightMapper
from smcfnet.dataset.dsec.dsec_sequence_index import build_dsec_sequence_index_datasets
from smcfnet.dataset.flow_provider import PrecomputedFlowProvider
from smcfnet.dataset.stem_feature_provider import StemFeatureProvider
from smcfnet.utils.event_utils import events_to_voxel_grid_torch
from smcfnet.utils.validation import (
    postprocess_predictions,
    decode_sttp_predictions,
    bbox_iou,
    _targets_cxcywh_to_xyxy_torch,
)
from .bundle import FrameBundle
from .decode import pair_ade


class SttpVizCore:
    def __init__(self, config_path: str, checkpoint: str, device: str = "cuda",
                 conf: Optional[float] = None):
        self.cfg = Config()
        update_config_from_dict(self.cfg, load_config_from_yaml(config_path))
        self.cfg.model.sttp.enabled = True
        self.cfg.model.mcm.enabled = False
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        # Viz-facing confidence threshold. cfg.eval.conf_threshold is the eval
        # default (0.001, mAP-oriented) and floods panels with near-zero boxes;
        # the trainer's viz path filters at 0.3, mirror that here.
        self.conf = 0.3 if conf is None else float(conf)
        # Input providers, mirroring the trainer's injection paths so flow-injecting
        # (Design C) and recurrent (ConvGRU) heads see the same inputs as eval.
        flow_dir = getattr(self.cfg.data, "precomputed_flow_dir", None)
        self._flow_provider = PrecomputedFlowProvider(flow_dir) if flow_dir else None
        self.model = build_model(self.cfg)
        ckpt = torch.load(checkpoint, map_location="cpu")
        state = ckpt.get("model", ckpt.get("state_dict", ckpt))
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing:
            import logging
            logging.getLogger(__name__).warning("viz load missing keys: %d", len(missing))
        self.model.eval().to(self.device)
        stem_dir = getattr(self.cfg.data, "stem_feature_dir", None)
        self._stem_provider = (
            StemFeatureProvider(stem_dir, context_window=int(self.cfg.model.sttp.context_window))
            if stem_dir else None
        )
        self.k = int(self.cfg.model.sttp.future_steps)
        self.target_mode = str(getattr(self.cfg.model.sttp, "target_mode", "absolute"))
        self._loss = STTPLoss(target_mode=self.target_mode,
                              image_width=self.cfg.data.image_width,
                              image_height=self.cfg.data.image_height)

    def _frame_inputs(self, sample):
        h, w = int(self.cfg.data.image_height), int(self.cfg.data.image_width)
        bins = int(self.cfg.data.event_bins)
        ev = sample["events"]
        ev = ev if isinstance(ev, torch.Tensor) else torch.as_tensor(ev)
        voxel = events_to_voxel_grid_torch(ev.to(self.device), num_bins=bins, width=w, height=h,
                                           source_width=w, source_height=h).unsqueeze(0)
        img = sample["image"]
        img = img if isinstance(img, torch.Tensor) else torch.as_tensor(img)
        img = img.to(self.device).float()
        if img.dim() == 3:
            img = img.unsqueeze(0)
        if img.shape[-2:] != (h, w):
            img = torch.nn.functional.interpolate(img, size=(h, w), mode="bilinear", align_corners=False)
        return img, voxel

    def iterate_sequence(self, seq_name: Optional[str], max_frames: int = 0,
                         frame_indices: Optional[Dict[str, List[int]]] = None) -> Iterator[FrameBundle]:
        if frame_indices:
            sequences = sorted(frame_indices)
        elif seq_name:
            sequences = [seq_name]
        else:
            sequences = None
        per_seq = build_dsec_sequence_index_datasets(
            dsec_yaml_path=Path(str(self.cfg.data.dataset_config)),
            split=str(getattr(self.cfg.eval, "split", "train") or "train"),
            dt_ms=float(self.cfg.data.event_window_ms),
            sequences=sequences,
            use_rectified_events=False, use_rectified_tracks=False,
        )
        if not per_seq:
            return
        # build_future_targets must use the run's k / target_mode so the future GT
        # the mapper produces matches the head the checkpoint was trained with.
        mapper = DSECLightweightMapper(
            per_seq, load_events=True, load_images=True, load_tracks=True,
            build_targets=True, build_future_targets=True,
            future_steps=self.k, future_target_mode=self.target_mode,
            image_height=int(self.cfg.data.image_height),
            image_width=int(self.cfg.data.image_width),
            prefer_distorted=bool(getattr(self.cfg.data, "mapper_prefer_distorted", True)),
        )
        # seq_name=None => iterate every val sequence (max_frames is per-sequence).
        # frame_indices => exactly those frames (mapper is random-access).
        for seq_ds in per_seq:
            if frame_indices:
                fis = [fi for fi in frame_indices.get(seq_ds.sequence, []) if 0 <= fi < len(seq_ds)]
            else:
                n = len(seq_ds) if max_frames <= 0 else min(max_frames, len(seq_ds))
                fis = range(n)
            for fi in fis:
                sample = mapper(seq_ds[fi])
                yield self.forward_frame(seq_ds.sequence, fi, sample)

    def forward_frame(self, seq_name: str, fi: int, sample) -> FrameBundle:
        """Single-frame frozen forward with the same input injection as eval:
        precomputed flow (Design C) and stem window (recurrent heads) when the
        run config declares them."""
        rgb, voxel = self._frame_inputs(sample)
        flow = None
        if self._flow_provider is not None:
            f = self._flow_provider.get(seq_name, fi)  # None at frame 0
            if f is not None:
                flow = f.unsqueeze(0).to(self.device)
        stem_window = None
        if self._stem_provider is not None:
            stem_window = [t.unsqueeze(0).to(self.device)
                           for t in self._stem_provider.get_window(seq_name, fi)]
        with torch.inference_mode():
            out = self.model(rgb, voxel, flow=flow, stem_window=stem_window)
        detections, fut_preds = out if isinstance(out, tuple) else (out, None)
        return self._build_bundle(seq_name, fi, rgb, detections, fut_preds, sample)

    def close(self) -> None:
        if self._flow_provider is not None:
            self._flow_provider.close()
        if self._stem_provider is not None:
            self._stem_provider.close()

    def _build_bundle(self, seq_name, fi, rgb, raw_outputs, fut_preds, sample) -> FrameBundle:
        # raw_outputs: [B, n_anchors, 5+ncls] decoded head outputs (pre-NMS, cxcywh).
        # Run NMS exactly as eval does, recovering the source anchor index per detection.
        num_classes = int(self.cfg.model.num_classes)
        preds_per_image, anchor_idx_per_image = postprocess_predictions(
            raw_outputs.clone(),
            num_classes=num_classes,
            conf_thre=self.conf,
            nms_thre=float(self.cfg.eval.nms_threshold),
            return_anchor_indices=True,
        )
        dets_t = preds_per_image[0]              # [N,7] xyxy,obj,cls_conf,cls  or None
        anchor_idx = anchor_idx_per_image[0]      # [N] LongTensor or None
        det = _detections_to_cxcywh(preds_per_image)

        # Decode future boxes at every anchor (identical math to eval), then index by
        # each surviving detection's source anchor.
        pred_future_steps = self._decode_future(fut_preds, dets_t, anchor_idx, raw_outputs)
        pred_future = pred_future_steps[:, -1, :]  # [N,4] endpoint, works for N==0

        # Match each detection to a GT track (IoU, greedy) and pull its future boxes.
        (gt_future, gt_current, valid,
         gt_future_steps, valid_steps) = self._gt_future_from_sample(sample, dets_t, det)
        ade = pair_ade(pred_future, gt_future, valid)
        rgb_np = rgb[0].detach().cpu().byte().permute(1, 2, 0).numpy()
        return FrameBundle(
            seq_name=seq_name, frame_index=fi, rgb=rgb_np,
            detections=det, pred_future=pred_future, gt_future=gt_future,
            gt_current=gt_current, valid=valid, per_obj_ade=ade,
            pred_future_steps=pred_future_steps, gt_future_steps=gt_future_steps,
            valid_steps=valid_steps,
            meta={"k": self.k, "run_name": Path(str(self.cfg.log.experiment_name)).name,
                  "frame_ade": float(np.nanmean(ade)) if np.any(~np.isnan(ade)) else float("nan")},
        )

    def _decode_future(self, fut_preds, dets_t, anchor_idx, raw_outputs) -> np.ndarray:
        """Decode the head's future predictions and index by each detection's source
        anchor (the exact pairing the trainer's TrajectoryMetricsAccumulator uses via
        postprocess_predictions(return_anchor_indices=True)). Returns [N,k,4] cxcywh px
        (full trajectory t+1..t+k; callers slice [:, -1, :] for the endpoint).

        In "delta" mode decode_sttp_predictions needs the per-detection current box;
        we supply the pre-NMS decoded detection box at the matched anchor (== the same
        current_boxes the trainer passes, which is det_pre_nms_cxcywh).
        """
        n_det = 0 if dets_t is None else int(dets_t.shape[0])
        if n_det == 0 or fut_preds is None or anchor_idx is None:
            return np.zeros((n_det, self.k, 4), dtype=np.float32)
        head = self.model.head
        # det_pre_nms_cxcywh for the batch (raw_outputs already cxcywh in [:, :, :4]).
        det_pre_nms = raw_outputs[:, :, :4]
        fut_decoded = decode_sttp_predictions(
            fut_preds,
            head.hw,
            head.strides,
            target_mode=self.target_mode,
            current_boxes=det_pre_nms,
            image_width=int(self.cfg.data.image_width),
            image_height=int(self.cfg.data.image_height),
        )[0]  # [n_anchors, k, 4]
        idx = anchor_idx.to(fut_decoded.device).long()
        out = fut_decoded.index_select(0, idx)  # [N,k,4] full trajectory
        return out.detach().cpu().numpy().astype(np.float32)

    def _anchor_grid(self):
        """Replicate YOLOXHead.decode_outputs' grid/stride generation so external code
        gets the identical anchor layout (scale-major, row-major) that decode_future_boxes
        flattens fut_preds into. Returns (expanded_strides [1,A], grids [1,A,2]).
        Requires the head to have run a forward pass first (populates head.hw).
        """
        head = self.model.head
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(head.hw, head.strides):
            yv, xv = torch.meshgrid(
                torch.arange(int(hsize)), torch.arange(int(wsize)), indexing="ij",
            )
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            strides.append(torch.full((1, grid.shape[1]), float(stride)))
        grids = torch.cat(grids, dim=1).float().to(self.device)          # [1,A,2]
        expanded_strides = torch.cat(strides, dim=1).float().to(self.device)  # [1,A]
        return expanded_strides, grids

    def _gt_future_from_sample(self, sample, dets_t, det_cxcywh):
        """Match each detection to a GT track by IoU (greedy, mirrors
        TrajectoryMetricsAccumulator.update), then return that track's future boxes.
        Returns (gt_future[N,4], gt_current[N,4], valid[N], gt_future_steps[N,k,4],
        valid_steps[N,k]) aligned to detection order. NaN rows where no GT match;
        per-step validity marks where the object left frame.

        Mapper future-target keys (build_future_targets=True): sample["future_targets"]
        is {"boxes":[M,k,4] cxcywh future, "validity":[M,k] bool, "current_boxes":[M,4]},
        index-aligned to sample["targets"] [M,5] = (class_id, cx, cy, w, h).
        """
        n_det = 0 if dets_t is None else int(dets_t.shape[0])
        nan = np.full((n_det, 4), np.nan, dtype=np.float32)
        gt_future = nan.copy()
        gt_current = nan.copy()
        valid = np.zeros((n_det,), dtype=bool)
        gt_future_steps = np.full((n_det, self.k, 4), np.nan, dtype=np.float32)
        valid_steps = np.zeros((n_det, self.k), dtype=bool)
        if n_det == 0:
            return gt_future, gt_current, valid, gt_future_steps, valid_steps

        targets = sample.get("targets")
        ft = sample.get("future_targets")
        if targets is None or ft is None or int(targets.shape[0]) == 0:
            return gt_future, gt_current, valid, gt_future_steps, valid_steps

        gt = targets if isinstance(targets, torch.Tensor) else torch.as_tensor(targets)
        gt = gt.to(dets_t.device).float()
        n_gt = int(gt.shape[0])
        fut_boxes_steps = ft["boxes"][:n_gt].to(dets_t.device).float()      # [n_gt,k,4] cxcywh
        fut_valid_steps = ft["validity"][:n_gt].to(dets_t.device).bool()    # [n_gt,k]
        # Endpoint (t+k) step for the single-arrow viz panel.
        fut_boxes = fut_boxes_steps[:, -1, :]                                # [n_gt,4]
        fut_valid = fut_valid_steps[:, -1]                                   # [n_gt]
        curr_boxes = ft["current_boxes"][:n_gt].to(dets_t.device).float()  # [n_gt,4]

        # IoU(det xyxy, gt xyxy), greedy best-det-per-GT at threshold 0.5 (eval default).
        gt_xyxy = _targets_cxcywh_to_xyxy_torch(gt[:, 1:5])
        iou = bbox_iou(dets_t[:, :4], gt_xyxy)  # [N_det, n_gt]
        iou_threshold = 0.5
        if iou.numel() > 0:
            max_iou_per_det, best_gt_per_det = iou.max(dim=1)
            sorted_det_idx = max_iou_per_det.argsort(descending=True)
            used_gt = set()
            for d_idx in sorted_det_idx.tolist():
                g_idx = int(best_gt_per_det[d_idx].item())
                if max_iou_per_det[d_idx] < iou_threshold:
                    continue
                if g_idx in used_gt:
                    continue
                used_gt.add(g_idx)
                gt_current[d_idx] = curr_boxes[g_idx].detach().cpu().numpy()
                gt_future_steps[d_idx] = fut_boxes_steps[g_idx].detach().cpu().numpy()
                valid_steps[d_idx] = fut_valid_steps[g_idx].detach().cpu().numpy()
                if bool(fut_valid[g_idx].item()):
                    gt_future[d_idx] = fut_boxes[g_idx].detach().cpu().numpy()
                    valid[d_idx] = True
        return gt_future, gt_current, valid, gt_future_steps, valid_steps


def _detections_to_cxcywh(detections) -> np.ndarray:
    # YOLOXHead postprocess returns per-image [n, 7]: x1,y1,x2,y2,obj_conf,cls_conf,cls.
    # Take batch 0; if None (no dets) return empty [0,6].
    d = detections[0] if isinstance(detections, (list, tuple)) else detections
    if d is None or len(d) == 0:
        return np.zeros((0, 6), dtype=np.float32)
    d = d.detach().cpu().numpy()
    x1, y1, x2, y2 = d[:, 0], d[:, 1], d[:, 2], d[:, 3]
    cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1), (y2 - y1)
    score = d[:, 4] * d[:, 5]
    cls = d[:, 6]
    return np.stack([cx, cy, w, h, score, cls], axis=1).astype(np.float32)
