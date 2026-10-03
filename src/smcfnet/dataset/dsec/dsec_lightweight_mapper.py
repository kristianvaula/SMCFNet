#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Lightweight mapper for DSEC.

Goal
- Keep CPU work minimal in the DataLoader workers.
- Materialize *raw* tensors (events, optionally rgb/labels) in a clean, tensor-first format.
- Defer expensive transforms (e.g. voxelization via `events_to_voxel_grid`) to a later stage
  (GPU or training step) to avoid dataloader becoming the bottleneck.

This mirrors Detectron2's guidance: DataLoader is fast when transferring `torch.Tensor`
objects and slow for large Python data structures.
"""

from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from smcfnet.dataset.dsec.dsec_slicer import DSECEventSlicer
from smcfnet.dataset.dsec.dsec_sequence_index import DSECSequenceIndexDataset


class DSECLightweightMapper:
    """Callable that maps a DSEC index item -> raw tensors.

    Inputs
    - `index_item`: output of `DSECSequenceIndexDataset.__getitem__` (typically through the
      concatenated `DSECDataset`), containing:
        {seq_idx, frame_index, t_start_us, t_end_us, t_ref_us}

    Outputs (default)
    - dict with:
        - seq_idx: int
        - frame_index: int
        - t_start_us/t_end_us/t_ref_us: int
        - events: torch.Tensor of shape (N,4) with columns [x,y,t,p]
          where `t` is controlled by `event_timestamp_unit`

    Notes
    - HDF5 files are opened lazily and per-process (safe with DataLoader workers).
    - We intentionally do NOT build voxel grids here.
    """

    def __init__(
        self,
        sequence_datasets: List[DSECSequenceIndexDataset],
        *,
        load_events: bool = True,
        load_tracks: bool = False,
        mmap_tracks: bool = True,
        load_images: bool = False,
        load_flow: bool = False,
        image_height: Optional[int] = None,
        image_width: Optional[int] = None,
        prefer_distorted: bool = True,
        build_targets: bool = False,
        build_future_targets: bool = False,
        future_steps: int = 5,
        future_target_mode: str = "absolute",
        event_timestamp_unit: str = "ms",
    ):
        if event_timestamp_unit not in {"ms", "us"}:
            raise ValueError(
                f"event_timestamp_unit must be 'ms' or 'us', got {event_timestamp_unit!r}"
            )
        if future_target_mode not in {"absolute", "delta"}:
            raise ValueError(
                f"future_target_mode must be 'absolute' or 'delta', got {future_target_mode!r}"
            )
        self.sequence_datasets = sequence_datasets
        self.load_events = load_events
        self.load_tracks = load_tracks
        self.mmap_tracks = mmap_tracks
        self.load_images = load_images
        self.load_flow = load_flow
        self.image_height = int(image_height) if image_height is not None else None
        self.image_width = int(image_width) if image_width is not None else None
        self.prefer_distorted = bool(prefer_distorted)
        self.build_targets = bool(build_targets)
        self.build_future_targets = bool(build_future_targets)
        self.future_steps = int(future_steps)
        self.future_target_mode = str(future_target_mode)
        self.event_timestamp_unit = str(event_timestamp_unit)

        # Per-process caches (re-initialized in each worker process).
        self._pid: Optional[int] = None
        self._h5: Dict[int, Any] = {}
        self._slicer: Dict[int, DSECEventSlicer] = {}
        self._tracks: Dict[int, Any] = {}
        self._tracks_t: Dict[int, Any] = {}

    def _reset_if_needed(self) -> None:
        pid = os.getpid()
        if self._pid == pid:
            return

        # New process (DataLoader worker fork/spawn): close inherited handles.
        for fh in self._h5.values():
            try:
                fh.close()
            except Exception:
                pass
        self._h5.clear()
        self._slicer.clear()
        self._tracks.clear()
        self._tracks_t.clear()
        self._pid = pid

    def reset_caches(self) -> None:
        """Explicitly reset all cached data. Called between epochs to free memory.
        
        Closes HDF5 file handles and clears track numpy array references.
        This is safe to call from the main process and prevents 4-20 GB accumulation
        of cached track arrays across epochs.
        """
        # Close HDF5 files
        for fh in self._h5.values():
            try:
                fh.close()
            except Exception:
                pass
        
        # Clear all mappings
        self._h5.clear()
        self._slicer.clear()
        self._tracks.clear()
        self._tracks_t.clear()
        
        import logging
        logger = logging.getLogger(__name__)
        logger.debug("Mapper caches reset: cleared HDF5 handles and track arrays")

    def _ensure_events_open(self, seq_idx: int) -> None:
        self._reset_if_needed()
        if seq_idx in self._slicer:
            return

        import os
        import h5py  # local import to keep module import cheap
        import hdf5plugin  # noqa: F401 (required for compressed DSEC events)

        # Disable blosc multi-threading to avoid heap corruption from the
        # blosc C extension's thread pool on mmap'd decompression buffers.
        os.environ.setdefault("BLOSC_NTHREADS", "1")

        meta = self.sequence_datasets[seq_idx].meta
        # 64 MB chunk cache (default 1 MB) keeps recently decompressed blosc
        # chunks in memory, drastically reducing decompression calls per frame.
        fh = h5py.File(
            str(meta.events_h5_path),
            "r",
            rdcc_nbytes=64 * 1024 * 1024,
            rdcc_nslots=10007,
        )
        self._h5[seq_idx] = fh
        self._slicer[seq_idx] = DSECEventSlicer(fh)

    def _ensure_tracks_open(self, seq_idx: int) -> None:
        self._reset_if_needed()
        if seq_idx in self._tracks:
            return

        meta = self.sequence_datasets[seq_idx].meta
        if not meta.has_labels or meta.tracks_npy_path is None:
            self._tracks[seq_idx] = None
            return

        mmap_mode = "r" if self.mmap_tracks else None
        tracks = np.load(str(meta.tracks_npy_path), mmap_mode=mmap_mode, allow_pickle=False)
        self._tracks[seq_idx] = tracks
        try:
            self._tracks_t[seq_idx] = tracks["t"]
        except Exception:
            self._tracks_t[seq_idx] = None

    def _load_image_tensor(self, seq_idx: int, frame_index: int) -> torch.Tensor:
        meta = self.sequence_datasets[seq_idx].meta
        h = int(self.image_height) if self.image_height is not None else 480
        w = int(self.image_width) if self.image_width is not None else 640

        if meta.images_dir is None:
            return torch.zeros((3, h, w), dtype=torch.uint8)

        from pathlib import Path

        img_dir = Path(meta.images_dir)
        img_name = f"{int(frame_index):06d}.png"

        # Prefer distorted images when available (matches event sensor resolution).
        if self.prefer_distorted and img_dir.name == "rectified":
            distorted_dir = img_dir.parent / "distorted"
            if distorted_dir.is_dir() and (distorted_dir / img_name).exists():
                img_dir = distorted_dir

        img_path = img_dir / img_name
        if not img_path.exists():
            return torch.zeros((3, h, w), dtype=torch.uint8)

        import cv2

        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            return torch.zeros((3, h, w), dtype=torch.uint8)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if img_rgb.shape[0] != h or img_rgb.shape[1] != w:
            img_rgb = cv2.resize(img_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        # Keep images compact in host memory; convert to float later on-device.
        return torch.from_numpy(img_rgb).permute(2, 0, 1).contiguous()

    def _targets_for_frame(self, seq_idx: int, t_ref_us: int, img_h: int, img_w: int) -> torch.Tensor:
        # Requires tracks loaded.
        self._ensure_tracks_open(seq_idx)
        tracks = self._tracks.get(seq_idx)
        t_arr = self._tracks_t.get(seq_idx)
        if tracks is None or t_arr is None:
            return torch.zeros((0, 5), dtype=torch.float32)

        t_ref_us = int(t_ref_us)
        left = int(np.searchsorted(t_arr, t_ref_us, side="left"))
        right = int(np.searchsorted(t_arr, t_ref_us, side="right"))
        if right <= left:
            return torch.zeros((0, 5), dtype=torch.float32)
        tr = tracks[left:right]

        labels = np.stack([tr["x"], tr["y"], tr["w"], tr["h"], tr["class_id"]], axis=1).astype(np.float32)
        labels_t = torch.from_numpy(labels)

        out = torch.zeros_like(labels_t)
        out[:, 0] = labels_t[:, 4]
        out[:, 1] = labels_t[:, 0] + labels_t[:, 2] / 2.0
        out[:, 2] = labels_t[:, 1] + labels_t[:, 3] / 2.0
        out[:, 3] = labels_t[:, 2]
        out[:, 4] = labels_t[:, 3]

        out[:, 1] = torch.clamp(out[:, 1], 0, float(img_w - 1))
        out[:, 2] = torch.clamp(out[:, 2], 0, float(img_h - 1))

        # Guard against degenerate / corrupt track boxes.
        # Downstream matching/loss code assumes strictly positive box sizes.
        valid = torch.isfinite(out).all(dim=1) & (out[:, 3] > 0) & (out[:, 4] > 0)
        if not bool(valid.all().item()):
            out = out[valid]
        return out

    def _future_targets_for_frame(
        self, seq_idx: int, t_ref_us: int, num_current: int, img_h: int, img_w: int,
    ) -> Dict[str, torch.Tensor]:
        """Look up GT boxes at t+k and pair with current-frame objects via track_id.

        Args:
            seq_idx: sequence index.
            t_ref_us: reference timestamp of current frame (microseconds).
            num_current: number of current-frame GT objects (determines output size).
            img_h, img_w: image dimensions for clamping.

        Returns:
            dict with:
              "boxes": [num_current, k, 4] future boxes (cxcywh) at steps t+1..t+k
                (0 for unmatched / out-of-range steps).
              "validity": [num_current, k] bool, True where the object has a
                track-id match at that step.
              "current_boxes": [num_current, 4] current-frame boxes (cxcywh).
        """
        k = self.future_steps
        empty = {
            "boxes": torch.zeros((num_current, k, 4), dtype=torch.float32),
            "validity": torch.zeros((num_current, k), dtype=torch.bool),
            "current_boxes": torch.zeros((num_current, 4), dtype=torch.float32),
        }
        if num_current == 0:
            return empty

        seq_ds = self.sequence_datasets[seq_idx]
        timestamps = seq_ds._timestamps_us
        # Find current frame index via bisect (O(log n) instead of O(n))
        import bisect
        curr_idx = bisect.bisect_left(timestamps, t_ref_us)
        if curr_idx >= len(timestamps) or timestamps[curr_idx] != t_ref_us:
            return empty
        # Need at least the first future step to exist.
        if curr_idx + 1 >= len(timestamps):
            return empty

        self._ensure_tracks_open(seq_idx)
        tracks = self._tracks.get(seq_idx)
        t_arr = self._tracks_t.get(seq_idx)
        if tracks is None or t_arr is None:
            return empty

        # Get current-frame tracks (with track_id) — step-independent, build once.
        left_c = int(np.searchsorted(t_arr, t_ref_us, side="left"))
        right_c = int(np.searchsorted(t_arr, t_ref_us, side="right"))
        if right_c <= left_c:
            return empty
        curr_tracks = tracks[left_c:right_c]
        curr_boxes_np = np.stack([
            curr_tracks["x"], curr_tracks["y"],
            curr_tracks["w"], curr_tracks["h"],
        ], axis=1).astype(np.float32)
        curr_cx = np.clip(curr_boxes_np[:, 0] + curr_boxes_np[:, 2] / 2.0, 0, img_w - 1)
        curr_cy = np.clip(curr_boxes_np[:, 1] + curr_boxes_np[:, 3] / 2.0, 0, img_h - 1)
        curr_cxcywh = np.stack([curr_cx, curr_cy, curr_boxes_np[:, 2], curr_boxes_np[:, 3]], axis=1)
        curr_valid = np.isfinite(curr_cxcywh).all(axis=1) & (curr_boxes_np[:, 2] > 0) & (curr_boxes_np[:, 3] > 0)
        curr_tracks = curr_tracks[curr_valid]
        curr_cxcywh = curr_cxcywh[curr_valid]
        if curr_tracks.shape[0] == 0:
            return empty
        curr_ids = curr_tracks["track_id"]

        boxes_out = torch.zeros((num_current, k, 4), dtype=torch.float32)
        valid_out = torch.zeros((num_current, k), dtype=torch.bool)
        current_boxes_out = torch.zeros((num_current, 4), dtype=torch.float32)
        for i, curr_box in enumerate(curr_cxcywh):
            if i >= num_current:
                break
            current_boxes_out[i] = torch.from_numpy(curr_box)

        # Trace each current object forward step by step via track_id.
        for s in range(1, k + 1):
            fut_idx = curr_idx + s
            if fut_idx >= len(timestamps):
                break  # no further future frames; remaining steps stay invalid
            t_fut_us = int(timestamps[fut_idx])

            left_f = int(np.searchsorted(t_arr, t_fut_us, side="left"))
            right_f = int(np.searchsorted(t_arr, t_fut_us, side="right"))
            if right_f <= left_f:
                continue  # no labelled objects at this step
            fut_tracks = tracks[left_f:right_f]
            fut_ids = fut_tracks["track_id"]
            fut_boxes_np = np.stack([
                fut_tracks["x"], fut_tracks["y"],
                fut_tracks["w"], fut_tracks["h"],
            ], axis=1).astype(np.float32)
            fut_id_to_idx = {int(tid): j for j, tid in enumerate(fut_ids)}

            for i, (tid, curr_box) in enumerate(zip(curr_ids, curr_cxcywh)):
                if i >= num_current:
                    break
                fut_i = fut_id_to_idx.get(int(tid))
                if fut_i is None:
                    continue
                fb = fut_boxes_np[fut_i]
                fut_cx = float(np.clip(fb[0] + fb[2] / 2.0, 0, img_w - 1))
                fut_cy = float(np.clip(fb[1] + fb[3] / 2.0, 0, img_h - 1))
                fut_box = np.array([fut_cx, fut_cy, float(fb[2]), float(fb[3])], dtype=np.float32)
                if self.future_target_mode == "delta":
                    boxes_out[i, s - 1, 0] = float((fut_box[0] - curr_box[0]) / float(img_w))
                    boxes_out[i, s - 1, 1] = float((fut_box[1] - curr_box[1]) / float(img_h))
                    boxes_out[i, s - 1, 2] = float((fut_box[2] - curr_box[2]) / float(img_w))
                    boxes_out[i, s - 1, 3] = float((fut_box[3] - curr_box[3]) / float(img_h))
                else:
                    boxes_out[i, s - 1] = torch.from_numpy(fut_box)
                valid_out[i, s - 1] = True

        return {"boxes": boxes_out, "validity": valid_out, "current_boxes": current_boxes_out}

    @staticmethod
    def _load_flow_png(flow_path: str) -> torch.Tensor:
        """Load a DSEC 16-bit PNG optical flow file.

        Returns a (3, H, W) float32 tensor where:
          - channel 0: flow_x (pixels)
          - channel 1: flow_y (pixels)
          - channel 2: validity mask (1.0 = valid, 0.0 = invalid)
        """
        import cv2

        # cv2 reads 16-bit PNG as (H, W, 3) uint16 in BGR order.
        # DSEC stores as RGB: ch0=flow_x, ch1=flow_y, ch2=valid.
        img = cv2.imread(flow_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Cannot read flow PNG: {flow_path}")
        assert img.dtype == np.uint16 and img.ndim == 3 and img.shape[2] == 3

        # Convert BGR → RGB channel order
        flow_16bit = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w, _ = flow_16bit.shape
        valid = (flow_16bit[:, :, 2] == 1).astype(np.float32)
        flow_float = flow_16bit.astype(np.float64)
        flow_x = np.where(valid > 0.5, (flow_float[:, :, 0] - 2**15) / 128.0, 0.0).astype(np.float32)
        flow_y = np.where(valid > 0.5, (flow_float[:, :, 1] - 2**15) / 128.0, 0.0).astype(np.float32)
        # Stack as (3, H, W): [flow_x, flow_y, valid]
        flow_t = torch.from_numpy(np.stack([flow_x, flow_y, valid], axis=0))
        return flow_t.contiguous()

    def __call__(self, index_item: Dict[str, int]) -> Dict[str, Any]:
        seq_idx = int(index_item["seq_idx"])
        frame_index = int(index_item["frame_index"])
        t_start_us = int(index_item["t_start_us"])
        t_end_us = int(index_item["t_end_us"])
        t_ref_us = int(index_item.get("t_ref_us", t_end_us))

        out: Dict[str, Any] = {
            "seq_idx": seq_idx,
            "frame_index": frame_index,
            "t_start_us": t_start_us,
            "t_end_us": t_end_us,
            "t_ref_us": t_ref_us,
            # The streaming path keeps an explicit reset flag in the batch
            # contract for known sequence starts. Runtime code still augments
            # this with inferred resets across shuffled batches.
            "sequence_reset": bool(frame_index <= 0),
        }

        if self.load_events:
            self._ensure_events_open(seq_idx)
            window = self._slicer[seq_idx].get_window_us(
                t_start_us,
                t_end_us,
                sequence_id=self.sequence_datasets[seq_idx].sequence,
                frame_index=frame_index,
                timestamp_unit=self.event_timestamp_unit,
            )
            if window is None:
                events = torch.empty((0, 4), dtype=torch.int64)
            else:
                events = window.events
                if not isinstance(events, torch.Tensor):
                    events = torch.as_tensor(events)
                # Ensure a stable, fast-to-collate layout.
                if events.numel() == 0:
                    events = events.reshape(0, 4)
                events = events.contiguous()
            out["events"] = events

        if self.load_tracks:
            self._ensure_tracks_open(seq_idx)
            tracks = self._tracks.get(seq_idx)
            # Detection training consumes compact per-frame targets. Do not send
            # full sequence track arrays through DataLoader queues for every sample.
            if not self.build_targets:
                out["tracks"] = tracks

        if self.load_images:
            out["image"] = self._load_image_tensor(seq_idx, frame_index)

        if self.build_targets:
            # If build_targets is enabled, we must have tracks available.
            img_h = int(self.image_height) if self.image_height is not None else 480
            img_w = int(self.image_width) if self.image_width is not None else 640
            out["targets"] = self._targets_for_frame(seq_idx, t_ref_us, img_h=img_h, img_w=img_w)

            if self.build_future_targets:
                num_current = int(out["targets"].shape[0])
                out["future_targets"] = self._future_targets_for_frame(
                    seq_idx, t_ref_us, num_current, img_h=img_h, img_w=img_w,
                )

        if self.load_flow:
            flow_path = index_item.get("flow_png_path")
            if flow_path is not None:
                out["flow_gt"] = self._load_flow_png(str(flow_path))

        return out
