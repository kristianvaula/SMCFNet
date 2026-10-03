from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch

from .annotations import PEODFrameAnnotation, load_peod_coco_annotations
from .peod_event_slicer import PEODEventSlicer
from .peod_sequence_index import PEODSequenceIndexDataset


class PEODLightweightMapper:
    """Map PEOD index items to the same raw sample contract as DSEC."""

    def __init__(
        self,
        sequence_datasets: List[PEODSequenceIndexDataset],
        *,
        load_events: bool = True,
        load_tracks: bool = False,
        mmap_tracks: bool = True,
        load_images: bool = False,
        load_flow: bool = False,
        image_height: Optional[int] = None,
        image_width: Optional[int] = None,
        prefer_distorted: bool = False,
        build_targets: bool = False,
        build_future_targets: bool = False,
        future_steps: int = 5,
        future_target_mode: str = "absolute",
        event_timestamp_unit: str = "ms",
    ):
        if event_timestamp_unit not in {"ms", "us"}:
            raise ValueError(f"event_timestamp_unit must be 'ms' or 'us', got {event_timestamp_unit!r}")
        self.sequence_datasets = sequence_datasets
        self.load_events = bool(load_events)
        self.load_images = bool(load_images)
        self.build_targets = bool(build_targets)
        self.image_height = int(image_height) if image_height is not None else None
        self.image_width = int(image_width) if image_width is not None else None
        self.event_timestamp_unit = str(event_timestamp_unit)

        # Accepted for API compatibility with DSECLightweightMapper.
        self.load_tracks = bool(load_tracks)
        self.mmap_tracks = bool(mmap_tracks)
        self.load_flow = bool(load_flow)
        self.prefer_distorted = bool(prefer_distorted)
        self.build_future_targets = bool(build_future_targets)
        self.future_steps = int(future_steps)
        self.future_target_mode = str(future_target_mode)

        self._pid: Optional[int] = None
        self._slicers: Dict[int, PEODEventSlicer] = {}
        self._annotations: Dict[int, list[PEODFrameAnnotation]] = {}
        self._flow_cache: Dict[int, tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]] = {}

    def _ensure_runtime_caches(self) -> None:
        if not hasattr(self, "_pid"):
            self._pid = None
        if not hasattr(self, "_slicers"):
            self._slicers = {}
        if not hasattr(self, "_annotations"):
            self._annotations = {}
        if not hasattr(self, "_flow_cache"):
            self._flow_cache = {}

    def _reset_if_needed(self) -> None:
        self._ensure_runtime_caches()
        pid = os.getpid()
        if self._pid == pid:
            return
        self._slicers.clear()
        self._annotations.clear()
        self._flow_cache.clear()
        self._pid = pid

    def reset_caches(self) -> None:
        self._ensure_runtime_caches()
        self._slicers.clear()
        self._annotations.clear()
        self._flow_cache.clear()

    def _ensure_slicer(self, seq_idx: int) -> PEODEventSlicer:
        self._reset_if_needed()
        if seq_idx not in self._slicers:
            meta = self.sequence_datasets[seq_idx].meta
            if meta.event_dat_path is None:
                raise FileNotFoundError(f"No PEOD event path recorded for {meta.sequence}")
            self._slicers[seq_idx] = PEODEventSlicer(meta.event_dat_path)
        return self._slicers[seq_idx]

    def _ensure_annotations(self, seq_idx: int) -> list[PEODFrameAnnotation]:
        self._reset_if_needed()
        if seq_idx not in self._annotations:
            seq_ds = self.sequence_datasets[seq_idx]
            if seq_ds.annotations is not None:
                self._annotations[seq_idx] = seq_ds.annotations
            else:
                meta = seq_ds.meta
                if meta.annotations_json_path is None:
                    self._annotations[seq_idx] = []
                else:
                    self._annotations[seq_idx] = load_peod_coco_annotations(meta.annotations_json_path)
        return self._annotations[seq_idx]

    def _load_image_tensor(self, seq_idx: int, frame_index: int) -> torch.Tensor:
        seq_ds = self.sequence_datasets[seq_idx]
        meta = seq_ds.meta
        h = int(self.image_height) if self.image_height is not None else int(meta.image_height)
        w = int(self.image_width) if self.image_width is not None else int(meta.image_width)

        if len(seq_ds.frame_file_names) > frame_index:
            img_path = Path(meta.rgb_dir) / seq_ds.frame_file_names[frame_index]
        else:
            img_path = Path(meta.rgb_dir) / f"{meta.sequence}_{frame_index + 1:04d}.png"

        if not img_path.exists():
            return torch.zeros((3, h, w), dtype=torch.uint8)

        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            return torch.zeros((3, h, w), dtype=torch.uint8)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if img_rgb.shape[0] != h or img_rgb.shape[1] != w:
            img_rgb = cv2.resize(img_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        return torch.from_numpy(img_rgb).permute(2, 0, 1).contiguous()

    def _targets_for_frame(self, seq_idx: int, frame_index: int) -> torch.Tensor:
        annotations = self._ensure_annotations(seq_idx)
        if frame_index < 0 or frame_index >= len(annotations):
            return torch.zeros((0, 5), dtype=torch.float32)
        return annotations[frame_index].targets.clone()

    @staticmethod
    def _resize_flow_masked(flow: np.ndarray, valid: np.ndarray, *, target_h: int, target_w: int) -> np.ndarray:
        """Resize [2,H,W] flow to target size without invalid-mask bleeding."""
        if flow.ndim != 3 or flow.shape[0] != 2:
            raise ValueError(f"Expected flow shape [2,H,W], got {flow.shape}")
        if valid.ndim != 2:
            raise ValueError(f"Expected valid shape [H,W], got {valid.shape}")

        src_h, src_w = int(flow.shape[1]), int(flow.shape[2])
        if src_h == int(target_h) and src_w == int(target_w):
            out = np.empty((3, src_h, src_w), dtype=np.float32)
            out[0:2] = flow.astype(np.float32, copy=False)
            out[2] = (valid > 0).astype(np.float32, copy=False)
            out[0:2] *= out[2:3]
            return out

        valid_f = (valid > 0).astype(np.float32, copy=False)
        den = cv2.resize(valid_f, (int(target_w), int(target_h)), interpolation=cv2.INTER_LINEAR)
        out = np.zeros((3, int(target_h), int(target_w)), dtype=np.float32)
        scale_x = float(target_w) / float(src_w)
        scale_y = float(target_h) / float(src_h)
        valid_out = den >= 0.5
        for c, scale in ((0, scale_x), (1, scale_y)):
            num = cv2.resize(
                flow[c].astype(np.float32, copy=False) * valid_f,
                (int(target_w), int(target_h)),
                interpolation=cv2.INTER_LINEAR,
            )
            vals = np.divide(num, np.maximum(den, 1.0e-6), out=np.zeros_like(num), where=den > 1.0e-6)
            vals *= float(scale)
            vals[~valid_out] = 0.0
            out[c] = vals
        out[2] = valid_out.astype(np.float32)
        return out

    def _ensure_flow_arrays(self, seq_idx: int) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        self._reset_if_needed()
        if seq_idx in self._flow_cache:
            return self._flow_cache[seq_idx]

        meta = self.sequence_datasets[seq_idx].meta
        flow_meta_path = getattr(meta, "flow_meta_path", None)
        if flow_meta_path is None:
            raise FileNotFoundError(f"No PEOD flow path recorded for {meta.sequence}")
        flow_meta_path = Path(flow_meta_path)
        if not flow_meta_path.exists():
            raise FileNotFoundError(
                f"PEOD flow metadata not found for {meta.sequence}: {flow_meta_path}"
            )

        t_ref_us: Optional[np.ndarray] = None
        if flow_meta_path.name.endswith(".meta.npz"):
            with np.load(flow_meta_path, allow_pickle=False) as meta_npz:
                flow_path = Path(str(meta_npz["flow_path"])) if "flow_path" in meta_npz else flow_meta_path.with_suffix("").with_suffix(".flow.npy")
                valid_path = Path(str(meta_npz["valid_path"])) if "valid_path" in meta_npz else flow_meta_path.with_suffix("").with_suffix(".valid.npy")
                if "t_ref_us" in meta_npz:
                    t_ref_us = np.asarray(meta_npz["t_ref_us"], dtype=np.int64)
        else:
            # Legacy single-NPZ support for early conversions.
            legacy = np.load(flow_meta_path, allow_pickle=False)
            if "flow" not in legacy or "valid" not in legacy:
                raise ValueError(f"Legacy PEOD flow NPZ missing flow/valid arrays: {flow_meta_path}")
            flow = np.asarray(legacy["flow"], dtype=np.float32)
            valid = np.asarray(legacy["valid"], dtype=np.uint8)
            t_ref_us = np.asarray(legacy["t_ref_us"], dtype=np.int64) if "t_ref_us" in legacy else None
            self._flow_cache[seq_idx] = (flow, valid, t_ref_us)
            return self._flow_cache[seq_idx]

        if not flow_path.is_absolute():
            flow_path = flow_meta_path.parent / flow_path
        if not valid_path.is_absolute():
            valid_path = flow_meta_path.parent / valid_path
        if not flow_path.exists():
            raise FileNotFoundError(f"PEOD flow array not found for {meta.sequence}: {flow_path}")
        if not valid_path.exists():
            raise FileNotFoundError(f"PEOD flow validity array not found for {meta.sequence}: {valid_path}")

        flow = np.load(flow_path, mmap_mode="r")
        valid = np.load(valid_path, mmap_mode="r")
        self._flow_cache[seq_idx] = (flow, valid, t_ref_us)
        return self._flow_cache[seq_idx]

    def _load_flow_tensor(self, seq_idx: int, frame_index: int, t_ref_us: int) -> torch.Tensor:
        flow_arr, valid_arr, t_ref_arr = self._ensure_flow_arrays(seq_idx)
        if frame_index < 0 or frame_index >= int(flow_arr.shape[0]):
            raise IndexError(
                f"PEOD flow frame_index out of range for {self.sequence_datasets[seq_idx].sequence}: "
                f"{frame_index} not in [0, {int(flow_arr.shape[0])})"
            )
        if t_ref_arr is not None and frame_index < len(t_ref_arr):
            expected = int(t_ref_arr[frame_index])
            if abs(expected - int(t_ref_us)) > 1:
                raise ValueError(
                    f"PEOD flow timestamp mismatch for {self.sequence_datasets[seq_idx].sequence} "
                    f"frame {frame_index}: index t_ref_us={t_ref_us}, flow t_ref_us={expected}"
                )

        flow_np = np.asarray(flow_arr[frame_index], dtype=np.float32)
        valid_np = np.asarray(valid_arr[frame_index], dtype=np.uint8)
        h = int(self.image_height) if self.image_height is not None else int(flow_np.shape[1])
        w = int(self.image_width) if self.image_width is not None else int(flow_np.shape[2])
        out_np = self._resize_flow_masked(flow_np, valid_np, target_h=h, target_w=w)
        return torch.from_numpy(out_np.copy()).contiguous()

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
            "sequence_reset": bool(frame_index <= 0),
        }

        if self.load_events:
            slicer = self._ensure_slicer(seq_idx)
            window = slicer.get_window_us(
                t_start_us,
                t_end_us,
                sequence_id=self.sequence_datasets[seq_idx].sequence,
                frame_index=frame_index,
                timestamp_unit=self.event_timestamp_unit,
            )
            out["events"] = window.events.contiguous()

        if self.load_images:
            out["image"] = self._load_image_tensor(seq_idx, frame_index)

        if self.build_targets:
            out["targets"] = self._targets_for_frame(seq_idx, frame_index)

        if self.load_flow:
            out["flow_gt"] = self._load_flow_tensor(seq_idx, frame_index, t_ref_us)

        if self.build_future_targets:
            raise NotImplementedError("PEOD future targets require persistent track ids, which are not available yet.")

        return out
