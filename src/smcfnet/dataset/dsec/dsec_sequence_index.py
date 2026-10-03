#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Lightweight, map-style datasets that only index DSEC sequences.

Purpose
- Provide deterministic, sampler-friendly indexing across DSEC sequences.
- Keep __getitem__ extremely cheap (no HDF5 open, no image decode, no voxelization).

This module intentionally returns only *metadata* needed to later fetch events/images
in a mapper/prefetcher stage, enabling a GPU-first preprocessing pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import yaml
from torch.utils.data import Dataset


@dataclass(frozen=True)
class DSECSequenceMeta:
    root: Path
    split: str
    sequence: str
    camera: str

    seq_path: Path
    events_h5_path: Path
    rectify_map_h5_path: Optional[Path]

    images_dir: Optional[Path]
    timestamps_path: Path

    tracks_npy_path: Optional[Path]
    has_labels: bool

    num_frames: int


def _abs_root(root: str | Path) -> Path:
    root_p = Path(root)
    if not root_p.is_absolute():
        root_p = (Path.cwd() / root_p).resolve()
    return root_p


def _read_timestamps_us(path: Path) -> List[int]:
    if not path.exists():
        raise FileNotFoundError(f"Timestamps not found: {path}")

    ts_us: List[int] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            ts_us.append(int(s))

    if len(ts_us) == 0:
        raise ValueError(f"No timestamps found in {path}")

    return ts_us


def _resolve_images_dir(seq_path: Path, camera: str) -> Optional[Path]:
    cand_rect = seq_path / "images" / camera / "rectified"
    cand_dist = seq_path / "images" / camera / "distorted"

    if cand_rect.is_dir():
        return cand_rect
    if cand_dist.is_dir():
        return cand_dist
    return None


def _load_dsec_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"DSEC dataset config not found: {path}")
    if path.is_dir():
        raise IsADirectoryError(
            f"DSEC dataset config path is a directory, expected a YAML file: {path}. "
            "This can happen if config.data.dataset_config is empty (Path('') -> '.')"
        )
    with path.open("r") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid DSEC dataset config (expected dict): {path}")
    return cfg


def sequences_from_dsec_yaml(cfg: Dict[str, Any], split: str) -> List[str]:
    ds_cfg = cfg.get("dataset", {})
    split_norm = str(split).lower()
    if split_norm == "train":
        key = "train_sequences"
    elif split_norm in {"val", "validation"}:
        key = "val_sequences"
    elif split_norm in {"test", "eval", "evaluation"}:
        key = "test_sequences"
    else:
        raise ValueError(
            f"Unsupported DSEC split '{split}'. Expected one of: train, val, test."
        )
    seqs = ds_cfg.get(key, [])

    if not isinstance(seqs, list) or len(seqs) == 0:
        raise ValueError(
            f"No sequences found for split='{split}'. "
            f"Expected dataset.{key} in YAML."
        )

    return [str(s) for s in seqs]


def _canonical_split(split: str) -> str:
    split_norm = str(split).lower()
    if split_norm == "train":
        return "train"
    if split_norm in {"val", "validation"}:
        return "val"
    if split_norm in {"test", "eval", "evaluation"}:
        return "test"
    raise ValueError(
        f"Unsupported DSEC split '{split}'. Expected one of: train, val, test."
    )


class DSECSequenceIndexDataset(Dataset):
    """Map-style dataset for a single DSEC sequence.

    __getitem__ returns only minimal metadata:
      - seq_idx: int
      - frame_index: int
      - t_ref_us: int
      - t_start_us: int
      - t_end_us: int

    The heavy data (HDF5 events, images, labels) are referenced via `meta` and
    should be loaded later in a mapper.
    """

    def __init__(
        self,
        meta: DSECSequenceMeta,
        *,
        seq_idx: int,
        dt_ms: float,
        clamp_start_us: bool = True,
    ):
        super().__init__()
        self.meta = meta
        self.seq_idx = int(seq_idx)
        self.sequence = meta.sequence  # for logging/introspection

        dt_us = int(round(float(dt_ms) * 1000.0))
        if dt_us <= 0:
            raise ValueError(f"dt_ms must be > 0, got {dt_ms}")
        self.dt_us = dt_us
        self._clamp_start_us = bool(clamp_start_us)

        # Keep timestamps in memory for fast random access.
        self._timestamps_us = _read_timestamps_us(meta.timestamps_path)
        if len(self._timestamps_us) != meta.num_frames:
            # Meta is best-effort, but keep it honest.
            raise ValueError(
                f"Meta num_frames={meta.num_frames} does not match timestamps "
                f"count={len(self._timestamps_us)} for sequence '{meta.sequence}'."
            )

    def __len__(self) -> int:
        return self.meta.num_frames

    def __getitem__(self, frame_index: int) -> Dict[str, int]:
        i = int(frame_index)
        if i < 0:
            i = self.meta.num_frames + i
        if i < 0 or i >= self.meta.num_frames:
            raise IndexError(f"frame_index out of range: {frame_index}")

        t_end_us = int(self._timestamps_us[i])
        t_start_us = t_end_us - self.dt_us
        if self._clamp_start_us and t_start_us < 0:
            t_start_us = 0

        return {
            "seq_idx": self.seq_idx,
            "frame_index": i,
            "t_ref_us": t_end_us,
            "t_start_us": t_start_us,
            "t_end_us": t_end_us,
        }

    def pull_item(self, frame_index: int) -> Dict[str, int]:
        """Alias for __getitem__ to match the concat wrapper API."""
        return self.__getitem__(frame_index)


def build_dsec_sequence_metas(
    *,
    dsec_yaml_path: str | Path,
    split: str,
    sequences: Optional[Sequence[str]] = None,
    use_rectified: bool = False,
    use_rectified_events: Optional[bool] = None,
    use_rectified_tracks: Optional[bool] = None,
    rectified_events_filename: str = "rectified_events.h5",
) -> List[DSECSequenceMeta]:
    """Scan the filesystem and build `DSECSequenceMeta` for each sequence.
    
    Args:
        dsec_yaml_path: Path to DSEC dataset config YAML
        split: Dataset split ('train', 'val', or 'test')
        sequences: Optional list of specific sequences to load
        use_rectified: If True, use rectified_events.h5 and rectified_tracks.npy
                      for full-resolution rectified images (1440x1080)
        use_rectified_events: If set, independently control whether to require/use
                              a rectified events HDF5 file.
        use_rectified_tracks: If set, independently control whether to prefer/use
                              rectified_tracks.npy.
        rectified_events_filename: Filename under events/<camera>/ to use when
                                   use_rectified_events is true.
    """

    # Backwards compat: if the caller only uses `use_rectified`, apply it to both.
    if use_rectified_events is None:
        use_rectified_events = bool(use_rectified)
    if use_rectified_tracks is None:
        use_rectified_tracks = bool(use_rectified)

    cfg = _load_dsec_yaml(dsec_yaml_path)
    ds_cfg = cfg.get("dataset", {})

    root = _abs_root(ds_cfg.get("root"))
    camera = str(ds_cfg.get("camera", "left"))
    # Allow pointing a logical split (e.g. "val") at a different on-disk
    # subdirectory (e.g. "train") so held-out train-directory sequences can be
    # used without pretending they are the train split.
    split_key = _canonical_split(split)
    split_dir = str(ds_cfg.get(f"{split_key}_split_dir", split_key))

    if sequences is None:
        sequences = sequences_from_dsec_yaml(cfg, split_key)

    metas: List[DSECSequenceMeta] = []
    for seq in sequences:
        seq = str(seq)
        seq_path = root / split_dir / seq
        if not seq_path.is_dir():
            raise FileNotFoundError(
                f"Sequence directory not found: {seq_path} (split='{split}', sequence='{seq}')."
            )

        # Select events file based on rectified-events flag
        if use_rectified_events:
            events_h5 = seq_path / "events" / camera / str(rectified_events_filename)
            if not events_h5.exists():
                raise FileNotFoundError(
                    f"Rectified events file not found: {events_h5}. "
                    f"Run preprocessing to generate {rectified_events_filename}."
                )
        else:
            events_h5 = seq_path / "events" / camera / "events.h5"
            if not events_h5.exists():
                raise FileNotFoundError(f"Events file not found: {events_h5}")

        rectify_map_h5 = seq_path / "events" / camera / "rectify_map.h5"
        rectify_map_h5 = rectify_map_h5 if rectify_map_h5.exists() else None

        timestamps_path = seq_path / "images" / "timestamps.txt"
        ts_us = _read_timestamps_us(timestamps_path)

        images_dir = _resolve_images_dir(seq_path, camera)

        # Select tracks file based on rectified-tracks flag
        if use_rectified_tracks:
            tracks_path = seq_path / "object_detections" / camera / "rectified_tracks.npy"
            has_labels = tracks_path.exists()
            if has_labels:
                tracks_path = tracks_path
            else:
                # Fall back to non-rectified if rectified doesn't exist
                tracks_path = seq_path / "object_detections" / camera / "tracks.npy"
                has_labels = tracks_path.exists()
                tracks_path = tracks_path if has_labels else None
        else:
            tracks_path = seq_path / "object_detections" / camera / "tracks.npy"
            has_labels = tracks_path.exists()
            tracks_path = tracks_path if has_labels else None

        metas.append(
            DSECSequenceMeta(
                root=root,
                split=split_key,
                sequence=seq,
                camera=camera,
                seq_path=seq_path,
                events_h5_path=events_h5,
                rectify_map_h5_path=rectify_map_h5,
                images_dir=images_dir,
                timestamps_path=timestamps_path,
                tracks_npy_path=tracks_path,
                has_labels=bool(has_labels),
                num_frames=len(ts_us),
            )
        )

    return metas


def build_dsec_sequence_index_datasets(
    *,
    dsec_yaml_path: str | Path,
    split: str,
    dt_ms: float,
    sequences: Optional[Sequence[str]] = None,
    use_rectified: bool = False,
    use_rectified_events: Optional[bool] = None,
    use_rectified_tracks: Optional[bool] = None,
    rectified_events_filename: str = "rectified_events.h5",
) -> List[DSECSequenceIndexDataset]:
    """Convenience: build per-sequence index datasets for a given split.
    
    Args:
        dsec_yaml_path: Path to DSEC dataset config YAML
        split: Dataset split ('train', 'val', or 'test')
        dt_ms: Event window duration in milliseconds
        sequences: Optional list of specific sequences to load
        use_rectified: If True, use rectified_events.h5 and rectified_tracks.npy
        use_rectified_events: If set, independently control whether to require/use
                              a rectified events HDF5 file.
        use_rectified_tracks: If set, independently control whether to prefer/use
                              rectified_tracks.npy.
        rectified_events_filename: Filename under events/<camera>/ to use when
                                   use_rectified_events is true.
    """

    metas = build_dsec_sequence_metas(
        dsec_yaml_path=dsec_yaml_path,
        split=split,
        sequences=sequences,
        use_rectified=use_rectified,
        use_rectified_events=use_rectified_events,
        use_rectified_tracks=use_rectified_tracks,
        rectified_events_filename=rectified_events_filename,
    )
    return [
        DSECSequenceIndexDataset(meta, seq_idx=i, dt_ms=dt_ms)
        for i, meta in enumerate(metas)
    ]
