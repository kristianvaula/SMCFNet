from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from torch.utils.data import Dataset

from .annotations import PEODFrameAnnotation, load_peod_coco_annotations
from .io import abs_root, load_peod_yaml, read_timestamps_us


@dataclass(frozen=True)
class PEODSequenceMeta:
    root: Path
    split: str
    subset: str
    sequence: str
    rgb_dir: Path
    event_dat_path: Optional[Path]
    timestamps_path: Optional[Path]
    annotations_json_path: Optional[Path]
    image_height: int
    image_width: int
    num_frames: int
    flow_meta_path: Optional[Path] = None


def _canonical_split(split: str) -> str:
    split_norm = str(split).lower()
    if split_norm == "train":
        return "train"
    if split_norm in {"val", "validation"}:
        return "val"
    if split_norm in {"test", "eval", "evaluation"}:
        return "test"
    raise ValueError(f"Unsupported PEOD split {split!r}; expected train, val, or test")


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        if value.lower() == "auto":
            return ["auto"]
        return [value]
    if isinstance(value, Iterable):
        return [str(v) for v in value]
    raise TypeError(f"Expected list/string/None, got {type(value)!r}")


def _find_annotation_root(root: Path) -> Optional[Path]:
    candidates = [
        root / "coco_0111_last",
        root / "annotations" / "coco_0111_last",
        root / "annotations",
    ]
    for cand in candidates:
        if cand.is_dir():
            return cand
    return None


def _annotation_jsons(annotation_root: Path, split_dir: str, subsets: Sequence[str]) -> list[Path]:
    paths: list[Path] = []
    for subset in subsets:
        subset_dir = annotation_root / split_dir / subset
        if subset_dir.is_dir():
            paths.extend(sorted(subset_dir.glob("*.json")))
    return paths


def _sequence_from_annotation_path(path: Path) -> str:
    return path.stem


def _resolve_rgb_dir(root: Path, split_dir: str, subset: str, sequence: str) -> Path:
    candidates = [
        root / "rgb" / split_dir / sequence,
        root / "rgb" / split_dir / subset / sequence,
        root / split_dir / subset / "rgb" / sequence,
    ]
    for cand in candidates:
        if cand.is_dir():
            return cand
    return candidates[0]


def _resolve_event_path(root: Path, split_dir: str, subset: str, sequence: str) -> Optional[Path]:
    candidates = [
        root / "events" / split_dir / subset / f"{sequence}.dat",
        root / "events" / split_dir / f"{sequence}.dat",
        root / "event" / split_dir / f"{sequence}.dat",
        root / "event" / split_dir / subset / f"{sequence}.dat",
        root / split_dir / subset / "event" / f"{sequence}.dat",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return candidates[0]


def _resolve_timestamp_path(root: Path, split_dir: str, subset: str, sequence: str) -> Optional[Path]:
    candidates = [
        root / "timestamp" / split_dir / subset / f"{sequence}.csv",
        root / "timestamp" / split_dir / f"{sequence}.csv",
        root / "timestamp" / split_dir / subset / f"{sequence}.txt",
        root / "timestamp" / split_dir / f"{sequence}.txt",
        root / split_dir / subset / "timestamp" / f"{sequence}.csv",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return candidates[0]


def _resolve_flow_meta_path(root: Path, split_dir: str, subset: str, sequence: str) -> Optional[Path]:
    candidates = [
        root / "flow" / split_dir / subset / f"{sequence}.meta.npz",
        root / "flow" / split_dir / f"{sequence}.meta.npz",
        root / split_dir / subset / "flow" / f"{sequence}.meta.npz",
        root / "flow" / split_dir / subset / f"{sequence}.npz",
        root / "flow" / split_dir / f"{sequence}.npz",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return candidates[0]


class PEODSequenceIndexDataset(Dataset):
    def __init__(
        self,
        meta: PEODSequenceMeta,
        *,
        seq_idx: int,
        dt_ms: float,
        annotations: Optional[list[PEODFrameAnnotation]] = None,
    ):
        super().__init__()
        self.meta = meta
        self.seq_idx = int(seq_idx)
        self.sequence = meta.sequence
        self.dt_us = int(round(float(dt_ms) * 1000.0))
        if self.dt_us <= 0:
            raise ValueError(f"dt_ms must be > 0, got {dt_ms}")

        self.annotations = annotations
        if annotations is not None:
            self.frame_file_names = [frame.file_name for frame in annotations]
        else:
            self.frame_file_names = []

        if meta.timestamps_path is not None and meta.timestamps_path.exists():
            self._timestamps_us = read_timestamps_us(meta.timestamps_path)
        else:
            # PEOD is documented as 30 Hz. This fallback is only for early
            # skeleton/debug use before timestamp archives are available.
            self._timestamps_us = [int(round(i * 1_000_000.0 / 30.0)) for i in range(meta.num_frames)]

        if len(self._timestamps_us) < meta.num_frames:
            raise ValueError(
                f"Timestamp count {len(self._timestamps_us)} is smaller than "
                f"frame count {meta.num_frames} for {meta.sequence}"
            )

    def __len__(self) -> int:
        return self.meta.num_frames

    def __getitem__(self, frame_index: int) -> Dict[str, int]:
        i = int(frame_index)
        if i < 0:
            i = self.meta.num_frames + i
        if i < 0 or i >= self.meta.num_frames:
            raise IndexError(f"frame_index out of range: {frame_index}")
        t_ref_us = int(self._timestamps_us[i])
        t_start_us = max(0, t_ref_us - self.dt_us)
        return {
            "seq_idx": self.seq_idx,
            "frame_index": i,
            "t_ref_us": t_ref_us,
            "t_start_us": t_start_us,
            "t_end_us": t_ref_us,
        }

    def pull_item(self, frame_index: int) -> Dict[str, int]:
        return self.__getitem__(frame_index)


def build_peod_sequence_metas(
    *,
    peod_yaml_path: str | Path,
    split: str,
    sequences: Optional[Sequence[str]] = None,
) -> list[PEODSequenceMeta]:
    cfg = load_peod_yaml(peod_yaml_path)
    ds_cfg = cfg.get("dataset", {})
    root = abs_root(ds_cfg["root"])
    split_key = _canonical_split(split)
    split_dir = str(ds_cfg.get(f"{split_key}_split_dir", "train" if split_key == "val" else split_key))

    subset_key = f"{split_key}_subsets"
    subsets = _as_list(ds_cfg.get(subset_key))
    if len(subsets) == 0 and split_key == "val":
        subsets = _as_list(ds_cfg.get("train_subsets"))
    if len(subsets) == 0:
        subsets = ["challenge", "normal"] if split_dir == "test" else ["challenge", "motion_blur", "normal"]

    seq_filter = set(str(s) for s in sequences) if sequences is not None else None
    seq_cfg = _as_list(ds_cfg.get(f"{split_key}_sequences"))
    if seq_filter is None and seq_cfg and seq_cfg != ["auto"]:
        seq_filter = set(seq_cfg)

    # Guard against silent train==val leakage. PEOD ships no official validation
    # split, so any val split is carved from the train directory. If the val split
    # reads the same on-disk directory as train, it MUST be narrowed to an explicit
    # held-out sequence list that is disjoint from train; otherwise validation
    # silently overlaps (or equals) training. See issue #108.
    if split_key == "val":
        train_split_dir = str(ds_cfg.get("train_split_dir", "train"))
        if split_dir == train_split_dir:
            if seq_filter is None:
                raise ValueError(
                    f"PEOD val split reads from the train directory "
                    f"(val_split_dir={split_dir!r}) but dataset.val_sequences is empty or "
                    f"'auto', so validation would cover the entire train split (data leak). "
                    f"Set dataset.val_sequences to an explicit held-out list disjoint from "
                    f"train_sequences (PEOD has no official validation split)."
                )
            train_seq_cfg = _as_list(ds_cfg.get("train_sequences"))
            if train_seq_cfg and train_seq_cfg != ["auto"]:
                overlap = seq_filter & set(train_seq_cfg)
                if overlap:
                    raise ValueError(
                        f"PEOD val_sequences overlap train_sequences on the shared train "
                        f"directory: {sorted(overlap)}. Validation must be disjoint from "
                        f"training."
                    )

    annotation_root = _find_annotation_root(root)
    if annotation_root is None:
        raise FileNotFoundError(
            f"Could not find extracted PEOD annotation directory under {root}. "
            "Extract annotations.zip before building PEOD loaders."
        )

    metas: list[PEODSequenceMeta] = []
    for ann_path in _annotation_jsons(annotation_root, split_dir, subsets):
        subset = ann_path.parent.name
        sequence = _sequence_from_annotation_path(ann_path)
        if seq_filter is not None and sequence not in seq_filter:
            continue
        frames = load_peod_coco_annotations(ann_path)
        if len(frames) == 0:
            continue
        image_height = int(frames[0].height)
        image_width = int(frames[0].width)
        metas.append(
            PEODSequenceMeta(
                root=root,
                split=split_key,
                subset=subset,
                sequence=sequence,
                rgb_dir=_resolve_rgb_dir(root, split_dir, subset, sequence),
                event_dat_path=_resolve_event_path(root, split_dir, subset, sequence),
                timestamps_path=_resolve_timestamp_path(root, split_dir, subset, sequence),
                annotations_json_path=ann_path,
                image_height=image_height,
                image_width=image_width,
                num_frames=len(frames),
                flow_meta_path=_resolve_flow_meta_path(root, split_dir, subset, sequence),
            )
        )

    if len(metas) == 0:
        raise ValueError(f"No PEOD sequences found for split={split!r} in {peod_yaml_path}")
    return metas


def build_peod_sequence_index_datasets(
    *,
    peod_yaml_path: str | Path,
    split: str,
    dt_ms: float,
    sequences: Optional[Sequence[str]] = None,
) -> list[PEODSequenceIndexDataset]:
    metas = build_peod_sequence_metas(
        peod_yaml_path=peod_yaml_path,
        split=split,
        sequences=sequences,
    )
    datasets: list[PEODSequenceIndexDataset] = []
    for seq_idx, meta in enumerate(metas):
        annotations = (
            load_peod_coco_annotations(meta.annotations_json_path)
            if meta.annotations_json_path is not None
            else None
        )
        datasets.append(
            PEODSequenceIndexDataset(
                meta,
                seq_idx=seq_idx,
                dt_ms=dt_ms,
                annotations=annotations,
            )
        )
    return datasets
