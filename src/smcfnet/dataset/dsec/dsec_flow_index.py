"""Flow-aligned index dataset for supervised optical flow training.

Only yields frames that have corresponding ground-truth optical flow from
the DSEC dataset.  Each sample provides the event time window that aligns
with a forward-flow displacement map and the path to the flow PNG file.

Usage:
  Triggered when ``config.data.mapper_load_flow = True`` in the data
  configuration.  See ``build.py::build_dsec_loader_cfg`` for wiring.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from torch.utils.data import Dataset

from smcfnet.dataset.dsec.dsec_sequence_index import (
    DSECSequenceMeta,
    _abs_root,
    _load_dsec_yaml,
    sequences_from_dsec_yaml,
)


def _read_flow_timestamps(ts_path: Path) -> np.ndarray:
    """Read forward_timestamps.txt → (N, 2) int64 array of (from_us, to_us)."""
    data = np.loadtxt(str(ts_path), delimiter=",", dtype="int64")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 2:
        raise ValueError(f"Expected at least 2 columns in {ts_path}, got {data.shape[1]}")
    return data[:, :2]


class DSECFlowIndexDataset(Dataset):
    """Map-style dataset yielding frames aligned with flow GT.

    __getitem__ returns:
      - seq_idx: int
      - frame_index: int  (flow index within the sequence)
      - t_start_us: int   (from_timestamp)
      - t_end_us: int     (to_timestamp)
      - t_ref_us: int     (= t_end_us, compat with mapper)
      - flow_png_path: str (absolute path to flow PNG)
    """

    def __init__(
        self,
        meta: DSECSequenceMeta,
        *,
        seq_idx: int,
        flow_direction: str = "forward",
        index_start: int = 0,
        index_count: Optional[int] = None,
        index_stride: int = 1,
    ):
        super().__init__()
        self.meta = meta
        self.seq_idx = int(seq_idx)
        self.sequence = meta.sequence
        self.flow_direction = str(flow_direction)

        flow_dir = meta.seq_path / "flow"
        if not flow_dir.is_dir():
            raise FileNotFoundError(
                f"Flow directory not found: {flow_dir} "
                f"(sequence '{meta.sequence}' may not have flow GT)"
            )

        ts_path = flow_dir / f"{flow_direction}_timestamps.txt"
        if not ts_path.exists():
            raise FileNotFoundError(f"Flow timestamps not found: {ts_path}")

        self._flow_timestamps = _read_flow_timestamps(ts_path)  # (N, 2)

        png_dir = flow_dir / flow_direction
        if not png_dir.is_dir():
            raise FileNotFoundError(f"Flow PNG directory not found: {png_dir}")

        self._flow_pngs = sorted(png_dir.glob("*.png"))
        if len(self._flow_pngs) != len(self._flow_timestamps):
            raise ValueError(
                f"Mismatch: {len(self._flow_pngs)} flow PNGs vs "
                f"{len(self._flow_timestamps)} flow timestamps "
                f"in {meta.sequence}/{flow_direction}"
            )
        self._selected_indices = self._build_selected_indices(
            total=len(self._flow_timestamps),
            index_start=index_start,
            index_count=index_count,
            index_stride=index_stride,
        )

    @staticmethod
    def _build_selected_indices(
        *,
        total: int,
        index_start: int,
        index_count: Optional[int],
        index_stride: int,
    ) -> np.ndarray:
        """Build the deterministic local flow-frame subset for one sequence."""
        start = int(index_start)
        stride = int(index_stride)
        if start < 0:
            raise ValueError(f"index_start must be non-negative, got {start}")
        if stride <= 0:
            raise ValueError(f"index_stride must be positive, got {stride}")
        if index_count is not None and int(index_count) < 0:
            raise ValueError(
                f"index_count must be non-negative or None, got {index_count}"
            )
        if start >= int(total):
            raise ValueError(
                f"index_start={start} is outside flow index length {total}"
            )

        selected = np.arange(start, int(total), stride, dtype=np.int64)
        if index_count is not None:
            selected = selected[: int(index_count)]
        if len(selected) == 0:
            raise ValueError(
                "DSEC-flow index subset is empty "
                f"(total={total}, index_start={start}, index_count={index_count}, "
                f"index_stride={stride})"
            )
        return selected

    def __len__(self) -> int:
        return len(self._selected_indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < 0:
            idx = len(self) + idx
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Flow index out of range: {idx}")

        source_idx = int(self._selected_indices[idx])
        t_from = int(self._flow_timestamps[source_idx, 0])
        t_to = int(self._flow_timestamps[source_idx, 1])

        return {
            "seq_idx": self.seq_idx,
            "frame_index": source_idx,
            "t_start_us": t_from,
            "t_end_us": t_to,
            "t_ref_us": t_to,
            "flow_png_path": str(self._flow_pngs[source_idx]),
        }

    def pull_item(self, idx: int) -> Dict[str, Any]:
        return self.__getitem__(idx)


def build_dsec_flow_index_datasets(
    *,
    dsec_yaml_path: str | Path,
    split: str = "train",
    sequences: Optional[Sequence[str]] = None,
    flow_direction: str = "forward",
    use_rectified_events: bool = False,
    rectified_events_filename: str = "rectified_events.h5",
    index_start: int = 0,
    index_count: Optional[int] = None,
    index_stride: int = 1,
) -> List[DSECFlowIndexDataset]:
    """Build per-sequence flow-aligned index datasets.

    Only sequences that have a ``flow/`` subdirectory are included.

    Args:
        dsec_yaml_path: Path to DSEC dataset YAML (e.g. optical_flow.yaml).
        split: Dataset split (default "train").
        sequences: Override sequence list (otherwise read from YAML).
        flow_direction: "forward" or "backward".
        use_rectified_events: Passed to meta construction.
        rectified_events_filename: Filename under events/<camera>/ to use when
                                   use_rectified_events is true.
        index_start: First local flow-frame index to expose for each sequence.
        index_count: Optional maximum number of local flow frames per sequence.
        index_stride: Local stride between exposed flow frames.
    """
    from smcfnet.dataset.dsec.dsec_sequence_index import build_dsec_sequence_metas

    metas = build_dsec_sequence_metas(
        dsec_yaml_path=dsec_yaml_path,
        split=split,
        sequences=sequences,
        use_rectified_events=use_rectified_events,
        rectified_events_filename=rectified_events_filename,
    )

    datasets: List[DSECFlowIndexDataset] = []
    for i, meta in enumerate(metas):
        flow_dir = meta.seq_path / "flow"
        if not flow_dir.is_dir():
            # Skip sequences without flow GT silently.
            continue
        datasets.append(
            DSECFlowIndexDataset(
                meta,
                seq_idx=i,
                flow_direction=flow_direction,
                index_start=index_start,
                index_count=index_count,
                index_stride=index_stride,
            )
        )

    if not datasets:
        raise ValueError(
            f"No sequences with flow GT found for split='{split}'. "
            f"Check that sequences listed in {dsec_yaml_path} have a flow/ directory."
        )

    return datasets
