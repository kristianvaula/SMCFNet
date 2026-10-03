"""
Precomputed optical-flow lookup for swapping MCM flow at the STTP head.

Stores E-RAFT (or any other) flow as per-sequence HDF5 files produced by
`tools/precompute_eraft_flow.py`. The STTP training path looks up flow by
`(sequence_name, frame_index)` and passes it through `model(..., flow=...)`,
which overrides the MCM's `_last_flow` at `yolox.py:134`.

Frame 0 of each sequence has no prior pair and is written as zeros (or
queried as `None`) because E-RAFT needs two consecutive voxels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch


class PrecomputedFlowProvider:
    """Lazy per-sequence HDF5 handle cache for precomputed optical flow.

    One instance per DataLoader worker. Handles open on first `get()` for a
    given sequence and stay open for the lifetime of the provider. Cross-file
    checkpoint-sha consistency is enforced: every sequence file must carry the
    same `checkpoint_sha` attr, otherwise we could silently mix flow from
    different estimator weights into the same training batch.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._handles: Dict[str, object] = {}
        self._checkpoint_sha: Optional[str] = None

    def get(self, sequence_name: str, frame_index: int) -> Optional[torch.Tensor]:
        """Return `(2, H, W)` fp32 flow tensor, or `None` for frame 0."""
        if frame_index == 0:
            return None
        if frame_index < 0:
            raise ValueError(f"frame_index must be >= 0, got {frame_index}")

        handle = self._open(sequence_name)
        dset = handle["flow"]
        n = dset.shape[0]
        if frame_index >= n:
            raise IndexError(
                f"frame_index {frame_index} out of range for sequence "
                f"'{sequence_name}' (frame_count={n})"
            )
        arr = dset[frame_index]
        return torch.from_numpy(np.asarray(arr)).float()

    def frame_count(self, sequence_name: str) -> int:
        handle = self._open(sequence_name)
        return int(handle["flow"].shape[0])

    def _open(self, sequence_name: str):
        if sequence_name in self._handles:
            return self._handles[sequence_name]

        import h5py  # local import mirrors dsec mapper pattern

        path = self.root / f"{sequence_name}.h5"
        if not path.exists():
            raise FileNotFoundError(
                f"No precomputed flow for sequence '{sequence_name}' at {path}"
            )
        handle = h5py.File(str(path), "r")

        sha = handle.attrs.get("checkpoint_sha", None)
        if isinstance(sha, bytes):
            sha = sha.decode("utf-8")
        if sha is None:
            handle.close()
            raise RuntimeError(
                f"Flow file {path} is missing required 'checkpoint_sha' attr"
            )
        if self._checkpoint_sha is None:
            self._checkpoint_sha = str(sha)
        elif str(sha) != self._checkpoint_sha:
            handle.close()
            raise RuntimeError(
                f"Checkpoint sha mismatch across flow files. "
                f"Sequence '{sequence_name}' has sha={sha!r} but earlier "
                f"files had sha={self._checkpoint_sha!r}. Refusing to mix "
                f"flow from different estimator weights."
            )

        self._handles[sequence_name] = handle
        return handle

    def close(self) -> None:
        for handle in self._handles.values():
            try:
                handle.close()
            except Exception:
                pass
        self._handles.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
