"""Precomputed per-frame stem-feature window lookup for the recurrent STTP head.

Mirrors PrecomputedFlowProvider: per-sequence HDF5 files (one dataset per FPN
scale, `p0`/`p1`/`p2`, each [n_frames, C, H, W] fp16) produced by
`scripts/precompute_stem_features.py`. Enforces a shared `checkpoint_sha`
across files so windows can never mix features from different detectors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch


class StemFeatureProvider:
    def __init__(self, root, context_window: int, num_scales: int = 3) -> None:
        if context_window < 1:
            raise ValueError(f"context_window must be >= 1, got {context_window}")
        self.root = Path(root)
        self.context_window = int(context_window)
        self.num_scales = int(num_scales)
        self._handles: Dict[str, object] = {}
        self._checkpoint_sha: Optional[str] = None

    def get_window(self, sequence_name: str, frame_index: int) -> List[torch.Tensor]:
        """Return num_scales tensors, each [context_window, C, H, W] (fp32).

        Indices are clamped to >= 0: at a sequence start the earliest frame is
        repeated (left-pad), so the LIF integrates fewer distinct steps with no
        special-casing in the head.
        """
        if frame_index < 0:
            raise ValueError(f"frame_index must be >= 0, got {frame_index}")
        handle = self._open(sequence_name)
        idxs = [max(0, frame_index - (self.context_window - 1) + j) for j in range(self.context_window)]
        out: List[torch.Tensor] = []
        for s in range(self.num_scales):
            dset = handle[f"p{s}"]
            n = dset.shape[0]
            if frame_index >= n:
                raise IndexError(
                    f"frame_index {frame_index} out of range for '{sequence_name}' (frame_count={n})"
                )
            frames = np.stack([np.asarray(dset[i]) for i in idxs])  # [K, C, H, W]
            out.append(torch.from_numpy(frames).float())
        return out

    def frame_count(self, sequence_name: str) -> int:
        return int(self._open(sequence_name)["p0"].shape[0])

    def _open(self, sequence_name: str):
        if sequence_name in self._handles:
            return self._handles[sequence_name]
        import h5py

        path = self.root / f"{sequence_name}.h5"
        if not path.exists():
            raise FileNotFoundError(f"No stem-feature cache for '{sequence_name}' at {path}")
        handle = h5py.File(str(path), "r")
        sha = handle.attrs.get("checkpoint_sha", None)
        if isinstance(sha, bytes):
            sha = sha.decode("utf-8")
        if sha is None:
            handle.close()
            raise RuntimeError(f"Stem cache {path} missing required 'checkpoint_sha' attr")
        if self._checkpoint_sha is None:
            self._checkpoint_sha = str(sha)
        elif str(sha) != self._checkpoint_sha:
            handle.close()
            raise RuntimeError(
                f"Checkpoint sha mismatch across stem caches. '{sequence_name}' has "
                f"sha={sha!r} but earlier files had sha={self._checkpoint_sha!r}. "
                "Refusing to mix features from different detectors."
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
