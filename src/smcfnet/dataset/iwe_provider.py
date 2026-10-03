"""Precomputed detector-facing SMCM IWE lookup.

The cache contract mirrors :mod:`smcfnet.dataset.flow_provider`: one lazy HDF5
handle cache per DataLoader worker, indexed by ``(sequence_name, frame_index)``.
Each sequence file stores the exact detector-facing event tensor that live SMCM
would have produced before EDUM/backbone, normally ``iwe[N, 3, 480, 640]``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import torch


class PrecomputedIWEProvider:
    """Lazy per-sequence HDF5 provider for SMCM detector IWE caches."""

    def __init__(
        self,
        root: str | Path,
        *,
        dataset_key: str = "iwe",
        expected_channels: int = 3,
        checkpoint_sha: Optional[str] = None,
        architecture: Optional[str] = None,
        renderer_version: Optional[str] = None,
        sequence_names: Optional[Sequence[str]] = None,
        expected_frame_counts: Optional[Dict[str, int]] = None,
        expected_height: Optional[int] = None,
        expected_width: Optional[int] = None,
        exclude_sequences: Optional[Iterable[str]] = None,
    ) -> None:
        self.root = Path(root)
        self.dataset_key = str(dataset_key or "iwe")
        self.expected_channels = int(expected_channels)
        self.expected_checkpoint_sha = str(checkpoint_sha) if checkpoint_sha else None
        self.expected_architecture = str(architecture) if architecture else None
        self.expected_renderer_version = str(renderer_version) if renderer_version else None
        self.expected_frame_counts = {
            str(k): int(v) for k, v in (expected_frame_counts or {}).items()
        }
        self.expected_height = int(expected_height) if expected_height is not None else None
        self.expected_width = int(expected_width) if expected_width is not None else None
        self.exclude_sequences = {str(s) for s in (exclude_sequences or [])}
        self._handles: Dict[str, object] = {}
        self._checkpoint_sha: Optional[str] = None
        self._architecture: Optional[str] = None
        self._renderer_version: Optional[str] = None

        if self.expected_channels <= 0:
            raise ValueError(f"expected_channels must be > 0, got {self.expected_channels}")
        if sequence_names is not None:
            self.validate_required_sequences(sequence_names)

    def get(self, sequence_name: str, frame_index: int) -> torch.Tensor:
        """Return a ``[C,H,W]`` float tensor for one cached frame."""
        sequence_name = str(sequence_name)
        if sequence_name in self.exclude_sequences:
            raise FileNotFoundError(
                f"Sequence '{sequence_name}' is excluded from this IWE cache provider"
            )
        if int(frame_index) < 0:
            raise ValueError(f"frame_index must be >= 0, got {frame_index}")

        handle = self._open(sequence_name)
        dset = handle[self.dataset_key]
        n = int(dset.shape[0])
        if int(frame_index) >= n:
            raise IndexError(
                f"frame_index {frame_index} out of range for sequence "
                f"'{sequence_name}' (frame_count={n})"
            )

        arr = np.asarray(dset[int(frame_index)])
        scale = handle.attrs.get("scale", None)
        dtype_attr = handle.attrs.get("dtype", None)
        if isinstance(dtype_attr, bytes):
            dtype_attr = dtype_attr.decode("utf-8")

        if arr.ndim != 3:
            raise RuntimeError(
                f"IWE cache '{sequence_name}' frame {frame_index} has shape "
                f"{arr.shape}; expected [C,H,W]"
            )
        if int(arr.shape[0]) != self.expected_channels:
            raise RuntimeError(
                f"IWE cache '{sequence_name}' channel count {arr.shape[0]} does "
                f"not match expected {self.expected_channels}"
            )

        tensor = torch.from_numpy(arr)
        if tensor.dtype == torch.uint8 or str(dtype_attr).lower() == "uint8":
            denom = float(scale) if scale is not None else 255.0
            if denom <= 0.0:
                raise RuntimeError(f"Invalid uint8 IWE scale attr in '{sequence_name}': {scale}")
            tensor = tensor.to(torch.float32) / denom
        else:
            tensor = tensor.to(torch.float32)
        return tensor.contiguous()

    def frame_count(self, sequence_name: str) -> int:
        return int(self._open(sequence_name)[self.dataset_key].shape[0])

    def validate_required_sequences(self, sequence_names: Sequence[str]) -> None:
        required = [str(s) for s in sequence_names if str(s) not in self.exclude_sequences]
        missing = [s for s in required if not (self.root / f"{s}.h5").exists()]
        if missing:
            preview = ", ".join(missing[:10])
            suffix = "" if len(missing) <= 10 else f", ... ({len(missing)} total)"
            raise FileNotFoundError(
                f"Missing precomputed IWE cache files under {self.root}: {preview}{suffix}"
            )

    @staticmethod
    def _attr_str(handle, key: str) -> Optional[str]:
        value = handle.attrs.get(key, None)
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if value is None:
            return None
        return str(value)

    def _open(self, sequence_name: str):
        if sequence_name in self._handles:
            return self._handles[sequence_name]

        import h5py

        path = self.root / f"{sequence_name}.h5"
        if not path.exists():
            raise FileNotFoundError(f"No precomputed IWE for sequence '{sequence_name}' at {path}")
        handle = h5py.File(str(path), "r")
        if self.dataset_key not in handle:
            handle.close()
            raise RuntimeError(f"IWE cache {path} is missing dataset '{self.dataset_key}'")

        dset = handle[self.dataset_key]
        if len(dset.shape) != 4:
            handle.close()
            raise RuntimeError(
                f"IWE dataset {path}:{self.dataset_key} has shape {dset.shape}; "
                "expected [N,C,H,W]"
            )
        if int(dset.shape[1]) != self.expected_channels:
            handle.close()
            raise RuntimeError(
                f"IWE dataset {path}:{self.dataset_key} has {dset.shape[1]} channels; "
                f"expected {self.expected_channels}"
            )
        if self.expected_height is not None and int(dset.shape[2]) != self.expected_height:
            handle.close()
            raise RuntimeError(
                f"IWE dataset {path}:{self.dataset_key} has height {dset.shape[2]}; "
                f"expected {self.expected_height}"
            )
        if self.expected_width is not None and int(dset.shape[3]) != self.expected_width:
            handle.close()
            raise RuntimeError(
                f"IWE dataset {path}:{self.dataset_key} has width {dset.shape[3]}; "
                f"expected {self.expected_width}"
            )
        expected_frames = self.expected_frame_counts.get(sequence_name)
        if expected_frames is not None and int(dset.shape[0]) != int(expected_frames):
            handle.close()
            raise RuntimeError(
                f"IWE dataset {path}:{self.dataset_key} has {dset.shape[0]} frames; "
                f"training dataset expects {expected_frames}"
            )

        sha = self._attr_str(handle, "checkpoint_sha")
        if sha is None:
            handle.close()
            raise RuntimeError(f"IWE cache {path} missing required 'checkpoint_sha' attr")
        if self.expected_checkpoint_sha is not None and sha != self.expected_checkpoint_sha:
            handle.close()
            raise RuntimeError(
                f"IWE cache {path} checkpoint_sha={sha!r} does not match expected "
                f"{self.expected_checkpoint_sha!r}"
            )
        if self._checkpoint_sha is None:
            self._checkpoint_sha = sha
        elif sha != self._checkpoint_sha:
            handle.close()
            raise RuntimeError(
                f"Checkpoint sha mismatch across IWE caches: {sequence_name} has "
                f"{sha!r}, earlier files had {self._checkpoint_sha!r}"
            )

        arch = self._attr_str(handle, "architecture")
        if self.expected_architecture is not None and arch != self.expected_architecture:
            handle.close()
            raise RuntimeError(
                f"IWE cache {path} architecture={arch!r} does not match expected "
                f"{self.expected_architecture!r}"
            )
        if arch is not None:
            if self._architecture is None:
                self._architecture = arch
            elif arch != self._architecture:
                handle.close()
                raise RuntimeError(
                    f"Architecture mismatch across IWE caches: {sequence_name} has "
                    f"{arch!r}, earlier files had {self._architecture!r}"
                )

        renderer = self._attr_str(handle, "renderer_version")
        if self.expected_renderer_version is not None and renderer != self.expected_renderer_version:
            handle.close()
            raise RuntimeError(
                f"IWE cache {path} renderer_version={renderer!r} does not match "
                f"expected {self.expected_renderer_version!r}"
            )
        if renderer is not None:
            if self._renderer_version is None:
                self._renderer_version = renderer
            elif renderer != self._renderer_version:
                handle.close()
                raise RuntimeError(
                    f"Renderer mismatch across IWE caches: {sequence_name} has "
                    f"{renderer!r}, earlier files had {self._renderer_version!r}"
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
