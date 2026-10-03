from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch


class BaseDSECPrefetcher(ABC):
    """Shared CUDA-prefetch abstraction for DSEC batch pipelines.

    Concrete subclasses decide what representation to prepare on-device:
    - packed events -> voxel grid
    - packed events only (streaming)
    - RGB-only batches
    """

    def __init__(self, loader, *, device: torch.device):
        self.device = device
        self._iter = iter(loader)
        self._stream: Optional[torch.cuda.Stream] = None
        if self.device.type == "cuda":
            self._stream = torch.cuda.Stream(device=self.device)

        self._next: Optional[Dict[str, Any]] = None
        self._preload()

    @property
    @abstractmethod
    def prepared_representation(self) -> str:
        """Name of the primary prepared representation produced by this prefetcher."""
        raise NotImplementedError

    @abstractmethod
    def _process(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Convert one host batch into the on-device representation."""
        raise NotImplementedError

    def _preload(self) -> None:
        try:
            batch = next(self._iter)
        except StopIteration:
            self._next = None
            return

        if self._stream is None:
            self._next = self._process(batch)
            return

        with torch.cuda.stream(self._stream):
            self._next = self._process(batch)

    def next(self) -> Dict[str, Any]:
        if self._next is None:
            raise StopIteration

        if self._stream is not None:
            torch.cuda.current_stream(device=self.device).wait_stream(self._stream)
            cur = torch.cuda.current_stream(device=self.device)
            for v in self._next.values():
                if isinstance(v, torch.Tensor) and v.is_cuda:
                    v.record_stream(cur)

        batch = self._next
        self._preload()
        return batch
