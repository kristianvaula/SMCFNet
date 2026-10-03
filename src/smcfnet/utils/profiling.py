"""Lightweight CUDA stage timing helpers for opt-in training profiling."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterator, Optional, Tuple

import torch


CudaEventPair = Tuple[torch.cuda.Event, torch.cuda.Event]


class CudaStageTimer:
    """Collect named CUDA event pairs on the current stream.

    This is intended for short-lived, opt-in profiling during training. When
    disabled, it adds near-zero overhead and behaves as a no-op.
    """

    def __init__(self, enabled: bool) -> None:
        self.enabled = bool(enabled and torch.cuda.is_available())
        self.events: Dict[str, CudaEventPair] = {}

    @contextmanager
    def section(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            self.events[str(name)] = (start, end)


def collect_cuda_stage_times_ms(
    events: Optional[Dict[str, CudaEventPair]],
    *,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Synchronize once and convert CUDA event pairs into millisecond scalars."""

    if not events:
        return {}

    if device is not None and getattr(device, "type", None) == "cuda":
        torch.cuda.synchronize(device)
    else:
        torch.cuda.synchronize()

    return {
        str(name): float(start.elapsed_time(end))
        for name, (start, end) in events.items()
    }
