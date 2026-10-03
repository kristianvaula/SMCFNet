from __future__ import annotations

from typing import Any, Dict

import torch

from smcfnet.dataset.dsec.prefetcher.base import BaseDSECPrefetcher


class DSECStreamingGPUPrefetcher(BaseDSECPrefetcher):
    """CUDA prefetch stage for streaming packed-event batches.

    This path preserves the packed `events` contract, including microsecond
    timestamps, and only handles device transfer. Temporal partitioning remains
    the responsibility of SMCM/SOFE.
    """

    _tensor_keys = (
        "events",
        "events_batch_idx",
        "events_offsets",
        "seq_idx",
        "frame_index",
        "t_ref_us",
        "t_start_us",
        "t_end_us",
        "sequence_reset",
        "image",
        "targets",
        "flow_gt",
    )

    def __init__(
        self,
        loader,
        *,
        device: torch.device,
        require_targets: bool = True,
    ):
        self.require_targets = bool(require_targets)
        super().__init__(loader, device=device)

    @property
    def prepared_representation(self) -> str:
        return "streaming_events"

    def _process(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = dict(batch)

        if "events" not in out or not isinstance(out["events"], torch.Tensor):
            raise ValueError("Streaming prefetcher requires tensor batch['events']")
        if "events_offsets" not in out or not isinstance(out["events_offsets"], torch.Tensor):
            raise ValueError("Streaming prefetcher requires tensor batch['events_offsets']")

        for key in self._tensor_keys:
            value = out.get(key)
            if not isinstance(value, torch.Tensor):
                continue
            if key == "image":
                out[key] = value.to(self.device, dtype=torch.float32, non_blocking=True)
            elif key == "targets":
                out[key] = value.to(self.device, dtype=torch.float32, non_blocking=True)
            elif key == "flow_gt":
                out[key] = value.to(self.device, dtype=torch.float32, non_blocking=True)
            elif key == "events_offsets":
                out[key] = value.to(self.device, dtype=torch.int64, non_blocking=True)
            else:
                out[key] = value.to(self.device, non_blocking=True)

        if self.require_targets:
            if "targets" not in out or not isinstance(out["targets"], torch.Tensor):
                raise ValueError("require_targets=True but batch has no tensor 'targets'")

        return out
