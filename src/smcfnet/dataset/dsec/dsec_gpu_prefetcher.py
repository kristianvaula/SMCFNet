
from typing import Any, Dict, List, Optional, Sequence

import torch

from smcfnet.utils.event_utils import events_to_voxel_grid_torch


class DSECGPUPrefetcher:
    """CUDA prefetch + GPU voxelization stage for the new packed-events loader.

    Expects batches with:
      - events: (N,4) int tensor [x,y,t_ms,p]
      - events_offsets: (B+1,) int64
    Optionally:
      - image: (B,3,H,W) float tensor
      - targets: (B,M,5) float tensor

    Produces:
      - events_voxel: (B,bins,H,W) float32 on `device`
      - image/targets moved to `device` when present
      - precomputed_flow: (B,2,H,W) float32 on `device` when a flow_provider
        is supplied. Frame 0 of each sequence contributes zeros.

    Notes:
      - Voxelization runs in the prefetch CUDA stream.
      - This keeps Trainer.run_step() free of CPU voxelization.
    """

    def __init__(
        self,
        loader,
        *,
        device: torch.device,
        num_bins: int,
        height: int,
        width: int,
        require_targets: bool = True,
        flow_provider: Optional[Any] = None,
        sequence_names: Optional[Sequence[str]] = None,
    ):
        self.device = device
        self.num_bins = int(num_bins)
        self.height = int(height)
        self.width = int(width)
        self.require_targets = bool(require_targets)
        self._flow_provider = flow_provider
        self._sequence_names: Optional[List[str]] = (
            list(sequence_names) if sequence_names is not None else None
        )
        if self._flow_provider is not None and self._sequence_names is None:
            raise ValueError(
                "flow_provider requires sequence_names to map seq_idx -> name"
            )

        self._iter = iter(loader)
        self._stream: Optional[torch.cuda.Stream] = None
        if self.device.type == "cuda":
            self._stream = torch.cuda.Stream(device=self.device)

        self._next: Optional[Dict[str, Any]] = None
        self._preload()

    def _voxelize_packed(self, events_packed: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        bsz = int(offsets.numel() - 1)
        voxels = []
        for i in range(bsz):
            start = int(offsets[i].item())
            end = int(offsets[i + 1].item())
            ev_i = events_packed[start:end]
            vox_i = events_to_voxel_grid_torch(
                ev_i,
                num_bins=self.num_bins,
                width=self.width,
                height=self.height,
            )
            voxels.append(vox_i)
        if len(voxels) == 0:
            return torch.zeros((0, self.num_bins, self.height, self.width), device=self.device, dtype=torch.float32)
        return torch.stack(voxels, dim=0)

    def _process(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = dict(batch)

        events = out.get("events")
        offsets = out.get("events_offsets")
        if events is None or offsets is None:
            raise ValueError("Batch missing required keys: events/events_offsets")

        if not isinstance(events, torch.Tensor):
            events = torch.as_tensor(events)
        if not isinstance(offsets, torch.Tensor):
            offsets = torch.as_tensor(offsets)

        events = events.to(self.device, non_blocking=True)
        offsets = offsets.to(torch.int64)
        offsets = offsets.to(self.device, non_blocking=True)

        out["events"] = events
        out["events_offsets"] = offsets

        if "image" in out and isinstance(out["image"], torch.Tensor):
            out["image"] = out["image"].to(self.device, non_blocking=True)

        if self.require_targets:
            if "targets" not in out or not isinstance(out["targets"], torch.Tensor):
                raise ValueError("require_targets=True but batch has no tensor 'targets'")
            out["targets"] = out["targets"].to(self.device, non_blocking=True)
        else:
            if "targets" in out and isinstance(out["targets"], torch.Tensor):
                out["targets"] = out["targets"].to(self.device, non_blocking=True)

        # Future targets for STTP (optional)
        if "future_targets" in out and isinstance(out["future_targets"], dict):
            ft = out["future_targets"]
            if isinstance(ft.get("boxes"), torch.Tensor):
                ft["boxes"] = ft["boxes"].to(self.device, non_blocking=True)
            if isinstance(ft.get("validity"), torch.Tensor):
                ft["validity"] = ft["validity"].to(self.device, non_blocking=True)
            if isinstance(ft.get("current_boxes"), torch.Tensor):
                ft["current_boxes"] = ft["current_boxes"].to(self.device, non_blocking=True)

        # Flow ground truth (optional)
        if "flow_gt" in out and isinstance(out["flow_gt"], torch.Tensor):
            out["flow_gt"] = out["flow_gt"].to(self.device, non_blocking=True)

        out["events_voxel"] = self._voxelize_packed(events, offsets)

        # Precomputed optical flow (optional, e.g. E-RAFT). Keys: seq_idx,
        # frame_index already present from the mapper/collate. Each sample's
        # frame_index=0 returns None from the provider and is substituted
        # with zeros here (no prior pair exists for E-RAFT warm-start).
        if self._flow_provider is not None:
            out["precomputed_flow"] = self._lookup_flow_batch(out)

        return out

    def _lookup_flow_batch(self, batch: Dict[str, Any]) -> torch.Tensor:
        assert self._sequence_names is not None  # guarded in __init__
        seq_idx = batch.get("seq_idx")
        frame_idx = batch.get("frame_index")
        if seq_idx is None or frame_idx is None:
            raise ValueError(
                "precomputed flow requires seq_idx + frame_index in batch"
            )
        if isinstance(seq_idx, torch.Tensor):
            seq_idx_list = seq_idx.tolist()
        else:
            seq_idx_list = list(seq_idx)
        if isinstance(frame_idx, torch.Tensor):
            frame_idx_list = frame_idx.tolist()
        else:
            frame_idx_list = list(frame_idx)

        flows = []
        zero_flow: Optional[torch.Tensor] = None
        for si, fi in zip(seq_idx_list, frame_idx_list):
            name = self._sequence_names[int(si)]
            flow = self._flow_provider.get(name, int(fi))
            if flow is None:
                if zero_flow is None:
                    zero_flow = torch.zeros(
                        2, self.height, self.width, dtype=torch.float32
                    )
                flow = zero_flow
            flows.append(flow)

        stacked = torch.stack(flows, dim=0)
        return stacked.to(self.device, non_blocking=True)

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
            # Ensure lifetime across streams.
            cur = torch.cuda.current_stream(device=self.device)
            for v in self._next.values():
                if isinstance(v, torch.Tensor) and v.is_cuda:
                    v.record_stream(cur)

        batch = self._next
        self._preload()
        return batch
