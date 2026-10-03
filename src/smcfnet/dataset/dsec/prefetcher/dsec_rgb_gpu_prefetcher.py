from __future__ import annotations

from typing import Any, Dict

import torch

from smcfnet.dataset.dsec.prefetcher.base import BaseDSECPrefetcher


class DSECRGBGPUPrefetcher(BaseDSECPrefetcher):
    """CUDA prefetch stage for RGB-only DSEC batches."""

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
        return "image"

    def _process(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = dict(batch)

        if "image" not in out or not isinstance(out["image"], torch.Tensor):
            raise ValueError("RGB prefetcher requires tensor batch['image']")
        image = out["image"]
        if image.dtype != torch.float32:
            image = image.to(self.device, dtype=torch.float32, non_blocking=True)
        else:
            image = image.to(self.device, non_blocking=True)
        out["image"] = image

        if self.require_targets:
            if "targets" not in out or not isinstance(out["targets"], torch.Tensor):
                raise ValueError("require_targets=True but batch has no tensor 'targets'")
            out["targets"] = out["targets"].to(self.device, non_blocking=True)
        elif "targets" in out and isinstance(out["targets"], torch.Tensor):
            out["targets"] = out["targets"].to(self.device, non_blocking=True)

        return out
