from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from smcfnet.dataset.dsec.prefetcher.base import BaseDSECPrefetcher
from smcfnet.utils.event_utils import events_to_cuadrado_histogram_torch, events_to_voxel_grid_torch
from smcfnet.utils.profiling import CudaStageTimer


class DSECGPUPrefetcher(BaseDSECPrefetcher):
    """CUDA prefetch + GPU voxelization stage for packed-event batches.

    When a ``flow_provider`` is supplied alongside ``sequence_names``, each
    processed batch gains a ``precomputed_flow`` key of shape ``(B, 2, H, W)``
    on ``device``. Frames where the provider returns ``None`` (e.g. frame 0
    of a sequence, which has no prior pair) contribute zeros.
    """

    def __init__(
        self,
        loader,
        *,
        device: torch.device,
        num_bins: int,
        height: int,
        width: int,
        source_height: int | None = None,
        source_width: int | None = None,
        require_targets: bool = True,
        augmentation_backend: str = "none",
        use_mosaic: bool = False,
        use_mixup: bool = False,
        mosaic_prob: float = 1.0,
        mixup_prob: float = 0.5,
        num_classes: int = 0,
        estimator_representation: str = "voxel",
        output_representation: str = "voxel",
        output_channels: int | None = None,
        build_voxel: bool = True,
        profile_stages: bool = False,
        flow_provider: Optional[Any] = None,
        sequence_names: Optional[Sequence[Optional[str]]] = None,
        stem_provider: Optional[Any] = None,
        stem_sequence_names: Optional[Sequence[Optional[str]]] = None,
        iwe_provider: Optional[Any] = None,
        iwe_sequence_names: Optional[Sequence[Optional[str]]] = None,
    ):
        self.num_bins = int(num_bins)
        self.height = int(height)
        self.width = int(width)
        self.source_height = int(source_height) if source_height is not None else self.height
        self.source_width = int(source_width) if source_width is not None else self.width
        self.require_targets = bool(require_targets)
        self.augmentation_backend = str(augmentation_backend or "none").lower()
        self.use_mosaic = bool(use_mosaic)
        self.use_mixup = bool(use_mixup)
        self.mosaic_prob = float(mosaic_prob)
        self.mixup_prob = float(mixup_prob)
        self.num_classes = int(num_classes)
        self.estimator_representation = str(estimator_representation or "voxel").lower()
        self.output_representation = str(output_representation or "voxel").lower()
        self.output_channels = int(output_channels) if output_channels is not None else self.num_bins
        self.build_voxel = bool(build_voxel)
        self.profile_stages = bool(profile_stages)
        if (
            self.estimator_representation == "cuadrado_histogram"
            and self.augmentation_backend == "gpu_voxel"
            and (self.use_mosaic or self.use_mixup)
        ):
            raise ValueError(
                "estimator_representation='cuadrado_histogram' cannot be combined with GPU voxel mosaic/mixup: "
                "the estimator histogram would no longer match the augmented event voxel. "
                "Disable mosaic/mixup or the native histogram path for this run."
            )
        self._flow_provider = flow_provider
        self._sequence_names: Optional[List[Optional[str]]] = (
            list(sequence_names) if sequence_names is not None else None
        )
        if self._flow_provider is not None and self._sequence_names is None:
            raise ValueError(
                "flow_provider requires sequence_names to map seq_idx -> name"
            )
        self._stem_provider = stem_provider
        self._stem_sequence_names: Optional[List[Optional[str]]] = (
            list(stem_sequence_names) if stem_sequence_names is not None else None
        )
        if self._stem_provider is not None and self._stem_sequence_names is None:
            raise ValueError(
                "stem_provider requires stem_sequence_names to map seq_idx -> name"
            )
        self._iwe_provider = iwe_provider
        self._iwe_sequence_names: Optional[List[Optional[str]]] = (
            list(iwe_sequence_names) if iwe_sequence_names is not None else None
        )
        if self._iwe_provider is not None and self._iwe_sequence_names is None:
            raise ValueError(
                "iwe_provider requires iwe_sequence_names to map seq_idx -> name"
            )
        super().__init__(loader, device=device)

    @property
    def prepared_representation(self) -> str:
        return "events_voxel"

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
                source_width=self.source_width,
                source_height=self.source_height,
            )
            voxels.append(vox_i)
        if len(voxels) == 0:
            return torch.zeros(
                (0, self.num_bins, self.height, self.width),
                device=self.device,
                dtype=torch.float32,
            )
        return torch.stack(voxels, dim=0)

    @staticmethod
    def _time_window_in_event_units(
        events_i: torch.Tensor,
        t_start_us_i: torch.Tensor | None,
        t_end_us_i: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if t_start_us_i is None or t_end_us_i is None or events_i.numel() == 0:
            return None, None
        t0_us = t_start_us_i.to(device=events_i.device, dtype=torch.float32)
        t1_us = t_end_us_i.to(device=events_i.device, dtype=torch.float32)
        ts = events_i[:, 2].to(torch.float32)
        # Voxel-mode DSEC batches currently carry millisecond event timestamps,
        # while metadata remains in microseconds. Detect that scale mismatch and
        # convert the window bounds instead of changing the packed event contract.
        window_scale = max(float(t0_us.abs()), float(t1_us.abs()))
        if window_scale > 0.0 and float(ts.abs().median()) < window_scale * 0.1:
            return t0_us / 1000.0, t1_us / 1000.0
        return t0_us, t1_us

    def _cuadrado_histogram_packed(
        self,
        events_packed: torch.Tensor,
        offsets: torch.Tensor,
        t_start_us: torch.Tensor | None = None,
        t_end_us: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bins = self.num_bins
        if self.estimator_representation != "cuadrado_histogram":
            raise RuntimeError("cuadrado histogram estimator representation is not enabled")
        bsz = int(offsets.numel() - 1)
        histograms = []
        for i in range(bsz):
            start = int(offsets[i].item())
            end = int(offsets[i + 1].item())
            ev_i = events_packed[start:end]
            t0 = t_start_us[i] if isinstance(t_start_us, torch.Tensor) else None
            t1 = t_end_us[i] if isinstance(t_end_us, torch.Tensor) else None
            t0, t1 = self._time_window_in_event_units(ev_i, t0, t1)
            hist_i = events_to_cuadrado_histogram_torch(
                ev_i,
                num_bins=bins,
                width=self.width,
                height=self.height,
                t_start=t0,
                t_end=t1,
                source_width=self.source_width,
                source_height=self.source_height,
            )
            histograms.append(hist_i)
        if len(histograms) == 0:
            return torch.zeros((0, 2, bins, self.height, self.width), device=self.device, dtype=torch.float32)
        return torch.stack(histograms, dim=0)

    def _filter_targets(self, targets_i: torch.Tensor) -> torch.Tensor:
        if not isinstance(targets_i, torch.Tensor) or targets_i.numel() == 0:
            return torch.zeros((0, 5), dtype=torch.float32, device=self.device)
        tt = targets_i.reshape(-1, 5).to(dtype=torch.float32, device=self.device)
        valid = torch.isfinite(tt).all(dim=1) & (tt[:, 3] > 1e-6) & (tt[:, 4] > 1e-6)
        if self.num_classes > 0:
            valid = valid & (tt[:, 0] >= 0) & (tt[:, 0] < float(self.num_classes))
        if not torch.any(valid):
            return torch.zeros((0, 5), dtype=torch.float32, device=self.device)
        return tt[valid].contiguous()

    def _pad_targets(self, per_sample: List[torch.Tensor]) -> torch.Tensor:
        if len(per_sample) == 0:
            return torch.zeros((0, 1, 5), dtype=torch.float32, device=self.device)
        max_num = max((int(t.shape[0]) for t in per_sample), default=0)
        if max_num <= 0:
            max_num = 1
        out = torch.zeros((len(per_sample), max_num, 5), dtype=torch.float32, device=self.device)
        for i, t in enumerate(per_sample):
            if not isinstance(t, torch.Tensor) or t.numel() == 0:
                continue
            n = min(int(t.shape[0]), max_num)
            out[i, :n].copy_(t[:n].to(dtype=torch.float32, device=self.device))
        return out

    @staticmethod
    def _resize_image(image: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
        if int(image.shape[-2]) == int(out_h) and int(image.shape[-1]) == int(out_w):
            return image.contiguous()
        img = F.interpolate(
            image.unsqueeze(0),
            size=(int(out_h), int(out_w)),
            mode="bilinear",
            align_corners=False,
        )
        return img.squeeze(0).contiguous()

    @staticmethod
    def _resize_voxel_preserve_mass(voxel: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
        in_h = int(voxel.shape[-2])
        in_w = int(voxel.shape[-1])
        if in_h == int(out_h) and in_w == int(out_w):
            return voxel.contiguous()
        vox = F.interpolate(
            voxel.unsqueeze(0),
            size=(int(out_h), int(out_w)),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        area_ratio = float(int(out_h) * int(out_w)) / float(max(in_h * in_w, 1))
        if area_ratio > 0.0:
            vox = vox / area_ratio
        return vox.contiguous()

    def _transform_targets_to_canvas(
        self,
        targets: torch.Tensor,
        *,
        src_h: int,
        src_w: int,
        dst_h: int,
        dst_w: int,
        offset_y: int,
        offset_x: int,
        canvas_h: int,
        canvas_w: int,
    ) -> torch.Tensor:
        if not isinstance(targets, torch.Tensor) or targets.numel() == 0:
            return torch.zeros((0, 5), dtype=torch.float32, device=self.device)

        tt = targets.reshape(-1, 5).to(dtype=torch.float32, device=self.device).clone()
        sx = float(dst_w) / float(max(int(src_w), 1))
        sy = float(dst_h) / float(max(int(src_h), 1))

        cx = tt[:, 1] * sx + float(offset_x)
        cy = tt[:, 2] * sy + float(offset_y)
        w = tt[:, 3] * sx
        h = tt[:, 4] * sy

        x1 = torch.clamp(cx - w / 2.0, 0.0, float(canvas_w))
        y1 = torch.clamp(cy - h / 2.0, 0.0, float(canvas_h))
        x2 = torch.clamp(cx + w / 2.0, 0.0, float(canvas_w))
        y2 = torch.clamp(cy + h / 2.0, 0.0, float(canvas_h))

        new_w = x2 - x1
        new_h = y2 - y1
        valid = torch.isfinite(tt).all(dim=1) & (new_w > 1e-3) & (new_h > 1e-3)
        if self.num_classes > 0:
            valid = valid & (tt[:, 0] >= 0) & (tt[:, 0] < float(self.num_classes))
        if not torch.any(valid):
            return torch.zeros((0, 5), dtype=torch.float32, device=self.device)

        out = torch.zeros((int(valid.sum().item()), 5), dtype=torch.float32, device=self.device)
        out[:, 0] = tt[valid, 0]
        out[:, 1] = x1[valid] + new_w[valid] / 2.0
        out[:, 2] = y1[valid] + new_h[valid] / 2.0
        out[:, 3] = new_w[valid]
        out[:, 4] = new_h[valid]
        return out.contiguous()

    def _compose_mosaic_sample(
        self,
        images: torch.Tensor,
        voxels: torch.Tensor,
        targets_list: List[torch.Tensor],
        source_indices: List[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img_h, img_w = int(images.shape[-2]), int(images.shape[-1])
        vox_h, vox_w = int(voxels.shape[-2]), int(voxels.shape[-1])
        half_img_h = int(img_h // 2)
        half_img_w = int(img_w // 2)
        half_vox_h = int(vox_h // 2)
        half_vox_w = int(vox_w // 2)

        img_placements = [
            (0, 0, half_img_h, half_img_w),
            (0, half_img_w, half_img_h, int(img_w - half_img_w)),
            (half_img_h, 0, int(img_h - half_img_h), half_img_w),
            (half_img_h, half_img_w, int(img_h - half_img_h), int(img_w - half_img_w)),
        ]
        vox_placements = [
            (0, 0, half_vox_h, half_vox_w),
            (0, half_vox_w, half_vox_h, int(vox_w - half_vox_w)),
            (half_vox_h, 0, int(vox_h - half_vox_h), half_vox_w),
            (half_vox_h, half_vox_w, int(vox_h - half_vox_h), int(vox_w - half_vox_w)),
        ]

        image_canvas = torch.zeros_like(images[0])
        voxel_canvas = torch.zeros_like(voxels[0])
        target_parts: List[torch.Tensor] = []

        for src_idx, img_place, vox_place in zip(source_indices[:4], img_placements, vox_placements):
            img_off_y, img_off_x, img_dst_h, img_dst_w = img_place
            vox_off_y, vox_off_x, vox_dst_h, vox_dst_w = vox_place

            resized_img = self._resize_image(images[int(src_idx)], img_dst_h, img_dst_w)
            image_canvas[:, img_off_y : img_off_y + img_dst_h, img_off_x : img_off_x + img_dst_w] = resized_img

            resized_vox = self._resize_voxel_preserve_mass(voxels[int(src_idx)], vox_dst_h, vox_dst_w)
            voxel_canvas[:, vox_off_y : vox_off_y + vox_dst_h, vox_off_x : vox_off_x + vox_dst_w] = resized_vox

            tg = self._transform_targets_to_canvas(
                targets_list[int(src_idx)],
                src_h=img_h,
                src_w=img_w,
                dst_h=img_dst_h,
                dst_w=img_dst_w,
                offset_y=img_off_y,
                offset_x=img_off_x,
                canvas_h=img_h,
                canvas_w=img_w,
            )
            if tg.numel() > 0:
                target_parts.append(tg)

        targets = (
            torch.cat(target_parts, dim=0).contiguous()
            if len(target_parts) > 0
            else torch.zeros((0, 5), dtype=torch.float32, device=self.device)
        )
        return image_canvas.contiguous(), voxel_canvas.contiguous(), targets

    def _compose_mixup_sample(
        self,
        base_image: torch.Tensor,
        base_voxel: torch.Tensor,
        base_targets: torch.Tensor,
        *,
        partner_idx: int,
        images: torch.Tensor,
        voxels: torch.Tensor,
        targets_list: List[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        partner_image = self._resize_image(images[int(partner_idx)], int(base_image.shape[-2]), int(base_image.shape[-1]))
        partner_voxel = self._resize_voxel_preserve_mass(
            voxels[int(partner_idx)],
            int(base_voxel.shape[-2]),
            int(base_voxel.shape[-1]),
        )
        mixed_image = (0.5 * base_image + 0.5 * partner_image).contiguous()
        mixed_voxel = (base_voxel + partner_voxel).contiguous()

        parts = [tg for tg in (base_targets, targets_list[int(partner_idx)]) if isinstance(tg, torch.Tensor) and tg.numel() > 0]
        mixed_targets = (
            torch.cat(parts, dim=0).contiguous()
            if len(parts) > 0
            else torch.zeros((0, 5), dtype=torch.float32, device=self.device)
        )
        return mixed_image, mixed_voxel, mixed_targets

    def _apply_gpu_voxel_augmentation(self, out: Dict[str, Any]) -> Dict[str, Any]:
        if not (self.use_mosaic or self.use_mixup):
            return out
        if "image" not in out or "targets" not in out or "events_voxel" not in out:
            return out

        images = out["image"]
        voxels = out["events_voxel"]
        targets = out["targets"]
        if not (isinstance(images, torch.Tensor) and isinstance(voxels, torch.Tensor) and isinstance(targets, torch.Tensor)):
            return out

        batch_size = int(images.shape[0])
        if batch_size <= 0:
            return out

        targets_list = [self._filter_targets(targets[i]) for i in range(batch_size)]
        aug_images: List[torch.Tensor] = []
        aug_voxels: List[torch.Tensor] = []
        aug_targets: List[torch.Tensor] = []

        for anchor_idx in range(batch_size):
            aug_image = images[anchor_idx]
            aug_voxel = voxels[anchor_idx]
            aug_target = targets_list[anchor_idx]

            if self.use_mosaic and float(torch.rand(1).item()) < self.mosaic_prob:
                indices = [anchor_idx]
                if batch_size > 1:
                    indices.extend(int(idx) for idx in torch.randint(0, batch_size, (3,), dtype=torch.int64).tolist())
                else:
                    indices.extend([anchor_idx, anchor_idx, anchor_idx])
                aug_image, aug_voxel, aug_target = self._compose_mosaic_sample(
                    images,
                    voxels,
                    targets_list,
                    indices,
                )

            if self.use_mixup and batch_size > 1 and float(torch.rand(1).item()) < self.mixup_prob:
                partner_idx = int(torch.randint(0, batch_size - 1, (1,), dtype=torch.int64).item())
                if partner_idx >= anchor_idx:
                    partner_idx += 1
                aug_image, aug_voxel, aug_target = self._compose_mixup_sample(
                    aug_image,
                    aug_voxel,
                    aug_target,
                    partner_idx=partner_idx,
                    images=images,
                    voxels=voxels,
                    targets_list=targets_list,
                )

            aug_images.append(aug_image.contiguous())
            aug_voxels.append(aug_voxel.contiguous())
            aug_targets.append(aug_target.contiguous())

        out["image"] = torch.stack(aug_images, dim=0)
        out["events_voxel"] = torch.stack(aug_voxels, dim=0)
        out["targets"] = self._pad_targets(aug_targets)
        return out

    def _process(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = dict(batch)
        stage_timer = CudaStageTimer(enabled=self.profile_stages and self.device.type == "cuda")

        events = out.get("events")
        offsets = out.get("events_offsets")
        if (events is None or offsets is None) and self._iwe_provider is None:
            raise ValueError("Batch missing required keys: events/events_offsets")

        if events is not None and not isinstance(events, torch.Tensor):
            events = torch.as_tensor(events)
        if offsets is not None and not isinstance(offsets, torch.Tensor):
            offsets = torch.as_tensor(offsets)

        with stage_timer.section("prefetch_h2d"):
            if isinstance(events, torch.Tensor):
                events = events.to(self.device, non_blocking=True)
                out["events"] = events
            if isinstance(offsets, torch.Tensor):
                offsets = offsets.to(torch.int64)
                offsets = offsets.to(self.device, non_blocking=True)
                out["events_offsets"] = offsets

            if "image" in out and isinstance(out["image"], torch.Tensor):
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
            else:
                if "targets" in out and isinstance(out["targets"], torch.Tensor):
                    out["targets"] = out["targets"].to(self.device, non_blocking=True)

            if "flow_gt" in out and isinstance(out["flow_gt"], torch.Tensor):
                out["flow_gt"] = out["flow_gt"].to(self.device, non_blocking=True)

        if "future_targets" in out and isinstance(out["future_targets"], dict):
            ft = out["future_targets"]
            for key in ("boxes", "validity", "current_boxes"):
                if isinstance(ft.get(key), torch.Tensor):
                    ft[key] = ft[key].to(self.device, non_blocking=True)

        with stage_timer.section("prefetch_voxel"):
            if self._iwe_provider is not None:
                out["events_voxel"] = self._lookup_iwe_batch(out)
            elif self.build_voxel:
                assert isinstance(events, torch.Tensor)
                assert isinstance(offsets, torch.Tensor)
                out["events_voxel"] = self._voxelize_packed(events, offsets)
            else:
                bsz = int(offsets.numel() - 1) if isinstance(offsets, torch.Tensor) else int(out["seq_idx"].numel())
                out["events_voxel"] = torch.zeros(
                    (bsz, self.output_channels, self.height, self.width),
                    device=self.device,
                    dtype=torch.float32,
                )
        if self.estimator_representation == "cuadrado_histogram":
            if not isinstance(events, torch.Tensor) or not isinstance(offsets, torch.Tensor):
                raise ValueError("cuadrado_histogram representation requires raw packed events")
            with stage_timer.section("prefetch_cuadrado_hist"):
                t_start_us = out.get("t_start_us")
                t_end_us = out.get("t_end_us")
                if isinstance(t_start_us, torch.Tensor):
                    t_start_us = t_start_us.to(self.device, non_blocking=True)
                    out["t_start_us"] = t_start_us
                if isinstance(t_end_us, torch.Tensor):
                    t_end_us = t_end_us.to(self.device, non_blocking=True)
                    out["t_end_us"] = t_end_us
                out["estimator_input"] = self._cuadrado_histogram_packed(events, offsets, t_start_us, t_end_us)
        if self.augmentation_backend == "gpu_voxel":
            with stage_timer.section("prefetch_gpu_aug"):
                out = self._apply_gpu_voxel_augmentation(out)

        if self._flow_provider is not None:
            out["precomputed_flow"] = self._lookup_flow_batch(out)

        if self._stem_provider is not None:
            out["stem_window"] = self._lookup_stem_window_batch(out)

        if stage_timer.enabled:
            out["_profile_events"] = dict(stage_timer.events)
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

        flows: List[torch.Tensor] = []
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

    def _lookup_iwe_batch(self, batch: Dict[str, Any]) -> torch.Tensor:
        assert self._iwe_sequence_names is not None  # guarded in __init__
        seq_idx = batch.get("seq_idx")
        frame_idx = batch.get("frame_index")
        if seq_idx is None or frame_idx is None:
            raise ValueError("precomputed IWE requires seq_idx + frame_index in batch")
        if isinstance(seq_idx, torch.Tensor):
            seq_idx_list = seq_idx.tolist()
        else:
            seq_idx_list = list(seq_idx)
        if isinstance(frame_idx, torch.Tensor):
            frame_idx_list = frame_idx.tolist()
        else:
            frame_idx_list = list(frame_idx)

        iwes: List[torch.Tensor] = []
        for si, fi in zip(seq_idx_list, frame_idx_list):
            name = self._iwe_sequence_names[int(si)]
            if name is None:
                raise RuntimeError(f"No sequence name for seq_idx={int(si)} in IWE lookup")
            iwe = self._iwe_provider.get(str(name), int(fi))
            if int(iwe.shape[0]) != int(self.output_channels):
                raise RuntimeError(
                    f"IWE cache for sequence '{name}' frame {fi} has {iwe.shape[0]} "
                    f"channels; expected {self.output_channels}"
                )
            if int(iwe.shape[-2]) != int(self.height) or int(iwe.shape[-1]) != int(self.width):
                raise RuntimeError(
                    f"IWE cache for sequence '{name}' frame {fi} has spatial shape "
                    f"{tuple(iwe.shape[-2:])}; expected {(self.height, self.width)}"
                )
            iwes.append(iwe)

        stacked = torch.stack(iwes, dim=0)
        return stacked.to(self.device, non_blocking=True)

    def _lookup_stem_window_batch(self, batch: Dict[str, Any]) -> List[torch.Tensor]:
        assert self._stem_sequence_names is not None  # guarded in __init__
        seq_idx = batch.get("seq_idx")
        frame_idx = batch.get("frame_index")
        if seq_idx is None or frame_idx is None:
            raise ValueError(
                "stem_window lookup requires seq_idx + frame_index in batch"
            )
        if isinstance(seq_idx, torch.Tensor):
            seq_idx_list = seq_idx.tolist()
        else:
            seq_idx_list = list(seq_idx)
        if isinstance(frame_idx, torch.Tensor):
            frame_idx_list = frame_idx.tolist()
        else:
            frame_idx_list = list(frame_idx)

        per_sample_windows: List[List[torch.Tensor]] = []
        for si, fi in zip(seq_idx_list, frame_idx_list):
            name = self._stem_sequence_names[int(si)]
            if name is not None:
                window = self._stem_provider.get_window(name, int(fi))
            else:
                window = None
            per_sample_windows.append(window)

        # Return [] only when no sample in the batch has a cached window; otherwise
        # use the first available window as the shape reference so mixed batches
        # (some sequences cached, some not) pad the missing samples with zeros.
        first_valid = next((w for w in per_sample_windows if w is not None), None)
        if first_valid is None:
            return []
        num_scales = len(first_valid)
        result: List[torch.Tensor] = []
        for si in range(num_scales):
            scale_tensors = [
                (w[si] if w is not None else torch.zeros_like(first_valid[si]))
                for w in per_sample_windows
            ]
            stacked = torch.stack(scale_tensors, dim=0)  # [B, K, C, H, W]
            result.append(stacked.to(self.device, non_blocking=True))
        return result
