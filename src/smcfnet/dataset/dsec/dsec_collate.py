from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch
import torch.nn.functional as F


def _dsec_packed_events_collate(
    batch: List[Dict[str, Any]],
    *,
    timestamp_unit: str,
) -> Dict[str, Any]:
    """Shared packed-events collate helper.

    Args:
      batch: Mapper outputs with `events` tensors shaped [Ni, 4].
      timestamp_unit: Human-readable timestamp unit used in docs/errors.
    """
    if timestamp_unit not in {"ms", "us"}:
        raise ValueError(f"timestamp_unit must be 'ms' or 'us', got {timestamp_unit!r}")

    event_label = f"(x, y, t_{timestamp_unit}, p)"
    empty_out: Dict[str, Any] = {
        "events": torch.empty((0, 4), dtype=torch.float32),
        "events_batch_idx": torch.empty((0,), dtype=torch.int64),
        "events_offsets": torch.zeros((1,), dtype=torch.int64),
    }

    batch_size = len(batch)
    if batch_size == 0:
        return empty_out

    # First pass: normalize to tensors, validate shapes, compute prefix sums.
    events_list: List[torch.Tensor] = []
    lengths: List[int] = []

    first_nonempty: torch.Tensor | None = None
    for sample in batch:
        events = sample.get("events", None)
        if events is None:
            events = torch.empty((0, 4), dtype=torch.int64)
        if not isinstance(events, torch.Tensor):
            events = torch.as_tensor(events)
        if events.numel() == 0:
            events = events.reshape(0, 4)
        if events.ndim != 2 or events.shape[1] != 4:
            raise ValueError(f"Expected events of shape [N,4] {event_label}, got {tuple(events.shape)}")

        # Keep dtype as-is; casting should happen in the model/prefetch stage if needed.
        events = events.contiguous()
        if first_nonempty is None and events.shape[0] > 0:
            first_nonempty = events

        events_list.append(events)
        lengths.append(int(events.shape[0]))

    total_events = int(sum(lengths))
    events_dtype = first_nonempty.dtype if first_nonempty is not None else torch.int64
    events_device = first_nonempty.device if first_nonempty is not None else torch.device("cpu")

    # Preallocate + copy instead of torch.cat to reduce overhead.
    events_packed = (
        torch.empty((total_events, 4), dtype=events_dtype, device=events_device)
        if total_events > 0
        else torch.empty((0, 4), dtype=events_dtype, device=events_device)
    )

    events_batch_idx = (
        torch.empty((total_events,), dtype=torch.int64, device=events_device)
        if total_events > 0
        else torch.empty((0,), dtype=torch.int64, device=events_device)
    )

    offsets_t = torch.empty((batch_size + 1,), dtype=torch.int64)
    offsets_t[0] = 0

    write = 0
    for i, (ev, n_i) in enumerate(zip(events_list, lengths)):
        offsets_t[i + 1] = offsets_t[i] + n_i
        if n_i == 0:
            continue
        events_packed[write : write + n_i].copy_(ev)
        events_batch_idx[write : write + n_i].fill_(i)
        write += n_i

    out: Dict[str, Any] = {
        "events": events_packed,
        "events_batch_idx": events_batch_idx,
        "events_offsets": offsets_t,
        "seq_idx": torch.as_tensor([int(s["seq_idx"]) for s in batch], dtype=torch.int64),
        "frame_index": torch.as_tensor([int(s["frame_index"]) for s in batch], dtype=torch.int64),
        "t_ref_us": torch.as_tensor([int(s["t_ref_us"]) for s in batch], dtype=torch.int64),
        "t_start_us": torch.as_tensor([int(s["t_start_us"]) for s in batch], dtype=torch.int64),
        "t_end_us": torch.as_tensor([int(s["t_end_us"]) for s in batch], dtype=torch.int64),
        "sequence_reset": torch.as_tensor(
            [bool(s.get("sequence_reset", int(s["frame_index"]) <= 0)) for s in batch],
            dtype=torch.bool,
        ),
    }

    # Pass through any extra keys as lists to avoid surprising tensorization.
    # Raw full-sequence tracks can be very large and should never be part of a
    # training batch once per-frame targets have been built.
    known = set(out.keys())
    known.update({"events", "tracks"})
    for key in batch[0].keys():
        if key in known:
            continue
        out[key] = [s.get(key) for s in batch]

    return out


def dsec_events_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate a batch of DSEC mapper outputs.

    Expects each sample dict to contain:
      - events: Tensor [Ni, 4] (x, y, t_ms, p)
      - seq_idx, frame_index, t_ref_us, t_start_us, t_end_us: int-like

        Returns a packed representation to avoid padding variable-length events:
      - events: Tensor [sum_i Ni, 4]
      - events_batch_idx: LongTensor [sum_i Ni] mapping each event -> batch index
      - events_offsets: LongTensor [B+1] prefix sums into the packed events tensor
      - seq_idx, frame_index, t_ref_us, t_start_us, t_end_us: LongTensor [B]

    Other keys are returned as lists (if present).
    """
    return _dsec_packed_events_collate(batch, timestamp_unit="ms")


def dsec_streaming_events_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate a batch of streaming DSEC mapper outputs.

    Expects each sample dict to contain:
      - events: Tensor [Ni, 4] (x, y, t_us, p)
      - seq_idx, frame_index, t_ref_us, t_start_us, t_end_us: int-like

    Returns the same packed schema as `dsec_events_collate`, but with
    microsecond timestamps preserved in `events[:, 2]`.
    """
    return _dsec_packed_events_collate(batch, timestamp_unit="us")


def dsec_train_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate for training/eval that optionally includes images/targets.

    - Always produces the packed events representation (same as `dsec_events_collate`).
    - If samples contain `image` tensors (C,H,W), stacks into `image` (B,C,H,W).
    - If samples contain `targets` tensors (Ni,5), pads into `targets` (B,M,5).

    This lets the DataLoader workers do image/target materialization while
    keeping voxelization for a later GPU stage.
    """
    out = dsec_events_collate(batch)
    batch_size = len(batch)
    if batch_size == 0:
        return out

    # Images (optional)
    if "image" in batch[0]:
        images = [s.get("image") for s in batch]
        if all(isinstance(im, torch.Tensor) for im in images):
            images_t = torch.stack([im.contiguous() for im in images], dim=0)
            out["image"] = images_t

    # Targets (optional)
    if "targets" in batch[0]:
        targets_list = [s.get("targets") for s in batch]
        if all(isinstance(t, torch.Tensor) for t in targets_list):
            max_num = max((int(t.shape[0]) for t in targets_list), default=0)
            if max_num <= 0:
                max_num = 1
            targets_t = torch.zeros((batch_size, max_num, 5), dtype=torch.float32)
            for i, t in enumerate(targets_list):
                if t is None or t.numel() == 0:
                    continue
                tt = t.to(torch.float32)
                n = min(int(tt.shape[0]), max_num)
                targets_t[i, :n].copy_(tt[:n])
            out["targets"] = targets_t

    # Future targets for STTP (optional)
    if "future_targets" in batch[0]:
        ft_list = [s.get("future_targets") for s in batch]
        if all(isinstance(ft, dict) for ft in ft_list):
            max_num = max((int(ft["boxes"].shape[0]) for ft in ft_list), default=0)
            if max_num <= 0:
                max_num = 1
            # boxes are [N, k, 4], validity [N, k] (multi-step trajectory). k is the
            # per-run horizon (sttp.future_steps), shared across samples.
            k_steps = int(ft_list[0]["boxes"].shape[1])
            fut_boxes = torch.zeros((batch_size, max_num, k_steps, 4), dtype=torch.float32)
            fut_valid = torch.zeros((batch_size, max_num, k_steps), dtype=torch.bool)
            fut_current = torch.zeros((batch_size, max_num, 4), dtype=torch.float32)
            for i, ft in enumerate(ft_list):
                n = int(ft["boxes"].shape[0])
                if n > 0:
                    fut_boxes[i, :n].copy_(ft["boxes"][:n])
                    fut_valid[i, :n].copy_(ft["validity"][:n])
                    if isinstance(ft.get("current_boxes"), torch.Tensor):
                        fut_current[i, :n].copy_(ft["current_boxes"][:n])
            out["future_targets"] = {"boxes": fut_boxes, "validity": fut_valid, "current_boxes": fut_current}

    # Flow ground truth (optional) — (B, 3, H, W) with [flow_x, flow_y, valid]
    if "flow_gt" in batch[0]:
        flow_list = [s.get("flow_gt") for s in batch]
        if all(isinstance(f, torch.Tensor) for f in flow_list):
            out["flow_gt"] = torch.stack(flow_list, dim=0)

    return out


def dsec_streaming_train_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Training/eval collate for streaming-event batches.

    Mirrors `dsec_train_collate`, but expects `events[:, 2]` to contain
    microsecond timestamps from a mapper configured with
    `event_timestamp_unit="us"`.
    """
    out = dsec_streaming_events_collate(batch)
    batch_size = len(batch)
    if batch_size == 0:
        return out

    if "image" in batch[0]:
        images = [s.get("image") for s in batch]
        if all(isinstance(im, torch.Tensor) for im in images):
            out["image"] = torch.stack([im.contiguous() for im in images], dim=0)

    if "targets" in batch[0]:
        targets_list = [s.get("targets") for s in batch]
        if all(isinstance(t, torch.Tensor) for t in targets_list):
            max_num = max((int(t.shape[0]) for t in targets_list), default=0)
            if max_num <= 0:
                max_num = 1
            targets_t = torch.zeros((batch_size, max_num, 5), dtype=torch.float32)
            for i, t in enumerate(targets_list):
                if t is None or t.numel() == 0:
                    continue
                tt = t.to(torch.float32)
                n = min(int(tt.shape[0]), max_num)
                targets_t[i, :n].copy_(tt[:n])
            out["targets"] = targets_t

    # Future targets for STTP (optional)
    if "future_targets" in batch[0]:
        ft_list = [s.get("future_targets") for s in batch]
        if all(isinstance(ft, dict) for ft in ft_list):
            max_num = max((int(ft["boxes"].shape[0]) for ft in ft_list), default=0)
            if max_num <= 0:
                max_num = 1
            # boxes are [N, k, 4], validity [N, k] (multi-step trajectory). k is the
            # per-run horizon (sttp.future_steps), shared across samples.
            k_steps = int(ft_list[0]["boxes"].shape[1])
            fut_boxes = torch.zeros((batch_size, max_num, k_steps, 4), dtype=torch.float32)
            fut_valid = torch.zeros((batch_size, max_num, k_steps), dtype=torch.bool)
            fut_current = torch.zeros((batch_size, max_num, 4), dtype=torch.float32)
            for i, ft in enumerate(ft_list):
                n = int(ft["boxes"].shape[0])
                if n > 0:
                    fut_boxes[i, :n].copy_(ft["boxes"][:n])
                    fut_valid[i, :n].copy_(ft["validity"][:n])
                    if isinstance(ft.get("current_boxes"), torch.Tensor):
                        fut_current[i, :n].copy_(ft["current_boxes"][:n])
            out["future_targets"] = {"boxes": fut_boxes, "validity": fut_valid, "current_boxes": fut_current}

    if "flow_gt" in batch[0]:
        flow_list = [s.get("flow_gt") for s in batch]
        if all(isinstance(f, torch.Tensor) for f in flow_list):
            out["flow_gt"] = torch.stack(flow_list, dim=0)

    return out


def dsec_rgb_train_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate for RGB-only training/eval (no events).

    Expects each sample dict to contain:
      - seq_idx, frame_index, t_ref_us, t_start_us, t_end_us: int-like
    Optionally:
      - image: Tensor (3,H,W)
      - targets: Tensor (Ni,5)

    Produces:
      - seq_idx, frame_index, t_ref_us, t_start_us, t_end_us: LongTensor (B,)
      - image: Tensor (B,3,H,W) if present
      - targets: Tensor (B,M,5) padded if present

    Other keys are returned as lists (if present).
    """

    batch_size = len(batch)
    if batch_size == 0:
        return {}

    out: Dict[str, Any] = {
        "seq_idx": torch.as_tensor([int(s["seq_idx"]) for s in batch], dtype=torch.int64),
        "frame_index": torch.as_tensor([int(s["frame_index"]) for s in batch], dtype=torch.int64),
        "t_ref_us": torch.as_tensor([int(s["t_ref_us"]) for s in batch], dtype=torch.int64),
        "t_start_us": torch.as_tensor([int(s["t_start_us"]) for s in batch], dtype=torch.int64),
        "t_end_us": torch.as_tensor([int(s["t_end_us"]) for s in batch], dtype=torch.int64),
    }

    # Images (optional)
    if "image" in batch[0]:
        images = [s.get("image") for s in batch]
        if all(isinstance(im, torch.Tensor) for im in images):
            out["image"] = torch.stack([im.contiguous() for im in images], dim=0)

    # Targets (optional)
    if "targets" in batch[0]:
        targets_list = [s.get("targets") for s in batch]
        if all(isinstance(t, torch.Tensor) for t in targets_list):
            max_num = max((int(t.shape[0]) for t in targets_list), default=0)
            if max_num <= 0:
                max_num = 1
            targets_t = torch.zeros((batch_size, max_num, 5), dtype=torch.float32)
            for i, t in enumerate(targets_list):
                if t is None or t.numel() == 0:
                    continue
                tt = t.to(torch.float32)
                n = min(int(tt.shape[0]), max_num)
                targets_t[i, :n].copy_(tt[:n])
            out["targets"] = targets_t

    # Future targets for STTP (optional)
    if "future_targets" in batch[0]:
        ft_list = [s.get("future_targets") for s in batch]
        if all(isinstance(ft, dict) for ft in ft_list):
            max_num = max((int(ft["boxes"].shape[0]) for ft in ft_list), default=0)
            if max_num <= 0:
                max_num = 1
            # boxes are [N, k, 4], validity [N, k] (multi-step trajectory). k is the
            # per-run horizon (sttp.future_steps), shared across samples.
            k_steps = int(ft_list[0]["boxes"].shape[1])
            fut_boxes = torch.zeros((batch_size, max_num, k_steps, 4), dtype=torch.float32)
            fut_valid = torch.zeros((batch_size, max_num, k_steps), dtype=torch.bool)
            fut_current = torch.zeros((batch_size, max_num, 4), dtype=torch.float32)
            for i, ft in enumerate(ft_list):
                n = int(ft["boxes"].shape[0])
                if n > 0:
                    fut_boxes[i, :n].copy_(ft["boxes"][:n])
                    fut_valid[i, :n].copy_(ft["validity"][:n])
                    if isinstance(ft.get("current_boxes"), torch.Tensor):
                        fut_current[i, :n].copy_(ft["current_boxes"][:n])
            out["future_targets"] = {"boxes": fut_boxes, "validity": fut_valid, "current_boxes": fut_current}

    # Pass through any extra keys as lists to avoid surprising tensorization.
    # Raw tracks are sequence-level metadata, not per-batch training tensors.
    known = set(out.keys())
    known.update({"tracks"})
    for key in batch[0].keys():
        if key in known:
            continue
        out[key] = [s.get(key) for s in batch]

    return out


def _resolve_sample_hw(sample: Dict[str, Any], default_height: int, default_width: int) -> tuple[int, int]:
    image = sample.get("image")
    if isinstance(image, torch.Tensor) and image.ndim == 3:
        return int(image.shape[-2]), int(image.shape[-1])
    return int(default_height), int(default_width)


def _resize_image_to_canvas(image: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    if image.ndim != 3:
        raise ValueError(f"Expected image tensor (C,H,W), got {tuple(image.shape)}")
    if int(image.shape[-2]) == int(out_h) and int(image.shape[-1]) == int(out_w):
        return image.contiguous()
    img = image.unsqueeze(0).to(torch.float32)
    img = F.interpolate(img, size=(int(out_h), int(out_w)), mode="bilinear", align_corners=False)
    return img.squeeze(0).contiguous().to(image.dtype)


def _transform_events_to_canvas(
    events: torch.Tensor,
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
    if not isinstance(events, torch.Tensor) or events.numel() == 0:
        dtype = events.dtype if isinstance(events, torch.Tensor) else torch.int64
        return torch.empty((0, 4), dtype=dtype)

    out = events.clone()
    sx = float(dst_w) / float(max(int(src_w), 1))
    sy = float(dst_h) / float(max(int(src_h), 1))

    x = torch.floor(out[:, 0].to(torch.float32) * sx) + float(offset_x)
    y = torch.floor(out[:, 1].to(torch.float32) * sy) + float(offset_y)

    out[:, 0] = x.to(out.dtype)
    out[:, 1] = y.to(out.dtype)

    valid = (
        (out[:, 0] >= 0)
        & (out[:, 0] < int(canvas_w))
        & (out[:, 1] >= 0)
        & (out[:, 1] < int(canvas_h))
    )
    return out[valid].contiguous()


def _transform_targets_to_canvas(
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
        return torch.zeros((0, 5), dtype=torch.float32)

    tt = targets.to(torch.float32).clone().reshape(-1, 5)
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
    if not torch.any(valid):
        return torch.zeros((0, 5), dtype=torch.float32)

    out = torch.zeros((int(valid.sum().item()), 5), dtype=torch.float32)
    out[:, 0] = tt[valid, 0]
    out[:, 1] = x1[valid] + new_w[valid] / 2.0
    out[:, 2] = y1[valid] + new_h[valid] / 2.0
    out[:, 3] = new_w[valid]
    out[:, 4] = new_h[valid]
    return out.contiguous()


def _compose_mosaic_sample(
    samples: List[Dict[str, Any]],
    *,
    use_events: bool,
    out_h: int,
    out_w: int,
) -> Dict[str, Any]:
    anchor = dict(samples[0])
    out: Dict[str, Any] = {
        k: v
        for k, v in anchor.items()
        if k not in {"events", "image", "targets", "flow_gt"}
    }

    half_h = int(out_h // 2)
    half_w = int(out_w // 2)
    placements = [
        (0, 0, half_h, half_w),
        (0, half_w, half_h, int(out_w - half_w)),
        (half_h, 0, int(out_h - half_h), half_w),
        (half_h, half_w, int(out_h - half_h), int(out_w - half_w)),
    ]

    images_present = all(isinstance(sample.get("image"), torch.Tensor) for sample in samples)
    if images_present:
        image_canvas = torch.zeros(
            (int(samples[0]["image"].shape[0]), int(out_h), int(out_w)),
            dtype=samples[0]["image"].dtype,
        )
    else:
        image_canvas = None

    events_parts: List[torch.Tensor] = []
    target_parts: List[torch.Tensor] = []

    for sample, (off_y, off_x, dst_h, dst_w) in zip(samples, placements):
        src_h, src_w = _resolve_sample_hw(sample, out_h, out_w)

        if image_canvas is not None:
            resized = _resize_image_to_canvas(sample["image"], dst_h, dst_w)
            image_canvas[:, off_y : off_y + dst_h, off_x : off_x + dst_w] = resized

        if use_events and isinstance(sample.get("events"), torch.Tensor):
            ev = _transform_events_to_canvas(
                sample["events"],
                src_h=src_h,
                src_w=src_w,
                dst_h=dst_h,
                dst_w=dst_w,
                offset_y=off_y,
                offset_x=off_x,
                canvas_h=out_h,
                canvas_w=out_w,
            )
            if ev.numel() > 0:
                events_parts.append(ev)

        if isinstance(sample.get("targets"), torch.Tensor):
            tg = _transform_targets_to_canvas(
                sample["targets"],
                src_h=src_h,
                src_w=src_w,
                dst_h=dst_h,
                dst_w=dst_w,
                offset_y=off_y,
                offset_x=off_x,
                canvas_h=out_h,
                canvas_w=out_w,
            )
            if tg.numel() > 0:
                target_parts.append(tg)

    if image_canvas is not None:
        out["image"] = image_canvas.contiguous()
    if use_events:
        event_dtype = (
            events_parts[0].dtype
            if len(events_parts) > 0
            else (
                samples[0]["events"].dtype
                if isinstance(samples[0].get("events"), torch.Tensor)
                else torch.int64
            )
        )
        out["events"] = (
            torch.cat(events_parts, dim=0).contiguous()
            if len(events_parts) > 0
            else torch.empty((0, 4), dtype=event_dtype)
        )
    if any(isinstance(sample.get("targets"), torch.Tensor) for sample in samples):
        out["targets"] = (
            torch.cat(target_parts, dim=0).contiguous()
            if len(target_parts) > 0
            else torch.zeros((0, 5), dtype=torch.float32)
        )
    return out


def _resize_sample_to_canvas(
    sample: Dict[str, Any],
    *,
    use_events: bool,
    out_h: int,
    out_w: int,
) -> Dict[str, Any]:
    out = {
        k: v
        for k, v in sample.items()
        if k not in {"events", "image", "targets", "flow_gt"}
    }
    src_h, src_w = _resolve_sample_hw(sample, out_h, out_w)
    if isinstance(sample.get("image"), torch.Tensor):
        out["image"] = _resize_image_to_canvas(sample["image"], out_h, out_w)
    if use_events and isinstance(sample.get("events"), torch.Tensor):
        out["events"] = _transform_events_to_canvas(
            sample["events"],
            src_h=src_h,
            src_w=src_w,
            dst_h=out_h,
            dst_w=out_w,
            offset_y=0,
            offset_x=0,
            canvas_h=out_h,
            canvas_w=out_w,
        )
    if isinstance(sample.get("targets"), torch.Tensor):
        out["targets"] = _transform_targets_to_canvas(
            sample["targets"],
            src_h=src_h,
            src_w=src_w,
            dst_h=out_h,
            dst_w=out_w,
            offset_y=0,
            offset_x=0,
            canvas_h=out_h,
            canvas_w=out_w,
        )
    return out


def _compose_mixup_sample(
    base: Dict[str, Any],
    partner: Dict[str, Any],
    *,
    use_events: bool,
    out_h: int,
    out_w: int,
) -> Dict[str, Any]:
    out = {
        k: v
        for k, v in base.items()
        if k not in {"events", "image", "targets", "flow_gt"}
    }
    base_resized = _resize_sample_to_canvas(base, use_events=use_events, out_h=out_h, out_w=out_w)
    partner_resized = _resize_sample_to_canvas(partner, use_events=use_events, out_h=out_h, out_w=out_w)

    if isinstance(base_resized.get("image"), torch.Tensor) and isinstance(partner_resized.get("image"), torch.Tensor):
        out["image"] = (0.5 * base_resized["image"].to(torch.float32) + 0.5 * partner_resized["image"].to(torch.float32)).contiguous()
    elif isinstance(base_resized.get("image"), torch.Tensor):
        out["image"] = base_resized["image"]

    if use_events:
        base_events = base_resized.get("events")
        partner_events = partner_resized.get("events")
        event_parts = [ev for ev in (base_events, partner_events) if isinstance(ev, torch.Tensor) and ev.numel() > 0]
        event_dtype = None
        if isinstance(base_events, torch.Tensor):
            event_dtype = base_events.dtype
        elif isinstance(partner_events, torch.Tensor):
            event_dtype = partner_events.dtype
        else:
            event_dtype = torch.int64
        out["events"] = (
            torch.cat(event_parts, dim=0).contiguous()
            if len(event_parts) > 0
            else torch.empty((0, 4), dtype=event_dtype)
        )

    base_targets = base_resized.get("targets")
    partner_targets = partner_resized.get("targets")
    target_parts = [
        tg.to(torch.float32).reshape(-1, 5)
        for tg in (base_targets, partner_targets)
        if isinstance(tg, torch.Tensor) and tg.numel() > 0
    ]
    if len(target_parts) > 0:
        out["targets"] = torch.cat(target_parts, dim=0).contiguous()
    elif isinstance(base_targets, torch.Tensor) or isinstance(partner_targets, torch.Tensor):
        out["targets"] = torch.zeros((0, 5), dtype=torch.float32)

    return out


def _build_train_collate_output(batch: List[Dict[str, Any]], *, timestamp_unit: str, use_events: bool) -> Dict[str, Any]:
    if use_events:
        out = _dsec_packed_events_collate(batch, timestamp_unit=timestamp_unit)
    else:
        out = {
            "seq_idx": torch.as_tensor([int(s["seq_idx"]) for s in batch], dtype=torch.int64),
            "frame_index": torch.as_tensor([int(s["frame_index"]) for s in batch], dtype=torch.int64),
            "t_ref_us": torch.as_tensor([int(s["t_ref_us"]) for s in batch], dtype=torch.int64),
            "t_start_us": torch.as_tensor([int(s["t_start_us"]) for s in batch], dtype=torch.int64),
            "t_end_us": torch.as_tensor([int(s["t_end_us"]) for s in batch], dtype=torch.int64),
        }

        known = set(out.keys())
        known.update({"tracks"})
        for key in batch[0].keys():
            if key in known:
                continue
            out[key] = [s.get(key) for s in batch]

    batch_size = len(batch)
    if batch_size == 0:
        return out

    if "image" in batch[0]:
        images = [s.get("image") for s in batch]
        if all(isinstance(im, torch.Tensor) for im in images):
            out["image"] = torch.stack([im.contiguous() for im in images], dim=0)

    if "targets" in batch[0]:
        targets_list = [s.get("targets") for s in batch]
        if all(isinstance(t, torch.Tensor) for t in targets_list):
            max_num = max((int(t.shape[0]) for t in targets_list), default=0)
            if max_num <= 0:
                max_num = 1
            targets_t = torch.zeros((batch_size, max_num, 5), dtype=torch.float32)
            for i, t in enumerate(targets_list):
                if t is None or t.numel() == 0:
                    continue
                tt = t.to(torch.float32)
                n = min(int(tt.shape[0]), max_num)
                targets_t[i, :n].copy_(tt[:n])
            out["targets"] = targets_t

    if "flow_gt" in batch[0]:
        flow_list = [s.get("flow_gt") for s in batch]
        if all(isinstance(f, torch.Tensor) for f in flow_list):
            out["flow_gt"] = torch.stack(flow_list, dim=0)

    if "future_targets" in batch[0]:
        ft_list = [s.get("future_targets") for s in batch]
        if all(isinstance(ft, dict) for ft in ft_list):
            max_num = max((int(ft["boxes"].shape[0]) for ft in ft_list), default=0)
            if max_num <= 0:
                max_num = 1
            # boxes are [N, k, 4], validity [N, k] (multi-step trajectory). k is the
            # per-run horizon (sttp.future_steps), shared across samples.
            k_steps = int(ft_list[0]["boxes"].shape[1])
            fut_boxes = torch.zeros((batch_size, max_num, k_steps, 4), dtype=torch.float32)
            fut_valid = torch.zeros((batch_size, max_num, k_steps), dtype=torch.bool)
            fut_current = torch.zeros((batch_size, max_num, 4), dtype=torch.float32)
            for i, ft in enumerate(ft_list):
                n = int(ft["boxes"].shape[0])
                if n > 0:
                    fut_boxes[i, :n].copy_(ft["boxes"][:n])
                    fut_valid[i, :n].copy_(ft["validity"][:n])
                    if isinstance(ft.get("current_boxes"), torch.Tensor):
                        fut_current[i, :n].copy_(ft["current_boxes"][:n])
            out["future_targets"] = {"boxes": fut_boxes, "validity": fut_valid, "current_boxes": fut_current}

    return out


def _augment_training_batch(
    batch: List[Dict[str, Any]],
    *,
    use_events: bool,
    use_mosaic: bool,
    use_mixup: bool,
    mosaic_prob: float,
    mixup_prob: float,
    image_height: int,
    image_width: int,
) -> List[Dict[str, Any]]:
    if len(batch) == 0:
        return batch
    if any("flow_gt" in sample for sample in batch):
        return batch
    if not any(isinstance(sample.get("targets"), torch.Tensor) for sample in batch):
        return batch
    if not any(isinstance(sample.get("image"), torch.Tensor) for sample in batch):
        return batch

    aug_batch: List[Dict[str, Any]] = []
    pool_size = len(batch)
    for anchor_idx, sample in enumerate(batch):
        aug_sample = sample

        if use_mosaic and float(torch.rand(1).item()) < float(mosaic_prob):
            indices = [anchor_idx]
            if pool_size > 1:
                extra = torch.randint(0, pool_size, (3,), dtype=torch.int64).tolist()
                indices.extend(int(idx) for idx in extra)
            else:
                indices.extend([anchor_idx, anchor_idx, anchor_idx])
            mosaic_sources = [batch[int(idx)] for idx in indices[:4]]
            aug_sample = _compose_mosaic_sample(
                mosaic_sources,
                use_events=use_events,
                out_h=image_height,
                out_w=image_width,
            )

        if use_mixup and pool_size > 1 and float(torch.rand(1).item()) < float(mixup_prob):
            partner_idx = int(torch.randint(0, pool_size - 1, (1,), dtype=torch.int64).item())
            if partner_idx >= anchor_idx:
                partner_idx += 1
            aug_sample = _compose_mixup_sample(
                aug_sample,
                batch[partner_idx],
                use_events=use_events,
                out_h=image_height,
                out_w=image_width,
            )

        aug_batch.append(aug_sample)

    return aug_batch


@dataclass
class DSECTrainCollate:
    """Configurable train collate with optional batch-local mosaic/mixup."""

    use_events: bool
    timestamp_unit: str = "ms"
    use_mosaic: bool = False
    use_mixup: bool = False
    mosaic_prob: float = 1.0
    mixup_prob: float = 0.5
    image_height: int = 480
    image_width: int = 640

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        if len(batch) == 0:
            return _build_train_collate_output(batch, timestamp_unit=self.timestamp_unit, use_events=self.use_events)

        batch_for_collate = batch
        if self.use_mosaic or self.use_mixup:
            batch_for_collate = _augment_training_batch(
                batch,
                use_events=bool(self.use_events),
                use_mosaic=bool(self.use_mosaic),
                use_mixup=bool(self.use_mixup),
                mosaic_prob=float(self.mosaic_prob),
                mixup_prob=float(self.mixup_prob),
                image_height=int(self.image_height),
                image_width=int(self.image_width),
            )

        return _build_train_collate_output(
            batch_for_collate,
            timestamp_unit=self.timestamp_unit,
            use_events=bool(self.use_events),
        )
