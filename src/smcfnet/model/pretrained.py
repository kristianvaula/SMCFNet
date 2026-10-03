from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Mapping, MutableMapping, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PretrainedLoadStats:
    loaded_keys: int
    skipped_shape_mismatch: int
    skipped_unmapped: int
    missing_keys: int
    unexpected_keys: int


def _extract_state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(checkpoint, Mapping):
        if "model" in checkpoint and isinstance(checkpoint["model"], Mapping):
            return checkpoint["model"]  # type: ignore[return-value]
        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], Mapping):
            return checkpoint["state_dict"]  # type: ignore[return-value]
    if isinstance(checkpoint, Mapping):
        return checkpoint  # type: ignore[return-value]
    raise TypeError(f"Unsupported checkpoint type for state_dict: {type(checkpoint)}")


def _strip_module_prefix(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            out[k[len("module.") :]] = v
        else:
            out[k] = v
    return out


def _adapt_conv_weight_channels(src_weight: torch.Tensor, target_shape: Tuple[int, ...]) -> Optional[torch.Tensor]:
    """Adapt Conv2d weights when only in_channels differ.

    This is primarily useful for loading RGB stem weights into an event stem
    with a different number of input channels.
    """
    if src_weight.ndim != 4 or len(target_shape) != 4:
        return None
    if src_weight.shape[0] != target_shape[0] or src_weight.shape[2] != target_shape[2] or src_weight.shape[3] != target_shape[3]:
        return None

    c_src = int(src_weight.shape[1])
    c_tgt = int(target_shape[1])
    if c_src == c_tgt:
        return src_weight

    new_w = src_weight.new_zeros(target_shape)
    c_copy = min(c_src, c_tgt)
    new_w[:, :c_copy] = src_weight[:, :c_copy]

    if c_tgt > c_src:
        fill = src_weight.mean(dim=1, keepdim=True)
        new_w[:, c_src:] = fill.repeat(1, c_tgt - c_src, 1, 1)

    return new_w


_DUALSTREAM_BACKBONE_MARKERS = ("backbone.stem_rgb.", "backbone.stem_event.", "backbone.dark2_rgb.")


def _is_dualstream_checkpoint(state_dict: Mapping[str, torch.Tensor]) -> bool:
    """Return True if checkpoint keys are already in DualStream naming."""
    return any(k.startswith(m) for m in _DUALSTREAM_BACKBONE_MARKERS for k in state_dict)


def _map_yolox_key_to_dualstream(key: str, *, stream: Optional[str]) -> Optional[str]:
    """Map common YOLOX key patterns to our DualStreamYOLOX names.

    Supported sources:
    - Official YOLOX checkpoints (often: backbone.backbone.* + backbone.* (neck) + head.*)
    - Some forks where keys already start with backbone.* / head.*
    """
    if key.startswith("backbone.backbone."):
        if stream is None:
            return None
        rest = key[len("backbone.backbone.") :]
        if rest.startswith("stem."):
            return f"backbone.stem_{stream}." + rest[len("stem.") :]
        for stage in ("dark2", "dark3", "dark4", "dark5"):
            if rest.startswith(stage + "."):
                return f"backbone.{stage}_{stream}." + rest[len(stage) + 1 :]
        return None

    # In official YOLOX, `backbone.*` (without `.backbone`) is the PAFPN neck.
    if key.startswith("backbone."):
        rest = key[len("backbone.") :]
        return "neck." + rest

    if key.startswith("head."):
        return key

    return None


def _load_dualstream_direct(
    model: nn.Module,
    model_dict: Dict[str, torch.Tensor],
    state_dict: Mapping[str, torch.Tensor],
    *,
    load_rgb: bool,
    load_event: bool,
    load_neck: bool,
    load_head: bool,
) -> PretrainedLoadStats:
    """Load a checkpoint that already uses DualStream key names."""
    filtered: Dict[str, torch.Tensor] = {}
    skipped_shape = 0
    skipped_filter = 0

    for k, v in state_dict.items():
        # Apply load_* filters
        if not load_rgb and "_rgb." in k:
            skipped_filter += 1
            continue
        if not load_event and "_event." in k:
            skipped_filter += 1
            continue
        if not load_neck and k.startswith("neck."):
            skipped_filter += 1
            continue
        if not load_head and k.startswith("head."):
            skipped_filter += 1
            continue

        if k not in model_dict:
            skipped_filter += 1
            continue

        if tuple(v.shape) != tuple(model_dict[k].shape):
            adapted = _adapt_conv_weight_channels(v, tuple(model_dict[k].shape))
            if adapted is None or tuple(adapted.shape) != tuple(model_dict[k].shape):
                skipped_shape += 1
                continue
            v = adapted

        filtered[k] = v

    incompatible = model.load_state_dict(filtered, strict=False)

    stats = PretrainedLoadStats(
        loaded_keys=len(filtered),
        skipped_shape_mismatch=skipped_shape,
        skipped_unmapped=skipped_filter,
        missing_keys=len(incompatible.missing_keys),
        unexpected_keys=len(incompatible.unexpected_keys),
    )
    logger.info(
        "Loaded DualStream checkpoint (direct): loaded=%d skipped_shape=%d skipped_nomodel=%d missing=%d unexpected=%d",
        stats.loaded_keys,
        stats.skipped_shape_mismatch,
        stats.skipped_unmapped,
        stats.missing_keys,
        stats.unexpected_keys,
    )
    return stats


def load_yolox_pretrained_dualstream(
    model: nn.Module,
    checkpoint: Any,
    *,
    load_rgb: bool = True,
    load_event: bool = True,
    load_neck: bool = True,
    load_head: bool = True,
) -> PretrainedLoadStats:
    """Load YOLOX(-style) pretrained weights into our dual-stream model.

    This performs:
    - key extraction (checkpoint may be raw state_dict or dict with `model`/`state_dict`)
    - optional removal of a leading `module.` prefix
    - key remapping into our module naming scheme
    - shape checking (skips incompatible tensors)
    - light adaptation for stem conv in-channel mismatch

    Returns a summary of what was loaded.
    """
    state_dict = _strip_module_prefix(_extract_state_dict(checkpoint))
    model_dict = model.state_dict()

    # ── Fast path: checkpoint is already in DualStream naming ──
    if _is_dualstream_checkpoint(state_dict):
        return _load_dualstream_direct(
            model, model_dict, state_dict,
            load_rgb=load_rgb, load_event=load_event,
            load_neck=load_neck, load_head=load_head,
        )

    # ── Standard path: remap from YOLOX naming ──
    remapped: Dict[str, torch.Tensor] = {}
    skipped_shape = 0
    skipped_unmapped = 0

    streams = []
    if load_rgb:
        streams.append("rgb")
    if load_event:
        streams.append("event")

    for k, v in state_dict.items():
        # Backbone/backbone.* keys can be replicated to both streams.
        if k.startswith("backbone.backbone."):
            for stream in streams:
                new_key = _map_yolox_key_to_dualstream(k, stream=stream)
                if new_key is None:
                    skipped_unmapped += 1
                    continue
                if new_key not in model_dict:
                    skipped_unmapped += 1
                    continue
                target_shape = tuple(model_dict[new_key].shape)
                vv = v
                if tuple(v.shape) != target_shape:
                    adapted = _adapt_conv_weight_channels(v, target_shape)
                    if adapted is None or tuple(adapted.shape) != target_shape:
                        skipped_shape += 1
                        continue
                    vv = adapted
                remapped[new_key] = vv
            continue

        # Neck + head: only load if requested.
        if k.startswith("backbone.") and not load_neck:
            continue
        if k.startswith("head.") and not load_head:
            continue

        new_key = _map_yolox_key_to_dualstream(k, stream=None)
        if new_key is None:
            skipped_unmapped += 1
            continue
        if new_key not in model_dict:
            skipped_unmapped += 1
            continue
        if tuple(v.shape) != tuple(model_dict[new_key].shape):
            skipped_shape += 1
            continue
        remapped[new_key] = v

    incompatible = model.load_state_dict(remapped, strict=False)

    stats = PretrainedLoadStats(
        loaded_keys=len(remapped),
        skipped_shape_mismatch=skipped_shape,
        skipped_unmapped=skipped_unmapped,
        missing_keys=len(incompatible.missing_keys),
        unexpected_keys=len(incompatible.unexpected_keys),
    )

    logger.info(
        "Loaded YOLOX pretrained weights (dual-stream mapping): loaded=%d skipped_shape=%d skipped_unmapped=%d missing=%d unexpected=%d",
        stats.loaded_keys,
        stats.skipped_shape_mismatch,
        stats.skipped_unmapped,
        stats.missing_keys,
        stats.unexpected_keys,
    )
    return stats
