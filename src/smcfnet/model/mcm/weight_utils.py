"""MCM / OFE weight loading, saving, and freezing utilities.

Supports the progressive training workflow:
  Phase A:  MCM pre-training (only OFE + warper + loss)
  Phase B:  Detection training with frozen pre-trained MCM
  Phase C:  Joint fine-tuning (unfreeze MCM with lower LR)

Key prefix conventions:
  - MCM-only checkpoint keys:  ``estimator.*``, ``warper.*``, (optional ``loss_fn.*``)
  - Full-model MCM keys:       ``backbone.mcm.estimator.*``, ``backbone.mcm.warper.*``
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _strip_module_prefix(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Drop DistributedDataParallel ``module.`` prefix when present."""
    out: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            out[k[len("module."):]] = v
        else:
            out[k] = v
    return out


def _extract_tensor_state_dict(blob: Any) -> Optional[Dict[str, torch.Tensor]]:
    """Extract a tensor-only state dict from common checkpoint container shapes."""
    if not isinstance(blob, Mapping):
        return None

    if "model" in blob and isinstance(blob["model"], Mapping):
        model_blob = blob["model"]
        if all(torch.is_tensor(v) for v in model_blob.values()):
            return dict(model_blob)

    if "state_dict" in blob and isinstance(blob["state_dict"], Mapping):
        state_blob = blob["state_dict"]
        if all(torch.is_tensor(v) for v in state_blob.values()):
            return dict(state_blob)

    if all(torch.is_tensor(v) for v in blob.values()):
        return dict(blob)

    for value in blob.values():
        if isinstance(value, Mapping) and value and all(torch.is_tensor(v) for v in value.values()):
            return dict(value)

    return None


def _as_estimator_state_dict(state: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Normalize arbitrary checkpoint keys into estimator-local key space."""
    state = _strip_module_prefix(state)

    prefix_full_est = "backbone.mcm.estimator."
    if any(k.startswith(prefix_full_est) for k in state.keys()):
        return {k[len(prefix_full_est):]: v for k, v in state.items() if k.startswith(prefix_full_est)}

    prefix_est = "estimator."
    if any(k.startswith(prefix_est) for k in state.keys()):
        return {k[len(prefix_est):]: v for k, v in state.items() if k.startswith(prefix_est)}

    prefix_full_mcm = "backbone.mcm."
    if any(k.startswith(prefix_full_mcm) for k in state.keys()):
        mcm_state = {k[len(prefix_full_mcm):]: v for k, v in state.items() if k.startswith(prefix_full_mcm)}
        if any(k.startswith(prefix_est) for k in mcm_state.keys()):
            return {k[len(prefix_est):]: v for k, v in mcm_state.items() if k.startswith(prefix_est)}
        raise KeyError(
            "Checkpoint has backbone.mcm.* keys but no backbone.mcm.estimator.* keys"
        )

    # Estimator-only/raw checkpoint (e.g. third-party OFE/SNN model).
    forbidden_prefixes = (
        "backbone.",
        "head.",
        "neck.",
        "optimizer",
        "scheduler",
    )
    if any(k.startswith(forbidden_prefixes) for k in state.keys()):
        raise KeyError(
            "Checkpoint does not contain estimator-prefixed keys and looks like a "
            "non-estimator model state_dict"
        )
    return dict(state)


# ---------------------------------------------------------------------------
# Save / Load MCM-only checkpoints
# ---------------------------------------------------------------------------

def save_mcm_checkpoint(
    mcm: nn.Module,
    path: Union[str, Path],
    *,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
    iteration: int = 0,
    best_metric: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Save an MCM-only checkpoint.

    The checkpoint contains the MCM ``state_dict`` plus optional training state,
    enabling both warm-restart of MCM pre-training and weight-only loading into
    the full detection model.
    """
    ckpt: Dict[str, Any] = {
        "mcm": mcm.state_dict(),
        "epoch": int(epoch),
        "iteration": int(iteration),
        "best_metric": float(best_metric),
    }
    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        ckpt["scheduler"] = scheduler.state_dict()
    if extra:
        ckpt.update(extra)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)
    logger.info("Saved MCM checkpoint to %s", path)


def load_mcm_checkpoint(path: Union[str, Path], map_location: str = "cpu") -> Dict[str, Any]:
    """Load an MCM-only checkpoint.

    Returns the raw checkpoint dict with at least ``mcm`` (state_dict).
    """
    ckpt = torch.load(str(path), map_location=map_location)

    if isinstance(ckpt, Mapping) and "mcm" in ckpt:
        mcm_blob = ckpt["mcm"]
        if isinstance(mcm_blob, Mapping):
            out = dict(ckpt)
            out["mcm"] = _strip_module_prefix(dict(mcm_blob))
            return out
        raise KeyError(f"Checkpoint at {path} has non-mapping 'mcm' entry")

    state = _extract_tensor_state_dict(ckpt)
    if state is None:
        raise KeyError(
            f"Checkpoint at {path} has no 'mcm' key and no detectable tensor state_dict"
        )
    state = _strip_module_prefix(state)

    # MCM-only state dict already in estimator/warper/loss namespace.
    if any(k.startswith(("estimator.", "warper.", "loss_fn.")) for k in state.keys()):
        return {"mcm": state}

    # Full-model checkpoint: slice out backbone.mcm.* keys.
    prefix_full_mcm = "backbone.mcm."
    if any(k.startswith(prefix_full_mcm) for k in state.keys()):
        mcm_state = {k[len(prefix_full_mcm):]: v for k, v in state.items() if k.startswith(prefix_full_mcm)}
        if not mcm_state:
            raise KeyError(f"No MCM keys found in checkpoint at {path}")
        return {"mcm": mcm_state}

    # Raw estimator-only checkpoint: wrap under estimator.* for MCM loading.
    estimator_state = _as_estimator_state_dict(state)
    wrapped = {f"estimator.{k}": v for k, v in estimator_state.items()}
    logger.info(
        "Interpreting checkpoint %s as estimator-only state_dict with %d parameters",
        str(path),
        len(wrapped),
    )
    return {"mcm": wrapped}


def load_mcm_weights_into_module(
    mcm: nn.Module,
    checkpoint_path: Union[str, Path],
    *,
    strict: bool = True,
) -> None:
    """Load MCM weights from an MCM-only checkpoint into an MCM module.

    Args:
        mcm: The ``MotionCompensationModule`` instance.
        checkpoint_path: Path to the MCM-only checkpoint.
        strict: Whether to enforce exact key matching (default True).
    """
    ckpt = load_mcm_checkpoint(checkpoint_path)
    mcm_state = ckpt["mcm"]

    try:
        missing, unexpected = mcm.load_state_dict(mcm_state, strict=strict)
        if missing:
            logger.warning("Missing keys when loading MCM weights: %s", missing)
        if unexpected:
            logger.warning("Unexpected keys when loading MCM weights: %s", unexpected)
        logger.info("Loaded MCM weights from %s (strict=%s)", checkpoint_path, strict)
        return
    except RuntimeError:
        # Fallback for estimator-only checkpoints that intentionally do not
        # include warper/loss state.
        if not hasattr(mcm, "estimator"):
            raise

    estimator_state = extract_ofe_state_dict(checkpoint_path)
    missing, unexpected = mcm.estimator.load_state_dict(estimator_state, strict=strict)
    if missing:
        logger.warning("Missing keys when loading OFE weights into estimator: %s", missing)
    if unexpected:
        logger.warning("Unexpected keys when loading OFE weights into estimator: %s", unexpected)
    logger.info(
        "Loaded estimator-only weights from %s into mcm.estimator (strict=%s)",
        checkpoint_path,
        strict,
    )


def load_mcm_weights_into_full_model(
    model: nn.Module,
    checkpoint_path: Union[str, Path],
    *,
    strict: bool = False,
) -> None:
    """Load MCM weights from an MCM-only checkpoint into the full detection model.

    Handles key remapping: MCM checkpoint uses ``estimator.*`` while the full
    model expects ``backbone.mcm.estimator.*``.

    Args:
        model: The full ``DualStreamYOLOX`` model (or DDP-wrapped).
        checkpoint_path: Path to the MCM-only checkpoint.
        strict: Whether to enforce exact key matching (default False — allows
                non-MCM parameters to remain untouched).
    """
    raw_model = model.module if hasattr(model, "module") else model

    # Locate the MCM sub-module
    mcm = None
    if hasattr(raw_model, "backbone") and hasattr(raw_model.backbone, "mcm"):
        mcm = raw_model.backbone.mcm
    elif hasattr(raw_model, "mcm"):
        mcm = raw_model.mcm

    if mcm is None:
        raise AttributeError(
            "Cannot find MCM sub-module in model. Ensure model.mcm.enabled=True."
        )

    load_mcm_weights_into_module(mcm, checkpoint_path, strict=strict)


def extract_mcm_state_dict(
    full_checkpoint_path: Union[str, Path],
    map_location: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """Extract MCM-only state_dict from a full-model checkpoint.

    Useful for converting a jointly-trained checkpoint back to MCM-only format
    so it can be reused for other experiments.

    Returns:
        State dict with keys like ``estimator.*``, ``warper.*``, ``loss_fn.*``
    """
    ckpt = torch.load(str(full_checkpoint_path), map_location=map_location)
    model_state = ckpt.get("model", ckpt.get("state_dict", ckpt))

    prefix = "backbone.mcm."
    mcm_state = {}
    for k, v in model_state.items():
        if k.startswith(prefix):
            mcm_state[k[len(prefix):]] = v

    if not mcm_state:
        raise KeyError(
            f"No keys with prefix '{prefix}' found in checkpoint at "
            f"{full_checkpoint_path}. Available prefixes: "
            f"{sorted(set(k.rsplit('.', 1)[0] for k in list(model_state.keys())[:20]))}"
        )

    logger.info(
        "Extracted %d MCM parameters from full model checkpoint", len(mcm_state)
    )
    return mcm_state


def extract_ofe_state_dict(
    checkpoint_path: Union[str, Path],
    map_location: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """Extract only the OFE (flow estimator) state_dict from any checkpoint.

    Works with both MCM-only and full-model checkpoints.

    Returns:
        State dict with keys like ``feature_encoder.*``, ``update_block.*``, etc.
        (estimator-internal keys, with ``estimator.`` prefix stripped).
    """
    ckpt = torch.load(str(checkpoint_path), map_location=map_location)

    if isinstance(ckpt, Mapping) and "mcm" in ckpt and isinstance(ckpt["mcm"], Mapping):
        state = dict(ckpt["mcm"])
    else:
        state = _extract_tensor_state_dict(ckpt)
        if state is None:
            raise KeyError(
                f"Could not locate a tensor state_dict in checkpoint at {checkpoint_path}"
            )

    ofe_state = _as_estimator_state_dict(state)

    logger.info("Extracted %d OFE parameters from checkpoint", len(ofe_state))
    return ofe_state


# ---------------------------------------------------------------------------
# Freeze / Unfreeze
# ---------------------------------------------------------------------------

def freeze_mcm_params(model: nn.Module) -> int:
    """Freeze all MCM parameters in the model. Returns count of frozen params."""
    raw_model = model.module if hasattr(model, "module") else model
    mcm = None
    if hasattr(raw_model, "backbone") and hasattr(raw_model.backbone, "mcm"):
        mcm = raw_model.backbone.mcm
    elif hasattr(raw_model, "mcm"):
        mcm = raw_model.mcm

    if mcm is None:
        logger.warning("No MCM module found — nothing to freeze")
        return 0

    count = 0
    for param in mcm.parameters():
        param.requires_grad = False
        count += 1
    logger.info("Frozen %d MCM parameters", count)
    return count


def unfreeze_mcm_params(model: nn.Module) -> int:
    """Unfreeze all MCM parameters in the model. Returns count of unfrozen params."""
    raw_model = model.module if hasattr(model, "module") else model
    mcm = None
    if hasattr(raw_model, "backbone") and hasattr(raw_model.backbone, "mcm"):
        mcm = raw_model.backbone.mcm
    elif hasattr(raw_model, "mcm"):
        mcm = raw_model.mcm

    if mcm is None:
        logger.warning("No MCM module found — nothing to unfreeze")
        return 0

    count = 0
    for param in mcm.parameters():
        param.requires_grad = True
        count += 1
    logger.info("Unfrozen %d MCM parameters", count)
    return count


def get_mcm_param_count(model: nn.Module) -> Dict[str, int]:
    """Report trainable/frozen MCM parameter counts."""
    raw_model = model.module if hasattr(model, "module") else model
    mcm = None
    if hasattr(raw_model, "backbone") and hasattr(raw_model.backbone, "mcm"):
        mcm = raw_model.backbone.mcm
    elif hasattr(raw_model, "mcm"):
        mcm = raw_model.mcm

    if mcm is None:
        return {"total": 0, "trainable": 0, "frozen": 0}

    total = sum(p.numel() for p in mcm.parameters())
    trainable = sum(p.numel() for p in mcm.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}
