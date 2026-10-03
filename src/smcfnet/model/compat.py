"""Checkpoint key remapping utilities.

Handles key remapping between older checkpoint state dicts and the
current unified MCM (Motion Compensation Module) architecture.

Two-pass remapping:
    Pass 1 (legacy internal):
        - ``smcm.sofe.*`` -> ``smcm.estimator.*``
        - ``smcm.ewm.*``  -> ``smcm.warper.*``
    Pass 2 (module rename):
        - ``backbone.ecm.{conv1,conv2,conv3,flow_head}.*`` -> ``backbone.mcm.estimator.{...}.*``
        - ``backbone.smcm.*`` -> ``backbone.mcm.*``

Also handles ``module.`` DDP prefix variants.
"""

from __future__ import annotations

import logging
from collections import OrderedDict

logger = logging.getLogger(__name__)

# Pass 1: Legacy internal SMCM key remaps (sofe/ewm -> estimator/warper)
_PASS1_REMAPS = [
    ("smcm.sofe.", "smcm.estimator."),
    ("smcm.ewm.", "smcm.warper."),
    # Legacy pre-modular SMCM referenced from backbone
    ("backbone.mcm.sofe.", "backbone.mcm.estimator."),
    ("backbone.mcm.ewm.", "backbone.mcm.warper."),
]

# Pass 2: Module rename (ECM/SMCM -> MCM)
_PASS2_REMAPS = [
    # ECM flow predictor layers -> MCM estimator
    ("backbone.ecm.conv1.", "backbone.mcm.estimator.conv1."),
    ("backbone.ecm.conv2.", "backbone.mcm.estimator.conv2."),
    ("backbone.ecm.conv3.", "backbone.mcm.estimator.conv3."),
    ("backbone.ecm.flow_head.", "backbone.mcm.estimator.flow_head."),
    # SMCM -> MCM (catch-all for remaining SMCM keys)
    ("backbone.smcm.", "backbone.mcm."),
]


def _apply_remaps(state_dict: dict, remaps: list) -> tuple[dict, int]:
    """Apply a list of prefix remaps, returning new dict and remap count."""
    new_sd = OrderedDict()
    remapped = 0
    for key, value in state_dict.items():
        new_key = key
        for old_prefix, new_prefix in remaps:
            if key.startswith(old_prefix):
                new_key = new_prefix + key[len(old_prefix):]
                remapped += 1
                break
        new_sd[new_key] = value
    return new_sd, remapped


def remap_checkpoint_keys(state_dict: dict) -> dict:
    """Remap old checkpoint keys to the current MCM layout.

    Applies two passes of prefix substitutions:
    1. Legacy SMCM internal renames (sofe->estimator, ewm->warper)
    2. Module renames (ECM/SMCM -> MCM)

    Handles ``module.`` DDP prefix by stripping it, remapping, then
    re-adding if needed.

    Args:
        state_dict: Raw model state dict (potentially from an older
            checkpoint).

    Returns:
        New state dict with remapped keys (same tensor objects, no copy).
    """
    # Strip DDP 'module.' prefix if present, remember to re-add
    has_module_prefix = any(k.startswith("module.") for k in state_dict)
    if has_module_prefix:
        state_dict = OrderedDict(
            (k[len("module."):] if k.startswith("module.") else k, v)
            for k, v in state_dict.items()
        )

    total_remapped = 0

    # Pass 1: legacy internal SMCM renames
    state_dict, n = _apply_remaps(state_dict, _PASS1_REMAPS)
    total_remapped += n

    # Pass 2: module rename (ECM/SMCM -> MCM)
    state_dict, n = _apply_remaps(state_dict, _PASS2_REMAPS)
    total_remapped += n

    if has_module_prefix:
        state_dict = OrderedDict(
            (f"module.{k}", v) for k, v in state_dict.items()
        )

    if total_remapped > 0:
        logger.info(
            "Remapped %d checkpoint keys to current MCM layout "
            "(ECM/SMCM -> MCM, sofe->estimator, etc.)",
            total_remapped,
        )

    return state_dict


# Aliases for backward compatibility
remap_mcm_keys = remap_checkpoint_keys
remap_smcm_keys = remap_checkpoint_keys


def load_checkpoint_compat(state_dict: dict, model) -> None:
    """Load a checkpoint with automatic key remapping.

    Tries strict loading first. If that fails due to key mismatches,
    applies remap and retries with ``strict=False`` (logging any
    remaining mismatches).

    Args:
        state_dict: Model state dict from checkpoint.
        model: The nn.Module to load into.
    """
    try:
        model.load_state_dict(state_dict, strict=True)
        return
    except RuntimeError:
        pass

    # Try with remapped keys
    remapped = remap_checkpoint_keys(state_dict)
    missing, unexpected = model.load_state_dict(remapped, strict=False)

    if missing:
        logger.warning("Missing keys after remap: %s", missing[:10])
    if unexpected:
        logger.warning("Unexpected keys after remap: %s", unexpected[:10])
