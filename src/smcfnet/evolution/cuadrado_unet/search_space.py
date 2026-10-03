"""Cuadrado U-Net EA search-space presets and guardrails."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Tuple

from smcfnet.evolution.framework import GeneSpec


CUADRADO_UNET_SEARCH_SPACE: Dict[str, Any] = {
    "num_bins": [13, 21],
    "mode": ["ann", "snn"],
    "kernel_size": [5, 7],
    "multiply_factor": [5.0, 10.0, 15.0],
}

CUADRADO_UNET_GENE_SPECS: Dict[Tuple[str, ...], GeneSpec] = {
    ("num_bins",): GeneSpec(
        path=("num_bins",),
        kind="categorical",
        values=CUADRADO_UNET_SEARCH_SPACE["num_bins"],
        sensitivity="heavy",
        ordered=True,
    ),
    ("mode",): GeneSpec(
        path=("mode",),
        kind="categorical",
        values=CUADRADO_UNET_SEARCH_SPACE["mode"],
        sensitivity="light",
        ordered=False,
    ),
    ("kernel_size",): GeneSpec(
        path=("kernel_size",),
        kind="categorical",
        values=CUADRADO_UNET_SEARCH_SPACE["kernel_size"],
        sensitivity="medium",
        ordered=True,
    ),
    ("multiply_factor",): GeneSpec(
        path=("multiply_factor",),
        kind="categorical",
        values=CUADRADO_UNET_SEARCH_SPACE["multiply_factor"],
        sensitivity="medium",
        ordered=True,
    ),
}

CUADRADO_UNET_CROSSOVER_GROUPS: Dict[str, Tuple[Tuple[str, ...], ...]] = {
    "input_contract": (
        ("num_bins",),
        ("mode",),
    ),
    "encoder_capacity": (
        ("kernel_size",),
        ("multiply_factor",),
    ),
}


@dataclass
class CuadradoUNetConstraintResult:
    """Result of applying Cuadrado-specific guardrails."""

    genome: Dict[str, Any]
    valid: bool = True
    repairs: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def apply_cuadrado_unet_guardrails(genome: Mapping[str, Any]) -> CuadradoUNetConstraintResult:
    """Validate and lightly repair a Cuadrado genome."""

    normalized = dict(genome or {})
    repairs: List[str] = []
    warnings: List[str] = []

    num_bins = int(normalized.get("num_bins", 21))
    if num_bins <= 0:
        raise ValueError("Cuadrado num_bins must be > 0")
    normalized["num_bins"] = num_bins

    mode = str(normalized.get("mode", "snn")).lower()
    if mode not in {"ann", "snn"}:
        raise ValueError(f"Cuadrado mode must be 'ann' or 'snn', got {mode!r}")
    normalized["mode"] = mode

    kernel_size = int(normalized.get("kernel_size", 7))
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("Cuadrado kernel_size must be a positive odd integer")
    normalized["kernel_size"] = kernel_size

    multiply_factor = float(normalized.get("multiply_factor", 10.0))
    if multiply_factor <= 0.0:
        raise ValueError("Cuadrado multiply_factor must be > 0")
    normalized["multiply_factor"] = multiply_factor

    return CuadradoUNetConstraintResult(
        genome=normalized,
        valid=True,
        repairs=repairs,
        warnings=warnings,
    )


def make_cuadrado_unet_target(search_space: Mapping[str, Any] | None = None):
    from smcfnet.evolution.targets import CuadradoUNetTarget

    return CuadradoUNetTarget(search_space=dict(search_space or CUADRADO_UNET_SEARCH_SPACE))


# ---------------------------------------------------------------------------
# PEOD-specific search space (num_bins and mode are fixed; not genes)
# ---------------------------------------------------------------------------

PEOD_CUADRADO_UNET_SEARCH_SPACE: Dict[str, Any] = {
    "kernel_size": [5, 7],
    "multiply_factor": [5.0, 10.0, 15.0],
    "smooth_weight": [0.3, 0.7, 1.0, 2.0],
    "loss_weight": [0.05, 0.1, 0.2],
}

PEOD_CUADRADO_UNET_GENE_SPECS: Dict[Tuple[str, ...], GeneSpec] = {
    ("kernel_size",): GeneSpec(
        path=("kernel_size",),
        kind="categorical",
        values=PEOD_CUADRADO_UNET_SEARCH_SPACE["kernel_size"],
        sensitivity="medium",
        ordered=True,
    ),
    ("multiply_factor",): GeneSpec(
        path=("multiply_factor",),
        kind="categorical",
        values=PEOD_CUADRADO_UNET_SEARCH_SPACE["multiply_factor"],
        sensitivity="medium",
        ordered=True,
    ),
    ("smooth_weight",): GeneSpec(
        path=("smooth_weight",),
        kind="categorical",
        values=PEOD_CUADRADO_UNET_SEARCH_SPACE["smooth_weight"],
        sensitivity="medium",
        ordered=True,
    ),
    ("loss_weight",): GeneSpec(
        path=("loss_weight",),
        kind="categorical",
        values=PEOD_CUADRADO_UNET_SEARCH_SPACE["loss_weight"],
        sensitivity="light",
        ordered=True,
    ),
}

PEOD_CUADRADO_UNET_CROSSOVER_GROUPS: Dict[str, Tuple[Tuple[str, ...], ...]] = {
    "encoder_capacity": (
        ("kernel_size",),
        ("multiply_factor",),
    ),
    "loss_balance": (
        ("smooth_weight",),
        ("loss_weight",),
    ),
}


def apply_peod_cuadrado_unet_guardrails(genome: Mapping[str, Any]) -> CuadradoUNetConstraintResult:
    """Validate and repair a PEOD Cuadrado genome, forcing fixed-topology fields.

    num_bins and mode must never appear in the search space — they are fixed by
    the PEOD input contract.  If either arrives here with an unexpected value,
    this is a framework bug (e.g. a leaked gene), not a normal repair case, so
    we raise immediately rather than silently overwriting.
    """

    normalized = dict(genome or {})
    repairs: List[str] = []
    warnings: List[str] = []

    num_bins = normalized.get("num_bins", 21)
    if int(num_bins) != 21:
        raise ValueError(
            f"apply_peod_cuadrado_unet_guardrails: num_bins={num_bins!r} — "
            "this field must be fixed at 21 and must not appear in the search space. "
            "A framework bug may have leaked it as a gene."
        )
    normalized["num_bins"] = 21

    mode = normalized.get("mode", "snn")
    if str(mode) != "snn":
        raise ValueError(
            f"apply_peod_cuadrado_unet_guardrails: mode={mode!r} — "
            "this field must be fixed at 'snn' and must not appear in the search space. "
            "A framework bug may have leaked it as a gene."
        )
    normalized["mode"] = "snn"

    normalized["image_height"] = 720
    normalized["image_width"] = 1280

    kernel_size = int(normalized.get("kernel_size", 7))
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(f"PEOD Cuadrado kernel_size must be a positive odd integer, got {kernel_size}")
    normalized["kernel_size"] = kernel_size

    multiply_factor = float(normalized.get("multiply_factor", 10.0))
    if multiply_factor <= 0.0:
        raise ValueError(f"PEOD Cuadrado multiply_factor must be > 0, got {multiply_factor}")
    normalized["multiply_factor"] = multiply_factor

    smooth_weight = float(normalized.get("smooth_weight", 1.0))
    if smooth_weight < 0.0:
        raise ValueError(f"smooth_weight must be >= 0, got {smooth_weight}")
    normalized["smooth_weight"] = smooth_weight

    loss_weight = float(normalized.get("loss_weight", 0.1))
    if loss_weight <= 0.0:
        raise ValueError(f"loss_weight must be > 0, got {loss_weight}")
    normalized["loss_weight"] = loss_weight

    return CuadradoUNetConstraintResult(
        genome=normalized,
        valid=True,
        repairs=repairs,
        warnings=warnings,
    )


def make_peod_cuadrado_unet_target(search_space: Mapping[str, Any] | None = None):
    from smcfnet.evolution.targets import PEODCuadradoUNetTarget

    return PEODCuadradoUNetTarget(search_space=dict(search_space or PEOD_CUADRADO_UNET_SEARCH_SPACE))


def make_peod_cuadrado_unet_v2_target(search_space: Mapping[str, Any] | None = None):
    from smcfnet.evolution.targets import PEODCuadradoUNetTarget

    return PEODCuadradoUNetTarget(
        search_space=dict(search_space or PEOD_CUADRADO_UNET_SEARCH_SPACE),
        name="cuadrado_unet_peod_v2",
        architecture_name="cuadrado_unet_peod_v2",
    )
