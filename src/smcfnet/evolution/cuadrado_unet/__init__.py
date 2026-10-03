"""Cuadrado U-Net evolutionary-search presets and constraints."""

from smcfnet.evolution.cuadrado_unet.search_space import (
    CUADRADO_UNET_CROSSOVER_GROUPS,
    CUADRADO_UNET_GENE_SPECS,
    CUADRADO_UNET_SEARCH_SPACE,
    CuadradoUNetConstraintResult,
    apply_cuadrado_unet_guardrails,
    make_cuadrado_unet_target,
)

__all__ = [
    "CUADRADO_UNET_CROSSOVER_GROUPS",
    "CUADRADO_UNET_GENE_SPECS",
    "CUADRADO_UNET_SEARCH_SPACE",
    "CuadradoUNetConstraintResult",
    "apply_cuadrado_unet_guardrails",
    "make_cuadrado_unet_target",
]