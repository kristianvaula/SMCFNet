"""SSNN evolutionary-search presets and constraints."""

from smcfnet.evolution.ssnn_net.search_space import (
    SSNN_NET_CROSSOVER_GROUPS,
    SSNN_NET_GENE_SPECS,
    SSNN_NET_SEARCH_SPACE,
    SSNN_NET_V2_GENE_SPECS,
    SSNN_NET_V2_SEARCH_SPACE,
    SSNNNetConstraintResult,
    apply_ssnn_net_guardrails,
    make_ssnn_net_target,
    make_ssnn_net_v2_target,
)

__all__ = [
    "SSNN_NET_CROSSOVER_GROUPS",
    "SSNN_NET_GENE_SPECS",
    "SSNN_NET_SEARCH_SPACE",
    "SSNN_NET_V2_GENE_SPECS",
    "SSNN_NET_V2_SEARCH_SPACE",
    "SSNNNetConstraintResult",
    "apply_ssnn_net_guardrails",
    "make_ssnn_net_target",
    "make_ssnn_net_v2_target",
]
