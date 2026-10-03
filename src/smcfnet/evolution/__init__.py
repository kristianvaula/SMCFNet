"""Generic evolutionary search scaffolding for architecture search."""

from smcfnet.evolution.framework import (
    Candidate,
    DistributedSearchContext,
    EvolutionResult,
    EvolutionarySearch,
    Fitness,
    GeneSpec,
    GroupedUniformCrossover,
    ParetoSelection,
    RandomCrossover,
    RandomMutation,
    SensitivityAwareMutation,
)
from smcfnet.evolution.evaluator import TrainerBackedEvaluator
from smcfnet.evolution.archive import SearchArchive
from smcfnet.evolution.analysis import analyze_ea_archive
from smcfnet.evolution.cuadrado_unet.search_space import (
    CUADRADO_UNET_CROSSOVER_GROUPS,
    CUADRADO_UNET_GENE_SPECS,
    CUADRADO_UNET_SEARCH_SPACE,
    CuadradoUNetConstraintResult,
    apply_cuadrado_unet_guardrails,
    make_cuadrado_unet_target,
)
from smcfnet.evolution.metrics import (
    build_convergence_fitness,
    parse_mcm_log_metrics,
    parse_mcm_validation_log_metrics,
    summarize_metric_rows,
)
from smcfnet.evolution.search_spaces import expand_parameter_grid, sample_parameter_space
from smcfnet.evolution.state import candidate_from_dict, load_search_state, save_search_state
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
from smcfnet.evolution.targets import (
    CuadradoUNetTarget,
    FireNetLIFTarget,
    HagenaarsFireNetLIFTarget,
    SSNNNetTarget,
    SSNNNetV2Target,
)

__all__ = [
    "Candidate",
    "DistributedSearchContext",
    "EvolutionResult",
    "EvolutionarySearch",
    "Fitness",
    "GeneSpec",
    "GroupedUniformCrossover",
    "ParetoSelection",
    "RandomCrossover",
    "RandomMutation",
    "SensitivityAwareMutation",
    "SearchArchive",
    "TrainerBackedEvaluator",
    "analyze_ea_archive",
    "CUADRADO_UNET_CROSSOVER_GROUPS",
    "CUADRADO_UNET_GENE_SPECS",
    "CUADRADO_UNET_SEARCH_SPACE",
    "CuadradoUNetConstraintResult",
    "build_convergence_fitness",
    "expand_parameter_grid",
    "sample_parameter_space",
    "load_search_state",
    "candidate_from_dict",
    "parse_mcm_log_metrics",
    "parse_mcm_validation_log_metrics",
    "save_search_state",
    "summarize_metric_rows",
    "SSNN_NET_CROSSOVER_GROUPS",
    "SSNN_NET_GENE_SPECS",
    "SSNN_NET_SEARCH_SPACE",
    "SSNN_NET_V2_GENE_SPECS",
    "SSNN_NET_V2_SEARCH_SPACE",
    "SSNNNetConstraintResult",
    "apply_cuadrado_unet_guardrails",
    "apply_ssnn_net_guardrails",
    "make_cuadrado_unet_target",
    "make_ssnn_net_target",
    "make_ssnn_net_v2_target",
    "CuadradoUNetTarget",
    "FireNetLIFTarget",
    "HagenaarsFireNetLIFTarget",
    "SSNNNetTarget",
    "SSNNNetV2Target",
]
