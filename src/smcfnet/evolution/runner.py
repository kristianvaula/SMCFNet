"""Command-line runner for evolutionary MCM/OFE searches."""

from __future__ import annotations

import argparse
import copy
import random
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import yaml

from smcfnet.config.config import Config, config_to_dict, load_config_from_yaml, update_config_from_dict
from smcfnet.evolution.archive import SearchArchive
from smcfnet.evolution.evaluator import TrainerBackedEvaluator
from smcfnet.evolution.metrics import (
    FLOW_CENTERED_RATIO_MIN,
    FLOW_MAG_MIN,
    FLOW_MEAN_RATIO_MAX,
    FLOW_SPATIAL_STD_MIN,
)
from smcfnet.evolution.cuadrado_unet.search_space import (
    CUADRADO_UNET_CROSSOVER_GROUPS,
    CUADRADO_UNET_GENE_SPECS,
    CUADRADO_UNET_SEARCH_SPACE,
    PEOD_CUADRADO_UNET_CROSSOVER_GROUPS,
    PEOD_CUADRADO_UNET_GENE_SPECS,
    PEOD_CUADRADO_UNET_SEARCH_SPACE,
    make_cuadrado_unet_target,
    make_peod_cuadrado_unet_target,
    make_peod_cuadrado_unet_v2_target,
)
from smcfnet.evolution.firenet.search_space import (
    FIRENET_LIF_ARITHMETIC_BLEND_CROSSOVER_FIELDS,
    FIRENET_LIF_CROSSOVER_GROUPS,
    FIRENET_LIF_GENE_SPECS,
    FIRENET_LIF_LOG_BLEND_CROSSOVER_FIELDS,
    FIRENET_LIF_SEARCH_SPACE,
    make_firenet_lif_target,
)
from smcfnet.evolution.hagenaars_firenet.search_space import (
    HAGENAARS_FIRENET_LIF_ARITHMETIC_BLEND_CROSSOVER_FIELDS,
    HAGENAARS_FIRENET_LIF_CROSSOVER_GROUPS,
    HAGENAARS_FIRENET_LIF_GENE_SPECS,
    HAGENAARS_FIRENET_LIF_LOG_BLEND_CROSSOVER_FIELDS,
    HAGENAARS_FIRENET_LIF_SEARCH_SPACE,
    make_hagenaars_firenet_lif_target,
)
from smcfnet.evolution.framework import DistributedSearchContext, EvolutionarySearch, GroupedUniformCrossover, SensitivityAwareMutation
from smcfnet.evolution.ssnn_net.search_space import (
    SSNN_NET_CROSSOVER_GROUPS,
    SSNN_NET_GENE_SPECS,
    SSNN_NET_SEARCH_SPACE,
    SSNN_NET_V2_GENE_SPECS,
    SSNN_NET_V2_SEARCH_SPACE,
    make_ssnn_net_target,
    make_ssnn_net_v2_target,
)
from smcfnet.evolution.state import candidate_from_dict, decode_rng_state, load_search_state, save_search_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run evolutionary search for MCM/OFE candidates")
    parser.add_argument("--config", required=True, help="Base YAML config")
    parser.add_argument(
        "--target",
        default=None,
        choices=[
            "firenet_lif",
            "hagenaars_firenet_lif",
            "cuadrado_unet",
            "cuadrado_unet_peod",
            "cuadrado_unet_peod_v2",
            "ssnn_net",
            "ssnn_net_v2",
        ],
        help="Search target",
    )
    parser.add_argument("--output-dir", default=None, help="Evolution output root")
    parser.add_argument("--experiment-name", default=None, help="Evolution experiment name")
    parser.add_argument("--population-size", type=int, default=None)
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--elite-size", type=int, default=None)
    parser.add_argument("--offspring-size", type=int, default=None)
    parser.add_argument("--max-steps-per-candidate", type=int, default=100)
    parser.add_argument(
        "--rung-budgets",
        default=None,
        help="Comma-separated multi-fidelity budgets, e.g. 100,3000,12000. Overrides --max-steps-per-candidate for scheduling.",
    )
    parser.add_argument("--rung-promote-fraction", type=float, default=None)
    parser.add_argument("--min-rung-survivors", type=int, default=None)
    parser.add_argument(
        "--final-rung-validation-fitness",
        action="store_true",
        help="Run MCM validation at the final rung and use validation EPE as f_quality when available.",
    )
    parser.add_argument("--timeout-sec", type=float, default=None)
    parser.add_argument("--evaluator-max-retries", type=int, default=None)
    parser.add_argument("--evaluator-retry-backoff-sec", type=float, default=None)
    parser.add_argument(
        "--evaluator-retry-failure-reasons",
        default=None,
        help="Comma-separated evaluator failure classes to retry",
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=None, help="Override data loader workers for candidate runs")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Resume from <evolution_root>/state.json")
    parser.add_argument("--resume-from", default=None, help="Explicit state JSON to resume from")
    parser.add_argument(
        "--initial-population-source",
        choices=["random", "base_genome", "config", "state"],
        default=None,
        help="How to seed generation 0 before filling with local mutations",
    )
    parser.add_argument(
        "--initial-population-state-path",
        default=None,
        help="Previous EA state.json, evolution directory, or run directory used when --initial-population-source=state",
    )
    parser.add_argument(
        "--flow-index-start",
        type=int,
        default=None,
        help="First per-sequence DSEC-flow frame exposed to candidate training",
    )
    parser.add_argument(
        "--flow-index-count",
        type=int,
        default=None,
        help="Maximum per-sequence DSEC-flow frames exposed to candidate training",
    )
    parser.add_argument(
        "--flow-index-stride",
        type=int,
        default=None,
        help="Per-sequence stride for DSEC-flow candidate windows",
    )
    parser.add_argument("--python", default=None, help="Python executable for candidate subprocesses")
    parser.add_argument("--crossover-probability", type=float, default=0.6)
    parser.add_argument("--group-swap-probability", type=float, default=0.5)
    parser.add_argument("--gene-swap-probability", type=float, default=0.5)
    parser.add_argument("--mutation-heavy-p", type=float, default=0.16)
    parser.add_argument("--mutation-medium-p", type=float, default=0.20)
    parser.add_argument("--mutation-light-p", type=float, default=0.20)
    parser.add_argument("--mutation-heavy-random-jump-p", type=float, default=0.25)
    parser.add_argument("--mutation-medium-random-jump-p", type=float, default=0.30)
    parser.add_argument("--mutation-light-random-jump-p", type=float, default=0.70)
    parser.add_argument("--min-mutations-per-child", type=int, default=2)
    parser.add_argument("--max-mutations-per-child", type=int, default=5)
    return parser.parse_args()


def load_config(path: str | Path) -> Config:
    cfg = Config()
    update_config_from_dict(cfg, load_config_from_yaml(str(path)))
    return cfg


def _parse_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value if int(item) > 0]
    text = str(value).strip()
    if not text:
        return []
    return [int(part.strip()) for part in text.split(",") if part.strip() and int(part.strip()) > 0]


def _is_hybrid_partition(cfg: Config) -> bool:
    return str(getattr(cfg.model.mcm, "partition_mode", "") or "").lower() == "hybrid"


def _remove_step_time_us_from_search_space(search_space: Dict[str, Any]) -> Dict[str, Any]:
    stripped = copy.deepcopy(dict(search_space or {}))
    mcm_space = stripped.get("mcm")
    if isinstance(mcm_space, dict):
        mcm_space.pop("step_time_us", None)
    return stripped


def _strip_step_time_us_for_hybrid(cfg: Config, target_name: str) -> None:
    """Remove fixed-time-only genes from hybrid streaming EA runs."""

    if target_name not in {"firenet_lif", "hagenaars_firenet_lif", "ssnn_net", "ssnn_net_v2"}:
        return
    if not _is_hybrid_partition(cfg):
        return
    cfg.model.mcm.step_time_us = None
    cfg.evolution.search_space = _remove_step_time_us_from_search_space(cfg.evolution.search_space or {})


def _filter_gene_specs_for_partition(cfg: Config, gene_specs: Dict[Any, Any]) -> Dict[Any, Any]:
    if not _is_hybrid_partition(cfg):
        return dict(gene_specs)
    return {
        path: spec
        for path, spec in dict(gene_specs).items()
        if tuple(spec.path) != ("mcm", "step_time_us")
    }


def _sync_gene_specs_with_search_space(
    cfg: Config, gene_specs: Dict[Any, Any]
) -> Dict[Any, Any]:
    """Rebuild each categorical GeneSpec's allowed values from cfg.evolution.search_space.

    The mutation operator's permissible moves otherwise come from the static,
    hardcoded per-target search-space module (e.g. SSNN_NET_V2_GENE_SPECS),
    completely independent of whatever a run's YAML config sets under
    ``evolution.search_space``. Without this sync, a config that narrows a
    gene to a single fixed option (e.g. restricting ssnn_encoder_signal to
    just ``["spike"]`` so the search never reconsiders the graded "membrane"
    mode) would still have that option silently reintroduced by mutation,
    since mutation reads from the unrestricted static defaults rather than
    the config override. This makes ``evolution.search_space`` the single
    source of truth for both initial sampling and ongoing mutation.
    """
    search_space = dict(cfg.evolution.search_space or {})
    synced: Dict[Any, Any] = {}
    for key, spec in dict(gene_specs).items():
        group_dict = search_space.get(spec.path[0]) if spec.path else None
        override_values = (
            group_dict.get(spec.path[-1])
            if isinstance(group_dict, dict) and len(spec.path) >= 2
            else None
        )
        if spec.kind == "categorical" and override_values is not None:
            synced[key] = replace(spec, values=tuple(copy.deepcopy(list(override_values))))
        else:
            synced[key] = spec
    return synced


def save_run_config(
    *,
    cfg: Config,
    args: argparse.Namespace,
    root_dir: Path,
    target_name: str,
    population_size: int,
    generations: int,
    elite_size: int,
    offspring_size: int,
    seed: int,
    rung_budgets: list[int],
    evaluator_budget: int,
) -> None:
    """Persist the resolved EA run configuration at search start."""

    cfg.evolution.target = str(target_name)
    cfg.evolution.population_size = int(population_size)
    cfg.evolution.generations = int(generations)
    cfg.evolution.elite_size = int(elite_size)
    cfg.evolution.offspring_size = int(offspring_size)
    cfg.evolution.seed = int(seed)
    cfg.evolution.rung_budgets = list(rung_budgets)

    resolved = config_to_dict(cfg)
    resolved["evolution_runtime"] = {
        "config_path": str(args.config),
        "target": str(target_name),
        "root_dir": str(root_dir),
        "population_size": int(population_size),
        "generations": int(generations),
        "elite_size": int(elite_size),
        "offspring_size": int(offspring_size),
        "seed": int(seed),
        "rung_budgets": list(rung_budgets),
        "evaluator_budget": int(evaluator_budget),
        "max_steps_per_candidate": int(args.max_steps_per_candidate),
        "timeout_sec": None if args.timeout_sec is None else float(args.timeout_sec),
        "device": str(args.device),
        "num_gpus": int(args.num_gpus),
        "num_workers": None if args.num_workers is None else int(args.num_workers),
        "initial_population_source": str(cfg.evolution.initial_population_source),
        "initial_population_state_path": getattr(cfg.evolution, "initial_population_state_path", None),
        "final_rung_validation_fitness": bool(cfg.evolution.final_rung_validation_fitness),
        "flow_index_start": getattr(cfg.data, "flow_index_start", None),
        "flow_index_count": getattr(cfg.data, "flow_index_count", None),
        "flow_index_stride": getattr(cfg.data, "flow_index_stride", None),
        "resume": bool(args.resume),
        "resume_from": None if args.resume_from is None else str(args.resume_from),
        "python": str(args.python) if args.python else sys.executable,
        "crossover_probability": float(args.crossover_probability),
        "group_swap_probability": float(args.group_swap_probability),
        "gene_swap_probability": float(args.gene_swap_probability),
        "mutation_heavy_p": float(args.mutation_heavy_p),
        "mutation_medium_p": float(args.mutation_medium_p),
        "mutation_light_p": float(args.mutation_light_p),
        "mutation_heavy_random_jump_p": float(args.mutation_heavy_random_jump_p),
        "mutation_medium_random_jump_p": float(args.mutation_medium_random_jump_p),
        "mutation_light_random_jump_p": float(args.mutation_light_random_jump_p),
        "min_mutations_per_child": int(args.min_mutations_per_child),
        "max_mutations_per_child": int(args.max_mutations_per_child),
        "evaluator_max_retries": int(getattr(cfg.evolution, "evaluator_max_retries", 0)),
        "evaluator_retry_backoff_sec": float(getattr(cfg.evolution, "evaluator_retry_backoff_sec", 5.0)),
        "evaluator_retry_failure_reasons": list(getattr(cfg.evolution, "evaluator_retry_failure_reasons", []) or []),
    }

    config_path = root_dir / "config.yaml"
    config_path.write_text(yaml.dump(resolved, default_flow_style=False, sort_keys=False))


def build_target(name: str, search_space: Dict[str, Any]):
    if name == "firenet_lif":
        return make_firenet_lif_target(search_space or FIRENET_LIF_SEARCH_SPACE)
    if name == "hagenaars_firenet_lif":
        return make_hagenaars_firenet_lif_target(search_space or HAGENAARS_FIRENET_LIF_SEARCH_SPACE)
    if name == "cuadrado_unet":
        return make_cuadrado_unet_target(search_space or CUADRADO_UNET_SEARCH_SPACE)
    if name == "cuadrado_unet_peod":
        return make_peod_cuadrado_unet_target(search_space or PEOD_CUADRADO_UNET_SEARCH_SPACE)
    if name == "cuadrado_unet_peod_v2":
        return make_peod_cuadrado_unet_v2_target(search_space or PEOD_CUADRADO_UNET_SEARCH_SPACE)
    if name == "ssnn_net":
        return make_ssnn_net_target(search_space or SSNN_NET_SEARCH_SPACE)
    if name == "ssnn_net_v2":
        return make_ssnn_net_v2_target(search_space or SSNN_NET_V2_SEARCH_SPACE)
    raise ValueError(f"Unsupported evolution target: {name!r}")


def _mutation_genome(result: Any) -> Dict[str, Any]:
    if isinstance(result, tuple):
        return copy.deepcopy(dict(result[0] or {}))
    return copy.deepcopy(dict(result or {}))


def _deep_merge_dict(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = copy.deepcopy(dict(base or {}))
    for key, value in dict(override or {}).items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _candidate_sort_key(candidate: Any) -> tuple[float, ...]:
    fitness = getattr(candidate, "fitness", None)
    if fitness is None:
        return (2.0, float("inf"))
    validity_penalty = 0.0 if bool(getattr(fitness, "valid", False)) else 1.0
    values = tuple(float(value) for value in (fitness.as_minimization_tuple() or (float("inf"),)))
    return (validity_penalty, *values)


def _resolve_initial_population_state_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    if (candidate / "state.json").is_file():
        return candidate / "state.json"
    if (candidate / "evolution" / "state.json").is_file():
        return candidate / "evolution" / "state.json"
    raise FileNotFoundError(f"Initial population state file not found: {candidate}")


def _load_state_seed_genomes(
    *,
    cfg: Config,
    target: Any,
    target_name: str,
    state_path: str | Path,
    population_size: int,
    mutation: Any,
    rng: random.Random,
) -> list[Dict[str, Any]]:
    resolved_state_path = _resolve_initial_population_state_path(state_path)
    state = load_search_state(resolved_state_path)
    if str(state.get("target")) != str(target_name):
        raise ValueError(
            f"Initial population state target mismatch: state has {state.get('target')!r}, "
            f"runner target is {target_name!r}"
        )

    candidates = [candidate_from_dict(item) for item in state.get("population", [])]
    if not candidates:
        raise ValueError(f"Initial population state has no saved population: {resolved_state_path}")

    config_seed = target.genome_from_config(cfg) if hasattr(target, "genome_from_config") else target.default_genome()
    ordered = sorted(candidates, key=_candidate_sort_key)
    initial: list[Dict[str, Any]] = []
    for candidate in ordered[: int(population_size)]:
        merged = _deep_merge_dict(config_seed, candidate.genome)
        initial.append(target.normalize_genome(merged))

    mutation_parent = initial[0]
    while len(initial) < int(population_size):
        mutated = _mutation_genome(mutation(mutation_parent, rng)) if mutation is not None else copy.deepcopy(mutation_parent)
        initial.append(target.normalize_genome(mutated))
    return initial[: int(population_size)]


def build_initial_genomes(
    *,
    cfg: Config,
    target: Any,
    target_name: str,
    source: str,
    population_size: int,
    mutation: Any,
    rng: random.Random,
    state_path: str | Path | None = None,
) -> list[Dict[str, Any]] | None:
    """Build a seeded generation-0 population.

    Candidate 0 is the seed genome. Remaining candidates are local mutation
    variants, so the initial population explores a neighborhood instead of
    discarding a known workable FireNet-LIF configuration.
    """

    source = str(source or "random")
    if source == "random":
        return None

    if source == "base_genome":
        if not cfg.evolution.base_genome:
            raise ValueError("evolution.initial_population_source='base_genome' requires evolution.base_genome")
        seed_genome = target.normalize_genome(cfg.evolution.base_genome)
    elif source == "config":
        if not hasattr(target, "genome_from_config"):
            raise ValueError(f"Target {target_name!r} does not support config-seeded initial populations")
        seed_genome = target.genome_from_config(cfg)
    elif source == "state":
        if state_path is None:
            raise ValueError("evolution.initial_population_source='state' requires initial_population_state_path")
        return _load_state_seed_genomes(
            cfg=cfg,
            target=target,
            target_name=target_name,
            state_path=state_path,
            population_size=population_size,
            mutation=mutation,
            rng=rng,
        )
    else:
        raise ValueError(f"Unsupported initial population source: {source!r}")

    initial = [target.normalize_genome(seed_genome)]
    while len(initial) < int(population_size):
        mutated = _mutation_genome(mutation(seed_genome, rng)) if mutation is not None else copy.deepcopy(seed_genome)
        initial.append(target.normalize_genome(mutated))
    return initial[: int(population_size)]


def build_state_saver(*, path: Path, target_name: str, rng: random.Random):
    def _save(generation, population, history, phase):
        save_search_state(
            path,
            target=target_name,
            generation=int(generation),
            population=population,
            rng=rng,
            extra={"history": list(history), "phase": str(phase)},
        )

    return _save


def _resume_state_is_completed_generation(
    resume_state: Mapping[str, Any],
    *,
    resume_generation: int,
    resume_history: Sequence[Mapping[str, Any]],
) -> bool:
    """Return True when a saved state represents selected survivors after a generation.

    Older final saves did not persist the explicit ``phase`` marker, so completed
    historical runs are detected from their final history entry as well.
    """

    extra = dict(resume_state.get("extra") or {})
    phase = str(extra.get("phase") or "")
    if phase in {"complete", "generation_end"}:
        return True
    if phase:
        return False
    if not resume_history:
        return False
    try:
        last_generation = int(dict(resume_history[-1]).get("generation", -1))
    except Exception:
        return False
    return int(last_generation) == int(resume_generation)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    target_name = str(args.target or cfg.evolution.target or "firenet_lif")
    population_size = int(args.population_size or cfg.evolution.population_size)
    generations = int(args.generations if args.generations is not None else cfg.evolution.generations)
    elite_size = int(args.elite_size or cfg.evolution.elite_size)
    offspring_size = int(args.offspring_size if args.offspring_size is not None else cfg.evolution.offspring_size)
    seed = int(args.seed if args.seed is not None else (cfg.evolution.seed if cfg.evolution.seed is not None else cfg.seed))

    if args.output_dir is not None:
        cfg.log.output_dir = str(args.output_dir)
    if args.experiment_name is not None:
        cfg.log.experiment_name = str(args.experiment_name)
    if args.initial_population_source is not None:
        cfg.evolution.initial_population_source = str(args.initial_population_source)
    if args.initial_population_state_path is not None:
        cfg.evolution.initial_population_state_path = str(args.initial_population_state_path)
    if args.rung_budgets is not None:
        cfg.evolution.rung_budgets = _parse_int_list(args.rung_budgets)
    if args.rung_promote_fraction is not None:
        cfg.evolution.rung_promote_fraction = float(args.rung_promote_fraction)
    if args.min_rung_survivors is not None:
        cfg.evolution.min_rung_survivors = int(args.min_rung_survivors)
    if args.final_rung_validation_fitness:
        cfg.evolution.final_rung_validation_fitness = True
    if args.evaluator_max_retries is not None:
        cfg.evolution.evaluator_max_retries = int(args.evaluator_max_retries)
    if args.evaluator_retry_backoff_sec is not None:
        cfg.evolution.evaluator_retry_backoff_sec = float(args.evaluator_retry_backoff_sec)
    if args.evaluator_retry_failure_reasons is not None:
        cfg.evolution.evaluator_retry_failure_reasons = [
            str(item).strip()
            for item in str(args.evaluator_retry_failure_reasons).split(",")
            if str(item).strip()
        ]
    if args.flow_index_start is not None:
        cfg.data.flow_index_start = int(args.flow_index_start)
    if args.flow_index_count is not None:
        cfg.data.flow_index_count = int(args.flow_index_count)
    if args.flow_index_stride is not None:
        cfg.data.flow_index_stride = int(args.flow_index_stride)
    if args.num_workers is not None:
        cfg.data.num_workers = max(0, int(args.num_workers))
    _strip_step_time_us_for_hybrid(cfg, target_name)

    root_dir = Path(cfg.log.output_dir) / str(cfg.log.experiment_name) / "evolution"
    root_dir.mkdir(parents=True, exist_ok=True)

    target = build_target(target_name, dict(cfg.evolution.search_space or {}))
    if _is_hybrid_partition(cfg) and target_name in {"firenet_lif", "hagenaars_firenet_lif", "ssnn_net", "ssnn_net_v2"}:
        target.search_space = _remove_step_time_us_from_search_space(target.search_space)
        setattr(target, "omit_step_time_us", True)
    elif target_name == "ssnn_net_v2":
        setattr(target, "omit_step_time_us", False)
    rung_budgets = _parse_int_list(getattr(cfg.evolution, "rung_budgets", []))
    evaluator_budget = int(rung_budgets[-1] if rung_budgets else args.max_steps_per_candidate)
    save_run_config(
        cfg=cfg,
        args=args,
        root_dir=root_dir,
        target_name=target_name,
        population_size=population_size,
        generations=generations,
        elite_size=elite_size,
        offspring_size=offspring_size,
        seed=seed,
        rung_budgets=rung_budgets,
        evaluator_budget=evaluator_budget,
    )
    evaluator = TrainerBackedEvaluator(
        root_dir=root_dir,
        max_steps_per_candidate=int(evaluator_budget),
        timeout_sec=args.timeout_sec,
        python_executable=str(args.python) if args.python else sys.executable,
        device=str(args.device),
        num_gpus=int(args.num_gpus),
        wall_ms_soft_limit=float(getattr(cfg.evolution, "fitness_wall_ms_soft_limit", 1000.0)),
        val_quality_key=getattr(cfg.evolution, "fitness_val_quality_key", None),
        val_max_batches=int(getattr(cfg.evolution, "fitness_val_max_batches", 50)),
        flow_mag_min=float(getattr(cfg.evolution, "fitness_flow_mag_min", FLOW_MAG_MIN)),
        flow_centered_ratio_min=float(
            getattr(cfg.evolution, "fitness_flow_centered_ratio_min", FLOW_CENTERED_RATIO_MIN)
        ),
        flow_mean_ratio_max=float(
            getattr(cfg.evolution, "fitness_flow_mean_ratio_max", FLOW_MEAN_RATIO_MAX)
        ),
        flow_spatial_std_min=float(
            getattr(cfg.evolution, "fitness_flow_spatial_std_min", FLOW_SPATIAL_STD_MIN)
        ),
        constraint_violation_weight=float(
            getattr(cfg.evolution, "fitness_constraint_violation_weight", 0.05)
        ),
        max_retries=int(getattr(cfg.evolution, "evaluator_max_retries", 0)),
        retry_backoff_sec=float(getattr(cfg.evolution, "evaluator_retry_backoff_sec", 5.0)),
        retry_failure_reasons=tuple(getattr(cfg.evolution, "evaluator_retry_failure_reasons", []) or []),
        cwd=Path.cwd(),
    )
    rng = random.Random(seed)
    resume_state = None
    resume_path = Path(args.resume_from) if args.resume_from else (root_dir / "state.json")
    if args.resume or args.resume_from:
        if not resume_path.exists():
            raise FileNotFoundError(f"Cannot resume evolution search; state file not found: {resume_path}")
        resume_state = load_search_state(resume_path)
        if str(resume_state.get("target")) != str(target_name):
            raise ValueError(
                f"Resume state target mismatch: state has {resume_state.get('target')!r}, "
                f"runner target is {target_name!r}"
            )
        if resume_state.get("rng_state"):
            rng.setstate(decode_rng_state(str(resume_state["rng_state"])))
    context = DistributedSearchContext.from_torch_distributed()
    crossover = None
    mutation = None
    if target_name in {
        "firenet_lif",
        "hagenaars_firenet_lif",
        "cuadrado_unet",
        "cuadrado_unet_peod",
        "cuadrado_unet_peod_v2",
        "ssnn_net",
        "ssnn_net_v2",
    }:
        if target_name == "hagenaars_firenet_lif":
            crossover_groups = HAGENAARS_FIRENET_LIF_CROSSOVER_GROUPS
            log_blend_paths = HAGENAARS_FIRENET_LIF_LOG_BLEND_CROSSOVER_FIELDS
            arithmetic_blend_paths = HAGENAARS_FIRENET_LIF_ARITHMETIC_BLEND_CROSSOVER_FIELDS
            gene_specs = HAGENAARS_FIRENET_LIF_GENE_SPECS
        elif target_name == "cuadrado_unet":
            crossover_groups = CUADRADO_UNET_CROSSOVER_GROUPS
            log_blend_paths = ()
            arithmetic_blend_paths = ()
            gene_specs = CUADRADO_UNET_GENE_SPECS
        elif target_name in {"cuadrado_unet_peod", "cuadrado_unet_peod_v2"}:
            crossover_groups = PEOD_CUADRADO_UNET_CROSSOVER_GROUPS
            log_blend_paths = ()
            arithmetic_blend_paths = ()
            gene_specs = PEOD_CUADRADO_UNET_GENE_SPECS
        elif target_name == "ssnn_net":
            crossover_groups = SSNN_NET_CROSSOVER_GROUPS
            log_blend_paths = ()
            arithmetic_blend_paths = ()
            gene_specs = SSNN_NET_GENE_SPECS
        elif target_name == "ssnn_net_v2":
            crossover_groups = SSNN_NET_CROSSOVER_GROUPS
            log_blend_paths = ()
            arithmetic_blend_paths = ()
            gene_specs = {**SSNN_NET_GENE_SPECS, **SSNN_NET_V2_GENE_SPECS}
        else:
            crossover_groups = FIRENET_LIF_CROSSOVER_GROUPS
            log_blend_paths = FIRENET_LIF_LOG_BLEND_CROSSOVER_FIELDS
            arithmetic_blend_paths = FIRENET_LIF_ARITHMETIC_BLEND_CROSSOVER_FIELDS
            gene_specs = FIRENET_LIF_GENE_SPECS
        gene_specs = _filter_gene_specs_for_partition(cfg, gene_specs)
        gene_specs = _sync_gene_specs_with_search_space(cfg, gene_specs)
        crossover = GroupedUniformCrossover(
            crossover_groups,
            crossover_probability=float(args.crossover_probability),
            group_swap_probability=float(args.group_swap_probability),
            gene_swap_probability=float(args.gene_swap_probability),
            log_blend_paths=log_blend_paths,
            arithmetic_blend_paths=arithmetic_blend_paths,
            record_metadata=True,
        )
        mutation = SensitivityAwareMutation(
            gene_specs,
            mutation_p={
                "heavy": float(args.mutation_heavy_p),
                "medium": float(args.mutation_medium_p),
                "light": float(args.mutation_light_p),
            },
            random_jump_p={
                "heavy": float(args.mutation_heavy_random_jump_p),
                "medium": float(args.mutation_medium_random_jump_p),
                "light": float(args.mutation_light_random_jump_p),
            },
            min_mutations=int(args.min_mutations_per_child),
            max_mutations=int(args.max_mutations_per_child),
        )
    archive = SearchArchive(
        root_dir=root_dir,
        budget_iters=int(evaluator_budget),
        seed=seed,
        use_wandb=bool(cfg.wandb.use_wandb),
        wandb_project=str(cfg.wandb.project_name),
        wandb_run_name=str(cfg.wandb.run_name or f"{cfg.log.experiment_name}_ea"),
        wandb_group=cfg.wandb.group,
        wandb_mode=str(cfg.wandb.mode),
    )
    state_saver = build_state_saver(path=root_dir / "state.json", target_name=target.name, rng=rng)

    engine = EvolutionarySearch(
        base_config=cfg,
        target=target,
        evaluator=evaluator,
        population_size=population_size,
        generations=generations,
        elite_size=elite_size,
        offspring_size=offspring_size,
        crossover=crossover,
        mutation=mutation,
        context=context,
        rng=rng,
        on_candidate_evaluated=archive.on_candidate_evaluated,
        on_generation_end=archive.on_generation_end,
        on_state_update=state_saver,
        rung_budgets=rung_budgets,
        rung_promote_fraction=float(getattr(cfg.evolution, "rung_promote_fraction", 0.5)),
        min_rung_survivors=int(getattr(cfg.evolution, "min_rung_survivors", 1)),
        final_rung_validation_fitness=bool(getattr(cfg.evolution, "final_rung_validation_fitness", False)),
    )

    resume_population = None
    resume_generation = 0
    resume_history = None
    initial_genomes = None
    if resume_state is not None:
        resume_population = [candidate_from_dict(item) for item in resume_state.get("population", [])]
        resume_generation = int(resume_state.get("generation", 0))
        resume_history = list(dict(resume_state.get("extra") or {}).get("history") or [])
        if (
            generations > resume_generation
            and resume_population
            and _resume_state_is_completed_generation(
                resume_state,
                resume_generation=resume_generation,
                resume_history=resume_history,
            )
        ):
            selected = list(resume_population)
            elites = selected[: min(elite_size, len(selected))]
            offspring = engine._breed(selected, generation=resume_generation + 1)
            resume_population = (list(elites) + offspring)[:population_size]
            resume_generation += 1
    else:
        initial_population_source = str(getattr(cfg.evolution, "initial_population_source", "random"))
        if initial_population_source == "random" and cfg.evolution.base_genome:
            initial_population_source = "base_genome"

        initial_genomes = build_initial_genomes(
            cfg=cfg,
            target=target,
            target_name=target_name,
            source=initial_population_source,
            state_path=getattr(cfg.evolution, "initial_population_state_path", None),
            population_size=population_size,
            mutation=mutation,
            rng=rng,
        )

    result = engine.run(
        initial_genomes=initial_genomes,
        initial_population=resume_population,
        start_generation=resume_generation,
        history=resume_history,
    )
    save_search_state(
        root_dir / "state.json",
        target=target.name,
        generation=generations,
        population=result.population,
        rng=rng,
        extra={"history": result.history, "phase": "complete"},
    )

    if result.best is not None and result.best.fitness is not None:
        print(f"Best candidate: {result.best.identifier}")
        print(f"Objectives: {result.best.fitness.objectives}")
        print(f"Valid: {result.best.fitness.valid}")
        print(f"Metrics: {result.best.fitness.metrics}")
    else:
        print("No evaluated candidate produced a fitness result")


if __name__ == "__main__":
    main()
