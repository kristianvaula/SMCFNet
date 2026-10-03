"""Generic EA framework used by DDP-backed architecture search jobs."""

from __future__ import annotations

import copy
import dataclasses
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple

import torch

from smcfnet.config.config import Config
from smcfnet.evolution.search_spaces import sample_parameter_space


Genome = Dict[str, Any]


@dataclass
class Fitness:
    """Multi-objective fitness payload.

    Objective values are stored in a fixed order. ``maximize`` flags indicate
    whether the corresponding objective should be maximized instead of
    minimized when comparing candidates.
    """

    metrics: Dict[str, float] = field(default_factory=dict)
    objectives: Tuple[float, ...] = ()
    maximize: Tuple[bool, ...] = ()
    valid: bool = True
    notes: Dict[str, Any] = field(default_factory=dict)

    def as_minimization_tuple(self) -> Tuple[float, ...]:
        values: List[float] = []
        for index, objective in enumerate(self.objectives):
            maximize = bool(self.maximize[index]) if index < len(self.maximize) else False
            values.append(-float(objective) if maximize else float(objective))
        return tuple(values)


@dataclass
class Candidate:
    """One genome in the search population."""

    genome: Genome
    fitness: Optional[Fitness] = None
    generation: int = 0
    parents: Tuple[str, ...] = ()
    identifier: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvolutionResult:
    """Result bundle returned by a search run."""

    best: Optional[Candidate]
    population: List[Candidate]
    history: List[Dict[str, Any]] = field(default_factory=list)


class SearchTarget(Protocol):
    """Architecture-specific bridge between genomes and concrete configs."""

    name: str

    def default_genome(self) -> Genome:
        ...

    def normalize_genome(self, genome: Mapping[str, Any]) -> Genome:
        ...

    def apply_to_config(self, cfg: Config, genome: Mapping[str, Any]) -> None:
        ...


class Evaluator(Protocol):
    """Consumes a concrete config and returns a fitness object."""

    def __call__(
        self,
        *,
        cfg: Config,
        target: SearchTarget,
        candidate: Candidate,
        context: "DistributedSearchContext",
    ) -> Fitness:
        ...


class MutationOperator(Protocol):
    def __call__(self, genome: Genome, rng: random.Random) -> Genome | Tuple[Genome, Mapping[str, Any]]:
        ...


class CrossoverOperator(Protocol):
    def __call__(self, left: Genome, right: Genome, rng: random.Random) -> Genome | Tuple[Genome, Mapping[str, Any]]:
        ...


class SelectionOperator(Protocol):
    def __call__(self, population: Sequence[Candidate], limit: int) -> List[Candidate]:
        ...


@dataclass
class DistributedSearchContext:
    """Small helper around torch.distributed for search-time synchronization."""

    rank: int = 0
    world_size: int = 1
    initialized: bool = False

    @classmethod
    def from_torch_distributed(cls) -> "DistributedSearchContext":
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return cls(
                rank=int(torch.distributed.get_rank()),
                world_size=int(torch.distributed.get_world_size()),
                initialized=True,
            )
        return cls()

    def is_distributed(self) -> bool:
        return bool(self.initialized and self.world_size > 1)

    def is_master(self) -> bool:
        return int(self.rank) == 0

    def barrier(self) -> None:
        if self.is_distributed():
            torch.distributed.barrier()

    def gather_metrics(self, metrics: Mapping[str, float]) -> Dict[str, float]:
        """All-gather metrics and average them on every rank.

        The caller stays responsible for rank-local evaluation. This helper is
        mainly for DDP jobs where every rank evaluates a shard and the final
        scalar metrics should be synchronized before selection/logging.
        """

        payload = {str(key): float(value) for key, value in metrics.items()}
        if not self.is_distributed():
            return payload

        gathered: List[Optional[Dict[str, float]]] = [None for _ in range(self.world_size)]
        torch.distributed.all_gather_object(gathered, payload)
        keys = {key for item in gathered if item is not None for key in item}
        reduced: Dict[str, float] = {}
        for key in keys:
            values = [float(item[key]) for item in gathered if item is not None and key in item]
            if values:
                reduced[key] = float(sum(values) / float(len(values)))
        return reduced


class RandomMutation:
    """No-op mutation helper that keeps the framework runnable out of the box."""

    def __call__(self, genome: Genome, rng: random.Random) -> Genome:
        del rng
        return copy.deepcopy(dict(genome))


@dataclass(frozen=True)
class GeneSpec:
    """Schema entry for a type-aware mutation gene."""

    path: Tuple[str, ...] | Sequence[str] | str
    kind: str
    sensitivity: str = "medium"
    values: Optional[Sequence[Any]] = None
    low: Optional[float | int] = None
    high: Optional[float | int] = None
    step: Optional[float | int] = None
    ordered: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _normalize_path(self.path))
        object.__setattr__(self, "kind", str(self.kind))
        object.__setattr__(self, "sensitivity", str(self.sensitivity))
        if self.values is not None:
            object.__setattr__(self, "values", tuple(copy.deepcopy(list(self.values))))


class SensitivityAwareMutation:
    """Type-aware local mutation over a nested genome schema."""

    def __init__(
        self,
        specs: Mapping[Sequence[str] | str, GeneSpec],
        *,
        mutation_p: Optional[Mapping[str, float]] = None,
        random_jump_p: Optional[Mapping[str, float]] = None,
        log_sigma: Optional[Mapping[str, float]] = None,
        min_mutations: int = 2,
        max_mutations: int = 5,
    ):
        self.specs = {tuple(spec.path): spec for spec in dict(specs or {}).values()}
        self.mutation_p = {"heavy": 0.16, "medium": 0.20, "light": 0.20}
        self.random_jump_p = {"heavy": 0.25, "medium": 0.30, "light": 0.70}
        self.log_sigma = {"heavy": 0.15, "medium": 0.30, "light": 0.50}
        if mutation_p:
            self.mutation_p.update({str(key): float(value) for key, value in mutation_p.items()})
        if random_jump_p:
            self.random_jump_p.update({str(key): float(value) for key, value in random_jump_p.items()})
        if log_sigma:
            self.log_sigma.update({str(key): float(value) for key, value in log_sigma.items()})
        self.min_mutations = max(0, int(min_mutations))
        self.max_mutations = max(self.min_mutations, int(max_mutations))

    def __call__(self, genome: Genome, rng: random.Random) -> Tuple[Genome, Dict[str, Any]]:
        child = copy.deepcopy(dict(genome))
        records: List[Dict[str, Any]] = []
        candidate_paths = list(self.specs.keys())
        rng.shuffle(candidate_paths)

        for path in candidate_paths:
            if len(records) >= self.max_mutations:
                break
            if not _has_path(child, path):
                continue
            spec = self.specs[path]
            probability = self.mutation_p.get(spec.sensitivity, 0.10)
            if rng.random() >= probability:
                continue
            self._try_mutate_path(child, spec, rng, records, forced=False)

        attempts = 0
        max_attempts = max(10, len(candidate_paths) * 4)
        while len(records) < self.min_mutations and candidate_paths and attempts < max_attempts:
            attempts += 1
            path = rng.choice(candidate_paths)
            if not _has_path(child, path):
                continue
            self._try_mutate_path(child, self.specs[path], rng, records, forced=True)

        metadata = {
            "mutation_op": "sensitivity_aware_type_mutation",
            "num_mutations": int(len(records)),
            "mutations": records,
        }
        return child, metadata

    def _try_mutate_path(
        self,
        child: Dict[str, Any],
        spec: GeneSpec,
        rng: random.Random,
        records: List[Dict[str, Any]],
        *,
        forced: bool,
    ) -> None:
        old = copy.deepcopy(_get_path(child, spec.path))
        new = self._mutate_value(old, spec, rng)
        if new == old:
            return
        _set_path(child, spec.path, new)
        record: Dict[str, Any] = {
            "path": ".".join(spec.path),
            "old": old,
            "new": copy.deepcopy(new),
            "kind": spec.kind,
            "sensitivity": spec.sensitivity,
        }
        if forced:
            record["forced"] = True
        records.append(record)

    def _mutate_value(self, old: Any, spec: GeneSpec, rng: random.Random) -> Any:
        if spec.kind == "bool":
            return not bool(old)
        if spec.kind == "categorical":
            return self._mutate_categorical(old, spec, rng)
        if spec.kind == "int":
            return self._mutate_int(old, spec, rng)
        if spec.kind == "float":
            return self._mutate_float(old, spec, rng)
        if spec.kind == "log_float":
            return self._mutate_log_float(old, spec, rng)
        raise ValueError(f"Unsupported gene kind: {spec.kind!r}")

    def _mutate_categorical(self, old: Any, spec: GeneSpec, rng: random.Random) -> Any:
        values = [copy.deepcopy(value) for value in (spec.values or ())]
        if not values:
            return old
        old_index = _option_index(old, values)
        if old_index is None:
            old_index = _nearest_numeric_option_index(old, values) if spec.ordered else None
        jump = rng.random() < self.random_jump_p.get(spec.sensitivity, 0.20)

        if old_index is not None and spec.ordered and not jump:
            neighbors = []
            if old_index > 0:
                neighbors.append(values[old_index - 1])
            if old_index < len(values) - 1:
                neighbors.append(values[old_index + 1])
            return copy.deepcopy(rng.choice(neighbors)) if neighbors else old

        choices = [value for value in values if value != old]
        return copy.deepcopy(rng.choice(choices)) if choices else old

    def _mutate_int(self, old: Any, spec: GeneSpec, rng: random.Random) -> int:
        old_i = int(old)
        if spec.values is not None:
            return int(self._mutate_categorical(old_i, spec, rng))
        low = int(spec.low if spec.low is not None else old_i - 1)
        high = int(spec.high if spec.high is not None else old_i + 1)
        step = max(1, int(spec.step or 1))
        jump = rng.random() < self.random_jump_p.get(spec.sensitivity, 0.20)
        if jump:
            values = list(range(low, high + 1, step))
            choices = [value for value in values if value != old_i]
            return int(rng.choice(choices)) if choices else old_i
        direction = rng.choice((-1, 1))
        return int(max(low, min(high, old_i + direction * step)))

    def _mutate_float(self, old: Any, spec: GeneSpec, rng: random.Random) -> float:
        old_f = float(old)
        low = float(spec.low if spec.low is not None else old_f - 1.0)
        high = float(spec.high if spec.high is not None else old_f + 1.0)
        sigma = {"heavy": 0.05, "medium": 0.10, "light": 0.20}.get(spec.sensitivity, 0.10)
        span = high - low
        return float(max(low, min(high, old_f + rng.gauss(0.0, sigma * span))))

    def _mutate_log_float(self, old: Any, spec: GeneSpec, rng: random.Random) -> float:
        old_f = max(float(old), 1e-12)
        low = float(spec.low if spec.low is not None else old_f * 0.1)
        high = float(spec.high if spec.high is not None else old_f * 10.0)
        sigma = self.log_sigma.get(spec.sensitivity, 0.30)
        return float(max(low, min(high, old_f * math.exp(rng.gauss(0.0, sigma)))))


class RandomCrossover:
    """Uniform left/right key mixing with left-biased defaults."""

    def __call__(self, left: Genome, right: Genome, rng: random.Random) -> Genome:
        merged = copy.deepcopy(dict(left))
        for key, value in right.items():
            if key not in merged or rng.random() < 0.5:
                merged[key] = copy.deepcopy(value)
        return merged


class GroupedUniformCrossover:
    """Grouped mixed-type uniform crossover over nested genome paths.

    Groups describe semantic blocks as nested paths. The operator itself is
    generic: target-specific code only supplies the path groups and optional
    numeric blend metadata.
    """

    def __init__(
        self,
        groups: Mapping[str, Sequence[Sequence[str] | str]],
        *,
        crossover_probability: float = 0.6,
        group_swap_probability: float = 0.5,
        gene_swap_probability: float = 0.5,
        log_blend_paths: Optional[Iterable[Sequence[str] | str]] = None,
        arithmetic_blend_paths: Optional[Iterable[Sequence[str] | str]] = None,
        blend_alpha: float = 0.5,
        record_metadata: bool = False,
    ):
        self.groups = {
            str(name): tuple(_normalize_path(path) for path in paths)
            for name, paths in dict(groups or {}).items()
        }
        self.crossover_probability = float(crossover_probability)
        self.group_swap_probability = float(group_swap_probability)
        self.gene_swap_probability = float(gene_swap_probability)
        self.log_blend_paths = {_normalize_path(path) for path in (log_blend_paths or ())}
        self.arithmetic_blend_paths = {_normalize_path(path) for path in (arithmetic_blend_paths or ())}
        self.blend_alpha = float(blend_alpha)
        self.record_metadata = bool(record_metadata)

    def __call__(self, left: Genome, right: Genome, rng: random.Random) -> Genome | Tuple[Genome, Dict[str, Any]]:
        if not self.groups or rng.random() > self.crossover_probability:
            child = copy.deepcopy(dict(left))
            if self.record_metadata:
                return child, {"crossover_op": "grouped_uniform", "performed": False, "groups": []}
            return child

        child = copy.deepcopy(dict(left))
        crossed_paths: set[Tuple[str, ...]] = set()
        group_records: List[Dict[str, Any]] = []
        for group_name, paths in self.groups.items():
            if rng.random() < self.group_swap_probability:
                donor = right if rng.random() < 0.5 else left
                donor_name = "right" if donor is right else "left"
                record = {
                    "group": str(group_name),
                    "mode": "group",
                    "donor": donor_name,
                    "paths": [".".join(path) for path in paths],
                }
                for path in paths:
                    if _has_path(donor, path):
                        _set_path(child, path, _get_path(donor, path))
                    crossed_paths.add(path)
                group_records.append(record)
                continue

            gene_records: List[Dict[str, Any]] = []
            for path in paths:
                crossed_paths.add(path)
                if rng.random() >= self.gene_swap_probability:
                    continue
                value = self._mix_gene(path, left, right, rng)
                if value is not _MISSING:
                    _set_path(child, path, value)
                    gene_records.append({"path": ".".join(path), "value": copy.deepcopy(value)})
            group_records.append({"group": str(group_name), "mode": "gene", "genes": gene_records})

        for key, value in right.items():
            if key not in child:
                child[str(key)] = copy.deepcopy(value)
        if self.record_metadata:
            return child, {
                "crossover_op": "grouped_uniform",
                "performed": True,
                "groups": group_records,
                "crossed_paths": [".".join(path) for path in sorted(crossed_paths)],
            }
        return child

    def _mix_gene(self, path: Tuple[str, ...], left: Genome, right: Genome, rng: random.Random) -> Any:
        if not _has_path(left, path) or not _has_path(right, path):
            source = right if _has_path(right, path) else left
            return copy.deepcopy(_get_path(source, path)) if _has_path(source, path) else _MISSING

        left_value = _get_path(left, path)
        right_value = _get_path(right, path)
        if path in self.log_blend_paths:
            blended = _blend_numeric(left_value, right_value, rng, alpha=self.blend_alpha, log_space=True)
            if blended is not _MISSING:
                return blended
        if path in self.arithmetic_blend_paths:
            blended = _blend_numeric(left_value, right_value, rng, alpha=self.blend_alpha, log_space=False)
            if blended is not _MISSING:
                return blended
        return copy.deepcopy(right_value if rng.random() < 0.5 else left_value)


class ParetoSelection:
    """Simple nondominated selection with deterministic tie-breaking."""

    def __call__(self, population: Sequence[Candidate], limit: int) -> List[Candidate]:
        evaluated = [candidate for candidate in population if candidate.fitness is not None]
        if not evaluated:
            return list(population[:limit])
        _assert_consistent_quality_source(evaluated)

        remaining = list(evaluated)
        survivors: List[Candidate] = []
        while remaining and len(survivors) < limit:
            front: List[Candidate] = []
            for candidate in remaining:
                if not any(_dominates(other, candidate) for other in remaining if other is not candidate):
                    front.append(candidate)
            front.sort(key=lambda cand: (_fitness_sort_key(cand), cand.identifier or ""))
            for candidate in front:
                if len(survivors) >= limit:
                    break
                survivors.append(candidate)
            remaining = [candidate for candidate in remaining if candidate not in front]
        return survivors


def _fitness_sort_key(candidate: Candidate) -> Tuple[float, ...]:
    if candidate.fitness is None:
        return (float("inf"),)
    validity_penalty = 0.0 if bool(candidate.fitness.valid) else 1.0
    return (validity_penalty, *(candidate.fitness.as_minimization_tuple() or (float("inf"),)))


def _dominates(left: Candidate, right: Candidate) -> bool:
    if left.fitness is None or right.fitness is None:
        return False
    if bool(left.fitness.valid) and not bool(right.fitness.valid):
        return True
    if not bool(left.fitness.valid) and bool(right.fitness.valid):
        return False
    left_values = left.fitness.as_minimization_tuple()
    right_values = right.fitness.as_minimization_tuple()
    if not left_values or not right_values or len(left_values) != len(right_values):
        return False
    return all(a <= b for a, b in zip(left_values, right_values)) and any(
        a < b for a, b in zip(left_values, right_values)
    )


def _assert_consistent_quality_source(population: Sequence[Candidate]) -> None:
    sources = {
        str(candidate.fitness.notes.get("quality_source"))
        for candidate in population
        if candidate.fitness is not None
        and candidate.fitness.notes.get("quality_source") not in (None, "")
    }
    if len(sources) > 1:
        raise ValueError(
            "ParetoSelection cannot compare candidates with mixed quality sources: "
            + ", ".join(sorted(sources))
        )


class _Missing:
    pass


_MISSING = _Missing()


def _normalize_path(path: Sequence[str] | str) -> Tuple[str, ...]:
    if isinstance(path, str):
        return tuple(part for part in path.split(".") if part)
    return tuple(str(part) for part in path)


def _has_path(mapping: Mapping[str, Any], path: Tuple[str, ...]) -> bool:
    current: Any = mapping
    for part in path:
        if not isinstance(current, Mapping) or part not in current:
            return False
        current = current[part]
    return True


def _get_path(mapping: Mapping[str, Any], path: Tuple[str, ...]) -> Any:
    current: Any = mapping
    for part in path:
        current = current[part]
    return current


def _set_path(mapping: Dict[str, Any], path: Tuple[str, ...], value: Any) -> None:
    current: Dict[str, Any] = mapping
    for part in path[:-1]:
        next_value = current.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            current[part] = next_value
        current = next_value
    current[path[-1]] = copy.deepcopy(value)


def _option_index(value: Any, options: Sequence[Any]) -> Optional[int]:
    for index, option in enumerate(options):
        if value == option:
            return int(index)
    return None


def _nearest_numeric_option_index(value: Any, options: Sequence[Any]) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric_options = [
        (index, float(option))
        for index, option in enumerate(options)
        if not isinstance(option, bool) and isinstance(option, (int, float))
    ]
    if not numeric_options:
        return None
    return min(numeric_options, key=lambda item: abs(float(value) - item[1]))[0]


def _blend_numeric(left: Any, right: Any, rng: random.Random, *, alpha: float, log_space: bool) -> Any:
    if isinstance(left, bool) or isinstance(right, bool):
        return _MISSING
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return _blend_scalar(float(left), float(right), rng, alpha=alpha, log_space=log_space, preserve_int=isinstance(left, int) and isinstance(right, int))
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)) and len(left) == len(right):
        blended: List[Any] = []
        for left_item, right_item in zip(left, right):
            item = _blend_numeric(left_item, right_item, rng, alpha=alpha, log_space=log_space)
            if item is _MISSING:
                return _MISSING
            blended.append(item)
        return blended if isinstance(left, list) else tuple(blended)
    return _MISSING


def _blend_scalar(left: float, right: float, rng: random.Random, *, alpha: float, log_space: bool, preserve_int: bool) -> float | int:
    if log_space:
        if left <= 0.0 or right <= 0.0:
            return _MISSING
        left_v = math.log(left)
        right_v = math.log(right)
    else:
        left_v = left
        right_v = right
    lam = rng.uniform(-float(alpha), 1.0 + float(alpha))
    value = lam * left_v + (1.0 - lam) * right_v
    if log_space:
        value = math.exp(value)
    if preserve_int:
        return int(round(value))
    return float(value)


class EvolutionarySearch:
    """Small orchestration layer for iterative architecture search.

    The engine intentionally stays agnostic about how a candidate is evaluated.
    The evaluator may run a local smoke test, a short trainer run, or a DDP
    job that reduces metrics across ranks before returning a ``Fitness``.
    """

    def __init__(
        self,
        *,
        base_config: Config,
        target: SearchTarget,
        evaluator: Evaluator,
        population_size: int = 8,
        generations: int = 1,
        elite_size: int = 2,
        offspring_size: Optional[int] = None,
        mutation: Optional[MutationOperator] = None,
        crossover: Optional[CrossoverOperator] = None,
        selection: Optional[SelectionOperator] = None,
        context: Optional[DistributedSearchContext] = None,
        rng: Optional[random.Random] = None,
        on_candidate_evaluated: Optional[Callable[[Candidate], None]] = None,
        on_generation_end: Optional[Callable[[int, Sequence[Candidate], Sequence[Candidate], Optional[Candidate]], None]] = None,
        on_state_update: Optional[Callable[[int, Sequence[Candidate], Sequence[Mapping[str, Any]], str], None]] = None,
        rung_budgets: Optional[Sequence[int]] = None,
        rung_promote_fraction: float = 0.5,
        min_rung_survivors: int = 1,
        final_rung_validation_fitness: bool = False,
    ):
        self.base_config = copy.deepcopy(base_config)
        self.target = target
        self.evaluator = evaluator
        self.population_size = int(population_size)
        self.generations = int(generations)
        self.elite_size = max(1, int(elite_size))
        self.offspring_size = int(offspring_size) if offspring_size is not None else max(self.population_size - self.elite_size, 0)
        self.mutation = mutation or RandomMutation()
        self.crossover = crossover or RandomCrossover()
        self.selection = selection or ParetoSelection()
        self.context = context or DistributedSearchContext.from_torch_distributed()
        self.rng = rng or random.Random()
        self.on_candidate_evaluated = on_candidate_evaluated
        self.on_generation_end = on_generation_end
        self.on_state_update = on_state_update
        budgets = [int(value) for value in (rung_budgets or []) if int(value) > 0]
        self.rung_budgets = tuple(budgets)
        self.rung_promote_fraction = min(max(float(rung_promote_fraction), 0.0), 1.0)
        self.min_rung_survivors = max(1, int(min_rung_survivors))
        self.final_rung_validation_fitness = bool(final_rung_validation_fitness)
        self._evaluation_cache: Dict[Tuple[str, int], Candidate] = {}
        self._eval_counts: Dict[Tuple[str, int], int] = {}

    def _make_candidate(self, genome: Mapping[str, Any], generation: int, index: int) -> Candidate:
        normalized = self.target.normalize_genome(genome)
        identifier = f"g{int(generation)}_{int(index)}"
        candidate = Candidate(genome=normalized, generation=int(generation), identifier=identifier)
        repair_metadata = getattr(self.target, "last_repair_metadata", None)
        if repair_metadata:
            candidate.metadata["repair"] = copy.deepcopy(dict(repair_metadata))
        return candidate

    def _evaluate_candidate(self, candidate: Candidate) -> Candidate:
        cfg = copy.deepcopy(self.base_config)
        self.target.apply_to_config(cfg, candidate.genome)
        self._assign_eval_id(candidate)
        candidate.fitness = self.evaluator(cfg=cfg, target=self.target, candidate=candidate, context=self.context)
        self._cache_candidate_evaluation(candidate)
        if self.on_candidate_evaluated is not None:
            self.on_candidate_evaluated(candidate)
        return candidate

    def _evaluate_or_reuse_candidate(self, candidate: Candidate) -> Candidate:
        cached = self._cached_candidate_evaluation(candidate)
        if cached is not None:
            candidate.fitness = copy.deepcopy(cached.fitness)
            candidate.metadata.update(copy.deepcopy(cached.metadata))
            candidate.metadata["reused_evaluation"] = True
            candidate.metadata["reused_from_candidate_id"] = cached.identifier
            return candidate
        return self._evaluate_candidate(candidate)

    def _candidate_for_rung(self, candidate: Candidate, *, rung_index: int, budget_iters: int) -> Candidate:
        rung_candidate = copy.deepcopy(candidate)
        base_id = self._stable_candidate_id(rung_candidate)
        rung_candidate.identifier = f"{base_id}_r{int(rung_index)}"
        rung_candidate.fitness = None
        rung_candidate.metadata["base_candidate_id"] = base_id
        rung_candidate.metadata["rung"] = {
            "index": int(rung_index),
            "budget_iters": int(budget_iters),
            "is_final": bool(int(rung_index) == len(self.rung_budgets) - 1),
            "use_validation_fitness": bool(
                self.final_rung_validation_fitness and int(rung_index) == len(self.rung_budgets) - 1
            ),
        }
        return rung_candidate

    @staticmethod
    def _stable_candidate_id(candidate: Candidate) -> str:
        if candidate.metadata.get("base_candidate_id"):
            return str(candidate.metadata["base_candidate_id"])
        identifier = str(candidate.identifier or "candidate")
        if "_r" in identifier:
            head, tail = identifier.rsplit("_r", 1)
            if tail.isdigit():
                return head
        return identifier

    @staticmethod
    def _rung_index(candidate: Candidate) -> Optional[int]:
        rung = dict(candidate.metadata.get("rung") or {})
        if "index" not in rung:
            return None
        return int(rung["index"])

    def _cache_key(self, candidate: Candidate) -> Optional[Tuple[str, int]]:
        rung_index = self._rung_index(candidate)
        if rung_index is None:
            return None
        return (self._stable_candidate_id(candidate), int(rung_index))

    def _assign_eval_id(self, candidate: Candidate) -> None:
        key = self._cache_key(candidate)
        if key is None:
            return
        eval_index = int(self._eval_counts.get(key, 0))
        self._eval_counts[key] = eval_index + 1
        candidate.metadata["eval_id"] = f"{candidate.identifier}_e{eval_index}"
        candidate.metadata["eval_index"] = eval_index

    def _cache_candidate_evaluation(self, candidate: Candidate) -> None:
        key = self._cache_key(candidate)
        if key is None or candidate.fitness is None or not bool(candidate.fitness.valid):
            return
        self._evaluation_cache[key] = copy.deepcopy(candidate)

    def _cached_candidate_evaluation(self, candidate: Candidate) -> Optional[Candidate]:
        key = self._cache_key(candidate)
        if key is None:
            return None
        cached = self._evaluation_cache.get(key)
        if cached is None or cached.fitness is None or not bool(cached.fitness.valid):
            return None
        return cached

    def _promote_limit(self, population_size: int, rung_index: int) -> int:
        if rung_index >= len(self.rung_budgets) - 1:
            return int(population_size)
        if population_size <= 0:
            return 0
        limit = int(math.ceil(float(population_size) * float(self.rung_promote_fraction)))
        return min(int(population_size), max(self.min_rung_survivors, limit))

    def _evaluate_rungs(
        self,
        population: Sequence[Candidate],
        *,
        generation: int,
        history_records: Sequence[Mapping[str, Any]],
    ) -> List[Candidate]:
        candidates = list(population)
        for rung_index, budget_iters in enumerate(self.rung_budgets):
            rung_candidates = [
                self._candidate_for_rung(candidate, rung_index=rung_index, budget_iters=int(budget_iters))
                for candidate in candidates
            ]
            for candidate in rung_candidates:
                self._evaluate_or_reuse_candidate(candidate)
                if self.on_state_update is not None:
                    self.on_state_update(generation, rung_candidates, history_records, f"rung_{rung_index}_candidate_evaluated")
            promote_limit = self._promote_limit(len(rung_candidates), rung_index)
            candidates = self.selection(rung_candidates, promote_limit)
            if self.on_state_update is not None:
                self.on_state_update(generation, candidates, history_records, f"rung_{rung_index}_complete")
            if not candidates:
                break
        return list(candidates)

    def _initial_genomes(
        self,
        initial_genomes: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> List[Genome]:
        if initial_genomes:
            return [self.target.normalize_genome(genome) for genome in initial_genomes]

        search_space = copy.deepcopy(getattr(self.target, "search_space", {}) or {})
        default_genome = self.target.default_genome()
        if search_space:
            return [
                self.target.normalize_genome(sample_parameter_space(default_genome, search_space, rng=self.rng))
                for _ in range(self.population_size)
            ]
        return [default_genome]

    def _breed(self, population: Sequence[Candidate], generation: int) -> List[Candidate]:
        if not population or self.offspring_size <= 0:
            return []

        parents = list(population)
        offspring: List[Candidate] = []
        for index in range(self.offspring_size):
            left = self.rng.choice(parents)
            right = self.rng.choice(parents) if len(parents) > 1 else left
            crossover_result = self.crossover(left.genome, right.genome, self.rng)
            crossover_metadata: Dict[str, Any] = {}
            if isinstance(crossover_result, tuple):
                child_genome = crossover_result[0]
                crossover_metadata = copy.deepcopy(dict(crossover_result[1] or {}))
            else:
                child_genome = crossover_result
            mutation_result = self.mutation(child_genome, self.rng)
            mutation_metadata: Dict[str, Any] = {}
            if isinstance(mutation_result, tuple):
                child_genome = mutation_result[0]
                mutation_metadata = copy.deepcopy(dict(mutation_result[1] or {}))
            else:
                child_genome = mutation_result
            child = self._make_candidate(child_genome, generation=generation, index=index)
            child.parents = tuple(parent.identifier or "" for parent in (left, right))
            if crossover_metadata:
                child.metadata["crossover"] = crossover_metadata
            if mutation_metadata:
                child.metadata["mutation"] = mutation_metadata
            offspring.append(child)
        return offspring

    def run(
        self,
        *,
        initial_genomes: Optional[Sequence[Mapping[str, Any]]] = None,
        initial_population: Optional[Sequence[Candidate]] = None,
        start_generation: int = 0,
        history: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> EvolutionResult:
        if initial_population is not None:
            population = [copy.deepcopy(candidate) for candidate in initial_population]
        else:
            population = [
                self._make_candidate(genome, generation=int(start_generation), index=index)
                for index, genome in enumerate(self._initial_genomes(initial_genomes))
            ]
        for candidate in population:
            self._cache_candidate_evaluation(candidate)
        history_records: List[Dict[str, Any]] = [copy.deepcopy(dict(item)) for item in (history or [])]

        best: Optional[Candidate] = None
        if population:
            evaluated_with_fitness = [candidate for candidate in population if candidate.fitness is not None]
            if evaluated_with_fitness:
                best = self.selection(evaluated_with_fitness, 1)[0]

        for generation in range(int(start_generation), self.generations + 1):
            if self.rung_budgets:
                evaluated = self._evaluate_rungs(population, generation=generation, history_records=history_records)
            else:
                for candidate in population:
                    if candidate.fitness is None:
                        self._evaluate_candidate(candidate)
                        if self.on_state_update is not None:
                            self.on_state_update(generation, population, history_records, "candidate_evaluated")
                evaluated = list(population)

            selected = self.selection(evaluated, min(self.population_size, len(evaluated)))
            elites = selected[: min(self.elite_size, len(selected))]
            if selected:
                best = selected[0]
            if self.on_generation_end is not None:
                self.on_generation_end(generation, evaluated, selected, best)
            history_records.append(
                {
                    "generation": int(generation),
                    "population_size": int(len(evaluated)),
                    "survivor_size": int(len(selected)),
                    "best_metrics": copy.deepcopy(best.fitness.metrics if best and best.fitness else {}),
                }
            )
            if self.on_state_update is not None:
                self.on_state_update(generation, selected, history_records, "generation_end")

            if generation >= self.generations:
                population = selected
                if self.on_state_update is not None:
                    self.on_state_update(generation, population, history_records, "complete")
                break

            offspring = self._breed(selected, generation=generation + 1)
            population = (list(elites) + offspring)[: self.population_size]
            if self.on_state_update is not None:
                self.on_state_update(generation + 1, population, history_records, "next_generation_ready")

        return EvolutionResult(best=best, population=population, history=history_records)
