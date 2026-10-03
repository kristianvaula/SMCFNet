"""Utilities for expanding discrete architecture search spaces."""

from __future__ import annotations

import copy
import itertools
import random
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


def _deep_merge(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _expand_options(value: Any) -> List[Any]:
    if isinstance(value, Mapping):
        return expand_parameter_grid({}, value)
    if isinstance(value, (list, tuple)) and not isinstance(value, (str, bytes)):
        return [copy.deepcopy(item) for item in value]
    return [copy.deepcopy(value)]


def expand_parameter_grid(
    base: Optional[Mapping[str, Any]] = None,
    search_space: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Expand a nested discrete search space into concrete specs.

    A search-space value can be:
    - a scalar, which is treated as a fixed value
    - a list/tuple, which is treated as a discrete option set
    - a nested dict, which is expanded recursively
    """

    base_spec = copy.deepcopy(dict(base or {}))
    search_space = dict(search_space or {})
    if not search_space:
        return [base_spec]

    items: List[tuple[str, List[Any]]] = []
    for key, value in search_space.items():
        items.append((str(key), _expand_options(value)))

    concrete_specs: List[Dict[str, Any]] = []
    for combo in itertools.product(*(values for _, values in items)):
        override = {key: copy.deepcopy(option) for (key, _), option in zip(items, combo)}
        concrete_specs.append(_deep_merge(base_spec, override))
    return concrete_specs


def sample_parameter_space(
    base: Optional[Mapping[str, Any]] = None,
    search_space: Optional[Mapping[str, Any]] = None,
    *,
    rng: Optional[random.Random] = None,
) -> Dict[str, Any]:
    """Sample one concrete spec from a nested discrete search space."""

    rng = rng or random.Random()
    sampled = copy.deepcopy(dict(base or {}))
    for key, value in dict(search_space or {}).items():
        key = str(key)
        if isinstance(value, Mapping):
            current = sampled.get(key, {})
            if not isinstance(current, Mapping):
                current = {}
            sampled[key] = sample_parameter_space(current, value, rng=rng)
        elif isinstance(value, (list, tuple)) and not isinstance(value, (str, bytes)):
            if not value:
                continue
            sampled[key] = copy.deepcopy(rng.choice(list(value)))
        else:
            sampled[key] = copy.deepcopy(value)
    return sampled
