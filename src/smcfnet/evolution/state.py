"""JSON persistence helpers for evolutionary search state."""

from __future__ import annotations

import base64
import json
import pickle
import random
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from smcfnet.evolution.framework import Candidate, Fitness


STATE_VERSION = 1


def encode_rng_state(rng: random.Random) -> str:
    """Serialize a ``random.Random`` state to a JSON-safe string."""

    return base64.b64encode(pickle.dumps(rng.getstate())).decode("ascii")


def decode_rng_state(payload: str) -> object:
    """Deserialize a state produced by ``encode_rng_state``."""

    return pickle.loads(base64.b64decode(payload.encode("ascii")))


def fitness_to_dict(fitness: Optional[Fitness]) -> Optional[Dict[str, Any]]:
    if fitness is None:
        return None
    return {
        "metrics": dict(fitness.metrics),
        "objectives": list(fitness.objectives),
        "maximize": list(fitness.maximize),
        "valid": bool(fitness.valid),
        "notes": dict(fitness.notes),
    }


def fitness_from_dict(data: Optional[Mapping[str, Any]]) -> Optional[Fitness]:
    if data is None:
        return None
    return Fitness(
        metrics={str(key): float(value) for key, value in dict(data.get("metrics") or {}).items()},
        objectives=tuple(float(value) for value in data.get("objectives", ())),
        maximize=tuple(bool(value) for value in data.get("maximize", ())),
        valid=bool(data.get("valid", True)),
        notes=dict(data.get("notes") or {}),
    )


def candidate_to_dict(candidate: Candidate) -> Dict[str, Any]:
    return {
        "identifier": candidate.identifier,
        "generation": int(candidate.generation),
        "parents": list(candidate.parents),
        "genome": candidate.genome,
        "fitness": fitness_to_dict(candidate.fitness),
        "metadata": dict(candidate.metadata),
    }


def candidate_from_dict(data: Mapping[str, Any]) -> Candidate:
    return Candidate(
        genome=dict(data.get("genome") or {}),
        fitness=fitness_from_dict(data.get("fitness")),
        generation=int(data.get("generation", 0)),
        parents=tuple(str(parent) for parent in data.get("parents", ())),
        identifier=str(data["identifier"]) if data.get("identifier") is not None else None,
        metadata=dict(data.get("metadata") or {}),
    )


def save_search_state(
    path: str | Path,
    *,
    target: str,
    generation: int,
    population: Sequence[Candidate],
    rng: Optional[random.Random] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> None:
    """Persist search state as local JSON."""

    state: Dict[str, Any] = {
        "version": STATE_VERSION,
        "target": str(target),
        "generation": int(generation),
        "population": [candidate_to_dict(candidate) for candidate in population],
        "extra": dict(extra or {}),
    }
    if rng is not None:
        state["rng_state"] = encode_rng_state(rng)

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(state, indent=2, sort_keys=True))
    os.replace(tmp_path, out_path)


def load_search_state(path: str | Path) -> Dict[str, Any]:
    """Load a JSON search state file."""

    state = json.loads(Path(path).read_text())
    if int(state.get("version", -1)) != STATE_VERSION:
        raise ValueError(f"Unsupported evolution state version: {state.get('version')!r}")
    return state
