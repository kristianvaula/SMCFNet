# src/smcfnet/tools/sttp_viz/registry.py
"""Registry for STTP visualization diagnostics. Mirrors model/ofe/__init__.py."""
from __future__ import annotations
from typing import Dict, List, Type

_DIAGNOSTIC_REGISTRY: Dict[str, type] = {}


def register_diagnostic(name: str):
    def deco(cls: type) -> type:
        if name in _DIAGNOSTIC_REGISTRY:
            raise ValueError(f"Diagnostic '{name}' already registered")
        cls.name = name
        _DIAGNOSTIC_REGISTRY[name] = cls
        return cls
    return deco


def get_diagnostic(name: str) -> type:
    if name not in _DIAGNOSTIC_REGISTRY:
        raise ValueError(
            f"Unknown diagnostic '{name}'. Available: {available_diagnostics()}"
        )
    return _DIAGNOSTIC_REGISTRY[name]


def available_diagnostics() -> List[str]:
    return sorted(_DIAGNOSTIC_REGISTRY)
