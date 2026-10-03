# src/smcfnet/tools/sttp_viz/base.py
"""Base class for diagnostics. Subclasses override the hooks they need."""
from __future__ import annotations
from pathlib import Path


class Diagnostic:
    name: str = "base"
    LEVEL: str = "frame"  # "frame" | "run"

    def __init__(self, out_dir, **kwargs):
        self.out_dir = Path(out_dir)

    def on_frame(self, bundle) -> None:  # frame-level
        pass

    def on_run(self, summary) -> None:   # run-level
        pass

    def finalize(self) -> None:
        pass
