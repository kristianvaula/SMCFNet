from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict

import yaml


def abs_root(root: str | Path) -> Path:
    root_p = Path(root)
    if not root_p.is_absolute():
        root_p = (Path.cwd() / root_p).resolve()
    return root_p


def load_peod_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PEOD dataset config not found: {path}")
    if path.is_dir():
        raise IsADirectoryError(f"PEOD dataset config path is a directory: {path}")
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid PEOD dataset config, expected dict: {path}")
    dataset = cfg.get("dataset", {})
    if str(dataset.get("name", "")).upper() != "PEOD":
        raise ValueError(f"Expected dataset.name=PEOD in {path}")
    return cfg


def read_timestamps_us(path: str | Path) -> list[int]:
    """Read PEOD frame timestamps.

    Extracted PEOD timestamp files are CSVs with rows like ``1,1590802``.
    The second column is the RGB frame timestamp in microseconds. This helper
    also accepts one-number-per-line files for robustness.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PEOD timestamp file not found: {path}")

    values: list[float] = []
    saw_multi_column = False
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if not row:
                continue
            if len(row) > 1:
                saw_multi_column = True
            values.append(float(row[-1]))

    if len(values) == 0:
        raise ValueError(f"No timestamps found in {path}")

    if saw_multi_column:
        return [int(round(v)) for v in values]

    max_value = max(abs(v) for v in values)
    if max_value < 1.0e5:
        scale = 1_000_000.0
    elif max_value < 1.0e8:
        scale = 1_000.0
    elif max_value > 1.0e12:
        scale = 0.001
    else:
        scale = 1.0

    return [int(round(v * scale)) for v in values]
