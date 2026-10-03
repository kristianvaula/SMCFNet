"""Shared run-provenance helpers (campaign #129, plan: recursive-jingling-reddy).

Single source of truth for stamping experiment runs so thesis numbers are
reproducible: full git SHA + dirty flag, a canonical hash of the RESOLVED config
(the full ``_base_`` chain, not just the leaf file) and of the EFFECTIVE config
(post CLI/SLURM override), plus the effective config written out 1:1 for rerun.

Design notes:
- ``git_head_sha`` returns the FULL sha (short shas collide).
- ``git_dirty`` is tracked-only (untracked scratch must not flag a clean tree).
- Floats are rounded in canonicalization so 1.0 vs 1.0000000001 collapse to one
  hash (cosmetic reserialization must not change the key).
- Volatile keys (output_dir / wandb / timestamp / device) are dropped from the
  hashed payload; the timestamp lives OUTSIDE every hash.
- ``provenance.json`` is written atomically (tmp + os.replace).
"""
from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

_VOLATILE_KEYS = {"output_dir", "wandb", "timestamp", "device"}
_FLOAT_NDIGITS = 9


def sha256_of_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_of_str(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git_head_sha() -> str:
    """Full HEAD sha, or '' if not a git repo."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return ""


def git_dirty() -> bool:
    """True if tracked files differ from HEAD (untracked files ignored)."""
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            stderr=subprocess.DEVNULL)
        return bool(out.decode().strip())
    except Exception:
        return False


def _to_primitive(obj: Any) -> Any:
    """Recursively cast to JSON/YAML-safe primitives, dropping volatile keys."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        obj = dataclasses.asdict(obj)
    if isinstance(obj, dict):
        return {
            str(k): _to_primitive(v)
            for k, v in obj.items()
            if k not in _VOLATILE_KEYS
        }
    if isinstance(obj, (list, tuple)):
        return [_to_primitive(v) for v in obj]
    if isinstance(obj, enum.Enum):
        return _to_primitive(obj.value)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, bool) or obj is None or isinstance(obj, (int, str)):
        return obj
    if isinstance(obj, float):
        return round(obj, _FLOAT_NDIGITS)
    # torch.device, numpy scalars, etc. -> str (stable, never crash the dump).
    return str(obj)


def canonical_dump(cfg_or_dict: Any) -> str:
    """Deterministic YAML of the resolved/effective config, volatile keys dropped."""
    import yaml
    prim = _to_primitive(cfg_or_dict)
    return yaml.safe_dump(prim, sort_keys=True, default_flow_style=False)


def resolved_config_sha256(config_path: str | Path) -> str:
    """Hash of the canonical dump of the full _base_-resolved chain (no overrides)."""
    from smcfnet.config.config import load_config_from_yaml
    merged = load_config_from_yaml(str(config_path))
    return sha256_of_str(canonical_dump(merged))


def write_effective_config(output_dir: str | Path, cfg: Any) -> Path:
    """Write the post-override effective config; this file IS a 1:1 rerun artifact."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "effective_config.yaml"
    path.write_text(canonical_dump(cfg))
    return path


def _slurm_job_id() -> str:
    arr = os.environ.get("SLURM_ARRAY_JOB_ID")
    task = os.environ.get("SLURM_ARRAY_TASK_ID")
    if arr and task:
        return f"{arr}_{task}"
    return os.environ.get("SLURM_JOB_ID", "local")


def _pip_freeze_sha256() -> str:
    try:
        out = subprocess.check_output(
            ["python", "-m", "pip", "freeze"], stderr=subprocess.DEVNULL)
        return sha256_of_str(out.decode())
    except Exception:
        return ""


def write_provenance(
    output_dir: str | Path,
    cfg: Any,
    config_path: str | Path,
    *,
    resume_path: str | None = None,
    wandb_run_id: str | None = None,
    dataset_id: str | None = None,
    timestamp: str | None = None,
) -> Path:
    """Write provenance.json atomically. Returns its path.

    The effective config is written alongside and its hash recorded. The
    timestamp is recorded OUTSIDE every hash payload.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    eff_path = write_effective_config(out, cfg)

    prov: dict[str, Any] = {
        "git_sha": git_head_sha(),
        "git_dirty": git_dirty(),
        "config_path": str(Path(config_path).resolve()),
        "effective_config_sha256": sha256_of_file(eff_path),
        "slurm_job_id": _slurm_job_id(),
        "pip_freeze_sha256": _pip_freeze_sha256(),
        "dataset_id": dataset_id,
        "wandb_run_id": wandb_run_id,
        "timestamp": timestamp,  # outside every hash
    }
    try:
        prov["resolved_config_sha256"] = resolved_config_sha256(config_path)
    except Exception as e:
        prov["resolved_config_sha256"] = ""
        prov["resolved_config_error"] = str(e)
    if resume_path:
        prov["resume_path"] = str(resume_path)
        prov["resume_sha256"] = (
            sha256_of_file(resume_path) if os.path.isfile(resume_path) else "")

    path = out / "provenance.json"
    tmp = out / "provenance.json.tmp"
    tmp.write_text(json.dumps(prov, indent=2, sort_keys=True))
    os.replace(tmp, path)
    return path


def provenance_dict(
    cfg: Any,
    config_path: str | Path,
    *,
    resume_path: str | None = None,
    wandb_run_id: str | None = None,
    dataset_id: str | None = None,
) -> dict:
    """In-memory provenance (for embedding in a checkpoint), no file writes."""
    d: dict[str, Any] = {
        "git_sha": git_head_sha(),
        "git_dirty": git_dirty(),
        "config_path": str(Path(config_path).resolve()),
        "effective_config_sha256": sha256_of_str(canonical_dump(cfg)),
        "slurm_job_id": _slurm_job_id(),
    }
    try:
        d["resolved_config_sha256"] = resolved_config_sha256(config_path)
    except Exception:
        d["resolved_config_sha256"] = ""
    if resume_path:
        d["resume_path"] = str(resume_path)
        d["resume_sha256"] = (
            sha256_of_file(resume_path) if os.path.isfile(resume_path) else "")
    return d
