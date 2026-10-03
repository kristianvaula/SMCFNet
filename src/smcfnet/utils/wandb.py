from __future__ import annotations

from dataclasses import asdict
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from smcfnet.config.config import Config


def _try_import_wandb():
  try:
    import wandb  # type: ignore

    return wandb
  except Exception:
    return None


class WandbMetricBuffer:
  """Buffers scalar metrics and flushes aggregated values to reduce wandb.log calls."""

  def __init__(self) -> None:
    self._buf: Dict[str, list[float]] = {}

  def add(self, metrics: Dict[str, Any]) -> None:
    for k, v in metrics.items():
      if v is None:
        continue
      if isinstance(v, (int, float, np.floating, np.integer)) and np.isfinite(v):
        self._buf.setdefault(k, []).append(float(v))

  def flush(self) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, values in self._buf.items():
      if not values:
        continue
      out[k] = float(np.mean(values))
    self._buf.clear()
    return out


def init_wandb(config: Config, output_dir: Path):
  """Initialize a wandb run. Returns wandb.Run or None."""

  wandb = _try_import_wandb()
  if wandb is None:
    return None

  if getattr(config, "wandb", None) is None:
    return None

  if not config.wandb.use_wandb:
    return None

  env_disabled = str(os.environ.get("WANDB_DISABLED", "")).strip().lower()
  if env_disabled in {"1", "true", "yes", "on"}:
    return None

  env_mode = str(os.environ.get("WANDB_MODE", "")).strip().lower()
  mode = env_mode or (config.wandb.mode or "online").lower()
  if mode == "disabled":
    return None

  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  run_name = config.wandb.run_name or config.log.experiment_name

  # Avoid uploading massive objects; keep config as plain nested dict.
  try:
    run = wandb.init(
      project=config.wandb.project_name,
      entity=config.wandb.entity,
      name=run_name,
      group=config.wandb.group,
      tags=config.wandb.tags,
      dir=str(output_dir),
      mode=mode,
      config=asdict(config),
    )
  except Exception as e:
    logger = logging.getLogger(__name__)
    logger.warning("wandb.init() failed (%s), falling back to offline mode", e)
    try:
      run = wandb.init(
        project=config.wandb.project_name,
        entity=config.wandb.entity,
        name=run_name,
        group=config.wandb.group,
        tags=config.wandb.tags,
        dir=str(output_dir),
        mode="offline",
        config=asdict(config),
      )
    except Exception:
      logger.warning("wandb.init() offline also failed, disabling wandb")
      return None
  return run
