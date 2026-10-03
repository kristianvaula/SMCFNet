"""Trainer-backed evaluators for evolutionary search."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from smcfnet.config.config import Config, save_config
from smcfnet.evolution.framework import Candidate, DistributedSearchContext, Fitness, SearchTarget
from smcfnet.evolution.metrics import (
    FLOW_CENTERED_RATIO_MIN,
    FLOW_MAG_MIN,
    FLOW_MEAN_RATIO_MAX,
    FLOW_SPATIAL_STD_MIN,
    build_convergence_fitness,
    parse_mcm_log_metrics,
    parse_mcm_validation_log_metrics,
    summarize_metric_rows,
)


@dataclass
class TrainerBackedEvaluator:
    """Evaluate candidates by running the existing training entrypoint.

    Each candidate is evaluated in its own subprocess and output directory.
    This is intentionally generic: target-specific behavior should be applied
    to ``cfg`` before the evaluator is called.
    """

    root_dir: Path
    max_steps_per_candidate: int = 100
    timeout_sec: Optional[float] = None
    python_executable: str = sys.executable
    module: str = "smcfnet.core.train"
    device: str = "cuda"
    num_gpus: int = 1
    memory_weight: float = 0.05
    wall_ms_soft_limit: float = 1000.0
    # Validation-prefixed quality key (None preserves the historical "mcm_epe"
    # default). Set to "total" for datasets without GT flow (e.g. PEOD).
    val_quality_key: Optional[str] = None
    # Flow-structure (DC/global-collapse) guardrail thresholds; defaults match
    # the historical hardcoded constants in evolution/metrics.py.
    flow_mag_min: float = FLOW_MAG_MIN
    flow_centered_ratio_min: float = FLOW_CENTERED_RATIO_MIN
    flow_mean_ratio_max: float = FLOW_MEAN_RATIO_MAX
    flow_spatial_std_min: float = FLOW_SPATIAL_STD_MIN
    constraint_violation_weight: float = 0.05
    # Cap applied to cfg.eval.max_batches when use_validation_fitness is set
    # for a rung, so "validation fitness" stays a short held-out check rather
    # than a full pass over the val loader (0 = no cap, validate() runs to
    # exhaustion; can be far slower than the training rung budget itself on
    # high-resolution datasets like PEOD).
    val_max_batches: int = 50
    max_retries: int = 0
    retry_backoff_sec: float = 5.0
    retry_failure_reasons: Sequence[str] = ("dataloader_transient", "filesystem_transient")
    command_prefix: Optional[Sequence[str]] = None
    env: Optional[Mapping[str, str]] = None
    cwd: Optional[Path] = None

    def __post_init__(self) -> None:
        self.root_dir = Path(self.root_dir)
        self.cwd = Path(self.cwd) if self.cwd is not None else Path.cwd()

    def __call__(
        self,
        *,
        cfg: Config,
        target: SearchTarget,
        candidate: Candidate,
        context: DistributedSearchContext,
    ) -> Fitness:
        if not context.is_master():
            return Fitness(valid=False, notes={"reason": "non-master rank skipped candidate evaluation"})

        candidate_dir = self._candidate_dir(candidate)
        candidate_dir.mkdir(parents=True, exist_ok=True)
        candidate.metadata["candidate_dir"] = str(candidate_dir)

        budget_iters = self._budget_iters_for_candidate(candidate)
        use_validation_fitness = self._use_validation_fitness(candidate)

        self._apply_budget_and_paths(
            cfg,
            candidate,
            candidate_dir,
            budget_iters=budget_iters,
            use_validation_fitness=use_validation_fitness,
        )
        config_path = candidate_dir / "candidate_config.yaml"
        save_config(cfg, str(config_path))

        started = time.time()
        command = self._build_command(config_path)
        metrics_path = candidate_dir / "metrics.json"

        completed: Optional[subprocess.CompletedProcess[str]] = None
        run_log = candidate_dir / "evaluator_subprocess.log"
        attempts: list[Dict[str, Any]] = []
        failure_reason: Optional[str] = None
        for attempt in range(max(0, int(self.max_retries)) + 1):
            run_log = self._attempt_log_path(candidate_dir, attempt)
            attempt_started = time.time()
            try:
                completed = self._run_command(command, run_log)
                failure_reason = None if completed.returncode == 0 else self._classify_failure(run_log)
            except subprocess.TimeoutExpired:
                completed = None
                failure_reason = "timeout"
            except RuntimeError as exc:
                completed = None
                failure_reason = self._classify_exception(exc)

            attempt_record = {
                "attempt": int(attempt),
                "reason": failure_reason,
                "returncode": int(completed.returncode) if completed is not None else None,
                "run_log": str(run_log),
                "elapsed_sec": float(time.time() - attempt_started),
            }
            attempts.append(attempt_record)
            if failure_reason is None:
                break
            if not self._should_retry(failure_reason, attempt):
                break
            time.sleep(self._retry_sleep_seconds(attempt))

        if completed is None:
            fitness = self._invalid_fitness(
                reason=str(failure_reason or "trainer_failed"),
                candidate_dir=candidate_dir,
                metrics_path=metrics_path,
                extra={
                    "timeout_sec": float(self.timeout_sec or 0.0),
                    "cmd": command,
                    "elapsed_sec": time.time() - started,
                    "attempts": attempts,
                    "retry_count": max(0, len(attempts) - 1),
                },
            )
            return fitness

        train_log = candidate_dir / "train.log"
        rows = parse_mcm_log_metrics(train_log, max_iter=budget_iters)
        val_rows = parse_mcm_validation_log_metrics(train_log, max_iter=budget_iters)
        elapsed_sec = float(time.time() - started)
        metrics = summarize_metric_rows(rows)
        if val_rows:
            metrics.update(summarize_metric_rows(val_rows, prefix="val"))
        iter_values = [int(row["iter"]) for row in rows if "iter" in row]
        iter_span = (max(iter_values) - min(iter_values) + 1) if iter_values else int(budget_iters)
        metrics.update(
            {
                "candidate/returncode": float(completed.returncode),
                "candidate/elapsed_sec": elapsed_sec,
                "candidate/retry_count": float(max(0, len(attempts) - 1)),
                "candidate/num_log_rows": float(len(rows)),
                "candidate/num_val_log_rows": float(len(val_rows)),
                "candidate/budget_iters": float(budget_iters),
                "candidate/wall_time_per_iter_sec": elapsed_sec / float(max(1, int(budget_iters))),
                "candidate/wall_time_per_logged_iter_sec": elapsed_sec / float(max(1, int(iter_span))),
                "candidate/wall_time_per_log_row_sec": elapsed_sec / float(max(1, len(rows))),
            }
        )

        if completed.returncode != 0:
            reason = str(failure_reason or self._classify_failure(run_log))
            fitness = self._invalid_fitness(
                reason=reason,
                candidate_dir=candidate_dir,
                metrics_path=metrics_path,
                metrics=metrics,
                extra={
                    "cmd": command,
                    "returncode": int(completed.returncode),
                    "attempts": attempts,
                    "retry_count": max(0, len(attempts) - 1),
                },
            )
            self._clear_cuda_cache()
            return fitness

        if not rows:
            fitness = self._invalid_fitness(
                reason="no_mcm_metrics",
                candidate_dir=candidate_dir,
                metrics_path=metrics_path,
                metrics=metrics,
                extra={
                    "cmd": command,
                    "returncode": int(completed.returncode),
                    "attempts": attempts,
                    "retry_count": max(0, len(attempts) - 1),
                },
            )
            return fitness

        threshold_kwargs = dict(
            flow_mag_min=self.flow_mag_min,
            flow_centered_ratio_min=self.flow_centered_ratio_min,
            flow_mean_ratio_max=self.flow_mean_ratio_max,
            flow_spatial_std_min=self.flow_spatial_std_min,
            constraint_violation_weight=self.constraint_violation_weight,
        )
        fitness = build_convergence_fitness(
            metrics,
            memory_weight=self.memory_weight,
            rows=rows,
            budget_iters=int(budget_iters),
            wall_ms_soft_limit=self.wall_ms_soft_limit,
            **threshold_kwargs,
        )
        if use_validation_fitness and val_rows:
            val_fitness = build_convergence_fitness(
                metrics,
                memory_weight=self.memory_weight,
                rows=val_rows,
                prefix="val",
                budget_iters=int(budget_iters),
                wall_ms_soft_limit=self.wall_ms_soft_limit,
                val_quality_key=self.val_quality_key,
                **threshold_kwargs,
            )
            objectives = list(fitness.objectives)
            if objectives and val_fitness.objectives:
                objectives[0] = float(val_fitness.objectives[0])
                fitness.objectives = tuple(objectives)
                fitness.valid = bool(all(math.isfinite(float(value)) for value in fitness.objectives))
                fitness.metrics["fitness/objective_quality"] = float(val_fitness.objectives[0])
                fitness.metrics["fitness/validation_quality"] = float(val_fitness.objectives[0])
                fitness.metrics["fitness/validation_mcm_epe_late"] = float(
                    val_fitness.metrics.get("fitness/epe_late", val_fitness.objectives[0])
                )
                fitness.notes["quality_source"] = "validation"
                fitness.notes["validation_rows"] = int(len(val_rows))
        elif use_validation_fitness:
            fitness.notes["quality_source"] = "train_no_validation_rows"
            fitness.notes["validation_rows"] = int(len(val_rows))
        fitness.notes.update(
            {
                "candidate_dir": str(candidate_dir),
                "config_path": str(config_path),
                "run_log": str(run_log),
                "train_log": str(train_log),
                "attempts": attempts,
                "retry_count": max(0, len(attempts) - 1),
            }
        )
        self._write_metrics(metrics_path, fitness)
        return fitness

    def _candidate_dir(self, candidate: Candidate) -> Path:
        generation_dir = self.root_dir / f"generation_{int(candidate.generation):03d}"
        return generation_dir / str(candidate.identifier or f"candidate_{int(candidate.generation):03d}")

    def _budget_iters_for_candidate(self, candidate: Candidate) -> int:
        rung = dict(candidate.metadata.get("rung") or {})
        value = rung.get("budget_iters", self.max_steps_per_candidate)
        return max(1, int(value))

    def _use_validation_fitness(self, candidate: Candidate) -> bool:
        rung = dict(candidate.metadata.get("rung") or {})
        return bool(rung.get("use_validation_fitness", False))

    def _apply_budget_and_paths(
        self,
        cfg: Config,
        candidate: Candidate,
        candidate_dir: Path,
        *,
        budget_iters: int,
        use_validation_fitness: bool,
    ) -> None:
        cfg.training.mode = "mcm_pretrain"
        cfg.training.max_steps_per_epoch = int(budget_iters)
        cfg.training.max_total_steps = int(budget_iters)
        cfg.training.num_epochs = 1
        if use_validation_fitness:
            cfg.eval.interval = 1
            cfg.eval.split = "val"
            cfg.eval.mon_eval.enabled = False
            if self.val_max_batches > 0:
                cfg.eval.max_batches = int(self.val_max_batches)
        cfg.log.output_dir = str(candidate_dir.parent)
        cfg.log.experiment_name = str(candidate_dir.name)
        cfg.wandb.mode = "disabled"
        cfg.wandb.use_wandb = False
        cfg.num_gpus = int(self.num_gpus)
        if getattr(cfg, "seed", None) is None:
            cfg.seed = 42

    def _build_command(self, config_path: Path) -> list[str]:
        if self.command_prefix is not None:
            return [*map(str, self.command_prefix), "--config", str(config_path)]
        return [
            str(self.python_executable),
            "-m",
            str(self.module),
            "--config",
            str(config_path),
            "--num-gpus",
            str(int(self.num_gpus)),
        ]

    def _attempt_log_path(self, candidate_dir: Path, attempt: int) -> Path:
        if int(attempt) <= 0:
            return candidate_dir / "evaluator_subprocess.log"
        return candidate_dir / f"evaluator_subprocess_retry_{int(attempt):02d}.log"

    def _run_command(self, command: Sequence[str], run_log: Path) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        if self.env:
            env.update({str(key): str(value) for key, value in self.env.items()})
        if self.device.lower() == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""
        env.setdefault("WANDB_MODE", "disabled")
        src_path = str((self.cwd / "src").resolve())
        env["PYTHONPATH"] = src_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

        with run_log.open("w") as stream:
            return subprocess.run(
                list(command),
                cwd=str(self.cwd),
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=self.timeout_sec,
                check=False,
            )

    def _classify_failure(self, run_log: Path) -> str:
        text = run_log.read_text(errors="replace") if run_log.exists() else ""
        lowered = text.lower()
        if "out of memory" in lowered or "cuda oom" in lowered or "cublas" in lowered:
            return "oom"
        # Word-boundary match so "inf"/"nan" don't match substrings of unrelated
        # words like "[info]" or "inference" (issue #109). "non-finite" stays an
        # explicit phrase check.
        if "non-finite" in lowered or re.search(r"\b(inf|nan)\b", lowered):
            return "non_finite"
        if self._looks_like_dataloader_transient(lowered):
            return "dataloader_transient"
        if self._looks_like_filesystem_transient(lowered):
            return "filesystem_transient"
        return "trainer_failed"

    def _classify_exception(self, exc: BaseException) -> str:
        lowered = str(exc).lower()
        if self._looks_like_filesystem_transient(lowered):
            return "filesystem_transient"
        if self._looks_like_dataloader_transient(lowered):
            return "dataloader_transient"
        return str(exc)

    def _should_retry(self, reason: str, attempt: int) -> bool:
        if int(attempt) >= max(0, int(self.max_retries)):
            return False
        return str(reason) in {str(item) for item in self.retry_failure_reasons}

    def _retry_sleep_seconds(self, attempt: int) -> float:
        if float(self.retry_backoff_sec) <= 0.0:
            return 0.0
        return float(self.retry_backoff_sec) * (2.0 ** max(0, int(attempt)))

    @staticmethod
    def _looks_like_dataloader_transient(text: str) -> bool:
        patterns = (
            "dataloader worker",
            "data loader worker",
            "worker exited unexpectedly",
            "killed by signal",
            "bus error",
            "resource_sharer",
            "multiprocessing",
            "broken pipe",
            "connection reset",
            "operation not permitted",
        )
        return any(pattern in text for pattern in patterns)

    @staticmethod
    def _looks_like_filesystem_transient(text: str) -> bool:
        patterns = (
            "input/output error",
            "i/o error",
            "stale file handle",
            "resource temporarily unavailable",
            "temporarily unavailable",
            "too many open files",
            "unable to open file",
            "unable to synchronously open file",
            "file locking",
            "errno 5",
            "errno 11",
            "errno 116",
            "hdf5",
        )
        return any(pattern in text for pattern in patterns)

    def _invalid_fitness(
        self,
        *,
        reason: str,
        candidate_dir: Path,
        metrics_path: Path,
        metrics: Optional[Mapping[str, float]] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> Fitness:
        fitness = Fitness(
            metrics=dict(metrics or {}),
            objectives=(float("inf"), float("inf"), float("inf"), float("inf")),
            maximize=(False, False, False, False),
            valid=False,
            notes={"reason": str(reason), "candidate_dir": str(candidate_dir), **dict(extra or {})},
        )
        self._write_metrics(metrics_path, fitness)
        return fitness

    @staticmethod
    def _write_metrics(path: Path, fitness: Fitness) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "metrics": dict(fitness.metrics),
                    "objectives": list(fitness.objectives),
                    "maximize": list(fitness.maximize),
                    "valid": bool(fitness.valid),
                    "notes": dict(fitness.notes),
                },
                indent=2,
                sort_keys=True,
            )
        )

    @staticmethod
    def _clear_cuda_cache() -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            return
