from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Iterable

import torch

from smcfnet.config.config import Config
from smcfnet.core.launch import is_main_process
from smcfnet.core.base.hookbase import HookBase
from smcfnet.utils.wandb import WandbMetricBuffer, _try_import_wandb, init_wandb


logger = logging.getLogger(__name__)


class FiniteModelStateHook(HookBase):
	"""Abort before eval/checkpoint if model parameters or buffers are corrupt."""

	def after_epoch(self) -> None:
		cfg = getattr(self.trainer, "config", None)
		training_cfg = getattr(cfg, "training", None) if cfg is not None else None
		if not bool(getattr(training_cfg, "abort_on_nonfinite_model_state", True)):
			return

		model = getattr(self.trainer, "model", None)
		if model is None:
			return

		bad = []
		total_bad = 0
		state = model.state_dict()
		for name, value in state.items():
			if not (torch.is_tensor(value) and value.is_floating_point()):
				continue
			finite = torch.isfinite(value)
			if bool(finite.all().item()):
				continue
			n_bad = int((~finite).sum().item())
			total_bad += n_bad
			if len(bad) < 8:
				bad.append(f"{name}: {n_bad}/{value.numel()} {tuple(value.shape)}")

		if total_bad > 0:
			raise RuntimeError(
				"Non-finite model state detected before eval/checkpoint at "
				f"epoch {int(self.trainer.epoch) + 1}, iteration {int(self.trainer.iteration)}: "
				f"{total_bad} NaN/Inf values. First tensors: {bad}"
			)


class PeriodicCheckpointer(HookBase):
	"""Hook for periodic checkpoint saving."""

	def __init__(
		self,
		save_interval: int,
		output_dir: Path,
		max_to_keep: int = 5,
		save_epochs: Iterable[int] | None = None,
	):
		self.save_interval = int(save_interval)
		self.output_dir = Path(output_dir)
		self.max_to_keep = int(max_to_keep)
		self.save_epochs = {
			int(epoch) for epoch in (save_epochs or []) if int(epoch) > 0
		}
		self.checkpoints = []

	def after_epoch(self):
		if not is_main_process():
			return

		epoch = int(self.trainer.epoch + 1)

		# Always save latest checkpoint (single file, overwritten each epoch)
		self.trainer.save_checkpoint("latest.pth")

		# Periodic and/or explicitly requested numbered checkpoints.
		should_save = False
		if self.save_interval > 0 and epoch % self.save_interval == 0:
			should_save = True
		if epoch in self.save_epochs:
			should_save = True
		if should_save:
			filename = f"epoch_{epoch}.pth"
			self.trainer.save_checkpoint(filename)

			# Keep only recent checkpoints
			self.checkpoints.append(self.output_dir / filename)
			if len(self.checkpoints) > self.max_to_keep:
				old_ckpt = self.checkpoints.pop(0)
				if old_ckpt.exists() and old_ckpt.name != "best.pth":
					old_ckpt.unlink()


class LRSchedulerHook(HookBase):
	"""Hook for learning rate scheduling."""

	def after_step(self):
		sched = getattr(self.trainer, "scheduler", None)
		if sched is None:
			return
		if bool(getattr(sched, "_step_per_batch", False)):
			sched.step()

	def after_epoch(self):
		sched = getattr(self.trainer, "scheduler", None)
		if sched is None:
			return
		if bool(getattr(sched, "_step_per_batch", False)):
			return
		# Support both styles:
		# 1) Warmup implemented *inside* the scheduler (e.g. SequentialLR): must step every epoch.
		# 2) Legacy warmup that manually sets LR for the first warmup_epochs: only step after warmup.
		cfg = getattr(self.trainer, "config", None)
		training_cfg = getattr(cfg, "training", None) if cfg is not None else None
		use_warmup_scheduler = bool(getattr(training_cfg, "warmup_use_scheduler", False))
		warmup_epochs = int(getattr(training_cfg, "warmup_epochs", 0)) if training_cfg is not None else 0

		if use_warmup_scheduler or warmup_epochs <= 0 or int(self.trainer.epoch) >= warmup_epochs:
			sched.step()


class EvaluationHook(HookBase):
	"""Hook for periodic evaluation."""

	def __init__(self, eval_interval: int):
		self.eval_interval = int(eval_interval)

	def after_epoch(self):
		# Run full eval periodically, but always run at the final epoch so that
		# short runs still produce a best model. If mon-eval is configured,
		# it runs on its own cadence between full eval passes.
		is_final_epoch = (self.trainer.epoch + 1) >= self.trainer.config.training.num_epochs
		epoch_num = int(self.trainer.epoch) + 1
		eval_interval = int(getattr(self.trainer, "_get_eval_interval", lambda: self.eval_interval)() or 0)
		mon_eval_interval = int(getattr(self.trainer, "_get_mon_eval_interval", lambda: 0)() or 0)

		# Keep rank control-flow identical in DDP. Loaders are intentionally
		# attached on rank 0 only; non-main ranks still call evaluate* so they
		# enter synchronize() and do not diverge into training collectives.
		run_eval = eval_interval > 0 and (
			(epoch_num % eval_interval == 0) or is_final_epoch
		)
		run_mon_eval = mon_eval_interval > 0 and not run_eval and (epoch_num % mon_eval_interval == 0)

		if run_mon_eval and hasattr(self.trainer, "evaluate_mon"):
			self.trainer.evaluate_mon()

		if run_eval:
			metrics = self.trainer.evaluate()

			if is_main_process():
				# Save best model
				preferred_metrics = getattr(self.trainer.config.eval, "metrics", None) or []
				metric_key = preferred_metrics[0] if len(preferred_metrics) > 0 else "mAP50"

				if metric_key in metrics and metrics[metric_key] > self.trainer.best_metric:
					self.trainer.best_metric = metrics[metric_key]
					self.trainer.save_checkpoint("best.pth")
					logger.info(f"New best {metric_key}: {self.trainer.best_metric:.4f}")


class DataTimeTracker(HookBase):
	"""Hook to track step time.

	The Trainer itself logs a more accurate dataloader wait time as `data_time`.
	This hook logs total step wall time as `step_time`.
	"""

	def __init__(self):
		self._t0 = None

	def before_step(self):
		self._t0 = time.perf_counter()

	def after_step(self):
		if self._t0 is None:
			return
		# Some trainers may already log step_time inside `run_step()`.
		# Avoid double-writing the same metric for the same iteration.
		try:
			if self.trainer.storage is not None:
				hist = self.trainer.storage.history("step_time")
				if len(hist) > 0 and int(hist[-1][0]) == int(self.trainer.storage.iter):
					return
		except Exception as exc:
			# Best-effort deduplication of step_time logging; failures here
			# must not break training, so we log and continue.
			logger.debug("Failed to read 'step_time' history for deduplication: %s", exc)
		step_time = time.perf_counter() - self._t0
		self.trainer.storage.put_scalar("step_time", step_time, smoothing_hint=True)


class EpochCleanupHook(HookBase):
	"""Cleanup non-training state at epoch boundaries.

	This keeps trainer classes focused on optimization logic by moving
	cache cleanup and metric-store maintenance into a hook.
	"""

	def __init__(
		self,
		*,
		clear_track_cache: bool = True,
		reset_mapper_cache: bool = True,
		clear_latest_scalars: bool = True,
	) -> None:
		self.clear_track_cache = bool(clear_track_cache)
		self.reset_mapper_cache = bool(reset_mapper_cache)
		self.clear_latest_scalars = bool(clear_latest_scalars)

	def after_epoch(self) -> None:
		if self.clear_track_cache:
			self._clear_track_cache()

		if self.reset_mapper_cache:
			self._reset_mapper_cache()

		if self.clear_latest_scalars:
			self._clear_latest_scalars()

	def _clear_track_cache(self) -> None:
		tracks = getattr(self.trainer, "_tracks_cache", None)
		tracks_t = getattr(self.trainer, "_tracks_t_cache", None)

		if not isinstance(tracks, dict):
			return

		num_cached = len(tracks)
		if num_cached <= 0:
			return

		tracks.clear()
		if isinstance(tracks_t, dict):
			tracks_t.clear()
		logger.info("Cleared %d cached track sequences", num_cached)

	def _reset_mapper_cache(self) -> None:
		for loader_name in ("train_loader", "data_loader"):
			loader = getattr(self.trainer, loader_name, None)
			if loader is None:
				continue

			dataset = getattr(loader, "dataset", None)
			mapper = self._find_mapper(dataset)
			if mapper is None:
				continue

			reset_fn = getattr(mapper, "reset_caches", None)
			if callable(reset_fn):
				try:
					reset_fn()
					logger.debug("Reset mapper caches after epoch")
				except Exception as exc:
					logger.debug("Could not reset mapper caches: %s", exc)
				break

	def _find_mapper(self, dataset: Any) -> Any:
		"""Best-effort mapper lookup through common dataset wrappers."""
		visited = set()
		current = dataset

		while current is not None:
			obj_id = id(current)
			if obj_id in visited:
				break
			visited.add(obj_id)

			mapper = getattr(current, "mapper", None)
			if mapper is None:
				mapper = getattr(current, "_mapper", None)
			if mapper is not None:
				return mapper

			next_dataset = getattr(current, "dataset", None)
			if next_dataset is None:
				next_dataset = getattr(current, "_dataset", None)
			current = next_dataset

		return None

	def _clear_latest_scalars(self) -> None:
		storage = getattr(self.trainer, "storage", None)
		if storage is None:
			return

		clear_fn = getattr(storage, "clear_latest_scalars", None)
		if callable(clear_fn):
			clear_fn()


class GradientMonitorHook(HookBase):
	"""Warn on non-finite gradients (instability signal only).

	Runs every ``interval`` steps. This hook intentionally avoids continuous
	gradient norm tracking; trainer code handles clipping and any clip-aware
	warnings with full step context.
	"""

	def __init__(self, interval: int = 50):
		self.interval = max(int(interval), 1)

	def after_backward(self):
		step = int(self.trainer.iteration)
		if step % self.interval != 0:
			return

		model = self.trainer.model
		# Unwrap DDP
		if hasattr(model, "module"):
			model = model.module

		has_nan = False

		for _name, p in model.named_parameters():
			if p.grad is None:
				continue
			g = p.grad
			if not g.isfinite().all():
				has_nan = True

		if has_nan:
			logger.warning(
				"Non-finite gradient detected before clipping at step %d",
				step,
			)


class FlowMagnitudeMonitorHook(HookBase):
	"""Log |flow| statistics from MCM._last_flow and optionally abort early.

	Catches the "decreasing CMax contrast but dead flow magnitude" failure
	mode (see docs/process/2026-04-14-flow-vs-gt-diagnostic.md): the training
	loss can appear healthy while the flow field collapses toward zero.

	Args:
		interval: log every N iterations.
		abort_after_iter: if >0, terminate training when flow is still under
			``abort_threshold`` after this many iterations. 0 disables.
		abort_threshold: |flow| mean (pixels/frame) below which we consider
			the estimator dead.
	"""

	def __init__(
		self,
		interval: int = 100,
		abort_after_iter: int = 0,
		abort_threshold: float = 1e-3,
	):
		self.interval = max(int(interval), 1)
		self.abort_after_iter = int(abort_after_iter)
		self.abort_threshold = float(abort_threshold)

	def after_step(self):
		step = int(self.trainer.iteration)
		if step % self.interval != 0:
			return

		import torch

		mcm = None
		model = self.trainer.model
		raw = model.module if hasattr(model, "module") else model
		# MCMTrainer sets .model = .mcm; full-model trainer nests under backbone.
		if hasattr(raw, "estimator") and hasattr(raw, "flow_scaling"):
			mcm = raw
		elif hasattr(raw, "backbone") and hasattr(raw.backbone, "mcm"):
			mcm = raw.backbone.mcm
		elif hasattr(raw, "mcm"):
			mcm = raw.mcm

		if mcm is None:
			return

		flow = getattr(mcm, "_last_flow", None)
		if flow is None:
			return

		with torch.no_grad():
			mag = flow.norm(dim=1)  # (B, H, W)
			mean_mag = float(mag.mean())
			p90_mag = float(mag.flatten().quantile(0.9))
			max_mag = float(mag.max())

		logger.info(
			"[flow-monitor] iter=%d  |flow| px/frame  mean=%.4e  p90=%.4e  max=%.4e",
			step, mean_mag, p90_mag, max_mag,
		)

		storage = getattr(self.trainer, "storage", None)
		if storage is not None and hasattr(storage, "put_scalar"):
			storage.put_scalar("flow/abs_mean", mean_mag)
			storage.put_scalar("flow/abs_p90", p90_mag)
			storage.put_scalar("flow/abs_max", max_mag)

		if (
			self.abort_after_iter > 0
			and step >= self.abort_after_iter
			and mean_mag < self.abort_threshold
		):
			logger.error(
				"[flow-monitor] ABORT: |flow| mean=%.4e < %.4e after %d iters. "
				"Flow estimator appears dead. Kill the job and revisit the "
				"training recipe (smooth_weight, loss_weight, crops) before "
				"wasting more cluster time.",
				mean_mag, self.abort_threshold, step,
			)
			raise RuntimeError(
				f"FlowMagnitudeMonitorHook abort: |flow| mean {mean_mag:.3e} "
				f"below threshold {self.abort_threshold:.3e} after {step} iters."
			)


class WandbHook(HookBase):
	"""Rank-0-only Weights & Biases logging via the trainer hook system."""

	def __init__(self, config: Config, output_dir: Path) -> None:
		self.cfg = config
		self.output_dir = Path(output_dir)
		self._run = None
		self._buf = WandbMetricBuffer()

	def before_train(self) -> None:
		if not is_main_process():
			return

		self._run = init_wandb(self.cfg, self.output_dir)
		if self._run is None:
			return

		wandb = _try_import_wandb()
		if wandb is None:
			return

		# Make step alignment explicit for charts.
		wandb.define_metric("train/step")
		wandb.define_metric("train/*", step_metric="train/step")
		wandb.define_metric("eval/*", step_metric="train/step")
		wandb.define_metric("mon_eval/*", step_metric="train/step")

	def after_step(self) -> None:
		if not is_main_process() or self._run is None:
			return

		step = int(self.trainer.iteration)

		reduced_step = getattr(self.trainer, "_latest_reduced_train_metrics_step", None)
		if reduced_step is None or int(reduced_step) != step:
			return
		latest: Dict[str, Any] = dict(getattr(self.trainer, "_latest_reduced_train_metrics", None) or {})
		if not latest:
			return

		payload: Dict[str, Any] = {
			"train/step": step,
			"train/epoch": float(self.trainer.epoch),
			"train/lr": float(self.trainer.optimizer.param_groups[0]["lr"]),
		}

		for k, v in latest.items():
			payload[f"train/{k}"] = v

		# GPU memory tracking (torch.cuda — nvidia-smi unavailable on IDUN)
		try:
			import torch
			if torch.cuda.is_available():
				payload["gpu/mem_alloc_mib"] = torch.cuda.memory_allocated() / (1024 ** 2)
				payload["gpu/mem_reserved_mib"] = torch.cuda.memory_reserved() / (1024 ** 2)
		except Exception:
			pass

		self._buf.add(payload)
		try:
			self.trainer._latest_reduced_train_metrics = None
			self.trainer._latest_reduced_train_metrics_step = None
		except Exception:
			pass

		# Flush aggregated metrics every M steps
		if self.cfg.wandb.flush_every_n_steps > 0 and (step % self.cfg.wandb.flush_every_n_steps) == 0:
			self._flush(step)

	def after_epoch(self) -> None:
		if not is_main_process() or self._run is None:
			return
		step = int(self.trainer.iteration)
		self._flush(step)
		self._log_validation_if_available(step)

	def after_train(self) -> None:
		if not is_main_process() or self._run is None:
			return
		step = int(self.trainer.iteration)
		self._flush(step)
		self._log_validation_if_available(step)
		try:
			self._run.finish()
		except Exception:
			pass

	def _log_validation_if_available(self, fallback_step: int) -> None:
		"""Log the most recent eval or mon-eval metrics."""
		self._log_eval_if_available(
			fallback_step=fallback_step,
			metrics_attr="_latest_eval_metrics",
			step_attr="_latest_eval_step",
			prefix="eval",
		)
		self._log_eval_if_available(
			fallback_step=fallback_step,
			metrics_attr="_latest_mon_eval_metrics",
			step_attr="_latest_mon_eval_step",
			prefix="mon_eval",
		)

	def _log_eval_if_available(self, *, fallback_step: int, metrics_attr: str, step_attr: str, prefix: str) -> None:
		wandb = _try_import_wandb()
		if wandb is None:
			return

		metrics = getattr(self.trainer, metrics_attr, None)
		if not metrics:
			return

		step = getattr(self.trainer, step_attr, None)
		if step is None:
			step = fallback_step

		payload: Dict[str, Any] = {
			"train/step": int(step),
			"train/epoch": float(self.trainer.epoch),
		}

		for k, v in dict(metrics).items():
			payload[f"{prefix}/{k}"] = v

		try:
			wandb.log(payload, step=int(step))
		finally:
			# Avoid re-logging the same eval payload on subsequent epochs/finalize.
			try:
				setattr(self.trainer, metrics_attr, None)
				setattr(self.trainer, step_attr, None)
			except Exception:
				pass

	def _flush(self, step: int) -> None:
		wandb = _try_import_wandb()
		if wandb is None:
			return

		agg = self._buf.flush()
		if not agg:
			return

		wandb.log(agg, step=step)
