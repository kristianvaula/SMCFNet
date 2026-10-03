"""
Main training script for RGB-E object detection with STTP.
"""
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from smcfnet.config.config import Config, get_config, save_config
from smcfnet.core.launch import (
	get_rank,
	get_world_size,
	get_local_rank,
	is_main_process,
	launch,
)
from smcfnet.core.trainer import Trainer
from smcfnet.model import build_eval_loader_cfg, build_test_loader_cfg
logger = logging.getLogger(__name__)

def setup_logging(output_dir: Path, rank: int) -> None:
	"""Setup logging configuration (rank-aware)."""

	class _RankFilter(logging.Filter):
		def __init__(self, rank: int):
			super().__init__()
			self._rank = int(rank)

		def filter(self, record: logging.LogRecord) -> bool:
			record.rank = self._rank
			record.pid = os.getpid()
			return True

	# Force handlers per rank (mp.spawn/torchrun can pre-configure logging).
	logging.basicConfig(
		level=logging.INFO if rank == 0 else logging.WARNING,
		format='[%(asctime)s] [rank=%(rank)s pid=%(pid)s] [%(name)s] [%(levelname)s] %(message)s',
		handlers=[
			logging.StreamHandler(),
			logging.FileHandler(output_dir / 'train.log') if rank == 0 else logging.NullHandler(),
		],
		force=True,
	)

	rank_filter = _RankFilter(rank)
	root_logger = logging.getLogger()
	for handler in root_logger.handlers:
		handler.addFilter(rank_filter)

	logging.getLogger("smcfnet").setLevel(logging.INFO if rank == 0 else logging.WARNING)


def set_random_seed(seed: int, rank: int) -> None:
	"""Set random seed for reproducibility."""
	seed = int(seed) + int(rank)
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

	torch.backends.cudnn.deterministic = False
	torch.backends.cudnn.benchmark = False


def _attach_eval_loader_if_available(trainer) -> None:
	"""Attach eval and optional mon-eval loaders (rank-0 only)."""
	if not is_main_process():
		return
	try:
		eval_cfg = getattr(trainer.config, "eval", None)
		eval_split = str(getattr(eval_cfg, "split", "val") or "val") if eval_cfg else "val"
		if not bool(getattr(trainer, "_eval_only", False)) and eval_split.lower() not in {"train", "val"}:
			raise ValueError(
				f"Training validation split must be 'train' or 'val', got '{eval_split}'. "
				"Use --eval-only or standalone evaluation scripts for test-split reporting."
			)
		trainer.val_loader = build_eval_loader_cfg(
			trainer.config,
			split=eval_split,
			sequences=list(getattr(eval_cfg, "sequences", []) or []) if eval_cfg else [],
			dataset_config_override=getattr(eval_cfg, "dataset_config", None) if eval_cfg else None,
		)
		# Derive per-sequence metadata directly from the loader's underlying dataset
		# (MappedDataset → DSECDataset/ConcatDataset → list of per-sequence datasets).
		# Avoids building the loader twice for the per-sequence flow lookup.
		try:
			mapped_ds = getattr(trainer.val_loader, "dataset", None)
			inner_ds = getattr(mapped_ds, "dataset", None)
			val_per_seq = getattr(inner_ds, "datasets", None)
			trainer._val_flow_sequence_names = (
				[getattr(ds, "sequence", None) for ds in val_per_seq]
				if val_per_seq is not None
				else None
		)
		except Exception:
			logger.exception("Failed to resolve val per-sequence metadata; flow lookup may misalign")
			trainer._val_flow_sequence_names = None
		
		# Also set eval_loader for backward compatibility
		trainer.eval_loader = trainer.val_loader
		logger.info(
			"Built validation loader with %d batches (split=%s)",
			len(trainer.val_loader),
			eval_split,
		)
	except (ValueError, TypeError, AttributeError):
		logger.exception("Invalid eval loader configuration")
		raise
	except Exception:
		logger.exception("Failed to build eval loader; continuing without eval")
		trainer.eval_loader = None
		trainer.val_loader = None
		trainer._val_flow_sequence_names = None
	try:
		mon_eval_cfg = getattr(getattr(trainer.config, "eval", None), "mon_eval", None)
		if mon_eval_cfg is None or not bool(getattr(mon_eval_cfg, "enabled", False)):
			trainer.mon_eval_loader = None
			trainer.monitor_val_loader = None
			return
		mon_eval_split = str(getattr(mon_eval_cfg, "split", "val") or "val")
		if not bool(getattr(trainer, "_eval_only", False)) and mon_eval_split.lower() not in {"train", "val"}:
			raise ValueError(
				f"Training monitor-eval split must be 'train' or 'val', got '{mon_eval_split}'. "
				"Use evaluation scripts for test-split reporting."
			)
		trainer.mon_eval_loader = build_eval_loader_cfg(
			trainer.config,
			split=mon_eval_split,
			sequences=list(getattr(mon_eval_cfg, "sequences", []) or []),
			dataset_config_override=getattr(mon_eval_cfg, "dataset_config", None),
		)
		trainer.monitor_val_loader = trainer.mon_eval_loader
		logger.info("Built mon-eval loader with %d batches", len(trainer.mon_eval_loader))
	except (ValueError, TypeError, AttributeError):
		logger.exception("Invalid mon-eval loader configuration")
		raise
	except Exception:
		logger.exception("Failed to build mon-eval loader; continuing without mon-eval")
		trainer.mon_eval_loader = None
		trainer.monitor_val_loader = None



def main_worker(cfg: Config, eval_only: bool = False, viz_only: bool = False,
                config_path: str | None = None) -> None:
	rank = get_rank()
	local_rank = get_local_rank()
	world_size = get_world_size()

	output_dir = Path(cfg.log.output_dir) / str(cfg.log.experiment_name)
	if is_main_process():
		output_dir.mkdir(parents=True, exist_ok=True)
	setup_logging(output_dir, rank)

	logger.info("Starting %s with world_size=%d", "evaluation" if eval_only else "training", world_size)
	set_random_seed(cfg.seed, rank)

	if is_main_process():
		save_config(cfg, str(output_dir / "config.yaml"))
		logger.info("Config saved to %s", output_dir / "config.yaml")
		# Campaign #129: stamp run provenance (git SHA + resolved/effective config
		# hashes + effective_config.yaml) so the run is reproducible. Source config
		# path drives resolved_config_sha256 over the full _base_ chain; fall back to
		# the saved flat config when --config was absent. Never fatal.
		try:
			from smcfnet.utils.provenance import write_provenance
			write_provenance(
				output_dir, cfg,
				config_path or str(output_dir / "config.yaml"),
				resume_path=getattr(cfg.log, "resume_from", None),
				wandb_run_id=os.environ.get("WANDB_RUN_ID"),
				dataset_id=str(getattr(cfg.eval, "split", "") or ""),
			)
			logger.info("Provenance written to %s", output_dir / "provenance.json")
		except Exception as e:  # pragma: no cover - provenance must never wedge a run
			logger.warning("Provenance stamp failed (non-fatal): %s", e)

	# Be explicit about device selection (launcher also sets this).
	if torch.cuda.is_available():
		torch.cuda.set_device(local_rank)

	# ---------------------------------------------------------------
	# Dispatch: MCM-only pre-training vs. joint detection training
	# ---------------------------------------------------------------
	training_mode = str(getattr(cfg.training, "mode", "joint") or "joint").lower()

	if training_mode == "mcm_pretrain":
		# MCM/OFE pre-training — no detection backbone involved.
		# The MCMTrainer takes care of data mapper flags internally.
		from smcfnet.core.mcm_trainer import MCMTrainer

		if eval_only or viz_only:
			raise ValueError("--eval-only / --viz-only are not supported for mcm_pretrain mode")

		mcm_trainer = MCMTrainer(cfg)
		mcm_trainer.train()
		return

	# ---------------------------------------------------------------
	# Standard joint detection training (default)
	# ---------------------------------------------------------------

	# Training pipeline expects RGB + per-frame targets to be materialized in the
	# DataLoader workers, while voxelization happens in the GPU prefetch stage.
	# We enable the fast-path defaults here (can still be overridden by YAML/CLI).
	cfg.data.mapper_load_images = True
	cfg.data.mapper_load_tracks = True
	cfg.data.mapper_build_targets = True

	# RGB-only mode: disable events so the loader uses the RGB-only collate.
	mode = str(getattr(getattr(cfg, "model", None), "input_mode", "dual") or "dual").lower()
	if mode in {"rgb", "rgb-only", "rgb_only"}:
		cfg.data.mapper_load_events = False

	trainer = Trainer(cfg)
	# Allow trainers to adjust behavior by run mode without changing public APIs.
	setattr(trainer, "_config_path", config_path)  # campaign #129 provenance
	setattr(trainer, "_eval_only", bool(eval_only))
	setattr(trainer, "_viz_only", bool(viz_only))
	_attach_eval_loader_if_available(trainer)

	if viz_only:
		if eval_only:
			raise ValueError("--viz-only and --eval-only are mutually exclusive")
		if getattr(cfg.log, "resume_from", None) is None:
			raise ValueError("--resume checkpoint path required for --viz-only")
		_maybe_swap_to_trivial_sttp_head(trainer, cfg)
		# Save PNG overlays (rank-0 only).
		if hasattr(trainer, "visualize"):
			trainer.visualize()
		else:
			raise AttributeError(f"Trainer {type(trainer).__name__} does not implement visualize()")
		return

	if eval_only:
		if getattr(cfg.log, "resume_from", None) is None:
			raise ValueError("--resume checkpoint path required for --eval-only")
		_maybe_swap_to_trivial_sttp_head(trainer, cfg)
		metrics = trainer.evaluate()
		if is_main_process():
			logger.info("Evaluation metrics: %s", metrics)
		return

	trainer.train()


def _maybe_swap_to_trivial_sttp_head(trainer, cfg) -> None:
	"""Replace the loaded STTP head with a zero-parameter FlowProjectionHead.

	Triggered by ``cfg.model.sttp.head_design == "trivial"``. Used in both
	``--eval-only`` and ``--viz-only`` modes (no training loop runs in either).
	The detector + MCM weights from the resumed checkpoint are kept; only the
	trajectory head is replaced. Any STTP weights in the resumed checkpoint are
	silently discarded.
	"""
	sttp_cfg = getattr(getattr(cfg, "model", None), "sttp", None)
	if sttp_cfg is None or not getattr(sttp_cfg, "enabled", False):
		return
	if str(getattr(sttp_cfg, "head_design", "C")).lower() != "trivial":
		return
	from smcfnet.model.sttp import FlowProjectionHead
	inner = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
	inner.sttp_head = FlowProjectionHead(
		future_steps=int(sttp_cfg.future_steps),
		num_scales=3,
		strides=(8, 16, 32),
	).to(next(inner.parameters()).device)
	if is_main_process():
		logger.info(
			"Swapped sttp_head to FlowProjectionHead (trivial baseline, "
			"future_steps=%d). Resumed STTP weights discarded.",
			int(sttp_cfg.future_steps),
		)


def main() -> None:
	cfg, args = get_config()
	eval_only = bool(getattr(args, "eval_only", False))
	viz_only = bool(getattr(args, "viz_only", False))

	num_gpus = int(getattr(cfg, "num_gpus", 1) or 1)
	launch(
		main_worker,
		num_gpus_per_machine=num_gpus,
		num_machines=1,
		machine_rank=0,
		dist_url=getattr(cfg, "dist_url", None),
		args=(cfg, eval_only, viz_only, getattr(args, "config", None)),
	)


if __name__ == "__main__":
	main()
