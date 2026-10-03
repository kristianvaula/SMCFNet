"""MCM-only trainer for pre-training flow estimators.

Trains *only* the Motion Compensation Module (OFE + warper + loss) on event
voxel data.  No detection backbone, neck, or head is involved.

This enables the progressive training workflow:
  Phase A:  ``MCMTrainer``  — pre-train OFE with self-supervised CMax loss
  Phase B:  ``Trainer``     — load MCM weights, freeze MCM, train detection
  Phase C:  ``Trainer``     — resume, unfreeze MCM with lower LR, fine-tune jointly

Usage:
  Triggered automatically when ``training.mode = "mcm_pretrain"`` in config.
  See ``train.py::main_worker`` for the dispatch logic.
"""

from __future__ import annotations

import logging
import math
import os
import subprocess
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel

from smcfnet.config.config import Config
from smcfnet.core.base import TrainerBase, HookBase
from smcfnet.core.hooks import (
    DataTimeTracker,
    EpochCleanupHook,
    LRSchedulerHook,
    PeriodicCheckpointer,
    GradientMonitorHook,
    WandbHook,
)
from smcfnet.core.launch import is_main_process, get_world_size, get_local_rank
from smcfnet.model.build import (
    _build_mcm,
    build_dsec_prefetcher,
    build_eval_loader_cfg,
    build_train_loader_cfg,
)
from smcfnet.model.mcm.weight_utils import save_mcm_checkpoint
from smcfnet.utils.metric_storage import MetricStorage
from smcfnet.dataset.dsec.prefetcher import BaseDSECPrefetcher
from smcfnet.dataset.samplers import InfiniteSampler

logger = logging.getLogger(__name__)


def _cuda_memory_snapshot(device: torch.device) -> Dict[str, float]:
    """Return a lightweight CUDA memory snapshot in MB for diagnostics."""
    mem = {
        "allocated_mb": 0.0,
        "reserved_mb": 0.0,
        "max_allocated_mb": 0.0,
        "max_reserved_mb": 0.0,
        "free_mb": 0.0,
        "total_mb": 0.0,
    }
    if not torch.cuda.is_available() or device.type != "cuda":
        return mem

    mem["allocated_mb"] = float(torch.cuda.memory_allocated(device) / (1024.0 ** 2))
    mem["reserved_mb"] = float(torch.cuda.memory_reserved(device) / (1024.0 ** 2))
    mem["max_allocated_mb"] = float(torch.cuda.max_memory_allocated(device) / (1024.0 ** 2))
    mem["max_reserved_mb"] = float(torch.cuda.max_memory_reserved(device) / (1024.0 ** 2))
    try:
        free_b, total_b = torch.cuda.mem_get_info(device)
        mem["free_mb"] = float(free_b / (1024.0 ** 2))
        mem["total_mb"] = float(total_b / (1024.0 ** 2))
    except Exception:
        pass
    return mem


def configure_mcm_pretrain_data_flags(cfg: Config) -> None:
    """Apply the dataset/materialization flags required for MCM pre-training."""
    cfg.data.mapper_load_images = False
    cfg.data.mapper_load_tracks = False
    cfg.data.mapper_build_targets = False
    cfg.data.mapper_load_events = True

    # Enable flow GT loading for supervised / combined loss
    if cfg.model.mcm.loss_type in (
        "supervised",
        "combined",
        "angular",
        "cmax_angular",
        "supervised_l1",
        "supervised_l1_tid",
        "supervised_angular",
        "event_warping_supervised",
        "event_warping_supervised_scheduled",
        "scheduled_event_warping_supervised",
    ):
        cfg.data.mapper_load_flow = True


def build_mcm_pretrain_forward_call(
    *,
    batch: Dict[str, Any],
    prepared_representation: str,
    mcm_module: nn.Module,
    tbptt_steps: int = 0,
    iteration: int = 0,
    epoch: int | None = None,
    num_epochs: int | None = None,
) -> Tuple[Optional[torch.Tensor], Dict[str, Any], str]:
    """Build representation-aware arguments for MCM pre-training forwards."""
    rep = str(prepared_representation)
    flow_gt = batch.get("flow_gt", None)

    forward_kwargs: Dict[str, Any] = {"flow_gt": flow_gt}
    if epoch is not None:
        forward_kwargs["training_epoch"] = int(epoch)
    if num_epochs is not None:
        forward_kwargs["training_num_epochs"] = int(num_epochs)
    forward_kwargs["training_iteration"] = int(iteration)

    # Forward raw events whenever available so event-warping losses keep
    # their packed-event path regardless of the primary representation.
    if "events" in batch and isinstance(batch["events"], torch.Tensor):
        forward_kwargs["events"] = batch["events"]
    if "events_offsets" in batch and isinstance(batch["events_offsets"], torch.Tensor):
        forward_kwargs["events_offsets"] = batch["events_offsets"]
    if "estimator_input" in batch and isinstance(batch["estimator_input"], torch.Tensor):
        forward_kwargs["estimator_input"] = batch["estimator_input"]
    if "mcm_estimator_input" in batch and isinstance(batch["mcm_estimator_input"], torch.Tensor):
        forward_kwargs["mcm_estimator_input"] = batch["mcm_estimator_input"]

    if rep == "events_voxel":
        voxel = batch.get("events_voxel", None)
        if not isinstance(voxel, torch.Tensor):
            raise ValueError("Voxel prefetcher requires tensor batch['events_voxel']")
        return voxel, forward_kwargs, rep

    if rep == "streaming_events":
        # Keep timestamps in absolute microseconds and delegate internal
        # temporal partitioning/state handling to SMCM/SOFE.
        required = ("events", "events_offsets", "t_start_us", "t_end_us", "t_ref_us")
        missing = [k for k in required if k not in batch]
        if missing:
            raise ValueError(f"Streaming batch missing required keys: {missing}")

        for key in required:
            forward_kwargs[key] = batch[key]

        if "seq_idx" in batch:
            forward_kwargs["seq_idx"] = batch["seq_idx"]
        if "frame_index" in batch:
            forward_kwargs["frame_index"] = batch["frame_index"]

        # Reset inference/consumption belongs to SMCM so the trainer does not
        # split ownership of sequence-boundary semantics.
        if "sequence_reset" in batch and isinstance(batch["sequence_reset"], torch.Tensor):
            forward_kwargs["sequence_reset"] = batch["sequence_reset"]

        # The trainer still performs backward/optimizer-step once per
        # detector-aligned batch, so carried state must be detached at each
        # outer iteration boundary. When configured, finer-grained intra-window
        # TBPTT is handled inside the streaming SMCM forward path.
        if int(iteration) > 0:
            forward_kwargs["detach_state"] = True

        return None, forward_kwargs, rep

    raise ValueError(f"Unsupported prefetch representation: {rep!r}")


def _build_mcm_optimizer(
    cfg: Config,
    mcm: nn.Module,
) -> torch.optim.Optimizer:
    """Build an optimizer for MCM-only training.

    Mirrors the main ``build_optimizer`` but operates only on MCM parameters.
    """
    tcfg = cfg.training
    opt_name = str(tcfg.optimizer).lower()
    lr = float(tcfg.learning_rate)
    wd = float(tcfg.weight_decay)

    params = [p for p in mcm.parameters() if p.requires_grad]
    if not params:
        raise ValueError("MCM module has no trainable parameters")

    if opt_name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if opt_name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if opt_name == "sgd":
        momentum = float(getattr(tcfg, "momentum", 0.9))
        return torch.optim.SGD(params, lr=lr, weight_decay=wd, momentum=momentum)
    raise ValueError(f"Unknown optimizer: {opt_name}")


def _build_mcm_scheduler(
    cfg: Config,
    optimizer: torch.optim.Optimizer,
) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    """Build LR scheduler for MCM-only training."""
    tcfg = cfg.training
    name = str(getattr(tcfg, "lr_scheduler", "cosine") or "cosine").lower()
    total_epochs = int(getattr(tcfg, "num_epochs", 1) or 1)

    if name == "cosine":
        eta_min = float(getattr(tcfg, "lr_eta_min", None) or getattr(tcfg, "warmup_lr", 0.0) or 0.0)
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs, eta_min=eta_min,
        )
    if name == "step":
        step_size = int(getattr(tcfg, "lr_step_size", 50))
        gamma = float(getattr(tcfg, "lr_gamma", 0.1))
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    if name == "exponential":
        gamma = float(getattr(tcfg, "lr_gamma", 0.1))
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    if name == "onecycle":
        steps_per_epoch = int(getattr(tcfg, "steps_per_epoch", 0) or 0)
        total_steps = int(getattr(tcfg, "total_steps", 0) or 0)
        if total_steps <= 0:
            if steps_per_epoch <= 0:
                raise ValueError("OneCycleLR requires training.steps_per_epoch or training.total_steps")
            total_steps = int(total_epochs) * int(steps_per_epoch)
        max_lr = float(getattr(tcfg, "learning_rate", 0.0) or 0.0)
        if max_lr <= 0:
            raise ValueError("OneCycleLR requires a positive learning_rate")
        pct_start = float(getattr(tcfg, "onecycle_pct_start", 0.3))
        div_factor = float(getattr(tcfg, "onecycle_div_factor", 25.0))
        final_div_factor = float(getattr(tcfg, "onecycle_final_div_factor", 1e4))
        sched = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=int(total_steps),
            pct_start=pct_start,
            div_factor=div_factor,
            final_div_factor=final_div_factor,
            anneal_strategy="cos",
        )
        sched._step_per_batch = True
        return sched
    if name == "none":
        return None

    logger.warning("Unknown scheduler '%s' for MCM trainer, using cosine", name)
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)


class MCMTrainer(TrainerBase):
    """Trainer for MCM/OFE pre-training (no detection backbone).

    Builds only the MCM module and trains it with its own loss function
    (CMaxLoss / SupervisedFlowLoss / CombinedLoss).

    Data flow per step:
      1.  Fetch batch from GPU prefetcher (packed events → voxel grid)
      2.  ``corrected, loss = mcm(voxel)``
      3.  Backprop loss, optimizer step

    Checkpoints are MCM-only (see ``weight_utils.save_mcm_checkpoint``).
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.config = cfg  # alias used by hooks

        self.device = self._get_device()
        if self.device.type == "cuda":
            torch.cuda.set_device(get_local_rank())
            self._log_cuda_device_context(tag="startup")
            self._log_cuda_memory(tag="startup_pre_build")
            try:
                torch.cuda.reset_peak_memory_stats(self.device)
            except Exception:
                pass

        # Build MCM module only
        if not cfg.model.mcm.enabled:
            raise ValueError(
                "MCM pre-training requires model.mcm.enabled=True"
            )
        try:
            self.mcm = _build_mcm(cfg).to(self.device)
        except RuntimeError as exc:
            if self.device.type == "cuda" and "out of memory" in str(exc).lower():
                logger.exception(
                    "CUDA OOM while moving MCM module to %s (local_rank=%d).",
                    str(self.device),
                    int(get_local_rank()),
                )
                self._log_cuda_memory(tag="startup_oom_on_to_cuda")
                self._log_nvidia_smi_compute_processes()
            raise

        if self.device.type == "cuda":
            self._log_cuda_memory(tag="startup_post_build")
        num_params = sum(p.numel() for p in self.mcm.parameters())
        logger.info("Built MCM module with %d parameters", int(num_params))

        # Load pre-existing MCM weights for warm-restart
        if getattr(cfg.model.mcm, "pretrained_weights", None):
            from smcfnet.model.mcm.weight_utils import load_mcm_weights_into_module
            load_mcm_weights_into_module(self.mcm, cfg.model.mcm.pretrained_weights)

        # Wrap with DDP when running multi-process
        self.mcm = self._wrap_ddp_if_needed(self.mcm)
        # Alias for hooks that expect trainer.model
        self.model = self.mcm

        self.optimizer = _build_mcm_optimizer(cfg, self.mcm)

        # Data loader (events only — no need for images or tracks)
        configure_mcm_pretrain_data_flags(cfg)

        self.data_loader = build_train_loader_cfg(cfg)
        self.train_loader = self.data_loader
        self.val_loader = self._build_validation_loader() if is_main_process() else None

        # Infer steps_per_epoch for OneCycle if not provided.
        if str(getattr(cfg.training, "lr_scheduler", "")).lower() == "onecycle":
            if getattr(cfg.training, "steps_per_epoch", None) in (None, 0):
                try:
                    cfg.training.steps_per_epoch = len(self.train_loader)
                except Exception:
                    pass

        self.scheduler = _build_mcm_scheduler(cfg, self.optimizer)

        # Tracking
        self.epoch: int = 0
        self.best_metric: float = 0.0
        self._best_metric_initialized: bool = False
        self._best_metric_name: Optional[str] = None
        self._best_metric_higher_is_better: bool = True
        self._latest_val_metrics: Optional[Dict[str, float]] = None
        self._latest_val_step: Optional[int] = None

        # Mixed precision
        self.use_amp = bool(cfg.training.use_amp and torch.cuda.is_available())
        self.scaler = GradScaler("cuda") if self.use_amp else None

        # Lazy loader/prefetcher
        self._train_loader_iter = None
        self._train_prefetcher: Optional[BaseDSECPrefetcher] = None
        self._active_event_representation: Optional[str] = None

        # Truncated-BPTT cadence for streaming/stateful estimators.
        # 0/None disables trainer-side state detachment.
        self._tbptt_steps = int(getattr(cfg.model.mcm, "bptt_steps", 0) or 0)
        self._streaming_step_policy = str(
            getattr(cfg.training, "streaming_step_policy", "outer_window") or "outer_window"
        ).lower()
        if self._streaming_step_policy == "chunk_accumulate":
            self._streaming_step_policy = "chunk_backward"
        if self._streaming_step_policy not in {"outer_window", "chunk_backward"}:
            raise ValueError(
                "training.streaming_step_policy must be 'outer_window' or "
                f"'chunk_backward', got {self._streaming_step_policy!r}"
            )
        if (
            str(getattr(cfg.model.mcm, "input_representation", "voxel")).lower() == "streaming_events"
            and self._tbptt_steps > 1
        ):
            logger.info(
                "Streaming MCM intra-window TBPTT enabled with model.mcm.bptt_steps=%d. "
                "streaming_step_policy=%s.",
                self._tbptt_steps,
                self._streaming_step_policy,
            )

        # Metric storage
        self.storage = MetricStorage(0)

        # Output directory
        self.output_dir = Path(cfg.log.output_dir) / str(cfg.log.experiment_name)
        if is_main_process():
            self.output_dir.mkdir(parents=True, exist_ok=True)
            from torch.utils.tensorboard import SummaryWriter
            self.writer: Optional[SummaryWriter] = SummaryWriter(log_dir=str(self.output_dir / "tb"))
        else:
            self.writer = None

        # Step timing
        self._last_step_end_time: Optional[float] = None
        self._nonfinite_debug_logged: bool = False

        # Memory logging cadence for long runs.
        self._mem_log_interval_steps = max(int(getattr(self.cfg.log, "memory_log_interval", 10) or 10), 1)

        # Register hooks
        self.register_hooks(self.build_hooks())

        # Resume from checkpoint
        if getattr(cfg.log, "resume_from", None):
            self.resume_checkpoint(str(cfg.log.resume_from))

    # ------------------------------------------------------------------
    # Data pipeline
    # ------------------------------------------------------------------

    @property
    def train_data_loader_iter(self):
        if self._train_loader_iter is None:
            self._train_loader_iter = iter(self.data_loader)
        return self._train_loader_iter

    def reset_data_loader_iter(self) -> None:
        self._train_loader_iter = None
        self._train_prefetcher = None

    def _train_sampler_is_infinite(self) -> bool:
        return isinstance(getattr(self.train_loader, "sampler", None), InfiniteSampler)

    def _set_train_sampler_epoch(self, epoch: int) -> None:
        sampler = getattr(self.train_loader, "sampler", None)
        set_epoch = getattr(sampler, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(int(epoch))

    def _prepare_train_epoch_data_stream(self, epoch: int) -> None:
        if self._train_sampler_is_infinite():
            if not getattr(self, "_logged_continuous_infinite_sampler", False) and is_main_process():
                logger.info(
                    "Keeping InfiniteSampler train stream continuous across logical epochs; "
                    "max_steps_per_epoch controls checkpoint/eval cadence only."
                )
                self._logged_continuous_infinite_sampler = True
            return

        self._set_train_sampler_epoch(epoch)
        self.reset_data_loader_iter()

    def _get_train_prefetcher(self) -> BaseDSECPrefetcher:
        if self._train_prefetcher is None:
            ev_h = int(getattr(self.cfg.data, "image_height", 480))
            ev_w = int(getattr(self.cfg.data, "image_width", 640))
            self._train_prefetcher = build_dsec_prefetcher(
                self.train_data_loader_iter,
                config=self.cfg,
                device=self.device,
                require_targets=False,  # MCM pre-training doesn't need targets
                voxel_height=ev_h,
                voxel_width=ev_w,
            )
            self._active_event_representation = str(self._train_prefetcher.prepared_representation)
        return self._train_prefetcher

    def _build_validation_loader(self):
        eval_cfg = getattr(self.cfg, "eval", None)
        eval_interval = int(getattr(eval_cfg, "interval", 0) or 0) if eval_cfg else 0
        if eval_interval <= 0:
            return None

        # Source order: mon_eval (if enabled) -> eval_cfg (NOT test split).
        # The earlier fallback to build_test_loader_cfg silently selected best
        # checkpoints on the official test split — see #022 (cause + audit) and
        # the symmetric fix in core/train.py:75 (#014).
        mon_cfg = getattr(eval_cfg, "mon_eval", None) if eval_cfg else None
        if mon_cfg is not None and bool(getattr(mon_cfg, "enabled", False)):
            source_cfg = mon_cfg
            source_label = "mon_eval"
        else:
            source_cfg = eval_cfg
            source_label = "eval"

        split = str(getattr(source_cfg, "split", "test") or "test").lower() if source_cfg else "test"
        if split not in {"train", "val", "test"}:
            raise ValueError(
                f"Unsupported MCM validation split '{split}'. Expected 'train', 'val', or 'test'."
            )

        sequences_raw = list(getattr(source_cfg, "sequences", []) or []) if source_cfg else []
        sequences = [str(s).strip() for s in sequences_raw if str(s).strip()]

        dataset_config_override = getattr(source_cfg, "dataset_config", None) if source_cfg else None
        if isinstance(dataset_config_override, str):
            dataset_config_override = dataset_config_override.strip() or None

        if split == "train" and len(sequences) == 0:
            logger.warning(
                "MCM validation split is 'train' but no heldout sequence override was provided. "
                "Validation will run over the full train split."
            )

        try:
            val_loader = build_eval_loader_cfg(
                self.cfg,
                split=split,
                sequences=sequences or None,
                dataset_config_override=dataset_config_override,
            )
        except (TypeError, AttributeError):
            logger.exception("MCM validation loader misconfigured (signature/config error)")
            raise
        except Exception:
            logger.exception("Failed to build MCM validation loader; continuing without validation")
            return None

        logger.info(
            "Built MCM validation loader (source=%s): split=%s batches=%d explicit_sequences=%d",
            source_label,
            split,
            len(val_loader),
            len(sequences),
        )
        return val_loader

    def get_validation_metric_spec(self) -> Tuple[str, bool]:
        """Return (metric_name, higher_is_better) for best-checkpoint selection."""
        return "val_total", False

    def _resolve_validation_metric_value(self, metrics: Dict[str, float]) -> Tuple[Optional[str], Optional[float], bool]:
        metric_name, higher_is_better = self.get_validation_metric_spec()
        metric_name = str(metric_name)
        candidates = [metric_name]
        if metric_name.startswith("val_"):
            candidates.append(metric_name[4:])
        else:
            candidates.append(f"val_{metric_name}")

        for key in candidates:
            if key not in metrics:
                continue
            value = float(metrics[key])
            return key, value, bool(higher_is_better)

        return None, None, bool(higher_is_better)

    def _should_run_validation_this_epoch(self) -> bool:
        eval_interval = int(getattr(getattr(self.config, "eval", None), "eval_interval", 0) or 0)
        if eval_interval <= 0 or self.val_loader is None:
            return False
        limited_smoke_run = (
            int(getattr(self.config.training, "max_steps_per_epoch", 0) or 0) > 0
            or int(getattr(self.config.training, "max_total_steps", 0) or 0) > 0
        )
        if limited_smoke_run:
            return (int(self.epoch) + 1) % eval_interval == 0
        is_final_epoch = (int(self.epoch) + 1) >= int(self.config.training.num_epochs)
        return ((int(self.epoch) + 1) % eval_interval == 0) or is_final_epoch

    def _update_best_checkpoint_from_metrics(self, metrics: Dict[str, float]) -> None:
        if not is_main_process():
            return

        metric_key, metric_value, higher_is_better = self._resolve_validation_metric_value(metrics)
        if metric_key is None or metric_value is None:
            logger.warning(
                "Validation finished but could not resolve best-checkpoint metric. "
                "Requested=%s available=%s",
                str(self.get_validation_metric_spec()[0]),
                sorted(metrics.keys()),
            )
            return

        if not self._best_metric_initialized:
            is_better = True
        elif higher_is_better:
            is_better = float(metric_value) > float(self.best_metric)
        else:
            is_better = float(metric_value) < float(self.best_metric)

        if not is_better:
            return

        self.best_metric = float(metric_value)
        self._best_metric_initialized = True
        self._best_metric_name = str(metric_key)
        self._best_metric_higher_is_better = bool(higher_is_better)
        self.save_checkpoint("best.pth")
        logger.info(
            "New best %s=%s (mode=%s)",
            str(metric_key),
            self._format_metric_value(float(metric_value)),
            "max" if higher_is_better else "min",
        )

    def _log_validation(self, metrics: Dict[str, float]) -> None:
        if not is_main_process():
            return
        if not metrics:
            logger.warning("[MCM][VAL] no metrics produced")
            return

        parts = [f"[MCM][VAL] epoch={self.epoch} iter={self.iteration}"]
        for k in sorted(metrics.keys()):
            parts.append(f"{k}={self._format_metric_value(float(metrics[k]))}")
        logger.info("  ".join(parts))

        if self.writer is not None:
            for k, v in metrics.items():
                self.writer.add_scalar(f"mcm_val/{k}", float(v), self.iteration)

    def validate(self) -> Dict[str, float]:
        """Run MCM validation (rank-0 only).

        Metrics are averaged per-batch over the configured validation loader.
        """
        if self.val_loader is None:
            return {}

        dist_ready = bool(dist.is_available() and dist.is_initialized())
        if dist_ready:
            dist.barrier()

        if not is_main_process():
            if dist_ready:
                dist.barrier()
            return {}

        mcm_wrapper = self.mcm
        mcm = self._unwrap_mcm()

        wrapper_was_training = bool(mcm_wrapper.training)
        module_was_training = bool(mcm.training)

        # Keep submodules in eval mode, but force the wrapper-level training
        # flag so loss_fn still executes in forward() for validation metrics.
        mcm_wrapper.eval()
        mcm.training = True
        mcm.reset_state()

        metrics_sums: Dict[str, float] = defaultdict(float)
        metrics_counts: Dict[str, int] = defaultdict(int)
        max_val_batches = int(getattr(getattr(self.config, "eval", None), "max_batches", 0) or 0)

        try:
            val_iter = iter(self.val_loader)
            ev_h = int(getattr(self.cfg.data, "image_height", 480))
            ev_w = int(getattr(self.cfg.data, "image_width", 640))
            prefetch = build_dsec_prefetcher(
                val_iter,
                config=self.cfg,
                device=self.device,
                require_targets=False,
                voxel_height=ev_h,
                voxel_width=ev_w,
            )

            with torch.no_grad():
                batches_seen = 0
                while True:
                    if max_val_batches > 0 and batches_seen >= max_val_batches:
                        break
                    try:
                        batch = prefetch.next()
                    except StopIteration:
                        break
                    batches_seen += 1

                    primary_input, forward_kwargs, rep = build_mcm_pretrain_forward_call(
                        batch=batch,
                        prepared_representation=str(prefetch.prepared_representation),
                        mcm_module=mcm,
                        tbptt_steps=0,
                        iteration=0,
                        epoch=self.epoch,
                        num_epochs=self.config.training.num_epochs,
                    )
                    if rep == "events_voxel":
                        mcm.reset_state()

                    if self.use_amp:
                        with autocast("cuda"):
                            _corrected, loss = mcm_wrapper(primary_input, **forward_kwargs)
                    else:
                        _corrected, loss = mcm_wrapper(primary_input, **forward_kwargs)

                    if isinstance(loss, torch.Tensor):
                        metrics_sums["val_total"] += float(loss.detach().item())
                        metrics_counts["val_total"] += 1

                    contrast = getattr(mcm, "_last_contrast_loss", None)
                    if isinstance(contrast, torch.Tensor):
                        metrics_sums["val_mcm_contrast"] += float(contrast.detach().item())
                        metrics_counts["val_mcm_contrast"] += 1

                    smooth = getattr(mcm, "_last_smooth_loss", None)
                    if isinstance(smooth, torch.Tensor):
                        metrics_sums["val_mcm_smooth"] += float(smooth.detach().item())
                        metrics_counts["val_mcm_smooth"] += 1

                    magnitude = getattr(mcm, "_last_magnitude_loss", None)
                    if isinstance(magnitude, torch.Tensor):
                        metrics_sums["val_mcm_magnitude"] += float(magnitude.detach().item())
                        metrics_counts["val_mcm_magnitude"] += 1

                    epe = getattr(mcm, "_last_epe", None)
                    if isinstance(epe, torch.Tensor):
                        metrics_sums["val_mcm_epe"] += float(epe.detach().item())
                        metrics_counts["val_mcm_epe"] += 1
        finally:
            mcm.reset_state()
            mcm.training = module_was_training
            mcm_wrapper.train(wrapper_was_training)

        metrics: Dict[str, float] = {}
        for key, total in metrics_sums.items():
            count = max(int(metrics_counts.get(key, 0)), 1)
            metrics[key] = float(total) / float(count)

        if "val_mcm_contrast" in metrics:
            metrics["val_iwe_contrast"] = -float(metrics["val_mcm_contrast"])

        self._latest_val_metrics = dict(metrics)
        self._latest_val_step = int(self.iteration)
        self._log_validation(metrics)

        if dist_ready:
            dist.barrier()
        return metrics

    def _log_cuda_device_context(self, *, tag: str) -> None:
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return
        dev_idx = int(self.device.index if self.device.index is not None else get_local_rank())
        device_name = torch.cuda.get_device_name(dev_idx)
        logger.info(
            "[CUDA][%s] device=%s index=%d local_rank=%d world_size=%d device_count=%d "
            "CUDA_VISIBLE_DEVICES=%s",
            str(tag),
            device_name,
            dev_idx,
            int(get_local_rank()),
            int(get_world_size()),
            int(torch.cuda.device_count()),
            os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        )

    def _log_cuda_memory(self, *, tag: str) -> Dict[str, float]:
        mem = _cuda_memory_snapshot(self.device)
        if self.device.type == "cuda" and torch.cuda.is_available():
            logger.info(
                "[MEM][%s] allocated=%.2fMB reserved=%.2fMB peak_alloc=%.2fMB "
                "peak_reserved=%.2fMB free=%.2fMB total=%.2fMB",
                str(tag),
                mem["allocated_mb"],
                mem["reserved_mb"],
                mem["max_allocated_mb"],
                mem["max_reserved_mb"],
                mem["free_mb"],
                mem["total_mb"],
            )
        return mem

    @staticmethod
    def _log_nvidia_smi_compute_processes() -> None:
        """Best-effort nvidia-smi process snapshot for OOM triage."""
        commands = [
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader",
            ],
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader",
            ],
        ]
        for cmd in commands:
            try:
                proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
                if proc.stdout.strip():
                    logger.info("[nvidia-smi] %s\n%s", " ".join(cmd[1:]), proc.stdout.strip())
                if proc.stderr.strip():
                    logger.warning("[nvidia-smi][stderr] %s", proc.stderr.strip())
            except Exception:
                logger.exception("Failed to run nvidia-smi diagnostics command: %s", " ".join(cmd))

    def _unwrap_mcm(self) -> nn.Module:
        return self.mcm.module if hasattr(self.mcm, "module") else self.mcm

    def _clip_gradients_and_check_finite(self) -> bool:
        grad_clip = float(getattr(self.config.training, "grad_clip", 0.0))
        if grad_clip <= 0:
            return True

        params = [p for p in self.mcm.parameters() if isinstance(getattr(p, "grad", None), torch.Tensor)]
        if not params:
            return True

        # PyTorch's default clip_grad_norm_ computes the aggregate norm in the
        # gradients' dtype. Large but finite fp32 gradients can overflow the
        # aggregate norm to inf, after which the trainer skips every step even
        # though no parameter gradient contains NaN/Inf. Accumulate in fp64 so
        # true-SNN supervised pretraining can recover via clipping.
        total_sq = 0.0
        for param in params:
            grad = param.grad.detach()
            if not bool(torch.isfinite(grad).all()):
                logger.warning("Non-finite pre-clip gradient norm at iter %d", self.iteration)
                return False
            grad_norm = float(torch.linalg.vector_norm(grad.double(), ord=2).item())
            if not math.isfinite(grad_norm):
                logger.warning("Non-finite pre-clip gradient norm at iter %d", self.iteration)
                return False
            total_sq += grad_norm * grad_norm

        preclip_norm_f = math.sqrt(total_sq)
        if not math.isfinite(preclip_norm_f):
            logger.warning("Non-finite pre-clip gradient norm at iter %d", self.iteration)
            return False

        clip_coef = grad_clip / (preclip_norm_f + 1e-6)
        if clip_coef < 1.0:
            clip_coef_t = torch.tensor(clip_coef, device=params[0].grad.device, dtype=params[0].grad.dtype)
            for param in params:
                param.grad.detach().mul_(clip_coef_t.to(device=param.grad.device, dtype=param.grad.dtype))
        return True

    def _skip_optimizer_step_for_nonfinite_grads(self, *, prev_scale: Optional[float] = None) -> None:
        logger.warning(
            "Skipping optimizer step and resetting streaming state due to non-finite gradients at iter %d",
            self.iteration,
        )
        self.optimizer.zero_grad(set_to_none=True)
        self._unwrap_mcm().reset_state()

        if self.scaler is not None and prev_scale is not None:
            new_scale = max(float(prev_scale) * 0.5, 1.0)
            self.scaler.update(new_scale=new_scale)
            logger.warning(
                "Reduced AMP GradScaler scale from %.1f to %.1f after non-finite gradients at iter %d",
                float(prev_scale),
                float(new_scale),
                self.iteration,
            )

    def _build_loss_dict(self, loss: torch.Tensor) -> Dict[str, Any]:
        loss_dict: Dict[str, Any] = {"total": loss}
        mcm = self._unwrap_mcm()
        if hasattr(mcm, "_last_contrast_loss"):
            loss_dict["mcm_contrast"] = mcm._last_contrast_loss
        if hasattr(mcm, "_last_smooth_loss"):
            loss_dict["mcm_smooth"] = mcm._last_smooth_loss
        if hasattr(mcm, "_last_smooth_loss_raw") and isinstance(mcm._last_smooth_loss_raw, torch.Tensor):
            loss_dict["mcm_smooth_raw"] = mcm._last_smooth_loss_raw
        if hasattr(mcm, "_last_magnitude_loss"):
            loss_dict["mcm_magnitude"] = mcm._last_magnitude_loss
        if hasattr(mcm, "_last_epe"):
            loss_dict["mcm_epe"] = mcm._last_epe
        if hasattr(mcm, "_last_schedule_ew_weight"):
            loss_dict["mcm_schedule_ew_weight"] = mcm._last_schedule_ew_weight
        if hasattr(mcm, "_last_schedule_sup_weight"):
            loss_dict["mcm_schedule_sup_weight"] = mcm._last_schedule_sup_weight
        if hasattr(mcm, "_last_center_loss") and isinstance(mcm._last_center_loss, torch.Tensor):
            loss_dict["mcm_center"] = mcm._last_center_loss
        if hasattr(mcm, "_last_centered_ratio") and isinstance(mcm._last_centered_ratio, torch.Tensor):
            loss_dict["mcm_centered_ratio"] = mcm._last_centered_ratio
        return loss_dict

    def _sync_skip_optimizer_step(self, local_skip: bool) -> bool:
        if int(get_world_size()) <= 1 or not dist.is_available() or not dist.is_initialized():
            return bool(local_skip)

        device = self.device if self.device.type == "cuda" else torch.device("cpu")
        flag = torch.tensor([1 if local_skip else 0], dtype=torch.int32, device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(int(flag.item()) > 0)

    def _sync_streaming_chunk_count(self, num_chunks: int) -> Tuple[int, int]:
        local = int(num_chunks)
        if int(get_world_size()) <= 1 or not dist.is_available() or not dist.is_initialized():
            return local, local

        device = self.device if self.device.type == "cuda" else torch.device("cpu")
        counts = torch.tensor([local, local], dtype=torch.int32, device=device)
        dist.all_reduce(counts[0:1], op=dist.ReduceOp.MIN)
        dist.all_reduce(counts[1:2], op=dist.ReduceOp.MAX)
        return int(counts[0].item()), int(counts[1].item())

    def _ddp_no_sync_context(self, enabled: bool):
        if enabled and hasattr(self.mcm, "no_sync"):
            return self.mcm.no_sync()
        return nullcontext()

    def _zero_loss_for_ddp_sync(self) -> torch.Tensor:
        zero: Optional[torch.Tensor] = None
        for param in self.mcm.parameters():
            if not param.requires_grad:
                continue
            term = param.float().sum() * 0.0
            zero = term if zero is None else zero + term
        if zero is None:
            raise RuntimeError("Cannot build dummy sync loss: MCM has no trainable parameters")
        return zero

    def _first_nonfinite_grad_param(self) -> Optional[Tuple[str, int, float]]:
        for name, param in self.mcm.named_parameters():
            grad = getattr(param, "grad", None)
            if not isinstance(grad, torch.Tensor):
                continue
            finite_mask = torch.isfinite(grad)
            if bool(finite_mask.all()):
                continue
            nonfinite_count = int((~finite_mask).sum().item())
            finite_vals = grad[finite_mask]
            max_abs_finite = float(finite_vals.abs().max().item()) if finite_vals.numel() > 0 else float("nan")
            return str(name), nonfinite_count, max_abs_finite
        return None

    def _log_nonfinite_diagnostics_once(
        self,
        *,
        reason: str,
        loss: torch.Tensor,
        loss_dict: Dict[str, Any],
        check_grads: bool,
    ) -> None:
        if self._nonfinite_debug_logged:
            return
        self._nonfinite_debug_logged = True

        logger.error("[MCM][NaN-Debug] First non-finite event at iter %d: %s", self.iteration, reason)
        if isinstance(loss, torch.Tensor):
            logger.error(
                "[MCM][NaN-Debug] total loss finite=%s value=%s",
                str(bool(torch.isfinite(loss.detach()).all())),
                str(float(loss.detach().item()) if loss.numel() == 1 else tuple(loss.shape)),
            )

        for key in sorted(loss_dict.keys()):
            val = loss_dict[key]
            if isinstance(val, torch.Tensor):
                t = val.detach()
                finite = bool(torch.isfinite(t).all())
                if t.numel() == 1:
                    logger.error("[MCM][NaN-Debug] %s finite=%s value=%s", key, str(finite), str(float(t.item())))
                else:
                    logger.error(
                        "[MCM][NaN-Debug] %s finite=%s shape=%s",
                        key,
                        str(finite),
                        str(tuple(int(v) for v in t.shape)),
                    )
            elif isinstance(val, (int, float)):
                logger.error("[MCM][NaN-Debug] %s value=%s", key, str(float(val)))

        if check_grads:
            bad = self._first_nonfinite_grad_param()
            if bad is None:
                logger.error("[MCM][NaN-Debug] No explicit non-finite parameter gradient found.")
            else:
                name, count, max_abs_finite = bad
                logger.error(
                    "[MCM][NaN-Debug] First non-finite grad param=%s nonfinite_count=%d max_abs_finite=%s",
                    name,
                    int(count),
                    str(max_abs_finite),
                )

    def _build_mcm_call(
        self,
        batch: Dict[str, Any],
    ) -> Tuple[Optional[torch.Tensor], Dict[str, Any], str]:
        """Build representation-aware arguments for ``self.mcm(...)``."""
        return build_mcm_pretrain_forward_call(
            batch=batch,
            prepared_representation=str(self._get_train_prefetcher().prepared_representation),
            mcm_module=self._unwrap_mcm(),
            tbptt_steps=self._tbptt_steps,
            iteration=self.iteration,
            epoch=self.epoch,
            num_epochs=self.config.training.num_epochs,
        )

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def _warmup_lr(self, batch_idx: int, num_batches: int) -> None:
        tcfg = self.config.training
        total_warmup_iters = int(tcfg.warmup_epochs) * int(num_batches)
        current_iter = int(self.epoch) * int(num_batches) + int(batch_idx)
        if total_warmup_iters <= 0 or current_iter >= total_warmup_iters:
            return
        warmup_factor = float(current_iter) / float(total_warmup_iters)
        base_warmup_lr = float(tcfg.warmup_lr)
        base_target_lr = float(tcfg.learning_rate)
        for param_group in self.optimizer.param_groups:
            # Prefer a stable scheduler-provided initial_lr if available.
            if "initial_lr" in param_group:
                target_lr = float(param_group["initial_lr"])
            else:
                # When no scheduler is used, cache a stable target LR the first time.
                if "_warmup_target_lr" not in param_group:
                    param_group["_warmup_target_lr"] = float(param_group["lr"])
                target_lr = float(param_group["_warmup_target_lr"])
            ratio = target_lr / base_target_lr if base_target_lr > 0 else 1.0
            group_warmup_lr = base_warmup_lr * ratio
            param_group["lr"] = group_warmup_lr + (target_lr - group_warmup_lr) * warmup_factor

    @staticmethod
    def _collect_flow_diagnostics(flow: torch.Tensor, *, prefix: str = "flow") -> Dict[str, torch.Tensor]:
        diagnostics: Dict[str, torch.Tensor] = {}
        if not isinstance(flow, torch.Tensor) or flow.numel() == 0:
            return diagnostics

        prefix = str(prefix)

        def key(name: str) -> str:
            return f"{prefix}_{name}"

        with torch.no_grad():
            flow_stats = flow.float()
            diagnostics[key("mean")] = flow_stats.mean()
            diagnostics[key("std")] = flow_stats.std()
            diagnostics[key("min")] = flow_stats.min()
            diagnostics[key("max")] = flow_stats.max()

            mag = flow_stats.norm(p=2, dim=1)
            diagnostics[key("mag_mean")] = mag.mean()
            diagnostics[key("mag_std")] = mag.std()

            if int(flow_stats.shape[1]) >= 2:
                u = flow_stats[:, 0]
                v = flow_stats[:, 1]
                diagnostics[key("u_mean")] = u.mean()
                diagnostics[key("v_mean")] = v.mean()

                u_hw = u.reshape(u.shape[0], -1)
                v_hw = v.reshape(v.shape[0], -1)
                diagnostics[key("u_spatial_std_mean")] = u_hw.std(dim=1).mean()
                diagnostics[key("v_spatial_std_mean")] = v_hw.std(dim=1).mean()
                diagnostics[key("u_range_mean")] = (u_hw.max(dim=1).values - u_hw.min(dim=1).values).mean()
                diagnostics[key("v_range_mean")] = (v_hw.max(dim=1).values - v_hw.min(dim=1).values).mean()

                mean_flow = flow_stats.mean(dim=(2, 3), keepdim=True)
                centered = flow_stats - mean_flow
                diagnostics[key("centered_mag_mean")] = centered.norm(p=2, dim=1).mean()

                eps = torch.finfo(flow_stats.dtype).eps
                mean_flow_mag = torch.sqrt(u.mean().square() + v.mean().square())
                diagnostics[key("global_flow_ratio")] = diagnostics[key("centered_mag_mean")] / (diagnostics[key("mag_mean")] + eps)
                diagnostics[key("mean_flow_ratio")] = mean_flow_mag / (diagnostics[key("mag_mean")] + eps)

            flow_dx = flow_stats[:, :, 1:, :] - flow_stats[:, :, :-1, :]
            flow_dy = flow_stats[:, :, :, 1:] - flow_stats[:, :, :, :-1]
            diagnostics[key("dx_mean")] = flow_dx.abs().mean()
            diagnostics[key("dy_mean")] = flow_dy.abs().mean()
            diagnostics[key("dx_max")] = flow_dx.abs().max()
            diagnostics[key("dy_max")] = flow_dy.abs().max()

            if int(flow_stats.shape[1]) >= 2:
                u = flow_stats[:, 0:1]
                v = flow_stats[:, 1:2]
                du_dx = u[:, :, :, 1:] - u[:, :, :, :-1]
                dv_dy = v[:, :, 1:, :] - v[:, :, :-1, :]
                min_h = min(int(du_dx.shape[2]), int(dv_dy.shape[2]))
                min_w = min(int(du_dx.shape[3]), int(dv_dy.shape[3]))
                if min_h > 0 and min_w > 0:
                    div = du_dx[:, :, :min_h, :min_w] + dv_dy[:, :, :min_h, :min_w]
                    diagnostics[key("div_mean")] = div.abs().mean()
                    diagnostics[key("div_max")] = div.abs().max()

                dv_dx = v[:, :, :, 1:] - v[:, :, :, :-1]
                du_dy = u[:, :, 1:, :] - u[:, :, :-1, :]
                min_h = min(int(dv_dx.shape[2]), int(du_dy.shape[2]))
                min_w = min(int(dv_dx.shape[3]), int(du_dy.shape[3]))
                if min_h > 0 and min_w > 0:
                    curl = dv_dx[:, :, :min_h, :min_w] - du_dy[:, :, :min_h, :min_w]
                    diagnostics[key("curl_mean")] = curl.abs().mean()
                    diagnostics[key("curl_max")] = curl.abs().max()

        return diagnostics

    @classmethod
    def _collect_readout_logit_diagnostics(cls, mcm: nn.Module) -> Dict[str, torch.Tensor]:
        logits = getattr(mcm, "_last_flow_logits", None)
        if not isinstance(logits, torch.Tensor):
            estimator = getattr(mcm, "estimator", None)
            logits = getattr(estimator, "_last_flow_logits", None) if estimator is not None else None
        if not isinstance(logits, torch.Tensor) or logits.numel() == 0:
            return {}
        return cls._collect_flow_diagnostics(logits, prefix="logits")

    @classmethod
    def _collect_readout_aux_diagnostics(cls, mcm: nn.Module) -> Dict[str, torch.Tensor]:
        estimator = getattr(mcm, "estimator", None)
        aux = getattr(estimator, "_last_step_readout_aux", None) if estimator is not None else None
        if not isinstance(aux, dict):
            return {}

        diagnostics: Dict[str, torch.Tensor] = {}
        flow_low = aux.get("flow_low")
        if isinstance(flow_low, torch.Tensor) and flow_low.numel() > 0:
            diagnostics.update(cls._collect_flow_diagnostics(flow_low, prefix="flow_low"))
        flow_full = aux.get("flow_full")
        if isinstance(flow_full, torch.Tensor) and flow_full.numel() > 0:
            diagnostics.update(cls._collect_flow_diagnostics(flow_full, prefix="flow_full"))

        for key in ("flow_upsample_scale_x", "flow_upsample_scale_y"):
            value = aux.get(key)
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                diagnostics[key] = value.detach().float()
            elif isinstance(value, (int, float)):
                diagnostics[key] = torch.tensor(float(value), dtype=torch.float32)
        return diagnostics

    @staticmethod
    def _collect_flow_sign_diagnostics(flow: torch.Tensor, *, prefix: str) -> Dict[str, torch.Tensor]:
        diagnostics: Dict[str, torch.Tensor] = {}
        if not isinstance(flow, torch.Tensor) or flow.numel() == 0 or flow.ndim < 4 or int(flow.shape[1]) < 2:
            return diagnostics

        with torch.no_grad():
            flow_stats = flow.float()
            u = flow_stats[:, 0]
            v = flow_stats[:, 1]
            diagnostics[f"{prefix}_u_pos_frac"] = (u > 0).to(torch.float32).mean()
            diagnostics[f"{prefix}_u_neg_frac"] = (u < 0).to(torch.float32).mean()
            diagnostics[f"{prefix}_v_pos_frac"] = (v > 0).to(torch.float32).mean()
            diagnostics[f"{prefix}_v_neg_frac"] = (v < 0).to(torch.float32).mean()
        return diagnostics

    @staticmethod
    def _collect_tree_l2_norm(value: Any) -> Optional[torch.Tensor]:
        sq_sum: Optional[torch.Tensor] = None
        elem_count: int = 0

        def _visit(obj: Any) -> None:
            nonlocal sq_sum, elem_count
            if torch.is_tensor(obj):
                if obj.numel() == 0:
                    return
                t = obj.detach().float()
                t_sq_sum = t.square().sum()
                sq_sum = t_sq_sum if sq_sum is None else (sq_sum + t_sq_sum)
                elem_count += int(t.numel())
                return
            if isinstance(obj, dict):
                for item in obj.values():
                    _visit(item)
                return
            if isinstance(obj, (list, tuple)):
                for item in obj:
                    _visit(item)
                return

        _visit(value)
        if sq_sum is None or elem_count <= 0:
            return None
        return torch.sqrt(sq_sum / float(elem_count))

    @staticmethod
    def _resolve_nested_module(root: Optional[nn.Module], path: List[str]) -> Optional[nn.Module]:
        module = root
        for part in path:
            if module is None:
                return None
            module = getattr(module, part, None)
        return module

    @classmethod
    def _collect_layer_state_diagnostics(
        cls,
        hidden: Any,
        *,
        core_module: Optional[nn.Module],
        path: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        diagnostics: Dict[str, torch.Tensor] = {}
        path = list(path or [])

        if not isinstance(hidden, dict):
            return diagnostics

        if "v" in hidden or "spike" in hidden:
            layer_name = "_".join(path)
            membrane = hidden.get("v")
            spike = hidden.get("spike")

            if isinstance(spike, torch.Tensor) and spike.numel() > 0:
                diagnostics[f"{layer_name}_spike_rate"] = spike.detach().float().mean()

            if isinstance(membrane, torch.Tensor) and membrane.numel() > 0:
                membrane_f = membrane.detach().float()
                diagnostics[f"{layer_name}_membrane_mean"] = membrane_f.mean()
                diagnostics[f"{layer_name}_membrane_std"] = membrane_f.std()

                module = cls._resolve_nested_module(core_module, path)
                threshold = getattr(module, "thresh", None) if module is not None else None
                if isinstance(threshold, torch.Tensor) and threshold.numel() > 0:
                    diagnostics[f"{layer_name}_membrane_threshold_gap_mean"] = (
                        threshold.detach().float() - membrane_f
                    ).mean()
            return diagnostics

        for key, value in hidden.items():
            diagnostics.update(
                cls._collect_layer_state_diagnostics(
                    value,
                    core_module=core_module,
                    path=path + [str(key)],
                )
            )
        return diagnostics

    @classmethod
    def _collect_streaming_state_diagnostics(cls, mcm: nn.Module) -> Dict[str, torch.Tensor]:
        estimator = getattr(mcm, "estimator", None)
        if estimator is None:
            return {}

        diagnostics: Dict[str, torch.Tensor] = {}
        state = getattr(estimator, "_state", None)
        if state is not None:
            core_norm = cls._collect_tree_l2_norm(getattr(state, "hidden", None))
            if core_norm is not None:
                diagnostics["core_hidden_norm"] = core_norm

            core_module = getattr(getattr(estimator, "sofe_core", None), "core", None)
            diagnostics.update(
                cls._collect_layer_state_diagnostics(
                    getattr(state, "hidden", None),
                    core_module=core_module,
                )
            )

        return diagnostics

    @staticmethod
    def _collect_streaming_debug_metrics(mcm: nn.Module) -> Dict[str, float]:
        debug = getattr(mcm, "_last_stream_debug", None)
        if not isinstance(debug, dict):
            return {}

        metrics: Dict[str, float] = {}
        for key in (
            "num_steps",
            "num_readouts",
            "tbptt_steps",
            "num_tbptt_chunks",
            "step_time_us",
            "streaming_step_policy_id",
            "chunk_event_count_mean",
            "chunk_event_count_min",
            "chunk_event_count_max",
            "chunk_loss_weight_mean",
            "chunk_loss_weight_min",
            "chunk_loss_weight_max",
            "cuda_memory_allocated_before_chunks_mb",
            "cuda_memory_allocated_after_chunks_mb",
            "cuda_peak_memory_allocated_before_chunks_mb",
            "cuda_peak_memory_allocated_after_chunks_mb",
            "head_spike_rate_final",
            "head_spike_rate_active_mean",
            "head_spike_rate_active_max",
            "head_spike_rate_readout_mean",
            "head_spike_rate_readout_last",
            "head_v_minus_thresh_active_max",
            "head_v_minus_thresh_active_p99",
        ):
            value = debug.get(key)
            if isinstance(value, (int, float)):
                metrics[key] = float(value)
        return metrics

    @staticmethod
    def _format_metric_value(value: float) -> str:
        value_f = float(value)
        abs_v = abs(value_f)
        if abs_v == 0.0:
            return "0"
        if abs_v < 1e-3 or abs_v >= 1e3:
            return f"{value_f:.3e}"
        return f"{value_f:.4f}"

    def _run_streaming_chunk_backward_step(
        self,
        *,
        primary_input: Optional[torch.Tensor],
        forward_kwargs: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], bool]:
        """Backward each streaming TBPTT chunk, then step once per outer batch."""

        mcm = self._unwrap_mcm()
        plan = mcm.prepare_streaming_tbptt_plan(voxel=primary_input, **forward_kwargs)
        num_chunks = int(plan.get("num_chunks", 0) or 0)
        min_chunks, max_chunks = self._sync_streaming_chunk_count(num_chunks)
        if min_chunks <= 0:
            raise RuntimeError(
                "streaming_step_policy='chunk_backward' requires at least one "
                f"TBPTT chunk on every rank; local chunks={num_chunks}, "
                f"global min={min_chunks}, global max={max_chunks}"
            )
        if max_chunks != min_chunks:
            raise RuntimeError(
                "streaming_step_policy='chunk_backward' requires equal TBPTT "
                "chunk counts across DDP ranks until padded chunk sync is "
                f"implemented; got min={min_chunks}, max={max_chunks} at "
                f"iter {self.iteration}"
            )

        self.optimizer.zero_grad(set_to_none=True)
        optimizer_stepped = False
        local_skip_step = False
        sync_backward_done = False
        loss_for_logging: Optional[torch.Tensor] = None
        loss_dict: Dict[str, Any] = {}
        prev_scale: Optional[float] = None

        if self.use_amp:
            assert self.scaler is not None
            prev_scale = float(self.scaler.get_scale())

        if self.device.type == "cuda":
            plan["cuda_memory_allocated_before_chunks_mb"] = float(torch.cuda.memory_allocated(self.device)) / (1024.0 * 1024.0)
            plan["cuda_peak_memory_allocated_before_chunks_mb"] = float(torch.cuda.max_memory_allocated(self.device)) / (1024.0 * 1024.0)

        for chunk_idx in range(num_chunks):
            is_final_local_chunk = chunk_idx == num_chunks - 1
            sync_this_backward = is_final_local_chunk
            with self._ddp_no_sync_context(enabled=not sync_this_backward):
                if self.use_amp:
                    with autocast("cuda"):
                        chunk_out = self.mcm(
                            None,
                            streaming_tbptt_plan=plan,
                            streaming_tbptt_chunk_index=chunk_idx,
                        )
                        raw_loss = chunk_out["loss"]
                        loss_weight = float(chunk_out.get("loss_weight", 1.0 / float(num_chunks)))
                        loss = raw_loss * loss_weight
                else:
                    chunk_out = self.mcm(
                        None,
                        streaming_tbptt_plan=plan,
                        streaming_tbptt_chunk_index=chunk_idx,
                    )
                    raw_loss = chunk_out["loss"]
                    loss_weight = float(chunk_out.get("loss_weight", 1.0 / float(num_chunks)))
                    loss = raw_loss * loss_weight

                raw_loss_finite = bool(torch.isfinite(raw_loss.detach()).all())
                scaled_loss_finite = bool(torch.isfinite(loss.detach()).all())
                if not raw_loss_finite or not scaled_loss_finite:
                    local_skip_step = True
                    loss_for_logging = raw_loss.detach()
                    loss_dict = self._build_loss_dict(loss_for_logging)
                    self._log_nonfinite_diagnostics_once(
                        reason="non-finite loss (chunk-backward path)",
                        loss=raw_loss,
                        loss_dict=loss_dict,
                        check_grads=False,
                    )
                    break

                if self.use_amp:
                    assert self.scaler is not None
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

            if sync_this_backward:
                sync_backward_done = True
            mcm.detach_streaming_tbptt_plan_state(plan)

        if self.device.type == "cuda":
            plan["cuda_memory_allocated_after_chunks_mb"] = float(torch.cuda.memory_allocated(self.device)) / (1024.0 * 1024.0)
            plan["cuda_peak_memory_allocated_after_chunks_mb"] = float(torch.cuda.max_memory_allocated(self.device)) / (1024.0 * 1024.0)

        if local_skip_step and not sync_backward_done and hasattr(self.mcm, "no_sync"):
            # A rank that aborts before its final real chunk still needs one
            # synchronized backward so other DDP ranks do not block at the
            # outer-batch all-reduce. The optimizer step is skipped below.
            dummy_loss = self._zero_loss_for_ddp_sync()
            if self.use_amp:
                assert self.scaler is not None
                self.scaler.scale(dummy_loss).backward()
            else:
                dummy_loss.backward()
            sync_backward_done = True

        local_skip_step = self._sync_skip_optimizer_step(local_skip_step)
        if local_skip_step:
            if prev_scale is not None:
                self._skip_optimizer_step_for_nonfinite_grads(prev_scale=prev_scale)
            else:
                self._skip_optimizer_step_for_nonfinite_grads()
            if loss_for_logging is None:
                loss_for_logging = torch.tensor(0.0, device=self.device)
            if not loss_dict:
                loss_dict = self._build_loss_dict(loss_for_logging)
            return loss_dict, False

        if self.use_amp:
            assert self.scaler is not None
            self.scaler.unscale_(self.optimizer)
            self.after_backward()
            assert prev_scale is not None
            grads_finite = self._clip_gradients_and_check_finite()
            grads_finite = not self._sync_skip_optimizer_step(not grads_finite)
            if grads_finite:
                self.scaler.step(self.optimizer)
                optimizer_stepped = True
                self.scaler.update()
                new_scale = float(self.scaler.get_scale())
                if new_scale < prev_scale:
                    logger.warning(
                        "AMP gradient overflow at iter %d: scaler reduced from %.1f to %.1f",
                        self.iteration,
                        prev_scale,
                        new_scale,
                    )
            else:
                weighted_loss = plan.get("weighted_loss_total")
                loss_for_logging = weighted_loss if isinstance(weighted_loss, torch.Tensor) else torch.tensor(0.0, device=self.device)
                loss_dict = self._build_loss_dict(loss_for_logging)
                self._log_nonfinite_diagnostics_once(
                    reason="non-finite gradients (chunk-backward AMP path)",
                    loss=loss_for_logging,
                    loss_dict=loss_dict,
                    check_grads=True,
                )
                self._skip_optimizer_step_for_nonfinite_grads(prev_scale=prev_scale)
                return loss_dict, False
        else:
            self.after_backward()
            grads_finite = self._clip_gradients_and_check_finite()
            grads_finite = not self._sync_skip_optimizer_step(not grads_finite)
            if grads_finite:
                self.optimizer.step()
                optimizer_stepped = True
            else:
                weighted_loss = plan.get("weighted_loss_total")
                loss_for_logging = weighted_loss if isinstance(weighted_loss, torch.Tensor) else torch.tensor(0.0, device=self.device)
                loss_dict = self._build_loss_dict(loss_for_logging)
                self._log_nonfinite_diagnostics_once(
                    reason="non-finite gradients (chunk-backward fp32 path)",
                    loss=loss_for_logging,
                    loss_dict=loss_dict,
                    check_grads=True,
                )
                self._skip_optimizer_step_for_nonfinite_grads()
                return loss_dict, False

        mcm.detach_streaming_tbptt_plan_state(plan)
        mcm.finalize_streaming_tbptt_plan(plan)
        weighted_loss = plan.get("weighted_loss_total")
        if isinstance(weighted_loss, torch.Tensor):
            loss_for_logging = weighted_loss
        else:
            loss_for_logging = torch.tensor(0.0, device=self.device)
        loss_dict = self._build_loss_dict(loss_for_logging)
        return loss_dict, optimizer_stepped

    def run_step(self) -> None:
        assert self.mcm.training, "MCM should be in training mode"

        data_start = time.perf_counter()
        try:
            batch = self._get_train_prefetcher().next()
        except StopIteration:
            self.reset_data_loader_iter()
            raise

        if self._last_step_end_time is not None:
            data_time = data_start - self._last_step_end_time
        else:
            data_time = time.perf_counter() - data_start

        primary_input, forward_kwargs, rep = self._build_mcm_call(batch)
        if rep == "events_voxel":
            # Voxel batches are independent event windows. Reset stateful
            # estimators here so recurrent/SNN state does not leak across
            # unrelated samples in standard MCM pretraining.
            self._unwrap_mcm().reset_state()

        if (
            rep == "streaming_events"
            and self._streaming_step_policy == "chunk_backward"
            and self._tbptt_steps > 0
        ):
            loss_dict, optimizer_stepped = self._run_streaming_chunk_backward_step(
                primary_input=primary_input,
                forward_kwargs=forward_kwargs,
            )
        elif self.use_amp:
            optimizer_stepped = False
            with autocast("cuda"):
                _corrected, loss = self.mcm(primary_input, **forward_kwargs)
            loss_dict = self._build_loss_dict(loss)

            # Always run the standard GradScaler pipeline so that inf
            # checks are properly recorded before scaler.update().  When
            # the loss (or resulting gradients) is non-finite the scaler
            # will detect it during unscale_, skip the optimizer step,
            # and reduce its scale factor — all without crashing.
            self.optimizer.zero_grad(set_to_none=True)
            assert self.scaler is not None
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            self.after_backward()
            prev_scale = float(self.scaler.get_scale())
            grads_finite = self._clip_gradients_and_check_finite()

            if grads_finite:
                self.scaler.step(self.optimizer)
                optimizer_stepped = True
                self.scaler.update()
                new_scale = float(self.scaler.get_scale())

                if new_scale < prev_scale:
                    logger.warning(
                        "AMP gradient overflow at iter %d: scaler reduced from %.1f to %.1f",
                        self.iteration,
                        prev_scale,
                        new_scale,
                    )
            else:
                self._log_nonfinite_diagnostics_once(
                    reason="non-finite gradients (AMP path)",
                    loss=loss,
                    loss_dict=loss_dict,
                    check_grads=True,
                )
                self._skip_optimizer_step_for_nonfinite_grads(prev_scale=prev_scale)

            if not torch.isfinite(loss):
                logger.warning("Non-finite loss at iter %d, step skipped by GradScaler", self.iteration)
                self._log_nonfinite_diagnostics_once(
                    reason="non-finite loss (AMP path)",
                    loss=loss,
                    loss_dict=loss_dict,
                    check_grads=False,
                )
                self._unwrap_mcm().reset_state()
        else:
            optimizer_stepped = False
            _corrected, loss = self.mcm(primary_input, **forward_kwargs)
            loss_dict = self._build_loss_dict(loss)

            if not torch.isfinite(loss):
                logger.warning("Non-finite loss at iter %d, skipping step", self.iteration)
                self._log_nonfinite_diagnostics_once(
                    reason="non-finite loss (fp32 path)",
                    loss=loss,
                    loss_dict=loss_dict,
                    check_grads=False,
                )
                self.optimizer.zero_grad(set_to_none=True)
                # Reset OFE streaming state to prevent NaN propagation.
                self._unwrap_mcm().reset_state()
            else:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.after_backward()
                if self._clip_gradients_and_check_finite():
                    self.optimizer.step()
                    optimizer_stepped = True
                else:
                    self._log_nonfinite_diagnostics_once(
                        reason="non-finite gradients (fp32 path)",
                        loss=loss,
                        loss_dict=loss_dict,
                        check_grads=True,
                    )
                    self._skip_optimizer_step_for_nonfinite_grads()

        # Build metrics dict
        mcm = self._unwrap_mcm()

        # Match the reference FireNet SNN training cadence: gradients flow
        # through the configured intra-window TBPTT span, then the carried
        # streaming state is preserved numerically but detached immediately
        # after each optimizer update.
        if optimizer_stepped and rep == "streaming_events":
            mcm.detach_state()

        flow = getattr(mcm, "_last_flow", None)
        if isinstance(flow, torch.Tensor) and flow.numel() > 0:
            loss_dict.update(self._collect_flow_diagnostics(flow))
            loss_dict.update(self._collect_flow_sign_diagnostics(flow, prefix="flow"))
            loss_dict.update(self._collect_flow_diagnostics(flow, prefix="flow_scaled"))
        estimator = getattr(mcm, "estimator", None)
        flow_unscaled = getattr(estimator, "_last_flow", None) if estimator is not None else None
        if isinstance(flow_unscaled, torch.Tensor) and flow_unscaled.numel() > 0:
            loss_dict.update(self._collect_flow_diagnostics(flow_unscaled, prefix="flow_unscaled"))
        loss_dict["flow_scaling"] = torch.tensor(float(getattr(mcm, "flow_scaling", 1.0)), dtype=torch.float32)
        loss_dict.update(self._collect_readout_logit_diagnostics(mcm))
        loss_dict.update(self._collect_readout_aux_diagnostics(mcm))
        loss_dict.update(self._collect_streaming_state_diagnostics(mcm))

        loss_dict.update(self._collect_streaming_debug_metrics(mcm))

        self._write_metrics(loss_dict, data_time)
        self._last_step_end_time = time.perf_counter()

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        logger.info(
            "Starting MCM pre-training for %d epochs", self.config.training.num_epochs
        )

        base_iters_per_epoch = len(self.train_loader)
        iters_per_epoch = int(base_iters_per_epoch)
        if iters_per_epoch <= 0:
            raise ValueError("Train loader has no iterations per epoch")

        max_steps_per_epoch = int(getattr(self.config.training, "max_steps_per_epoch", 0) or 0)
        if max_steps_per_epoch > 0:
            iters_per_epoch = min(int(iters_per_epoch), int(max_steps_per_epoch))

        max_total_steps = int(getattr(self.config.training, "max_total_steps", 0) or 0)

        if is_main_process():
            per_gpu_batch = int(getattr(self.cfg.training, "batch_size", 1))
            global_batch = per_gpu_batch * max(int(get_world_size()), 1)
            logger.info(
                "MCM train epoch: %d iters/epoch, batch=%d per GPU (global=%d)",
                iters_per_epoch, per_gpu_batch, global_batch,
            )
            if max_steps_per_epoch > 0 or max_total_steps > 0:
                logger.info(
                    "MCM smoke/profiling limits active: base_iters=%d, max_steps_per_epoch=%s, max_total_steps=%s",
                    base_iters_per_epoch,
                    str(max_steps_per_epoch) if max_steps_per_epoch > 0 else "off",
                    str(max_total_steps) if max_total_steps > 0 else "off",
                )

        with MetricStorage(self.iteration) as self.storage:
            self.before_train()
            try:
                stop_training = False
                for epoch in range(int(self.epoch), int(self.config.training.num_epochs)):
                    if stop_training:
                        break
                    self.epoch = int(epoch)
                    self.mcm.train()
                    self._prepare_train_epoch_data_stream(epoch)
                    self._last_step_end_time = None

                    losses: Dict[str, float] = defaultdict(float)
                    processed = 0

                    self.before_epoch()
                    epoch_start = time.perf_counter()

                    for batch_idx in range(int(iters_per_epoch)):
                        if max_total_steps > 0 and int(self.iteration) >= int(max_total_steps):
                            stop_training = True
                            break
                        self.storage.iter = int(self.iteration)

                        if int(self.epoch) < int(self.config.training.warmup_epochs):
                            self._warmup_lr(batch_idx, int(iters_per_epoch))

                        self.before_step()
                        self.run_step()
                        self.after_step()
                        processed += 1

                        latest = dict(self.storage.latest())
                        for key in ["total", "mcm_contrast", "mcm_smooth", "mcm_magnitude", "mcm_epe"]:
                            if key in latest:
                                losses[key] += float(latest[key])

                        self.iteration += 1

                    epoch_time = time.perf_counter() - epoch_start
                    self.after_epoch()

                    if self._should_run_validation_this_epoch():
                        val_metrics = self.validate()
                        if val_metrics:
                            self._update_best_checkpoint_from_metrics(val_metrics)

                    denom = max(processed, 1)
                    avg = {k: (v / denom) for k, v in losses.items()}
                    self._log_epoch(avg, epoch_time)

            except Exception:
                logger.exception("Exception during MCM pre-training:")
                raise
            finally:
                self.after_train()
                if self.writer is not None:
                    try:
                        self.writer.flush()
                        self.writer.close()
                    except Exception:
                        pass

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def build_hooks(self) -> List[HookBase]:
        hooks: List[HookBase] = []
        hooks.append(DataTimeTracker())

        if self.scheduler is not None:
            hooks.append(LRSchedulerHook())

        # Periodic checkpointing
        save_interval = int(
            getattr(getattr(self.config, "eval", None), "save_interval", 0) or 0
        )
        save_epochs = list(getattr(getattr(self.config, "eval", None), "save_epochs", []) or [])
        if (save_interval > 0 or save_epochs) and is_main_process():
            max_to_keep = (
                999
                if bool(getattr(getattr(self.config, "log", None), "save_all_checkpoints", False))
                else 5
            )
            if save_epochs:
                max_to_keep = max(max_to_keep, len({int(epoch) for epoch in save_epochs if int(epoch) > 0}))
            hooks.append(
                PeriodicCheckpointer(
                    save_interval,
                    self.output_dir,
                    max_to_keep=max_to_keep,
                    save_epochs=save_epochs,
                )
            )

        if bool(getattr(self.config.training, "log_grad_norm", False)):
            interval = int(getattr(self.config.training, "log_grad_norm_interval", 50))
            hooks.append(GradientMonitorHook(interval=interval))

        # Flow-magnitude sanity monitor. Logs |flow| stats every N iters and
        # can auto-abort if flow stays near zero (the failure mode the
        # flow-vs-GT diagnostic surfaced on mcm_cmax_full). Opt-in via config.
        flow_mon_interval = int(getattr(self.config.training, "flow_monitor_interval", 0))
        if flow_mon_interval > 0:
            from smcfnet.core.hooks import FlowMagnitudeMonitorHook
            hooks.append(FlowMagnitudeMonitorHook(
                interval=flow_mon_interval,
                abort_after_iter=int(getattr(self.config.training, "flow_monitor_abort_after_iter", 0)),
                abort_threshold=float(getattr(self.config.training, "flow_monitor_abort_threshold", 1e-3)),
            ))

        wandb_cfg = getattr(self.config, "wandb", None)
        if wandb_cfg is not None and bool(getattr(wandb_cfg, "use_wandb", False)):
            hooks.append(WandbHook(self.config, self.output_dir))

        hooks.append(EpochCleanupHook())

        return hooks

    # ------------------------------------------------------------------
    # Metrics / Logging
    # ------------------------------------------------------------------

    def _write_metrics(self, loss_dict: Dict[str, Any], data_time: float) -> None:
        if self.storage is None:
            return
        metrics: Dict[str, float] = {}
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor):
                metrics[k] = float(v.detach().item())
            elif isinstance(v, (int, float)):
                metrics[k] = float(v)
        metrics["data_time"] = float(data_time)
        metrics["lr"] = float(self.optimizer.param_groups[0]["lr"])

        should_log_mem = (
            self.device.type == "cuda"
            and torch.cuda.is_available()
            and int(self.iteration) % int(self._mem_log_interval_steps) == 0
        )
        if should_log_mem:
            mem = _cuda_memory_snapshot(self.device)
            metrics["gpu_mem_allocated_mb"] = mem["allocated_mb"]
            metrics["gpu_mem_reserved_mb"] = mem["reserved_mb"]
            metrics["gpu_mem_peak_allocated_mb"] = mem["max_allocated_mb"]
            metrics["gpu_mem_peak_reserved_mb"] = mem["max_reserved_mb"]
            metrics["gpu_mem_free_mb"] = mem["free_mb"]
            metrics["gpu_mem_total_mb"] = mem["total_mb"]

        for k, v in metrics.items():
            self.storage.put_scalar(k, v)

        log_interval = int(getattr(self.config.log, "log_interval", 50))
        should_emit_log = bool(
            is_main_process()
            and (
                int(self.iteration) % log_interval == 0
                or should_log_mem
            )
        )
        if should_emit_log:
            parts = [f"[MCM] iter={self.iteration}"]
            for k in [
                "total",
                "mcm_contrast",
                "mcm_smooth",
                "mcm_smooth_raw",
                "mcm_center",
                "mcm_centered_ratio",
                "mcm_epe",
                "num_steps",
                "num_readouts",
                "tbptt_steps",
                "num_tbptt_chunks",
                "streaming_step_policy_id",
                "chunk_event_count_mean",
                "chunk_event_count_min",
                "chunk_event_count_max",
                "chunk_loss_weight_mean",
                "chunk_loss_weight_min",
                "chunk_loss_weight_max",
                "flow_mag_mean",
                "flow_unscaled_mag_mean",
                "flow_scaled_mag_mean",
                "flow_low_mag_mean",
                "flow_full_mag_mean",
                "flow_scaling",
                "flow_upsample_scale_x",
                "flow_upsample_scale_y",
                "flow_centered_mag_mean",
                "flow_global_flow_ratio",
                "flow_mean_flow_ratio",
                "flow_u_spatial_std_mean",
                "flow_v_spatial_std_mean",
                "flow_u_mean",
                "flow_v_mean",
                "core_hidden_norm",
                "gpu_mem_allocated_mb",
                "gpu_mem_reserved_mb",
                "gpu_mem_peak_allocated_mb",
                "gpu_mem_peak_reserved_mb",
                "gpu_mem_free_mb",
                "cuda_memory_allocated_before_chunks_mb",
                "cuda_memory_allocated_after_chunks_mb",
                "cuda_peak_memory_allocated_before_chunks_mb",
                "cuda_peak_memory_allocated_after_chunks_mb",
                "lr",
                "head_spike_rate_final",
                "head_spike_rate_active_mean",
                "head_spike_rate_active_max",
                "head_spike_rate_readout_mean",
                "head_spike_rate_readout_last",
                "head_v_minus_thresh_active_max",
                "head_v_minus_thresh_active_p99",
                "head_spike_rate",
                "head_membrane_mean",
                "head_membrane_std",
                "head_membrane_threshold_gap_mean",
                "G1_spike_rate",
                "G1_membrane_mean",
                "G1_membrane_std",
                "G1_membrane_threshold_gap_mean",
                "R1a_spike_rate",
                "R1a_membrane_mean",
                "R1a_membrane_std",
                "R1a_membrane_threshold_gap_mean",
                "R1b_spike_rate",
                "R1b_membrane_mean",
                "R1b_membrane_std",
                "R1b_membrane_threshold_gap_mean",
                "G2_spike_rate",
                "G2_membrane_mean",
                "G2_membrane_std",
                "G2_membrane_threshold_gap_mean",
                "R2a_spike_rate",
                "R2a_membrane_mean",
                "R2a_membrane_std",
                "R2a_membrane_threshold_gap_mean",
                "R2b_spike_rate",
                "R2b_membrane_mean",
                "R2b_membrane_std",
                "R2b_membrane_threshold_gap_mean",
            ]:
                if k in metrics:
                    parts.append(f"{k}={self._format_metric_value(metrics[k])}")
            logger.info("  ".join(parts))

            if self.writer is not None:
                for k, v in metrics.items():
                    self.writer.add_scalar(f"mcm_train/{k}", v, self.iteration)

    def _log_epoch(self, avg_losses: Dict[str, float], epoch_time: float) -> None:
        if not is_main_process():
            return
        parts = [f"[MCM] Epoch {self.epoch} done in {epoch_time:.1f}s"]
        for k, v in avg_losses.items():
            parts.append(f"{k}={v:.4f}")
        logger.info("  ".join(parts))

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    @property
    def iteration(self) -> int:
        return self.iter

    @iteration.setter
    def iteration(self, val: int) -> None:
        self.iter = int(val)

    def state_dict(self) -> Dict:
        sd = {
            "mcm": self.mcm.state_dict(),
            "optimizer": self.optimizer.state_dict() if self.optimizer else None,
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "epoch": int(self.epoch),
            "iteration": int(self.iteration),
            "best_metric": float(self.best_metric),
            "best_metric_initialized": bool(self._best_metric_initialized),
            "best_metric_name": self._best_metric_name,
            "best_metric_higher_is_better": bool(self._best_metric_higher_is_better),
        }
        if self.scaler is not None:
            sd["scaler"] = self.scaler.state_dict()
        return sd

    def load_state_dict(self, state_dict: Dict) -> None:
        if "mcm" in state_dict:
            self.mcm.load_state_dict(state_dict["mcm"])
        self.epoch = int(state_dict.get("epoch", 0))
        self.iteration = int(state_dict.get("iteration", 0))
        self.best_metric = float(state_dict.get("best_metric", 0.0))
        self._best_metric_initialized = bool(state_dict.get("best_metric_initialized", False))
        self._best_metric_name = state_dict.get("best_metric_name", None)
        self._best_metric_higher_is_better = bool(state_dict.get("best_metric_higher_is_better", True))
        if self.optimizer is not None and state_dict.get("optimizer") is not None:
            self.optimizer.load_state_dict(state_dict["optimizer"])
        if self.scheduler is not None and state_dict.get("scheduler") is not None:
            self.scheduler.load_state_dict(state_dict["scheduler"])
        if self.scaler is not None and state_dict.get("scaler") is not None:
            self.scaler.load_state_dict(state_dict["scaler"])

    def save_checkpoint(self, filename: str) -> None:
        """Save an MCM-only checkpoint (compatible with weight_utils loaders)."""
        extra = {}
        if self.scaler is not None:
            extra["scaler"] = self.scaler.state_dict()
        save_mcm_checkpoint(
            self.mcm,
            path=Path(self.output_dir) / str(filename),
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.epoch,
            iteration=self.iteration,
            best_metric=self.best_metric,
            extra=extra or None,
        )

    def resume_checkpoint(self, checkpoint_path: str) -> None:
        """Resume MCM pre-training from a checkpoint."""
        logger.info("Resuming MCM training from %s", checkpoint_path)
        from smcfnet.model.mcm.weight_utils import load_mcm_checkpoint
        ckpt = load_mcm_checkpoint(checkpoint_path, map_location=str(self.device))

        weights_only = bool(getattr(self.config.log, "load_weights_only", False))
        if weights_only:
            self.mcm.load_state_dict(ckpt["mcm"])
            self.epoch = 0
            self.iteration = 0
            self.best_metric = 0.0
            logger.info("Loaded MCM weights only; starting fresh (epoch=0)")
        else:
            self.load_state_dict(ckpt)
            logger.info(
                "Resumed MCM training from epoch=%d, iter=%d",
                self.epoch, self.iteration,
            )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _wrap_ddp_if_needed(self, model: nn.Module) -> nn.Module:
        if int(get_world_size()) <= 1:
            return model

        if not dist.is_available():
            raise RuntimeError("torch.distributed is not available, cannot use DDP")

        if not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend, init_method="env://")

        ddp_kwargs = {"broadcast_buffers": False}
        if self.device.type == "cuda":
            ddp_kwargs["device_ids"] = [int(get_local_rank())]
        return DistributedDataParallel(model, **ddp_kwargs)

    def _get_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda", get_local_rank())
        return torch.device("cpu")
