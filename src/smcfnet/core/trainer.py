import gc
import hashlib
import os
import psutil
import datetime
import random
from contextlib import nullcontext

import logging
import pickle
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import OrderedDict, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from smcfnet.model import (
  build_model,
  build_dsec_prefetcher,
  build_optimizer,
  build_train_loader_cfg,
  build_test_loader_cfg,
  build_lr_scheduler,
)
from smcfnet.config.config import (
  Config,
  DataConfig,
  MCMConfig,
  EDUMConfig,
  CMMConfig,
  DETRHeadConfig,
  ModelConfig,
  TrainingConfig,
  EvalConfig,
  VizConfig,
  LogConfig,
  WandbConfig,
  config_to_dict,
)
from smcfnet.utils.metric_storage import MetricStorage
from smcfnet.core.base.hookbase import HookBase
from smcfnet.core.hooks import (
  DataTimeTracker,
  EpochCleanupHook,
  FiniteModelStateHook,
  LRSchedulerHook,
  EvaluationHook,
  PeriodicCheckpointer,
  WandbHook,
  GradientMonitorHook,
)
from smcfnet.core.launch import is_main_process, synchronize, get_local_rank, get_world_size
from smcfnet.utils.validation import (
  postprocess_predictions,
  postprocess_detr_predictions,
  StreamingMAPAccumulator,
  decode_sttp_predictions,
  TrajectoryMetricsAccumulator,
)
from smcfnet.dataset.dsec.prefetcher import BaseDSECPrefetcher, DSECGPUPrefetcher
from smcfnet.dataset.samplers import InfiniteSampler
from smcfnet.utils.event_utils import events_to_voxel_grid
from smcfnet.utils.viz_utils import visualize_detections
from smcfnet.model.pretrained import load_yolox_pretrained_dualstream
from smcfnet.utils.profiling import CudaStageTimer, collect_cuda_stage_times_ms

logger = logging.getLogger(__name__)

from smcfnet.core.base import TrainerBase


def _sha256_file(path: str | Path) -> str:
  h = hashlib.sha256()
  with Path(path).open("rb") as fh:
    for chunk in iter(lambda: fh.read(1024 * 1024), b""):
      h.update(chunk)
  return h.hexdigest()


def _checkpoint_safe_globals():
  """Allowlist config dataclasses for torch weights_only loading.

  Older trainer checkpoints stored the live Config dataclass object directly.
  PyTorch 2.4+ rejects those globals under weights_only=True unless allowlisted.
  """
  return [
    Config,
    DataConfig,
    MCMConfig,
    EDUMConfig,
    CMMConfig,
    DETRHeadConfig,
    ModelConfig,
    TrainingConfig,
    EvalConfig,
    VizConfig,
    LogConfig,
    WandbConfig,
  ]


def _torch_safe_globals_context():
  safe_globals = getattr(torch.serialization, "safe_globals", None)
  if callable(safe_globals):
    return safe_globals(_checkpoint_safe_globals())
  return nullcontext()


def log_memory_stats(tag: str = "", logger=logger, every_n_steps: int = 1, step: Optional[int] = None):
  """Log GPU and CPU memory usage."""
  every_n_steps = int(every_n_steps)
  if every_n_steps <= 0:
    return
  if step is not None and (int(step) % every_n_steps != 0):
    return
  if torch.cuda.is_available():
    mem_alloc = torch.cuda.memory_allocated() / 1024 ** 2
    mem_reserved = torch.cuda.memory_reserved() / 1024 ** 2
    max_mem_alloc = torch.cuda.max_memory_allocated() / 1024 ** 2
    max_mem_reserved = torch.cuda.max_memory_reserved() / 1024 ** 2
    logger.info(f"[MEM][{tag}] CUDA mem_alloc={mem_alloc:.1f}MB mem_reserved={mem_reserved:.1f}MB max_alloc={max_mem_alloc:.1f}MB max_reserved={max_mem_reserved:.1f}MB (step={step})")
  process = psutil.Process()
  cpu_mem = process.memory_info().rss / 1024 ** 2
  logger.info(f"[MEM][{tag}] CPU rss={cpu_mem:.1f}MB (step={step})")
"""
Training engine for RGB-E object detection.
Incorporates best practices from detectron2: hooks, event storage, and robust metric tracking.
"""
class Trainer(TrainerBase):
  """
  Training engine with hooks system for extensibility
  """
  def __init__(
    self, 
    cfg: Config,
  ): 
    super().__init__()
    # Keep both names for compatibility with existing hooks/older code.
    self.cfg = cfg
    self.config = cfg

    self.device = self._get_device()
    if self.device.type == "cuda":
      torch.cuda.set_device(self.device)

    model = self.build_model(cfg).to(self.device)

    # Freeze everything except STTP head for isolated trajectory training.
    # Must happen BEFORE optimizer construction so frozen params are excluded
    # from param groups (get_default_optimizer_params skips requires_grad=False).
    self._freeze_detector = bool(getattr(cfg.model.sttp, "freeze_detector", False))
    if self._freeze_detector:
      frozen = 0
      for name, param in model.named_parameters():
        if "sttp_head" not in name:
          param.requires_grad = False
          frozen += 1
      # Set frozen modules to eval so BatchNorm uses pretrained running stats.
      # Use direct assignment instead of module.eval() which is recursive and
      # would also switch sttp_head to eval mode.
      for name, module in model.named_modules():
        if "sttp_head" not in name:
          module.training = False
      # Tell the model to run detector forward under torch.no_grad() to
      # avoid storing activations for frozen params (~50% VRAM savings).
      model.detector_frozen = True
      trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
      logger.info("freeze_detector: froze %d params, %d trainable values remain (sttp_head only)", frozen, trainable)

    optimizer = self.build_optimizer(cfg, model)
    # Build loader + keep access to DSEC per-sequence metadata.
    try:
      components = build_train_loader_cfg(cfg, return_components=True)
      self._dsec_per_sequence = components.get("per_sequence", None)
    except Exception:
      self._dsec_per_sequence = None

    # Optional precomputed flow provider (e.g. E-RAFT). Overrides MCM flow
    # at yolox.py:134 via the `flow` kwarg on model.forward.
    self._flow_provider = None
    self._flow_sequence_names: Optional[list] = None
    flow_dir = getattr(cfg.data, "precomputed_flow_dir", None)
    if flow_dir:
      if self._dsec_per_sequence is None:
        raise RuntimeError(
          "data.precomputed_flow_dir is set but DSEC per-sequence metadata is "
          "unavailable; flow lookup requires seq_idx -> sequence_name mapping."
        )
      from smcfnet.dataset.flow_provider import PrecomputedFlowProvider
      self._flow_provider = PrecomputedFlowProvider(flow_dir)
      self._flow_sequence_names = [
        getattr(ds, "sequence", None) for ds in self._dsec_per_sequence
      ]
      if any(n is None for n in self._flow_sequence_names):
        raise RuntimeError(
          "One or more per-sequence datasets lack a .sequence attribute; "
          "cannot build seq_idx -> name mapping for flow provider."
        )
      logger.info(
        "Precomputed flow provider initialised: dir=%s, sequences=%d",
        flow_dir, len(self._flow_sequence_names),
      )

    # Optional precomputed stem-feature provider for the recurrent STTP head.
    self._stem_provider = None
    self._stem_sequence_names = None
    stem_dir = getattr(cfg.data, "stem_feature_dir", None)
    if stem_dir:
      if self._dsec_per_sequence is None:
        raise RuntimeError(
          "data.stem_feature_dir is set but DSEC per-sequence metadata is "
          "unavailable; stem lookup requires seq_idx -> sequence_name mapping."
        )
      from smcfnet.dataset.stem_feature_provider import StemFeatureProvider
      self._stem_provider = StemFeatureProvider(
        stem_dir,
        context_window=int(cfg.model.sttp.context_window),
        num_scales=3,
      )
      self._stem_sequence_names = [
        getattr(ds, "sequence", None) for ds in self._dsec_per_sequence
      ]
      if any(n is None for n in self._stem_sequence_names):
        raise RuntimeError(
          "One or more per-sequence datasets lack a .sequence attribute; "
          "cannot build seq_idx -> name mapping for stem provider."
        )
      logger.info(
        "Stem-feature provider initialised: dir=%s, K_ctx=%d",
        stem_dir, int(cfg.model.sttp.context_window),
      )

    # Optional precomputed detector-facing SMCM IWE provider. This turns the
    # detector job back into a normal shuffled supervised run: SMCM state is
    # advanced offline in sequence order and each batch only looks up the cached
    # [3,H,W] IWE by (sequence, frame_index).
    self._iwe_provider = None
    self._iwe_sequence_names = None
    self._live_iwe_reference_provider = None
    self._live_iwe_reference_sequence_names = None
    iwe_dir = getattr(cfg.data, "precomputed_iwe_dir", None)
    if iwe_dir:
      if self._dsec_per_sequence is None:
        raise RuntimeError(
          "data.precomputed_iwe_dir is set but DSEC per-sequence metadata is "
          "unavailable; IWE lookup requires seq_idx -> sequence_name mapping."
        )
      mcm_ckpt = getattr(getattr(cfg.model, "mcm", None), "pretrained_weights", None)
      if not mcm_ckpt:
        raise RuntimeError(
          "data.precomputed_iwe_dir requires model.mcm.pretrained_weights so "
          "the cache can be bound to the intended frozen SMCM checkpoint."
        )
      ckpt_path = Path(str(mcm_ckpt))
      if not ckpt_path.exists():
        project_root = Path(__file__).resolve().parents[3]
        candidate = project_root / ckpt_path
        if candidate.exists():
          ckpt_path = candidate
      if not ckpt_path.exists():
        raise FileNotFoundError(f"SMCM checkpoint for precomputed IWE validation not found: {mcm_ckpt}")
      checkpoint_sha = _sha256_file(ckpt_path)
      self._iwe_sequence_names = [
        getattr(ds, "sequence", None) for ds in self._dsec_per_sequence
      ]
      if any(n is None for n in self._iwe_sequence_names):
        raise RuntimeError(
          "One or more per-sequence datasets lack a .sequence attribute; "
          "cannot build seq_idx -> name mapping for IWE provider."
        )
      iwe_h, iwe_w = self._event_voxel_resolution()
      frame_counts = {
        str(getattr(ds, "sequence")): int(len(ds))
        for ds in self._dsec_per_sequence
      }
      from smcfnet.dataset.iwe_provider import PrecomputedIWEProvider
      self._iwe_provider = PrecomputedIWEProvider(
        iwe_dir,
        dataset_key=str(getattr(cfg.data, "precomputed_iwe_key", "iwe") or "iwe"),
        expected_channels=int(getattr(cfg.model, "precomputed_event_channels", 3)),
        checkpoint_sha=checkpoint_sha,
        architecture=str(getattr(cfg.model.mcm, "architecture", "")),
        sequence_names=[str(n) for n in self._iwe_sequence_names],
        expected_frame_counts=frame_counts,
        expected_height=int(iwe_h),
        expected_width=int(iwe_w),
        exclude_sequences=getattr(cfg.data, "precomputed_iwe_exclude_sequences", None),
      )
      logger.info(
        "Precomputed IWE provider initialised: dir=%s, sequences=%d, checkpoint_sha=%s",
        iwe_dir, len(self._iwe_sequence_names), checkpoint_sha[:12],
      )
    live_iwe_reference_dir = getattr(cfg.data, "live_iwe_reference_dir", None)
    if live_iwe_reference_dir:
      if self._dsec_per_sequence is None:
        raise RuntimeError(
          "data.live_iwe_reference_dir is set but DSEC per-sequence metadata is "
          "unavailable; live/cache seam verification requires seq_idx -> name mapping."
        )
      mcm_ckpt = getattr(getattr(cfg.model, "mcm", None), "pretrained_weights", None)
      if not mcm_ckpt:
        raise RuntimeError(
          "data.live_iwe_reference_dir requires model.mcm.pretrained_weights so "
          "the reference cache can be bound to the intended live SMCM checkpoint."
        )
      ckpt_path = Path(str(mcm_ckpt))
      if not ckpt_path.exists():
        project_root = Path(__file__).resolve().parents[3]
        candidate = project_root / ckpt_path
        if candidate.exists():
          ckpt_path = candidate
      if not ckpt_path.exists():
        raise FileNotFoundError(f"SMCM checkpoint for live/cache IWE verification not found: {mcm_ckpt}")
      checkpoint_sha = _sha256_file(ckpt_path)
      self._live_iwe_reference_sequence_names = [
        getattr(ds, "sequence", None) for ds in self._dsec_per_sequence
      ]
      if any(n is None for n in self._live_iwe_reference_sequence_names):
        raise RuntimeError(
          "One or more per-sequence datasets lack a .sequence attribute; "
          "cannot build seq_idx -> name mapping for live/cache IWE verification."
        )
      iwe_h, iwe_w = self._event_voxel_resolution()
      frame_counts = {
        str(getattr(ds, "sequence")): int(len(ds))
        for ds in self._dsec_per_sequence
      }
      from smcfnet.dataset.iwe_provider import PrecomputedIWEProvider
      self._live_iwe_reference_provider = PrecomputedIWEProvider(
        live_iwe_reference_dir,
        dataset_key=str(getattr(cfg.data, "live_iwe_reference_key", "iwe") or "iwe"),
        expected_channels=int(getattr(cfg.model, "precomputed_event_channels", 3)),
        checkpoint_sha=checkpoint_sha,
        architecture=str(getattr(cfg.model.mcm, "architecture", "")),
        sequence_names=[str(n) for n in self._live_iwe_reference_sequence_names],
        expected_frame_counts=frame_counts,
        expected_height=int(iwe_h),
        expected_width=int(iwe_w),
        exclude_sequences=getattr(cfg.data, "precomputed_iwe_exclude_sequences", None),
      )
      logger.info(
        "Live/cache IWE seam reference initialised: dir=%s, sequences=%d, checkpoint_sha=%s",
        live_iwe_reference_dir, len(self._live_iwe_reference_sequence_names), checkpoint_sha[:12],
      )

    self.data_loader = self.build_train_loader(cfg)
    # Backward-compatible alias used by older code/tests.
    self.train_loader = self.data_loader
    # Optional eval loaders; attached by the entrypoint on rank 0.
    self.eval_loader: Optional[DataLoader] = None
    self.mon_eval_loader: Optional[DataLoader] = None
    # Backward-compatible aliases.
    self.val_loader: Optional[DataLoader] = None
    self.monitor_val_loader: Optional[DataLoader] = None

    # Infer steps_per_epoch for OneCycle if not provided.
    if str(getattr(cfg.training, "lr_scheduler", "")).lower() == "onecycle":
      if getattr(cfg.training, "steps_per_epoch", None) in (None, 0):
        try:
          cfg.training.steps_per_epoch = len(self.train_loader)
        except Exception:
          pass
    self.scheduler = self.build_lr_scheduler(cfg, optimizer)
    self.optimizer = optimizer

    # Wrap with DDP when running multi-process (torchrun / our launcher).
    self.model = self._wrap_ddp_if_needed(model)
    
    # Tracking
    self.epoch: int = 0
    self.best_metric: float = 0.0

    # Expose latest eval metrics for hooks (e.g. WandbHook).
    self._latest_eval_metrics: Optional[Dict[str, float]] = None
    self._latest_eval_step: Optional[int] = None
    self._latest_mon_eval_metrics: Optional[Dict[str, float]] = None
    self._latest_mon_eval_step: Optional[int] = None
    self._latest_local_train_metrics: Dict[str, float] = {}
    self._latest_reduced_train_metrics: Optional[Dict[str, float]] = None
    self._latest_reduced_train_metrics_step: Optional[int] = None

    # Mixed precision
    self.use_amp = bool(cfg.training.use_amp and torch.cuda.is_available())
    self.scaler = GradScaler("cuda") if self.use_amp else None

    # Lazy data loader iterator (detectron2 pattern)
    self._train_loader_iter = None

    # GPU prefetcher (created per-epoch; wraps the loader iterator)
    self._train_prefetcher: Optional[BaseDSECPrefetcher] = None

    # Metric storage (our train() opens a context-managed one).
    self.storage = MetricStorage(0)

    # Logging output
    self.output_dir = Path(self.cfg.log.output_dir) / str(self.cfg.log.experiment_name)
    if is_main_process():
      self.output_dir.mkdir(parents=True, exist_ok=True)
      self.writer = SummaryWriter(log_dir=str(self.output_dir / "tensorboard"))
    else:
      self.writer = None

    # Dataloader timing
    self._last_step_end_time: Optional[float] = None

    # Memory logging cadence (reduce log volume during long runs).
    self._mem_log_interval_steps = int(getattr(self.cfg.log, "memory_log_interval", 100) or 0)

    # Caches for per-sequence labels/images
    self._tracks_cache: Dict[int, Any] = {}
    self._tracks_t_cache: Dict[int, Any] = {}

    # Register hooks (empty by default until this trainer is fully implemented).
    self.register_hooks(self.build_hooks())

    # Signal handler for graceful SLURM preemption (saves latest.pth before exit).
    self._signal_received = False
    self._setup_signal_handler()

    # Auto-resume: if enabled and no explicit resume path, look for latest.pth.
    is_auto_resume = False
    if getattr(self.cfg.log, "auto_resume", False) and not getattr(self.cfg.log, "resume_from", None):
      latest = self.output_dir / "latest.pth"
      if latest.exists():
        self.cfg.log.resume_from = str(latest)
        is_auto_resume = True
        logger.info("Auto-resuming from %s", latest)

    # Resume from checkpoint if specified.
    if getattr(self.cfg.log, "resume_from", None):
      try:
        self.resume_checkpoint(
          str(self.cfg.log.resume_from),
          weights_only=bool(getattr(self.cfg.log, "load_weights_only", False)),
        )
        if bool(getattr(self.cfg.training, "unfreeze_mcm_after_resume", False)):
          self._unfreeze_mcm_after_resume()
      except RuntimeError:
        if not is_auto_resume:
          raise
        # latest.pth was likely truncated by a SIGKILL mid-save (e.g. SLURM
        # preemption). Don't crash the run over an unreadable auto-resume
        # checkpoint; start fresh instead.
        logger.warning(
          "Auto-resume checkpoint %s is unreadable (likely truncated by a "
          "preemption SIGKILL mid-save); starting fresh instead of resuming.",
          self.cfg.log.resume_from,
          exc_info=True,
        )
        self.cfg.log.resume_from = None

  def _setup_signal_handler(self) -> None:
    """Register handlers for graceful shutdown on SLURM preemption.

    SLURM sends SIGUSR1 when ``#SBATCH --signal=B:USR1@<seconds>`` is set,
    giving the job time to save a checkpoint before the hard kill.
    """
    def _handler(signum, frame):
      logger.warning("Received signal %s — will save checkpoint after current batch", signum)
      self._signal_received = True

    signal.signal(signal.SIGUSR1, _handler)
    signal.signal(signal.SIGTERM, _handler)

  @property
  def train_data_loader_iter(self):
    if self._train_loader_iter is None:
      self._train_loader_iter = iter(self.train_loader)
    return self._train_loader_iter

  def reset_data_loader_iter(self) -> None:
    self._train_loader_iter = None
    self._train_prefetcher = None

  def _release_train_pipeline_for_eval(self) -> None:
    """Release the training input pipeline before inline eval (anti-OOM).

    Eval-only has a sub-GB host footprint, but leaving the training prefetcher +
    persistent DataLoader workers (and their pinned buffers / allocator pool)
    alive during inline eval makes host RSS balloon (could OOM the job). Shut the
    train workers down + drop the iterator so inline eval runs in the same clean
    state as eval-only; the next train step lazily rebuilds it. Bump the
    InfiniteSampler epoch so the rebuild reshuffles instead of replaying indices.
    """
    if self._train_loader_iter is None and self._train_prefetcher is None:
      return
    try:
      sampler = getattr(self.train_loader, "sampler", None)
      if isinstance(sampler, InfiniteSampler):
        sampler.set_epoch(int(getattr(sampler, "_epoch", 0)) + 1)
      self._shutdown_dataloader_workers(self.train_loader, loader_name="train loader")
      self.reset_data_loader_iter()
      gc.collect()
      if torch.cuda.is_available():
        torch.cuda.empty_cache()
      try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
      except Exception:
        pass
      logger.info("Released training input pipeline before inline eval (anti-OOM).")
    except Exception as exc:
      logger.warning("train-pipeline release before eval failed: %s", exc)

  def _set_train_sampler_epoch(self, epoch: int) -> None:
    sampler = getattr(self.train_loader, "sampler", None)
    if sampler is None:
      return
    set_epoch = getattr(sampler, "set_epoch", None)
    if callable(set_epoch):
      set_epoch(int(epoch))

  def _train_sampler_is_infinite(self) -> bool:
    return isinstance(getattr(self.train_loader, "sampler", None), InfiniteSampler)

  def _prepare_train_epoch_data_stream(self, epoch: int, *, rebuild_prefetcher: bool = False) -> None:
    if self._train_sampler_is_infinite():
      if rebuild_prefetcher:
        self._train_prefetcher = None
      if not getattr(self, "_logged_continuous_infinite_sampler", False) and is_main_process():
        logger.info(
          "Keeping InfiniteSampler train stream continuous across logical epochs; "
          "max_steps_per_epoch controls checkpoint/eval cadence only."
        )
        self._logged_continuous_infinite_sampler = True
      return

    self._set_train_sampler_epoch(epoch)
    self.reset_data_loader_iter()

  def _event_voxel_resolution(self):
    """Return (height, width) for event voxelization.

    When EDUM is enabled the model upsamples internally, so we voxelize
    at the native event camera resolution.  Otherwise we use the image
    resolution (current Phase 1 behaviour).
    """
    edum_cfg = getattr(self.cfg.model, "edum", None)
    if bool(getattr(edum_cfg, "enabled", False)):
      return (
        int(getattr(edum_cfg, "event_native_height", 480)),
        int(getattr(edum_cfg, "event_native_width", 640)),
      )
    return (
      int(getattr(self.cfg.data, "image_height", 480)),
      int(getattr(self.cfg.data, "image_width", 640)),
    )

  def _event_coordinate_resolution(self, voxel_height: int, voxel_width: int):
    """Return source coordinate resolution for raw event tensors."""
    img_h = int(getattr(self.cfg.data, "image_height", voxel_height))
    img_w = int(getattr(self.cfg.data, "image_width", voxel_width))
    prefer_distorted = bool(getattr(self.cfg.data, "mapper_prefer_distorted", True))
    fullres_rectified = (not prefer_distorted) and (img_h == 1080 and img_w == 1440)
    edum_cfg = getattr(self.cfg.model, "edum", None)
    rectified_event_resolution = str(
      getattr(self.cfg.data, "mapper_rectified_events_resolution", "auto") or "auto"
    ).lower()
    native_rectified_events = (
      fullres_rectified
      and bool(getattr(edum_cfg, "enabled", False))
      and rectified_event_resolution in {"auto", "orig", "native", "orig-res", "event"}
    )
    if native_rectified_events:
      return int(voxel_height), int(voxel_width)
    if fullres_rectified:
      return img_h, img_w
    return int(voxel_height), int(voxel_width)

  def _get_train_prefetcher(self) -> BaseDSECPrefetcher:
    if self._train_prefetcher is None:
      ev_h, ev_w = self._event_voxel_resolution()
      self._train_prefetcher = build_dsec_prefetcher(
        self.train_data_loader_iter,
        config=self.cfg,
        device=self.device,
        require_targets=True,
        voxel_height=ev_h,
        voxel_width=ev_w,
        train_augment=True,
        flow_provider=self._flow_provider,
        sequence_names=self._flow_sequence_names if self._flow_provider is not None else None,
        stem_provider=self._stem_provider,
        stem_sequence_names=self._stem_sequence_names if self._stem_provider is not None else None,
        iwe_provider=self._iwe_provider,
        iwe_sequence_names=self._iwe_sequence_names if self._iwe_provider is not None else None,
      )
      if self._train_prefetcher.prepared_representation == "streaming_events":
        self._validate_streaming_train_prefetcher()
      # Apply no_aug-phase override: built prefetcher has aug from config, force off
      # when the closing no_aug_epochs window says so (see epoch-loop logic).
      if getattr(self, "_force_no_aug_this_epoch", False):
        self._train_prefetcher.use_mosaic = False
        self._train_prefetcher.use_mixup = False
    return self._train_prefetcher

  @staticmethod
  def _sequence_names_for_loader(loader: Optional[DataLoader]) -> Optional[List[Optional[str]]]:
    if loader is None:
      return None
    dataset = getattr(loader, "dataset", None)
    wrapped = getattr(dataset, "dataset", None)
    if wrapped is not None:
      dataset = wrapped
    names = getattr(dataset, "sequence_names", None)
    if names is None:
      return None
    return list(names)

  def _validate_streaming_train_prefetcher(self) -> None:
    """Validate the narrow streaming-SMCM detector path before first batch."""

    raw_model = self.model.module if hasattr(self.model, "module") else self.model
    backbone = getattr(raw_model, "backbone", None)
    mcm = getattr(backbone, "mcm", None)
    if backbone is None or mcm is None or not bool(getattr(backbone, "use_mcm", False)):
      raise NotImplementedError(
        "Streaming-event batches require an enabled streaming SMCM on the detector backbone."
      )
    if str(getattr(mcm, "primary_event_representation", "")) != "streaming_events":
      raise NotImplementedError(
        "Streaming-event batches require model.backbone.mcm.primary_event_representation="
        "'streaming_events'."
      )
    if not bool(mcm.should_use_frozen_inference_path()):
      raise NotImplementedError(
        "Detector Trainer supports streaming-event batches only for frozen SMCM probes. "
        "Set model.mcm.freeze=true or use the dedicated SMCM trainer for trainable streaming MCM."
      )
    logger.info(
      "Streaming-event train prefetcher enabled for frozen SMCM detector probe "
      "(mcm=%s, required_keys=%s).",
      type(mcm).__name__,
      tuple(getattr(mcm, "required_batch_keys", ())),
    )

  def _get_seq_meta(self, seq_idx: int):
    if self._dsec_per_sequence is None:
      return None
    if seq_idx < 0 or seq_idx >= len(self._dsec_per_sequence):
      return None
    return getattr(self._dsec_per_sequence[seq_idx], "meta", None)

  def _load_image_tensor(self, seq_idx: int, frame_index: int) -> torch.Tensor:
    meta = self._get_seq_meta(seq_idx)
    if meta is None or meta.images_dir is None:
      # Fallback: black image
      h = int(getattr(self.cfg.data, "image_height", 480))
      w = int(getattr(self.cfg.data, "image_width", 640))
      return torch.zeros((3, h, w), dtype=torch.float32)

    img_dir = Path(meta.images_dir)
    img_name = f"{int(frame_index):06d}.png"

    # Prefer distorted images when available (they match the event sensor resolution).
    if img_dir.name == "rectified":
      distorted_dir = img_dir.parent / "distorted"
      if distorted_dir.is_dir() and (distorted_dir / img_name).exists():
        img_dir = distorted_dir

    img_path = img_dir / img_name
    if not img_path.exists():
      h = int(getattr(self.cfg.data, "image_height", 480))
      w = int(getattr(self.cfg.data, "image_width", 640))
      return torch.zeros((3, h, w), dtype=torch.float32)

    import cv2

    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
      h = int(getattr(self.cfg.data, "image_height", 480))
      w = int(getattr(self.cfg.data, "image_width", 640))
      return torch.zeros((3, h, w), dtype=torch.float32)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(img_rgb).permute(2, 0, 1).contiguous().float()

  def _ensure_tracks_loaded(self, seq_idx: int) -> None:
    if seq_idx in self._tracks_cache:
      return
    meta = self._get_seq_meta(seq_idx)
    if meta is None or (not getattr(meta, "has_labels", False)) or getattr(meta, "tracks_npy_path", None) is None:
      self._tracks_cache[seq_idx] = None
      self._tracks_t_cache[seq_idx] = None
      return

    import numpy as np

    tracks = np.load(str(meta.tracks_npy_path), mmap_mode="r", allow_pickle=False)
    self._tracks_cache[seq_idx] = tracks
    self._tracks_t_cache[seq_idx] = tracks["t"]

  def _tracks_for_frame(self, seq_idx: int, t_ref_us: int):
    import numpy as np

    self._ensure_tracks_loaded(seq_idx)
    tracks = self._tracks_cache.get(seq_idx)
    t_arr = self._tracks_t_cache.get(seq_idx)
    if tracks is None or t_arr is None:
      return None
    t_ref_us = int(t_ref_us)
    left = int(np.searchsorted(t_arr, t_ref_us, side="left"))
    right = int(np.searchsorted(t_arr, t_ref_us, side="right"))
    if right <= left:
      return None
    return tracks[left:right]

  def _build_targets(self, seq_idx: int, t_ref_us: int, img_h: int, img_w: int) -> torch.Tensor:
    tracks = self._tracks_for_frame(seq_idx, t_ref_us)
    if tracks is None or len(tracks) == 0:
      return torch.zeros((0, 5), dtype=torch.float32)

    import numpy as np

    labels = np.stack(
      [tracks["x"], tracks["y"], tracks["w"], tracks["h"], tracks["class_id"]],
      axis=1,
    ).astype(np.float32)
    labels_t = torch.from_numpy(labels)

    # Convert [x,y,w,h,class_id] (top-left) -> [class_id,cx,cy,w,h]
    out = torch.zeros_like(labels_t)
    out[:, 0] = labels_t[:, 4]
    out[:, 1] = labels_t[:, 0] + labels_t[:, 2] / 2.0
    out[:, 2] = labels_t[:, 1] + labels_t[:, 3] / 2.0
    out[:, 3] = labels_t[:, 2]
    out[:, 4] = labels_t[:, 3]

    out[:, 1] = torch.clamp(out[:, 1], 0, float(img_w - 1))
    out[:, 2] = torch.clamp(out[:, 2], 0, float(img_h - 1))
    return out

  def _empty_event_voxels_like(self, images: torch.Tensor) -> torch.Tensor:
    """Build zero event voxels for RGB-only batches.

    RGB mode still uses the dual-stream model signature, so we provide
    zero-valued event tensors when the loader/prefetcher intentionally omits
    event representations.
    """
    if not isinstance(images, torch.Tensor) or images.ndim != 4:
      raise ValueError(f"Expected images tensor with shape (B,C,H,W), got {type(images)}")
    bins = int(getattr(self.cfg.data, "event_bins", 5))
    b, _, h, w = images.shape
    return torch.zeros((int(b), bins, int(h), int(w)), dtype=images.dtype, device=images.device)

  def _prepare_batch(self, batch: Dict[str, Any], *, require_targets: bool) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Convert a packed-events batch into model inputs.

    Returns:
      images: (B,3,H,W) float32
      voxels: (B,BINS,H,W) float32
      targets: (B,max_boxes,5) float32 or None
    """
    seq_idx_t = batch.get("seq_idx")
    frame_index_t = batch.get("frame_index")
    t_ref_us_t = batch.get("t_ref_us")

    if seq_idx_t is None or frame_index_t is None:
      raise ValueError("Batch missing required keys: seq_idx/frame_index")

    seq_idx_list = seq_idx_t.tolist() if isinstance(seq_idx_t, torch.Tensor) else list(seq_idx_t)
    frame_list = frame_index_t.tolist() if isinstance(frame_index_t, torch.Tensor) else list(frame_index_t)
    t_ref_list = (
      t_ref_us_t.tolist() if isinstance(t_ref_us_t, torch.Tensor) else [None] * len(seq_idx_list)
    )

    # Images
    images = [self._load_image_tensor(int(s), int(f)) for s, f in zip(seq_idx_list, frame_list)]
    images_t = torch.stack(images, dim=0)
    img_h = int(images_t.shape[-2])
    img_w = int(images_t.shape[-1])

    # Events -> voxel grid (CPU) then stack
    bins = int(getattr(self.cfg.data, "event_bins", 5))
    ev_h, ev_w = self._event_voxel_resolution()
    src_h, src_w = self._event_coordinate_resolution(ev_h, ev_w)

    events_packed: torch.Tensor = batch.get("events")
    offsets: torch.Tensor = batch.get("events_offsets")
    if events_packed is None or offsets is None:
      raise ValueError("Batch missing required keys: events/events_offsets")

    voxels: List[torch.Tensor] = []
    for i in range(len(seq_idx_list)):
      start = int(offsets[i].item())
      end = int(offsets[i + 1].item())
      ev_i = events_packed[start:end]
      if isinstance(ev_i, torch.Tensor):
        ev_np = ev_i.cpu().numpy()
      else:
        ev_np = ev_i
      voxel = events_to_voxel_grid(
        ev_np,
        num_bins=bins,
        width=ev_w,
        height=ev_h,
        source_width=src_w,
        source_height=src_h,
      )
      voxels.append(torch.as_tensor(voxel, dtype=torch.float32))
    voxels_t = torch.stack(voxels, dim=0)

    targets_t: Optional[torch.Tensor] = None
    if require_targets:
      per_sample = []
      for s, t_ref in zip(seq_idx_list, t_ref_list):
        if t_ref is None:
          per_sample.append(torch.zeros((0, 5), dtype=torch.float32))
        else:
          per_sample.append(self._build_targets(int(s), int(t_ref), img_h=img_h, img_w=img_w))

      max_num = max((int(x.shape[0]) for x in per_sample), default=0)
      if max_num <= 0:
        max_num = 1
      targets_t = torch.zeros((len(per_sample), max_num, 5), dtype=torch.float32)
      for i, t in enumerate(per_sample):
        if t.numel() == 0:
          continue
        n = min(int(t.shape[0]), max_num)
        targets_t[i, :n].copy_(t[:n])

    return images_t, voxels_t, targets_t

  def _batch_forward_kwargs(self, batch: Dict[str, Any]) -> Dict[str, Any]:
    """Collect optional per-sample context consumed by MCM/event renderers."""
    extra_kwargs: Dict[str, Any] = {}
    for key in (
      "events",
      "events_offsets",
      "t_start_us",
      "t_end_us",
      "t_ref_us",
      "seq_idx",
      "frame_index",
      "sequence_reset",
      "estimator_input",
      "mcm_estimator_input",
      "iwe_mapping",
      "event_to_rgb_mapping",
      "mapping",
    ):
      value = batch.get(key)
      if value is not None:
        extra_kwargs[key] = value
    return extra_kwargs

  def _warmup_lr(self, batch_idx: int, num_batches: int) -> None:
    cfg = self.config.training
    total_warmup_iters = int(cfg.warmup_epochs) * int(num_batches)
    current_iter = int(self.epoch) * int(num_batches) + int(batch_idx)
    if total_warmup_iters <= 0 or current_iter >= total_warmup_iters:
      return

    warmup_factor = float(current_iter) / float(total_warmup_iters)
    base_warmup_lr = float(cfg.warmup_lr)
    base_target_lr = float(cfg.learning_rate)
    for param_group in self.optimizer.param_groups:
      # Each group may have its own target LR (e.g. smcm_lr_factor).
      # Use the group's "initial_lr" if set by the scheduler, otherwise
      # scale proportionally from the group's current "lr" setting.
      target_lr = float(param_group.get("initial_lr", param_group["lr"]))
      # Scale warmup_lr proportionally to preserve the per-group ratio.
      ratio = target_lr / base_target_lr if base_target_lr > 0 else 1.0
      group_warmup_lr = base_warmup_lr * ratio
      param_group["lr"] = group_warmup_lr + (target_lr - group_warmup_lr) * warmup_factor

  def _grad_total_norm(self, parameters, norm_type: float = 2.0) -> torch.Tensor:
    """Compute total grad norm without modifying gradients."""
    grads = [p.grad.detach() for p in parameters if p.grad is not None]
    if not grads:
      return torch.zeros((), device=self.device)

    norm_type = float(norm_type)
    if norm_type == float("inf"):
      norms = [g.abs().max().to(device=self.device) for g in grads]
      return torch.stack(norms).max()

    norms = [
      torch.linalg.vector_norm(g, ord=norm_type).to(device=self.device)
      for g in grads
    ]
    return torch.linalg.vector_norm(torch.stack(norms), ord=norm_type)

  def _raw_model(self) -> nn.Module:
    return self.model.module if hasattr(self.model, "module") else self.model

  @staticmethod
  def _is_mcm_state_key(key: str) -> bool:
    key = str(key)
    return (
      ".mcm." in key
      or key.startswith("backbone.mcm")
      or key.startswith("module.backbone.mcm")
    )

  def _mcm_module(self) -> Optional[nn.Module]:
    raw = self._raw_model()
    backbone = getattr(raw, "backbone", None)
    return getattr(backbone, "mcm", None) if backbone is not None else None

  def _reset_mcm_state_for_step_if_requested(self) -> None:
    if not bool(getattr(self.cfg.training, "reset_mcm_state_each_step", False)):
      return
    mcm = self._mcm_module()
    if mcm is None:
      if not getattr(self, "_reset_mcm_state_missing_warned", False):
        logger.warning(
          "training.reset_mcm_state_each_step=true but the model has no live MCM module."
        )
        self._reset_mcm_state_missing_warned = True
      return
    reset = getattr(mcm, "reset_state", None)
    if not callable(reset):
      raise RuntimeError(
        "training.reset_mcm_state_each_step=true but the live MCM does not expose reset_state()."
      )
    reset()

  def _mcm_parameters(self) -> List[nn.Parameter]:
    mcm = self._mcm_module()
    if mcm is None:
      return []
    return [p for p in mcm.parameters()]

  def _assert_mcm_gradients_reached(self) -> None:
    if not bool(getattr(self.cfg.training, "require_mcm_grad_check", False)):
      return
    if bool(getattr(self, "_mcm_grad_check_passed", False)):
      return
    params = [p for p in self._mcm_parameters() if p.requires_grad]
    if not params:
      raise RuntimeError("MCM gradient check requested, but no trainable MCM parameters were found.")
    grads = [p.grad for p in params]
    missing = sum(1 for grad in grads if grad is None)
    finite = [
      bool(torch.isfinite(grad.detach()).all().item())
      for grad in grads
      if grad is not None
    ]
    if missing > 0 or not finite or not all(finite):
      raise RuntimeError(
        "MCM gradient reachability check failed: "
        f"params={len(params)} missing_grad={missing} finite_grads={sum(finite)}/{len(finite)}"
      )
    self._mcm_grad_check_passed = True
    logger.info("MCM gradient reachability check passed at iter %d", int(self.iteration))

  @staticmethod
  def _safe_metric_name(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(name).strip().lower()).strip("_")

  def _optimizer_lr_metrics(self) -> Dict[str, float]:
    if self.optimizer is None:
      return {}
    metrics: Dict[str, float] = {}
    for idx, group in enumerate(self.optimizer.param_groups):
      lr = float(group.get("lr", 0.0))
      metrics[f"lr_group_{idx}"] = lr
      name = group.get("name")
      if name:
        safe_name = self._safe_metric_name(str(name))
        if safe_name:
          metrics[f"lr_{safe_name}"] = lr
    return metrics

  def _assert_scheduler_param_groups_current(self) -> None:
    sched = getattr(self, "scheduler", None)
    if sched is None or self.optimizer is None:
      return
    n_groups = len(self.optimizer.param_groups)

    def _check_one(obj: Any, label: str) -> None:
      for attr in ("base_lrs", "_last_lr"):
        values = getattr(obj, attr, None)
        if values is not None and len(values) != n_groups:
          raise RuntimeError(
            f"Scheduler {label}.{attr} has {len(values)} entries but optimizer has "
            f"{n_groups} param groups. The terminal_mcm group would not follow "
            "the shared LR schedule."
          )

    _check_one(sched, type(sched).__name__)
    for child_idx, child in enumerate(getattr(sched, "_schedulers", []) or []):
      _check_one(child, f"{type(sched).__name__}._schedulers[{child_idx}]/{type(child).__name__}")

  def _patch_scheduler_for_added_group(self, *, base_lr: float, current_lr: float) -> None:
    sched = getattr(self, "scheduler", None)
    if sched is None:
      return
    if hasattr(sched, "base_lrs"):
      sched.base_lrs.append(float(base_lr))
    if hasattr(sched, "_last_lr"):
      sched._last_lr.append(float(current_lr))
    for child in getattr(sched, "_schedulers", []) or []:
      if hasattr(child, "base_lrs"):
        child.base_lrs.append(float(base_lr))
      if hasattr(child, "_last_lr"):
        child._last_lr.append(float(current_lr))
    self._assert_scheduler_param_groups_current()

  def _unfreeze_mcm_after_resume(self) -> None:
    """Enable live MCM training after restoring a frozen-prefix checkpoint."""
    if getattr(self.cfg.data, "precomputed_iwe_dir", None):
      raise RuntimeError(
        "training.unfreeze_mcm_after_resume requires live MCM; "
        "data.precomputed_iwe_dir must be null/empty."
      )
    if not bool(getattr(getattr(self.cfg.model, "mcm", None), "enabled", False)):
      raise RuntimeError("training.unfreeze_mcm_after_resume requires model.mcm.enabled=true")

    raw = self._raw_model()
    mcm = self._mcm_module()
    if mcm is None:
      raise RuntimeError("Could not find raw_model.backbone.mcm for terminal unfreeze.")

    existing_param_ids = {
      id(p)
      for group in self.optimizer.param_groups
      for p in group.get("params", [])
    }
    new_params: List[nn.Parameter] = []
    for param in mcm.parameters():
      param.requires_grad = True
      if id(param) not in existing_param_ids:
        new_params.append(param)

    if not new_params:
      raise RuntimeError("MCM unfreeze requested, but all MCM parameters were already in the optimizer.")

    base_ref = float(self.optimizer.param_groups[0].get("initial_lr", getattr(self.cfg.training, "learning_rate", 0.0)))
    current_ref = float(self.optimizer.param_groups[0].get("lr", base_ref))
    factor = float(getattr(getattr(self.cfg.model, "mcm", None), "lr_factor", 1.0))
    mcm_base_lr = base_ref * factor
    mcm_current_lr = current_ref * factor
    weight_decay = float(getattr(self.cfg.training, "weight_decay", 0.0))
    self.optimizer.add_param_group({
      "params": new_params,
      "lr": mcm_current_lr,
      "initial_lr": mcm_base_lr,
      "weight_decay": weight_decay,
      "name": "terminal_mcm",
    })
    self._patch_scheduler_for_added_group(base_lr=mcm_base_lr, current_lr=mcm_current_lr)

    if isinstance(self.model, DistributedDataParallel):
      self.model = raw
      self.model = self._wrap_ddp_if_needed(self.model)

    trainable_values = sum(p.numel() for p in new_params)
    self._assert_scheduler_param_groups_current()
    logger.info(
      "Phase2.2 terminal MCM unfreeze: added %d tensors (%d values) as optimizer group %d "
      "with lr=%g initial_lr=%g weight_decay=%g",
      len(new_params),
      trainable_values,
      len(self.optimizer.param_groups) - 1,
      mcm_current_lr,
      mcm_base_lr,
      weight_decay,
    )

  def _lookup_live_iwe_reference_batch(self, batch: Dict[str, Any]) -> torch.Tensor:
    provider = self._live_iwe_reference_provider
    sequence_names = self._live_iwe_reference_sequence_names
    if provider is None or sequence_names is None:
      raise RuntimeError("Live/cache IWE reference provider is not initialised.")
    seq_idx = batch.get("seq_idx")
    frame_idx = batch.get("frame_index")
    if seq_idx is None or frame_idx is None:
      raise RuntimeError("Live/cache IWE verification requires seq_idx + frame_index in the batch.")
    seq_idx_list = seq_idx.detach().cpu().tolist() if isinstance(seq_idx, torch.Tensor) else list(seq_idx)
    frame_idx_list = frame_idx.detach().cpu().tolist() if isinstance(frame_idx, torch.Tensor) else list(frame_idx)

    iwes: List[torch.Tensor] = []
    for si, fi in zip(seq_idx_list, frame_idx_list):
      name = sequence_names[int(si)]
      if name is None:
        raise RuntimeError(f"No sequence name for seq_idx={int(si)} in live/cache IWE verification")
      iwes.append(provider.get(str(name), int(fi)))
    return torch.stack(iwes, dim=0).to(self.device, non_blocking=True)

  def _assert_live_iwe_matches_reference_cache(self, batch: Dict[str, Any]) -> Dict[str, float]:
    if not bool(getattr(self.cfg.training, "require_live_iwe_cache_match", False)):
      return {}
    if bool(getattr(self, "_live_iwe_cache_match_passed", False)):
      return {}
    if self._live_iwe_reference_provider is None:
      raise RuntimeError(
        "training.require_live_iwe_cache_match=true but data.live_iwe_reference_dir is not set."
      )

    raw = self._raw_model()
    viz = getattr(raw, "_mcm_viz", None)
    live = viz.get("output") if isinstance(viz, dict) else None
    if not isinstance(live, torch.Tensor):
      raise RuntimeError(
        "Live/cache IWE verification requested, but the live model did not expose "
        "_mcm_viz['output'] after forward."
      )

    ref = self._lookup_live_iwe_reference_batch(batch)
    if tuple(live.shape) != tuple(ref.shape):
      raise RuntimeError(
        "Live/cache IWE verification shape mismatch: "
        f"live={tuple(live.shape)} cache={tuple(ref.shape)}"
      )

    live_f = live.detach().to(dtype=torch.float32)
    ref_f = ref.detach().to(dtype=torch.float32)
    diff = (live_f - ref_f).abs()
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    mean_abs = float(diff.mean().item()) if diff.numel() else 0.0
    denom = torch.linalg.vector_norm(ref_f).clamp_min(1e-12)
    rel_l2 = float((torch.linalg.vector_norm(live_f - ref_f) / denom).item())
    atol = float(getattr(self.cfg.training, "live_iwe_cache_match_atol", 0.02))
    rtol = float(getattr(self.cfg.training, "live_iwe_cache_match_rtol", 0.05))
    if max_abs > atol and rel_l2 > rtol:
      raise RuntimeError(
        "Live/cache IWE seam check failed before optimizer step: "
        f"max_abs={max_abs:.6g} mean_abs={mean_abs:.6g} rel_l2={rel_l2:.6g} "
        f"(limits: max_abs<={atol:g} or rel_l2<={rtol:g}). "
        "This indicates the live MCM output does not match the cached prefix IWE."
      )

    self._live_iwe_cache_match_passed = True
    logger.info(
      "Live/cache IWE seam check passed at iter %d: shape=%s max_abs=%.6g mean_abs=%.6g rel_l2=%.6g",
      int(self.iteration), tuple(live.shape), max_abs, mean_abs, rel_l2,
    )
    return {
      "live_iwe_cache_max_abs": max_abs,
      "live_iwe_cache_mean_abs": mean_abs,
      "live_iwe_cache_rel_l2": rel_l2,
    }

  def _sync_skip_optimizer_step(self, local_skip: bool) -> bool:
    """Return True when any DDP rank needs to skip the optimizer step."""
    skip = bool(local_skip)
    if int(get_world_size()) <= 1 or not dist.is_available() or not dist.is_initialized():
      return skip

    device = self.device if self.device.type == "cuda" else torch.device("cpu")
    flag = torch.tensor([1 if skip else 0], device=device, dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(int(flag.item()))

  def _sync_and_abort_if_bad_step(
    self,
    local_bad: bool,
    *,
    reason: str,
    loss_dict: Optional[Dict[str, Any]] = None,
    value: Optional[float] = None,
  ) -> None:
    """Abort all DDP ranks when any rank detects an unrecoverable bad batch."""
    bad = bool(local_bad)
    if int(get_world_size()) > 1 and dist.is_available() and dist.is_initialized():
      device = self.device if self.device.type == "cuda" else torch.device("cpu")
      flag = torch.tensor([1 if bad else 0], device=device, dtype=torch.int32)
      dist.all_reduce(flag, op=dist.ReduceOp.MAX)
      bad = bool(int(flag.item()) > 0)

    if not bad:
      return

    if bool(local_bad):
      logger.error(
        "Fatal non-finite training state at iter %d on rank %s: %s value=%s loss_dict=%s",
        self.iteration,
        os.environ.get("RANK", os.environ.get("LOCAL_RANK", "?")),
        reason,
        value,
        loss_dict,
      )
    else:
      logger.error(
        "Fatal non-finite training state at iter %d detected on another DDP rank: %s",
        self.iteration,
        reason,
      )

    raise FloatingPointError(
      f"Fatal non-finite training state at iteration {self.iteration}: {reason}. "
      "All DDP ranks abort synchronously to avoid collective desynchronization."
    )

  def _check_consec_nan_skip_limit(self) -> None:
    """Abort all DDP ranks when consecutive skipped steps exceed the configured limit."""
    limit = int(getattr(self.config.training, "max_consecutive_nan_skips", 0) or 0)
    if limit <= 0:
      return
    n = int(getattr(self, "_n_consec_skipped_steps", 0))
    if n < limit:
      return
    logger.error(
      "Consecutive NaN/Inf skip limit reached at iter %d: %d consecutive bad steps (limit=%d). "
      "Model has diverged — aborting to avoid wasting compute.",
      self.iteration, n, limit,
    )
    self._sync_and_abort_if_bad_step(
      True,
      reason=f"max_consecutive_nan_skips={limit} exceeded ({n} consecutive bad steps)",
    )

  def _assert_loss_finite_across_ranks(self, loss: torch.Tensor, loss_dict: Dict[str, Any]) -> bool:
    """Return True when any rank has a non-finite forward loss. Logs but does not raise."""
    loss_value = float(loss.detach().float().item())
    local_bad = not np.isfinite(loss_value)
    bad = bool(local_bad)
    if int(get_world_size()) > 1 and dist.is_available() and dist.is_initialized():
      device = self.device if self.device.type == "cuda" else torch.device("cpu")
      flag = torch.tensor([1 if local_bad else 0], device=device, dtype=torch.int32)
      dist.all_reduce(flag, op=dist.ReduceOp.MAX)
      bad = bool(int(flag.item()) > 0)
    if not bad:
      return False
    rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "?"))
    if local_bad:
      logger.warning(
        "Non-finite forward loss at iter %d (rank %s): value=%s — skipping step. loss_dict=%s",
        self.iteration, rank, loss_value,
        {k: str(v) for k, v in loss_dict.items() if not k.startswith("_")},
      )
    else:
      logger.warning(
        "Non-finite forward loss at iter %d on another DDP rank — rank %s skipping step too.",
        self.iteration, rank,
      )
    if bool(getattr(self.config.training, "abort_on_nonfinite_forward_loss", False)):
      self._sync_and_abort_if_bad_step(
        local_bad,
        reason="non-finite forward loss",
        loss_dict={k: str(v) for k, v in loss_dict.items() if not k.startswith("_")},
        value=loss_value if local_bad else None,
      )
    return True

  def _snapshot_floating_buffers(self) -> Dict[str, torch.Tensor]:
    """Capture mutable floating buffers so skipped bad forwards do not poison state."""

    model = self.model.module if hasattr(self.model, "module") else self.model
    return {
      name: buffer.detach().clone()
      for name, buffer in model.named_buffers()
      if torch.is_tensor(buffer) and buffer.is_floating_point()
    }

  def _restore_floating_buffers(self, snapshot: Dict[str, torch.Tensor]) -> None:
    if not snapshot:
      return
    model = self.model.module if hasattr(self.model, "module") else self.model
    buffers = dict(model.named_buffers())
    restored = 0
    with torch.no_grad():
      for name, saved in snapshot.items():
        current = buffers.get(name)
        if current is None or not torch.is_tensor(current):
          continue
        if current.shape != saved.shape:
          continue
        current.copy_(saved.to(device=current.device, dtype=current.dtype))
        restored += 1
    if restored:
      logger.warning(
        "Restored %d floating model buffers after skipped non-finite forward at iter %d.",
        restored,
        self.iteration,
      )

  def run_step(self) -> None:
    # Log memory before forward
    log_memory_stats(
      "before_forward",
      logger,
      every_n_steps=self._mem_log_interval_steps,
      step=self.iteration,
    )
    assert self.model.training, "Model should be in training mode"

    step_start = time.perf_counter()
    try:
      batch = self._get_train_prefetcher().next()
    except StopIteration:
      self.reset_data_loader_iter()
      raise

    # Measure actual batch fetch wait from the prefetcher.
    data_time = time.perf_counter() - step_start
    prefetch_profile_events = batch.get("_profile_events")

    if "image" not in batch or "targets" not in batch:
      # Compatibility fallback (slower): old path that prepares everything inside trainer.
      images, voxels, targets = self._prepare_batch(batch, require_targets=True)
      images = images.to(self.device, non_blocking=True)
      voxels = voxels.to(self.device, non_blocking=True)
      assert targets is not None
      targets = targets.to(self.device, non_blocking=True)
    else:
      images = batch["image"]
      voxels = batch.get("events_voxel")
      if not isinstance(voxels, torch.Tensor):
        voxels = self._empty_event_voxels_like(images)
      targets = batch["targets"]

    # Forward sparse-event metadata when available. Some detector-facing MCM
    # outputs render from packed events rather than from the dense voxel grid.
    extra_kwargs = self._batch_forward_kwargs(batch)
    if bool(getattr(self.config.training, "profile_model_stages", False)):
      extra_kwargs["profile_stages"] = True
    # Forward raw events when available (needed by EventWarpingLoss).
    if "events" in batch and batch["events"] is not None:
      extra_kwargs["events"] = batch["events"]
    if "events_offsets" in batch and batch["events_offsets"] is not None:
      extra_kwargs["events_offsets"] = batch["events_offsets"]
    if "future_targets" in batch and batch["future_targets"] is not None:
      extra_kwargs["future_targets"] = batch["future_targets"]
    # Precomputed OFE flow (e.g. E-RAFT) overrides MCM flow at the STTP
    # head via yolox.py:134. No-op when data.precomputed_flow_dir is unset.
    precomp_flow = batch.get("precomputed_flow")
    if precomp_flow is not None:
      extra_kwargs["flow"] = precomp_flow
    if not getattr(self, "_flow_injection_verified", False) and self._flow_provider is not None:
      if precomp_flow is None:
        raise RuntimeError(
          "data.precomputed_flow_dir is set but batch lacks 'precomputed_flow'. "
          "Train-side flow injection plumbing is broken — refusing to silently "
          "train on the wrong flow source."
        )
      logger.info(
        "Train-side precomputed_flow active: shape=%s dtype=%s",
        tuple(precomp_flow.shape), precomp_flow.dtype,
      )
      self._flow_injection_verified = True

    stem_window = batch.get("stem_window")
    if stem_window is not None:
      extra_kwargs["stem_window"] = stem_window
    if not getattr(self, "_stem_window_verified", False) and self._stem_provider is not None:
      if stem_window is None:
        raise RuntimeError(
          "data.stem_feature_dir is set but batch lacks 'stem_window'. "
          "Train-side stem-window plumbing is broken — refusing to silently "
          "train the recurrent head on degenerate (single-frame) context."
        )
      first = stem_window[0]
      if first.dim() != 5 or first.shape[1] != int(self.config.model.sttp.context_window):
        raise RuntimeError(
          f"stem_window has wrong shape {tuple(first.shape)}; expected 5D with "
          f"K_ctx={self.config.model.sttp.context_window} at dim 1."
        )
      logger.info(
        "Train-side stem_window active: scales=%d shape=%s dtype=%s",
        len(stem_window), tuple(first.shape), first.dtype,
      )
      self._stem_window_verified = True

    if not getattr(self, "_iwe_cache_verified", False) and self._iwe_provider is not None:
      if not isinstance(voxels, torch.Tensor):
        raise RuntimeError(
          "data.precomputed_iwe_dir is set but batch lacks tensor 'events_voxel'. "
          "Train-side IWE cache plumbing is broken."
        )
      expected_channels = int(getattr(self.config.model, "precomputed_event_channels", 3))
      if int(voxels.shape[1]) != expected_channels:
        raise RuntimeError(
          f"Cached IWE batch has {voxels.shape[1]} channels; expected {expected_channels}."
        )
      logger.info(
        "Train-side precomputed IWE active: shape=%s dtype=%s",
        tuple(voxels.shape), voxels.dtype,
      )
      self._iwe_cache_verified = True

    buffer_snapshot = self._snapshot_floating_buffers()
    self._reset_mcm_state_for_step_if_requested()

    if self.use_amp:
      with autocast("cuda"):
        loss_dict = self.model(images, voxels, targets, **extra_kwargs)
        loss = loss_dict["total"]
      loss_dict.update(self._assert_live_iwe_matches_reference_cache(batch))
      if self._assert_loss_finite_across_ranks(loss, loss_dict):
        self._restore_floating_buffers(buffer_snapshot)
        self.optimizer.zero_grad(set_to_none=True)
        self._n_skipped_bad_steps = getattr(self, "_n_skipped_bad_steps", 0) + 1
        self._n_consec_skipped_steps = getattr(self, "_n_consec_skipped_steps", 0) + 1
        self._check_consec_nan_skip_limit()
        return
      profile_events = loss_dict.pop("_profile_events", None)
      if isinstance(prefetch_profile_events, dict):
        profile_events = dict(profile_events or {})
        profile_events.update(prefetch_profile_events)

      log_memory_stats(
        "after_forward",
        logger,
        every_n_steps=self._mem_log_interval_steps,
        step=self.iteration,
      )
      self.optimizer.zero_grad(set_to_none=True)
      assert self.scaler is not None
      backward_timer = CudaStageTimer(
        enabled=isinstance(profile_events, dict) and self.device.type == "cuda"
      )
      prev_scale = float(self.scaler.get_scale())
      with backward_timer.section("backward"):
        self.scaler.scale(loss).backward()
      if backward_timer.enabled:
        profile_events = dict(profile_events or {})
        profile_events.update(backward_timer.events)
      log_memory_stats(
        "after_backward",
        logger,
        every_n_steps=self._mem_log_interval_steps,
        step=self.iteration,
      )

      # Unscale before after_backward so hooks (e.g. GradientMonitorHook) see
      # real gradient magnitudes.  scaler.step() already handles inf detection.
      self.scaler.unscale_(self.optimizer)
      self._assert_mcm_gradients_reached()
      self.after_backward()

      grad_clip = float(getattr(self.config.training, "grad_clip", 0.0))
      skip_optimizer_step = False
      preclip_norm_f: Optional[float] = None
      if grad_clip > 0:
        preclip_total_norm = self._grad_total_norm(self.model.parameters())
        preclip_norm_f = float(preclip_total_norm.item() if isinstance(preclip_total_norm, torch.Tensor) else preclip_total_norm)
        if not torch.isfinite(torch.as_tensor(preclip_norm_f)):
          logger.warning("Non-finite pre-clip gradient norm at iter %d", self.iteration)
          skip_optimizer_step = True
        else:
          torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

      skip_optimizer_step = self._sync_skip_optimizer_step(skip_optimizer_step)
      if skip_optimizer_step:
        self._restore_floating_buffers(buffer_snapshot)
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.update()
        logger.warning(
          "Non-finite gradient norm at iter %d (%.4g) — skipping optimizer step.",
          self.iteration, preclip_norm_f if preclip_norm_f is not None else float("nan"),
        )
        self._n_skipped_bad_steps = getattr(self, "_n_skipped_bad_steps", 0) + 1
        self._n_consec_skipped_steps = getattr(self, "_n_consec_skipped_steps", 0) + 1
        self._check_consec_nan_skip_limit()
        return
      self.scaler.step(self.optimizer)
      self.scaler.update()
      new_scale = float(self.scaler.get_scale())

      if new_scale < prev_scale:
        self._restore_floating_buffers(buffer_snapshot)
        logger.warning(
          "AMP gradient overflow at iter %d: scaler reduced from %.1f to %.1f",
          self.iteration,
          prev_scale,
          new_scale,
        )
    else:
      loss_dict = self.model(images, voxels, targets, **extra_kwargs)
      loss = loss_dict["total"]
      loss_dict.update(self._assert_live_iwe_matches_reference_cache(batch))
      if self._assert_loss_finite_across_ranks(loss, loss_dict):
        self._restore_floating_buffers(buffer_snapshot)
        self.optimizer.zero_grad(set_to_none=True)
        self._n_skipped_bad_steps = getattr(self, "_n_skipped_bad_steps", 0) + 1
        self._n_consec_skipped_steps = getattr(self, "_n_consec_skipped_steps", 0) + 1
        self._check_consec_nan_skip_limit()
        return
      profile_events = loss_dict.pop("_profile_events", None)
      if isinstance(prefetch_profile_events, dict):
        profile_events = dict(profile_events or {})
        profile_events.update(prefetch_profile_events)

      log_memory_stats(
        "after_forward",
        logger,
        every_n_steps=self._mem_log_interval_steps,
        step=self.iteration,
      )
      self.optimizer.zero_grad(set_to_none=True)
      backward_timer = CudaStageTimer(
        enabled=isinstance(profile_events, dict) and self.device.type == "cuda"
      )
      with backward_timer.section("backward"):
        loss.backward()
      if backward_timer.enabled:
        profile_events = dict(profile_events or {})
        profile_events.update(backward_timer.events)
      log_memory_stats(
        "after_backward",
        logger,
        every_n_steps=self._mem_log_interval_steps,
        step=self.iteration,
      )
      self.after_backward()
      self._assert_mcm_gradients_reached()

      grad_clip = float(getattr(self.config.training, "grad_clip", 0.0))
      skip_optimizer_step = False
      # Always check gradient finiteness, independent of grad_clip, so a NaN/Inf
      # gradient never reaches optimizer.step() even when grad clipping is disabled
      # (the non-finite-loss guard in _write_metrics relies on this skip happening).
      # The AMP branch is covered separately by the GradScaler. Clip only if grad_clip > 0.
      preclip_total_norm = self._grad_total_norm(self.model.parameters())
      preclip_norm_f = float(preclip_total_norm.item() if isinstance(preclip_total_norm, torch.Tensor) else preclip_total_norm)
      if not torch.isfinite(torch.as_tensor(preclip_norm_f)):
        logger.warning("Non-finite pre-clip gradient norm at iter %d", self.iteration)
        skip_optimizer_step = True
      elif grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

      skip_optimizer_step = self._sync_skip_optimizer_step(skip_optimizer_step)
      if skip_optimizer_step:
        self._restore_floating_buffers(buffer_snapshot)
        self.optimizer.zero_grad(set_to_none=True)
        logger.warning(
          "Non-finite gradient norm at iter %d (%.4g) — skipping optimizer step.",
          self.iteration, preclip_norm_f if preclip_norm_f is not None else float("nan"),
        )
        self._n_skipped_bad_steps = getattr(self, "_n_skipped_bad_steps", 0) + 1
        self._n_consec_skipped_steps = getattr(self, "_n_consec_skipped_steps", 0) + 1
        self._check_consec_nan_skip_limit()
        return
      self.optimizer.step()

    if isinstance(profile_events, dict) and profile_events:
      stage_times_ms = collect_cuda_stage_times_ms(profile_events, device=self.device)
      for name, value in stage_times_ms.items():
        loss_dict[f"prof_{name}_ms"] = float(value)

    step_time = time.perf_counter() - step_start
    self._write_metrics(loss_dict, data_time, step_time)
    self._last_step_end_time = time.perf_counter()

  def train(self) -> None:
    logger.info("==== Starting training ====")
    log_memory_stats("train_start", logger)
    logger.info(f"Starting training for {self.config.training.num_epochs} epochs")

    base_iters_per_epoch = len(self.train_loader)
    iters_per_epoch = int(base_iters_per_epoch)
    if iters_per_epoch <= 0:
      raise ValueError("Train loader has no iterations per epoch")

    max_steps_per_epoch = int(getattr(self.config.training, "max_steps_per_epoch", 0) or 0)
    if max_steps_per_epoch > 0:
      iters_per_epoch = min(int(iters_per_epoch), int(max_steps_per_epoch))

    max_total_steps = int(getattr(self.config.training, "max_total_steps", 0) or 0)

    # Log epoch size for visibility. Note: our sampler shards indices by distributed rank,
    # so `iters_per_epoch` corresponds to the number of iterations each rank will run.
    if is_main_process():
      world_size = int(get_world_size())
      per_gpu_batch = int(getattr(self.cfg.training, "batch_size", 1))
      global_batch = per_gpu_batch * max(world_size, 1)

      try:
        dataset_len = len(self.train_loader.dataset)
      except Exception:
        dataset_len = None

      approx_global_samples = int(iters_per_epoch) * int(global_batch)
      if dataset_len is None:
        logger.info(
          "Train epoch size: %d iters/epoch (per rank), batch=%d per GPU (global=%d), approx %d samples/epoch",
          int(iters_per_epoch),
          int(per_gpu_batch),
          int(global_batch),
          int(approx_global_samples),
        )
      else:
        logger.info(
          "Train epoch size: %d iters/epoch (per rank), batch=%d per GPU (global=%d), dataset=%d samples, approx %d samples/epoch",
          int(iters_per_epoch),
          int(per_gpu_batch),
          int(global_batch),
          int(dataset_len),
          int(approx_global_samples),
        )
      if max_steps_per_epoch > 0 or max_total_steps > 0:
        logger.info(
          "Joint smoke/profiling limits active: base_iters=%d, max_steps_per_epoch=%s, max_total_steps=%s",
          int(base_iters_per_epoch),
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
          self.model.train()
          # YOLOX-style "close mosaic": disable mosaic+mixup in the final
          # no_aug_epochs of training so the model fine-tunes on clean
          # (unaugmented) images. The prefetcher is rebuilt lazily on the
          # next _get_train_prefetcher() call (reset above sets it to None),
          # and this flag forces the rebuild to skip aug. 0 = legacy behavior.
          prev_no_aug_phase = bool(getattr(self, "_force_no_aug_this_epoch", False))
          no_aug_epochs = int(getattr(self.config.training, "no_aug_epochs", 0))
          in_no_aug_phase = (
            no_aug_epochs > 0
            and epoch >= int(self.config.training.num_epochs) - no_aug_epochs
          )
          if in_no_aug_phase and not getattr(self, "_no_aug_logged", False):
            logger.info(
              "no_aug phase active at epoch %d (last %d epochs of %d): mosaic+mixup disabled",
              epoch, no_aug_epochs, int(self.config.training.num_epochs),
            )
            self._no_aug_logged = True
          self._force_no_aug_this_epoch = in_no_aug_phase
          self._prepare_train_epoch_data_stream(
            epoch,
            rebuild_prefetcher=bool(prev_no_aug_phase != in_no_aug_phase),
          )
          self._last_step_end_time = None

          # Re-apply eval() to frozen detector BN so running stats are used
          if self._freeze_detector:
            for name, module in self.model.named_modules():
              if "sttp_head" not in name and isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()

          losses = defaultdict(float)
          processed = 0

          self.before_epoch()
          # Reset per-epoch MCM NaN warning so it warns once per epoch
          _raw = self.model.module if hasattr(self.model, 'module') else self.model
          if hasattr(_raw, '_mcm_nan_warned'):
            _raw._mcm_nan_warned = False
          # Legacy
          if hasattr(_raw, '_smcm_nan_warned'):
            _raw._smcm_nan_warned = False
          epoch_start = time.perf_counter()

          log_memory_stats(f"epoch_{epoch+1}_start", logger)

          for batch_idx in range(int(iters_per_epoch)):
            if max_total_steps > 0 and int(self.iteration) >= int(max_total_steps):
              stop_training = True
              break
            self.storage.iter = int(self.iteration)

            if (not bool(getattr(self.config.training, "warmup_use_scheduler", False))) and int(self.epoch) < int(self.config.training.warmup_epochs):
              self._warmup_lr(batch_idx, int(iters_per_epoch))

            self.before_step()
            self.run_step()
            self.after_step()
            processed += 1

            # Coarse periodic epoch-progress memory probe.
            log_memory_stats(
              f"epoch_{epoch+1}_batch_{batch_idx}",
              logger,
              every_n_steps=self._mem_log_interval_steps,
              step=batch_idx,
            )

            # Log GPU memory once after first iteration for diagnostics
            if batch_idx == 0 and self.epoch == 0 and torch.cuda.is_available():
              alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
              resrv = torch.cuda.max_memory_reserved() / (1024 ** 2)
              logger.info("GPU memory after first iter: allocated=%.0f MiB, reserved=%.0f MiB", alloc, resrv)

            latest = dict(self._latest_local_train_metrics)
            for key in ["total", "loss_obj", "loss_cls", "loss_box", "loss_mcm",
                        "loss_ce", "loss_bbox", "loss_giou", "loss_fut", "loss_tc"]:
              if key in latest:
                losses[key] += float(latest[key])

            self.iteration += 1

            if self._signal_received:
              logger.info("Signal received — saving latest.pth and exiting")
              if is_main_process():
                self.save_checkpoint("latest.pth")
              self.after_train()
              sys.exit(0)

          log_memory_stats(f"epoch_{epoch+1}_end", logger)
          epoch_time = time.perf_counter() - epoch_start
          self.after_epoch()

          reduced_losses, reduced_processed = self._reduce_epoch_loss_sums(losses, processed)
          denom = max(reduced_processed, 1)
          avg = {k: (v / denom) for k, v in reduced_losses.items()}
          if is_main_process():
            self._log_epoch(avg, epoch_time)

      except Exception:
        logger.exception("Exception during training:")
        raise
      finally:
        self.after_train()
        if self.writer is not None:
          try:
            self.writer.flush()
            self.writer.close()
          except Exception:
            pass

  @property
  def iteration(self) -> int:
    return int(self.iter)

  @iteration.setter
  def iteration(self, value: int) -> None:
    self.iter = int(value)

  def _write_metrics(self, loss_dict: Dict[str, torch.Tensor], data_time: float, step_time: float) -> None:
    """Write metrics to MetricStorage (and TensorBoard on rank 0).

    This helper mirrors the old trainer behavior and is safe to call from a future `run_step()`.
    """
    if self.storage is None:
      return

    metrics_dict: Dict[str, float] = {
      k: (v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v))
      for k, v in loss_dict.items()
    }
    metrics_dict["data_time"] = float(data_time)
    metrics_dict["step_time"] = float(step_time)
    metrics_dict.update(self._optimizer_lr_metrics())
    self._latest_local_train_metrics = dict(metrics_dict)

    total_loss = metrics_dict.get("total", metrics_dict.get("loss_total", 0.0))
    local_nonfinite = not np.isfinite(float(total_loss))

    # Synchronize the non-finite flag across all DDP ranks. In DDP each rank
    # processes a different mini-batch shard, so one rank can have a NaN loss
    # while others have finite losses. Without this sync, the NaN rank takes
    # the early-return path below while the finite ranks enter
    # _reduce_metrics_for_logging and block in dist.all_reduce — deadlock.
    any_nonfinite = local_nonfinite
    if int(get_world_size()) > 1 and dist.is_available() and dist.is_initialized():
      _nf_flag = torch.tensor(
        [1 if local_nonfinite else 0], device=self.device, dtype=torch.int32
      )
      dist.all_reduce(_nf_flag, op=dist.ReduceOp.MAX)
      any_nonfinite = bool(int(_nf_flag.item()) > 0)

    if any_nonfinite:
      self._sync_and_abort_if_bad_step(
        True,
        reason="non-finite metric loss before logging",
        loss_dict=metrics_dict,
        value=float(total_loss),
      )
    self._consec_nonfinite = 0
    self._n_consec_skipped_steps = 0

    for k, v in metrics_dict.items():
      self.storage.put_scalar(k, float(v))

    log_interval = max(int(getattr(self.cfg.log, "log_interval", 0) or 0), 0)
    wandb_cfg = getattr(self.cfg, "wandb", None)
    wandb_log_interval = 0
    if self._is_wandb_train_logging_active():
      wandb_log_interval = max(int(getattr(wandb_cfg, "log_every_n_steps", 0) or 0), 0)

    emit_train_log = bool(log_interval > 0 and (self.iteration % log_interval) == 0)
    emit_wandb = bool(wandb_log_interval > 0 and (self.iteration % wandb_log_interval) == 0)

    self._latest_reduced_train_metrics = None
    self._latest_reduced_train_metrics_step = None
    if emit_train_log or emit_wandb:
      reduced_metrics = self._reduce_metrics_for_logging(metrics_dict)
      if is_main_process():
        self._latest_reduced_train_metrics = dict(reduced_metrics)
        self._latest_reduced_train_metrics_step = int(self.iteration)
        if emit_train_log:
          self._log_training(reduced_metrics)

  def _reduce_metrics_for_logging(self, metrics_dict: Dict[str, float]) -> Dict[str, float]:
    """Reduce step metrics across ranks only when they are about to be emitted."""
    if int(get_world_size()) <= 1 or not dist.is_available() or not dist.is_initialized():
      return dict(metrics_dict)

    device = self.device if self.device.type == "cuda" else torch.device("cpu")
    world_size = float(int(get_world_size()))
    keys = list(metrics_dict.keys())
    mean_keys = [k for k in keys if (k not in ("data_time", "step_time") and not k.startswith("prof_"))]
    max_keys = [k for k in keys if k not in mean_keys]

    reduced: Dict[str, float] = {}
    if mean_keys:
      mean_tensor = torch.tensor([float(metrics_dict[k]) for k in mean_keys], device=device, dtype=torch.float64)
      dist.all_reduce(mean_tensor, op=dist.ReduceOp.SUM)
      mean_tensor /= world_size
      for idx, key in enumerate(mean_keys):
        reduced[key] = float(mean_tensor[idx].item())

    if max_keys:
      max_tensor = torch.tensor([float(metrics_dict[k]) for k in max_keys], device=device, dtype=torch.float64)
      dist.all_reduce(max_tensor, op=dist.ReduceOp.MAX)
      for idx, key in enumerate(max_keys):
        reduced[key] = float(max_tensor[idx].item())

    return {k: reduced[k] for k in keys}

  def _is_wandb_train_logging_active(self) -> bool:
    wandb_cfg = getattr(self.cfg, "wandb", None)
    if wandb_cfg is None or not bool(getattr(wandb_cfg, "use_wandb", False)):
      return False

    env_disabled = str(os.environ.get("WANDB_DISABLED", "")).strip().lower()
    if env_disabled in {"1", "true", "yes", "on"}:
      return False

    env_mode = str(os.environ.get("WANDB_MODE", "")).strip().lower()
    cfg_mode = str(getattr(wandb_cfg, "mode", "online") or "online").strip().lower()
    mode = env_mode or cfg_mode
    return mode != "disabled"

  def _reduce_epoch_loss_sums(self, losses: Dict[str, float], processed: int) -> tuple[Dict[str, float], int]:
    """Reduce per-epoch summed losses and processed-step counts across ranks."""
    reduced_losses = {str(k): float(v) for k, v in losses.items()}
    reduced_processed = int(processed)

    if int(get_world_size()) <= 1 or not dist.is_available() or not dist.is_initialized():
      return reduced_losses, reduced_processed

    device = self.device if self.device.type == "cuda" else torch.device("cpu")
    loss_keys = list(reduced_losses.keys())
    if loss_keys:
      loss_tensor = torch.tensor([reduced_losses[k] for k in loss_keys], device=device, dtype=torch.float64)
      dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
      for idx, key in enumerate(loss_keys):
        reduced_losses[key] = float(loss_tensor[idx].item())

    processed_tensor = torch.tensor([float(reduced_processed)], device=device, dtype=torch.float64)
    dist.all_reduce(processed_tensor, op=dist.ReduceOp.SUM)
    reduced_processed = int(round(float(processed_tensor[0].item())))
    return reduced_losses, reduced_processed

  def _log_training(self, metrics_dict: Dict[str, float]) -> None:
    lr = float(self.optimizer.param_groups[0]["lr"]) if self.optimizer is not None else 0.0
    log_str = (
      f"Epoch [{self.epoch}/{self.cfg.training.num_epochs}] "
      f"Iter {self.iteration} "
      f"LR: {lr:.6f} "
      f"Loss: {metrics_dict.get('total', 0.0):.4f}"
    )
    if "loss_obj" in metrics_dict:
      log_str += f" Obj: {metrics_dict['loss_obj']:.4f}"
    if "loss_cls" in metrics_dict:
      log_str += f" Cls: {metrics_dict['loss_cls']:.4f}"
    if "loss_box" in metrics_dict:
      log_str += f" Box: {metrics_dict['loss_box']:.4f}"
    if "fg_ratio" in metrics_dict:
      log_str += f" FG: {metrics_dict['fg_ratio']:.4f}"
    if "lr_terminal_mcm" in metrics_dict:
      log_str += f" MCM_LR: {metrics_dict['lr_terminal_mcm']:.6g}"
    if "loss_iou" in metrics_dict:
      log_str += f" IoU: {metrics_dict['loss_iou']:.4f}"
    # DETR loss keys
    if "loss_ce" in metrics_dict:
      log_str += f" CE: {metrics_dict['loss_ce']:.4f}"
    if "loss_bbox" in metrics_dict:
      log_str += f" BBox: {metrics_dict['loss_bbox']:.4f}"
    if "loss_giou" in metrics_dict:
      log_str += f" GIoU: {metrics_dict['loss_giou']:.4f}"
    # MCM loss + components (unified; only shown when non-zero)
    mcm_val = metrics_dict.get("loss_mcm", 0.0)
    if mcm_val != 0.0:
      log_str += f" MCM: {mcm_val:.4f}"
      mcm_c = metrics_dict.get("mcm_contrast", 0.0)
      mcm_s = metrics_dict.get("mcm_smooth", 0.0)
      if mcm_c != 0.0 or mcm_s != 0.0:
        log_str += f" (C:{mcm_c:.4f} S:{mcm_s:.4f})"
    # Legacy ECM/SMCM keys (only shown when non-zero and different from MCM)
    ecm_val = metrics_dict.get("loss_ecm", 0.0)
    if ecm_val != 0.0 and ecm_val != mcm_val:
      log_str += f" ECM: {ecm_val:.4f}"
      ecm_c = metrics_dict.get("ecm_contrast", 0.0)
      ecm_s = metrics_dict.get("ecm_smooth", 0.0)
      if ecm_c != 0.0 or ecm_s != 0.0:
        log_str += f" (C:{ecm_c:.4f} S:{ecm_s:.4f})"
    smcm_val = metrics_dict.get("loss_smcm", 0.0)
    if smcm_val != 0.0 and smcm_val != mcm_val:
      log_str += f" SMCM: {smcm_val:.4f}"
      smcm_c = metrics_dict.get("smcm_contrast", 0.0)
      smcm_s = metrics_dict.get("smcm_smooth", 0.0)
      if smcm_c != 0.0 or smcm_s != 0.0:
        log_str += f" (C:{smcm_c:.4f} S:{smcm_s:.4f})"
    # STTP losses (only shown when non-zero)
    fut_val = metrics_dict.get("loss_fut", 0.0)
    if fut_val != 0.0:
      log_str += f" Fut: {fut_val:.4f}"
    tc_val = metrics_dict.get("loss_tc", 0.0)
    if tc_val != 0.0:
      log_str += f" TC: {tc_val:.4f}"
    log_str += f" Data: {metrics_dict.get('data_time', 0.0):.4f}s"
    log_str += f" Step: {metrics_dict.get('step_time', 0.0):.4f}s"
    # GPU memory (logged at every log interval for monitoring)
    if torch.cuda.is_available():
      gpu_alloc = torch.cuda.memory_allocated() / (1024 ** 2)
      gpu_resrv = torch.cuda.memory_reserved() / (1024 ** 2)
      log_str += f" GPU: {gpu_alloc:.0f}/{gpu_resrv:.0f}MiB"
    prof_parts = []
    for key, label in (
      ("prof_prefetch_h2d_ms", "h2d"),
      ("prof_prefetch_voxel_ms", "vox"),
      ("prof_prefetch_cuadrado_hist_ms", "hist"),
      ("prof_prefetch_gpu_aug_ms", "aug"),
      ("prof_mcm_ms", "mcm"),
      ("prof_mcm_estimator_ms", "mcm_est"),
      ("prof_mcm_flow_scale_ms", "mcm_scale"),
      ("prof_mcm_renderer_ms", "mcm_render"),
      ("prof_mcm_loss_ms", "mcm_loss"),
      ("prof_rgb_backbone_ms", "rgb"),
      ("prof_edum_ms", "edum"),
      ("prof_event_backbone_ms", "event"),
      ("prof_cmm_ms", "cmm"),
      ("prof_neck_ms", "neck"),
      ("prof_head_ms", "head"),
      ("prof_backward_ms", "bwd"),
    ):
      if key in metrics_dict:
        prof_parts.append(f"{label}={metrics_dict[key]:.1f}")
    if prof_parts:
      log_str += " Prof(ms): " + " ".join(prof_parts)
    logger.info(log_str)

    if self.writer is not None:
      for k, v in metrics_dict.items():
        self.writer.add_scalar(f"train/{k}", float(v), int(self.iteration))
      self.writer.add_scalar("train/lr", float(lr), int(self.iteration))
      if torch.cuda.is_available():
        self.writer.add_scalar("train/gpu_mem_alloc_mib", torch.cuda.memory_allocated() / (1024 ** 2), int(self.iteration))
        self.writer.add_scalar("train/gpu_mem_reserved_mib", torch.cuda.memory_reserved() / (1024 ** 2), int(self.iteration))

  def _log_epoch(self, losses: Dict[str, float], epoch_time: float) -> None:
    log_str = (
      f"Epoch [{self.epoch}/{self.cfg.training.num_epochs}] completed in {epoch_time:.2f}s "
      f"Avg Loss: {losses.get('total', 0.0):.4f}"
    )

    logger.info(log_str)

    if self.writer is not None:
      for key, value in losses.items():
        self.writer.add_scalar(f"epoch/{key}", float(value), int(self.epoch))

  def _log_validation(
    self,
    metrics: Dict[str, float],
    *,
    writer_prefix: str = "eval",
    display_name: str = "Eval",
  ) -> None:
    if not metrics:
      return
    if "mAP" in metrics and "mAP50" in metrics:
      msg = f"{display_name} - mAP: {metrics['mAP']:.4f}, mAP50: {metrics['mAP50']:.4f}"
      if "ADE" in metrics:
        msg += f", ADE: {metrics['ADE']:.4f}px"
      logger.info(msg)

      # If we have debug metrics, print a compact summary (they're also written to TB/W&B).
      dbg_keys = [
        "dbg_best_iou_mean",
        "dbg_best_iou_median",
        "dbg_recall_iou50",
        "dbg_pred_count_mean",
        "dbg_pred_has_any_frac",
        "dbg_pred_score_max_p95",
      ]
      present = [k for k in dbg_keys if k in metrics]
      if present:
        parts = [f"{k}={float(metrics[k]):.4f}" for k in present]
        logger.info("%s dbg - %s", display_name, ", ".join(parts))
    else:
      logger.info("%s metrics: %s", display_name, metrics)

    if self.writer is not None:
      for key, value in metrics.items():
        self.writer.add_scalar(f"{writer_prefix}/{key}", float(value), int(self.epoch))

  def _get_device(self) -> torch.device:
    if torch.cuda.is_available():
      return torch.device("cuda", int(get_local_rank()))
    return torch.device("cpu")

  def _wrap_ddp_if_needed(self, model: nn.Module) -> nn.Module:
    if int(get_world_size()) <= 1:
      return model

    if not dist.is_available():
      raise RuntimeError("torch.distributed is not available, cannot use DDP")

    # Prefer an already-initialized process group (e.g. when using our launch())
    # but also support torchrun by initializing from env:// if needed.
    if not dist.is_initialized():
      backend = "nccl" if torch.cuda.is_available() else "gloo"
      dist.init_process_group(backend=backend, init_method="env://")

    ddp_kwargs = {"broadcast_buffers": False}
    if self.device.type == "cuda":
      ddp_kwargs["device_ids"] = [int(get_local_rank())]
    return DistributedDataParallel(model, **ddp_kwargs)
    
  @classmethod
  def build_model(cls, cfg):
    """Returns torch.nn.Module"""
    model = build_model(cfg)
    logger = logging.getLogger(__name__)
    logger.info("Model:\n{}".format(model))
    return model
  
  @classmethod
  def build_optimizer(cls, cfg, model):
    """Returns torch.optim.Optimizer"""
    optimizer = build_optimizer(cfg, model)
    logger = logging.getLogger(__name__)
    logger.info("Optimizer:\n{}".format(optimizer))
    return optimizer
  
  @classmethod
  def build_lr_scheduler(cls, cfg, optimizer):
    """Returns scheduler via call to builder functions"""
    return build_lr_scheduler(cfg, optimizer)
  
  @classmethod
  def build_train_loader(cls, cfg):
    """Returns train loader via call to builder function"""
    return build_train_loader_cfg(cfg)

  def build_hooks(self) -> List[HookBase]:
    """Build and return the default hooks for this trainer.

    We largely mirror the proven ordering from the previous implementation:
    - Data time tracking (per-step)
    - LR scheduling (per-epoch)
    - Evaluation (per-epoch, periodic)
    - Checkpointing (per-epoch, rank-0)
    - Optional W&B logging (rank-0)

    Hooks are kept robust to partially-implemented trainer features by enabling
    them conditionally when the required trainer methods/data are available.
    """
    hooks: List[HookBase] = []

    # Track wall time per step.
    hooks.append(DataTimeTracker())

    # LR scheduling is typically epoch-based for our configs.
    if getattr(self, "scheduler", None) is not None:
      hooks.append(LRSchedulerHook())

    hooks.append(FiniteModelStateHook())

    # Periodic evaluation (safe even if validate() returns {} / is a no-op).
    eval_cfg = getattr(self.config, "eval", None)
    eval_interval = int(getattr(eval_cfg, "interval", 0) or 0)
    mon_eval_cfg = getattr(eval_cfg, "mon_eval", None)
    mon_eval_interval = int(getattr(mon_eval_cfg, "interval", 0) or 0) if mon_eval_cfg is not None else 0
    if max(eval_interval, mon_eval_interval) > 0 and hasattr(self, "evaluate"):
      hooks.append(EvaluationHook(eval_interval))

    # Periodic checkpointing (rank-0 only).
    save_interval = int(getattr(getattr(self.config, "eval", None), "save_interval", 0) or 0)
    save_epochs = list(getattr(getattr(self.config, "eval", None), "save_epochs", []) or [])
    if (save_interval > 0 or save_epochs) and is_main_process() and hasattr(self, "save_checkpoint"):
      max_to_keep = 999 if bool(getattr(getattr(self.config, "log", None), "save_all_checkpoints", False)) else 5
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

    # Optional gradient monitoring (useful for MCM experiments).
    if bool(getattr(self.config.training, "log_grad_norm", False)):
      interval = int(getattr(self.config.training, "log_grad_norm_interval", 50))
      hooks.append(GradientMonitorHook(interval=interval))

    # Optional W&B logging.
    wandb_cfg = getattr(self.config, "wandb", None)
    if wandb_cfg is not None and bool(getattr(wandb_cfg, "use_wandb", False)):
      hooks.append(WandbHook(self.config, self.output_dir))

    # Keep non-training cleanup out of trainer loop logic.
    hooks.append(EpochCleanupHook())

    return hooks

  def _collect_rng_state(self) -> Dict[str, Any]:
    state: Dict[str, Any] = {
      "python": random.getstate(),
      "numpy": np.random.get_state(),
      "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
      state["cuda"] = torch.cuda.get_rng_state_all()
    return state

  def _restore_rng_state(self, state: Dict[str, Any]) -> None:
    if not isinstance(state, dict):
      raise TypeError(f"rng_state must be a dict, got {type(state).__name__}")
    strict_resume = bool(getattr(self.cfg.training, "strict_resume", False))

    def _cpu_byte_tensor(value: Any, *, key: str) -> Optional[torch.ByteTensor]:
      if value is None:
        return None
      try:
        if torch.is_tensor(value):
          tensor = value.detach().to(device="cpu", dtype=torch.uint8)
        else:
          tensor = torch.as_tensor(value, dtype=torch.uint8, device="cpu")
      except Exception as exc:
        message = f"Could not restore rng_state[{key!r}] from {type(value).__name__}: {exc}"
        if strict_resume:
          raise TypeError(message) from exc
        logger.warning("%s; skipping this RNG state for non-strict resume.", message)
        return None
      return tensor.contiguous()

    if "python" in state:
      random.setstate(state["python"])
    if "numpy" in state:
      np.random.set_state(state["numpy"])
    if "torch" in state:
      torch_state = _cpu_byte_tensor(state["torch"], key="torch")
      if torch_state is not None:
        torch.set_rng_state(torch_state)
    if "cuda" in state and torch.cuda.is_available():
      cuda_state = state["cuda"]
      if torch.is_tensor(cuda_state):
        cuda_states = [cuda_state]
      else:
        cuda_states = list(cuda_state) if isinstance(cuda_state, (list, tuple)) else []
      restored_cuda_states = [
        tensor
        for index, value in enumerate(cuda_states)
        if (tensor := _cpu_byte_tensor(value, key=f"cuda[{index}]")) is not None
      ]
      if restored_cuda_states:
        torch.cuda.set_rng_state_all(restored_cuda_states)
      elif strict_resume:
        raise TypeError("Strict resume checkpoint has cuda rng_state but no restorable CUDA RNG tensors.")

  def state_dict(self) -> Dict:
    ret: Dict = {
      "epoch": int(self.epoch),
      "iteration": int(self.iteration),
      "model": self.model.state_dict(),
      "optimizer": self.optimizer.state_dict() if self.optimizer is not None else None,
      "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
      "scaler": self.scaler.state_dict() if self.scaler is not None else None,
      "best_metric": float(self.best_metric),
      "rng_state": self._collect_rng_state(),
    }
    sampler = getattr(self.train_loader, "sampler", None)
    if isinstance(sampler, InfiniteSampler):
      ret["sampler_epoch"] = int(getattr(sampler, "_epoch", 0))

    base_state = super().state_dict()
    if "hooks" in base_state:
      ret["hooks"] = base_state["hooks"]
    return ret

  def _align_model_state_prefix(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Align checkpoint key prefixing (module. vs non-module.) with current model."""
    if not isinstance(state_dict, dict) or not state_dict:
      return state_dict

    model_keys = list(self.model.state_dict().keys())
    if not model_keys:
      return state_dict

    model_has_module = any(k.startswith("module.") for k in model_keys)
    ckpt_has_module = any(k.startswith("module.") for k in state_dict.keys())

    if ckpt_has_module and not model_has_module:
      return OrderedDict(
        (k[len("module."):] if k.startswith("module.") else k, v)
        for k, v in state_dict.items()
      )

    if model_has_module and not ckpt_has_module:
      return OrderedDict((f"module.{k}", v) for k, v in state_dict.items())

    return state_dict

  def _infer_resume_epoch(self, loaded_epoch: int, loaded_iteration: int) -> int:
    """Handle older end-of-epoch checkpoints that stored the completed epoch."""
    loaded_epoch = int(loaded_epoch)
    loaded_iteration = int(loaded_iteration)
    if loaded_iteration <= 0:
      return loaded_epoch

    try:
      iters_per_epoch = int(len(self.train_loader))
    except Exception:
      return loaded_epoch

    max_steps_per_epoch = int(getattr(self.config.training, "max_steps_per_epoch", 0) or 0)
    if max_steps_per_epoch > 0:
      iters_per_epoch = min(iters_per_epoch, max_steps_per_epoch)
    if iters_per_epoch <= 0 or (loaded_iteration % iters_per_epoch) != 0:
      return loaded_epoch

    completed_epochs = loaded_iteration // iters_per_epoch
    if completed_epochs > loaded_epoch:
      logger.info(
        "Checkpoint is at an epoch boundary (iteration=%d, %d iters/epoch); resuming from epoch %d",
        loaded_iteration,
        iters_per_epoch,
        completed_epochs,
      )
      return int(completed_epochs)
    return loaded_epoch

  def load_state_dict(self, state_dict: Dict) -> None:
    strict_resume = bool(getattr(self.cfg.training, "strict_resume", False))
    if strict_resume:
      required = ["model", "optimizer", "scheduler", "epoch", "iteration"]
      if self.scaler is not None:
        required.append("scaler")
      missing_required = [key for key in required if key not in state_dict or state_dict.get(key) is None]
      if missing_required:
        raise RuntimeError(f"Strict resume checkpoint missing required fields: {missing_required}")

    has_explicit_epoch = "epoch" in state_dict
    loaded_epoch = int(state_dict.get("epoch", 0))
    self.iteration = int(state_dict.get("iteration", state_dict.get("iter", 0)))
    epoch_override = getattr(self.cfg.training, "resume_epoch_override", None)
    if epoch_override is not None:
      self.epoch = int(epoch_override)
      logger.info(
        "Resume epoch overridden by config: checkpoint_epoch=%d iteration=%d -> epoch=%d",
        loaded_epoch,
        int(self.iteration),
        int(self.epoch),
      )
    else:
      infer_epoch = (not has_explicit_epoch) or bool(getattr(self.cfg.training, "resume_infer_epoch_boundary", False))
      self.epoch = self._infer_resume_epoch(loaded_epoch, int(self.iteration)) if infer_epoch else loaded_epoch
    self.best_metric = float(state_dict.get("best_metric", 0.0))

    if "model" in state_dict:
      model_state = self._align_model_state_prefix(state_dict["model"])
      allow_missing_mcm = bool(getattr(self.cfg.training, "resume_allow_missing_mcm", False))
      if strict_resume and allow_missing_mcm:
        incompatible = self.model.load_state_dict(model_state, strict=False)
        bad_missing = [k for k in incompatible.missing_keys if not self._is_mcm_state_key(k)]
        bad_unexpected = [k for k in incompatible.unexpected_keys if not self._is_mcm_state_key(k)]
        if bad_missing or bad_unexpected:
          raise RuntimeError(
            "Strict resume with resume_allow_missing_mcm only permits MCM key differences; "
            f"bad_missing={bad_missing[:10]} bad_unexpected={bad_unexpected[:10]}"
          )
        logger.info(
          "Strict resume accepted checkpoint with %d intentionally missing live-MCM keys.",
          len(incompatible.missing_keys),
        )
      elif strict_resume:
        self.model.load_state_dict(model_state, strict=True)
      else:
        try:
          self.model.load_state_dict(model_state, strict=True)
        except RuntimeError:
          # Try SMCM key remapping for pre-refactor checkpoints, then fall
          # back to strict=False so STTP-on-detector checkpoints (saved ckpt
          # lacks sttp_head.*) and cross-arch checkpoints load cleanly.
          from smcfnet.model.compat import remap_smcm_keys
          remapped = self._align_model_state_prefix(remap_smcm_keys(model_state))
          incompatible = self.model.load_state_dict(remapped, strict=False)
          if incompatible.missing_keys:
            logger.warning(
              "Resume load: %d missing keys after remap (fresh-init modules like sttp_head are expected). First few: %s",
              len(incompatible.missing_keys), incompatible.missing_keys[:5],
            )
          if incompatible.unexpected_keys:
            logger.warning(
              "Resume load: %d unexpected keys after remap (stale modules from older arch). First few: %s",
              len(incompatible.unexpected_keys), incompatible.unexpected_keys[:5],
            )
    elif strict_resume:
      raise RuntimeError("Strict resume checkpoint is missing model state.")
    if self.optimizer is not None and state_dict.get("optimizer") is not None:
      try:
        self.optimizer.load_state_dict(state_dict["optimizer"])
      except ValueError:
        if strict_resume:
          raise
        # Param groups changed shape (e.g. fusion_dark* -> cmm_dark* across
        # an architecture change), so the saved optimizer state no longer
        # matches the current model. Drop it and start fresh rather than
        # crashing the resume.
        logger.warning(
          "Resume load: optimizer state incompatible with current model "
          "(likely architecture change); skipping optimizer state restore "
          "and starting with fresh optimizer state."
        )
    elif strict_resume and self.optimizer is not None:
      raise RuntimeError("Strict resume checkpoint is missing optimizer state.")
    if self.scheduler is not None and state_dict.get("scheduler") is not None:
      self.scheduler.load_state_dict(state_dict["scheduler"])
    elif strict_resume and self.scheduler is not None:
      raise RuntimeError("Strict resume checkpoint is missing scheduler state.")
    if self.scaler is not None and state_dict.get("scaler") is not None:
      self.scaler.load_state_dict(state_dict["scaler"])
    elif strict_resume and self.scaler is not None:
      raise RuntimeError("Strict resume checkpoint is missing AMP GradScaler state.")

    rng_state = state_dict.get("rng_state", None)
    if rng_state is not None:
      self._restore_rng_state(rng_state)
    elif strict_resume and not bool(getattr(self.cfg.training, "resume_allow_legacy_missing_rng", False)):
      raise RuntimeError("Strict resume checkpoint is missing rng_state.")
    elif strict_resume:
      logger.warning("Strict resume allowed legacy checkpoint without rng_state by config override.")

    sampler_epoch = state_dict.get("sampler_epoch", None)
    sampler = getattr(self.train_loader, "sampler", None)
    if sampler_epoch is not None and isinstance(sampler, InfiniteSampler):
      sampler.set_epoch(int(sampler_epoch))

    if "hooks" in state_dict:
      super().load_state_dict({"iteration": int(self.iteration), "hooks": state_dict.get("hooks", {})})

  def save_checkpoint(self, filename: str) -> None:
    if bool(getattr(self.config.training, "abort_on_nonfinite_model_state", True)):
      bad = []
      total_bad = 0
      for name, value in self.model.state_dict().items():
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
          f"Refusing to save non-finite checkpoint {filename!r} at "
          f"epoch {int(self.epoch) + 1}, iteration {int(self.iteration)}: "
          f"{total_bad} NaN/Inf values. First tensors: {bad}"
        )

    checkpoint = self.state_dict()
    # Store config as plain containers so future weights_only loads remain safe.
    checkpoint["config"] = config_to_dict(self.config)
    save_path = Path(self.output_dir) / str(filename)
    # Write to a temp file and rename atomically so a SIGKILL mid-save (e.g.
    # SLURM preemption hard-killing a rank before it finishes writing) can't
    # leave a truncated/corrupt checkpoint at the final path.
    tmp_path = save_path.with_suffix(save_path.suffix + f".tmp{os.getpid()}")
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, save_path)
    logger.info(f"Saved checkpoint to {save_path}")

  def resume_checkpoint(self, checkpoint_path: str, weights_only: bool = False) -> None:
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    try:
      with _torch_safe_globals_context():
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=bool(weights_only))
    except pickle.UnpicklingError as e:
      if bool(weights_only):
        logger.error(
          "Secure checkpoint loading with weights_only=True failed due to UnpicklingError: %s. "
          "For security reasons, automatic fallback to unsafe pickle loading is disabled. "
          "If you trust this checkpoint and understand the risks, "
          "retry with weights_only=False.",
          e,
        )
        raise RuntimeError(
          "Failed to load checkpoint safely with weights_only=True. "
          "See log for details; explicit weights_only=False is required to "
          "allow unsafe pickle loading."
        ) from e
      else:
        raise
    except TypeError:
      # Older torch versions don't support `weights_only`.
      checkpoint = torch.load(checkpoint_path, map_location=self.device)

    if isinstance(checkpoint, dict) and ("model" in checkpoint or "state_dict" in checkpoint):
      model_state = checkpoint.get("model", checkpoint.get("state_dict"))
    else:
      model_state = checkpoint

    if isinstance(model_state, dict):
      model_state = self._align_model_state_prefix(model_state)

    if weights_only:
      try:
        # Fast path: exact match (e.g., our own checkpoints saved with model.state_dict()).
        self.model.load_state_dict(model_state)
      except RuntimeError as e:
        # Try SMCM key remapping (old smcm.sofe.* -> smcm.estimator.*)
        from smcfnet.model.compat import remap_smcm_keys
        remapped = self._align_model_state_prefix(remap_smcm_keys(model_state))
        try:
          self.model.load_state_dict(remapped)
          logger.info("Loaded checkpoint after SMCM key remapping")
        except RuntimeError:
          # Common footgun: providing an official YOLOX pretrained checkpoint via
          # `log.resume_from` (weights_only) even though module names differ.
          model_for_load = self.model.module if hasattr(self.model, "module") else self.model
          logger.warning(
            "Strict weight loading failed; attempting YOLOX pretrained key-mapped loading. "
            "If you intended to resume training, set `log.resume_from` to a Trainer checkpoint. "
            "If you intended pretrained init, prefer `model.pretrained=True` + `model.pretrained_path`."
          )
          load_yolox_pretrained_dualstream(model_for_load, model_state)

      # Ensure we don't pretend we resumed a full training state.
      self.epoch = 0
      self.iteration = 0
      self.best_metric = 0.0
      logger.info("Loaded pretrained weights; starting fresh (epoch=0, iteration=0)")
      return
    else:
      assert isinstance(checkpoint, dict), "Full resume requires a dict checkpoint"
      self.load_state_dict(checkpoint)

    logger.info(f"Resumed from epoch {self.epoch}, iteration {self.iteration}")

  def _shutdown_dataloader_workers(self, loader: Optional[DataLoader], *, loader_name: str) -> None:
    """Explicitly tear down persistent worker state after validation.

    Validation can keep CPU/pinned-memory buffers alive across epochs when
    persistent workers are enabled. Releasing those workers is a cheap way to
    avoid carrying validation-side memory into the next training epoch.
    """
    if loader is None:
      return

    iterator = getattr(loader, "_iterator", None)
    if iterator is None:
      return

    shutdown = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown):
      try:
        shutdown()
        logger.info("Shut down %s persistent workers after evaluation", loader_name)
      except Exception:
        logger.exception("Failed to shut down %s workers after evaluation", loader_name)
    try:
      loader._iterator = None
    except Exception:
      pass

  def _get_eval_interval(self) -> int:
    eval_cfg = getattr(self.cfg, "eval", None)
    return int(getattr(eval_cfg, "interval", 0) or 0)

  def _get_mon_eval_interval(self) -> int:
    eval_cfg = getattr(self.cfg, "eval", None)
    mon_eval_cfg = getattr(eval_cfg, "mon_eval", None)
    if mon_eval_cfg is None or not bool(getattr(mon_eval_cfg, "enabled", False)):
      return 0
    return int(getattr(mon_eval_cfg, "interval", 0) or 0)

  def _run_validation_loader(
    self,
    *,
    loader: Optional[DataLoader],
    display_name: str,
    writer_prefix: str,
    latest_metrics_attr: str,
    latest_step_attr: str,
    loader_name: str,
  ) -> Dict[str, float]:
    """Run validation on an arbitrary eval loader (rank-0 only)."""
    synchronize()

    if (not is_main_process()) or (loader is None):
      synchronize()
      return {}

    self._release_train_pipeline_for_eval()

    model_for_eval = self.model.module if hasattr(self.model, "module") else self.model
    model_for_eval.eval()
    logger.info("Running %s...", display_name.lower())
    log_memory_stats(f"{writer_prefix}_start", logger)

    evaluator = StreamingMAPAccumulator(num_classes=self.config.model.num_classes)
    traj_evaluator: Optional[TrajectoryMetricsAccumulator] = None
    sttp_enabled = getattr(self.config.model.sttp, "enabled", False)
    if sttp_enabled:
      traj_evaluator = TrajectoryMetricsAccumulator()
    prefetch: Optional[DSECGPUPrefetcher] = None
    compute_val_loss = bool(getattr(getattr(self.cfg, "eval", None), "compute_loss", False))
    val_loss_sums: Dict[str, float] = defaultdict(float)
    val_loss_weight = 0

    def _filter_targets_padded(t: torch.Tensor) -> torch.Tensor:
      if t is None:
        return torch.zeros((0, 5), dtype=torch.float32)
      if not isinstance(t, torch.Tensor):
        t = torch.as_tensor(t)
      if t.numel() == 0:
        return t.reshape(0, 5).to(torch.float32)
      tt = t.to(torch.float32)
      if tt.ndim != 2 or tt.shape[-1] != 5:
        return tt.reshape(-1, 5)
      valid = (tt[:, 3] > 0) & (tt[:, 4] > 0)
      valid = valid & (tt[:, 0] >= 0) & (tt[:, 0] < float(self.cfg.model.num_classes))
      return tt[valid]

    with torch.no_grad():
      ev_h, ev_w = self._event_voxel_resolution()
      bins = int(getattr(self.cfg.data, "event_bins", 5))
      val_seq_names = getattr(self, "_val_flow_sequence_names", None) or self._flow_sequence_names
      val_iwe_seq_names = self._sequence_names_for_loader(loader) if self._iwe_provider is not None else None
      
      traj_evaluator: Optional[TrajectoryMetricsAccumulator] = None
      sttp_enabled = getattr(self.config.model.sttp, "enabled", False)
      if sttp_enabled:
        traj_evaluator = TrajectoryMetricsAccumulator()
      prefetch = build_dsec_prefetcher(
        loader,
        config=self.cfg,
        device=self.device,
        require_targets=True,
        voxel_height=ev_h,
        voxel_width=ev_w,
        flow_provider=self._flow_provider,
        sequence_names=val_seq_names if self._flow_provider is not None else None,
        stem_provider=self._stem_provider,
        # eval-time: use the val-loader's per-sequence names (same list flow uses),
        # NOT the training _stem_sequence_names — else val seq_idx resolves through the
        # train mapping and looks up the wrong sequence (PR #104 omission).
        stem_sequence_names=val_seq_names if self._stem_provider is not None else None,
        iwe_provider=self._iwe_provider,
        iwe_sequence_names=val_iwe_seq_names if self._iwe_provider is not None else None,
      )

      batch_idx = 0
      while True:
        try:
          batch = prefetch.next()
        except StopIteration:
          break

        if "image" not in batch or "targets" not in batch:
          images, voxels, targets = self._prepare_batch(batch, require_targets=True)
          images = images.to(self.device, non_blocking=True)
          voxels = voxels.to(self.device, non_blocking=True)
        else:
          images = batch["image"]
          voxels = batch.get("events_voxel")
          if not isinstance(voxels, torch.Tensor):
            voxels = self._empty_event_voxels_like(images)
          targets = batch.get("targets")

        if targets is None:
          raise ValueError(f"{display_name} requires targets but batch['targets'] is None")

        # Forward raw events plus streaming-SMCM metadata (needed by sparse_event
        # warper and streaming SOFE estimators like firenet, which require
        # t_start_us/t_end_us/t_ref_us to be present even during eval).
        extra_kwargs: dict = self._batch_forward_kwargs(batch)
        if batch.get("precomputed_flow") is not None:
          extra_kwargs["flow"] = batch["precomputed_flow"]
        stem_window = batch.get("stem_window")
        if stem_window is not None:
          extra_kwargs["stem_window"] = stem_window

        val_loss_dict = None
        if compute_val_loss:
          combined = model_for_eval(
            images,
            voxels,
            targets=targets,
            loss_mode=True,
            return_detections=True,
            **extra_kwargs,
          )
          if not (isinstance(combined, tuple) and len(combined) == 2 and isinstance(combined[0], dict)):
            raise RuntimeError(
              "eval.compute_loss=true requires model.forward(..., loss_mode=True, "
              "return_detections=True) to return (loss_dict, detections)."
            )
          val_loss_dict, outputs = combined
          loss_weight = int(targets.shape[0]) if isinstance(targets, torch.Tensor) and targets.ndim >= 1 else 1
          for key, value in val_loss_dict.items():
            if str(key).startswith("_"):
              continue
            if isinstance(value, torch.Tensor):
              if value.numel() != 1:
                continue
              scalar = float(value.detach().float().item())
            else:
              try:
                scalar = float(value)
              except (TypeError, ValueError):
                continue
            if np.isfinite(scalar):
              metric_key = "val_loss_total" if str(key) == "total" else f"val_{key}"
              val_loss_sums[metric_key] += scalar * loss_weight
          val_loss_weight += loss_weight
        else:
          outputs = model_for_eval(images, voxels, **extra_kwargs)

        # Unpack STTP future predictions if returned
        fut_preds = None
        if isinstance(outputs, tuple):
          outputs, fut_preds = outputs

        # Clone pre-NMS detection boxes (cxcywh) before postprocess mutates to xyxy
        det_pre_nms_cxcywh = None
        if fut_preds is not None and isinstance(outputs, torch.Tensor) and outputs.ndim == 3:
          det_pre_nms_cxcywh = outputs[:, :, :4].clone()

        head_type = getattr(self.config.model, "head_type", "yolox")
        anchor_indices_per_image = None
        if head_type == "detr":
          predictions_per_image = postprocess_detr_predictions(
            outputs,
            img_h=int(images.shape[-2]),
            img_w=int(images.shape[-1]),
            num_select=getattr(self.config.model.detr_head, "num_select", 300),
            conf_thre=self.config.eval.conf_threshold,
          )
        else:
          predictions_per_image, anchor_indices_per_image = postprocess_predictions(
            outputs,
            num_classes=self.config.model.num_classes,
            conf_thre=self.config.eval.conf_threshold,
            nms_thre=self.config.eval.nms_threshold,
            return_anchor_indices=True,
          )

        bsz = int(targets.shape[0])
        batch_targets: List[torch.Tensor] = []
        for i in range(bsz):
          tgt_i = _filter_targets_padded(targets[i])
          batch_targets.append(tgt_i.cpu())

        evaluator.update(predictions_per_image, batch_targets)

        # Accumulate trajectory metrics if STTP predictions available
        if (
          traj_evaluator is not None
          and fut_preds is not None
          and det_pre_nms_cxcywh is not None
          and batch.get("future_targets") is not None
        ):
          head = model_for_eval.head
          fut_decoded = decode_sttp_predictions(
            fut_preds,
            head.hw,
            head.strides,
            target_mode=str(getattr(self.cfg.model.sttp, "target_mode", "absolute")),
            current_boxes=det_pre_nms_cxcywh,
            image_width=int(images.shape[-1]),
            image_height=int(images.shape[-2]),
          )
          traj_evaluator.update(
            det_pre_nms_cxcywh,
            fut_decoded,
            predictions_per_image,
            batch_targets,
            batch["future_targets"],
            anchor_indices_per_image=anchor_indices_per_image,
          )

        if (batch_idx + 1) % 10 == 0:
          logger.info("%s: %d/%d batches", display_name, batch_idx + 1, len(loader))
        batch_idx += 1

    if batch_idx == 0:
      logger.warning("Validation skipped mAP calculation because no validation batches were processed.")
      metrics: Dict[str, float] = {}
    else:
      logger.info("Calculating mAP metrics...")
      metrics = evaluator.compute(verbose=self.config.log.verbose_validation)
      if compute_val_loss and val_loss_weight > 0:
        for key, total in sorted(val_loss_sums.items()):
          metrics[key] = float(total) / float(val_loss_weight)

    # Merge trajectory metrics (ADE) if available
    if traj_evaluator is not None:
      traj_metrics = traj_evaluator.compute()
      if traj_metrics:
        metrics.update(traj_metrics)
        logger.info("Trajectory metrics: ADE=%.4fpx", traj_metrics["ADE"])

    self._log_validation(metrics, writer_prefix=writer_prefix, display_name=display_name)
    setattr(self, latest_metrics_attr, dict(metrics))
    setattr(self, latest_step_attr, int(self.iteration))

    # Explicitly release validation-only references to reduce memory carryover
    # into the next training epoch.
    cleanup_after_eval = bool(getattr(getattr(self.cfg, "eval", None), "cleanup_after_eval", True))
    empty_cache_after_eval = bool(getattr(getattr(self.cfg, "eval", None), "empty_cache_after_eval", True))
    shutdown_val_workers = bool(getattr(getattr(self.cfg, "eval", None), "shutdown_val_loader_workers", True))
    if cleanup_after_eval:
      evaluator = None
      traj_evaluator = None
      prefetch = None
      if shutdown_val_workers:
        self._shutdown_dataloader_workers(self.val_loader, loader_name="validation loader")
      gc.collect()
      if torch.cuda.is_available() and empty_cache_after_eval:
        torch.cuda.empty_cache()
      try:
        import ctypes as _ct
        _ct.CDLL("libc.so.6").malloc_trim(0)
      except Exception:
        pass

    synchronize()
    return metrics

  def evaluate(self) -> Dict[str, float]:
    """Run eval (rank-0 only). Metrics-only (no visualizations)."""
    return self._run_validation_loader(
      loader=self.eval_loader,
      display_name="Eval",
      writer_prefix="eval",
      latest_metrics_attr="_latest_eval_metrics",
      latest_step_attr="_latest_eval_step",
      loader_name="eval loader",
    )

  def evaluate_mon(self) -> Dict[str, float]:
    """Run lightweight mon-eval on a representative subset."""
    return self._run_validation_loader(
      loader=self.mon_eval_loader,
      display_name="Monitor Validation",
      writer_prefix="mon_eval",
      latest_metrics_attr="_latest_mon_eval_metrics",
      latest_step_attr="_latest_mon_eval_step",
      loader_name="mon-eval loader",
    )

  def validate(self) -> Dict[str, float]:
    """Backward-compatible alias for evaluate()."""
    return self.evaluate()

  def validate_monitor(self) -> Dict[str, float]:
    """Backward-compatible alias for evaluate_mon()."""
    return self.evaluate_mon()

  def visualize(self) -> int:
    """Run a visualization-only inference pass and save PNG overlays (rank-0 only)."""
    import json
    import os
    import secrets

    synchronize()

    if (not is_main_process()) or (self.eval_loader is None):
      synchronize()
      return 0

    model_for_eval = self.model.module if hasattr(self.model, "module") else self.model
    model_for_eval.eval()

    vis_max = int(getattr(getattr(self.cfg, "viz", None), "max_images", 0) or 0)
    per_seq_cap = int(getattr(getattr(self.cfg, "viz", None), "frames_per_sequence", 0) or 0)
    if vis_max <= 0 and per_seq_cap <= 0:
      logger.info("viz-only: nothing to do (viz.max_images <= 0 and viz.frames_per_sequence <= 0)")
      synchronize()
      return 0

    vis_conf = float(getattr(getattr(self.cfg, "viz", None), "conf_threshold", 0.3))
    vis_dirname = str(getattr(getattr(self.cfg, "viz", None), "dir", "eval_vis"))
    random_seq = bool(getattr(getattr(self.cfg, "viz", None), "random_sequence", True))

    # Per-sequence mode overrides random-sequence picking and sets a global
    # ceiling of (cap * num_sequences) frames.
    if per_seq_cap > 0:
      random_seq = False

    # Group each invocation's outputs under eval_vis/<run_tag>/ so multiple
    # viz jobs (different checkpoints, sweeps) don't collide and can be
    # reviewed or deleted independently.
    run_tag = os.environ.get("VIZ_RUN_TAG") or time.strftime("run_%Y%m%d_%H%M%S")
    vis_dir = Path(self.output_dir) / vis_dirname / run_tag
    vis_dir.mkdir(parents=True, exist_ok=True)
    det_dir = vis_dir / "detections"
    det_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = vis_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    sttp_metrics_path = metrics_dir / "sttp_per_track.jsonl"
    sttp_metrics_records: list[dict] = []

    # Val batches emit seq_idx indexing into the val-split per-sequence list
    # (not the train-split list). Use val names when present to avoid picking
    # a seq_idx that never appears in val batches.
    val_seq_names: Optional[List[Optional[str]]] = getattr(self, "_val_flow_sequence_names", None)
    val_iwe_seq_names: Optional[List[Optional[str]]] = (
      self._sequence_names_for_loader(self.val_loader) if self._iwe_provider is not None else None
    )
    pool_len = len(val_seq_names) if val_seq_names else (
      len(self._dsec_per_sequence) if self._dsec_per_sequence is not None else 0
    )

    vis_only_seq_idx: Optional[int] = None
    if random_seq and pool_len > 0:
      vis_only_seq_idx = int(secrets.randbelow(pool_len))
      if val_seq_names is not None and 0 <= vis_only_seq_idx < len(val_seq_names):
        seq_name = val_seq_names[vis_only_seq_idx]
      elif self._dsec_per_sequence is not None and 0 <= vis_only_seq_idx < len(self._dsec_per_sequence):
        seq_name = getattr(self._dsec_per_sequence[vis_only_seq_idx], "sequence", None)
      else:
        seq_name = None
      if seq_name:
        logger.info("viz-only: sampling sequence '%s' (seq_idx=%d)", seq_name, vis_only_seq_idx)
      else:
        logger.info("viz-only: sampling seq_idx=%d", vis_only_seq_idx)

    def _filter_targets_padded(t: torch.Tensor) -> torch.Tensor:
      if t is None:
        return torch.zeros((0, 5), dtype=torch.float32)
      if not isinstance(t, torch.Tensor):
        t = torch.as_tensor(t)
      if t.numel() == 0:
        return t.reshape(0, 5).to(torch.float32)
      tt = t.to(torch.float32)
      if tt.ndim != 2 or tt.shape[-1] != 5:
        return tt.reshape(-1, 5)
      valid = (tt[:, 3] > 0) & (tt[:, 4] > 0)
      valid = valid & (tt[:, 0] >= 0) & (tt[:, 0] < float(self.cfg.model.num_classes))
      return tt[valid]

    num_saved = 0
    flow_gt_missing = 0
    flow_gt_present = 0
    with torch.no_grad():
      ev_h, ev_w = self._event_voxel_resolution()
      bins = int(getattr(self.cfg.data, "event_bins", 5))
      val_seq_names = getattr(self, "_val_flow_sequence_names", None) or self._flow_sequence_names
      val_iwe_seq_names = (
        self._sequence_names_for_loader(self.val_loader) if self._iwe_provider is not None else None
      )
      prefetch = build_dsec_prefetcher(
        self.val_loader,
        config=self.cfg,
        device=self.device,
        require_targets=True,
        voxel_height=ev_h,
        voxel_width=ev_w,
        flow_provider=self._flow_provider,
        sequence_names=val_seq_names if self._flow_provider is not None else None,
        stem_provider=self._stem_provider,
        # eval-time: use the val-loader's per-sequence names (same list flow uses),
        # NOT the training _stem_sequence_names — else val seq_idx resolves through the
        # train mapping and looks up the wrong sequence (PR #104 omission).
        stem_sequence_names=val_seq_names if self._stem_provider is not None else None,
        iwe_provider=self._iwe_provider,
        iwe_sequence_names=val_iwe_seq_names if self._iwe_provider is not None else None,
      )

      # In per-sequence mode, replace the global cap with per-seq accounting.
      per_seq_saved: dict[int, int] = {}
      target_total = per_seq_cap * pool_len if per_seq_cap > 0 else vis_max

      batch_idx = 0
      while num_saved < target_total:
        # Early exit when every sequence has hit its cap.
        if per_seq_cap > 0 and pool_len > 0 and all(
          per_seq_saved.get(s, 0) >= per_seq_cap for s in range(pool_len)
        ):
          break
        try:
          batch = prefetch.next()
        except StopIteration:
          break

        if "image" not in batch or "targets" not in batch:
          images, voxels, targets = self._prepare_batch(batch, require_targets=True)
          images = images.to(self.device, non_blocking=True)
          voxels = voxels.to(self.device, non_blocking=True)
        else:
          images = batch["image"]
          voxels = batch.get("events_voxel")
          if not isinstance(voxels, torch.Tensor):
            voxels = self._empty_event_voxels_like(images)
          targets = batch.get("targets")

        if targets is None:
          raise ValueError("viz-only requires targets but batch['targets'] is None")

        # Forward raw events plus streaming-SMCM metadata (needed by sparse_event
        # warper and streaming SOFE estimators like firenet, which require
        # t_start_us/t_end_us/t_ref_us to be present even during eval).
        extra_kwargs: dict = self._batch_forward_kwargs(batch)
        if batch.get("precomputed_flow") is not None:
          extra_kwargs["flow"] = batch["precomputed_flow"]
        stem_window = batch.get("stem_window")
        if stem_window is not None:
          extra_kwargs["stem_window"] = stem_window

        outputs = model_for_eval(images, voxels, **extra_kwargs)

        # Unpack STTP future predictions if the head is present.
        fut_preds_batch = None
        if isinstance(outputs, tuple):
          outputs, fut_preds_batch = outputs

        # Clone pre-NMS detection cxcywh before postprocess mutates to xyxy.
        det_pre_nms_cxcywh = None
        if fut_preds_batch is not None and isinstance(outputs, torch.Tensor) and outputs.ndim == 3:
          det_pre_nms_cxcywh = outputs[:, :, :4].clone()

        head_type = getattr(self.config.model, "head_type", "yolox")
        anchor_indices_per_image = None
        if head_type == "detr":
          predictions_per_image = postprocess_detr_predictions(
            outputs,
            img_h=int(images.shape[-2]),
            img_w=int(images.shape[-1]),
            num_select=getattr(self.config.model.detr_head, "num_select", 300),
            conf_thre=self.config.eval.conf_threshold,
          )
        else:
          predictions_per_image, anchor_indices_per_image = postprocess_predictions(
            outputs,
            num_classes=self.config.model.num_classes,
            conf_thre=self.config.eval.conf_threshold,
            nms_thre=self.config.eval.nms_threshold,
            return_anchor_indices=True,
          )

        # Decode STTP future predictions once per batch.
        fut_decoded_batch = None
        if fut_preds_batch is not None and det_pre_nms_cxcywh is not None:
          from smcfnet.utils.validation import decode_sttp_predictions
          head = model_for_eval.head
          fut_decoded_batch = decode_sttp_predictions(
            fut_preds_batch,
            head.hw,
            head.strides,
            target_mode=str(getattr(self.cfg.model.sttp, "target_mode", "absolute")),
            current_boxes=det_pre_nms_cxcywh,
            image_width=int(images.shape[-1]),
            image_height=int(images.shape[-2]),
          )

        bsz = int(targets.shape[0])
        for i in range(bsz):
          if num_saved >= target_total:
            break

          seq_idx = batch.get("seq_idx")
          frame_index = batch.get("frame_index")
          t_ref_us = batch.get("t_ref_us")
          seq_val = int(seq_idx[i].item()) if isinstance(seq_idx, torch.Tensor) else int(seq_idx[i])
          if vis_only_seq_idx is not None and seq_val != int(vis_only_seq_idx):
            continue
          # Per-sequence cap: skip this frame if its sequence is already at quota.
          if per_seq_cap > 0 and per_seq_saved.get(seq_val, 0) >= per_seq_cap:
            continue

          frame_val = int(frame_index[i].item()) if isinstance(frame_index, torch.Tensor) else int(frame_index[i])
          tref_val = int(t_ref_us[i].item()) if isinstance(t_ref_us, torch.Tensor) else int(t_ref_us[i])

          seq_name = None
          if val_seq_names is not None and 0 <= seq_val < len(val_seq_names):
            seq_name = val_seq_names[seq_val]
          elif self._dsec_per_sequence is not None and 0 <= seq_val < len(self._dsec_per_sequence):
            seq_name = getattr(self._dsec_per_sequence[seq_val], "sequence", None)

          stem = f"seq{seq_val:02d}_frame{frame_val:06d}_t{tref_val}"
          if seq_name:
            stem = f"{seq_name}_{stem}"
          out_path = det_dir / f"{stem}.png"

          # Resolve GT flow once per frame (shared across mcm/flow_mcm/flow_precomputed
          # panels). When load_flow_gt is requested but GT isn't available for a given
          # frame, we render the no-GT layout instead of skipping — otherwise a broken
          # val-side flow source would silently produce zero outputs for the whole run.
          # Mixed layouts are a minor visual annoyance; zero frames is a hard failure.
          want_flow_gt = bool(getattr(self.cfg.viz, "load_flow_gt", False))
          flow_gt_i: Optional[torch.Tensor] = None
          if want_flow_gt:
            bf = batch.get("flow_gt") if isinstance(batch, dict) else None
            if bf is not None and i < len(bf) and bf[i] is not None:
              flow_gt_i = bf[i]
            else:
              seq_path = None
              if self._dsec_per_sequence is not None and 0 <= seq_val < len(self._dsec_per_sequence):
                meta = getattr(self._dsec_per_sequence[seq_val], "meta", None)
                seq_path = getattr(meta, "seq_path", None) if meta is not None else None
              if seq_path is not None:
                from smcfnet.utils.flow_viz import load_gt_flow_for_frame
                flow_gt_i = load_gt_flow_for_frame(seq_path, tref_val)
            if flow_gt_i is None:
              flow_gt_missing += 1
              if flow_gt_missing == 1:
                logger.warning(
                  "viz-only: load_flow_gt=true but GT flow unavailable for this frame "
                  "(seq_idx=%d, t_ref_us=%d); rendering without GT panel. "
                  "This message will not repeat.",
                  seq_val, tref_val,
                )
            else:
              flow_gt_present += 1

          pred_i = predictions_per_image[i] if i < len(predictions_per_image) else None
          pred_for_vis = pred_i
          if pred_for_vis is not None and isinstance(pred_for_vis, torch.Tensor) and pred_for_vis.numel() > 0:
            conf = (pred_for_vis[:, 4] * pred_for_vis[:, 5])
            pred_for_vis = pred_for_vis[conf >= float(vis_conf)]

          tgt_i = _filter_targets_padded(targets[i])
          img_i = batch["image"][i] if ("image" in batch) else images[i]
          visualize_detections(
            image=img_i,
            predictions=pred_for_vis,
            targets=tgt_i,
            save_path=str(out_path),
            title=f"{stem}",
            conf_threshold=float(vis_conf),
          )

          # MCM flow + IWE debug panel (when MCM is enabled; legacy alias _ecm_viz also works)
          mcm_viz = getattr(model_for_eval, "_mcm_viz", None) or getattr(model_for_eval, "_ecm_viz", None)
          if mcm_viz and mcm_viz.get("flow") is not None:
            from smcfnet.utils.flow_viz import visualize_mcm_debug, visualize_flow
            mcm_dir = vis_dir / "mcm"
            mcm_dir.mkdir(parents=True, exist_ok=True)
            visualize_mcm_debug(
              voxel_before=mcm_viz["input"][i],
              voxel_after=mcm_viz["output"][i],
              flow=mcm_viz["flow"][i],
              save_path=mcm_dir / f"{stem}_mcm.png",
              title=stem,
              flow_gt=flow_gt_i,
            )
            # Standalone MCM flow panel for consistency with flow_precomputed/
            mcm_flow_dir = vis_dir / "flow_mcm"
            mcm_flow_dir.mkdir(parents=True, exist_ok=True)
            visualize_flow(
              flow=mcm_viz["flow"][i],
              save_path=mcm_flow_dir / f"{stem}_flow.png",
              title=f"{stem} (model MCM flow)",
              flow_gt=flow_gt_i,
            )

          # Precomputed flow panel (the flow actually used by STTP, which differs
          # from the model's internal MCM flow for Kristian/MPCM/E-RAFT matched-pair
          # configs).
          precomp_flow = batch.get("precomputed_flow") if isinstance(batch, dict) else None
          if precomp_flow is not None and i < len(precomp_flow):
            from smcfnet.utils.flow_viz import visualize_flow
            precomp_dir = vis_dir / "flow_precomputed"
            precomp_dir.mkdir(parents=True, exist_ok=True)
            visualize_flow(
              flow=precomp_flow[i],
              save_path=precomp_dir / f"{stem}_flow.png",
              title=f"{stem} (precomputed flow used by STTP)",
              flow_gt=flow_gt_i,
            )

          # STTP trajectory panel (current + t+k future boxes for GT and preds).
          if fut_decoded_batch is not None and batch.get("future_targets") is not None:
            from smcfnet.utils.viz_utils import visualize_sttp_trajectories

            fut_i = fut_decoded_batch[i]                # [n_anchors, 4] cxcywh
            pre_nms_i = det_pre_nms_cxcywh[i]           # [n_anchors, 4] cxcywh

            # Map each post-NMS detection to its source anchor. Prefer the exact
            # indices returned by postprocess_predictions; fall back to a reverse-NN
            # cdist match only if they are unavailable.
            det_i = pred_for_vis
            pred_fut_np = None
            det_np = None
            if det_i is not None and isinstance(det_i, torch.Tensor) and det_i.numel() > 0:
              anchor_idx_i = (
                anchor_indices_per_image[i]
                if anchor_indices_per_image is not None and i < len(anchor_indices_per_image)
                else None
              )
              if anchor_idx_i is not None:
                anchor_idx = anchor_idx_i.to(pre_nms_i.device)
              else:
                det_cx = (det_i[:, 0] + det_i[:, 2]) / 2
                det_cy = (det_i[:, 1] + det_i[:, 3]) / 2
                det_w = det_i[:, 2] - det_i[:, 0]
                det_h = det_i[:, 3] - det_i[:, 1]
                det_cxcywh = torch.stack([det_cx, det_cy, det_w, det_h], dim=1).to(pre_nms_i.device)
                dist = torch.cdist(det_cxcywh.float(), pre_nms_i.float(), p=1)
                anchor_idx = dist.argmin(dim=1)
              pred_fut_np = fut_i[anchor_idx].detach().cpu().numpy()
              det_np = det_i.detach().cpu().numpy()

            ft = batch["future_targets"]
            gt_cur_np = ft["current_boxes"][i].detach().cpu().numpy()
            gt_fut_np = ft["boxes"][i].detach().cpu().numpy()
            gt_valid_np = ft["validity"][i].detach().cpu().numpy().astype(bool)

            sttp_dir = vis_dir / "sttp"
            sttp_dir.mkdir(parents=True, exist_ok=True)
            future_steps = int(getattr(self.cfg.model.sttp, "future_steps", 5))
            sttp_matches = visualize_sttp_trajectories(
              image=img_i,
              det_current_xyxy=det_np,
              det_future_cxcywh=pred_fut_np,
              gt_current_cxcywh=gt_cur_np,
              gt_future_cxcywh=gt_fut_np,
              gt_valid=gt_valid_np,
              save_path=str(sttp_dir / f"{stem}_sttp.png"),
              title=f"{stem} (t -> t+{future_steps})",
            )
            if sttp_matches:
              frame_ctx = {
                "seq_idx": seq_val,
                "seq_name": seq_name,
                "frame_index": frame_val,
                "t_ref_us": tref_val,
                "future_steps": future_steps,
                "stem": stem,
              }
              for m in sttp_matches:
                rec = dict(frame_ctx)
                rec.update(m)
                sttp_metrics_records.append(rec)

          num_saved += 1
          per_seq_saved[seq_val] = per_seq_saved.get(seq_val, 0) + 1

        batch_idx += 1
        if batch_idx % 10 == 0:
          logger.info(
            "viz-only: processed %d batches, saved %d/%d images%s",
            batch_idx,
            num_saved,
            target_total,
            f" ({len(per_seq_saved)}/{pool_len} sequences covered)" if per_seq_cap > 0 else "",
          )

    logger.info("viz-only: wrote %d PNGs to %s", num_saved, vis_dir)
    if per_seq_cap > 0:
      missed = [s for s in range(pool_len) if per_seq_saved.get(s, 0) == 0]
      if missed:
        logger.warning("viz-only: %d sequences received no frames: seq_idx=%s", len(missed), missed)
    if flow_gt_missing > 0 or flow_gt_present > 0:
      total = flow_gt_missing + flow_gt_present
      logger.info(
        "viz-only: GT flow coverage %d/%d frames (%.1f%%); %d rendered without GT panel.",
        flow_gt_present, total, 100.0 * flow_gt_present / max(total, 1), flow_gt_missing,
      )

    # Persist STTP per-track metrics + aggregate summary for LLM-assisted review.
    if sttp_metrics_records:
      with sttp_metrics_path.open("w") as fh:
        for rec in sttp_metrics_records:
          fh.write(json.dumps(rec) + "\n")

      def _arr(key: str) -> np.ndarray:
        vals = np.array([r[key] for r in sttp_metrics_records], dtype=np.float64)
        return vals[np.isfinite(vals)]

      def _stats(vals: np.ndarray) -> dict:
        if vals.size == 0:
          return {"count": 0}
        return {
          "count": int(vals.size),
          "mean": float(np.mean(vals)),
          "median": float(np.median(vals)),
          "p10": float(np.percentile(vals, 10)),
          "p90": float(np.percentile(vals, 90)),
          "min": float(np.min(vals)),
          "max": float(np.max(vals)),
        }

      cos_vals = _arr("cosine")
      epe_vals = _arr("epe")
      ratio_vals = _arr("mag_ratio")
      iou_tk_vals = _arr("iou_future")
      seqs_seen = sorted({r["seq_name"] for r in sttp_metrics_records if r.get("seq_name")})
      summary = {
        "run_tag": run_tag,
        "experiment": getattr(self.cfg.log, "experiment_name", None),
        "future_steps": int(getattr(self.cfg.model.sttp, "future_steps", 5))
          if hasattr(self.cfg.model, "sttp") else None,
        "frames_rendered": int(num_saved),
        "tracks_matched": len(sttp_metrics_records),
        "sequences": seqs_seen,
        "cosine": _stats(cos_vals),
        "epe_px": _stats(epe_vals),
        "mag_ratio": _stats(ratio_vals),
        "iou_future": _stats(iou_tk_vals),
        "cosine_buckets": {
          "opposite(<0)": int(np.sum(cos_vals < 0.0)),
          "orthogonal[0,0.5)": int(np.sum((cos_vals >= 0.0) & (cos_vals < 0.5))),
          "aligned[0.5,0.9)": int(np.sum((cos_vals >= 0.5) & (cos_vals < 0.9))),
          "tight[>=0.9]": int(np.sum(cos_vals >= 0.9)),
        },
      }
      with (metrics_dir / "sttp_summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2)
      logger.info(
        "viz-only: STTP metrics -> %s (%d tracks, mean cos %.2f, mean EPE %.2f px)",
        metrics_dir,
        len(sttp_metrics_records),
        summary["cosine"].get("mean", float("nan")),
        summary["epe_px"].get("mean", float("nan")),
      )

    synchronize()
    return int(num_saved)
