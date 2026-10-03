import torch
import torch.nn as nn
import torch.optim as optim
import logging
import copy
import inspect
import itertools
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Callable
from pathlib import Path
from torch.utils.data import DataLoader
import yaml

from smcfnet.config.config import Config
from smcfnet.dataset.dsec import DSECDataset, DSECLightweightMapper, build_dsec_sequence_index_datasets
from smcfnet.dataset.dsec import build_dsec_flow_index_datasets
from smcfnet.dataset.peod import PEODDataset, PEODLightweightMapper, build_peod_sequence_index_datasets
from smcfnet.dataset.dsec.dsec_collate import (
  DSECTrainCollate,
  dsec_events_collate,
  dsec_rgb_train_collate,
  dsec_streaming_train_collate,
  dsec_train_collate,
)
from smcfnet.dataset.dsec.prefetcher import (
  BaseDSECPrefetcher,
  DSECGPUPrefetcher,
  DSECRGBGPUPrefetcher,
  DSECStreamingGPUPrefetcher,
)
from smcfnet.dataset.mapped_dataset import MappedDataset
from smcfnet.dataset.samplers import InfiniteSampler
from torch.utils.data import SequentialSampler
from .pretrained import load_yolox_pretrained_dualstream
from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN
from .yolox import DualStreamYOLOX


def _create_flow_estimator_for_config(name: str, **kwargs):
  """Create a registered estimator with only constructor-supported kwargs.

  The MCM config has a broad shared surface because different OFE families use
  overlapping but non-identical knobs. Keep the registry factory strict for
  direct callers, but make config-driven construction tolerant of irrelevant
  knobs for a selected architecture.
  """
  from .ofe import _ESTIMATOR_REGISTRY, create_flow_estimator

  cls = _ESTIMATOR_REGISTRY.get(name)
  if cls is None:
    return create_flow_estimator(name, **kwargs)

  sig = inspect.signature(cls.__init__)
  if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
    return create_flow_estimator(name, **kwargs)

  accepted = {
    key
    for key, param in sig.parameters.items()
    if key != "self"
    and param.kind
    in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
  }
  filtered = {key: value for key, value in kwargs.items() if key in accepted}
  return create_flow_estimator(name, **filtered)


def _resolve_path_with_project_fallback(path_raw: str | Path) -> Path:
  """Resolve a path, trying workspace-root fallback when relative."""
  path = Path(path_raw)
  if not path.is_absolute() and not path.exists():
    project_root = Path(__file__).resolve().parents[3]
    candidate = (project_root / path).resolve()
    if candidate.exists():
      path = candidate
  return path


def _resolve_dataset_config_path(config: Config, dataset_config_override: Optional[str] = None) -> Path:
  dataset_cfg_raw = str(
    dataset_config_override
    if dataset_config_override is not None
    else getattr(config.data, "dataset_config", "")
  or "").strip()
  if dataset_cfg_raw == "":
    raise ValueError(
      "config.data.dataset_config is empty. "
      "Fix by passing a valid training config or override explicitly with --dataset-config."
    )

  dataset_cfg = _resolve_path_with_project_fallback(dataset_cfg_raw)
  if not dataset_cfg.exists():
    raise FileNotFoundError(
      f"Dataset config not found: {dataset_cfg}. "
      f"(config.data.dataset_config={getattr(config.data, 'dataset_config', None)})"
    )
  if dataset_cfg.is_dir():
    raise IsADirectoryError(
      f"Dataset config path points to a directory: {dataset_cfg}. "
      f"(config.data.dataset_config={getattr(config.data, 'dataset_config', None)})"
    )
  return dataset_cfg


def _dataset_name_from_config_path(dataset_cfg: Path) -> str:
  with dataset_cfg.open("r", encoding="utf-8") as fh:
    cfg = yaml.safe_load(fh)
  if not isinstance(cfg, dict):
    raise ValueError(f"Invalid dataset config, expected dict: {dataset_cfg}")
  dataset = cfg.get("dataset", {})
  if not isinstance(dataset, dict):
    raise ValueError(f"Invalid dataset config, expected dataset mapping: {dataset_cfg}")
  return str(dataset.get("name", "DSEC")).upper()


_RECTIFIED_EVENTS_ORIG_ALIASES = {"orig", "native", "orig-res", "event"}
_RECTIFIED_EVENTS_FULL_ALIASES = {"full", "rgb", "full-res"}


def _has_precomputed_iwe(config: Config) -> bool:
  return bool(str(getattr(config.data, "precomputed_iwe_dir", "") or "").strip())


def _normalize_rectified_events_resolution(config: Config) -> str:
  return str(
    getattr(config.data, "mapper_rectified_events_resolution", "auto") or "auto"
  ).lower()


def _is_orig_rectified_events_resolution(value: str) -> bool:
  return str(value).lower() in _RECTIFIED_EVENTS_ORIG_ALIASES


def _resolve_rectified_events_filename(
  *,
  rectified_event_resolution: str,
  edum_enabled: bool,
) -> str:
  if _is_orig_rectified_events_resolution(rectified_event_resolution):
    return "rectified_events_orig_res.h5"
  if str(rectified_event_resolution).lower() in _RECTIFIED_EVENTS_FULL_ALIASES:
    return "rectified_events.h5"
  if str(rectified_event_resolution).lower() == "auto":
    return "rectified_events_orig_res.h5" if bool(edum_enabled) else "rectified_events.h5"
  raise ValueError(
    "data.mapper_rectified_events_resolution must be one of auto/full/orig, "
    f"got {rectified_event_resolution!r}"
  )


def _should_use_rectified_events(
  *,
  mapper_load_events: bool,
  prefer_distorted: bool,
  image_height: int,
  image_width: int,
  rectified_event_resolution: str,
) -> bool:
  """Return whether the dataset index should select a rectified events HDF5.

  Full-resolution rectified RGB training selects rectified events implicitly.
  Native-size MCM pretraining selects rectified events only when the config
  explicitly requests the native-size rectified file.
  """
  if not bool(mapper_load_events) or bool(prefer_distorted):
    return False
  if int(image_height) == 1080 and int(image_width) == 1440:
    return True
  return _is_orig_rectified_events_resolution(rectified_event_resolution)


def _resolve_mcm_event_resolution(config: Config) -> tuple[int, int]:
  """Resolve (H, W) for MCM sparse warping/losses.

  ``data.image_height`` / ``data.image_width`` describe RGB geometry. MCM should
  use the actual event representation geometry. For DSEC native/orig rectified
  events this is the event-camera resolution, even when RGB images are full-res.
  """
  edum_cfg = getattr(getattr(config, "model", None), "edum", None)
  event_native_h = int(getattr(edum_cfg, "event_native_height", 480))
  event_native_w = int(getattr(edum_cfg, "event_native_width", 640))
  if bool(getattr(edum_cfg, "enabled", False)):
    return (event_native_h, event_native_w)

  rectified_event_resolution = _normalize_rectified_events_resolution(config)
  if _is_orig_rectified_events_resolution(rectified_event_resolution):
    return (event_native_h, event_native_w)

  return (
    int(getattr(config.data, "image_height", 480)),
    int(getattr(config.data, "image_width", 640)),
  )


_SUPERVISED_MCM_LOSS_KWARGS: dict[str, set[str]] = {
  "supervised": {"max_epe", "eps"},
  "supervised_l1": set(),
  "supervised_l1_tid": {"gamma"},
  "angular": set(),
  "supervised_angular": {"alpha", "beta", "max_epe"},
}


def _filter_supervised_mcm_loss_kwargs(loss_type: str, loss_kwargs: dict[str, Any]) -> dict[str, Any]:
  """Drop inherited self-supervised kwargs for pure supervised MCM losses.

  YAML configs are merged recursively in this project, so switching a PEOD
  self-supervised base config from ``event_warping`` to ``supervised`` can leave
  event-warping-only keys such as ``smoothness_mode`` and ``res`` under
  ``model.mcm.loss_kwargs``. Those keys are valid for the parent loss but not
  for supervised EPE constructors. Filtering here keeps inherited configs usable
  while preserving strict constructor checks for unrelated loss types.
  """
  allowed = _SUPERVISED_MCM_LOSS_KWARGS.get(str(loss_type))
  if allowed is None:
    return loss_kwargs
  filtered = {k: v for k, v in loss_kwargs.items() if k in allowed}
  dropped = sorted(set(loss_kwargs) - set(filtered))
  if dropped:
    logging.getLogger(__name__).warning(
      "Dropping MCM loss_kwargs incompatible with loss_type=%s: %s",
      loss_type,
      dropped,
    )
  return filtered




def _normalize_estimator_representation(config: Config) -> str:
  """Resolve the estimator-side event representation with legacy compatibility."""
  mcm = getattr(config.model, "mcm", None)
  if mcm is None:
    return "voxel"

  explicit = str(getattr(mcm, "estimator_representation", "auto") or "auto").lower()
  if explicit != "auto":
    value = explicit
  elif bool(getattr(mcm, "cuadrado_native_histogram", False)):
    value = "cuadrado_histogram"
  else:
    value = str(getattr(mcm, "input_representation", "voxel") or "voxel").lower()

  aliases = {
    "histogram": "cuadrado_histogram",
    "cuadrado": "cuadrado_histogram",
    "ofev_histogram": "cuadrado_histogram",
    "of_ev_snn": "cuadrado_histogram",
  }
  value = aliases.get(value, value)
  if value not in {"voxel", "cuadrado_histogram", "streaming_events"}:
    raise ValueError(
      f"Unsupported model.mcm.estimator_representation={value!r}; "
      "expected 'voxel', 'cuadrado_histogram', or 'streaming_events'"
    )
  return value


def _normalize_output_representation(config: Config) -> str:
  """Resolve the detector-facing MCM output representation."""
  mcm = getattr(config.model, "mcm", None)
  if mcm is None or not bool(getattr(mcm, "enabled", False)):
    return "voxel"

  explicit = str(getattr(mcm, "output_representation", "auto") or "auto").lower()
  if explicit != "auto":
    value = explicit
  else:
    output_cfg = getattr(mcm, "output", None)
    output_type = str(getattr(output_cfg, "type", "voxel_grid") or "voxel_grid").lower()
    legacy_warper_type = str(getattr(mcm, "warper_type", "sparse_event") or "sparse_event").lower()
    if output_type in {"iwe", "iwe_liu_rgb", "liu_rgb"}:
      value = "iwe"
    elif legacy_warper_type == "voxel":
      value = "voxel"
    else:
      value = "revoxelized"

  aliases = {
    "voxel_grid": "revoxelized",
    "sparse_event": "revoxelized",
    "sparse_revoxelized": "revoxelized",
    "iwe_liu_rgb": "iwe",
    "liu_rgb": "iwe",
  }
  value = aliases.get(value, value)
  if value not in {"voxel", "revoxelized", "iwe"}:
    raise ValueError(
      f"Unsupported model.mcm.output_representation={value!r}; "
      "expected 'voxel', 'revoxelized', or 'iwe'"
    )
  return value


def _mcm_output_type_from_representation(output_representation: str) -> str:
  if output_representation == "iwe":
    return "iwe_liu_rgb"
  return "voxel_grid"


def _mcm_warper_type_from_representation(output_representation: str, legacy_warper_type: str) -> str:
  if output_representation == "voxel":
    return "voxel"
  if output_representation in {"revoxelized", "iwe"}:
    return "sparse_event"
  return legacy_warper_type


def _build_mcm(config: Config):
  """Build a MotionCompensationModule from config.

  Reads architecture, loss, and neuron settings from ``config.model.mcm``
  (an ``MCMConfig`` dataclass). Handles both the lightweight ECM estimator
  and the heavier recurrent / spiking estimators (``st_flownet``,
  ``cnn_baseline``, ``cuadrado_unet``, ``sofe``, etc.).
  """
  from .losses import create_mc_loss
  from .event_outputs import create_event_output_renderer
  from .mcm import (
    MotionCompensationModule,
    EventWarpingModule,
    StreamingMotionCompensationModule,
  )

  mcm = config.model.mcm
  arch = str(mcm.architecture).lower()
  estimator_representation = _normalize_estimator_representation(config)
  input_representation = estimator_representation  # legacy local name used by streaming branch
  ev_h, ev_w = _resolve_mcm_event_resolution(config)
  output_cfg = getattr(mcm, "output", None)
  output_representation = _normalize_output_representation(config)
  output_type = _mcm_output_type_from_representation(output_representation)
  output_voxel_bins = getattr(output_cfg, "voxel_bins", None)
  if output_voxel_bins is None:
    output_voxel_bins = int(config.data.event_bins)
  legacy_warper_type = str(getattr(mcm, "warper_type", "sparse_event") or "sparse_event").lower()
  warper_type = _mcm_warper_type_from_representation(output_representation, legacy_warper_type)

  if estimator_representation == "streaming_events" and str(arch).lower() != "sofe":
    raise ValueError(
      "model.mcm.estimator_representation='streaming_events' requires "
      "model.mcm.architecture='sofe'."
    )
  cuadrado_arches = {
    "cuadrado_unet",
    "cuadrado_unet_peod",
    "cuadrado_unet_peod_v2",
    "cuadrado_unet_bottom_sep",
    "cuadrado_unet_factorized",
  }
  if estimator_representation == "cuadrado_histogram" and str(arch).lower() not in cuadrado_arches:
    raise ValueError(
      "model.mcm.estimator_representation='cuadrado_histogram' requires "
      "a Cuadrado-family estimator architecture."
    )

  if arch in {"ecm", "ecm_peod"}:
    estimator = _create_flow_estimator_for_config(
      arch,
      num_bins=config.data.event_bins,
      mid_channels=mcm.mid_channels,
      act=config.model.activation,
      mode=mcm.mode,
      image_height=int(ev_h),
      image_width=int(ev_w),
    )
  else:
    neuron_kwargs = dict(mcm.neuron_kwargs or {})
    neuron_kwargs.setdefault("alpha", mcm.lif_alpha)
    neuron_kwargs.setdefault("theta", mcm.lif_theta)

    streaming_kwargs = {}
    if arch == "sofe":
      streaming_kwargs = dict(
        step_time_us=getattr(mcm, "step_time_us", None),
        partition_mode=getattr(mcm, "partition_mode", "fixed_time"),
        events_per_step_target=getattr(mcm, "events_per_step_target", None),
        step_dt_max_us=getattr(mcm, "step_dt_max_us", None),
        step_dt_min_us=getattr(mcm, "step_dt_min_us", None),
        max_steps_per_sample=getattr(mcm, "max_steps_per_sample", None),
        readout_every_n_steps=getattr(mcm, "readout_every_n_steps", None),
        readout_interval_us=getattr(mcm, "readout_interval_us", None),
        decay_tau_us=getattr(mcm, "decay_tau_us", None),
        reset_on_sequence_change=getattr(mcm, "reset_on_sequence_change", True),
        bptt_steps=getattr(mcm, "bptt_steps", None),
        image_height=int(ev_h),
        image_width=int(ev_w),
        activation=getattr(config.model, "activation", "silu"),
        architecture_spec=dict(getattr(mcm, "sofe_architecture", {}) or {}),
        architecture_search_space=dict(getattr(mcm, "sofe_search_space", {}) or {}),
      )

    st_flownet_kwargs = {}
    if arch == "st_flownet":
      st_flownet_kwargs["poly_basis"] = mcm.poly_basis
    elif mcm.poly_basis != 0:
      import logging
      logging.getLogger(__name__).warning(
        "poly_basis=%d is set but architecture=%r does not support it; ignoring",
        mcm.poly_basis, arch,
      )

    cuadrado_kwargs = {}
    if arch in cuadrado_arches:
      cuadrado_kwargs = dict(
        image_height=int(ev_h),
        image_width=int(ev_w),
        kernel_size=int(getattr(mcm, "cuadrado_kernel_size", 7)),
        multiply_factor=float(getattr(mcm, "cuadrado_multiply_factor", 10.0)),
      )

    estimator = _create_flow_estimator_for_config(
      arch,
      num_bins=config.data.event_bins,
      encoder_channels=list(mcm.encoder_channels),
      num_steps=getattr(mcm, "num_steps", None),
      stage_depths=getattr(mcm, "stage_depths", None),
      stage_num_heads=getattr(mcm, "stage_num_heads", None),
      window_size=getattr(mcm, "window_size", 7),
      mlp_ratio=getattr(mcm, "mlp_ratio", 4.0),
      num_res_blocks=mcm.num_res_blocks,
      alpha=mcm.lif_alpha,
      theta=mcm.lif_theta,
      mode=mcm.mode,
      neuron_type=mcm.neuron_type,
      neuron_kwargs=neuron_kwargs,
      **streaming_kwargs,
      **st_flownet_kwargs,
      **cuadrado_kwargs,
    )

  legacy_dense_voxel_warper = (
    warper_type == "voxel"
    and output_type == "voxel_grid"
    and int(output_voxel_bins) == int(config.data.event_bins)
    and float(getattr(output_cfg, "t_ref", 1.0)) == 1.0
  )
  if legacy_dense_voxel_warper:
    warper = EventWarpingModule()
  elif output_type in {"voxel_grid", "iwe_liu_rgb"}:
    warper = create_event_output_renderer(
      output_type=output_type,
      num_bins=int(output_voxel_bins),
      res=(ev_h, ev_w),
      t_ref=float(getattr(output_cfg, "t_ref", 1.0)),
      iwe_channels=str(getattr(output_cfg, "iwe_channels", "liu_rgb")),
      iwe_normalize=str(getattr(output_cfg, "iwe_normalize", "p99")),
      iwe_scale=float(getattr(output_cfg, "iwe_scale", 1.0)),
      iwe_splat=str(getattr(output_cfg, "iwe_splat", "bilinear")),
      iwe_align_to_rgb=str(getattr(output_cfg, "iwe_align_to_rgb", "auto")),
    )
  else:
    raise ValueError(f"Unsupported model.mcm.output.type={output_type!r}")

  loss_kwargs = dict(mcm.loss_kwargs or {})
  if mcm.loss_type in ("cmax", "combined", "cmax_angular"):
    loss_kwargs.setdefault("contrast_weight", mcm.contrast_weight)
    loss_kwargs.setdefault("smooth_weight", mcm.smooth_weight)
    loss_kwargs.setdefault("contrast_clamp", mcm.contrast_clamp)
    loss_kwargs.setdefault("magnitude_weight", mcm.magnitude_weight)
    loss_kwargs.setdefault("use_backward", mcm.use_backward)
  elif mcm.loss_type == "cmax_multiscale":
    loss_kwargs.setdefault("contrast_weight", mcm.contrast_weight)
    loss_kwargs.setdefault("smooth_weight", mcm.smooth_weight)
    loss_kwargs.setdefault("contrast_clamp", mcm.contrast_clamp)
    loss_kwargs.setdefault("magnitude_weight", mcm.magnitude_weight)
    loss_kwargs.setdefault("num_reference_times", mcm.num_reference_times)
  elif mcm.loss_type in (
    "event_warping",
    "event_warping_supervised",
    "event_warping_supervised_scheduled",
    "scheduled_event_warping_supervised",
  ):
    loss_kwargs.setdefault("smooth_weight", mcm.smooth_weight)
    loss_kwargs.setdefault("loss_scaling", mcm.loss_scaling)
    if input_representation == "streaming_events":
      # Streaming SOFE contract: res must match the camera's event resolution
      # and flow_scaling must be 1.0 (flow is already pre-scaled by the MCM).
      # Force these values so a stale DSEC config cannot silently corrupt a
      # PEOD run (or any dataset with a different native resolution).
      existing_res = loss_kwargs.get("res")
      if existing_res is not None and tuple(existing_res) != (ev_h, ev_w):
        logging.getLogger(__name__).warning(
          "loss_kwargs.res=%r overridden to %r for streaming SOFE path; "
          "res must match the resolved event camera resolution.",
          tuple(existing_res), (ev_h, ev_w),
        )
      loss_kwargs["res"] = (ev_h, ev_w)
      existing_fs = loss_kwargs.get("flow_scaling")
      if existing_fs is not None and float(existing_fs) != 1.0:
        logging.getLogger(__name__).warning(
          "loss_kwargs.flow_scaling=%r overridden to 1.0 for streaming SOFE path "
          "(flow is pre-scaled by MCM.flow_scaling before loss evaluation).",
          existing_fs,
        )
      loss_kwargs["flow_scaling"] = 1.0
    else:
      # Non-streaming path: allow user-provided res/flow_scaling, default to
      # the resolved event resolution.
      loss_kwargs.setdefault("res", (ev_h, ev_w))
      # flow_scaling=1.0 in the loss because the MCM already pre-scales flow
      loss_kwargs.setdefault("flow_scaling", 1.0)

  loss_kwargs = _filter_supervised_mcm_loss_kwargs(mcm.loss_type, loss_kwargs)
  loss_fn = create_mc_loss(mcm.loss_type, **loss_kwargs)

  if input_representation == "streaming_events":
    aux_scales_raw = getattr(mcm, "auxiliary_flow_scales", None)
    aux_scales = list(aux_scales_raw) if aux_scales_raw else None
    return StreamingMotionCompensationModule(
      estimator=estimator,
      warper=warper,
      loss_fn=loss_fn,
      flow_scaling=mcm.flow_scaling,
      reset_on_sequence_change=bool(getattr(mcm, "reset_on_sequence_change", True)),
      auxiliary_flow_loss=bool(getattr(mcm, "auxiliary_flow_loss", False)),
      auxiliary_flow_loss_weight=float(getattr(mcm, "auxiliary_flow_loss_weight", 0.25)),
      auxiliary_flow_scales=aux_scales,
      auxiliary_warp_loss=bool(getattr(mcm, "auxiliary_warp_loss", False)),
      auxiliary_warp_loss_weight=float(getattr(mcm, "auxiliary_warp_loss_weight", 0.10)),
    )

  return MotionCompensationModule(
    estimator=estimator,
    warper=warper,
    loss_fn=loss_fn,
    flow_scaling=mcm.flow_scaling,
  )


def resolve_dsec_event_pipeline(config: Config) -> Dict[str, Any]:
  """Resolve mapper, collate, and prefetch settings for the current config."""
  if _has_precomputed_iwe(config):
    return {
      "mapper_load_events": False,
      "input_representation": "precomputed_iwe",
      "event_timestamp_unit": "ms",
      "collate_fn": dsec_rgb_train_collate,
      "prefetcher_type": "voxel",
    }

  mapper_load_events = bool(getattr(config.data, "mapper_load_events", True))
  if _has_precomputed_iwe(config):
    mapper_load_events = False
  if not mapper_load_events:
    return {
      "mapper_load_events": False,
      "input_representation": "rgb",
      "event_timestamp_unit": "ms",
      "collate_fn": dsec_rgb_train_collate,
      "prefetcher_type": "rgb",
    }

  input_representation = _normalize_estimator_representation(config)

  prefetcher_type = str(
    getattr(getattr(config.model, "mcm", None), "prefetcher_type", "auto") or "auto"
  ).lower()
  if prefetcher_type == "auto":
    prefetcher_type = "streaming" if input_representation == "streaming_events" else "voxel"
  if prefetcher_type not in {"voxel", "streaming", "rgb"}:
    raise ValueError(
      f"Unsupported model.mcm.prefetcher_type={prefetcher_type!r}; "
      "expected 'auto', 'voxel', 'streaming', or 'rgb'"
    )

  if input_representation == "streaming_events":
    return {
      "mapper_load_events": True,
      "input_representation": input_representation,
      "event_timestamp_unit": "us",
      "collate_fn": dsec_streaming_train_collate,
      "prefetcher_type": prefetcher_type,
    }

  return {
    "mapper_load_events": True,
    "input_representation": input_representation,
    "event_timestamp_unit": "ms",
    "collate_fn": dsec_train_collate,
    "prefetcher_type": prefetcher_type,
  }


def resolve_training_augmentation_backend(
  config: Config,
  *,
  event_pipeline: Optional[Dict[str, Any]] = None,
) -> str:
  """Resolve the train-time augmentation backend.

  Backends:
  - ``cpu_raw``: current collate-time raw-event augmentation path
  - ``gpu_voxel``: augment after prefetch using image + voxel + targets
  - ``none``: no train-time mosaic/mixup requested
  """
  use_mosaic = bool(getattr(config.training, "use_mosaic", False))
  use_mixup = bool(getattr(config.training, "use_mixup", False))
  if not (use_mosaic or use_mixup):
    return "none"

  backend = str(getattr(config.training, "augmentation_backend", "auto") or "auto").lower()
  if backend not in {"auto", "cpu_raw", "gpu_voxel"}:
    raise ValueError(
      f"Unsupported training.augmentation_backend={backend!r}; "
      "expected 'auto', 'cpu_raw', or 'gpu_voxel'"
    )

  pipeline = event_pipeline if event_pipeline is not None else resolve_dsec_event_pipeline(config)
  mcm_enabled = bool(getattr(getattr(config.model, "mcm", None), "enabled", False)) and not _has_precomputed_iwe(config)
  has_events = bool(pipeline.get("mapper_load_events", False))
  prefetcher_type = str(pipeline.get("prefetcher_type", "rgb"))
  precomputed_iwe_enabled = _has_precomputed_iwe(config)

  if backend == "auto":
    if precomputed_iwe_enabled:
      return "gpu_voxel"
    if (not mcm_enabled) and has_events and prefetcher_type == "voxel":
      return "gpu_voxel"
    return "cpu_raw"

  if backend == "gpu_voxel":
    if mcm_enabled:
      raise ValueError(
        "training.augmentation_backend='gpu_voxel' is not supported when model.mcm.enabled=true; "
        "raw-event training requires the cpu_raw backend."
      )
    if (not precomputed_iwe_enabled) and (not has_events or prefetcher_type != "voxel"):
      raise ValueError(
        "training.augmentation_backend='gpu_voxel' requires a voxel-event detector pipeline "
        "(mapper_load_events=true and prefetcher_type='voxel')."
      )

  if backend == "cpu_raw" and precomputed_iwe_enabled:
    raise ValueError(
      "training.augmentation_backend='cpu_raw' is invalid with data.precomputed_iwe_dir: "
      "raw events are not loaded, and CPU-side image/target augmentation would leave "
      "the cached IWE unaugmented. Use augmentation_backend='auto'/'gpu_voxel' or "
      "disable mosaic/mixup."
    )

  return backend


def build_dsec_prefetcher(
  loader,
  *,
  config: Config,
  device: torch.device,
  require_targets: bool,
  voxel_height: Optional[int] = None,
  voxel_width: Optional[int] = None,
  train_augment: bool = False,
  flow_provider: Optional[Any] = None,
  sequence_names: Optional[Sequence[Optional[str]]] = None,
  stem_provider: Optional[Any] = None,
  stem_sequence_names: Optional[Sequence[Optional[str]]] = None,
  iwe_provider: Optional[Any] = None,
  iwe_sequence_names: Optional[Sequence[Optional[str]]] = None,
) -> BaseDSECPrefetcher:
  """Build the configured DSEC prefetcher for a loader or loader iterator.

  When ``flow_provider`` is supplied alongside ``sequence_names``, the voxel
  prefetcher attaches a per-batch ``precomputed_flow`` tensor consumed by the
  STTP head via ``yolox.py`` (raw events / RGB-only pipelines do not yet
  support precomputed flow injection).
  """
  pipeline = resolve_dsec_event_pipeline(config)
  prefetcher_type = str(pipeline["prefetcher_type"])

  if prefetcher_type == "rgb":
    if flow_provider is not None or iwe_provider is not None:
      raise NotImplementedError(
        "precomputed flow/IWE injection is not implemented for the RGB-only prefetcher"
      )
    return DSECRGBGPUPrefetcher(
      loader,
      device=device,
      require_targets=require_targets,
    )

  if prefetcher_type == "streaming":
    if flow_provider is not None or iwe_provider is not None:
      raise NotImplementedError(
        "precomputed flow/IWE injection is not implemented for the streaming prefetcher"
      )
    return DSECStreamingGPUPrefetcher(
      loader,
      device=device,
      require_targets=require_targets,
    )

  ev_h = int(voxel_height if voxel_height is not None else getattr(config.data, "image_height", 480))
  ev_w = int(voxel_width if voxel_width is not None else getattr(config.data, "image_width", 640))
  img_h = int(getattr(config.data, "image_height", ev_h))
  img_w = int(getattr(config.data, "image_width", ev_w))
  prefer_distorted = bool(getattr(config.data, "mapper_prefer_distorted", True))
  fullres_rectified = (not prefer_distorted) and (img_h == 1080 and img_w == 1440)
  rectified_event_resolution = _normalize_rectified_events_resolution(config)
  edum_enabled = bool(getattr(getattr(config.model, "edum", None), "enabled", False))
  native_rectified_events = (
    fullres_rectified
    and edum_enabled
    and (rectified_event_resolution == "auto" or _is_orig_rectified_events_resolution(rectified_event_resolution))
  ) or (
    (not fullres_rectified)
    and (not prefer_distorted)
    and _is_orig_rectified_events_resolution(rectified_event_resolution)
  )
  source_h = ev_h if native_rectified_events else (img_h if fullres_rectified else ev_h)
  source_w = ev_w if native_rectified_events else (img_w if fullres_rectified else ev_w)
  bins = int(getattr(config.data, "event_bins", 5))
  estimator_representation = _normalize_estimator_representation(config)
  output_representation = _normalize_output_representation(config)
  mcm_enabled = bool(getattr(getattr(config.model, "mcm", None), "enabled", False))
  precomputed_iwe_enabled = _has_precomputed_iwe(config)
  if precomputed_iwe_enabled:
    if iwe_provider is None:
      raise ValueError("data.precomputed_iwe_dir is set but no IWE provider was supplied")
    output_representation = "iwe"
    mcm_enabled = False
    # The live estimator (e.g. cuadrado_histogram) never runs in cache mode:
    # events_voxel comes from the IWE provider lookup below, not from raw
    # events. Neutralize estimator_representation so DSECGPUPrefetcher does
    # not also try to build a live cuadrado histogram from raw events.
    estimator_representation = "voxel"
  output_channels = bins
  if (mcm_enabled and output_representation == "iwe") or precomputed_iwe_enabled:
    output_channels = 3
  if precomputed_iwe_enabled:
    output_channels = int(getattr(config.model, "precomputed_event_channels", output_channels))
  build_voxel = (not mcm_enabled) or estimator_representation == "voxel" or output_representation == "voxel"
  return DSECGPUPrefetcher(
    loader,
    device=device,
    num_bins=bins,
    height=ev_h,
    width=ev_w,
    source_height=source_h,
    source_width=source_w,
    require_targets=require_targets,
    augmentation_backend=resolve_training_augmentation_backend(config, event_pipeline=pipeline)
    if bool(train_augment)
    else "none",
    use_mosaic=bool(getattr(config.training, "use_mosaic", False)) if bool(train_augment) else False,
    use_mixup=bool(getattr(config.training, "use_mixup", False)) if bool(train_augment) else False,
    mosaic_prob=float(getattr(config.training, "mosaic_prob", 1.0)),
    mixup_prob=float(getattr(config.training, "mixup_prob", 0.5)),
    num_classes=int(getattr(config.model, "num_classes", 0) or 0),
    estimator_representation=estimator_representation,
    output_representation=output_representation,
    output_channels=output_channels,
    build_voxel=build_voxel,
    profile_stages=bool(getattr(config.training, "profile_model_stages", False)),
    flow_provider=flow_provider,
    sequence_names=sequence_names,
    stem_provider=stem_provider,
    stem_sequence_names=stem_sequence_names,
    iwe_provider=iwe_provider,
    iwe_sequence_names=iwe_sequence_names,
  )


def build_model(config: Config) -> nn.Module:
  """
  Build the YOLOX model with dual-stream backbone for RGB-Event detection.
  
  Phase 1 Configuration:
  - CSPDarknet dual-stream backbone
  - Simple concatenation fusion
  - Standard YOLOX PAFPN neck
  - Standard YOLOX detection head
  
  Args:
      config: Training configuration
  
  Returns:
      Complete YOLOX model ready for training
  """
  logger = logging.getLogger(__name__)
  model_size = config.model.size
  size_config = {
    'nano': (0.33, 0.25),
    'tiny': (0.33, 0.375),
    's': (0.33, 0.5),
    'm': (0.67, 0.75),
    'l': (1.0, 1.0),
    'x': (1.33, 1.25),
  }
  
  if model_size not in size_config:
    raise ValueError(f"Invalid model size: {model_size}") 
  
  depth_mul, width_mul = size_config[model_size]

  cmm_cfg = getattr(config.model, 'cmm', None)
  cmm_enabled = bool(getattr(cmm_cfg, 'enabled', False) or getattr(config.model, 'use_cmm', False))
  legacy_fusion_conv = bool(getattr(config.model, 'legacy_fusion_conv', False))
  cmm_kwargs = {
    'fusion_layers': list(getattr(cmm_cfg, 'fusion_layers', getattr(config.model, 'fusion_layers', [3, 4, 5]))),
    'd_state': int(getattr(cmm_cfg, 'd_state', 16)),
    'd_conv': int(getattr(cmm_cfg, 'd_conv', 3)),
    'expand': float(getattr(cmm_cfg, 'expand', 2.0)),
    'dt_rank': getattr(cmm_cfg, 'dt_rank', 'auto'),
    'dropout': float(getattr(cmm_cfg, 'dropout', 0.0)),
  }

  precomputed_iwe_enabled = _has_precomputed_iwe(config)
  mcm_instance = None
  event_input_channels = int(config.data.event_bins)
  if precomputed_iwe_enabled:
    event_input_channels = int(getattr(config.model, "precomputed_event_channels", 3))
  elif config.model.mcm.enabled:
    mcm_instance = _build_mcm(config)
    event_input_channels = int(getattr(mcm_instance, "output_channels", event_input_channels) or event_input_channels)

  edum_kwargs = None
  if config.model.edum.enabled:
    edum = config.model.edum
    edum_kwargs = {
      'num_bins': event_input_channels,
      'target_height': config.data.image_height,
      'target_width': config.data.image_width,
      'mid_channels': edum.mid_channels,
      'kernel_size': edum.kernel_size,
      'scale_factor': edum.scale_factor,
      'rgb_in_channels': 3,
      'act': config.model.activation,
    }

  # Build Backbone (Dual-Stream CSPDarknet with Simple Fusion)
  from .backbone import CSPDarknetDualStream
  backbone = CSPDarknetDualStream(
    dep_mul=depth_mul,
    wid_mul=width_mul,
    out_features=("dark3", "dark4", "dark5"),
    depthwise=config.model.depthwise,
    act=config.model.activation,
    event_in_channels=event_input_channels,
    use_mcm=bool(config.model.mcm.enabled and not precomputed_iwe_enabled),
    mcm_instance=mcm_instance,
    use_edum=config.model.edum.enabled,
    edum_kwargs=edum_kwargs,
    use_cmm=cmm_enabled,
    cmm_kwargs=cmm_kwargs,
    legacy_fusion_conv=legacy_fusion_conv,
    event_ablation_zero_input=bool(getattr(config.model, "event_ablation_zero_input", False)),
  )

  in_channels = [256, 512, 1024]

  neck = YOLOPAFPN(
    depth=depth_mul,
    width=width_mul,
    in_channels=in_channels,
    depthwise=config.model.depthwise,
    act=config.model.activation,
  )

  num_classes = config.model.num_classes
  head_type = getattr(config.model, "head_type", "yolox")
  head_label = "DETR" if head_type == "detr" else "YOLOX"

  if head_type == "detr":
    from .detr_head import DETRDecoderHead
    detr_cfg = config.model.detr_head
    head = DETRDecoderHead(
      num_classes=num_classes,
      in_channels=in_channels,
      width=width_mul,
      hidden_dim=detr_cfg.hidden_dim,
      num_queries=detr_cfg.num_queries,
      num_select=detr_cfg.num_select,
      dec_layers=detr_cfg.dec_layers,
      sa_nheads=detr_cfg.sa_nheads,
      ca_nheads=detr_cfg.ca_nheads,
      dec_n_points=detr_cfg.dec_n_points,
      dim_feedforward=detr_cfg.dim_feedforward,
      dropout=detr_cfg.dropout,
      group_detr=detr_cfg.group_detr,
      aux_loss=detr_cfg.aux_loss,
      two_stage=detr_cfg.two_stage,
      bbox_reparam=detr_cfg.bbox_reparam,
      lite_refpoint_refine=detr_cfg.lite_refpoint_refine,
      focal_alpha=detr_cfg.focal_alpha,
      ia_bce_loss=detr_cfg.ia_bce_loss,
      cls_loss_coef=detr_cfg.cls_loss_coef,
      bbox_loss_coef=detr_cfg.bbox_loss_coef,
      giou_loss_coef=detr_cfg.giou_loss_coef,
      set_cost_class=detr_cfg.set_cost_class,
      set_cost_bbox=detr_cfg.set_cost_bbox,
      set_cost_giou=detr_cfg.set_cost_giou,
      img_h=config.data.image_height,
      img_w=config.data.image_width,
    )
  else:
    head = YOLOXHead(
      num_classes=num_classes,
      width=width_mul,
      strides=[8, 16, 32],
      in_channels=in_channels,
      act=config.model.activation,
      depthwise=config.model.depthwise,
    )
    head.initialize_biases(1e-2)

  # --- STTP Head (optional) ---
  sttp_head = None
  sttp_loss = None
  sttp_cfg = config.model.sttp
  if sttp_cfg.enabled:
    from .sttp import STTPHead, STTPLoss
    stem_channels = int(256 * width_mul)
    head_design = str(getattr(sttp_cfg, "head_design", "C")).lower()

    if head_design == "spiking_recurrent":
      from .sttp import SpikingRecurrentSTTPHead
      sttp_head = SpikingRecurrentSTTPHead(
        in_channels=stem_channels,
        num_scales=3,
        hidden_dim=int(sttp_cfg.recurrent_hidden_dim),
        neuron_model=str(sttp_cfg.neuron_model),
        act=config.model.activation,
      )
      design_label = "spiking_recurrent"
    elif head_design == "convgru":
      from .sttp import ConvGRUSTTPHead
      sttp_head = ConvGRUSTTPHead(
        in_channels=stem_channels,
        num_scales=3,
        hidden_dim=int(sttp_cfg.recurrent_hidden_dim),
        act=config.model.activation,
      )
      design_label = "convgru"
    elif head_design == "flow_only":
      from .sttp import FlowOnlySTTPHead
      sttp_head = FlowOnlySTTPHead(
        num_scales=3,
        flow_channels=sttp_cfg.flow_channels,
        flow_proj_dim=sttp_cfg.flow_proj_dim,
        act=config.model.activation,
      )
      design_label = "D (flow-only)"
    else:
      # head_design "C" or "trivial". The trivial swap, if configured, fires
      # later in train.py:_maybe_swap_to_trivial_sttp_head; we still build
      # STTPHead here so checkpoint loading remains symmetric.
      sttp_head = STTPHead(
        in_channels=stem_channels,
        num_scales=3,
        flow_channels=sttp_cfg.flow_channels,
        flow_proj_dim=sttp_cfg.flow_proj_dim,
        act=config.model.activation,
      )
      design_label = "C" if sttp_cfg.flow_channels > 0 else "B"

    sttp_loss = STTPLoss(
      temporal_consistency_weight=sttp_cfg.temporal_consistency_weight,
      target_mode=sttp_cfg.target_mode,
      image_width=config.data.image_width,
      image_height=config.data.image_height,
    )
    logger.info(
      f"  - STTP: enabled (design={design_label}, "
      f"λ_fut={sttp_cfg.loss_weight}, k={sttp_cfg.future_steps}, "
      f"target_mode={sttp_cfg.target_mode})"
    )

  model = DualStreamYOLOX(backbone, neck, head, sttp_head=sttp_head, sttp_loss=sttp_loss)

  # Update auxiliary loss weight buffers (registered in DualStreamYOLOX.__init__)
  model._loss_mcm_weight.fill_(float(config.model.mcm.loss_weight))
  model._mcm_contrast_clamp.fill_(float(config.model.mcm.contrast_clamp))
  if sttp_cfg.enabled:
    model._loss_sttp_weight.fill_(float(sttp_cfg.loss_weight))

  # Load Pretrained Weights (if specified)
  if config.model.pretrained:
    if not getattr(config.model, "pretrained_path", None):
      logger.warning("config.model.pretrained=True but no config.model.pretrained_path was provided.")
    else:
      logger.info(f"Loading pretrained weights from {config.model.pretrained_path}")
      checkpoint = torch.load(config.model.pretrained_path, map_location="cpu")
      load_yolox_pretrained_dualstream(
        model,
        checkpoint,
        load_rgb=True,
        load_event=True,
        load_neck=True,
        load_head=(head_type == "yolox"),  # Don't load YOLOX head weights for DETR
      )

  # ------------------------------------------------------------------
  # MCM pretrained weight loading + optional freeze (modular training)
  # ------------------------------------------------------------------
  mcm_cfg = config.model.mcm
  mcm_pretrained = getattr(mcm_cfg, "pretrained_weights", None)
  mcm_freeze = bool(getattr(mcm_cfg, "freeze", False))

  if mcm_cfg.enabled and mcm_pretrained and not precomputed_iwe_enabled:
    from .mcm.weight_utils import load_mcm_weights_into_full_model
    logger.info(f"Loading pre-trained MCM weights from {mcm_pretrained}")
    load_mcm_weights_into_full_model(model, mcm_pretrained, strict=False)

  if mcm_cfg.enabled and mcm_freeze and not precomputed_iwe_enabled:
    from .mcm.weight_utils import freeze_mcm_params, get_mcm_param_count
    frozen = freeze_mcm_params(model)
    counts = get_mcm_param_count(model)
    logger.info(
      f"Frozen MCM: {frozen} params frozen, "
      f"{counts['trainable']} trainable, {counts['total']} total"
    )

  phase = "Phase 3" if cmm_enabled else (
    "Phase 2" if (config.model.mcm.enabled or config.model.edum.enabled) else "Phase 1"
  )
  logger.info(f"Built {phase} {head_label} model: {model_size.upper()}")
  logger.info(f"  - Backbone: Dual-Stream CSPDarknet (depth={depth_mul}, width={width_mul})")
  if cmm_enabled:
    fusion_desc = "CMM + concat"
  elif legacy_fusion_conv:
    fusion_desc = "Legacy concat + 1x1 conv"
  else:
    fusion_desc = "Elementwise sum"
  logger.info(f"  - Fusion: {fusion_desc}")
  mcm_desc = "disabled"
  if precomputed_iwe_enabled:
    mcm_desc = (
      f"disabled (precomputed_iwe={Path(str(config.data.precomputed_iwe_dir)).name}, "
      f"{event_input_channels}ch)"
    )
  elif config.model.mcm.enabled:
    mcm_desc = f"enabled (arch={config.model.mcm.architecture}, mode={config.model.mcm.mode})"
    mcm_desc += f", output={getattr(mcm_instance, 'output_type', 'legacy')}:{event_input_channels}ch"
    if mcm_freeze:
      mcm_desc += ", FROZEN"
    if mcm_pretrained:
      mcm_desc += f", pretrained={Path(mcm_pretrained).name}"
  logger.info(f"  - MCM: {mcm_desc}")
  logger.info(f"  - EDUM: {'enabled' if config.model.edum.enabled else 'disabled'}")
  logger.info(f"  - CMM: {'enabled' if cmm_enabled else 'disabled'}")
  logger.info(f"  - Classes: {num_classes}")
  return model

def _supports_foreach(optim_cls: type) -> bool:
  try:
    sig = inspect.signature(optim_cls.__init__)
  except (TypeError, ValueError):
    return False
  return "foreach" in sig.parameters

def _expand_param_groups(params: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
  ret = defaultdict(dict)
  for item in params:
    assert "params" in item
    cur_params = {x: y for x, y in item.items() if x != "params" and x != "param_names"}
    if "param_names" in item:
      for param_name, param in zip(item["param_names"], item["params"]):
        ret[param].update({"param_names": [param_name], "params": [param], **cur_params})
    else:
      for param in item["params"]:
        ret[param].update({"params": [param], **cur_params})
  return list(ret.values())

def reduce_param_groups(params: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
  """Merge parameter groups with identical hyperparameters.

  This keeps the number of optimizer param groups small, which improves performance
  especially when `foreach=True` is enabled.
  """
  params = _expand_param_groups(params)
  groups = defaultdict(list)
  for item in params:
    cur_params = tuple((x, y) for x, y in item.items() if x != "params" and x != "param_names")
    groups[cur_params].append({"params": item["params"]})
    if "param_names" in item:
      groups[cur_params][-1]["param_names"] = item["param_names"]

  ret = []
  for param_keys, param_values in groups.items():
    cur = {kv[0]: kv[1] for kv in param_keys}
    cur["params"] = list(itertools.chain.from_iterable([p["params"] for p in param_values]))
    if len(param_values) > 0 and "param_names" in param_values[0]:
      cur["param_names"] = list(
        itertools.chain.from_iterable([p["param_names"] for p in param_values])
      )
    ret.append(cur)
  return ret

def get_default_optimizer_params(
  model: torch.nn.Module,
  *,
  base_lr: float,
  weight_decay: float,
  weight_decay_norm: Optional[float] = 0.0,
  overrides: Optional[Dict[str, Dict[str, float]]] = None,
  lr_factor_func: Optional[Callable[[str], float]] = None,
) -> List[Dict[str, Any]]:
  """Build param groups with sensible defaults.

  Detectron2-style behavior we keep:
  - optionally set different weight decay for normalization layers
  - allow overrides by parameter name (e.g. "head" vs "backbone")
  - merge groups for optimizer efficiency
  """
  if overrides is None:
    overrides = {}

  defaults: Dict[str, Any] = {"lr": float(base_lr), "weight_decay": float(weight_decay)}
  if lr_factor_func is not None and base_lr is None:
    raise ValueError("lr_factor_func requires base_lr")

  norm_module_types = (
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.BatchNorm3d,
    torch.nn.SyncBatchNorm,
    torch.nn.GroupNorm,
    torch.nn.InstanceNorm1d,
    torch.nn.InstanceNorm2d,
    torch.nn.InstanceNorm3d,
    torch.nn.LayerNorm,
    torch.nn.LocalResponseNorm,
  )

  params: List[Dict[str, Any]] = []
  memo: Set[torch.nn.parameter.Parameter] = set()
  for module_name, module in model.named_modules():
    for module_param_name, value in module.named_parameters(recurse=False):
      if not value.requires_grad:
        continue
      if value in memo:
        continue
      memo.add(value)

      hyperparams = copy.copy(defaults)
      if isinstance(module, norm_module_types) and weight_decay_norm is not None:
        hyperparams["weight_decay"] = float(weight_decay_norm)
      if lr_factor_func is not None:
        hyperparams["lr"] *= float(lr_factor_func(f"{module_name}.{module_param_name}"))

      # Allow overrides by exact parameter name (e.g. "bias"), matching detectron2 style.
      hyperparams.update(overrides.get(module_param_name, {}))
      params.append({"params": [value], **hyperparams})
  return reduce_param_groups(params)

def build_optimizer(config: Config, model: torch.nn.Module) -> optim.Optimizer:
  """Build an optimizer for the RGB+Event model.

  Functionality implemented (Detectron2-inspired):
  - param groups with optional zero weight decay for normalization layers
  - reduced/merged param groups for better optimizer performance
  - optional `foreach=True` when supported by the optimizer
  """
  cfg = config.training
  opt_name = str(cfg.optimizer).lower()

  base_lr = float(cfg.learning_rate)
  weight_decay = float(cfg.weight_decay)
  weight_decay_norm = getattr(cfg, "weight_decay_norm", 0.0)

  mcm_lr_factor = float(getattr(config.model.mcm, "lr_factor", 1.0))
  detr_lr_factor = float(getattr(getattr(config.model, "detr_head", None), "lr_factor", 1.0))

  def _lr_factor(param_name: str) -> float:
    """Scale LR for MCM flow estimator and DETR head parameters."""
    if "mcm." in param_name:
      return mcm_lr_factor
    if detr_lr_factor != 1.0 and "head." in param_name:
      return detr_lr_factor
    return 1.0

  need_lr_factor = (mcm_lr_factor != 1.0 or detr_lr_factor != 1.0)
  params = get_default_optimizer_params(
    model,
    base_lr=base_lr,
    weight_decay=weight_decay,
    weight_decay_norm=weight_decay_norm,
    lr_factor_func=_lr_factor if need_lr_factor else None,
  )

  foreach = _supports_foreach(getattr(optim, opt_name.upper(), optim.Adam))

  # Some installations expose foreach but may error depending on device/dtype.
  # We'll pass it optimistically and fall back if needed.
  def _try_make(optim_cls, kwargs: Dict[str, Any]) -> optim.Optimizer:
    try:
      return optim_cls(**kwargs)
    except TypeError:
      kwargs.pop("foreach", None)
      return optim_cls(**kwargs)

  if opt_name == "adam":
    kwargs = {"params": params, "lr": base_lr, "weight_decay": weight_decay}
    if _supports_foreach(optim.Adam):
      kwargs["foreach"] = True
    return _try_make(optim.Adam, kwargs)

  if opt_name == "adamw":
    kwargs = {"params": params, "lr": base_lr, "weight_decay": weight_decay}
    if _supports_foreach(optim.AdamW):
      kwargs["foreach"] = True
    return _try_make(optim.AdamW, kwargs)

  if opt_name == "sgd":
    momentum = float(getattr(cfg, "momentum", 0.9))
    nesterov = bool(getattr(cfg, "nesterov", True))
    kwargs = {
      "params": params,
      "lr": base_lr,
      "momentum": momentum,
      "nesterov": nesterov,
      "weight_decay": weight_decay,
    }
    if _supports_foreach(optim.SGD):
      kwargs["foreach"] = True
    return _try_make(optim.SGD, kwargs)

  raise ValueError(f"Unknown optimizer: {cfg.optimizer}")

def build_lr_scheduler(config: Config, optimizer: optim.Optimizer):
  """Build a LR scheduler from config.

  Notes:
  - By default, this returns an epoch-based scheduler (call `scheduler.step()` once per epoch).
  - Warmup is often handled explicitly in the training loop. If you want warmup to be part of the
    scheduler itself, set `config.training.warmup_use_scheduler=True`.
  """

  logger = logging.getLogger(__name__)
  cfg = config.training

  name = str(getattr(cfg, "lr_scheduler", "cosine") or "cosine").lower()

  lr_gamma = float(getattr(cfg, "lr_gamma", 0.1))
  lr_step_size = int(getattr(cfg, "lr_step_size", 50))
  lr_milestones = list(getattr(cfg, "lr_milestones", [150, 225]))

  # Common warmup settings.
  warmup_epochs = int(getattr(cfg, "warmup_epochs", 0) or 0)
  warmup_lr = float(getattr(cfg, "warmup_lr", 0.0) or 0.0)
  use_warmup_scheduler = bool(getattr(cfg, "warmup_use_scheduler", False))

  total_epochs = int(getattr(cfg, "num_epochs", 0) or 0)
  if total_epochs <= 0:
    total_epochs = 1

  def _build_main_scheduler(num_epochs: int):
    if name == "cosine":
      # Use warmup_lr as a sensible default for eta_min.
      eta_min = getattr(cfg, "lr_eta_min", None)
      if eta_min is None:
        eta_min = warmup_lr
      t_max = getattr(cfg, "lr_t_max", None)
      if t_max is None:
        t_max = num_epochs
      return optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(int(t_max), 1),
        eta_min=float(eta_min),
      )

    if name == "step":
      return optim.lr_scheduler.StepLR(
        optimizer,
        step_size=max(int(lr_step_size), 1),
        gamma=float(lr_gamma),
      )

    if name == "multistep":
      milestones = [int(m) for m in lr_milestones]
      return optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=milestones,
        gamma=float(lr_gamma),
      )

    if name == "exponential":
      return optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=float(lr_gamma),
      )

    if name == "onecycle":
      steps_per_epoch = int(getattr(cfg, "steps_per_epoch", 0) or 0)
      total_steps = int(getattr(cfg, "total_steps", 0) or 0)
      if total_steps <= 0:
        if steps_per_epoch <= 0:
          raise ValueError("OneCycleLR requires training.steps_per_epoch or training.total_steps")
        total_steps = int(num_epochs) * int(steps_per_epoch)

      max_lr = float(getattr(cfg, "learning_rate", 0.0) or 0.0)
      if max_lr <= 0:
        raise ValueError("OneCycleLR requires a positive learning_rate")

      pct_start = float(getattr(cfg, "onecycle_pct_start", 0.3))
      div_factor = float(getattr(cfg, "onecycle_div_factor", 25.0))
      final_div_factor = float(getattr(cfg, "onecycle_final_div_factor", 1e4))
      sched = optim.lr_scheduler.OneCycleLR(
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

    raise ValueError(f"Unknown scheduler: {cfg.lr_scheduler}")

  # Optional: include warmup directly in the scheduler.
  if use_warmup_scheduler and warmup_epochs > 0 and warmup_lr > 0:
    # Linear warmup from warmup_lr -> base_lr over warmup_epochs.
    # torch LinearLR uses factors relative to optimizer base_lrs.
    base_lr = float(optimizer.param_groups[0].get("lr", getattr(cfg, "learning_rate", 0.0)))
    if base_lr <= 0:
      base_lr = 1e-6
    start_factor = float(warmup_lr) / float(base_lr)
    # Clamp to a reasonable positive factor to avoid scheduler errors.
    start_factor = max(float(start_factor), 1e-8)

    warmup_sched = optim.lr_scheduler.LinearLR(
      optimizer,
      start_factor=start_factor,
      end_factor=1.0,
      total_iters=int(warmup_epochs),
    )
    main_epochs = max(int(total_epochs) - int(warmup_epochs), 1)
    main_sched = _build_main_scheduler(main_epochs)

    sched = optim.lr_scheduler.SequentialLR(
      optimizer,
      schedulers=[warmup_sched, main_sched],
      milestones=[int(warmup_epochs)],
    )
    logger.info(
      "Built LR scheduler=%s with warmup (epochs=%d, warmup_lr=%g)",
      name,
      warmup_epochs,
      warmup_lr,
    )
    return sched

  sched = _build_main_scheduler(total_epochs)
  logger.info(
    "Built LR scheduler=%s (total_epochs=%d)",
    name,
    total_epochs,
  )
  return sched

def build_dsec_loader_cfg(
  config: Config,
  *,
  split: str,
  is_train: bool,
  sequences: Optional[List[str]] = None,
  dataset_config_override: Optional[str] = None,
  return_components: bool = False,
):
  """Build a DSEC DataLoader from config.

  This is the shared implementation behind train/test loaders.

  For training, we use an InfiniteSampler (Detectron2-style iteration-based stream).
  For evaluation, we build a finite, deterministic SequentialSampler.
  """
  logger = logging.getLogger(__name__)

  # Resolve dataset config path robustly.
  dataset_cfg_raw = str(
    dataset_config_override
    if dataset_config_override is not None
    else getattr(config.data, "dataset_config", "")
  or "").strip()
  if dataset_cfg_raw == "":
    raise ValueError(
      "config.data.dataset_config is empty. "
      "This commonly happens when the YAML passed via --config was not found/loaded. "
      "Fix by passing a valid training config (e.g. --config src/smcfnet/config/...) "
      "or override explicitly with --dataset-config src/smcfnet/config/dsec/dsec.yaml."
    )

  dsec_cfg = _resolve_path_with_project_fallback(dataset_cfg_raw)

  if not dsec_cfg.exists():
    raise FileNotFoundError(
      f"DSEC dataset config not found: {dsec_cfg}. "
      f"(config.data.dataset_config={config.data.dataset_config})"
    )

  if dsec_cfg.is_dir():
    raise IsADirectoryError(
      f"DSEC dataset config path points to a directory: {dsec_cfg}. "
      f"(config.data.dataset_config={config.data.dataset_config})"
    )

  sequence_overrides: Optional[List[str]] = None
  if sequences is not None and len(sequences) > 0:
    sequence_overrides = [str(s) for s in sequences]

  if sequence_overrides is not None:
    logger.info(
      "Applying explicit sequence override for split='%s': %d sequences",
      str(split),
      len(sequence_overrides),
    )

  # Determine if we should use rectified events/tracks based on image resolution
  # and explicit native-size rectified-event requests.
  image_height = int(getattr(config.data, "image_height", 1080))
  image_width = int(getattr(config.data, "image_width", 1440))
  prefer_distorted = bool(getattr(config.data, "mapper_prefer_distorted", False))

  # Mapper toggles (used to decide whether rectified files are actually needed).
  mapper_load_events = bool(getattr(config.data, "mapper_load_events", True))
  if _has_precomputed_iwe(config):
    mapper_load_events = False
  mapper_load_tracks = bool(getattr(config.data, "mapper_load_tracks", False))
  mapper_build_targets = bool(getattr(config.data, "mapper_build_targets", False))
  mapper_load_flow = bool(getattr(config.data, "mapper_load_flow", False))
  
  fullres_image = (image_height == 1080 and image_width == 1440)
  fullres_rectified = (not prefer_distorted) and fullres_image

  if fullres_image and prefer_distorted and (mapper_load_tracks or mapper_build_targets):
    raise ValueError(
      "Invalid DSEC coordinate configuration: image_height/image_width are 1080x1440 "
      "but mapper_prefer_distorted=True while tracks/targets are enabled. Distorted "
      "images/tracks are in the native event-camera frame (~640x480), so resizing "
      "distorted images to full-res without rectified_tracks.npy corrupts detection "
      "supervision. Use either full-res rectified data "
      "(mapper_prefer_distorted=False) or native distorted resolution "
      "(image_height=480, image_width=640)."
    )

  edum_enabled = bool(getattr(getattr(config.model, "edum", None), "enabled", False))
  rectified_event_resolution = _normalize_rectified_events_resolution(config)
  # Full-res rectified images require rectified events/tracks. EDUM may provide
  # the upsampling path, but its input events must already be rectified to match
  # the high-resolution RGB camera coordinate frame. MCM pretraining can request
  # the same view model at native event tensor size with resolution="orig".
  use_rectified_events = _should_use_rectified_events(
    mapper_load_events=mapper_load_events,
    prefer_distorted=prefer_distorted,
    image_height=image_height,
    image_width=image_width,
    rectified_event_resolution=rectified_event_resolution,
  )
  rectified_events_filename = "rectified_events.h5"
  if use_rectified_events:
    rectified_events_filename = _resolve_rectified_events_filename(
      rectified_event_resolution=rectified_event_resolution,
      edum_enabled=edum_enabled,
    )

  # Prefer rectified *tracks* for full-res rectified images when tracks/targets are used.
  # (Targets are computed directly from track coords; no warping happens in the mapper.)
  use_rectified_tracks = bool(fullres_rectified and (mapper_load_tracks or mapper_build_targets))

  if use_rectified_events and use_rectified_tracks:
    logger.info(
      "Using rectified events and tracks for full-resolution images (%dx%d)",
      image_width, image_height
    )
    logger.info("Using rectified events file: %s", rectified_events_filename)
  elif use_rectified_tracks and not use_rectified_events:
    if mapper_load_events:
      logger.info(
        "Using rectified tracks with native/distorted events for full-resolution images (%dx%d)",
        image_width, image_height
      )
    else:
      logger.info(
        "Using rectified tracks (events disabled) for full-resolution images (%dx%d)",
        image_width, image_height
      )
  elif use_rectified_events and not use_rectified_tracks:
    if fullres_rectified:
      logger.info(
        "Using rectified events (tracks disabled) for full-resolution images (%dx%d)",
        image_width, image_height
      )
    else:
      logger.info(
        "Using native-resolution rectified events (tracks disabled) for event-only training (%dx%d)",
        image_width, image_height
      )
    logger.info("Using rectified events file: %s", rectified_events_filename)

  per_sequence = build_dsec_sequence_index_datasets(
    dsec_yaml_path=dsec_cfg,
    split=str(split),
    sequences=sequence_overrides,
    dt_ms=float(config.data.event_window_ms),
    use_rectified_events=use_rectified_events,
    use_rectified_tracks=use_rectified_tracks,
    rectified_events_filename=rectified_events_filename,
  )

  # When flow GT is requested, build a separate flow-aligned index dataset.
  # The flow datasets use timestamps from the flow GT files instead of image
  # timestamps, so each sample is guaranteed to have a matching flow PNG.
  if mapper_load_flow:
    flow_index_start = int(getattr(config.data, "flow_index_start", 0))
    flow_index_count = getattr(config.data, "flow_index_count", None)
    if flow_index_count is not None:
      flow_index_count = int(flow_index_count)
    flow_index_stride = int(getattr(config.data, "flow_index_stride", 1))
    flow_per_sequence = build_dsec_flow_index_datasets(
      dsec_yaml_path=dsec_cfg,
      split=str(split),
      sequences=sequence_overrides,
      flow_direction="forward",
      use_rectified_events=use_rectified_events,
      rectified_events_filename=rectified_events_filename,
      index_start=flow_index_start,
      index_count=flow_index_count,
      index_stride=flow_index_stride,
    )
    dataset = DSECDataset(flow_per_sequence)
    # The mapper still needs the *event* sequence datasets for HDF5 access.
    mapper_sequence_ref = per_sequence
    logger.info(
      "Built flow-aligned DSEC dataset: %d sequences, %d total flow-GT frames "
      "(index_start=%d, index_count=%s, index_stride=%d)",
      len(flow_per_sequence), len(dataset),
      flow_index_start, str(flow_index_count), flow_index_stride,
    )
  else:
    flow_per_sequence = None
    dataset = DSECDataset(per_sequence)
    mapper_sequence_ref = per_sequence

  event_pipeline = resolve_dsec_event_pipeline(config)

  # Mapper: materialize raw tensors with minimal CPU work.
  mapper = DSECLightweightMapper(
    mapper_sequence_ref,
    load_events=bool(mapper_load_events),
    load_tracks=bool(mapper_load_tracks),
    mmap_tracks=bool(getattr(config.data, "mapper_mmap_tracks", True)),
    load_images=bool(getattr(config.data, "mapper_load_images", False)),
    load_flow=bool(mapper_load_flow),
    image_height=int(getattr(config.data, "image_height", 480)),
    image_width=int(getattr(config.data, "image_width", 640)),
    prefer_distorted=bool(getattr(config.data, "mapper_prefer_distorted", True)),
    build_targets=bool(mapper_build_targets),
    event_timestamp_unit=str(event_pipeline["event_timestamp_unit"]),
    build_future_targets=bool(mapper_build_targets and config.model.sttp.enabled),
    future_steps=int(config.model.sttp.future_steps),
    future_target_mode=str(getattr(config.model.sttp, "target_mode", "absolute")),
  )

  sampler = None
  sampler_seed = None
  if is_train:
    # Sampler: infinite stream of indices, sharded across distributed ranks.
    # Important (Detectron2 note): this sampler is meant to be used as DataLoader(sampler=...)
    # with a map-style dataset and therefore does NOT shard by DataLoader worker id.
    sampler_seed = getattr(config.data, "sampler_seed", None)
    if sampler_seed is None:
      sampler_seed = int(getattr(config, "seed", 0))
    sampler = InfiniteSampler(
      len(dataset),
      shuffle=bool(getattr(config.data, "sampler_shuffle", True)),
      seed=int(sampler_seed),
    )
  logger.info(
    "Built DSEC index dataset with %d sequences, %d total frames. Sequences=%s",
    len(per_sequence),
    len(dataset),
    getattr(dataset, "sequence_names", None),
  )

  logger.info(
    "Initialized loader components: mapper=%s, sampler=%s(shuffle=%s, seed=%s)",
    mapper.__class__.__name__,
    sampler.__class__.__name__,
    getattr(config.data, "sampler_shuffle", True),
    sampler_seed,
  )

  components = {
    "dataset": dataset,
    "per_sequence": per_sequence,
    "mapper": mapper,
    "sampler": sampler,
    "event_pipeline": event_pipeline,
  }

  if return_components:
    return components

  if is_train:
    use_mosaic = bool(getattr(config.training, "use_mosaic", False))
    use_mixup = bool(getattr(config.training, "use_mixup", False))
    if bool(getattr(config.model.sttp, "enabled", False)) and (use_mosaic or use_mixup):
      # Mosaic/mixup rebuild `targets` from multiple source samples but pass
      # `future_targets` through from the anchor only (see _compose_mosaic_sample
      # and _compose_mixup_sample in dsec_collate.py). SimOTA `matched_gt_inds`
      # then overruns `future_targets["boxes"].shape[1]` and triggers a CUDA OOB
      # in _gather_future_targets. Force them off for STTP runs.
      logger.warning(
        "STTP enabled with use_mosaic=%s, use_mixup=%s. Forcing both off "
        "to keep targets/future_targets row alignment.",
        use_mosaic,
        use_mixup,
      )
      use_mosaic = False
      use_mixup = False
    aug_backend = resolve_training_augmentation_backend(config, event_pipeline=event_pipeline)
    num_workers = int(getattr(config.data, "num_workers", 0))
    pin_memory = bool(getattr(config.data, "pin_memory", False))
    prefetch_factor = int(getattr(config.data, "prefetch_factor", 2))
    persistent_workers = bool(getattr(config.data, "persistent_workers", True))
    world_size = max(1, int(getattr(config, "num_gpus", 1)))
    if aug_backend == "gpu_voxel":
      logger.info(
        "Train-time mosaic/mixup backend resolved to gpu_voxel "
        "(world_size=%d, num_workers=%d). Worker-side collate augmentation is disabled.",
        world_size,
        num_workers,
      )
    elif bool(event_pipeline["mapper_load_events"]) and (use_mosaic or use_mixup):
      # Raw-event batch augmentation is primarily sensitive to aggregate worker
      # fan-out under DDP. Local A100 probing showed that the old pinned/
      # prefetched loader settings remain viable at 2 workers on a single GPU,
      # while 4 workers causes a much larger host-memory spike. Warn on truly
      # high-risk settings rather than every non-minimal loader.
      risky_loader = (
        num_workers >= 4
        or (world_size > 1 and num_workers > 2)
      )
      elevated_loader = (
        not risky_loader
        and world_size > 1
        and num_workers > 0
        and (pin_memory or prefetch_factor > 1 or persistent_workers)
      )
      if risky_loader:
        logger.warning(
          "Raw-event train-time mosaic/mixup is host-memory heavy. Current loader settings "
          "(world_size=%d, num_workers=%d, pin_memory=%s, prefetch_factor=%d, "
          "persistent_workers=%s) are likely to cause host-memory pressure or "
          "cgroup OOM under DDP. Prefer num_workers<=2 for multi-GPU diagnostics "
          "and reduce worker-side prefetching first if memory spikes on step 0.",
          world_size,
          num_workers,
          pin_memory,
          prefetch_factor,
          persistent_workers,
        )
      elif elevated_loader:
        logger.info(
          "Raw-event train-time mosaic/mixup with DDP uses elevated host memory "
          "(world_size=%d, num_workers=%d, pin_memory=%s, prefetch_factor=%d, "
          "persistent_workers=%s). This is acceptable if step-0 RSS stays bounded, "
          "but reduce num_workers first if SLURM reports oom_kill events.",
          world_size,
          num_workers,
          pin_memory,
          prefetch_factor,
          persistent_workers,
        )

    collate = DSECTrainCollate(
      use_events=bool(event_pipeline["mapper_load_events"]),
      timestamp_unit=str(event_pipeline["event_timestamp_unit"]),
      use_mosaic=use_mosaic if aug_backend == "cpu_raw" else False,
      use_mixup=use_mixup if aug_backend == "cpu_raw" else False,
      mosaic_prob=float(getattr(config.training, "mosaic_prob", 1.0)),
      mixup_prob=float(getattr(config.training, "mixup_prob", 0.5)),
      image_height=int(getattr(config.data, "image_height", 480)),
      image_width=int(getattr(config.data, "image_width", 640)),
    )
    loader = build_train_loader(
      dataset,
      mapper=mapper,
      sampler=sampler,
      total_batch_size=int(getattr(config.training, "batch_size", 1)),
      num_workers=num_workers,
      collate_fn=collate,
      pin_memory=pin_memory,
      prefetch_factor=prefetch_factor,
      persistent_workers=persistent_workers,
      drop_last=bool(getattr(config.data, "drop_last", True)),
    )
    return loader

  # Eval: finite stream, deterministic order.
  mapped_dataset = MappedDataset(dataset, mapper)
  eval_sampler = SequentialSampler(mapped_dataset)
  collate = event_pipeline["collate_fn"]
  dl_kwargs: Dict[str, Any] = dict(
    dataset=mapped_dataset,
    batch_size=int(getattr(config.training, "batch_size", 1)),
    sampler=eval_sampler,
    num_workers=int(getattr(config.data, "num_workers", 0)),
    pin_memory=bool(getattr(config.data, "pin_memory", False)),
    drop_last=False,
    collate_fn=collate,
  )
  if int(getattr(config.data, "num_workers", 0)) > 0:
    dl_kwargs["persistent_workers"] = bool(getattr(config.data, "persistent_workers", True))
    dl_kwargs["prefetch_factor"] = int(getattr(config.data, "prefetch_factor", 2))
  return DataLoader(**dl_kwargs)


def build_peod_loader_cfg(
  config: Config,
  *,
  split: str,
  is_train: bool,
  sequences: Optional[List[str]] = None,
  dataset_config_override: Optional[str] = None,
  return_components: bool = False,
):
  """Build a PEOD DataLoader from config.

  PEOD reuses the existing packed-event collate and GPU prefetcher contracts,
  but has its own filesystem/index/mapper path. DSEC-specific rectification and
  flow-GT indexing are intentionally not applied here.
  """
  logger = logging.getLogger(__name__)
  peod_cfg = _resolve_dataset_config_path(config, dataset_config_override)

  sequence_overrides: Optional[List[str]] = None
  if sequences is not None and len(sequences) > 0:
    sequence_overrides = [str(s) for s in sequences]
    logger.info(
      "Applying explicit PEOD sequence override for split='%s': %d sequences",
      str(split),
      len(sequence_overrides),
    )

  mapper_load_events = bool(getattr(config.data, "mapper_load_events", True))
  mapper_load_tracks = bool(getattr(config.data, "mapper_load_tracks", False))
  mapper_build_targets = bool(getattr(config.data, "mapper_build_targets", False))
  mapper_load_flow = bool(getattr(config.data, "mapper_load_flow", False))

  per_sequence = build_peod_sequence_index_datasets(
    peod_yaml_path=peod_cfg,
    split=str(split),
    sequences=sequence_overrides,
    dt_ms=float(config.data.event_window_ms),
  )
  dataset = PEODDataset(per_sequence)
  event_pipeline = resolve_dsec_event_pipeline(config)

  sttp_cfg = getattr(config.model, "sttp", None)
  sttp_enabled = bool(getattr(sttp_cfg, "enabled", False))
  future_steps = int(getattr(sttp_cfg, "future_steps", 5)) if sttp_cfg is not None else 5
  future_target_mode = str(getattr(sttp_cfg, "target_mode", "absolute")) if sttp_cfg is not None else "absolute"

  mapper = PEODLightweightMapper(
    per_sequence,
    load_events=bool(mapper_load_events),
    load_tracks=bool(mapper_load_tracks),
    mmap_tracks=bool(getattr(config.data, "mapper_mmap_tracks", True)),
    load_images=bool(getattr(config.data, "mapper_load_images", False)),
    load_flow=bool(mapper_load_flow),
    image_height=int(getattr(config.data, "image_height", 720)),
    image_width=int(getattr(config.data, "image_width", 1280)),
    prefer_distorted=False,
    build_targets=bool(mapper_build_targets),
    event_timestamp_unit=str(event_pipeline["event_timestamp_unit"]),
    build_future_targets=bool(mapper_build_targets and sttp_enabled),
    future_steps=future_steps,
    future_target_mode=future_target_mode,
  )

  sampler = None
  sampler_seed = None
  if is_train:
    sampler_seed = getattr(config.data, "sampler_seed", None)
    if sampler_seed is None:
      sampler_seed = int(getattr(config, "seed", 0))
    sampler = InfiniteSampler(
      len(dataset),
      shuffle=bool(getattr(config.data, "sampler_shuffle", True)),
      seed=int(sampler_seed),
    )

  logger.info(
    "Built PEOD index dataset with %d sequences, %d total frames. Sequences=%s",
    len(per_sequence),
    len(dataset),
    getattr(dataset, "sequence_names", None),
  )
  logger.info(
    "Initialized PEOD loader components: mapper=%s, sampler=%s(shuffle=%s, seed=%s)",
    mapper.__class__.__name__,
    sampler.__class__.__name__ if sampler is not None else None,
    getattr(config.data, "sampler_shuffle", True),
    sampler_seed,
  )

  components = {
    "dataset": dataset,
    "per_sequence": per_sequence,
    "mapper": mapper,
    "sampler": sampler,
    "event_pipeline": event_pipeline,
  }
  if return_components:
    return components

  if is_train:
    use_mosaic = bool(getattr(config.training, "use_mosaic", False))
    use_mixup = bool(getattr(config.training, "use_mixup", False))
    aug_backend = resolve_training_augmentation_backend(config, event_pipeline=event_pipeline)
    num_workers = int(getattr(config.data, "num_workers", 0))
    pin_memory = bool(getattr(config.data, "pin_memory", False))
    prefetch_factor = int(getattr(config.data, "prefetch_factor", 2))
    persistent_workers = bool(getattr(config.data, "persistent_workers", True))
    mcm_cfg = getattr(getattr(config, "model", None), "mcm", None)
    full_res_events = int(getattr(config.data, "image_height", 720)) >= 720 and int(getattr(config.data, "image_width", 1280)) >= 1280

    collate = DSECTrainCollate(
      use_events=bool(event_pipeline["mapper_load_events"]),
      timestamp_unit=str(event_pipeline["event_timestamp_unit"]),
      use_mosaic=use_mosaic if aug_backend == "cpu_raw" else False,
      use_mixup=use_mixup if aug_backend == "cpu_raw" else False,
      mosaic_prob=float(getattr(config.training, "mosaic_prob", 1.0)),
      mixup_prob=float(getattr(config.training, "mixup_prob", 0.5)),
      image_height=int(getattr(config.data, "image_height", 720)),
      image_width=int(getattr(config.data, "image_width", 1280)),
    )
    return build_train_loader(
      dataset,
      mapper=mapper,
      sampler=sampler,
      total_batch_size=int(getattr(config.training, "batch_size", 1)),
      num_workers=num_workers,
      collate_fn=collate,
      pin_memory=pin_memory,
      prefetch_factor=prefetch_factor,
      persistent_workers=persistent_workers,
      drop_last=bool(getattr(config.data, "drop_last", True)),
    )

  mapped_dataset = MappedDataset(dataset, mapper)
  eval_sampler = SequentialSampler(mapped_dataset)
  dl_kwargs: Dict[str, Any] = dict(
    dataset=mapped_dataset,
    batch_size=int(getattr(config.training, "batch_size", 1)),
    sampler=eval_sampler,
    num_workers=int(getattr(config.data, "num_workers", 0)),
    pin_memory=bool(getattr(config.data, "pin_memory", False)),
    drop_last=False,
    collate_fn=event_pipeline["collate_fn"],
  )
  if int(getattr(config.data, "num_workers", 0)) > 0:
    dl_kwargs["persistent_workers"] = bool(getattr(config.data, "persistent_workers", True))
    dl_kwargs["prefetch_factor"] = int(getattr(config.data, "prefetch_factor", 2))
  return DataLoader(**dl_kwargs)


def build_dataset_loader_cfg(
  config: Config,
  *,
  split: str,
  is_train: bool,
  sequences: Optional[List[str]] = None,
  dataset_config_override: Optional[str] = None,
  return_components: bool = False,
):
  dataset_cfg = _resolve_dataset_config_path(config, dataset_config_override)
  dataset_name = _dataset_name_from_config_path(dataset_cfg)
  if dataset_name == "PEOD":
    return build_peod_loader_cfg(
      config,
      split=split,
      is_train=is_train,
      sequences=sequences,
      dataset_config_override=str(dataset_cfg),
      return_components=return_components,
    )
  if dataset_name == "DSEC":
    return build_dsec_loader_cfg(
      config,
      split=split,
      is_train=is_train,
      sequences=sequences,
      dataset_config_override=str(dataset_cfg),
      return_components=return_components,
    )
  raise ValueError(f"Unsupported dataset.name={dataset_name!r} in {dataset_cfg}")


def build_train_loader_cfg(config: Config, *, return_components: bool = False):
  """Backwards-compatible wrapper for the training DataLoader."""
  return build_dataset_loader_cfg(
    config,
    split="train",
    is_train=True,
    return_components=return_components,
  )

def build_eval_loader_cfg(
  config: Config,
  *,
  split: str = "test",
  sequences: Optional[List[str]] = None,
  dataset_config_override: Optional[str] = None,
  return_components: bool = False,
):
  """Build a finite deterministic eval loader with optional sequence overrides."""
  return build_dataset_loader_cfg(
    config,
    split=str(split),
    is_train=False,
    sequences=sequences,
    dataset_config_override=dataset_config_override,
    return_components=return_components,
  )


def build_test_loader_cfg(config: Config, *, return_components: bool = False):
  """Build the official test DataLoader (finite, deterministic)."""
  return build_eval_loader_cfg(
    config,
    split="test",
    return_components=return_components,
  )

def build_train_loader(
  dataset,
  *,
  mapper,
  sampler=None,
  total_batch_size,
  aspect_ratio_grouping=True,
  num_workers=0,
  collate_fn=None,
  pin_memory=False,
  prefetch_factor=2,
  persistent_workers=True,
  drop_last=True,
  **kwargs,
):
  """
  Build a dataloader 
  """
  if collate_fn is None:
    collate_fn = dsec_events_collate

  mapped_dataset = MappedDataset(dataset, mapper)

  if sampler is None:
    # Fallback: still allow a usable loader without explicitly passing a sampler.
    sampler = InfiniteSampler(len(mapped_dataset), shuffle=True, seed=0)

  dl_kwargs: Dict[str, Any] = dict(
    dataset=mapped_dataset,
    batch_size=int(total_batch_size),
    sampler=sampler,
    num_workers=int(num_workers),
    pin_memory=bool(pin_memory),
    drop_last=bool(drop_last),
    collate_fn=collate_fn,
  )

  # Only valid when using multiprocessing workers.
  if int(num_workers) > 0:
    dl_kwargs["persistent_workers"] = bool(persistent_workers)
    dl_kwargs["prefetch_factor"] = int(prefetch_factor)

  return DataLoader(**dl_kwargs)
