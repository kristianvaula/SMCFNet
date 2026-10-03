"""
Configuration system with YAML file support and command-line overrides.
"""
import argparse
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class DataConfig:
    """Dataset configuration."""
    dataset_config: str = "src/smcfnet/config/dsec/dsec.yaml"
    event_bins: int = 5
    event_window_ms: float = 50.0
    image_height: int = 480
    image_width: int = 640
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = True
    drop_last: bool = True
    cache_train: bool = False  # Cache training data in memory
    cache_val: bool = False    # Cache validation data in memory

    # Sampler (iteration-based training)
    sampler_shuffle: bool = True
    sampler_seed: Optional[int] = None  # If None, will default to Config.seed

    # Lightweight mapper (raw materialization only; defer heavy transforms like voxelization)
    mapper_load_events: bool = True
    mapper_load_tracks: bool = False
    mapper_mmap_tracks: bool = True

    # Optional: materialize RGB + per-frame targets in DataLoader workers.
    # Voxelization should still be deferred to a GPU stage (prefetcher) for speed.
    mapper_load_images: bool = False
    mapper_prefer_distorted: bool = True
    mapper_rectified_events_resolution: str = "auto"  # auto | full | orig
    mapper_build_targets: bool = False
    mapper_load_flow: bool = False  # Load optical flow GT (supervised MCM training)
    flow_index_start: int = 0        # First per-sequence DSEC-flow frame index to expose
    flow_index_count: Optional[int] = None  # Optional per-sequence DSEC-flow frame count
    flow_index_stride: int = 1       # Per-sequence DSEC-flow frame stride

    # Precomputed per-frame optical flow (e.g. E-RAFT) that overrides MCM
    # flow at the STTP head via yolox.py's `flow` kwarg. When set, a
    # PrecomputedFlowProvider is built against this directory and each batch
    # gains a `precomputed_flow` tensor shaped (B, 2, image_height, image_width).
    # Layout: <dir>/<sequence_name>.h5 with dataset `flow` of shape
    # (frame_count, 2, H, W). See tools/precompute_eraft_flow.py.
    precomputed_flow_dir: Optional[str] = None
    stem_feature_dir: Optional[str] = None  # per-sequence HDF5 stem-feature cache for the recurrent STTP head
    # Precomputed detector-facing SMCM IWE cache. When set, training resolves
    # the event stream from per-sequence HDF5 files instead of running live SMCM.
    precomputed_iwe_dir: Optional[str] = None
    precomputed_iwe_key: str = "iwe"
    precomputed_iwe_dtype: str = "fp16"  # fp16 | uint8 | float32
    precomputed_iwe_sequences: Optional[list] = None
    precomputed_iwe_exclude_sequences: Optional[list] = None
    live_iwe_reference_dir: Optional[str] = None
    live_iwe_reference_key: str = "iwe"


@dataclass
class MCMOutputConfig:
    """Detector-facing event output rendered by MCM/SMCM."""

    type: str = "voxel_grid"          # voxel_grid | iwe_liu_rgb
    t_ref: float = 1.0

    # Voxel-grid output. None means use data.event_bins at build time.
    voxel_bins: Optional[int] = None

    # Liu-style IWE output.
    iwe_channels: str = "liu_rgb"
    iwe_normalize: str = "p99"        # p99 | none
    iwe_scale: float = 1.0            # set 255.0 for uint8-range Liu rendering
    iwe_splat: str = "bilinear"
    iwe_align_to_rgb: str = "auto"    # auto | none | mapping


@dataclass
class MCMConfig:
    """Motion Compensation Module configuration."""
    enabled: bool = False
    architecture: str = "st_flownet"   # flow estimator registry key
    mode: str = "ann"                  # "ann" (ClipFloor pre-train) or "snn" (LIF fine-tune)
    # New explicit event-representation contract.
    # estimator_representation controls the generated input consumed by the flow estimator.
    # output_representation controls the detector-facing event representation emitted by MCM.
    estimator_representation: str = "auto"  # auto | voxel | cuadrado_histogram | streaming_events
    output_representation: str = "auto"     # auto | voxel | revoxelized | iwe

    # Legacy aliases kept during migration. Prefer estimator_representation/output_representation.
    input_representation: str = "voxel"  # "voxel" or "streaming_events"
    prefetcher_type: str = "auto"        # "auto", "voxel", "streaming", or "rgb"
    cuadrado_native_histogram: bool = False  # legacy alias for estimator_representation='cuadrado_histogram'
    reset_on_sequence_change: bool = True
    bptt_steps: Optional[int] = None

    # Streaming step scheduling knobs (used by SMCM/SOFE paths).
    step_time_us: Optional[int] = None
    partition_mode: str = "fixed_time"   # fixed_time | hybrid
    events_per_step_target: Optional[int] = None
    step_dt_max_us: Optional[int] = None
    step_dt_min_us: Optional[int] = None
    max_steps_per_sample: Optional[int] = None
    readout_every_n_steps: Optional[int] = None
    readout_interval_us: Optional[int] = None
    decay_tau_us: Optional[float] = None

    # SOFE architecture build/search surface.
    # ``sofe_architecture`` is the concrete spec used for model construction.
    # ``sofe_search_space`` can hold array-valued option sets for external
    # search/ablation code to expand into concrete specs.
    sofe_architecture: dict = field(default_factory=dict)
    sofe_search_space: dict = field(default_factory=dict)

    encoder_channels: list = field(default_factory=lambda: [32, 64, 128, 256])
    num_steps: Optional[int] = None
    stage_depths: Optional[list] = None
    stage_num_heads: Optional[list] = None
    window_size: int | list[int] = 7
    mlp_ratio: float = 4.0
    num_res_blocks: int = 2
    contrast_weight: float = 1.0
    smooth_weight: float = 0.001
    smooth_loss_scale: float = 1.0   # extra scale on the raw smoothness term before smooth_weight
    contrast_clamp: float = 10.0      # clamp contrast loss to [-clamp, 0]
    magnitude_weight: float = 0.0     # L2 flow magnitude penalty (self-supervised regularizer)
    use_backward: bool = True          # Bidirectional contrast (forward + backward warping)
    num_reference_times: int = 11      # Multi-timescale CMax: R+1 reference times (only used when loss_type='cmax_multiscale', Paredes-Valles 2023)
    poly_basis: int = 0                # 0 = Conv2d flow generator (default); >=1 = PolynomialFlowHead (MPCM Tab4 L7 uses k=1)
    lif_alpha: float = 0.9
    lif_theta: float = 1.0
    neuron_type: str = "lif"           # neuron registry key (lif, plif)
    neuron_kwargs: Optional[dict] = None
    loss_type: str = "event_warping"   # MC loss registry key (event_warping, event_warping_supervised, cmax, supervised, supervised_l1, supervised_l1_tid, supervised_angular, combined, angular, cmax_angular)
    loss_kwargs: Optional[dict] = None # smoothness_mode: "mean_l1" keeps the current normalized L1 smoothness. "original" restores the original MCFNet ECM
    mid_channels: int = 32             # hidden channels for the ECM flow predictor
    loss_weight: float = 0.1           # MCM loss weight
    lr_factor: float = 1.0             # LR multiplier for MCM flow estimator
    flow_scaling: float = 640.0        # Flow multiplier: 1.0 = raw pixel units, 640 = MCFNet-style
    loss_scaling: bool = True          # Normalize event_warping loss by nonzero pixel count
    warper_type: str = "sparse_event" # Warping strategy: "voxel" or "sparse_event" (MCFNet-style)
    cuadrado_kernel_size: int = 7      # Cuadrado/OF_EV_SNN estimator kernel size
    cuadrado_multiply_factor: float = 10.0  # Cuadrado/OF_EV_SNN pre-IF activation scale
    output: MCMOutputConfig = field(default_factory=MCMOutputConfig)

    # Auxiliary pyramid flow loss (Phase 2, ssnn_net v2 + pyramid_residual readout).
    auxiliary_flow_loss: bool = False
    auxiliary_flow_loss_weight: float = 0.25
    auxiliary_flow_scales: Optional[list] = None  # defaults to ["h16","h8","h4","h2"] when None

    # Auxiliary self-supervised pyramid warping loss (Phase 3).
    # Upsamples coarse pyramid flows to full resolution and evaluates the main
    # event-warping objective at each scale; reuses auxiliary_flow_scales.
    auxiliary_warp_loss: bool = False
    auxiliary_warp_loss_weight: float = 0.10

    # Modular training support
    pretrained_weights: Optional[str] = None  # path to MCM or estimator-only checkpoint for weight loading
    freeze: bool = False                      # freeze MCM params (use with pretrained_weights)

@dataclass
class EDUMConfig:
    """Event Dynamic Upsampling Module configuration."""
    enabled: bool = False
    kernel_size: int = 3    # CARAFE reassembly kernel size
    mid_channels: int = 32  # internal feature channels
    scale_factor: int = 2   # integer CARAFE scale (2x, then bilinear to exact target)
    event_native_height: int = 480  # native event camera resolution
    event_native_width: int = 640


@dataclass
class CMMConfig:
    """Cross-Modal Mamba Fusion (CMM) configuration."""
    enabled: bool = False
    fusion_layers: list = field(default_factory=lambda: [3, 4, 5])
    d_state: int = 16
    d_conv: int = 3
    expand: float = 2.0
    dt_rank: str = "auto"
    dropout: float = 0.0


@dataclass
class STTPConfig:
    """Short-Term Trajectory Prediction head configuration."""
    enabled: bool = False
    flow_channels: int = 0       # 0 = Design B (stem only), 2 = Design C (stem + flow)
    flow_proj_dim: int = 16      # projected flow channels (Design C only)
    loss_weight: float = 0.5     # λ_fut
    temporal_consistency_weight: float = 0.0  # 0.0 = disabled
    future_steps: int = 5        # k in t+k (at 20Hz: k=5 → 250ms)
    freeze_detector: bool = False  # freeze everything except sttp_head
    target_mode: str = "absolute"  # "absolute" cxcywh or normalized "delta"
    # head_design selects the trajectory head architecture (Stage 1 / Stage 2 ablations).
    #   "C"        — current Design C (stem features + flow), trainable
    #   "trivial"  — zero-parameter analytical projection: future = current + future_steps * flow
    #   "flow_only" — Design D (flow-only, no stem), trainable [Stage 2]
    head_design: str = "C"
    # Spiking-recurrent head (head_design == "spiking_recurrent" / "convgru"):
    context_window: int = 1        # K_ctx: past frames in the temporal window (DISTINCT from future_steps)
    recurrent_hidden_dim: int = 64  # H: encoder/recurrent channel width
    neuron_model: str = "plif"      # "plif" or "lif" (spiking_recurrent only)


@dataclass
class DETRHeadConfig:
    """RF-DETR decoder head configuration."""
    hidden_dim: int = 256
    num_queries: int = 300
    num_select: int = 300        # top-k at inference
    dec_layers: int = 3
    sa_nheads: int = 8           # self-attention heads
    ca_nheads: int = 16          # cross-attention heads
    dec_n_points: int = 2        # deformable attention sampling points
    dim_feedforward: int = 2048
    dropout: float = 0.0
    group_detr: int = 13         # query groups during training (1 at inference)
    aux_loss: bool = True        # auxiliary losses at each decoder layer
    two_stage: bool = False      # encoder proposals (start False)
    bbox_reparam: bool = True    # output delta to reference point
    lite_refpoint_refine: bool = True
    focal_alpha: float = 0.25
    ia_bce_loss: bool = True     # intersection-area weighted BCE
    cls_loss_coef: float = 1.0
    bbox_loss_coef: float = 5.0
    giou_loss_coef: float = 2.0
    set_cost_class: float = 2.0  # Hungarian matcher class cost
    set_cost_bbox: float = 5.0   # Hungarian matcher bbox L1 cost
    set_cost_giou: float = 2.0   # Hungarian matcher GIoU cost
    lr_factor: float = 1.0       # LR multiplier for DETR head params


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # Input modality / trainer selection.
    # - "dual": RGB + events (default)
    # - "rgb":  RGB only
    input_mode: str = "dual"

    backbone: str = "cspdarknet"
    size: str = "s"  # nano, tiny, s, m, l, x
    num_classes: int = 8  # DSEC-Det classes
    pretrained: bool = False
    pretrained_path: Optional[str] = None

    # Detection head selection
    head_type: str = "yolox"  # "yolox" or "detr"

    # Model architecture details
    depthwise: bool = False
    activation: str = "silu"

    # Event stream settings
    event_representation: str = "voxel"  # voxel, timestamp, dist
    precomputed_event_channels: int = 3

    # MCM (Motion Compensation Module)
    mcm: MCMConfig = field(default_factory=MCMConfig)

    # EDUM (Event Dynamic Upsampling Module)
    edum: EDUMConfig = field(default_factory=EDUMConfig)

    # CMM (Cross-Modal Mamba Fusion Module)
    cmm: CMMConfig = field(default_factory=CMMConfig)

    # CMM usage flag (kept for backward compatibility with older YAML files).
    # When False, fall back to simple feature concatenation.
    use_cmm: bool = False
    # Legacy RGBE-Concat checkpoint compatibility. Older baselines fused RGB
    # and event features with learned 1x1 conv projections named
    # backbone.fusion_dark{3,4,5}. Keep disabled for current experiments.
    legacy_fusion_conv: bool = False
    # Eval-only diagnostic: zero the detector-facing event tensor after the
    # optional MCM and before EDUM/event backbone. This keeps the RGB path and
    # weights unchanged while measuring detector reliance on event evidence.
    event_ablation_zero_input: bool = False

    # STTP (Short-Term Trajectory Prediction)
    sttp: STTPConfig = field(default_factory=STTPConfig)

    # DETR Head (RF-DETR decoder)
    detr_head: DETRHeadConfig = field(default_factory=DETRHeadConfig)

    fusion_layers: list = field(default_factory=lambda: [3, 4, 5])


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    # Training mode:
    #   "joint"        — full detection + MCM joint training (default)
    #   "mcm_pretrain" — train only the MCM/OFE module (no detection backbone)
    mode: str = "joint"

    batch_size: int = 2
    num_epochs: int = 300
    max_steps_per_epoch: Optional[int] = None
    max_total_steps: Optional[int] = None
    profile_model_stages: bool = False

    # Optimizer
    optimizer: str = "adam"
    learning_rate: float = 5e-4
    # SGD-specific knobs (ignored by Adam/AdamW)
    momentum: float = 0.9
    nesterov: bool = True
    weight_decay: float = 1e-4

    # Learning rate schedule
    lr_scheduler: str = "cosine"  # cosine, step, multistep, exponential, onecycle
    warmup_epochs: int = 5
    warmup_lr: float = 1e-6

    # Scheduler knobs (optional)
    # If True, warmup is implemented via the returned scheduler (SequentialLR).
    # If False (default), warmup is expected to be handled in the training loop.
    warmup_use_scheduler: bool = False

    # Step / MultiStep
    lr_step_size: int = 50
    lr_gamma: float = 0.1
    lr_milestones: list = field(default_factory=lambda: [150, 225])

    # Cosine
    lr_t_max: Optional[int] = None
    lr_eta_min: Optional[float] = None

    # OneCycle
    steps_per_epoch: Optional[int] = None
    total_steps: Optional[int] = None
    onecycle_pct_start: float = 0.3
    onecycle_div_factor: float = 25.0
    onecycle_final_div_factor: float = 1e4

    # Loss weights
    loss_obj: float = 1.0
    loss_cls: float = 1.0
    loss_box: float = 5.0


    # Gradient clipping
    grad_clip: float = 10.0

    # Streaming MCM/SOFE trainer cadence.
    # outer_window: current behavior, one backward over the aggregated window loss.
    # chunk_backward: backward each TBPTT chunk immediately, accumulate grads,
    #                 and keep one optimizer step per outer event window.
    streaming_step_policy: str = "outer_window"

    # Gradient monitoring
    log_grad_norm: bool = False
    log_grad_norm_interval: int = 50

    # Flow-magnitude monitor (MCM training). Logs |flow| stats every N iters
    # and optionally early-aborts when flow stays near zero.
    flow_monitor_interval: int = 0          # 0 disables
    flow_monitor_abort_after_iter: int = 0  # 0 disables abort
    flow_monitor_abort_threshold: float = 1e-3

    # Mixed precision training
    use_amp: bool = True

    # Data augmentation
    use_mosaic: bool = True
    use_mixup: bool = True
    mosaic_prob: float = 1.0
    mixup_prob: float = 0.5
    augmentation_backend: str = "auto"  # auto | cpu_raw | gpu_voxel
    # YOLOX-style "close mosaic" — disable mosaic+mixup for the final N epochs so the model
    # fine-tunes on clean (unaugmented) images. 0 = no closing phase (legacy behavior).
    no_aug_epochs: int = 0
    # Abort training only after this many CONSECUTIVE non-finite losses (transient
    # spikes are skipped + logged by the trainer's _write_metrics guard; persistent
    # non-finite = real divergence). See trainer.py:_write_metrics.
    max_consecutive_nonfinite: int = 50
    # Abort training when this many consecutive steps are skipped due to NaN/Inf
    # loss or gradient norms. Resets to 0 on any successful optimizer step.
    # 0 = disabled. At ~150 iters/min a dead run hits 1000 in under 7 minutes.
    max_consecutive_nan_skips: int = 1000
    # Abort before checkpoint/eval if any model parameter or buffer has become
    # NaN/Inf. This catches BatchNorm running-stat corruption that can happen
    # during a forward pass even when the non-finite optimizer step is skipped.
    abort_on_nonfinite_model_state: bool = True
    # Abort immediately when the detector forward loss is NaN/Inf. Unlike an
    # overflowed backward pass, a bad forward can already have mutated BatchNorm
    # running buffers before the optimizer-step skip logic runs.
    abort_on_nonfinite_forward_loss: bool = False

    # Phase 2.2 terminal-stage integration support. These are opt-in so legacy
    # resume behavior remains unchanged for existing experiments.
    strict_resume: bool = False
    resume_infer_epoch_boundary: bool = False
    resume_epoch_override: Optional[int] = None
    resume_allow_missing_mcm: bool = False
    resume_allow_legacy_missing_rng: bool = False
    unfreeze_mcm_after_resume: bool = False
    reset_mcm_state_each_step: bool = False
    require_mcm_grad_check: bool = False
    require_live_iwe_cache_match: bool = False
    live_iwe_cache_match_atol: float = 0.02
    live_iwe_cache_match_rtol: float = 0.05


@dataclass
class EAConfig:
    """Evolutionary search configuration."""

    enabled: bool = False
    target: str = "firenet_lif"      # firenet_lif | cuadrado_unet
    population_size: int = 8
    generations: int = 1
    elite_size: int = 2
    offspring_size: int = 6
    mutation_rate: float = 0.2
    crossover_rate: float = 0.5
    tournament_size: int = 3
    objective_names: list = field(default_factory=lambda: ["val_loss"])
    maximize_objectives: list = field(default_factory=list)
    search_space: dict = field(default_factory=dict)
    base_genome: dict = field(default_factory=dict)
    initial_population_source: str = "random"  # random | base_genome | config | state
    initial_population_state_path: Optional[str] = None
    rung_budgets: list = field(default_factory=list)
    rung_promote_fraction: float = 0.5
    min_rung_survivors: int = 1
    final_rung_validation_fitness: bool = False
    fitness_wall_ms_soft_limit: float = 1000.0
    # Quality key used for validation-prefixed rows in fitness scoring. None
    # preserves the historical default ("mcm_epe", requires GT flow). Set to
    # "total" for datasets without GT flow (e.g. PEOD) so validation fitness
    # falls back to the held-out self-supervised contrast+smoothness loss
    # instead of being unconditionally invalid.
    fitness_val_quality_key: Optional[str] = None
    # Cap on validation batches consumed when use_validation_fitness is set
    # for the final rung (0 = unlimited, validate() runs to exhaustion).
    # Without this, "validation fitness" can run a full pass over the entire
    # val loader (thousands of batches on high-resolution datasets), far
    # exceeding the training rung budget it is meant to score.
    fitness_val_max_batches: int = 50
    # Flow-structure (DC/global-collapse) guardrail thresholds. Defaults match
    # the historical hardcoded constants in evolution/metrics.py so existing
    # DSEC/other-target configs are unaffected. Tighten for PEOD-style data
    # where a global ego-motion DC shift is a cheap, well-rewarded local
    # minimum for the unsupervised event_warping objective.
    fitness_flow_mag_min: float = 0.10
    fitness_flow_centered_ratio_min: float = 0.25
    fitness_flow_mean_ratio_max: float = 0.995
    fitness_flow_spatial_std_min: float = 0.002
    # Weight applied to constraint_violation (flow_structure + snn_stability +
    # compute_constraint) inside f_quality. Default matches the historical
    # hardcoded 0.05.
    fitness_constraint_violation_weight: float = 0.05
    evaluator_max_retries: int = 1
    evaluator_retry_backoff_sec: float = 5.0
    evaluator_retry_failure_reasons: list = field(
        default_factory=lambda: ["dataloader_transient", "filesystem_transient"]
    )
    evaluator: str = "mcm"
    ddp_mode: str = "master_only"    # master_only | all_ranks
    seed: Optional[int] = None
    checkpoint_interval: int = 1


@dataclass
class MonEvalConfig:
    """Representative lightweight eval subset run during training."""
    enabled: bool = False
    interval: int = 0
    split: str = "val"  # train | val | test
    dataset_config: Optional[str] = None
    sequences: list = field(default_factory=list)


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    interval: int = 10
    save_interval: int = 10
    save_epochs: list = field(default_factory=list)
    cleanup_after_eval: bool = True
    empty_cache_after_eval: bool = True
    shutdown_val_loader_workers: bool = True
    metrics: list = field(default_factory=lambda: ["mAP50", "mAP", "mAP75"])
    iou_threshold: float = 0.5
    conf_threshold: float = 0.001
    nms_threshold: float = 0.65
    compute_loss: bool = False
    split: str = "val"  # train | val | test
    dataset_config: Optional[str] = None
    sequences: list = field(default_factory=list)
    mon_eval: MonEvalConfig = field(default_factory=MonEvalConfig)
    # Cap on the number of batches consumed by MCMTrainer.validate() (0 =
    # unlimited, the historical default). Without a cap, a full validation
    # pass iterates the entire val loader regardless of how small the
    # training budget is — for PEOD-resolution data this can take far longer
    # than the training rung budget itself (e.g. 5444 val batches vs. an
    # 8-15 iter EA proxy rung), defeating the point of a "short validation."
    max_batches: int = 0

    @property
    def eval_interval(self) -> int:
        """Backward-compatible alias for interval."""
        return int(self.interval)

    @eval_interval.setter
    def eval_interval(self, value: int) -> None:
        self.interval = int(value)

    @property
    def full_eval_interval(self) -> int:
        """Backward-compatible alias for interval."""
        return int(self.interval)

    @full_eval_interval.setter
    def full_eval_interval(self, value: int) -> None:
        self.interval = int(value)

    @property
    def monitor_validation(self) -> MonEvalConfig:
        """Backward-compatible alias for mon_eval."""
        return self.mon_eval

    @monitor_validation.setter
    def monitor_validation(self, value: MonEvalConfig) -> None:
        self.mon_eval = value


@dataclass
class VizConfig:
    """Visualization-only inference configuration.

    Intended to be used with a dedicated visualization run mode (e.g. --viz-only)
    so training/validation stays clean and low-overhead.
    """

    max_images: int = 200
    conf_threshold: float = 0.3
    dir: str = "eval_vis"
    random_sequence: bool = True

    # Per-sequence viz mode. When > 0, visualize() caps each val sequence at
    # this number of frames instead of honouring max_images globally. Enables
    # the "one GIF per sequence" workflow.
    frames_per_sequence: int = 0

    # Optional: dump raw tensors for debugging (off by default).
    dump_predictions: bool = False
    dump_max_images: int = 200
    dump_file: str = "predictions_debug.pt"

    # When true, viz loads DSEC GT optical flow (from batch["flow_gt"] if
    # data.mapper_load_flow is on, otherwise via on-disk lookup) and renders
    # it alongside MCM/precomputed flow with a per-pixel EPE panel. Frames
    # from sequences without GT flow silently fall back to the no-GT layout.
    load_flow_gt: bool = False


@dataclass
class LogConfig:
    """Logging and checkpointing configuration."""
    output_dir: str = "./outputs"
    experiment_name: str = "phase1_baseline"
    log_interval: int = 50
    memory_log_interval: int = 0
    save_all_checkpoints: bool = False
    auto_resume: bool = False
    resume_from: Optional[str] = None
    load_weights_only: bool = False
    verbose_validation: bool = False

@dataclass
class WandbConfig:
    """Weights & Biases logging configuration."""
    use_wandb: bool = False
    project_name: str = "smcfnet_dsec"
    entity: Optional[str] = None

    # Optional run metadata
    run_name: Optional[str] = None          # default: experiment_name
    group: Optional[str] = None
    tags: list = field(default_factory=list)

    # Logging behavior
    log_every_n_steps: int = 10             # collect samples every N steps
    flush_every_n_steps: int = 50           # push aggregated metrics every N steps
    mode: str = "online"                    # "online" | "offline" | "disabled"


@dataclass
class Config:
    """Master configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evolution: EAConfig = field(default_factory=EAConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    viz: VizConfig = field(default_factory=VizConfig)
    log: LogConfig = field(default_factory=LogConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    num_gpus: int = 1
    dist_url: str = "auto"
    seed: int = 42


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dicts; override wins on conflicts."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config_from_yaml(yaml_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file, with optional _base_ inheritance."""
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with yaml_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    # Normalize legacy eval field names (eval_interval, full_eval_interval,
    # monitor_validation) → canonical names (interval, mon_eval) at THIS level
    # before merging with base. Without this, a child that uses `eval_interval`
    # would be silently dropped when the base has `interval` already set,
    # because _deep_merge keeps both keys and the post-merge normalize then
    # only promotes the legacy field when the canonical is absent. See #013/#021.
    eval_section = config.get("eval")
    if isinstance(eval_section, dict):
        config["eval"] = _normalize_eval_updates(eval_section)

    base_ref = config.pop("_base_", None)
    if base_ref:
        base_path = yaml_path.parent / base_ref
        base = load_config_from_yaml(str(base_path))
        config = _deep_merge(base, config)

    return config


def update_config_from_dict(config: Config, updates: Dict[str, Any]) -> None:
    """Update config object from nested dictionary."""
    for key, value in updates.items():
        if hasattr(config, key):
            attr = getattr(config, key)
            if isinstance(value, dict) and hasattr(attr, '__dict__'):
                _update_nested(attr, value, key)
            else:
                setattr(config, key, value)
        else:
            print(f"Warning: Unknown config key '{key}' - skipping")


def _update_nested(obj: Any, updates: Dict[str, Any], path: str) -> None:
    """Recursively update a nested dataclass from a dict."""
    if path == "eval" and isinstance(updates, dict):
        updates = _normalize_eval_updates(updates)
    for sub_key, sub_value in updates.items():
        if hasattr(obj, sub_key):
            sub_attr = getattr(obj, sub_key)
            if isinstance(sub_value, dict) and hasattr(sub_attr, '__dict__'):
                _update_nested(sub_attr, sub_value, f"{path}.{sub_key}")
            else:
                setattr(obj, sub_key, sub_value)
        else:
            print(f"Warning: Unknown config key '{path}.{sub_key}' - skipping")


def _normalize_eval_updates(updates: Dict[str, Any]) -> Dict[str, Any]:
    """Map legacy eval config keys onto the current API.

    New API:
    - eval.interval
    - eval.mon_eval.*

    Legacy aliases still accepted:
    - eval.eval_interval
    - eval.full_eval_interval
    - eval.monitor_validation
    """
    normalized = dict(updates)

    if "monitor_validation" in normalized and "mon_eval" not in normalized:
        normalized["mon_eval"] = normalized["monitor_validation"]

    if "interval" not in normalized:
        if "full_eval_interval" in normalized:
            normalized["interval"] = normalized["full_eval_interval"]
        elif "eval_interval" in normalized:
            normalized["interval"] = normalized["eval_interval"]

    normalized.pop("monitor_validation", None)
    normalized.pop("full_eval_interval", None)
    normalized.pop("eval_interval", None)
    return normalized


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train RGB-E Object Detection with STTP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default="src/smcfnet/config/config.yaml",
        help="Path to YAML config file"
    )

    # Data
    parser.add_argument("--dataset-config", type=str, help="Path to DSEC config YAML")
    parser.add_argument("--num-workers", type=int, help="Number of data loading workers")

    # Model
    parser.add_argument(
        "--input-mode",
        type=str,
        choices=["dual", "rgb"],
        help="Input mode: 'dual' (RGB+events) or 'rgb' (RGB-only)",
    )
    parser.add_argument("--backbone", type=str, help="Backbone architecture")
    parser.add_argument("--model-size", type=str, choices=['nano', 'tiny', 's', 'm', 'l', 'x'], help="Model size")
    parser.add_argument("--pretrained-path", type=str, help="Override model.pretrained_path from YAML")
    parser.add_argument("--no-mcm", action="store_true", dest="no_mcm", help="Disable MCM module")
    parser.add_argument("--no-edum", action="store_true", help="Disable EDUM module")
    parser.add_argument("--no-cmm", action="store_true", help="Disable CMM module")
    parser.add_argument("--enable-mcm", action="store_true", dest="enable_mcm", help="Enable MCM module")
    parser.add_argument("--enable-edum", action="store_true", help="Enable EDUM module")
    parser.add_argument("--enable-cmm", action="store_true", help="Enable CMM module")

    # Training
    parser.add_argument("--batch-size", type=int, help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument(
        "--max-steps-per-epoch",
        type=int,
        help="Optional limit on train iterations per epoch for smoke/profiling runs",
    )
    parser.add_argument(
        "--max-total-steps",
        type=int,
        help="Optional hard stop on total training iterations for smoke/profiling runs",
    )
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--momentum", type=float, help="SGD momentum")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")

    # Logging
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument("--experiment-name", type=str, help="Experiment name")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")

    # Distributed
    parser.add_argument("--num-gpus", type=int, help="Number of GPUs")
    parser.add_argument("--dist-url", type=str, help="Distributed URL")

    # Evaluation
    parser.add_argument("--eval-only", action="store_true", help="Run evaluation only (compute metrics)")

    # Visualization-only inference
    parser.add_argument(
        "--viz-only",
        "--save-vis",
        dest="viz_only",
        action="store_true",
        help="Run visualization-only inference and save PNG overlays",
    )
    parser.add_argument("--vis-num", type=int, help="Max number of visualization images to save")
    parser.add_argument("--vis-conf", type=float, help="Confidence threshold for visualized predictions")
    parser.add_argument("--vis-dir", type=str, help="Subfolder under output_dir/experiment_name for PNGs")
    parser.add_argument(
        "--no-random-seq",
        action="store_true",
        help="Disable random sequence selection for viz-only (save first frames encountered)",
    )
    parser.add_argument("--dump-preds", action="store_true", help="Dump per-image predictions/targets during viz-only")
    parser.add_argument("--dump-max", type=int, help="Max number of images to include in prediction dump")
    parser.add_argument(
        "--vis-per-seq",
        type=int,
        help="Per-sequence frame cap for viz-only. When >0, save this many frames per val sequence (ignores --vis-num).",
    )


    # Misc
    parser.add_argument("--seed", type=int, help="Random seed")

    return parser.parse_args()


def get_config() -> tuple[Config, argparse.Namespace]:
    """
    Build configuration from YAML file and command line arguments.
    Command line arguments override YAML settings.
    """
    args = parse_args()
    config = Config()

    # Load from YAML if provided and exists
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            yaml_config = load_config_from_yaml(args.config)
            update_config_from_dict(config, yaml_config)
        else:
            print(f"Warning: Config file not found: {args.config}, using defaults")

    # Override with command line arguments
    if args.dataset_config:
        config.data.dataset_config = args.dataset_config
    if args.num_workers is not None:
        config.data.num_workers = args.num_workers

    if args.backbone:
        config.model.backbone = args.backbone
    if getattr(args, "input_mode", None):
        config.model.input_mode = str(args.input_mode)
    if args.model_size:
        config.model.size = args.model_size
    if getattr(args, "pretrained_path", None):
        config.model.pretrained_path = str(args.pretrained_path)
    if getattr(args, "no_mcm", False):
        config.model.mcm.enabled = False
    if args.no_edum:
        config.model.edum.enabled = False
    if args.no_cmm:
        config.model.cmm.enabled = False
    if getattr(args, "enable_mcm", False):
        config.model.mcm.enabled = True
    if args.enable_edum:
        config.model.edum.enabled = True
    if args.enable_cmm:
        config.model.cmm.enabled = True
    # Mirror legacy fusion_layers into cmm config when provided.
    if getattr(config.model, "fusion_layers", None):
        config.model.cmm.fusion_layers = list(config.model.fusion_layers)

    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.epochs is not None:
        config.training.num_epochs = args.epochs
    if getattr(args, "max_steps_per_epoch", None) is not None:
        config.training.max_steps_per_epoch = int(args.max_steps_per_epoch)
    if getattr(args, "max_total_steps", None) is not None:
        config.training.max_total_steps = int(args.max_total_steps)
    if args.lr is not None:
        config.training.learning_rate = args.lr
    if getattr(args, "momentum", None) is not None:
        config.training.momentum = float(args.momentum)
    if args.no_amp:
        config.training.use_amp = False

    if args.output_dir:
        config.log.output_dir = args.output_dir
    if args.experiment_name:
        config.log.experiment_name = args.experiment_name
    if args.resume:
        config.log.resume_from = args.resume

    if args.num_gpus is not None:
        config.num_gpus = args.num_gpus
    if args.dist_url:
        config.dist_url = args.dist_url
    if args.seed is not None:
        config.seed = args.seed

    # Viz-only overrides
    if getattr(args, "viz_only", False):
        pass
    if getattr(args, "vis_num", None) is not None:
        config.viz.max_images = int(args.vis_num)
    if getattr(args, "vis_conf", None) is not None:
        config.viz.conf_threshold = float(args.vis_conf)
    if getattr(args, "vis_dir", None):
        config.viz.dir = str(args.vis_dir)
    if getattr(args, "no_random_seq", False):
        config.viz.random_sequence = False
    if getattr(args, "vis_per_seq", None) is not None:
        config.viz.frames_per_sequence = int(args.vis_per_seq)
    if getattr(args, "dump_preds", False):
        config.viz.dump_predictions = True
    if getattr(args, "dump_max", None) is not None:
        config.viz.dump_max_images = int(args.dump_max)

    return config, args


def save_config(config: Config, save_path: str) -> None:
    """Save configuration to YAML file."""
    config_dict = config_to_dict(config)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


def config_to_dict(config: Config) -> Dict[str, Any]:
    """Convert nested config dataclasses to plain Python containers."""
    return {
        'data': asdict(config.data),
        'model': asdict(config.model),
        'training': asdict(config.training),
        'eval': asdict(config.eval),
        'evolution': asdict(config.evolution),
        'viz': asdict(config.viz),
        'log': asdict(config.log),
        'wandb': asdict(config.wandb),
        'num_gpus': config.num_gpus,
        'dist_url': config.dist_url,
        'seed': config.seed,
    }
