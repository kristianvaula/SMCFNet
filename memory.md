# SMCM-MCFNet Memory

## What this project is

SMCM-MCFNet is an RGB-Event object detection research codebase built around the DSEC and PEOD dataset. The main task is 2D object detection from synchronized RGB frames and event-camera data, with optional motion-compensation and fusion modules motivated by a thesis on low-latency, resource-aware RGB-E perception.

Core research direction:
- Preserve more event-native temporal information than standard dense event-frame pipelines.
- Use motion compensation that can be ANN- or SNN-driven.
- Reuse motion estimates inside the detector instead of treating motion as a disposable preprocessing step.
- Keep the system compatible with Linux HPC training, SLURM, conda, PyTorch DDP, and checkpointed experiment workflows.

## Main entrypoints

- `src/smcfnet/core/train.py`: top-level training entrypoint (`python -m smcfnet.core.train`).
- `src/smcfnet/core/launch.py`: single-node multi-GPU launcher based on `torch.distributed` and `mp.spawn`.
- `src/smcfnet/core/trainer.py`: standard joint detector trainer.
- `src/smcfnet/core/mcm_trainer.py`: MCM-only pretraining trainer.

Training dispatch:
- `training.mode == "joint"`: full detection training.
- `training.mode == "mcm_pretrain"`: train only the motion compensation module.

`train.py` also supports:
- `--eval-only`: load a checkpoint and run validation.
- `--viz-only`: load a checkpoint and save visualizations.

## High-level pipeline

1. YAML config is loaded into dataclasses from `src/smcfnet/config/config.py`.
2. `launch.py` sets up rank/local-rank/world-size and DDP.
3. A DSEC DataLoader is built in `src/smcfnet/model/build.py`.
4. The lightweight mapper reads raw events, optional RGB, optional targets, and optional flow GT.
5. `DSECGPUPrefetcher` moves packed events to GPU and voxelizes them into `(B, bins, H, W)`.
6. The model consumes:
   - RGB image tensor `(B, 3, H, W)`
   - event voxel tensor `(B, event_bins, H, W)`
   - detection targets `(B, M, 5)` during training
7. The dual-stream backbone extracts RGB and event features separately, optionally applying:
   - MCM before the event backbone
   - EDUM on event features
   - CMM at selected fusion stages
8. Fused multi-scale features go through YOLOX PAFPN and then either:
   - YOLOX head
   - DETR-style decoder head
9. Total training loss is detection loss plus optional weighted MCM loss.

## Data pipeline details

Dataset stack is centered on DSEC.

Relevant files:
- `src/smcfnet/dataset/dsec/dsec_lightweight_mapper.py`
- `src/smcfnet/dataset/dsec/dsec_gpu_prefetcher.py`
- `src/smcfnet/model/build.py`

Important design choices:
- DataLoader workers do minimal CPU work.
- Events stay as packed tensors for as long as possible.
- Voxelization is intentionally deferred to the GPU prefetch stage.
- RGB images and per-frame targets can be materialized in workers when joint detection training is used.
- Flow GT can be loaded for supervised MCM training.

Batch contents typically include:
- `events`: packed `(N, 4)` tensor with `[x, y, t_ms, p]`
- `events_offsets`: offsets delimiting per-sample event windows
- `image`: RGB tensor if enabled
- `targets`: YOLO-style targets if enabled
- `flow_gt`: optical flow supervision if enabled

This means the project is not fundamentally “event-frame first”; it is “packed events first, voxelize late”.

## Model architecture

Main builder:
- `src/smcfnet/model/build.py`

Backbone:
- `src/smcfnet/model/backbone.py`

Wrapper:
- `src/smcfnet/model/yolox.py`

Default detector structure:
- Dual-stream CSPDarknet backbone
- One RGB branch and one event branch
- Fusion at `dark3`, `dark4`, `dark5`
- Baseline fusion is concat + `1x1` channel reduction
- YOLOPAFPN neck
- YOLOX head by default, DETR head optionally

The detector wrapper is `DualStreamYOLOX`, which:
- runs the backbone
- extracts optional MCM auxiliary outputs
- feeds fused features into neck and head
- adds `loss_mcm` into total loss using `model.mcm.loss_weight`

## MCM: Motion Compensation Module

Key files:
- `src/smcfnet/model/mcm/mcm.py`
- `src/smcfnet/model/build.py`
- `src/smcfnet/core/mcm_trainer.py`

MCM is a generic wrapper around:
- a flow estimator
- a warper
- a motion-compensation loss

Contract:
- input: event voxel grid
- output: corrected voxel grid plus scalar loss

What MCM does:
1. Predict optical flow from event voxels.
2. Scale the flow if configured.
3. Warp voxel bins or sparse events to a reference time.
4. Compute self-supervised or supervised motion loss during training.
5. Return corrected event representation for the event backbone.

Supported ideas in config:
- ANN or SNN-style estimators
- architectures such as `st_flownet`, `conv_flownet`, `cuadrado_unet`, `idnet_ann`, `ecm`
- loss types including `event_warping`, `event_warping_supervised`, `cmax`, `combined`, and supervised flow variants
- optional pretrained MCM weights
- optional freezing of MCM during staged training

Current SOFE focus (preliminary and ongoing experiments):
- Cuadrado U-Net (`cuadrado_unet`)
- Xu FireNet
- Hagenaars FireNet
- SSNN

These four are the prioritized Spiking Optical Flow Estimators (SOFE) for comparative motion-compensation studies and downstream detector integration.

Current training workflow explicitly supports:
- Phase A: MCM pretrain only
- Phase B: load MCM into detector and freeze it
- Phase C: unfreeze and fine-tune jointly

This staged setup matters for checkpoint compatibility and experiment stability.

## Optional modules

### EDUM

Event Dynamic Upsampling Module:
- enabled via `model.edum.enabled`
- used when event resolution should stay native initially and be upsampled inside the model

### CMM

Cross-Modal Mamba Fusion:
- enabled via `model.cmm.enabled` or legacy `model.use_cmm`
- can fuse RGB and event features at selected levels, usually `[3, 4, 5]`

These modules are incremental extensions on top of the same dual-stream detector skeleton rather than a separate architecture family.

## Config system

File:
- `src/smcfnet/config/config.py`

Main config groups:
- `data`
- `model`
- `training`
- `eval`
- `viz`
- `log`
- `wandb`

Important knobs for reasoning about experiments:
- `model.input_mode`: dual vs RGB-only
- `model.head_type`: `yolox` vs `detr`
- `model.mcm.enabled`
- `model.edum.enabled`
- `model.cmm.enabled`
- `training.mode`: `joint` vs `mcm_pretrain`
- `data.mapper_*` flags: decide whether workers load events, images, targets, tracks, or flow

YAML runs live under:
- `src/smcfnet/config/runs/`

Example:
- `src/smcfnet/config/runs/cmm/cmm_m_raw.yaml`

## Evolution for SOFE starting configs

Preliminary architecture-search experiments use Evolution to discover strong starting configurations before longer training runs.

Scope of these preliminary EA runs:
- cover all prioritized SOFE architectures (Cuadrado U-Net, Xu FireNet, Hagenaars FireNet, SSNN)
- search for robust initial hyperparameter and architecture settings
- rank candidate configs with short-budget, multi-objective fitness signals
- export top candidates as starting points for subsequent full pretraining and joint training

In practice, Evolution is treated as an initializer/discovery stage, not the final training stage.

## Critical files for prioritized SOFE architectures

Shared SOFE and integration files (all four architectures):
- `src/smcfnet/model/ofe/sofe.py` (streaming estimator runtime, partitioning, state handling)
- `src/smcfnet/model/ofe/sofe_builder.py` (SOFE family normalization/build registry)
- `src/smcfnet/model/build.py` (MCM/SOFE construction from config)
- `src/smcfnet/config/config.py` (MCM + SOFE config surface)
- `src/smcfnet/evolution/runner.py` (EA entrypoint and target dispatch)
- `src/smcfnet/evolution/framework.py` (EA loop/operators/selection)

Cuadrado U-Net:
- `src/smcfnet/model/ofe/cuadrado_unet.py` (estimator implementation)
- `src/smcfnet/evolution/cuadrado_unet/search_space.py` (EA target/search dimensions)
- `src/smcfnet/config/runs/mcm/cuadrado/` (supervised + finetune train configs)
- `slurmjobs/cuadrado/` (batch scripts for Cuadrado runs)

Xu FireNet:
- `src/smcfnet/model/ofe/firenet_lif.py` (Xu FireNet-LIF implementation)
- `src/smcfnet/evolution/firenet/search_space.py` (EA target/search dimensions)
- `src/smcfnet/config/runs/sofe/firenet_lif/` (SOFE pretrain + EA configs)
- `slurmjobs/sofe/run_firenet_lif_*.slurm` and `slurmjobs/ea/run_firenet_lif_*.slurm` (training and EA launchers)

Hagenaars FireNet:
- `src/smcfnet/model/ofe/hagenaars_firenet_lif.py` (Hagenaars FireNet-LIF implementation)
- `src/smcfnet/evolution/hagenaars_firenet/search_space.py` (EA target/search dimensions)
- `src/smcfnet/config/runs/sofe/hagenaars_firenet_lif/` (SOFE pretrain + EA configs)
- `slurmjobs/sofe/run_hagenaars_firenet_lif_*.slurm` and `slurmjobs/ea/run_hagenaars_firenet_lif_*.slurm` (training and EA launchers)

SSNN:
- `src/smcfnet/model/ofe/ssnn_net.py` (SSNN core/readout implementation, including v2)
- `src/smcfnet/evolution/ssnn_net/search_space.py` (EA target/search dimensions)
- `src/smcfnet/config/runs/sofe/ssnn_net/` (v2 pretrain + EA configs)
- `slurmjobs/sofe/run_ssnn_net_v2_*.slurm` and `slurmjobs/ea/run_ssnn_net_v2_*.slurm` (training and EA launchers)

## Distributed and experiment assumptions

The code assumes:
- PyTorch DDP correctness matters
- multi-GPU training may be launched through the internal launcher
- rank-aware logging and checkpointing are important
- SLURM preemption is expected

`Trainer` handles:
- DDP wrapping
- AMP
- LR scheduling
- hook-based logging/eval/checkpointing
- graceful checkpoint save on `SIGUSR1` / `SIGTERM`
- optional auto-resume from `latest.pth`

This project values stable experiment continuation over ad hoc scripting.

## SLURM and conda note

When writing SLURM scripts for this project, do not activate the `smcm-mcfnet`
conda environment under `set -u` / `nounset` directly. The conda activation
hooks can reference unset variables and fail unexpectedly.

Use this pattern:

```bash
# Conda activation hooks can reference unset variables under nounset.
set +u
source "/cluster/work/${CURRENT_USER}/miniconda3/etc/profile.d/conda.sh"
echo "Activating conda env: smcm-mcfnet"
conda activate smcm-mcfnet || { echo "Failed to activate conda env 'smcm-mcfnet'"; exit 1; }
set -u
```

This should be treated as the default activation pattern for new SLURM scripts
in this repo.

## What is implemented vs. thesis direction

Implemented now:
- RGB + event 2D detection on DSEC
- packed-event loader with late voxelization
- dual-stream CSPDarknet fusion detector
- optional MCM / EDUM / CMM modules
- MCM-only pretraining and joint detector training
- optional YOLOX or DETR head

Motivating thesis direction:
- eventually integrate motion more deeply for short-term trajectory prediction and hazard anticipation
- investigate SNN-based motion compensation for efficiency and temporal fidelity
- reduce redundant downstream tracking-style motion computation

In other words, the current repo is already a modular RGB-E detector with motion-compensation research hooks, while the broader thesis aims go beyond plain detection.

## Good first files to read

If a new model needs fast orientation, read in this order:
- `src/smcfnet/core/train.py`
- `src/smcfnet/core/trainer.py`
- `src/smcfnet/core/mcm_trainer.py`
- `src/smcfnet/config/config.py`
- `src/smcfnet/model/build.py`
- `src/smcfnet/model/backbone.py`
- `src/smcfnet/model/yolox.py`
- `src/smcfnet/model/mcm/mcm.py`
- `src/smcfnet/dataset/dsec/dsec_lightweight_mapper.py`
- `src/smcfnet/dataset/dsec/dsec_gpu_prefetcher.py`

## Mental model to keep

Think of SMCM-MCFNet as:
- a DSEC-focused RGB-E detector
- with a dual-stream YOLOX-style backbone/neck/head core
- where event data is kept raw and lightweight in the loader, voxelized late, and optionally corrected by a motion module before fusion
- and where most research variation is introduced through config-controlled modules rather than by replacing the whole training stack
