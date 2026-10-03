# SMCM-MCFNet

This repository contains the core code for the SMCM-MCFNet thesis project in a
trimmed handoff form. The goal of this version is to keep the main training,
evaluation, data-pipeline, and trajectory-prediction code while removing most
thesis-specific sweeps, repair runs, and analysis artifacts.

## What Is Kept

- core model, training, dataset, and evaluation code under `src/smcfnet/`
- a small set of reusable top-level tools under `tools/`
- a curated set of example configs under `src/smcfnet/config/`
- a small set of representative SLURM jobs under `slurmjobs/`
- two short docs for the data pipeline and cache workflow under `docs/`

## Main Workflow

The intended high-level workflow is:

1. Set up DSEC and the Python environment.
2. Preprocess rectified events and tracks if you want full rectified training.
3. Optionally pretrain or precompute motion representations.
4. Train the detector.
5. Optionally train STTP on top of a motion source.

The main curated examples in this handoff are:

- SOFE pretraining: `src/smcfnet/config/runs/sofe/ssnn_net/ssnn_net_v2_fixed_time_g16_0_pretrain.yaml`
- detector training: `src/smcfnet/config/runs/baselines/train/rgbe_ssnn_net_v2_iwe_edum_cmm_m.yaml`
- STTP with shared motion: `src/smcfnet/config/runs/sttp/sttp_flowinj_mcm.yaml`
- STTP with precomputed E-RAFT flow: `src/smcfnet/config/runs/sttp/sttp_flowinj_eraft_v4.yaml`
- Cuadrado flow precompute for STTP: `src/smcfnet/config/runs/sttp/cuadrado_e35_precompute.yaml`

## Data Setup

The project uses DSEC-Detection and related DSEC assets.

Set base paths in your shell:

```bash
export DSEC_ROOT="/cluster/work/${USER}/datasets/dsec"
export PROJECT_ROOT="/cluster/work/${USER}/SMCM-MCFNet"
```

Download the dataset with the upstream `dsec-det` scripts:

```bash
cd /cluster/work/${USER}
git clone git@github.com:uzh-rpg/dsec-det.git
cd dsec-det

bash scripts/download_dsec.sh ${DSEC_ROOT}
bash scripts/download_dsec_extra.sh ${DSEC_ROOT}
bash scripts/download_remapped_images.sh ${DSEC_ROOT}
```

For the expected on-disk layout and the distorted-versus-rectified data
contract, see `docs/dataset-data-pipeline.md`.

## Installation

On IDUN or a similar cluster setup:

```bash
module load HDF5/1.14.0-iimpi-2023a
module load Anaconda3/2024.02-1

conda create -n smcm-mcfnet python=3.10 -y
conda activate smcm-mcfnet

cd ${PROJECT_ROOT}
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.0+cu121.html
pip install -r requirements.txt
pip install -e .
```

If `requirements.txt` is not sufficient for your cluster image, use it as the
baseline and add platform-specific packages as needed.

## Preprocessing

For full-resolution rectified image training, preprocess rectified events and
tracks:

```bash
cd ${PROJECT_ROOT}/src/smcfnet/dataset/dsec/preprocessing
cp progress_template.txt progress.txt

./run_preprocessing.sh both
```

See `src/smcfnet/dataset/dsec/preprocessing/README.md` for preprocessing
details.

## Example Runs

All training uses the same entrypoint:

```bash
python -m smcfnet.core.train --config <path/to/config.yaml>
```

Useful starting points:

```bash
# Detector training
python -m smcfnet.core.train \
  --config src/smcfnet/config/runs/baselines/train/rgbe_ssnn_net_v2_iwe_edum_cmm_m.yaml

# SOFE pretraining
python -m smcfnet.core.train \
  --config src/smcfnet/config/runs/sofe/ssnn_net/ssnn_net_v2_fixed_time_g16_0_pretrain.yaml

# STTP training with shared motion
python -m smcfnet.core.train \
  --config src/smcfnet/config/runs/sttp/sttp_flowinj_mcm.yaml
```

## SLURM Jobs

The kept SLURM examples are meant as templates, not a complete job archive:

- `slurmjobs/baseline/run_rgbe_ssnn_net_v2_iwe_edum_cmm_m.slurm`
- `slurmjobs/sofe/run_ssnn_net_v2_fixed_time_g16_0_pretrain.slurm`
- `slurmjobs/cache/run_precompute_smcm_iwe_template.slurm`
- `slurmjobs/sttp/run_sttp_train.slurm`
- `slurmjobs/run_precompute_cuadrado_e35_flow.slurm`

See `slurmjobs/README.md` for the conventions used by these scripts.

## Top-Level Tools

The `tools/` folder now keeps only reusable helpers such as:

- `eval_checkpoint.py`
- `precompute_smcm_iwe.py`
- `precompute_cuadrado_flow.py`
- `precompute_stem_features.py`
- `profile_dataset_loader.py`
- `visualize_dsec.py`
- `visualize_flow_qualitative.py`
- STTP plotting, dumping, aggregation, and TTC utilities

## Testing

Tests live in `src/smcfnet/test/`.

Use:

```bash
python -m pytest
```

For a compact testing guide with example commands, see `TESTING.md`.

## References

If you use DSEC-Detection, please cite:

```bibtex
@Article{Gehrig24nature,
  author    = {Gehrig, Daniel and Scaramuzza, Davide},
  title     = {Low Latency Automotive Vision with Event Cameras},
  journal   = {Nature},
  year      = {2024}
}

@InProceedings{Gehrig21ral,
  author  = {Mathias Gehrig and Willem Aarents and Daniel Gehrig and Davide Scaramuzza},
  title   = {{DSEC}: A Stereo Event Camera Dataset for Driving Scenarios},
  journal = {{IEEE} Robotics and Automation Letters},
  year    = {2021},
  doi     = {10.1109/LRA.2021.3068942}
}
```
