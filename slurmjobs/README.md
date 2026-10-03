# SLURM Jobs Guide

This handoff keeps a small set of example jobs that cover the main workflow
used in the thesis without shipping every one-off sweep and repair script.

## Kept Examples

- `baseline/run_rgbe_ssnn_net_v2_iwe_edum_cmm_m.slurm`: final detector-style training example.
- `sofe/run_ssnn_net_v2_fixed_time_g16_0_pretrain.slurm`: selected SSNN SOFE pretraining example.
- `cache/run_precompute_smcm_iwe_template.slurm`: offline IWE cache generation template.
- `sttp/run_sttp_train.slurm`: STTP training example.
- `run_precompute_cuadrado_e35_flow.slurm`: precompute flow for STTP from a trained Cuadrado checkpoint.

## Pattern To Follow

All example jobs use the same conventions:

- `set -euo pipefail`
- repo-root resolution from `SLURM_SUBMIT_DIR` with cluster fallbacks
- nounset-safe module and conda bootstrap
- per-run stdout and stderr redirected into the experiment directory
- `PYTHONPATH="$PWD/src:${PYTHONPATH:-}"`

The reliable cluster bootstrap pattern is:

```bash
export xml_catalog_files_libxml2="${xml_catalog_files_libxml2-}"
set +u
module purge
module load HDF5/1.14.0-iimpi-2023a
module load Anaconda3/2024.02-1
source "/cluster/work/${CURRENT_USER}/miniconda3/etc/profile.d/conda.sh"
conda activate smcm-mcfnet || { echo "Failed to activate conda env 'smcm-mcfnet'"; exit 1; }
export xml_catalog_files_libxml2="${xml_catalog_files_libxml2-}"
set -u
```

## Logging

The default recommendation is to disable SLURM bootstrap logs and redirect the
real runtime logs into the experiment directory:

```bash
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

RUN_STDOUT="${RUN_DIR}/stdout_${SLURM_JOB_ID:-manual}.log"
RUN_STDERR="${RUN_DIR}/stderr_${SLURM_JOB_ID:-manual}.log"
exec >>"${RUN_STDOUT}" 2>>"${RUN_STDERR}"
```

That keeps each run self-contained and avoids duplicate log files.

## Common Failure Modes

- `xml_catalog_files_libxml2` unbound variable: keep the `set +u` conda bootstrap wrapper.
- missing runtime logs: create the run directory before `exec >> ...`.
- wrong repo root: submit from the repo root or update the fallback paths.
- silent config drift: echo the selected config and experiment name before training starts.
- Create directories before `exec` redirection.
- Prefer `#SBATCH --output=/dev/null` and `#SBATCH --error=/dev/null` when runtime logs are already redirected into the run directory.

### Error: strict mode aborts on non-critical diagnostics

Cause:
- `pipefail` and `-e` terminate on optional command failures.

Fix:
- Add `|| true` to optional diagnostics.
- Do not add `|| true` to required training/evaluation commands.

### Error: unset SLURM variables when running manually

Cause:
- Running script outside `sbatch` with `set -u`.

Fix:
- Use defaults for optional variables, for example:
  - `JOB_ID="${SLURM_JOB_ID:-manual}"`
  - `TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"`

## 5) Pre-Submit Checklist

Before submitting a new or edited script:
- Script starts with `set -euo pipefail`.
- Module + conda bootstrap block is fully inside the nounset-safe guard.
- `#SBATCH --output=/dev/null` and `#SBATCH --error=/dev/null` are set unless duplicate bootstrap logs are explicitly needed.
- Runtime log directories are created before `exec >>` redirection.
- Array bounds are validated for sweep jobs.
- Required commands fail loudly; optional diagnostics are guarded.
- Training command uses the expected config and output path.

## 6) Submission Examples

Single job:

```bash
sbatch slurmjobs/baseline/run_baseline150_rgb_m_distorted.slurm
```

Array job:

```bash
sbatch slurmjobs/run_sofe_stflow_lite_alignment_qual_sweep_array.slurm
```

Array job with overrides:

```bash
TRAIN_EPOCHS=20 TRAIN_MAX_STEPS_PER_EPOCH=64 sbatch slurmjobs/run_sofe_spikingjelly_unet_persistent_stabilization_qual_sweep_array.slurm
```
