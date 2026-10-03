#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=sttp-train
#SBATCH --time=24:00:00

#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu40g|gpu80g"
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

#SBATCH --mail-type=ALL
#SBATCH --mail-user=ecschoye@stud.ntnu.no

# SLURM bootstrap logs land in .slurm-logs/ until conda is up and the
# experiment name + MCM checkpoint are resolved; runtime logs go to the
# per-experiment dir via exec >>. NOTE: .slurm-logs/ must exist at submit time.
#   mkdir -p .slurm-logs
#SBATCH --output=.slurm-logs/sttp-train-%j.out
#SBATCH --error=.slurm-logs/sttp-train-%j.err

# ---------------------------------------------------------------------------
# Generic STTP trainer. Defaults to flow-injection on R3 MCM, but accepts
# any STTP config via CONFIG env override (no-flow / flow-injection /
# flow-only / baseline). Resolves experiment_name + pretrained_path from
# the config AFTER conda is up so banners show real paths.
# Resumes from ${EXPERIMENT}/latest.pth if present.
#
# Override the MCM checkpoint without editing YAML by setting MCM_CKPT:
#
#   MCM_CKPT=./outputs/mcm/mcm_cmax_R3_halfres/best.pth \
#     sbatch --export=ALL,CONFIG=src/smcfnet/config/runs/sttp/sttp_flowinj_eraft.yaml \
#     slurmjobs/sttp/run_sttp_train.slurm
#
# If MCM_CKPT is unset, the value from cfg.model.pretrained_path is used.
# ---------------------------------------------------------------------------

set -euo pipefail

CONFIG="${CONFIG:-src/smcfnet/config/runs/sttp/sttp_flowinj_mcm.yaml}"

JOB_ID="${SLURM_JOB_ID:-manual}"
WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${WORKDIR}"

CURRENT_USER="$(whoami)"
if [ ! -f "pyproject.toml" ]; then
  for candidate in \
      "/cluster/work/${CURRENT_USER}/SMCM-MCFNet" \
      "/cluster/work/${CURRENT_USER}/projects/smcm-mcfnet/SMCM-MCFNet"; do
    if [ -f "${candidate}/pyproject.toml" ]; then
      cd "${candidate}"
      break
    fi
  done
fi
if [ ! -f "pyproject.toml" ]; then
  echo "ERROR: cannot locate SMCM-MCFNet repo root (pyproject.toml not found in WORKDIR or known candidates)."
  exit 1
fi

# nounset-safe module + conda bootstrap (per slurmjobs/README.md).
export xml_catalog_files_libxml2="${xml_catalog_files_libxml2-}"
set +u
module purge
module load HDF5/1.14.0-iimpi-2023a
module load Anaconda3/2024.02-1
if [ -f "/cluster/work/${CURRENT_USER}/miniconda3/etc/profile.d/conda.sh" ]; then
  source "/cluster/work/${CURRENT_USER}/miniconda3/etc/profile.d/conda.sh"
elif [ -n "${EBROOTANACONDA3:-}" ] && [ -f "${EBROOTANACONDA3}/etc/profile.d/conda.sh" ]; then
  source "${EBROOTANACONDA3}/etc/profile.d/conda.sh"
else
  echo "ERROR: no conda.sh found (looked under per-user miniconda and module-provided \$EBROOTANACONDA3)"
  exit 1
fi
conda activate smcm-mcfnet || { echo "Failed to activate conda env 'smcm-mcfnet'"; exit 1; }
export xml_catalog_files_libxml2="${xml_catalog_files_libxml2-}"
set -u

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
pip install -e . --no-deps || echo "WARN: pip install -e failed (env may be stale)"

# Resolve config fields with the real env available. MCM_CKPT (env) takes
# precedence over the YAML's model.pretrained_path so the same launcher can
# swap checkpoints without YAML edits.
read -r MCM_CKPT_FROM_CFG EXPERIMENT < <(python -c "
from smcfnet.config.config import Config, load_config_from_yaml, update_config_from_dict
cfg = Config()
update_config_from_dict(cfg, load_config_from_yaml('${CONFIG}'))
print(cfg.model.pretrained_path or '', cfg.log.experiment_name)
")
MCM_CKPT="${MCM_CKPT:-${MCM_CKPT_FROM_CFG}}"
if [ -z "${MCM_CKPT}" ]; then
  echo "ERROR: MCM checkpoint is empty. Set MCM_CKPT env or model.pretrained_path in ${CONFIG}."
  exit 1
fi
if [ -z "${EXPERIMENT}" ]; then
  echo "ERROR: failed to resolve experiment_name from ${CONFIG}"
  exit 1
fi

OUTPUT_ROOT="outputs/sttp"
RUN_DIR="${OUTPUT_ROOT}/${EXPERIMENT}"
mkdir -p "${RUN_DIR}"

RUN_STDOUT="${RUN_DIR}/stdout_${JOB_ID}.log"
RUN_STDERR="${RUN_DIR}/stderr_${JOB_ID}.log"
exec >"${RUN_STDOUT}" 2>"${RUN_STDERR}"

RESUME_CKPT="./outputs/sttp/${EXPERIMENT}/latest.pth"

echo "========================================"
echo "STTP train (frozen detector, resumes if latest.pth present)"
echo "========================================"
echo "Job ID:     ${JOB_ID}"
echo "Host:       $(hostname)"
echo "Date:       $(date)"
echo "Config:     ${CONFIG}"
echo "Experiment: ${EXPERIMENT}"
echo "Checkpoint: ${MCM_CKPT}"
echo "Resume:     ${RESUME_CKPT}"

if [ ! -f "${MCM_CKPT}" ]; then
  echo "ERROR: MCM checkpoint not found at ${MCM_CKPT}"
  exit 1
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L || true
command -v nvidia-smi >/dev/null 2>&1 \
  && nvidia-smi --query-gpu=memory.free,memory.used,memory.total --format=csv \
  || true

TRAIN_CMD=(python -m smcfnet.core.train --config "${CONFIG}" --pretrained-path "${MCM_CKPT}")
if [ -f "${RESUME_CKPT}" ]; then
  echo "Resuming from ${RESUME_CKPT}"
  TRAIN_CMD+=(--resume "${RESUME_CKPT}")
fi

if "${TRAIN_CMD[@]}"; then
  echo ""
  echo "STTP train finished at $(date)"
  exit 0
else
  EXIT_CODE=$?
  echo ""
  echo "STTP train FAILED (exit ${EXIT_CODE}) at $(date)"
  exit "${EXIT_CODE}"
fi
