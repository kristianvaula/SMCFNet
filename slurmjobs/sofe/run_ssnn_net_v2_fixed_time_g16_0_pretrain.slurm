#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=ssnn-v2-g16-pre
#SBATCH --time=32:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu40g|gpu80g"
#SBATCH --mem=60G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

CONFIG="${CONFIG:-src/smcfnet/config/runs/sofe/ssnn_net/ssnn_net_v2_fixed_time_g16_0_pretrain.yaml}"
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
  echo "ERROR: cannot locate SMCM-MCFNet repo root"
  exit 1
fi

OUTPUT_ROOT="outputs/sofe/ssnn_net_v2_fixed_time"
EXPERIMENT_NAME="ssnn_net_v2_fixed_time_g16_0_pretrain"
RUN_DIR="${OUTPUT_ROOT}/${EXPERIMENT_NAME}"
mkdir -p "${RUN_DIR}"

RUN_STDOUT="${RUN_DIR}/stdout_${JOB_ID}.log"
RUN_STDERR="${RUN_DIR}/stderr_${JOB_ID}.log"
exec >>"${RUN_STDOUT}" 2>>"${RUN_STDERR}"

export xml_catalog_files_libxml2="${xml_catalog_files_libxml2-}"
set +u
module purge
module load HDF5/1.14.0-iimpi-2023a
module load Anaconda3/2024.02-1
source "/cluster/work/${CURRENT_USER}/miniconda3/etc/profile.d/conda.sh"
conda activate smcm-mcfnet || { echo "Failed to activate conda env 'smcm-mcfnet'"; exit 1; }
export xml_catalog_files_libxml2="${xml_catalog_files_libxml2-}"
set -u

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1

echo "Starting SOFE pretraining"
echo "Config: ${CONFIG}"
echo "Run dir: ${RUN_DIR}"
echo "Host: $(hostname)"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L || true

exec python -m smcfnet.core.train \
  --config "${CONFIG}" \
  --output-dir "${OUTPUT_ROOT}" \
  --experiment-name "${EXPERIMENT_NAME}"