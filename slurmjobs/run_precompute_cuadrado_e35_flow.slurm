#!/bin/bash
#SBATCH --job-name=precompute-cuadrado-e35-flow
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --time=0-12:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80g
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ecschoye@stud.ntnu.no
#SBATCH --output=/cluster/work/ecschoye/SMCM-MCFNet/.slurm-logs/precompute-cuadrado-e35-%j.out
#SBATCH --error=/cluster/work/ecschoye/SMCM-MCFNet/.slurm-logs/precompute-cuadrado-e35-%j.err

# ---------------------------------------------------------------------------
# Precompute optical flow from Kristian's 35-epoch Cuadrado MCM (edge-aware-L1,
# sw0.0005) once per DSEC-Det frame at 480x640. Writes per-sequence HDF5 under
# outputs/cache/cuadrado_e35_flow_480x640/ for STTP via PrecomputedFlowProvider.
# Controlled swap vs the old epoch_10/sw150 cache (cuadrado_flow_480x640).
#
#   sbatch slurmjobs/run_precompute_cuadrado_e35_flow.slurm
#   sbatch --export=ALL,SEQUENCES="thun_01_a",SPLITS="test" \
#     slurmjobs/run_precompute_cuadrado_e35_flow.slurm   # single-seq smoke
# ---------------------------------------------------------------------------
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/cluster/work/ecschoye/SMCM-MCFNet}"

module load HDF5/1.14.0-iimpi-2023a Anaconda3/2024.02-1
source activate smcm-mcfnet

CONFIG=${CONFIG:-src/smcfnet/config/runs/sttp/cuadrado_e35_precompute.yaml}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/cache/cuadrado_e35_flow_480x640}
SPLITS=${SPLITS:-"train test"}
SEED=${SEED:-0}
SEQUENCES=${SEQUENCES:-}
OVERWRITE=${OVERWRITE:-if-sha-mismatch}

mkdir -p "${OUTPUT_DIR}" .slurm-logs
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PYTHONUNBUFFERED=1

echo "Job ${SLURM_JOB_ID:-?} on $(hostname) @ $(date)"
echo "CONFIG=${CONFIG}  OUTPUT_DIR=${OUTPUT_DIR}  SPLITS=${SPLITS}  OVERWRITE=${OVERWRITE}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

CMD=(
  python3 -m tools.precompute_cuadrado_flow
  --config "${CONFIG}"
  --output-dir "${OUTPUT_DIR}"
  --seed "${SEED}"
  --overwrite "${OVERWRITE}"
)
read -r -a SPLIT_ARRAY <<< "${SPLITS}"
CMD+=(--splits "${SPLIT_ARRAY[@]}")
if [ -n "${SEQUENCES}" ]; then
  read -r -a SEQ_ARRAY <<< "${SEQUENCES}"
  CMD+=(--sequences "${SEQ_ARRAY[@]}")
fi

printf 'Running:'; printf ' %q' "${CMD[@]}"; echo
"${CMD[@]}"
echo "Finished at $(date)"
