#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=dsec_preprocess
#SBATCH --time=0-8:00:00
#SBATCH --partition=CPUQ
#SBATCH --mem=256G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --output=outputs/preprocess_%j.txt
#SBATCH --error=outputs/preprocess_%j.err

WORKDIR=${SLURM_SUBMIT_DIR:-$(pwd)}
cd "${WORKDIR}"
echo "Running from: ${WORKDIR}"
echo "Job name: ${SLURM_JOB_NAME}"
echo "Job ID:   ${SLURM_JOB_ID}"
echo "Nodes:    ${SLURM_JOB_NODELIST}"
echo "Host:     $(hostname)"
echo "Date:     $(date)"

module purge
module load HDF5/1.14.0-iimpi-2023a
module load Anaconda3/2024.02-1

conda activate smcfnet || { echo "Failed to activate conda env 'smcfnet'"; exit 1; }

python - <<'PY'
import sys, h5py, numpy
print("Python:", sys.executable)
print("H5py:", h5py.__version__)
print("Numpy:", numpy.__version__)
PY

MODE="${1:-both}"
DRY_RUN="${2:-}"

mkdir -p outputs

# Parallel workers for sequence-level parallelism
WORKERS="${SLURM_CPUS_PER_TASK:-32}"

# Limit per-worker thread count to avoid oversubscription
# Each worker spawns numpy/HDF5 threads; total threads = WORKERS * OMP_NUM_THREADS
THREADS_PER_WORKER=$(( SLURM_CPUS_PER_TASK / WORKERS ))
if [ "$THREADS_PER_WORKER" -lt 1 ]; then THREADS_PER_WORKER=1; fi

export OMP_NUM_THREADS=$THREADS_PER_WORKER
export MKL_NUM_THREADS=$THREADS_PER_WORKER
export OPENBLAS_NUM_THREADS=$THREADS_PER_WORKER
export NUMEXPR_NUM_THREADS=$THREADS_PER_WORKER
export WORKERS
export PYTHONUNBUFFERED=1

echo "Starting preprocessing at $(date)"
echo "Workers: ${WORKERS}, threads per worker: ${THREADS_PER_WORKER}"

if [ "${DRY_RUN}" = "--dry-run" ]; then
    ./run_preprocessing.sh "${MODE}" --dry-run
else
    ./run_preprocessing.sh "${MODE}"
fi

EXIT_CODE=$?

echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Preprocessing completed successfully!"
else
    echo "Preprocessing failed with exit code: $EXIT_CODE"
fi
echo "Finished at $(date)"
echo "=========================================="

exit $EXIT_CODE
