#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=dsec_test
#SBATCH --time=00:15:00
#SBATCH --partition=short
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=outputs/test_%j.txt
#SBATCH --error=outputs/test_%j.err

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
module load HDF5/1.14.0-iimpi-2023a Anaconda3/2024.02-1
conda activate smcfnet
export PYTHONUNBUFFERED=1

# Test with one sequence only
./run_preprocessing.sh train --dry-run
