#!/bin/bash
# Convenience script for running DSEC preprocessing

set -e  # Exit on error

# Activate conda environment
# eval "$(conda shell.bash hook)"
# conda activate smcm-mcfnet

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Data root (adjust if needed)
USER=$(whoami)
DATA_ROOT="/cluster/work/${USER}/datasets/dsec"

echo "=========================================="
echo "DSEC Dataset Preprocessing"
echo "=========================================="
echo ""

usage() {
    echo "Usage: $0 [train|test|both|reset] [--dry-run]"
    echo ""
    echo "Examples:"
    echo "  $0 train"
    echo "  $0 test --dry-run"
    echo "  $0 both"
    echo "  $0 reset --dry-run"
}

# Parse command line arguments
MODE="${1:-train}"  # Default to train
DRY_RUN="${2:-}"

if [ "$MODE" = "-h" ] || [ "$MODE" = "--help" ]; then
    usage
    exit 0
fi

if [ "$MODE" = "reset" ]; then
    echo "Resetting preprocessing outputs and progress..."
    if [ "$DRY_RUN" = "--dry-run" ]; then
        echo "Running in DRY RUN mode..."
        python preprocessor.py --data-root "$DATA_ROOT" --reset --dry-run
    else
        python preprocessor.py --data-root "$DATA_ROOT" --reset
    fi

    echo ""
    echo "=========================================="
    echo "Done!"
    echo "=========================================="
    exit 0
fi

# Check if progress file exists (for processing modes)
if [ ! -f "progress.txt" ]; then
    echo "Progress file not found. Initializing..."
    python preprocessor.py --data-root "$DATA_ROOT" --init-progress
    echo ""
fi

if [ "$MODE" = "both" ]; then
    SPLITS=("train" "test")
elif [ "$MODE" = "train" ] || [ "$MODE" = "test" ]; then
    SPLITS=("$MODE")
else
    echo "Unknown mode: $MODE"
    echo ""
    usage
    exit 1
fi

if [ "$DRY_RUN" = "--dry-run" ]; then
    echo "Running in DRY RUN mode..."
fi

WORKERS="${WORKERS:-1}"

for SPLIT in "${SPLITS[@]}"; do
    echo "Processing $SPLIT split with $WORKERS workers..."
    if [ "$DRY_RUN" = "--dry-run" ]; then
        python preprocessor.py \
            --data-root "$DATA_ROOT" \
            --split "$SPLIT" \
            --workers "$WORKERS" \
            --dry-run
    else
        python preprocessor.py \
            --data-root "$DATA_ROOT" \
            --split "$SPLIT" \
            --workers "$WORKERS"
    fi
done

echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="
