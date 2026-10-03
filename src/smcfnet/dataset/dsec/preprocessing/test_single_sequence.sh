#!/bin/bash
# Quick test script - process just one sequence to verify everything works

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

USER=$(whoami)

DATA_ROOT="/cluster/work/${USER}/projects/smcm-mcfnet/data"
TEST_SEQUENCE="interlaken_00_a"  # Small sequence for testing

echo "=========================================="
echo "Quick Test: Processing single sequence"
echo "Sequence: $TEST_SEQUENCE"
echo "=========================================="
echo ""

python preprocessor.py \
    --data-root "$DATA_ROOT" \
    --split test \
    --sequences "$TEST_SEQUENCE"

echo ""
echo "=========================================="
echo "Test complete! Check the logs above."
echo "=========================================="
echo ""
echo "If successful, you should see:"
echo "  ✓ Saved rectified events to rectified_events.h5"
echo "  ✓ Saved rectified tracks to rectified_tracks.npy"
echo ""
echo "To process all sequences, run:"
echo "  ./run_preprocessing.sh train"
echo "  ./run_preprocessing.sh test"
