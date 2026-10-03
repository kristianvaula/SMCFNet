#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=smcm-iwe-cache
#SBATCH --time=0-08:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# ---------------------------------------------------------------------------
# Template: precompute detector-facing SMCM IWEs for one trained variant.
#
# This job runs frozen MCM/SMCM inference sequence-by-sequence and writes an
# offline detector cache. Detector training then sets data.precomputed_iwe_dir
# to the produced cache directory; the live MCM path is disabled by config
# resolution while the checkpoint SHA remains bound for compatibility checks.
#
# Required:
#   CONFIG    Training/probe YAML that defines the MCM variant and DSEC data.
#   VARIANT   Stable cache name, usually the experiment/checkpoint selection.
#
# Optional:
#   CHECKPOINT          Overrides model.mcm.pretrained_weights from CONFIG.
#   DATASET_CONFIG      Optional dataset YAML override. Use when CONFIG is the
#                       trained MCM/run config but the cache should be built
#                       over another detector split, e.g. full PEOD train/val.
#   SPLITS              Space-separated split list. Default: "train".
#   OUTPUT_ROOT         Default: outputs/cache/smcm_iwe
#   DTYPE               fp16|float32|uint8. Default: fp16.
#   OVERWRITE           never|if-compatible|always. Default: if-compatible.
#   SEQUENCES           Space-separated allowlist for smoke/subset cache.
#   EXCLUDE_SEQUENCES   Space-separated denylist.
#   DEVICE              Default: cuda.
#   RESET_EACH_FRAME    1|0. Default: 0. Pass --reset-each-frame to the
#                       precompute tool. Use 1 for one-shot Cuadrado caches;
#                       leave 0 for streaming recurrent SMCM caches.
#   MERGE_ALL            1|0. Default: 1. Refresh <OUTPUT_ROOT>/<VARIANT>/all,
#                         a symlink directory merging every split subdirectory
#                         present on disk for this variant. Detector training
#                         points data.precomputed_iwe_dir at this "all" dir,
#                         not at a single split dir: PrecomputedIWEProvider
#                         takes one root and the trainer reuses the same
#                         provider instance for both the train and eval
#                         loaders, which normally read disjoint DSEC splits.
#                         See docs/.caching-readme for the full rationale.
#   SKIP_INTEGRITY_CHECK 1|0. Default: 0. Skip the post-write h5py open +
#                         frame_count check on every file in each processed
#                         split. Leave this on; a corrupted cache file
#                         (truncated write, disk/filesystem fault) opens fine
#                         as a stat-able file but fails h5py.File(...) with
#                         "bad object header version number", and otherwise
#                         surfaces only much later as a mid-epoch detector
#                         training crash on whatever sequence the shuffled
#                         sampler happens to draw first.
#
# CONFIG must be the exact probe/training YAML used to select this variant's
# detector contract (model.mcm.output: type/iwe_channels/iwe_normalize/...),
# not a reconstructed config pointing at the same checkpoint. precompute's
# renderer_version is a hash of that resolved output block, not of code or
# the checkpoint — two configs with identical architecture+checkpoint but a
# different model.mcm.output (e.g. a probe's restated voxel_grid+iwe_* block
# vs. the generic rgbe_<arch>_iwe_edum_cmm_m.yaml family base's iwe_liu_rgb)
# silently produce different renderer_version hashes. This is invisible
# until detector training opens two files from the "same" cache with
# different hashes, at which point PrecomputedIWEProvider raises a renderer
# mismatch. This matters most when backfilling/regenerating a single
# sequence after the fact (e.g. SEQUENCES=<one seq> OVERWRITE=always against
# an existing cache) — always pass the same CONFIG that built the rest of
# that cache, never a reconstructed equivalent.
#
# Examples:
#   # Full train cache for the Cuadrado iter6 probe:
#   sbatch --export=ALL,\
#CONFIG=src/smcfnet/config/runs/baselines/train/dsec_probe/probe_iter6_cuadrado_g07_e35.yaml,\
#VARIANT=probe_iter6_cuadrado_g07_e35 \
#     slurmjobs/cache/run_precompute_smcm_iwe_template.slurm
#
#   # Two-sequence timing smoke:
#   sbatch --time=01:00:00 --export=ALL,\
#CONFIG=src/smcfnet/config/runs/baselines/train/dsec_probe/probe_iter6_cuadrado_g07_e35.yaml,\
#VARIANT=probe_iter6_cuadrado_g07_e35,\
#SEQUENCES="thun_00_a interlaken_00_c",OVERWRITE=always \
#     slurmjobs/cache/run_precompute_smcm_iwe_template.slurm
#
#   # Backfill one corrupted/missing sequence in an existing cache (must use
#   # the same CONFIG that built the rest of the cache — see note above):
#   sbatch --export=ALL,\
#CONFIG=src/smcfnet/config/runs/baselines/train/dsec_probe/probe_iter6_cuadrado_g07_e35.yaml,\
#VARIANT=probe_iter6_cuadrado_g07_e35,SPLITS=train,\
#SEQUENCES=interlaken_00_e,OVERWRITE=always \
#     slurmjobs/cache/run_precompute_smcm_iwe_template.slurm
# ---------------------------------------------------------------------------

set -euo pipefail

JOB_ID="${SLURM_JOB_ID:-manual}"
WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${WORKDIR}"

CURRENT_USER="$(whoami)"
REPO_ROOT_DEFAULT="/cluster/work/${CURRENT_USER}/projects/smcm-mcfnet/SMCM-MCFNet"
if [ ! -f "pyproject.toml" ] && [ -d "${REPO_ROOT_DEFAULT}" ]; then
  cd "${REPO_ROOT_DEFAULT}"
fi
if [ ! -f "pyproject.toml" ]; then
  echo "ERROR: cannot locate SMCM-MCFNet repo root."
  exit 2
fi

CONFIG="${CONFIG:-}"
CHECKPOINT="${CHECKPOINT:-}"
DATASET_CONFIG="${DATASET_CONFIG:-}"
VARIANT="${VARIANT:-}"
SPLITS="${SPLITS:-train}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/cache/smcm_iwe}"
DTYPE="${DTYPE:-fp16}"
OVERWRITE="${OVERWRITE:-if-compatible}"
SEQUENCES="${SEQUENCES:-}"
EXCLUDE_SEQUENCES="${EXCLUDE_SEQUENCES:-}"
DEVICE="${DEVICE:-cuda}"
RESET_EACH_FRAME="${RESET_EACH_FRAME:-0}"
MERGE_ALL="${MERGE_ALL:-1}"
SKIP_INTEGRITY_CHECK="${SKIP_INTEGRITY_CHECK:-0}"

if [ -z "${CONFIG}" ]; then
  echo "ERROR: CONFIG must be set to the YAML used by the trained variant."
  exit 2
fi
if [ -z "${VARIANT}" ]; then
  echo "ERROR: VARIANT must be set to a stable cache name."
  exit 2
fi
if [ ! -f "${CONFIG}" ]; then
  echo "ERROR: CONFIG not found: ${CONFIG}"
  exit 2
fi
if [ -n "${CHECKPOINT}" ] && [ ! -f "${CHECKPOINT}" ]; then
  echo "ERROR: CHECKPOINT not found: ${CHECKPOINT}"
  exit 2
fi
if [ -n "${DATASET_CONFIG}" ] && [ ! -f "${DATASET_CONFIG}" ]; then
  echo "ERROR: DATASET_CONFIG not found: ${DATASET_CONFIG}"
  exit 2
fi

RUN_DIR="${OUTPUT_ROOT}/${VARIANT}/slurm_logs"
mkdir -p "${RUN_DIR}"
RUN_STDOUT="${RUN_DIR}/stdout_${JOB_ID}.log"
RUN_STDERR="${RUN_DIR}/stderr_${JOB_ID}.log"
exec >>"${RUN_STDOUT}" 2>>"${RUN_STDERR}"

# Module/conda init can reference unset module variables under set -u.
export xml_catalog_files_libxml2="${xml_catalog_files_libxml2-}"
set +u
module purge
module load HDF5/1.14.0-iimpi-2023a
module load Anaconda3/2024.02-1
source "/cluster/work/${CURRENT_USER}/miniconda3/etc/profile.d/conda.sh"
conda activate smcm-mcfnet || { echo "Failed to activate conda env 'smcm-mcfnet'"; exit 1; }
set -u

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export BLOSC_NTHREADS=1
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

echo "========================================"
echo "SMCM IWE Cache Precompute"
echo "========================================"
echo "Job ID:        ${JOB_ID}"
echo "Host:          $(hostname)"
echo "Date:          $(date)"
echo "Repo:          ${PWD}"
echo "CONFIG:        ${CONFIG}"
echo "CHECKPOINT:    ${CHECKPOINT:-<from config>}"
echo "DATASET_CONFIG:${DATASET_CONFIG:-<from config>}"
echo "VARIANT:       ${VARIANT}"
echo "SPLITS:        ${SPLITS}"
echo "OUTPUT_ROOT:   ${OUTPUT_ROOT}"
echo "DTYPE:         ${DTYPE}"
echo "OVERWRITE:     ${OVERWRITE}"
echo "SEQUENCES:     ${SEQUENCES:-<all>}"
echo "EXCLUDE:       ${EXCLUDE_SEQUENCES:-<none>}"
echo "DEVICE:        ${DEVICE}"
echo "RESET_EACH_FRAME: ${RESET_EACH_FRAME}"
echo "MERGE_ALL:     ${MERGE_ALL}"
echo "SKIP_INTEGRITY_CHECK: ${SKIP_INTEGRITY_CHECK}"
echo "Stdout:        ${RUN_STDOUT}"
echo "Stderr:        ${RUN_STDERR}"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L || true

read -r -a SPLIT_ARRAY <<< "${SPLITS}"
for SPLIT in "${SPLIT_ARRAY[@]}"; do
  OUT="${OUTPUT_ROOT}/${VARIANT}/${SPLIT}"
  mkdir -p "${OUT}"
  START="$(date +%s)"

  CMD=(
    python -m tools.precompute_smcm_iwe
    --config "${CONFIG}"
    --variant "${VARIANT}"
    --output-dir "${OUT}"
    --split "${SPLIT}"
    --dtype "${DTYPE}"
    --overwrite "${OVERWRITE}"
    --device "${DEVICE}"
  )
  if [ -n "${CHECKPOINT}" ]; then
    CMD+=(--checkpoint "${CHECKPOINT}")
  fi
  if [ -n "${DATASET_CONFIG}" ]; then
    CMD+=(--dataset-config "${DATASET_CONFIG}")
  fi
  if [ "${RESET_EACH_FRAME}" = "1" ]; then
    CMD+=(--reset-each-frame)
  fi
  if [ -n "${SEQUENCES}" ]; then
    read -r -a SEQUENCE_ARRAY <<< "${SEQUENCES}"
    CMD+=(--sequences "${SEQUENCE_ARRAY[@]}")
  fi
  if [ -n "${EXCLUDE_SEQUENCES}" ]; then
    read -r -a EXCLUDE_ARRAY <<< "${EXCLUDE_SEQUENCES}"
    CMD+=(--exclude-sequences "${EXCLUDE_ARRAY[@]}")
  fi

  echo ""
  echo "===== Split ${SPLIT} ====="
  printf ' %q' "${CMD[@]}"
  echo ""

  "${CMD[@]}"

  END="$(date +%s)"
  ELAPSED="$((END - START))"
  export OUT SPLIT ELAPSED
  python - <<'PY'
import json
import os
from pathlib import Path

out = Path(os.environ["OUT"])
split = os.environ["SPLIT"]
elapsed = int(os.environ["ELAPSED"])
manifest_path = out / "manifest.json"
manifest = json.load(open(manifest_path))
frames = sum(int(s["frame_count"]) for s in manifest["sequences"])
fps = frames / max(elapsed, 1)
print("===== Cache timing summary =====")
print(f"split={split}")
print(f"manifest={manifest_path}")
print(f"frames={frames}")
print(f"elapsed_s={elapsed}")
print(f"frames_per_s={fps:.4f}")
PY

  if [ "${SKIP_INTEGRITY_CHECK}" != "1" ]; then
    echo ""
    echo "===== Integrity check: ${OUT} ====="
    export OUT
    python - <<'PY'
import os
import sys
from pathlib import Path

import h5py

out = Path(os.environ["OUT"])
files = sorted(out.glob("*.h5"))
bad = []
for f in files:
    try:
        with h5py.File(f, "r") as fh:
            if "iwe" not in fh:
                bad.append((str(f), "missing 'iwe' dataset"))
                continue
            shape = fh["iwe"].shape
            n_attr = fh.attrs.get("frame_count", None)
            if n_attr is not None and int(shape[0]) != int(n_attr):
                bad.append((str(f), f"frame_count mismatch: dataset={shape[0]} attr={int(n_attr)}"))
    except Exception as e:
        bad.append((str(f), str(e)))

if bad:
    print(f"INTEGRITY CHECK FAILED: {len(bad)}/{len(files)} unreadable/corrupt file(s)")
    for f, e in bad:
        print(f"  {f}: {e}")
    print("Re-run with OVERWRITE=always SEQUENCES=<bad sequence names> to backfill, "
          "using the SAME CONFIG that built the rest of this cache (see header notes).")
    sys.exit(1)
print(f"Integrity check passed: {len(files)} file(s) OK")
PY
  fi
done

if [ "${MERGE_ALL}" = "1" ]; then
  ALL_DIR="${OUTPUT_ROOT}/${VARIANT}/all"
  mkdir -p "${ALL_DIR}"
  echo ""
  echo "===== Merging all available splits into ${ALL_DIR} ====="
  shopt -s nullglob
  for SPLIT_DIR in "${OUTPUT_ROOT}/${VARIANT}"/*/; do
    SPLIT_NAME="$(basename "${SPLIT_DIR%/}")"
    case "${SPLIT_NAME}" in
      all|slurm_logs) continue ;;
    esac
    for SEQ_FILE in "${SPLIT_DIR}"*.h5; do
      [ -e "${SEQ_FILE}" ] || continue
      ln -sfn "../${SPLIT_NAME}/$(basename "${SEQ_FILE}")" "${ALL_DIR}/$(basename "${SEQ_FILE}")"
    done
  done
  shopt -u nullglob
  MERGED_COUNT="$(find "${ALL_DIR}" -maxdepth 1 -name '*.h5' | wc -l)"
  echo "Merged ${MERGED_COUNT} sequence symlink(s) into ${ALL_DIR}"
  echo "Point data.precomputed_iwe_dir at this directory, not at a single split dir."
fi

echo ""
echo "Finished at $(date)"
