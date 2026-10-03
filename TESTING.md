# Testing Guide

This trimmed handoff keeps the test suite intact, but the public-facing testing
instructions below only reference files and workflows that still exist in the
cleaned repository.

## Environment

For tests that read event data, load HDF5 first:

```bash
module load HDF5/1.14.0-iimpi-2023a
module load Anaconda3/2024.02-1
conda activate smcm-mcfnet
```

Prefer `python -m pytest` over bare `pytest` so the active environment is used.

## Fast Smoke Tests

These do not require the full DSEC dataset setup.

```bash
python -m pytest src/smcfnet/test/mcm/ofe/test_mcm_simple.py -v
python -m pytest src/smcfnet/test/test_edum.py -v
python -m pytest src/smcfnet/test/test_sttp_head.py -v
python -m pytest src/smcfnet/test/test_plot_results_smoke.py -v
```

## Dataset And Loader Tests

These require a working DSEC tree and the HDF5 module.

```bash
python -m pytest src/smcfnet/test/dataset/dsec/test_dsec_sequence_index_dataset.py -v
python -m pytest src/smcfnet/test/dataset/dsec/test_dsec_lightweight_mapper.py -v
python -m pytest src/smcfnet/test/dataset/dsec/test_dsec_train_loader_images_targets.py -v
python -m pytest src/smcfnet/test/dataset/dsec/test_dsec_train_loader_e2e.py -v
```

Additional useful data-path tests:

```bash
python -m pytest src/smcfnet/test/test_event_rectifier.py -v
python -m pytest src/smcfnet/test/dataset/dsec/test_dsec_collate_fast.py -v
python -m pytest src/smcfnet/test/dataset/test_gpu_prefetcher_synthetic.py -v
```

## STTP And Tooling Tests

These are useful after touching STTP logic or top-level tool scripts:

```bash
python -m pytest src/smcfnet/test/test_sttp_loss.py -v
python -m pytest src/smcfnet/test/test_sttp_integration.py -v
python -m pytest src/smcfnet/test/test_sttp_baseline_tracker.py -v
python -m pytest src/smcfnet/test/test_ttc_sim.py -v
```

## Config And Provenance Checks

These are good low-cost checks after editing YAML configs or config loading:

```bash
python -m pytest src/smcfnet/test/test_provenance.py -v
python -m pytest src/smcfnet/test/mcm/test_event_representation_pipeline.py -v
```

## Run A Broader CPU-Only Slice

If you want a broader pass without the full dataset-dependent suite:

```bash
python -m pytest \
  src/smcfnet/test/mcm/ofe/test_mcm_simple.py \
  src/smcfnet/test/test_edum.py \
  src/smcfnet/test/test_sttp_head.py \
  src/smcfnet/test/test_sttp_loss.py \
  src/smcfnet/test/test_plot_results_smoke.py \
  src/smcfnet/test/test_provenance.py -v
```

## Run The Full Suite

If the environment and datasets are available, run everything with:

```bash
python -m pytest src/smcfnet/test -v
```

If that is too broad for a first pass, start with the smoke tests and then add
the DSEC loader tests.

## Notes

- Some tests are cluster-oriented and may expect GPU or dataset availability.
- Some tests import top-level modules from `tools/`, so if you trim that folder
  further, rerun the STTP and plotting smoke tests.
- The cleaned handoff keeps a smaller config set, so when adding new tests,
  prefer the curated configs under `src/smcfnet/config/runs/` instead of old
  thesis-specific paths.
