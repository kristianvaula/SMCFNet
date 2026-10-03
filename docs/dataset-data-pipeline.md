# Dataset And Data Pipeline

This repository expects DSEC data to be indexed first and materialized late.
The practical rule is: keep dataset workers cheap, then choose the final event
representation in the collate or prefetch path.

## Dataset YAMLs

The dataset YAMLs under `src/smcfnet/config/dsec/` define:

- `dataset.root`: absolute DSEC root.
- `dataset.camera`: usually `left`.
- `dataset.train_sequences`: sequence names under `${root}/train/`.
- `dataset.val_sequences` or `dataset.test_sequences`: held-out sequences.

The most useful starter files are:

- `dsec.yaml`: main detector split.
- `dsec_debug.yaml`: tiny smoke-test split.
- `dsec_short.yaml`: reduced subset for faster iteration.
- `optical_flow.yaml`: flow-GT subset.
- `optical_flow_train_heldout.yaml`: flow pretraining split with train-held-out validation.

## Expected DSEC Layout

```text
${root}/{train|test}/${sequence}/
  images/
    timestamps.txt
    left/
      rectified/*.png
      distorted/*.png
  events/
    left/
      events.h5
      rectify_map.h5
      rectified_events.h5
      rectified_events_orig_res.h5
  object_detections/
    left/
      tracks.npy
      rectified_tracks.npy
  flow/
    forward_timestamps.txt
    forward/*.png
    backward_timestamps.txt
    backward/*.png
  calibration/
    cam_to_cam.yaml
```

Not every training mode uses every file. Detection mainly needs image
timestamps, events, and tracks. Supervised SOFE pretraining also needs the
`flow/` tree.

## Rectified Versus Distorted Data

There are two coordinate systems in play:

- distorted native event-camera coordinates, roughly `640x480`
- rectified RGB coordinates, `1440x1080`

Important config switches:

- `mapper_prefer_distorted: true` keeps distorted images, events, and tracks.
- `mapper_prefer_distorted: false` uses the rectified view.
- `mapper_rectified_events_resolution: "full"` selects `rectified_events.h5`.
- `mapper_rectified_events_resolution: "orig"` selects `rectified_events_orig_res.h5`.

For detector-side EDUM or IWE fusion, the common setup is rectified RGB with
`mapper_rectified_events_resolution: "orig"`, so events stay at native event
resolution while still matching the rectified camera view.

## Loader Design

The DSEC pipeline is structured in three stages:

1. Sequence indexing builds cheap metadata items such as sequence id, frame id,
   and time window.
2. The lightweight mapper performs actual HDF5 reads, optional image decode,
   optional target construction, and optional flow loading.
3. Collate or prefetch code converts those raw events into the representation
   needed by the model, such as voxels, Cuadrado histograms, or streaming event
   chunks.

That split keeps worker-side overhead low and makes it easier to swap event
representations without changing the on-disk contract.

## Relevant Code Paths

- `src/smcfnet/dataset/dsec/dsec_sequence_index.py`: per-sequence filesystem contract and frame indexing.
- `src/smcfnet/dataset/dsec/dsec_flow_index.py`: flow-GT indexing for supervised motion pretraining.
- `src/smcfnet/dataset/dsec/dsec_lightweight_mapper.py`: event, image, and target fetching.
- `src/smcfnet/model/build.py`: resolves event resolution and chooses the prefetch path.
- `src/smcfnet/dataset/dsec/preprocessing/README.md`: rectification preprocessing details.

## Practical Guidance

- Start with `dsec_debug.yaml` when validating a new config.
- Use `optical_flow_train_heldout.yaml` for supervised SOFE pretraining.
- Use `dsec.yaml` for main detector training unless you intentionally define a
  smaller split.
- If boxes or event overlays look misaligned, check the distorted-versus-
  rectified settings before changing model code.