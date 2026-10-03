# DSEC Dataset Preprocessing

This directory contains tools for rectifying DSEC event camera data and object tracking annotations to match the full resolution rectified RGB images (1440×1080).

## Prerequisites

### HDF5 Compression Plugins

The DSEC `events.h5` files use compression that requires HDF5 external filter plugins. If you get an error like:

```
Can't synchronously read data (can't open directory: .../lib/hdf5/plugin)
```

Install the required plugins:

```bash
conda install -c conda-forge hdf5-external-filter-plugins
```

Or with pip:

```bash
pip install hdf5plugin
```

The script will attempt to auto-detect and configure the plugin path, but manual installation may be needed.

## Overview

The DSEC dataset provides:
- **Events** at 640×480 resolution in distorted event camera space
- **RGB images** at 1440×1080 resolution (available in both distorted and rectified forms)
- **Object tracks** with bounding boxes in distorted event camera coordinates

For training with full resolution RGB images, we need to rectify both events and tracks to match the RGB coordinate space. This preprocessing pipeline does exactly that.

## Files Created

After preprocessing, each sequence will have:
```
sequence_name/
├── events/left/
│   ├── events.h5                    # Original distorted events
│   ├── rectified_events.h5          # ✨ NEW: Rectified events  
│   └── rectify_map.h5
└── object_detections/left/
    ├── tracks.npy                   # Original distorted tracks
    └── rectified_tracks.npy         # ✨ NEW: Rectified tracks
```

## Usage

### 1. Initialize Progress Tracking

First, initialize the progress file to scan all available sequences:

```bash
cd ${PROJECT_ROOT}/src/smcfnet/dataset/dsec/preprocessing

python preprocessor.py \
    --data-root ${DSEC_ROOT} \
    --init-progress
```

This creates `progress.txt` with all train and test sequences.

### 2. Run Preprocessing

Process all pending sequences:
```bash
# Process train split
python preprocessor.py \
    --data-root ${DSEC_ROOT} \
    --split train

# Process test split
python preprocessor.py \
    --data-root ${DSEC_ROOT} \
    --split test
```


### 3. Process Specific Sequences

To process only certain sequences (useful for testing or reprocessing):
```bash
python preprocessor.py \
    --data-root ${DSEC_ROOT} \
    --split train \
    --sequences interlaken_00_c zurich_city_00_a
```

### 4. Skip Events or Tracks

Only process events or tracks:
```bash
# Only rectify events
python preprocessor.py \
    --data-root ${DSEC_ROOT} \
    --split train \
    --skip-tracks

# Only rectify tracks
python preprocessor.py \
    --data-root ${DSEC_ROOT} \
    --split train \
    --skip-events
```

### 5. Dry Run

Preview what would be processed without actually doing anything:
```bash
python preprocessor.py \
    --data-root ${DSEC_ROOT} \
    --split train \
    --dry-run
```

## How It Works

### Event Rectification

1. **Load rectification map**: From `rectify_map.h5` and calibration from `cam_to_cam.yaml`
2. **Compute remapping**: Build lookup table from event coords (640×480) → RGB coords (1440×1080)
3. **Transform events**: For each event (x, y, p, t):
   - Lookup rectified coordinates (x', y') using the remap
   - Validate coordinates are within RGB bounds
   - Filter invalid events
4. **Save**: Store rectified events in same HDF5 format as original

**Statistics**: Typically ~95-98% of events successfully rectify to valid RGB coordinates.

### Track Rectification

1. **Load tracks**: From `tracks.npy` (structured array with t, x, y, w, h, class_id)
2. **Convert bounding boxes**: From (x, y, w, h) to corner format (x1, y1, x2, y2)
3. **Warp corners**: Transform all 4 corners using rectification map
4. **Recompute boxes**: Take axis-aligned bounding box of warped corners
5. **Validate**: Check corners are in bounds and box is non-degenerate
6. **Save**: Store rectified tracks in same numpy format

**Statistics**: Typically ~98-99% of tracks successfully rectify.

## Progress Tracking

The `progress.txt` file tracks which sequences have been processed:

```yaml
jobs:
  train_sequences:
    - interlaken_00_c
    - zurich_city_00_a
    # ... all train sequences
  test_sequences:
    - interlaken_00_a
    # ... all test sequences
finished:
  train_sequences:
    - interlaken_00_c  # ✓ Completed
  test_sequences: []
```

The script automatically:
- Saves progress after each successful sequence
- Skips already-processed sequences
- Allows resuming if interrupted
- Can be reset by manually editing `progress.txt`

## Output Format

### Rectified Events (HDF5)

Same structure as original `events.h5`:
```python
{
    'events': {
        'x': uint16[N],      # X coordinate in [0, 1439]
        'y': uint16[N],      # Y coordinate in [0, 1079]
        'p': int32[N],       # Polarity (-1 or 1)
        't': int64[N]        # Timestamp in microseconds
    },
    't_offset': int64,       # Time offset
    'ms_to_idx': int64[M]    # Millisecond → event index lookup
}
```

### Rectified Tracks (NPY)

Same structured array as original `tracks.npy`:
```python
dtype = [
    ('t', '<i8'),         # Timestamp in microseconds
    ('x', '<f4'),         # Bounding box X (top-left corner)
    ('y', '<f4'),         # Bounding box Y (top-left corner)
    ('w', '<f4'),         # Bounding box width
    ('h', '<f4'),         # Bounding box height
    ('class_id', '<i8')   # Object class ID
]
```

All coordinates are in rectified RGB space (1440×1080).

## Logging

The script provides detailed logging:
```
2026-02-08 14:30:15 [INFO] ================================================================================
2026-02-08 14:30:15 [INFO] Processing sequence: interlaken_00_c
2026-02-08 14:30:15 [INFO] Path: ${DSEC_ROOT}/train/interlaken_00_c
2026-02-08 14:30:16 [INFO] Rectifying events...
2026-02-08 14:30:16 [INFO]   Loading rectification map from rectify_map.h5
2026-02-08 14:30:16 [INFO]   Remap shape: (480, 640, 2), event res: 640×480, RGB res: 1440×1080
2026-02-08 14:30:16 [INFO]   Total events: 15,234,567
2026-02-08 14:30:17 [INFO]   Rectifying events...
2026-02-08 14:30:18 [INFO]   Valid rectified events: 14,897,234 (97.8%)
2026-02-08 14:30:19 [INFO]   ✓ Saved rectified events to rectified_events.h5
2026-02-08 14:30:19 [INFO] Rectifying tracks...
2026-02-08 14:30:19 [INFO]   Loading tracks from tracks.npy
2026-02-08 14:30:19 [INFO]   Total tracks: 1,234
2026-02-08 14:30:19 [INFO]   Rectifying track bounding boxes...
2026-02-08 14:30:19 [INFO]   Valid rectified tracks: 1,215 (98.5%)
2026-02-08 14:30:19 [INFO]   ✓ Saved rectified tracks to rectified_tracks.npy
2026-02-08 14:30:19 [INFO] ✓ Sequence interlaken_00_c processed successfully
```

## Error Handling

The script handles various error cases gracefully:

- **Missing files**: Warns but continues to next sequence
- **Already processed**: Skips existing output files
- **Processing errors**: Logs error and continues to next sequence
- **Interrupted**: Can resume from progress file

## Performance

Processing time per sequence (approximate):
- **Events**: 5-15 seconds (depends on event count, typically 10-50M events/sequence)
- **Tracks**: <1 second (typically 1-5K tracks/sequence)

Total dataset processing time:
- **Train split** (49 sequences): ~8-12 minutes
- **Test split** (13 sequences): ~2-3 minutes

## Validation

After preprocessing, you can validate the output using the test script:

```bash
cd ${PROJECT_ROOT}/src/smcfnet/test
python test_event_rectifier.py
```

This will create visualizations comparing:
- Raw events vs rectified events
- Distorted tracks vs rectified tracks
- Alignment with RGB images

## Troubleshooting

### "Calibration file not found"
Ensure the sequence has `calibration/cam_to_cam.yaml`. This should be present in all DSEC sequences.

### "Rectification map not found"
Ensure the sequence has `events/left/rectify_map.h5`. This is provided by DSEC for the left event camera.

### Very low rectification percentage (<90%)
This may indicate:
- Corrupted rectification map
- Mismatched calibration files
- Events outside camera FOV (rare)

Check the logs for specific error messages.

### Out of memory
Reduce the `max_batch_size` parameter in `rectify_events_h5()` function (default: 1M events).

## Integration with Training

After preprocessing, update your config to use rectified data:

```yaml
data:
  mapper_prefer_distorted: false  # Use rectified RGB images
  mapper_load_rectified: true     # Load rectified events/tracks
  image_height: 1080
  image_width: 1440
```

The dataset mapper will automatically load `rectified_events.h5` and `rectified_tracks.npy` when `mapper_load_rectified: true` and `mapper_prefer_distorted: false`.
