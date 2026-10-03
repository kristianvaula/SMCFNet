"""Preprocessor for rectifying DSEC events and tracks to rectified RGB space.

Current train and test data structure:
 
│   ├── interlaken_01_a
│   │   ├── calibration
│   │   │   ├── cam_to_cam.yaml
│   │   │   └── cam_to_lidar.yaml
│   │   ├── events
│   │   │   ├── left
│   │   │   │   ├── events.h5
│   │   │   │   ├── rectified_events.h5            ** Created by this script **
│   │   │   │   ├── rectified_events_orig_res.h5   ** Created by this script **
│   │   │   │   └── rectify_map.h5
│   │   │   └── right
│   │   ├── images
│   │   │   ├── left
│   │   │   │   ├── distorted
│   │   │   │   └── rectified
│   │   │   ├── right
│   │   │   └── timestamps.txt
│   │   └── object_detections
│   │       └── left
│   │           ├── tracks.npy
│   │           └── rectified_tracks.npy   ** Created by this script **

In this script we are going to run two different preprocessing jobs of the whole dataset.
We will go through each sequence and (1) fetch events and (2) tracks.npy, rectify them to fit the rectified
image frame and save them as 'rectified_events.h5', optionally 'rectified_events_orig_res.h5',
and 'rectified_tracks.npy'.
This way we can simply choose to load the rectified data if we train/test on full resolution images. 
"""

import sys
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse

import yaml
import h5py
import hdf5plugin  # Required for DSEC events.h5 blosc compression
import numpy as np

# Fix HDF5 plugin path issue - DSEC events use compression that requires plugins
if 'HDF5_PLUGIN_PATH' not in os.environ:
    # Try to find hdf5 plugin directory from conda environment
    possible_paths = []
    
    # Check if we're in a conda environment
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        possible_paths.append(Path(conda_prefix) / 'lib' / 'hdf5' / 'plugin')
        possible_paths.append(Path(conda_prefix) / 'share' / 'hdf5_plugins')
    
    # Also check common locations
    possible_paths.append(Path.home() / 'miniconda3' / 'envs' / 'smcm-mcfnet' / 'lib' / 'hdf5' / 'plugin')
    possible_paths.append(Path('/usr') / 'lib' / 'x86_64-linux-gnu' / 'hdf5' / 'plugins')
    
    # Use first existing path, or create the first conda path
    plugin_dir = None
    for path in possible_paths:
        if path.exists():
            plugin_dir = path
            break
    
    if plugin_dir is None and conda_prefix:
        # Create the plugin directory in conda env
        plugin_dir = Path(conda_prefix) / 'lib' / 'hdf5' / 'plugin'
        try:
            plugin_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
    
    if plugin_dir:
        os.environ['HDF5_PLUGIN_PATH'] = str(plugin_dir)

# Add parent directory to path for imports
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Add dsec-det to path
_DSEC_DET_SRC = (_REPO_ROOT.parent / "dsec-det" / "src").resolve()
if str(_DSEC_DET_SRC) not in sys.path:
    sys.path.insert(0, str(_DSEC_DET_SRC))

from smcfnet.dataset.dsec.io import yaml_file_to_dict, h5_file_to_dict
from smcfnet.dataset.dsec.remapping import compute_remapping
from smcfnet.dataset.dsec import build_dsec_sequence_index_datasets


# Constants
EV_W, EV_H = 640, 480  # Event camera resolution
RGB_W, RGB_H = 1440, 1080  # Rectified RGB resolution
RECTIFIED_EVENTS_FULL_NAME = "rectified_events.h5"
RECTIFIED_EVENTS_ORIG_RES_NAME = "rectified_events_orig_res.h5"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ==================== Progress Tracking ====================

def _scan_sequences(data_root: Path) -> Tuple[List[str], List[str]]:
    train_dir = data_root / "train"
    test_dir = data_root / "test"

    train_seqs = sorted([d.name for d in train_dir.iterdir() if d.is_dir()]) if train_dir.exists() else []
    test_seqs = sorted([d.name for d in test_dir.iterdir() if d.is_dir()]) if test_dir.exists() else []
    return train_seqs, test_seqs


def _initialize_progress_dict(data_root: Path) -> Dict[str, Dict[str, List[str]]]:
    train_seqs, test_seqs = _scan_sequences(data_root)
    return {
        "jobs": {
            "train_sequences": train_seqs,
            "test_sequences": test_seqs,
        },
        "finished": {
            "train_sequences": [],
            "test_sequences": [],
        },
    }


def _delete_rectified_outputs(data_root: Path, dry_run: bool = False) -> Tuple[int, int]:
    """Delete all generated rectified outputs under the dataset root.

    Deletes:
      - events/left/rectified_events.h5
      - object_detections/left/rectified_tracks.npy
      - object_detections/left/rectified_tracks.h5 (if present)
      - object_detections/left/recitifed_tracks.h5 (typo-variant, if present)

    Returns:
        (deleted_files, missing_files)
    """
    deleted = 0
    missing = 0

    for split in ("train", "test"):
        split_dir = data_root / split
        if not split_dir.exists():
            continue

        for seq_path in sorted([p for p in split_dir.iterdir() if p.is_dir()]):
            candidates = [
                seq_path / "events" / "left" / RECTIFIED_EVENTS_FULL_NAME,
                seq_path / "events" / "left" / RECTIFIED_EVENTS_ORIG_RES_NAME,
                seq_path / "object_detections" / "left" / "rectified_tracks.npy",
                seq_path / "object_detections" / "left" / "rectified_tracks.h5",
                seq_path / "object_detections" / "left" / "recitifed_tracks.h5",
            ]

            for fp in candidates:
                if fp.exists():
                    logger.info(f"Deleting: {fp}")
                    if not dry_run:
                        try:
                            fp.unlink()
                        except Exception as e:
                            logger.error(f"Failed to delete {fp}: {e}")
                            raise
                    deleted += 1
                else:
                    missing += 1

    return deleted, missing

def load_progress(progress_file: Path) -> Dict:
    """Load progress tracking file."""
    if not progress_file.exists():
        logger.warning(f"Progress file not found: {progress_file}")
        return {
            "jobs": {
                "train_sequences": [],
                "test_sequences": []
            },
            "finished": {
                "train_sequences": [],
                "test_sequences": []
            }
        }
    
    with progress_file.open('r') as f:
        progress = yaml.safe_load(f)
    
    # Handle None values in YAML (empty lists that were saved as just ':')
    if "jobs" not in progress or progress["jobs"] is None:
        progress["jobs"] = {}
    if "finished" not in progress or progress["finished"] is None:
        progress["finished"] = {}
    
    # Ensure all lists exist and are not None
    for key in ["train_sequences", "test_sequences"]:
        if key not in progress["jobs"] or progress["jobs"][key] is None:
            progress["jobs"][key] = []
        if key not in progress["finished"] or progress["finished"][key] is None:
            progress["finished"][key] = []
    
    return progress


def save_progress(progress_file: Path, progress: Dict) -> None:
    """Save progress tracking file."""
    # Ensure empty lists are saved as [] not None
    for section in ["jobs", "finished"]:
        if section in progress and progress[section] is not None:
            for key in ["train_sequences", "test_sequences"]:
                if key in progress[section] and progress[section][key] is None:
                    progress[section][key] = []
    
    with progress_file.open('w') as f:
        yaml.safe_dump(progress, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Progress saved to {progress_file}")


def mark_sequence_finished(progress: Dict, split: str, sequence: str) -> None:
    """Mark a sequence as finished in the progress dict."""
    split_key = f"{split}_sequences"
    
    # Ensure the finished list exists and is not None
    if "finished" not in progress or progress["finished"] is None:
        progress["finished"] = {}
    if split_key not in progress["finished"] or progress["finished"][split_key] is None:
        progress["finished"][split_key] = []
    
    if sequence not in progress["finished"][split_key]:
        progress["finished"][split_key].append(sequence)


def get_pending_sequences(progress: Dict, split: str) -> List[str]:
    """Get list of sequences that haven't been processed yet."""
    split_key = f"{split}_sequences"
    all_seqs = progress["jobs"][split_key]
    finished_seqs = progress["finished"][split_key]
    return [seq for seq in all_seqs if seq not in finished_seqs]


# ==================== Event Rectification ====================

def rectify_events_h5(
    events_h5_path: Path,
    rectify_map_h5_path: Path,
    cam_to_cam_path: Path,
    output_h5_path: Path,
    max_batch_size: int = 1_000_000,
    output_resolution: str = "full",
) -> Tuple[int, int]:
    """Rectify events from distorted event space to rectified space.
    
    Args:
        events_h5_path: Path to input events.h5
        rectify_map_h5_path: Path to rectify_map.h5
        cam_to_cam_path: Path to cam_to_cam.yaml
        output_h5_path: Path to save rectified_events.h5
        max_batch_size: Process events in batches to manage memory
        output_resolution: "full" writes 1440x1080 rectified RGB coordinates.
            "orig" writes 640x480 native-resolution rectified coordinates for EDUM.
        
    Returns:
        Tuple of (total_events, valid_events)
    """
    output_resolution = str(output_resolution).lower()
    if output_resolution in {"native", "orig-res", "original", "event"}:
        output_resolution = "orig"
    if output_resolution in {"rgb", "full-res"}:
        output_resolution = "full"
    if output_resolution not in {"full", "orig"}:
        raise ValueError(f"output_resolution must be 'full' or 'orig', got {output_resolution!r}")

    out_w, out_h = (RGB_W, RGB_H) if output_resolution == "full" else (EV_W, EV_H)
    scale_x = float(out_w) / float(RGB_W)
    scale_y = float(out_h) / float(RGB_H)

    logger.info(f"  Loading rectification map from {rectify_map_h5_path.name}")
    
    # Load calibration and rectification map
    calib = yaml_file_to_dict(cam_to_cam_path)
    rectification_map = h5_file_to_dict(rectify_map_h5_path)
    remap = compute_remapping(calib, rectification_map)
    
    H_ev, W_ev = remap.shape[:2]
    logger.info(
        f"  Remap shape: {remap.shape}, event res: {W_ev}×{H_ev}, "
        f"RGB res: {RGB_W}×{RGB_H}, output res: {out_w}×{out_h}"
    )
    
    # Open input events file
    try:
        h5_in = h5py.File(events_h5_path, 'r')
    except Exception as e:
        logger.error(f"  Failed to open events file: {e}")
        logger.info(f"  Hint: This may be due to HDF5 blosc compression. Try installing: pip install hdf5plugin")
        raise
    
    try:
        events_group = h5_in['events']
        t_offset = h5_in['t_offset'][()]
        ms_to_idx_src = h5_in['ms_to_idx'][:]

        total_events = int(events_group['x'].shape[0])
        logger.info(f"  Total events: {total_events:,}")

        # Process events in batches to avoid OOM.
        # IMPORTANT: do not accumulate all batches in RAM; stream-write to HDF5.
        logger.info(f"  Processing events in batches of {max_batch_size:,}...")

        num_batches = (total_events + max_batch_size - 1) // max_batch_size if total_events > 0 else 0
        events_filtered_fov = 0
        events_filtered_invalid = 0

        # Pre-size ms_to_idx table based on *source* max timestamp; rectified output is a subset.
        if total_events > 0:
            max_ms_est = int(events_group['t'][-1] // 1000)
        else:
            max_ms_est = 0

        logger.info(f"  Writing to {output_h5_path.name} (streaming; copying HDF5 filter settings from source)")

        with h5py.File(output_h5_path, 'w') as h5_out:
            events_out = h5_out.create_group('events')

            x_kw = _copy_h5_filter_and_chunking(events_group['x'])
            y_kw = _copy_h5_filter_and_chunking(events_group['y'])
            p_kw = _copy_h5_filter_and_chunking(events_group['p'])
            t_kw = _copy_h5_filter_and_chunking(events_group['t'])

            # Ensure chunking exists for extendable datasets (required by HDF5 for efficient resize).
            if "chunks" not in x_kw or x_kw["chunks"] is None:
                x_kw["chunks"] = (min(max_batch_size, 1_000_000),)
            if "chunks" not in y_kw or y_kw["chunks"] is None:
                y_kw["chunks"] = x_kw["chunks"]
            if "chunks" not in p_kw or p_kw["chunks"] is None:
                p_kw["chunks"] = x_kw["chunks"]
            if "chunks" not in t_kw or t_kw["chunks"] is None:
                t_kw["chunks"] = x_kw["chunks"]

            x_ds = events_out.create_dataset('x', shape=(0,), maxshape=(None,), dtype='<u2', **x_kw)
            y_ds = events_out.create_dataset('y', shape=(0,), maxshape=(None,), dtype='<u2', **y_kw)
            p_ds = events_out.create_dataset('p', shape=(0,), maxshape=(None,), dtype='<u1', **p_kw)  # uint8 like original
            t_ds = events_out.create_dataset('t', shape=(0,), maxshape=(None,), dtype='<u4', **t_kw)  # uint32 like original

            # Copy metadata
            h5_out.create_dataset('t_offset', data=t_offset)
            h5_out.attrs["coordinate_frame"] = "rectified_rgb_full" if output_resolution == "full" else "rectified_rgb_orig_res"
            h5_out.attrs["output_width"] = int(out_w)
            h5_out.attrs["output_height"] = int(out_h)
            h5_out.attrs["rectified_rgb_width"] = int(RGB_W)
            h5_out.attrs["rectified_rgb_height"] = int(RGB_H)

            # Streaming ms_to_idx computation.
            # We fill first-occurrence indices for ms values that appear, then backfill missing ms.
            ms_to_idx_out = np.full(max_ms_est + 2, -1, dtype=np.int64)

            written = 0

            for batch_idx in range(num_batches):
                start_idx = batch_idx * max_batch_size
                end_idx = min(start_idx + max_batch_size, total_events)

                if batch_idx % 100 == 0:  # Log progress every 100 batches
                    logger.info(
                        f"    Batch {batch_idx + 1}/{num_batches} ({100.0 * start_idx / total_events:.1f}%)"
                    )

                # Load batch
                x = events_group['x'][start_idx:end_idx]
                y = events_group['y'][start_idx:end_idx]
                p = events_group['p'][start_idx:end_idx]
                t = events_group['t'][start_idx:end_idx]

                # Filter events outside event camera FOV
                valid_mask = (x >= 0) & (x < W_ev) & (y >= 0) & (y < H_ev)
                events_filtered_fov += int((~valid_mask).sum())

                x_valid = x[valid_mask]
                y_valid = y[valid_mask]
                p_valid = p[valid_mask]
                t_valid = t[valid_mask]

                if x_valid.size == 0:
                    continue

                # Lookup rectified coordinates
                # (Use intp for indexing to reduce memory overhead vs int64.)
                xy_rect = remap[y_valid.astype(np.intp, copy=False), x_valid.astype(np.intp, copy=False)]

                # Filter events with non-finite rectified coordinates
                finite_mask = np.isfinite(xy_rect[:, 0]) & np.isfinite(xy_rect[:, 1])
                events_filtered_invalid += int((~finite_mask).sum())

                xy_rect_valid = xy_rect[finite_mask]
                p_batch = p_valid[finite_mask]
                t_batch = t_valid[finite_mask]

                if xy_rect_valid.shape[0] == 0:
                    continue

                # Convert rectified RGB coordinates to the requested output grid.
                # The orig-res variant keeps rectified geometry but stores coordinates
                # on the native event grid expected by EDUM.
                if output_resolution == "full":
                    x_rect = np.rint(xy_rect_valid[:, 0])
                    y_rect = np.rint(xy_rect_valid[:, 1])
                else:
                    x_rect = np.floor(xy_rect_valid[:, 0] * scale_x)
                    y_rect = np.floor(xy_rect_valid[:, 1] * scale_y)
                x_rect = np.clip(x_rect, 0, out_w - 1).astype(np.uint16, copy=False)
                y_rect = np.clip(y_rect, 0, out_h - 1).astype(np.uint16, copy=False)

                n = int(x_rect.shape[0])

                # Append to HDF5 datasets
                new_size = written + n
                x_ds.resize((new_size,))
                y_ds.resize((new_size,))
                p_ds.resize((new_size,))
                t_ds.resize((new_size,))

                x_ds[written:new_size] = x_rect
                y_ds[written:new_size] = y_rect
                p_ds[written:new_size] = p_batch
                t_ds[written:new_size] = t_batch

                # Update ms_to_idx first-occurrence table for this batch.
                t_ms_batch = (t_batch // 1000).astype(np.int64, copy=False)
                if t_ms_batch.size > 0:
                    # Find first index of each ms inside this batch (t_ms is non-decreasing).
                    change_idx = np.nonzero(np.diff(t_ms_batch))[0] + 1
                    starts = np.concatenate((np.array([0], dtype=np.int64), change_idx.astype(np.int64, copy=False)))
                    ms_vals = t_ms_batch[starts]
                    for local_start, ms_val in zip(starts.tolist(), ms_vals.tolist()):
                        ms_val_int = int(ms_val)
                        if 0 <= ms_val_int < (ms_to_idx_out.shape[0] - 1) and ms_to_idx_out[ms_val_int] == -1:
                            ms_to_idx_out[ms_val_int] = written + int(local_start)

                written = new_size

            valid_events = int(written)

            # Log filtering statistics
            if events_filtered_fov > 0:
                logger.info(f"  Filtered {events_filtered_fov:,} events outside event camera FOV")
            if events_filtered_invalid > 0:
                logger.info(f"  Filtered {events_filtered_invalid:,} events with non-finite rectified coordinates")

            logger.info(f"  Valid rectified events: {valid_events:,} ({100.0 * valid_events / total_events:.1f}%)")

            # Finalize ms_to_idx
            ms_kw = _copy_h5_filter_and_chunking(h5_in['ms_to_idx'])
            if valid_events > 0:
                logger.info(f"  Computing ms_to_idx lookup table...")
                # Sentinel: insertion point for (max_ms + 1) is always end.
                ms_to_idx_out[-1] = valid_events
                # Backfill missing milliseconds to match `np.searchsorted(t_ms, all_ms, side='left')`.
                for i in range(ms_to_idx_out.shape[0] - 2, -1, -1):
                    if ms_to_idx_out[i] == -1:
                        ms_to_idx_out[i] = ms_to_idx_out[i + 1]

                logger.info(f"  ✓ Created ms_to_idx table: {len(ms_to_idx_out):,} entries")
                h5_out.create_dataset('ms_to_idx', data=ms_to_idx_out.astype(np.uint64), dtype='<u8', **ms_kw)
            else:
                # No output events; fall back to source ms_to_idx (keeps file structurally valid).
                h5_out.create_dataset('ms_to_idx', data=ms_to_idx_src, dtype='<u8', **ms_kw)

        logger.info(f"  ✓ Saved rectified events to {output_h5_path}")
    finally:
        h5_in.close()
        
    return total_events, valid_events


def _copy_h5_filter_and_chunking(src_ds: h5py.Dataset) -> Dict[str, object]:
    """Return kwargs to replicate chunking + filter pipeline from a source dataset.

    DSEC `events.h5` uses the Blosc (filter id 32001) plugin. We copy the raw
    `cd_values` exactly to avoid subtle ordering/meaning mistakes.
    """
    kwargs: Dict[str, object] = {}
    if src_ds.chunks is not None:
        kwargs["chunks"] = src_ds.chunks

    plist = src_ds.id.get_create_plist()
    nfilters = plist.get_nfilters()
    for i in range(nfilters):
        filter_id, _flags, cd_values, _name = plist.get_filter(i)
        # Copy only the compression filter. (DSEC uses blosc id=32001.)
        if int(filter_id) == 32001:
            kwargs["compression"] = int(filter_id)
            kwargs["compression_opts"] = tuple(cd_values)
            break

    return kwargs


# ==================== Track Rectification ====================

def rectify_tracks_npy(
    tracks_npy_path: Path,
    rectify_map_h5_path: Path,
    cam_to_cam_path: Path,
    output_npy_path: Path
) -> Tuple[int, int]:
    """Rectify tracks from distorted event space to rectified RGB space.
    
    Tracks are stored as structured numpy arrays with fields:
    - t: timestamp (int64)
    - x, y, w, h: bounding box in distorted event coordinates (float32)
    - class_id: object class (int)
    
    Args:
        tracks_npy_path: Path to input tracks.npy
        rectify_map_h5_path: Path to rectify_map.h5
        cam_to_cam_path: Path to cam_to_cam.yaml
        output_npy_path: Path to save rectified_tracks.npy
        
    Returns:
        Tuple of (total_tracks, valid_tracks)
    """
    logger.info(f"  Loading tracks from {tracks_npy_path.name}")
    
    # Load calibration and rectification map
    calib = yaml_file_to_dict(cam_to_cam_path)
    rectification_map = h5_file_to_dict(rectify_map_h5_path)
    remap = compute_remapping(calib, rectification_map)
    
    H_ev, W_ev = remap.shape[:2]
    
    # Load tracks
    tracks = np.load(tracks_npy_path, mmap_mode=None, allow_pickle=False)
    total_tracks = len(tracks)
    logger.info(f"  Total tracks: {total_tracks:,}")
    
    if total_tracks == 0:
        logger.warning(f"  No tracks to rectify, creating empty output")
        np.save(output_npy_path, tracks)
        return 0, 0
    
    # Extract bounding boxes in [x1, y1, x2, y2] format
    x = tracks["x"].astype(np.float32, copy=False)
    y = tracks["y"].astype(np.float32, copy=False)
    w = tracks["w"].astype(np.float32, copy=False)
    h = tracks["h"].astype(np.float32, copy=False)

    # Pre-filter invalid source tracks. These are guaranteed to cause issues later.
    finite_src = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & np.isfinite(h)
    pos_src = (w > 0) & (h > 0)
    keep_src = finite_src & pos_src
    dropped_src_nonfinite = int((~finite_src).sum())
    dropped_src_nonpos = int((finite_src & ~pos_src).sum())

    if dropped_src_nonfinite or dropped_src_nonpos:
        logger.info(
            "  Dropping invalid source tracks: "
            f"nonfinite={dropped_src_nonfinite:,}, nonpos_wh={dropped_src_nonpos:,}"
        )

    if not keep_src.any():
        logger.warning("  All source tracks invalid after filtering; writing empty rectified output")
        np.save(output_npy_path, tracks[:0])
        return total_tracks, 0

    tracks = tracks[keep_src]
    x = x[keep_src]
    y = y[keep_src]
    w = w[keep_src]
    h = h[keep_src]
    
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    
    # Warp boxes using rectification map
    logger.info(f"  Rectifying track bounding boxes...")
    xyxy_ev = np.stack([x1, y1, x2, y2], axis=1)
    xyxy_rect, valid_warp = _warp_xyxy_boxes_via_remap(xyxy_ev, remap, (RGB_H, RGB_W))

    # Post-filter: ensure rectified boxes are finite and strictly positive area.
    finite_dst = np.isfinite(xyxy_rect).all(axis=1)
    pos_dst = (xyxy_rect[:, 2] > xyxy_rect[:, 0]) & (xyxy_rect[:, 3] > xyxy_rect[:, 1])
    keep_dst = valid_warp & finite_dst & pos_dst

    dropped_warp_invalid = int((~valid_warp).sum())
    dropped_dst_nonfinite = int((valid_warp & ~finite_dst).sum())
    dropped_dst_nonpos = int((valid_warp & finite_dst & ~pos_dst).sum())

    if dropped_warp_invalid or dropped_dst_nonfinite or dropped_dst_nonpos:
        logger.info(
            "  Dropping invalid rectified tracks: "
            f"warp_invalid={dropped_warp_invalid:,}, nonfinite={dropped_dst_nonfinite:,}, nonpos_wh={dropped_dst_nonpos:,}"
        )

    valid_tracks = int(keep_dst.sum())
    logger.info(
        f"  Valid rectified tracks: {valid_tracks:,} "
        f"({100.0 * valid_tracks / total_tracks:.1f}% of original)"
    )
    
    # Create new tracks array with rectified coordinates
    tracks_rect = tracks.copy()
    
    # Convert back to x, y, w, h format
    tracks_rect['x'] = xyxy_rect[:, 0]
    tracks_rect['y'] = xyxy_rect[:, 1]
    tracks_rect['w'] = xyxy_rect[:, 2] - xyxy_rect[:, 0]
    tracks_rect['h'] = xyxy_rect[:, 3] - xyxy_rect[:, 1]
    
    # Filter to only valid tracks
    tracks_rect = tracks_rect[keep_dst]
    
    # Save rectified tracks
    logger.info(f"  Writing to {output_npy_path.name}")
    np.save(output_npy_path, tracks_rect)
    logger.info(f"  ✓ Saved rectified tracks to {output_npy_path}")
    
    return total_tracks, valid_tracks


def _warp_xyxy_boxes_via_remap(
    xyxy_ev: np.ndarray,
    remap_ev_to_rect: np.ndarray,
    rgb_hw: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Warp axis-aligned boxes from event/distorted pixel coords to rectified RGB.

    Uses the `rectify_map`-based lookup remap: (x_ev, y_ev) -> (x_rect, y_rect)
    on the four corners, then returns the enclosing axis-aligned rectangle.
    
    Conservative filtering: Only marks boxes as invalid if we cannot compute valid
    rectified coordinates (i.e., all corners map to non-finite values). Boxes that
    extend outside bounds are clamped rather than filtered.

    Args:
        xyxy_ev: (N, 4) boxes in event pixel coords [x1,y1,x2,y2].
        remap_ev_to_rect: (H_ev, W_ev, 2) float map giving rectified coords.
        rgb_hw: (H_rgb, W_rgb)

    Returns:
        warped_xyxy: (N, 4) float boxes in rectified coords [x1,y1,x2,y2]
        valid: (N,) bool mask for boxes successfully warped.
    """
    if xyxy_ev.size == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=bool)

    H_ev, W_ev = remap_ev_to_rect.shape[:2]
    H_rgb, W_rgb = rgb_hw

    xyxy_ev = np.asarray(xyxy_ev, dtype=np.float32)
    x1, y1, x2, y2 = [xyxy_ev[:, i] for i in range(4)]

    # 4 corners: TL, TR, BL, BR
    corners = np.stack(
        [
            np.stack([x1, y1], axis=1),
            np.stack([x2, y1], axis=1),
            np.stack([x1, y2], axis=1),
            np.stack([x2, y2], axis=1),
        ],
        axis=1,
    )  # (N, 4, 2)

    corners_int = np.rint(corners).astype(np.int64)
    cx = corners_int[:, :, 0]
    cy = corners_int[:, :, 1]

    # Clamp coordinates for safe lookup (handles boxes partially outside event camera FOV)
    cx_clamped = np.clip(cx, 0, W_ev - 1)
    cy_clamped = np.clip(cy, 0, H_ev - 1)

    # Lookup rectified coordinates
    warped = remap_ev_to_rect[cy_clamped, cx_clamped]  # (N, 4, 2)

    # Corner is valid only if both x/y are finite
    corner_valid = np.isfinite(warped).all(axis=2)  # (N, 4)
    has_any_valid_corner = corner_valid.any(axis=1)  # (N,)

    warped_x = warped[:, :, 0].astype(np.float32, copy=False)  # (N, 4)
    warped_y = warped[:, :, 1].astype(np.float32, copy=False)  # (N, 4)

    # Fill invalid corners with per-box mean of valid corners (vectorized; avoids Python loops).
    warped_x_nan = np.where(corner_valid, warped_x, np.nan)
    warped_y_nan = np.where(corner_valid, warped_y, np.nan)

    # Compute per-box means only for boxes that have at least one valid corner to
    # avoid RuntimeWarnings from np.nanmean on all-NaN slices.
    mean_x = np.zeros(warped_x_nan.shape[0], dtype=np.float32)
    mean_y = np.zeros(warped_y_nan.shape[0], dtype=np.float32)
    valid_boxes = has_any_valid_corner
    if np.any(valid_boxes):
        mean_x_valid = np.nanmean(warped_x_nan[valid_boxes], axis=1)
        mean_y_valid = np.nanmean(warped_y_nan[valid_boxes], axis=1)
        mean_x_valid = np.where(np.isfinite(mean_x_valid), mean_x_valid, 0.0)
        mean_y_valid = np.where(np.isfinite(mean_y_valid), mean_y_valid, 0.0)
        mean_x[valid_boxes] = mean_x_valid.astype(np.float32, copy=False)
        mean_y[valid_boxes] = mean_y_valid.astype(np.float32, copy=False)

    warped_x_filled = np.where(np.isfinite(warped_x_nan), warped_x_nan, mean_x[:, None])
    warped_y_filled = np.where(np.isfinite(warped_y_nan), warped_y_nan, mean_y[:, None])

    wx1 = np.where(has_any_valid_corner, warped_x_filled.min(axis=1), np.nan)
    wy1 = np.where(has_any_valid_corner, warped_y_filled.min(axis=1), np.nan)
    wx2 = np.where(has_any_valid_corner, warped_x_filled.max(axis=1), np.nan)
    wy2 = np.where(has_any_valid_corner, warped_y_filled.max(axis=1), np.nan)

    warped_xyxy = np.stack([wx1, wy1, wx2, wy2], axis=1).astype(np.float32)

    # Clip to RGB bounds (keeps all boxes, just ensures they're within frame)
    warped_xyxy[:, 0] = np.clip(warped_xyxy[:, 0], 0, W_rgb - 1)
    warped_xyxy[:, 2] = np.clip(warped_xyxy[:, 2], 0, W_rgb - 1)
    warped_xyxy[:, 1] = np.clip(warped_xyxy[:, 1], 0, H_rgb - 1)
    warped_xyxy[:, 3] = np.clip(warped_xyxy[:, 3], 0, H_rgb - 1)

    # Only mark as invalid if we couldn't compute any valid rectified coordinates
    valid = has_any_valid_corner

    return warped_xyxy, valid


# ==================== Output Validation ====================

def _validate_rectified_events_h5(path: Path) -> bool:
    """Check that a rectified_events.h5 file is complete and readable.

    Verifies the file can be opened and contains the expected datasets
    with non-zero event data.  Returns False for corrupt / partial files.
    """
    try:
        with h5py.File(path, 'r') as f:
            if 'events' not in f:
                return False
            ev = f['events']
            for key in ('x', 'y', 'p', 't'):
                if key not in ev:
                    return False
                if ev[key].shape[0] == 0:
                    return False
            if 't_offset' not in f or 'ms_to_idx' not in f:
                return False
        return True
    except Exception:
        return False


def _event_output_jobs(event_output_resolution: str) -> List[Tuple[str, str]]:
    mode = str(event_output_resolution).lower()
    aliases = {
        "rgb": "full",
        "full-res": "full",
        "native": "orig",
        "orig-res": "orig",
        "original": "orig",
        "event": "orig",
    }
    mode = aliases.get(mode, mode)
    if mode == "full":
        return [("full", RECTIFIED_EVENTS_FULL_NAME)]
    if mode == "orig":
        return [("orig", RECTIFIED_EVENTS_ORIG_RES_NAME)]
    if mode == "both":
        return [
            ("full", RECTIFIED_EVENTS_FULL_NAME),
            ("orig", RECTIFIED_EVENTS_ORIG_RES_NAME),
        ]
    raise ValueError(
        f"event_output_resolution must be one of full/orig/both, got {event_output_resolution!r}"
    )


def _validate_rectified_tracks_npy(path: Path) -> bool:
    """Check that a rectified_tracks.npy file is complete and readable."""
    try:
        arr = np.load(path, allow_pickle=False)
        # Empty is technically valid (some sequences have 0 tracks)
        return arr is not None
    except Exception:
        return False


def validate_all_outputs(data_root: Path) -> bool:
    """Scan all sequences in train/test and validate rectified outputs.

    Returns True if every sequence has valid outputs, False otherwise.
    """
    logger.info(f"\n{'='*80}")
    logger.info("Validating all rectified outputs")
    logger.info(f"{'='*80}")
    logger.info(f"Data root: {data_root}")

    all_ok = True

    for split in ("train", "test"):
        split_dir = data_root / split
        if not split_dir.exists():
            continue

        seqs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        logger.info(f"\n--- {split} ({len(seqs)} sequences) ---")

        for seq_path in seqs:
            seq_name = seq_path.name
            issues: List[str] = []

            # Check rectified events
            rect_events = seq_path / "events" / "left" / RECTIFIED_EVENTS_FULL_NAME
            src_events = seq_path / "events" / "left" / "events.h5"
            if src_events.exists():
                if not rect_events.exists():
                    issues.append("rectified_events.h5 MISSING")
                elif not _validate_rectified_events_h5(rect_events):
                    issues.append("rectified_events.h5 CORRUPT")

            # Check rectified tracks
            rect_tracks = seq_path / "object_detections" / "left" / "rectified_tracks.npy"
            src_tracks = seq_path / "object_detections" / "left" / "tracks.npy"
            if src_tracks.exists():
                if not rect_tracks.exists():
                    issues.append("rectified_tracks.npy MISSING")
                elif not _validate_rectified_tracks_npy(rect_tracks):
                    issues.append("rectified_tracks.npy CORRUPT")

            if issues:
                all_ok = False
                logger.warning(f"  ✗ {seq_name}: {', '.join(issues)}")
            else:
                logger.info(f"  ✓ {seq_name}")

    logger.info(f"\n{'='*80}")
    if all_ok:
        logger.info("All outputs are valid!")
    else:
        logger.error("Some outputs are missing or corrupt (see above)")
    logger.info(f"{'='*80}")

    return all_ok


# ==================== Main Preprocessing Engine ====================

def process_sequence(
    sequence_name: str,
    seq_path: Path,
    skip_events: bool = False,
    skip_tracks: bool = False,
    overwrite: bool = False,
    event_output_resolution: str = "full",
) -> Dict[str, any]:
    """Process a single sequence, rectifying events and tracks.
    
    Args:
        sequence_name: Name of the sequence (e.g., 'interlaken_00_a')
        seq_path: Path to sequence directory
        skip_events: Skip event rectification
        skip_tracks: Skip track rectification
        event_output_resolution: Which event output to write: full, orig, or both
        
    Returns:
        Dictionary with processing statistics
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Processing sequence: {sequence_name}")
    logger.info(f"Path: {seq_path}")
    
    stats = {
        "sequence": sequence_name,
        "success": False,
        "events_processed": 0,
        "events_valid": 0,
        "tracks_processed": 0,
        "tracks_valid": 0,
        "errors": []
    }
    
    # Check paths
    cam_to_cam = seq_path / "calibration" / "cam_to_cam.yaml"
    events_h5 = seq_path / "events" / "left" / "events.h5"
    rectify_map_h5 = seq_path / "events" / "left" / "rectify_map.h5"
    tracks_npy = seq_path / "object_detections" / "left" / "tracks.npy"
    
    if not cam_to_cam.exists():
        error = f"Calibration file not found: {cam_to_cam}"
        logger.error(f"  ✗ {error}")
        stats["errors"].append(error)
        return stats
    
    if not rectify_map_h5.exists():
        error = f"Rectification map not found: {rectify_map_h5}"
        logger.error(f"  ✗ {error}")
        stats["errors"].append(error)
        return stats
    
    # Process events
    if not skip_events:
        if not events_h5.exists():
            error = f"Events file not found: {events_h5}"
            logger.warning(f"  ⚠ {error}")
            stats["errors"].append(error)
        else:
            for output_mode, output_name in _event_output_jobs(event_output_resolution):
                output_events_h5 = seq_path / "events" / "left" / output_name

                should_process_events = True
                if output_events_h5.exists():
                    if overwrite:
                        logger.info(f"  Overwriting existing {output_events_h5.name}")
                        try:
                            output_events_h5.unlink()
                        except Exception as e:
                            logger.warning(f"  ⚠ Failed to delete {output_events_h5}: {e}")
                            should_process_events = False
                    else:
                        if _validate_rectified_events_h5(output_events_h5):
                            logger.info(
                                f"  ⚠ {output_events_h5.name} already exists and is valid, skipping event rectification"
                            )
                            should_process_events = False
                        else:
                            logger.warning(
                                f"  ⚠ {output_events_h5.name} exists but is corrupt/incomplete, deleting and re-processing"
                            )
                            try:
                                output_events_h5.unlink()
                            except Exception as e:
                                logger.warning(f"  ⚠ Failed to delete {output_events_h5}: {e}")
                                should_process_events = False

                if should_process_events:
                    try:
                        logger.info(f"Rectifying events ({output_mode} -> {output_events_h5.name})...")
                        total, valid = rectify_events_h5(
                            events_h5,
                            rectify_map_h5,
                            cam_to_cam,
                            output_events_h5,
                            output_resolution=output_mode,
                        )
                        stats["events_processed"] = max(int(stats["events_processed"]), int(total))
                        stats["events_valid"] = max(int(stats["events_valid"]), int(valid))
                    except Exception as e:
                        error = f"Event rectification failed for {output_events_h5.name}: {e}"
                        logger.error(f"  ✗ {error}")
                        stats["errors"].append(error)
    
    # Process tracks
    if not skip_tracks:
        if not tracks_npy.exists():
            error = f"Tracks file not found: {tracks_npy}"
            logger.warning(f"  ⚠ {error}")
            stats["errors"].append(error)
        else:
            output_tracks_npy = seq_path / "object_detections" / "left" / "rectified_tracks.npy"

            should_process_tracks = True
            if output_tracks_npy.exists():
                if overwrite:
                    logger.info(f"  Overwriting existing {output_tracks_npy.name}")
                    try:
                        output_tracks_npy.unlink()
                    except Exception as e:
                        logger.warning(f"  ⚠ Failed to delete {output_tracks_npy}: {e}")
                        should_process_tracks = False
                else:
                    if _validate_rectified_tracks_npy(output_tracks_npy):
                        logger.info(
                            f"  ⚠ {output_tracks_npy.name} already exists and is valid, skipping track rectification"
                        )
                        should_process_tracks = False
                    else:
                        logger.warning(
                            f"  ⚠ {output_tracks_npy.name} exists but is corrupt/incomplete, deleting and re-processing"
                        )
                        try:
                            output_tracks_npy.unlink()
                        except Exception as e:
                            logger.warning(f"  ⚠ Failed to delete {output_tracks_npy}: {e}")
                            should_process_tracks = False

            if should_process_tracks:
                try:
                    logger.info(f"Rectifying tracks...")
                    total, valid = rectify_tracks_npy(
                        tracks_npy,
                        rectify_map_h5,
                        cam_to_cam,
                        output_tracks_npy
                    )
                    stats["tracks_processed"] = total
                    stats["tracks_valid"] = valid
                except Exception as e:
                    error = f"Track rectification failed: {e}"
                    logger.error(f"  ✗ {error}")
                    stats["errors"].append(error)
    
    # Mark as successful if no critical errors
    stats["success"] = len(stats["errors"]) == 0 or all("not found" in e for e in stats["errors"])
    
    if stats["success"]:
        logger.info(f"✓ Sequence {sequence_name} processed successfully")
    else:
        logger.error(f"✗ Sequence {sequence_name} had errors")
    
    return stats


def _process_sequence_worker(
    seq_name: str,
    seq_path: Path,
    skip_events: bool,
    skip_tracks: bool,
    overwrite: bool,
    event_output_resolution: str,
) -> Dict[str, any]:
    """Top-level worker for ProcessPoolExecutor (must be picklable).

    Each worker independently processes one sequence and returns its stats dict.
    """
    try:
        return process_sequence(
            seq_name,
            seq_path,
            skip_events=skip_events,
            skip_tracks=skip_tracks,
            overwrite=overwrite,
            event_output_resolution=event_output_resolution,
        )
    except Exception as e:
        import traceback
        logger.error(f"Unexpected error processing {seq_name}: {e}")
        traceback.print_exc()
        return {
            "sequence": seq_name,
            "success": False,
            "events_processed": 0,
            "events_valid": 0,
            "tracks_processed": 0,
            "tracks_valid": 0,
            "errors": [str(e)],
        }


def run_preprocessing(
    data_root: Path,
    split: str,
    progress_file: Path,
    sequences: Optional[List[str]] = None,
    skip_events: bool = False,
    skip_tracks: bool = False,
    overwrite: bool = False,
    dry_run: bool = False,
    max_workers: int = 1,
    event_output_resolution: str = "full",
) -> None:
    """Run preprocessing on all pending sequences.

    Args:
        data_root: Root directory containing train/test subdirectories
        split: 'train' or 'test'
        progress_file: Path to progress.txt YAML file
        sequences: Optional list of specific sequences to process (overrides progress tracking)
        skip_events: Skip event rectification
        skip_tracks: Skip track rectification
        overwrite: Overwrite existing rectified outputs (events/tracks)
        dry_run: If True, only print what would be done
        max_workers: Number of parallel workers (1 = serial, >1 = parallel via ProcessPoolExecutor)
        event_output_resolution: Which event output to write: full, orig, or both
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"DSEC Preprocessing Pipeline")
    logger.info(f"{'='*80}")
    logger.info(f"Data root: {data_root}")
    logger.info(f"Split: {split}")
    logger.info(f"Progress file: {progress_file}")
    logger.info(f"Skip events: {skip_events}")
    logger.info(f"Skip tracks: {skip_tracks}")
    logger.info(f"Overwrite outputs: {overwrite}")
    logger.info(f"Dry run: {dry_run}")
    logger.info(f"Workers: {max_workers}")
    logger.info(f"Event output resolution: {event_output_resolution}")

    # Load progress
    progress = load_progress(progress_file)

    # Determine sequences to process
    if sequences is not None:
        seqs_to_process = sequences
        logger.info(f"\nProcessing specified sequences: {len(seqs_to_process)}")
    else:
        seqs_to_process = get_pending_sequences(progress, split)
        logger.info(f"\nPending sequences: {len(seqs_to_process)}")
        logger.info(f"Already finished: {len(progress['finished'][f'{split}_sequences'])}")

    if len(seqs_to_process) == 0:
        logger.info("No sequences to process!")
        return

    logger.info(f"\nSequences to process:")
    for seq in seqs_to_process:
        logger.info(f"  - {seq}")

    if dry_run:
        logger.info("\n[DRY RUN] Would process the above sequences.")
        return

    # Validate sequence paths before spawning workers
    valid_jobs: List[Tuple[str, Path]] = []
    for seq_name in seqs_to_process:
        seq_path = data_root / split / seq_name
        if not seq_path.exists():
            logger.error(f"Sequence directory not found: {seq_path}")
            continue
        valid_jobs.append((seq_name, seq_path))

    if len(valid_jobs) == 0:
        logger.error("No valid sequence directories found!")
        return

    # Process sequences (serial or parallel)
    all_stats: List[Dict] = []
    successful = 0
    failed = len(seqs_to_process) - len(valid_jobs)  # count missing dirs

    if max_workers <= 1:
        # Serial execution (original behaviour)
        for i, (seq_name, seq_path) in enumerate(valid_jobs, 1):
            logger.info(f"\n[{i}/{len(valid_jobs)}] Processing {seq_name}")
            stats = _process_sequence_worker(
                seq_name,
                seq_path,
                skip_events,
                skip_tracks,
                overwrite,
                event_output_resolution,
            )
            all_stats.append(stats)

            if stats["success"]:
                successful += 1
                mark_sequence_finished(progress, split, seq_name)
                save_progress(progress_file, progress)
            else:
                failed += 1
    else:
        # Parallel execution via ProcessPoolExecutor
        logger.info(f"\nLaunching {max_workers} parallel workers for {len(valid_jobs)} sequences...")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_seq = {
                executor.submit(
                    _process_sequence_worker,
                    seq_name,
                    seq_path,
                    skip_events,
                    skip_tracks,
                    overwrite,
                    event_output_resolution,
                ): seq_name
                for seq_name, seq_path in valid_jobs
            }

            for future in as_completed(future_to_seq):
                seq_name = future_to_seq[future]
                try:
                    stats = future.result()
                except Exception as e:
                    logger.error(f"Worker for {seq_name} raised: {e}")
                    stats = {
                        "sequence": seq_name,
                        "success": False,
                        "events_processed": 0,
                        "events_valid": 0,
                        "tracks_processed": 0,
                        "tracks_valid": 0,
                        "errors": [str(e)],
                    }

                all_stats.append(stats)
                completed = successful + failed + 1 - (len(seqs_to_process) - len(valid_jobs))

                if stats["success"]:
                    successful += 1
                    mark_sequence_finished(progress, split, seq_name)
                    save_progress(progress_file, progress)
                    logger.info(
                        f"[{completed}/{len(valid_jobs)}] {seq_name} finished successfully"
                    )
                else:
                    failed += 1
                    logger.error(
                        f"[{completed}/{len(valid_jobs)}] {seq_name} failed: "
                        f"{stats.get('errors', [])}"
                    )

    # Summary
    logger.info(f"\n{'='*80}")
    logger.info(f"Preprocessing Summary")
    logger.info(f"{'='*80}")
    logger.info(f"Total sequences: {len(seqs_to_process)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")

    total_events = sum(s["events_processed"] for s in all_stats)
    total_events_valid = sum(s["events_valid"] for s in all_stats)
    total_tracks = sum(s["tracks_processed"] for s in all_stats)
    total_tracks_valid = sum(s["tracks_valid"] for s in all_stats)

    if total_events > 0:
        logger.info(f"\nEvents: {total_events_valid:,} / {total_events:,} ({100.0 * total_events_valid / total_events:.1f}%) valid")
    if total_tracks > 0:
        logger.info(f"Tracks: {total_tracks_valid:,} / {total_tracks:,} ({100.0 * total_tracks_valid / total_tracks:.1f}%) valid")

    logger.info(f"\n✓ Preprocessing complete!")


def main():
    """Main entry point for preprocessing script."""
    parser = argparse.ArgumentParser(
        description="Rectify DSEC events and tracks to full resolution RGB space"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("./data"),
        help="Root directory containing train/test subdirectories (default: ./data)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Which split to process (default: train)"
    )
    parser.add_argument(
        "--progress-file",
        type=Path,
        default=_SCRIPT_DIR / "progress.txt",
        help="Path to progress.txt YAML file (default: ./progress.txt)"
    )
    parser.add_argument(
        "--sequences",
        type=str,
        nargs="+",
        help="Specific sequences to process (overrides progress tracking)"
    )
    parser.add_argument(
        "--skip-events",
        action="store_true",
        help="Skip event rectification"
    )
    parser.add_argument(
        "--skip-tracks",
        action="store_true",
        help="Skip track rectification"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without actually processing"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing rectified outputs (events/tracks)"
    )
    parser.add_argument(
        "--event-output-resolution",
        type=str,
        default="full",
        choices=["full", "orig", "both", "rgb", "native", "orig-res", "event"],
        help=(
            "Event rectification output to generate. "
            "'full' writes rectified_events.h5 in 1440x1080 RGB coordinates; "
            "'orig' writes rectified_events_orig_res.h5 in 640x480 EDUM-native coordinates; "
            "'both' writes both. Default: full."
        ),
    )
    parser.add_argument(
        "--init-progress",
        action="store_true",
        help="Initialize progress file from data directory"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset progress file and delete all rectified outputs under --data-root"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers for sequence processing (default: 1 = serial). "
             "Recommended: 8-16 on HPC nodes with many cores."
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate all existing rectified outputs and report missing/corrupt files (no processing)"
    )

    args = parser.parse_args()

    # Validate all outputs if requested
    if args.validate:
        ok = validate_all_outputs(args.data_root)
        sys.exit(0 if ok else 1)

    # Reset progress + delete generated outputs if requested
    if args.reset:
        logger.info("Reset requested: deleting rectified outputs and resetting progress file...")
        deleted, _missing = _delete_rectified_outputs(args.data_root, dry_run=args.dry_run)
        progress = _initialize_progress_dict(args.data_root)
        if args.dry_run:
            logger.info(
                f"[DRY RUN] Would delete {deleted} files and reset progress at: {args.progress_file}"
            )
        else:
            save_progress(args.progress_file, progress)
            logger.info(f"Deleted {deleted} files and reset progress at: {args.progress_file}")
        return
    
    # Initialize progress file if requested
    if args.init_progress:
        logger.info("Initializing progress file...")

        progress = _initialize_progress_dict(args.data_root)
        save_progress(args.progress_file, progress)
        logger.info(
            "Initialized progress file with "
            f"{len(progress['jobs']['train_sequences'])} train and "
            f"{len(progress['jobs']['test_sequences'])} test sequences"
        )
        return
    
    # Run preprocessing
    run_preprocessing(
        data_root=args.data_root,
        split=args.split,
        progress_file=args.progress_file,
        sequences=args.sequences,
        skip_events=args.skip_events,
        skip_tracks=args.skip_tracks,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        max_workers=args.workers,
        event_output_resolution=args.event_output_resolution,
    )


if __name__ == "__main__":
    main()
