"""Verify that _targets_for_frame and _future_targets_for_frame produce objects
in the same order, so that SimOTA assignments on current-frame targets index
correctly into future_targets.

The concern: both methods use np.searchsorted on the same tracks array, so raw
ordering should match. BUT _targets_for_frame can filter degenerate boxes
(zero-size, non-finite) via a valid mask, dropping rows from the output while
_future_targets_for_frame iterates the *unfiltered* raw track array. If any box
gets filtered, the index correspondence breaks silently — object i in targets
would map to a different track_id than object i in future_targets.

This test empirically checks that pairing by verifying track_id order.
"""

from __future__ import annotations

import bisect
from pathlib import Path

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Locate project and DSEC config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DSEC_YAML = PROJECT_ROOT / "src" / "smcfnet" / "config" / "dsec" / "dsec.yaml"

# Skip early if DSEC data is unavailable
if not DSEC_YAML.exists():
    pytest.skip("DSEC dataset config not found", allow_module_level=True)

from smcfnet.dataset.dsec.dsec_sequence_index import (
    DSECSequenceIndexDataset,
    build_dsec_sequence_metas,
)
from smcfnet.dataset.dsec.dsec_lightweight_mapper import DSECLightweightMapper

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FUTURE_STEPS = 5
IMG_H, IMG_W = 480, 640
DT_MS = 50.0
MIN_OBJECTS = 3
NUM_FRAMES = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_track_ids_at(mapper: DSECLightweightMapper, seq_idx: int, t_us: int) -> np.ndarray:
    """Extract raw track_ids at a timestamp, same searchsorted as the mapper uses."""
    mapper._ensure_tracks_open(seq_idx)
    tracks = mapper._tracks.get(seq_idx)
    t_arr = mapper._tracks_t.get(seq_idx)
    if tracks is None or t_arr is None:
        return np.array([], dtype=np.int64)
    t_us = int(t_us)
    left = int(np.searchsorted(t_arr, t_us, side="left"))
    right = int(np.searchsorted(t_arr, t_us, side="right"))
    if right <= left:
        return np.array([], dtype=np.int64)
    return tracks[left:right]["track_id"].copy()


def _get_valid_mask_at(mapper: DSECLightweightMapper, seq_idx: int, t_us: int) -> np.ndarray:
    """Reproduce the degenerate-box filter from _targets_for_frame."""
    mapper._ensure_tracks_open(seq_idx)
    tracks = mapper._tracks.get(seq_idx)
    t_arr = mapper._tracks_t.get(seq_idx)
    if tracks is None or t_arr is None:
        return np.array([], dtype=bool)
    t_us = int(t_us)
    left = int(np.searchsorted(t_arr, t_us, side="left"))
    right = int(np.searchsorted(t_arr, t_us, side="right"))
    if right <= left:
        return np.array([], dtype=bool)
    tr = tracks[left:right]
    x = tr["x"].astype(np.float32)
    y = tr["y"].astype(np.float32)
    w = tr["w"].astype(np.float32)
    h = tr["h"].astype(np.float32)
    cx = np.clip(x + w / 2.0, 0, IMG_W - 1)
    cy = np.clip(y + h / 2.0, 0, IMG_H - 1)
    finite = np.isfinite(cx) & np.isfinite(cy) & np.isfinite(w) & np.isfinite(h)
    return finite & (w > 0) & (h > 0)


# ---------------------------------------------------------------------------
# Fixture: one labelled DSEC sequence with mapper
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mapper_and_seq():
    """Build a mapper for the first labelled training sequence."""
    try:
        metas = build_dsec_sequence_metas(dsec_yaml_path=DSEC_YAML, split="train")
    except (FileNotFoundError, ValueError) as exc:
        pytest.skip(f"Cannot build DSEC metas: {exc}")

    labelled = [m for m in metas if m.has_labels]
    if not labelled:
        pytest.skip("No labelled DSEC training sequences found")

    meta = labelled[0]
    seq_ds = DSECSequenceIndexDataset(meta, seq_idx=0, dt_ms=DT_MS)
    mapper = DSECLightweightMapper(
        [seq_ds],
        load_events=False,
        load_tracks=True,
        mmap_tracks=True,
        load_images=False,
        build_targets=True,
        build_future_targets=True,
        future_steps=FUTURE_STEPS,
        image_height=IMG_H,
        image_width=IMG_W,
    )
    return mapper, seq_ds


@pytest.fixture(scope="module")
def rich_frames(mapper_and_seq):
    """Find frames with >= MIN_OBJECTS current-frame objects, away from sequence end."""
    mapper, seq_ds = mapper_and_seq
    max_frame = len(seq_ds) - FUTURE_STEPS - 1
    if max_frame < 0:
        pytest.skip("Sequence too short for future_steps")

    found: list[int] = []
    # Scan from frame 0 upward to find qualifying frames deterministically
    for fi in range(0, max_frame + 1):
        if len(found) >= NUM_FRAMES:
            break
        t_ref = seq_ds[fi]["t_ref_us"]
        targets = mapper._targets_for_frame(0, t_ref, IMG_H, IMG_W)
        if targets.shape[0] >= MIN_OBJECTS:
            found.append(fi)

    if len(found) < 1:
        pytest.skip(f"No frames with >= {MIN_OBJECTS} objects found")

    return found


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTrackIdOrdering:
    """Verify that current-frame and future-frame targets use the same object ordering."""

    def test_raw_ordering_matches(self, mapper_and_seq, rich_frames):
        """Both methods use searchsorted on the same array, so raw track_id
        ordering at t_ref should be identical between the two code paths."""
        mapper, seq_ds = mapper_and_seq
        mismatches = []

        for fi in rich_frames:
            t_ref = seq_ds[fi]["t_ref_us"]

            # Raw track_ids as returned by searchsorted (the ground truth order)
            raw_ids = _get_track_ids_at(mapper, 0, t_ref)

            # _future_targets_for_frame re-does the same searchsorted and
            # iterates curr_tracks["track_id"] — extract that order
            # by calling it with num_current = len(raw_ids) and checking
            # which indices got validity=True
            targets = mapper._targets_for_frame(0, t_ref, IMG_H, IMG_W)
            num_current = targets.shape[0]
            future = mapper._future_targets_for_frame(0, t_ref, num_current, IMG_H, IMG_W)

            # The future method internally iterates raw_ids[0..num_current-1].
            # If _targets_for_frame filtered some boxes, num_current < len(raw_ids),
            # and the two arrays diverge.
            valid_mask = _get_valid_mask_at(mapper, 0, t_ref)
            filtered_ids = raw_ids[valid_mask]  # what _targets_for_frame kept
            future_iterated_ids = raw_ids[:num_current]  # what _future_targets iterates

            if not np.array_equal(filtered_ids, future_iterated_ids):
                mismatches.append({
                    "frame": fi,
                    "t_ref_us": t_ref,
                    "raw_ids": raw_ids.tolist(),
                    "targets_ids (filtered)": filtered_ids.tolist(),
                    "future_ids (first N raw)": future_iterated_ids.tolist(),
                    "valid_mask": valid_mask.tolist(),
                })

        if mismatches:
            lines = ["Track-ID ordering mismatch detected!\n"]
            for m in mismatches:
                lines.append(f"  Frame {m['frame']} (t_ref={m['t_ref_us']}us):")
                lines.append(f"    raw track_ids:            {m['raw_ids']}")
                lines.append(f"    _targets kept (filtered): {m['targets_ids (filtered)']}")
                lines.append(f"    _future iterates (raw):   {m['future_ids (first N raw)']}")
                lines.append(f"    valid_mask:               {m['valid_mask']}")
                lines.append("")
            msg = "\n".join(lines)
            # This is a real bug if it fires — fail hard
            pytest.fail(msg)

    def test_future_validity_matches_track_presence(self, mapper_and_seq, rich_frames):
        """For each valid future target, its track_id should actually exist
        at the future timestamp. For each invalid one, the track_id should
        be absent from the future frame."""
        mapper, seq_ds = mapper_and_seq
        inconsistencies = []

        for fi in rich_frames:
            t_ref = seq_ds[fi]["t_ref_us"]
            timestamps = seq_ds._timestamps_us
            curr_idx = bisect.bisect_left(timestamps, t_ref)
            fut_idx = curr_idx + FUTURE_STEPS
            if fut_idx >= len(timestamps):
                continue
            t_fut = timestamps[fut_idx]

            raw_curr_ids = _get_track_ids_at(mapper, 0, t_ref)
            raw_fut_ids = set(int(x) for x in _get_track_ids_at(mapper, 0, t_fut))

            targets = mapper._targets_for_frame(0, t_ref, IMG_H, IMG_W)
            num_current = targets.shape[0]
            future = mapper._future_targets_for_frame(0, t_ref, num_current, IMG_H, IMG_W)
            validity = future["validity"]

            # _future_targets_for_frame iterates raw_curr_ids[0..num_current-1]
            for i in range(min(num_current, len(raw_curr_ids))):
                tid = int(raw_curr_ids[i])
                is_valid = bool(validity[i].item())
                in_future = tid in raw_fut_ids

                if is_valid and not in_future:
                    inconsistencies.append(
                        f"Frame {fi}: track_id {tid} at index {i} marked valid "
                        f"but NOT in future frame (t_fut={t_fut}us)"
                    )
                elif not is_valid and in_future:
                    # This can happen legitimately if filtering caused index shift,
                    # but it's still a consistency concern worth flagging
                    inconsistencies.append(
                        f"Frame {fi}: track_id {tid} at index {i} marked INVALID "
                        f"but IS in future frame (t_fut={t_fut}us) — possible index misalignment"
                    )

        if inconsistencies:
            msg = "Future validity inconsistencies:\n  " + "\n  ".join(inconsistencies)
            pytest.fail(msg)

    def test_no_degenerate_box_filtering(self, mapper_and_seq, rich_frames):
        """Check whether any of the sampled frames have degenerate boxes that
        would be filtered by _targets_for_frame but not by _future_targets_for_frame.

        If this test fails, the index-alignment bug is *live* for real data."""
        mapper, seq_ds = mapper_and_seq
        frames_with_filtering = []

        for fi in rich_frames:
            t_ref = seq_ds[fi]["t_ref_us"]
            raw_ids = _get_track_ids_at(mapper, 0, t_ref)
            valid_mask = _get_valid_mask_at(mapper, 0, t_ref)

            num_raw = len(raw_ids)
            num_valid = int(valid_mask.sum())
            if num_valid < num_raw:
                dropped_ids = raw_ids[~valid_mask].tolist()
                frames_with_filtering.append(
                    f"Frame {fi}: {num_raw} raw -> {num_valid} valid "
                    f"(dropped track_ids: {dropped_ids})"
                )

        if frames_with_filtering:
            msg = (
                "Degenerate boxes found — _targets_for_frame filters rows that "
                "_future_targets_for_frame does NOT, causing index misalignment:\n  "
                + "\n  ".join(frames_with_filtering)
            )
            # Warn rather than fail: the filtering itself isn't wrong, but the
            # mismatch between the two methods IS a real alignment bug.
            pytest.fail(msg)

    def test_output_shapes_consistent(self, mapper_and_seq, rich_frames):
        """Basic shape sanity: future boxes and validity have correct dimensions."""
        mapper, seq_ds = mapper_and_seq

        for fi in rich_frames:
            t_ref = seq_ds[fi]["t_ref_us"]
            targets = mapper._targets_for_frame(0, t_ref, IMG_H, IMG_W)
            n = targets.shape[0]
            assert targets.shape == (n, 5), f"targets shape {targets.shape}"

            future = mapper._future_targets_for_frame(0, t_ref, n, IMG_H, IMG_W)
            assert future["boxes"].shape == (n, 4), (
                f"Frame {fi}: future boxes shape {future['boxes'].shape}, expected ({n}, 4)"
            )
            assert future["validity"].shape == (n,), (
                f"Frame {fi}: validity shape {future['validity'].shape}, expected ({n},)"
            )

    def test_future_boxes_physically_plausible(self, mapper_and_seq, rich_frames):
        """For matched objects, future boxes should be finite, positive-sized,
        and within image bounds (the mapper clamps cx/cy)."""
        mapper, seq_ds = mapper_and_seq

        for fi in rich_frames:
            t_ref = seq_ds[fi]["t_ref_us"]
            targets = mapper._targets_for_frame(0, t_ref, IMG_H, IMG_W)
            n = targets.shape[0]
            future = mapper._future_targets_for_frame(0, t_ref, n, IMG_H, IMG_W)
            valid = future["validity"]
            boxes = future["boxes"][valid]

            if boxes.shape[0] == 0:
                continue

            assert torch.isfinite(boxes).all(), f"Frame {fi}: non-finite future boxes"
            assert (boxes[:, 2] > 0).all(), f"Frame {fi}: zero/negative future width"
            assert (boxes[:, 3] > 0).all(), f"Frame {fi}: zero/negative future height"
            assert (boxes[:, 0] >= 0).all() and (boxes[:, 0] < IMG_W).all(), (
                f"Frame {fi}: future cx out of bounds"
            )
            assert (boxes[:, 1] >= 0).all() and (boxes[:, 1] < IMG_H).all(), (
                f"Frame {fi}: future cy out of bounds"
            )
