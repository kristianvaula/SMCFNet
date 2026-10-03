"""Unit tests for smcfnet.dataset.flow_provider.

Handcrafts tiny HDF5 flow files and exercises the provider's lookup,
frame-0 handling, missing-file errors, handle caching, and cross-file
checkpoint-sha consistency.

Runs on macOS without CUDA.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

h5py = pytest.importorskip("h5py")

from smcfnet.dataset.flow_provider import PrecomputedFlowProvider


def _write_flow_file(
    path: Path,
    frame_count: int,
    sha: str,
    *,
    shape=(2, 8, 12),
    dtype=np.float16,
) -> np.ndarray:
    """Write a synthetic flow file and return the raw array for comparison."""
    rng = np.random.default_rng(seed=int(sha[:8], 16))
    flow = rng.normal(size=(frame_count, *shape)).astype(dtype)
    flow[0] = 0  # frame 0 convention: zeros (no prior pair)
    with h5py.File(str(path), "w") as f:
        f.create_dataset("flow", data=flow, dtype=dtype)
        f.attrs["checkpoint_sha"] = sha
        f.attrs["dsec_sequence"] = path.stem
        f.attrs["frame_count"] = frame_count
    return flow


def test_get_returns_expected_tensor(tmp_path):
    sha = "a" * 64
    flow = _write_flow_file(tmp_path / "seq_a.h5", frame_count=5, sha=sha)
    provider = PrecomputedFlowProvider(tmp_path)

    out = provider.get("seq_a", 3)

    assert out is not None
    assert out.shape == (2, 8, 12)
    assert out.dtype == torch.float32
    np.testing.assert_allclose(out.numpy(), flow[3].astype(np.float32), rtol=0, atol=0)


def test_frame_zero_returns_none(tmp_path):
    _write_flow_file(tmp_path / "seq_a.h5", frame_count=5, sha="b" * 64)
    provider = PrecomputedFlowProvider(tmp_path)

    assert provider.get("seq_a", 0) is None


def test_missing_sequence_raises(tmp_path):
    provider = PrecomputedFlowProvider(tmp_path)

    with pytest.raises(FileNotFoundError, match="seq_missing"):
        provider.get("seq_missing", 1)


def test_out_of_range_raises(tmp_path):
    _write_flow_file(tmp_path / "seq_a.h5", frame_count=5, sha="c" * 64)
    provider = PrecomputedFlowProvider(tmp_path)

    with pytest.raises(IndexError, match="frame_count=5"):
        provider.get("seq_a", 5)


def test_handle_is_cached_across_calls(tmp_path):
    _write_flow_file(tmp_path / "seq_a.h5", frame_count=5, sha="d" * 64)
    provider = PrecomputedFlowProvider(tmp_path)

    provider.get("seq_a", 1)
    handle_first = provider._handles["seq_a"]
    provider.get("seq_a", 2)
    handle_second = provider._handles["seq_a"]

    assert handle_first is handle_second  # same h5py.File instance


def test_checkpoint_sha_must_match_across_files(tmp_path):
    _write_flow_file(tmp_path / "seq_a.h5", frame_count=3, sha="a" * 64)
    _write_flow_file(tmp_path / "seq_b.h5", frame_count=3, sha="b" * 64)
    provider = PrecomputedFlowProvider(tmp_path)

    provider.get("seq_a", 1)  # establishes expected sha
    with pytest.raises(RuntimeError, match="Checkpoint sha mismatch"):
        provider.get("seq_b", 1)


def test_missing_sha_attr_raises(tmp_path):
    path = tmp_path / "seq_a.h5"
    with h5py.File(str(path), "w") as f:
        f.create_dataset("flow", data=np.zeros((3, 2, 4, 4), dtype=np.float16))
    provider = PrecomputedFlowProvider(tmp_path)

    with pytest.raises(RuntimeError, match="missing required 'checkpoint_sha'"):
        provider.get("seq_a", 1)


def test_frame_count_helper(tmp_path):
    _write_flow_file(tmp_path / "seq_a.h5", frame_count=7, sha="e" * 64)
    provider = PrecomputedFlowProvider(tmp_path)

    assert provider.frame_count("seq_a") == 7


def test_close_releases_handles(tmp_path):
    _write_flow_file(tmp_path / "seq_a.h5", frame_count=3, sha="f" * 64)
    provider = PrecomputedFlowProvider(tmp_path)
    provider.get("seq_a", 1)

    assert "seq_a" in provider._handles
    provider.close()
    assert provider._handles == {}
