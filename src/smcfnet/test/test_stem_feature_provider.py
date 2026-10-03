import h5py
import numpy as np
import torch
from smcfnet.dataset.stem_feature_provider import StemFeatureProvider


def _make_cache(tmp_path, seq="seq_a", n=10, channels=4, shapes=((6, 8), (3, 4), (2, 2)), sha="abc"):
    path = tmp_path / f"{seq}.h5"
    with h5py.File(str(path), "w") as f:
        for i, (h, w) in enumerate(shapes):
            # frame index encoded in the data so window order is checkable
            data = np.stack([np.full((channels, h, w), t, dtype=np.float16) for t in range(n)])
            f.create_dataset(f"p{i}", data=data)
        f.attrs["checkpoint_sha"] = sha
        f.attrs["num_scales"] = len(shapes)
    return path


def test_window_returns_k_frames_per_scale(tmp_path):
    _make_cache(tmp_path)
    p = StemFeatureProvider(tmp_path, context_window=3, num_scales=3)
    window = p.get_window("seq_a", frame_index=5)  # list of [K, C, H, W]
    assert len(window) == 3
    assert window[0].shape == (3, 4, 6, 8)
    # frames 3,4,5 in order
    assert window[0][:, 0, 0, 0].tolist() == [3.0, 4.0, 5.0]


def test_window_clamp_pads_at_sequence_start(tmp_path):
    _make_cache(tmp_path)
    p = StemFeatureProvider(tmp_path, context_window=3, num_scales=3)
    window = p.get_window("seq_a", frame_index=0)  # clamp: repeat frame 0
    assert window[0].shape == (3, 4, 6, 8)
    assert window[0][:, 0, 0, 0].tolist() == [0.0, 0.0, 0.0]


def test_sha_mismatch_refused(tmp_path):
    _make_cache(tmp_path, seq="seq_a", sha="aaa")
    _make_cache(tmp_path, seq="seq_b", sha="bbb")
    p = StemFeatureProvider(tmp_path, context_window=2, num_scales=3)
    p.get_window("seq_a", 1)
    try:
        p.get_window("seq_b", 1)
        assert False, "expected sha mismatch RuntimeError"
    except RuntimeError as e:
        assert "sha" in str(e).lower()
