"""Regression tests for issue #108: PEOD val split must not silently overlap train.

PEOD ships no official validation split, so a val split is carved from the train
directory (``val_split_dir: train``). ``peod.yaml`` shipped with
``val_sequences: []``, which (being falsy) skipped the sequence filter and made
validation cover the entire train split. ``build_peod_sequence_metas`` now raises
when a val split reads the train directory without an explicit, disjoint
``val_sequences`` list.

These tests use synthetic configs with a non-existent root, so the leak guard
(which runs before any disk access) is exercised without the PEOD dataset.
"""

import tempfile
import unittest
from pathlib import Path

import yaml

from smcfnet.dataset.peod.peod_sequence_index import build_peod_sequence_metas

_NONEXISTENT_ROOT = "/nonexistent/peod_root_issue108"


def _base_dataset(**overrides):
    ds = {
        "name": "PEOD",
        "root": _NONEXISTENT_ROOT,
        "image_height": 720,
        "image_width": 1280,
        "classes": ["car"],
        "train_subsets": ["challenge"],
        "val_split_dir": "train",
        "val_subsets": ["challenge"],
        "test_subsets": ["challenge"],
        "train_sequences": "auto",
        "val_sequences": [],
        "test_sequences": "auto",
    }
    ds.update(overrides)
    return ds


class _PeodYaml:
    """Write a temp PEOD yaml and return its path."""

    def __init__(self, **dataset_overrides):
        self._tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        )
        yaml.safe_dump({"dataset": _base_dataset(**dataset_overrides)}, self._tmp)
        self._tmp.flush()
        self._tmp.close()
        self.path = self._tmp.name

    def cleanup(self):
        Path(self.path).unlink(missing_ok=True)


class TestPeodValLeakGuard(unittest.TestCase):
    def _build_val(self, **overrides):
        cfg = _PeodYaml(**overrides)
        self.addCleanup(cfg.cleanup)
        return build_peod_sequence_metas(peod_yaml_path=cfg.path, split="val")

    def test_empty_val_sequences_on_train_dir_raises_leak(self):
        with self.assertRaises(ValueError) as ctx:
            self._build_val(val_split_dir="train", val_sequences=[])
        self.assertIn("data leak", str(ctx.exception))

    def test_auto_val_sequences_on_train_dir_raises_leak(self):
        with self.assertRaises(ValueError) as ctx:
            self._build_val(val_split_dir="train", val_sequences="auto")
        self.assertIn("data leak", str(ctx.exception))

    def test_explicit_val_sequences_overlapping_train_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self._build_val(
                val_split_dir="train",
                train_sequences=["seq_a", "seq_b"],
                val_sequences=["seq_b"],
            )
        self.assertIn("overlap", str(ctx.exception))

    def test_disjoint_explicit_val_sequences_pass_the_guard(self):
        # Guard passes (disjoint), so it proceeds to disk and fails on the
        # missing root with FileNotFoundError -- proving it got past the guard.
        with self.assertRaises(FileNotFoundError):
            self._build_val(
                val_split_dir="train",
                train_sequences=["seq_a"],
                val_sequences=["seq_b"],
            )

    def test_val_from_test_dir_is_not_treated_as_train_leak(self):
        # val_split_dir != train dir -> the train-leak guard must not fire even
        # with 'auto'. It proceeds to disk and raises FileNotFoundError.
        with self.assertRaises(FileNotFoundError):
            self._build_val(val_split_dir="test", val_sequences="auto")


if __name__ == "__main__":
    unittest.main()
