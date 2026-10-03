"""Regression tests for issue #107: validation silently disabled by a stale
``sequence_manifest=`` kwarg passed to ``build_eval_loader_cfg``.

The manifest feature was intentionally removed (commit e72bf1b) in favour of the
YAML mechanism (``dsec.yaml`` ``val_sequences`` + ``val_split_dir`` selected via
``eval.split: "val"``), but two callers kept passing ``sequence_manifest=...``.
The resulting ``TypeError`` was swallowed by a bare ``except Exception`` at both
callsites, so every detector/MCM run trained with no validation loader.

These tests are CPU-only and need no dataset, so they run on macOS.
"""

import ast
import inspect
import sys
import unittest
from pathlib import Path
from unittest import mock

from smcfnet.model.build import build_dataset_loader_cfg, build_eval_loader_cfg

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CALLSITE_FILES = (
    PROJECT_ROOT / "src/smcfnet/core/train.py",
    PROJECT_ROOT / "src/smcfnet/core/mcm_trainer.py",
)


class TestEvalLoaderSignatureContract(unittest.TestCase):
    """Test A: the callee no longer accepts sequence_manifest."""

    def test_build_eval_loader_cfg_param_set(self):
        params = set(inspect.signature(build_eval_loader_cfg).parameters)
        self.assertEqual(
            params,
            {"config", "split", "sequences", "dataset_config_override", "return_components"},
        )
        self.assertNotIn("sequence_manifest", params)

    def test_build_dataset_loader_cfg_param_set(self):
        params = set(inspect.signature(build_dataset_loader_cfg).parameters)
        self.assertEqual(
            params,
            {
                "config",
                "split",
                "is_train",
                "sequences",
                "dataset_config_override",
                "return_components",
            },
        )
        self.assertNotIn("sequence_manifest", params)

    def test_passing_sequence_manifest_raises_typeerror(self):
        # The unexpected-kwarg TypeError is raised at call binding, before the
        # body runs, so a dummy config never reaches any dataset access.
        with self.assertRaises(TypeError):
            build_eval_loader_cfg(object(), sequence_manifest=None)


class TestCallsitesDoNotPassManifest(unittest.TestCase):
    """Test B: no caller re-introduces the kwarg (contract test only covers the
    callee; this guards the callers)."""

    def test_no_sequence_manifest_reference_in_callsites(self):
        for path in CALLSITE_FILES:
            self.assertTrue(path.exists(), f"callsite file missing: {path}")
            text = path.read_text()
            self.assertNotIn(
                "sequence_manifest",
                text,
                msg=(
                    f"{path.name} still references sequence_manifest; the kwarg was "
                    "removed in e72bf1b and must not be reintroduced (issue #107)."
                ),
            )

    def test_build_eval_loader_calls_have_no_sequence_manifest_kwarg(self):
        # Stronger, AST-level check: any build_eval_loader_cfg(...) call must not
        # pass a sequence_manifest keyword.
        for path in CALLSITE_FILES:
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
                if name != "build_eval_loader_cfg":
                    continue
                kwargs = {kw.arg for kw in node.keywords if kw.arg is not None}
                self.assertNotIn(
                    "sequence_manifest",
                    kwargs,
                    msg=f"build_eval_loader_cfg call in {path.name} passes sequence_manifest",
                )


class TestEvalLoaderFailureSurfaces(unittest.TestCase):
    """Test C: a signature/config error during eval-loader build is re-raised,
    not silently swallowed into val_loader=None (the root cause of #107)."""

    @unittest.skipIf(
        sys.platform == "darwin",
        "importing smcfnet.core.train aborts on macOS (pyarrow/keras conflict); runs on cluster",
    )
    def test_typeerror_is_reraised_not_swallowed(self):
        from smcfnet.config.config import Config
        from smcfnet.core import train as train_mod

        class _StubTrainer:
            def __init__(self, cfg):
                self.config = cfg
                self._eval_only = False
                self.val_loader = None
                self.eval_loader = None
                self._val_flow_sequence_names = None

        trainer = _StubTrainer(Config())

        with mock.patch.object(
            train_mod, "build_eval_loader_cfg", side_effect=TypeError("unexpected kwarg")
        ), mock.patch.object(train_mod, "is_main_process", return_value=True):
            with self.assertRaises(TypeError):
                train_mod._attach_eval_loader_if_available(trainer)


if __name__ == "__main__":
    unittest.main()
