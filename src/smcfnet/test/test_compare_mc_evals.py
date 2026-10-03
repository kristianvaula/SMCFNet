from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_compare_module():
    root = Path(__file__).resolve().parents[3]
    path = root / "experiments" / "p1" / "compare_mc_evals.py"
    spec = importlib.util.spec_from_file_location("compare_mc_evals", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_checkpoint_result_labels_include_run_identity():
    mod = _load_compare_module()
    result_dir = Path("results/ssnn_net_v2_hybrid/epoch_30")

    label, run_label, checkpoint_label = mod._result_labels(result_dir)

    assert label == "ssnn_net_v2_hybrid/epoch_30"
    assert run_label == "ssnn_net_v2_hybrid"
    assert checkpoint_label == "epoch_30"


def test_display_label_is_compact_within_run_and_unique_across_runs():
    mod = _load_compare_module()
    record = {
        "label": "ssnn_net_v2_hybrid/epoch_30",
        "run_label": "ssnn_net_v2_hybrid",
        "checkpoint_label": "epoch_30",
    }

    assert mod._display_label(record, multiple_runs=False) == "epoch_30"
    assert (
        mod._display_label(record, multiple_runs=True)
        == "ssnn_net_v2_hybrid/epoch_30"
    )
