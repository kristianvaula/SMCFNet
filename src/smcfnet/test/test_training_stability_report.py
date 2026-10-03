from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


def _load_report_module():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-smcfnet-tests")
    root = Path(__file__).resolve().parents[3]
    path = root / "experiments" / "training_stability_report.py"
    spec = importlib.util.spec_from_file_location("training_stability_report", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_log_scalars_uses_iteration_steps(tmp_path):
    mod = _load_report_module()
    log_path = tmp_path / "train.log"
    log_path.write_text(
        "\n".join(
            [
                "[INFO] [MCM] iter=10 loss_total=5.0 mcm_epe=4.0",
                "[INFO] [MCM] iter=20 loss_total=3.0 mcm_epe=2.0",
                "[INFO] [MCM][VAL] epoch=0 iter=20 val_loss_total=2.5 val_mcm_epe=1.5",
            ]
        ),
        encoding="utf-8",
    )

    series = mod.parse_log_scalars(
        log_path,
        ["mcm_train/loss_total", "mcm_val/loss_total", "mcm_val/mcm_epe"],
        max_step=None,
    )

    assert series["mcm_train/loss_total"].steps == [10, 20]
    assert series["mcm_train/loss_total"].values == [5.0, 3.0]
    assert series["mcm_val/loss_total"].steps == [20]
    assert series["mcm_val/mcm_epe"].values == [1.5]


def test_best_point_defaults_to_lower_is_better():
    mod = _load_report_module()
    series = mod.ScalarSeries(steps=[10, 20, 30], values=[3.0, 1.0, 2.0], source="test")

    assert mod.best_point(series, higher_is_better=False) == {"step": 20, "value": 1.0}
