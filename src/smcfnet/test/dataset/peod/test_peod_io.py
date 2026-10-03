from __future__ import annotations

import locale

import pytest

from smcfnet.config.config import load_config_from_yaml
from smcfnet.dataset.peod.io import load_peod_yaml, read_timestamps_us


def _set_ascii_locale() -> str:
    previous = locale.setlocale(locale.LC_CTYPE)
    try:
        locale.setlocale(locale.LC_CTYPE, "C")
    except locale.Error:
        pytest.skip("The C locale is unavailable")
    return previous


def test_peod_text_inputs_are_utf8_independent_of_process_locale(tmp_path):
    dataset_path = tmp_path / "peod.yaml"
    dataset_path.write_text(
        "# 30 epochs x 1000 steps - UTF-8: \u00d7 \u2014\n"
        "dataset:\n"
        "  name: PEOD\n",
        encoding="utf-8",
    )
    timestamps_path = tmp_path / "timestamps.csv"
    timestamps_path.write_text("1,1590802\n", encoding="utf-8")

    previous = _set_ascii_locale()
    try:
        assert load_peod_yaml(dataset_path)["dataset"]["name"] == "PEOD"
        assert load_config_from_yaml(str(dataset_path))["dataset"]["name"] == "PEOD"
        assert read_timestamps_us(timestamps_path) == [1590802]
    finally:
        locale.setlocale(locale.LC_CTYPE, previous)
