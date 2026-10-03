"""Unit tests for the shared provenance util (campaign #129)."""
import json
import re
from pathlib import Path

import pytest

from smcfnet.utils import provenance as P


def test_canonical_dump_handles_mixed_types():
    import enum

    class Color(enum.Enum):
        RED = "red"

    d = {"p": Path("/tmp/x"), "c": Color.RED, "t": (1, 2, 3), "f": 0.1}
    out = P.canonical_dump(d)  # must not raise
    assert "red" in out and "/tmp/x" in out
    # deterministic
    assert out == P.canonical_dump(d)


def test_float_collapse():
    assert P.canonical_dump({"a": 1.0}) == P.canonical_dump({"a": 1.0000000001})
    assert P.canonical_dump({"a": 1.0}) != P.canonical_dump({"a": 1.01})


def test_volatile_keys_dropped():
    base = {"x": 1, "log": {"output_dir": "./a", "experiment_name": "e"}}
    other = {"x": 1, "log": {"output_dir": "./DIFFERENT", "experiment_name": "e"},
             "wandb": {"entity": "z"}, "timestamp": "now"}
    # output_dir/wandb/timestamp must not affect the canonical payload
    assert P.canonical_dump(base) == P.canonical_dump(other)


def test_git_head_sha_shape():
    sha = P.git_head_sha()
    assert sha == "" or re.fullmatch(r"[0-9a-f]{40}", sha)


def _write(p: Path, text: str):
    p.write_text(text)
    return p


def test_resolved_hash_same_chain_equal_and_base_sensitive(tmp_path):
    base = _write(tmp_path / "base.yaml", "model:\n  num_classes: 8\n")
    child = _write(tmp_path / "child.yaml", f'_base_: "{base}"\nmodel:\n  size: s\n')
    h1 = P.resolved_config_sha256(child)
    h2 = P.resolved_config_sha256(child)
    assert h1 == h2  # two load paths of the same chain -> equal
    # change the _base_ parent -> resolved hash must change
    _write(base, "model:\n  num_classes: 3\n")
    assert P.resolved_config_sha256(child) != h1


def test_effective_changes_under_override_resolved_stable(tmp_path):
    base = _write(tmp_path / "base.yaml", "model:\n  num_classes: 8\n")
    child = _write(tmp_path / "child.yaml", f'_base_: "{base}"\nmodel:\n  size: s\n')
    from smcfnet.config.config import load_config_from_yaml
    merged = load_config_from_yaml(str(child))
    resolved_before = P.resolved_config_sha256(child)
    eff_no_override = P.sha256_of_str(P.canonical_dump(merged))
    # simulate a CLI/SLURM override on the live config
    merged["model"]["size"] = "m"
    eff_override = P.sha256_of_str(P.canonical_dump(merged))
    assert eff_override != eff_no_override            # effective hash moves
    assert P.resolved_config_sha256(child) == resolved_before  # resolved does not


def test_write_provenance_atomic_and_complete(tmp_path):
    base = _write(tmp_path / "base.yaml", "model:\n  num_classes: 8\n")
    child = _write(tmp_path / "child.yaml", f'_base_: "{base}"\nmodel:\n  size: s\n')
    from smcfnet.config.config import load_config_from_yaml
    cfg = load_config_from_yaml(str(child))
    out = tmp_path / "run"
    path = P.write_provenance(out, cfg, child, timestamp="2026-06-23T00:00:00")
    assert path.exists() and (out / "effective_config.yaml").exists()
    assert not (out / "provenance.json.tmp").exists()  # atomic cleanup
    prov = json.loads(path.read_text())
    for k in ("git_sha", "git_dirty", "config_path", "effective_config_sha256",
              "resolved_config_sha256", "slurm_job_id", "timestamp"):
        assert k in prov
    assert prov["timestamp"] == "2026-06-23T00:00:00"
    # effective hash in json matches the written effective file
    assert prov["effective_config_sha256"] == P.sha256_of_file(out / "effective_config.yaml")
