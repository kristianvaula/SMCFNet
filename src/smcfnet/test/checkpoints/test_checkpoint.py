"""Checkpoint tests for the current Trainer API.

These tests are intentionally lightweight and avoid DSEC filesystem
requirements by monkeypatching Trainer builders with synthetic components.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from smcfnet.config.config import Config
from smcfnet.core.trainer import Trainer


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def _patch_trainer_builders(monkeypatch: pytest.MonkeyPatch) -> None:
    def _build_model(cls, cfg):
        return _TinyModel()

    def _build_optimizer(cls, cfg, model):
        return torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.0)

    def _build_scheduler(cls, cfg, optimizer):
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    def _build_loader(cls, cfg):
        # Trainer construction only needs an iterable with length for these tests.
        return [
            {
                "events": torch.zeros((1, 4), dtype=torch.float32),
                "events_offsets": torch.tensor([0, 1], dtype=torch.int64),
                "events_batch_idx": torch.tensor([0], dtype=torch.int64),
            }
        ]

    monkeypatch.setattr(Trainer, "build_model", classmethod(_build_model))
    monkeypatch.setattr(Trainer, "build_optimizer", classmethod(_build_optimizer))
    monkeypatch.setattr(Trainer, "build_lr_scheduler", classmethod(_build_scheduler))
    monkeypatch.setattr(Trainer, "build_train_loader", classmethod(_build_loader))


@pytest.fixture()
def cfg(tmp_path: Path) -> Config:
    cfg = Config()
    cfg.log.output_dir = str(tmp_path)
    cfg.log.experiment_name = "checkpoint_test"
    cfg.wandb.use_wandb = False
    cfg.training.use_amp = False
    return cfg


def test_save_checkpoint_roundtrip(monkeypatch: pytest.MonkeyPatch, cfg: Config) -> None:
    _patch_trainer_builders(monkeypatch)

    trainer = Trainer(cfg)
    trainer.epoch = 5
    trainer.iteration = 100
    trainer.best_metric = 0.75

    trainer.save_checkpoint("test_checkpoint.pth")

    ckpt_path = Path(trainer.output_dir) / "test_checkpoint.pth"
    assert ckpt_path.exists()

    ckpt = torch.load(ckpt_path, map_location="cpu")
    for key in ["epoch", "iteration", "model", "optimizer", "scheduler", "best_metric", "config"]:
        assert key in ckpt

    assert int(ckpt["epoch"]) == 5
    assert int(ckpt["iteration"]) == 100
    assert float(ckpt["best_metric"]) == pytest.approx(0.75)


def test_resume_full_state(monkeypatch: pytest.MonkeyPatch, cfg: Config) -> None:
    _patch_trainer_builders(monkeypatch)

    t1 = Trainer(cfg)
    t1.epoch = 7
    t1.iteration = 321
    t1.best_metric = 0.88
    t1.save_checkpoint("resume_full.pth")

    t2 = Trainer(cfg)
    t2.resume_checkpoint(str(Path(t1.output_dir) / "resume_full.pth"), weights_only=False)

    assert t2.epoch == 7
    assert t2.iteration == 321
    assert t2.best_metric == pytest.approx(0.88)


def test_resume_weights_only_resets_training_state(monkeypatch: pytest.MonkeyPatch, cfg: Config) -> None:
    _patch_trainer_builders(monkeypatch)

    t1 = Trainer(cfg)
    t1.epoch = 9
    t1.iteration = 999
    t1.best_metric = 0.42
    t1.save_checkpoint("resume_weights_only.pth")

    t2 = Trainer(cfg)
    # Deliberately mutate state first to ensure reset behavior is explicit.
    t2.epoch = 123
    t2.iteration = 456
    t2.best_metric = 0.11
    t2.resume_checkpoint(str(Path(t1.output_dir) / "resume_weights_only.pth"), weights_only=True)

    assert t2.epoch == 0
    assert t2.iteration == 0
    assert t2.best_metric == pytest.approx(0.0)
