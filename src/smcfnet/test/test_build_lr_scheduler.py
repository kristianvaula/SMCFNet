import unittest

import torch

from smcfnet.config.config import Config
from smcfnet.model.build import build_lr_scheduler


class TestBuildLRScheduler(unittest.TestCase):
  def _make(self):
    cfg = Config()
    cfg.training.num_epochs = 10
    cfg.training.learning_rate = 1e-2
    cfg.training.warmup_lr = 1e-4

    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=float(cfg.training.learning_rate))
    return cfg, optimizer

  def test_cosine(self):
    cfg, opt = self._make()
    cfg.training.lr_scheduler = "cosine"
    sched = build_lr_scheduler(cfg, opt)
    self.assertIsInstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)

  def test_step(self):
    cfg, opt = self._make()
    cfg.training.lr_scheduler = "step"
    cfg.training.lr_step_size = 3
    cfg.training.lr_gamma = 0.5
    sched = build_lr_scheduler(cfg, opt)
    self.assertIsInstance(sched, torch.optim.lr_scheduler.StepLR)

  def test_multistep(self):
    cfg, opt = self._make()
    cfg.training.lr_scheduler = "multistep"
    cfg.training.lr_milestones = [3, 7]
    cfg.training.lr_gamma = 0.2
    sched = build_lr_scheduler(cfg, opt)
    self.assertIsInstance(sched, torch.optim.lr_scheduler.MultiStepLR)

  def test_exponential(self):
    cfg, opt = self._make()
    cfg.training.lr_scheduler = "exponential"
    cfg.training.lr_gamma = 0.95
    sched = build_lr_scheduler(cfg, opt)
    self.assertIsInstance(sched, torch.optim.lr_scheduler.ExponentialLR)

  def test_onecycle(self):
    cfg, opt = self._make()
    cfg.training.lr_scheduler = "onecycle"
    cfg.training.steps_per_epoch = 5
    sched = build_lr_scheduler(cfg, opt)
    self.assertIsInstance(sched, torch.optim.lr_scheduler.OneCycleLR)

  def test_warmup_sequential_optional(self):
    cfg, opt = self._make()
    cfg.training.lr_scheduler = "cosine"
    cfg.training.warmup_use_scheduler = True
    cfg.training.warmup_epochs = 2
    cfg.training.warmup_lr = 1e-4

    sched = build_lr_scheduler(cfg, opt)
    self.assertIsInstance(sched, torch.optim.lr_scheduler.SequentialLR)


if __name__ == "__main__":
  unittest.main()
