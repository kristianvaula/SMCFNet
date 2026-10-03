"""Based on https://github.com/facebookresearch/detectron2/blob/main/detectron2/engine/defaults.py"""
from typing import List, Optional
import weakref
import logging

from smcfnet.core.base import HookBase
from smcfnet.utils.metric_storage import MetricStorage

class TrainerBase:
  """
  Base class for iterative trainer with hooks.
    
  The only assumption they made here is: the training runs in a loop.
  A subclass can implement what the loop is.
  They made no assumptions about the existence of dataloader, optimizer, model, etc.

  Attributes:
    iter(int): the current iteration.

    start_iter(int): The iteration to start with.
        By convention the minimum possible value is 0.

    max_iter(int): The iteration to end training.

    storage(EventStorage): An EventStorage that's opened during the course of training.
  """
  
  def __init__(self) -> None:
    self._hooks: List[HookBase] = []
    self.iter: int = 0
    self.start_iter: int = 0
    self.max_iter: int = 0
    self.storage: Optional[MetricStorage] = None
  
  def register_hooks(self, hooks: List[Optional[HookBase]]) -> None:
    """Register hooks with the trainer."""
    hooks = [h for h in hooks if h is not None]
    for h in hooks:
      assert isinstance(h, HookBase)
      h.trainer = weakref.proxy(self)
    self._hooks.extend(hooks)
  
  def train(self, start_iter: int, max_iter: int) -> None:
    """Main training loop"""
    
    logger = logging.getLogger(__name__)
    logger.info("Starting training from iteration {}".format(start_iter))
    
    self.iter = self.start_iter = start_iter
    self.max_iter = max_iter
    
    with MetricStorage(start_iter) as self.storage:
      try:
        self.before_train()
        for self.iter in range(start_iter, max_iter):
          self.before_step()
          self.run_step()
          self.after_step()
        self.iter += 1
      except Exception:
        logger.exception("Exception during training:")
        raise
      finally:
        self.after_train()

  def before_train(self):
    for h in self._hooks:
      h.before_train()

  def before_epoch(self):
    for h in self._hooks:
      h.before_epoch()

  def after_epoch(self):
    for h in self._hooks:
      h.after_epoch()

  def after_train(self):
    if self.storage is not None:
      self.storage.iter = self.iter
    for h in self._hooks:
      h.after_train()
      
  def before_step(self):
    if self.storage is not None:
      self.storage.iter = self.iter

    for h in self._hooks:
      h.before_step()

  def after_backward(self):
    for h in self._hooks:
      h.after_backward()

  def after_step(self):
    for h in self._hooks:
      h.after_step()

  def run_step(self):
    raise NotImplementedError

  def state_dict(self):
    ret = {"iteration": self.iter}
    hooks_state = {}
    for h in self._hooks:
      sd = h.state_dict()
      if sd:
        name = type(h).__qualname__
        if name in hooks_state:
          continue
        hooks_state[name] = sd
    if hooks_state:
      ret["hooks"] = hooks_state
    return ret

  def load_state_dict(self, state_dict):
    logger = logging.getLogger(__name__)
    self.iter = state_dict["iteration"]
    for key, value in state_dict.get("hooks", {}).items():
      for h in self._hooks:
        try:
          name = type(h).__qualname__
        except AttributeError:
          continue
        if name == key:
          h.load_state_dict(value)
          break
      else:
        logger.warning(f"Cannot find the hook '{key}', its state_dict is ignored.")