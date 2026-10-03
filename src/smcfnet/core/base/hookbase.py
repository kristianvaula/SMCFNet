from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Only import for type checking, not at runtime
    from smcfnet.core.trainer import Trainer

class HookBase:
  """
  Base class for hooks that can be registered with Trainer.
  Hooks allow modular extension of training loop behavior.
  """
  trainer: "Trainer" = None
  
  def before_train(self):
    """Called before training starts."""
    pass
  
  def after_train(self):
    """Called after training finishes."""
    pass
  
  def before_epoch(self):
    """Called before each epoch."""
    pass
  
  def after_epoch(self):
    """Called after each epoch."""
    pass
  
  def before_step(self):
    """Called before each training step."""
    pass
  
  def after_step(self):
    """Called after each training step."""
    pass
  
  def after_backward(self):
    """Called after backward pass."""
    pass

  def state_dict(self):
    """Return hook state for checkpointing."""
    return {}

  def load_state_dict(self, state_dict):
    """Restore hook state from checkpointing."""
    return
