import logging
from collections import defaultdict, deque
from typing import Dict

logger = logging.getLogger(__name__)


class MetricStorage:
  """
  Simple metric storage for tracking metrics during training.
  Inspired by detectron2's EventStorage but simplified.
  
  Uses bounded history to prevent unbounded memory growth across epochs.
  """
  def __init__(self, start_iter: int = 0, max_history_per_key: int = 2000):
    """
    Args:
      start_iter: Starting iteration number.
      max_history_per_key: Maximum number of history entries per metric key.
        When exceeded, oldest entries are discarded (FIFO). Default 2000 keeps
        ~20 epochs of 100 iters * 10 metrics in memory (~1 MB total).
    """
    self._history = defaultdict(lambda: deque(maxlen=max_history_per_key))
    self._smoothing_hints = {}
    self._latest_scalars = {}
    self.iter = int(start_iter)

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc, tb):
    # Do not suppress exceptions
    return False
  
  def put_scalar(self, name: str, value: float, smoothing_hint: bool = True):
    """Store a scalar value."""
    self._history[name].append((self.iter, value))
    self._latest_scalars[name] = value
    self._smoothing_hints[name] = smoothing_hint
  
  def put_scalars(self, **kwargs):
    """Store multiple scalars at once."""
    for k, v in kwargs.items():
        self.put_scalar(k, v)
  
  def latest(self) -> Dict[str, float]:
    """Get latest values of all scalars."""
    return self._latest_scalars.copy()
  
  def history(self, name: str) -> list[tuple[int, float]]:
    """Get a snapshot of one metric's bounded history.

    Returns a list copy of the internal deque to provide a stable public
    interface while storage remains deque-backed and size-limited.
    """
    return list(self._history[name])
  
  def clear_history(self):
    """Clear all history."""
    self._history.clear()
    self._latest_scalars.clear()
  
  def clear_latest_scalars(self):
    """Clear latest scalars while preserving history. Useful between epochs."""
    self._latest_scalars.clear()
