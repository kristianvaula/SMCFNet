from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import numpy as np
import torch
import math
import h5py
from numba import njit


DEFAULT_DT = 0.05  # 50 ms in seconds

@dataclass
class EventWindow:
  """
  Event window with configurable integer timestamp units.

  events tensor columns:
      [x, y, t, p]
  where:
      t: int64 timestamps in ``timestamp_unit`` units
      p: polarity in {-1, +1}
  """
  t_ref_ms: int
  dt_ms: int
  events: torch.Tensor
  timestamp_unit: str = "ms"
  sequence_id: Optional[str] = None
  frame_index: Optional[int] = None
  
  @property
  def t0_ms(self) -> int:
    return self.t_ref_ms - self.dt_ms

  def is_empty(self) -> bool:
    return self.events.numel() == 0
  
class DSECEventSlicer:
    """
    Efficient slicer for DSEC-style h5 event files.

    Usage:
        with h5py.File(events_h5_path, "r") as fh:
            slicer = DSECEventSlicer(fh)
            window = slicer.get_window_us(t_start_us, t_end_us)
    """
    
    def __init__(self, h5f: h5py.File):
      self.h5f = h5f
      # expect events under 'events/x','events/y','events/t','events/p'
      evgrp = h5f["events"] if "events" in h5f else h5f
      self.x_ds = evgrp["x"]
      self.y_ds = evgrp["y"]
      self.t_ds = evgrp["t"]   # microseconds on DSEC
      self.p_ds = evgrp["p"]

      # load the mapping from ms to event indices
      self.ms_to_idx = np.asarray(h5f["ms_to_idx"], dtype=np.int64)
      
      # t_offset is the time offset in microseconds that must be added to the timestamps of events
      self.t_offset = int(h5f["t_offset"][()]) if "t_offset" in h5f else 0
      # final timestamp in microseconds
      self.t_final_us = int(self.t_ds[-1]) + self.t_offset
      
    def get_conservative_window_ms(self, t_start_us: int, t_end_us: int) -> Tuple[int, int]:
      assert t_end_us > t_start_us
      win_start_ms = math.floor(t_start_us / 1000)
      win_end_ms = math.ceil(t_end_us / 1000)
      return int(win_start_ms), int(win_end_ms)
    
    def _refine_offsets_in_slice(self, time_array: np.ndarray, t_start_us: int, t_end_us: int) -> Tuple[int, int]:
      """
      Given time_array (numpy array of microsecond timestamps for the conservative slice),
      return local start and end indices such that:
        time_array[idx_start] >= t_start_us
        time_array[idx_end-1] < t_end_us
      Use numpy.searchsorted (vectorized) or optional numba linear scan for speed on small arrays
      """
      idx0 = int(np.searchsorted(time_array, t_start_us, side='left'))
      idx1 = int(np.searchsorted(time_array, t_end_us, side='left'))
      return idx0, idx1
    
    def ms_to_index(self, time_ms: int) -> Optional[int]:
      if time_ms < 0: 
        return 0
      if time_ms >= self.ms_to_idx.size:
        return None
      return int(self.ms_to_idx[time_ms])
    
    def get_window_us(
      self,
      t_start_us: int, 
      t_end_us: int,
      sequence_id: Optional[str]=None,
      frame_index: Optional[int]=None, 
      dt_ms: Optional[int]=None,
      timestamp_unit: str = "ms",
    ) -> Optional[EventWindow]:
      """
      Returns EventWindow for (t_start_us, t_end_us], where input times are in microseconds.
      If dt_ms is None, computed as ceil((t_end_us - t_start_us)/1000).
      """
      assert t_end_us > t_start_us
      if timestamp_unit not in {"ms", "us"}:
        raise ValueError(
          f"timestamp_unit must be 'ms' or 'us', got {timestamp_unit!r}"
        )
      
      # account for offset stored in file 
      t_start_us_local = int(t_start_us - self.t_offset)
      t_end_us_local = int(t_end_us - self.t_offset)
      
      if dt_ms is None:
        dt_ms = int(math.ceil((t_end_us - t_start_us) / 1000.0))
        
      # conservative ms window
      win_start_ms, win_end_ms = self.get_conservative_window_ms(t_start_us_local, t_end_us_local)
      
      # map ms -> coarse indices using ms_to_idx
      idx_start_coarse = self.ms_to_idx[win_start_ms]
      idx_end_coarse = self.ms_to_idx[win_end_ms]
      if idx_start_coarse is None or idx_end_coarse is None:
        # requested window is out of sequence time bounds
        return None
      
      # Read only the conservative timestamps for refinement
      # Using h5py slicing -> return numpy array
      time_slice = np.asarray(self.t_ds[idx_start_coarse:idx_end_coarse], dtype=np.int64)
      
      # refine offsets within this small slice
      local_offset_start, local_offset_end = self._refine_offsets_in_slice(time_slice, t_start_us_local, t_end_us_local)
      
      # compute absolute indices in whole dataset
      idx_start = idx_start_coarse + local_offset_start
      idx_end = idx_start_coarse + local_offset_end
      
      # fallback if no events in exact microsecond window
      if idx_end <= idx_start:
        empty = torch.empty((0,4), dtype=torch.int64)
        t_ref_ms = int(math.floor(t_end_us / 1000))
        return EventWindow(
          t_ref_ms=t_ref_ms,
          dt_ms=dt_ms,
          events=empty,
          timestamp_unit=timestamp_unit,
          sequence_id=sequence_id,
          frame_index=frame_index,
        )

      # now read the exact slices for x,y,p,t
      xs = np.asarray(self.x_ds[idx_start:idx_end], dtype=np.int32)
      ys = np.asarray(self.y_ds[idx_start:idx_end], dtype=np.int32)
      ps = np.asarray(self.p_ds[idx_start:idx_end], dtype=np.int8)
      ts_us = np.asarray(self.t_ds[idx_start:idx_end], dtype=np.int64)  # microseconds
      
      if timestamp_unit == "us":
        ts = ts_us + self.t_offset
      else:
        ts = (ts_us + self.t_offset) // 1000  # integer division: ms
      
      # assemble Nx4 int64 numpy array then to torch
      ev_np = np.stack([xs, ys, ts, ps], axis=1).astype(np.int64)
      ev_torch = torch.from_numpy(ev_np)
      
      t_ref_ms = int(math.floor((t_end_us) / 1000))
      return EventWindow(
        t_ref_ms=t_ref_ms,
        dt_ms=dt_ms,
        events=ev_torch,
        timestamp_unit=timestamp_unit,
        sequence_id=sequence_id,
        frame_index=frame_index,
      )
