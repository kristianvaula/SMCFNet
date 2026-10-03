"""
Utility functions for event processing.
Compatible with DSEC dataset format.
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Union


def events_to_voxel_grid(
  events: Union[Dict, np.ndarray],
  num_bins: int,
  width: int,
  height: int,
  source_width: int | None = None,
  source_height: int | None = None,
) -> torch.Tensor:
  """Convert event stream to voxel grid representation (NumPy backend).

  Args:
    events: Either dict with keys {'x','y','t_ms','p'} or Nx4 array (x, y, t_ms, p)
    num_bins: Number of temporal bins
    width: Image width
    height: Image height

  Returns:
    Voxel grid of shape (num_bins, height, width)
  """
  voxel = np.zeros((num_bins, height, width), dtype=np.float32)

  if isinstance(events, dict):
    if len(events["x"]) == 0:
      return torch.from_numpy(voxel)

    x = events["x"] if isinstance(events["x"], np.ndarray) else events["x"].numpy()
    y = events["y"] if isinstance(events["y"], np.ndarray) else events["y"].numpy()
    t_ms = events["t_ms"] if isinstance(events["t_ms"], np.ndarray) else events["t_ms"].numpy()
    p = events["p"] if isinstance(events["p"], np.ndarray) else events["p"].numpy()
  else:
    if len(events) == 0:
      return torch.from_numpy(voxel)
    x = events[:, 0]
    y = events[:, 1]
    t_ms = events[:, 2]
    p = events[:, 3]

  if len(t_ms) > 0:
    t_min, t_max = t_ms.min(), t_ms.max()
    if t_max > t_min:
      t = (t_ms - t_min) / (t_max - t_min)
    else:
      t = np.zeros_like(t_ms, dtype=np.float32)
  else:
    return torch.from_numpy(voxel)

  src_w = int(source_width) if source_width is not None else int(width)
  src_h = int(source_height) if source_height is not None else int(height)
  if src_w <= 0 or src_h <= 0:
    raise ValueError(f"source_width/source_height must be positive, got {src_w}x{src_h}")

  if src_w != int(width) or src_h != int(height):
    x = np.floor(x.astype(np.float32, copy=False) * (float(width) / float(src_w))).astype(np.int32)
    y = np.floor(y.astype(np.float32, copy=False) * (float(height) / float(src_h))).astype(np.int32)
  else:
    x = x.astype(np.int32)
    y = y.astype(np.int32)

  valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
  x, y, t, p = x[valid], y[valid], t[valid], p[valid]

  if len(x) == 0:
    return torch.from_numpy(voxel)

  bin_pos = t.astype(np.float32) * (num_bins - 1)
  lower_bin = np.floor(bin_pos).astype(np.int64)
  upper_bin = np.ceil(bin_pos).astype(np.int64)

  lower_bin = np.clip(lower_bin, 0, num_bins - 1)
  upper_bin = np.clip(upper_bin, 0, num_bins - 1)

  frac = (bin_pos - lower_bin.astype(np.float32)).astype(np.float32)
  lower_weight = (1.0 - frac).astype(np.float32)
  upper_weight = frac

  p = p.astype(np.float32)
  y = y.astype(np.int64)
  x = x.astype(np.int64)

  np.add.at(voxel, (lower_bin, y, x), p * lower_weight)

  mask = upper_bin != lower_bin
  if np.any(mask):
    np.add.at(voxel, (upper_bin[mask], y[mask], x[mask]), p[mask] * upper_weight[mask])

  return torch.from_numpy(voxel)

def events_to_cuadrado_histogram_torch(
  events: torch.Tensor,
  num_bins: int,
  width: int,
  height: int,
  *,
  t_start: float | torch.Tensor | None = None,
  t_end: float | torch.Tensor | None = None,
  source_width: int | None = None,
  source_height: int | None = None,
) -> torch.Tensor:
  """Build OF_EV_SNN/Cuadrado split-polarity event-count histograms.

  Returns a tensor with shape ``(2, num_bins, height, width)`` where channel 0
  contains positive-event counts and channel 1 contains negative-event counts.
  Unlike ``events_to_voxel_grid_torch``, this representation uses hard temporal
  bins and count accumulation, matching the OF_EV_SNN input contract.

  ``t_start``/``t_end`` should describe the detector-aligned event window in
  the same timestamp unit as ``events[:, 2]``. If they are omitted, the function
  falls back to the min/max event timestamps for the sample.
  """
  if not isinstance(events, torch.Tensor):
    events = torch.as_tensor(events)

  device = events.device
  b = int(num_bins)
  h = int(height)
  w = int(width)
  if b <= 0:
    raise ValueError(f"num_bins must be > 0, got {b}")
  if h <= 0 or w <= 0:
    raise ValueError(f"height/width must be positive, got {h}x{w}")

  out = torch.zeros((2, b, h, w), device=device, dtype=torch.float32)
  if events.numel() == 0:
    return out
  if events.ndim != 2 or events.shape[1] != 4:
    raise ValueError(f"Expected events of shape [N,4], got {tuple(events.shape)}")

  src_w = int(source_width) if source_width is not None else w
  src_h = int(source_height) if source_height is not None else h
  if src_w <= 0 or src_h <= 0:
    raise ValueError(f"source_width/source_height must be positive, got {src_w}x{src_h}")

  if src_w != w or src_h != h:
    x = torch.floor(events[:, 0].to(torch.float32) * (float(w) / float(src_w))).to(torch.int64)
    y = torch.floor(events[:, 1].to(torch.float32) * (float(h) / float(src_h))).to(torch.int64)
  else:
    x = events[:, 0].to(torch.int64)
    y = events[:, 1].to(torch.int64)
  t = events[:, 2].to(torch.float32)
  p = events[:, 3].to(torch.float32)

  valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
  if not torch.any(valid):
    return out

  x = x[valid]
  y = y[valid]
  t = t[valid]
  p = p[valid]

  if t_start is None or t_end is None:
    t0 = torch.min(t)
    t1 = torch.max(t)
  else:
    t0 = torch.as_tensor(t_start, device=device, dtype=torch.float32)
    t1 = torch.as_tensor(t_end, device=device, dtype=torch.float32)

  denom = t1 - t0
  if float(denom) <= 0.0:
    t0 = torch.min(t)
    t1 = torch.max(t)
    denom = t1 - t0

  if float(denom) > 0.0:
    rel = torch.clamp((t - t0) / denom, 0.0, 1.0)
  else:
    rel = torch.zeros_like(t, dtype=torch.float32)

  bin_idx = torch.floor(rel * float(b)).to(torch.int64)
  bin_idx = torch.clamp(bin_idx, 0, b - 1)
  pol_idx = torch.where(p > 0, torch.zeros_like(bin_idx), torch.ones_like(bin_idx))

  hw = int(h * w)
  flat = pol_idx * (b * hw) + bin_idx * hw + y * w + x
  out.view(-1).scatter_add_(0, flat, torch.ones_like(t, dtype=torch.float32))
  return out


def events_to_voxel_grid_torch(
  events: torch.Tensor,
  num_bins: int,
  width: int,
  height: int,
  source_width: int | None = None,
  source_height: int | None = None,
) -> torch.Tensor:
  """Torch implementation of voxelization (CUDA-friendly)."""
  if not isinstance(events, torch.Tensor):
    events = torch.as_tensor(events)

  device = events.device
  if events.numel() == 0:
    return torch.zeros((int(num_bins), int(height), int(width)), device=device, dtype=torch.float32)

  if events.ndim != 2 or events.shape[1] != 4:
    raise ValueError(f"Expected events of shape [N,4], got {tuple(events.shape)}")

  src_w = int(source_width) if source_width is not None else int(width)
  src_h = int(source_height) if source_height is not None else int(height)
  if src_w <= 0 or src_h <= 0:
    raise ValueError(f"source_width/source_height must be positive, got {src_w}x{src_h}")

  if src_w != int(width) or src_h != int(height):
    x = torch.floor(events[:, 0].to(torch.float32) * (float(width) / float(src_w))).to(torch.int64)
    y = torch.floor(events[:, 1].to(torch.float32) * (float(height) / float(src_h))).to(torch.int64)
  else:
    x = events[:, 0].to(torch.int64)
    y = events[:, 1].to(torch.int64)
  t_ms = events[:, 2].to(torch.float32)
  p = events[:, 3].to(torch.float32)

  w = int(width)
  h = int(height)
  b = int(num_bins)
  if b <= 0:
    raise ValueError(f"num_bins must be > 0, got {b}")

  valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
  if not torch.any(valid):
    return torch.zeros((b, h, w), device=device, dtype=torch.float32)

  x = x[valid]
  y = y[valid]
  t_ms = t_ms[valid]
  p = p[valid]

  t_min = torch.min(t_ms)
  t_max = torch.max(t_ms)
  denom = (t_max - t_min)
  if float(denom) > 0.0:
    t = (t_ms - t_min) / denom
  else:
    t = torch.zeros_like(t_ms, dtype=torch.float32)

  bin_pos = t * float(b - 1)
  lower = torch.floor(bin_pos).to(torch.int64)
  upper = torch.ceil(bin_pos).to(torch.int64)
  lower = torch.clamp(lower, 0, b - 1)
  upper = torch.clamp(upper, 0, b - 1)

  frac = (bin_pos - lower.to(torch.float32)).to(torch.float32)
  lower_w = (1.0 - frac)
  upper_w = frac

  hw = int(h * w)
  flat0 = lower * hw + y * w + x
  voxel_flat = torch.zeros((b * hw,), device=device, dtype=torch.float32)
  voxel_flat.scatter_add_(0, flat0, p * lower_w)

  mask = upper != lower
  if torch.any(mask):
    flat1 = upper[mask] * hw + y[mask] * w + x[mask]
    voxel_flat.scatter_add_(0, flat1, p[mask] * upper_w[mask])

  return voxel_flat.view(b, h, w)

def upscale_voxel_grid(
  voxel: torch.Tensor,
  target_height: int,
  target_width: int,
  mode: str = 'bilinear',
  preserve_density: bool = True,
) -> torch.Tensor:
  """
  Upscale event voxel grid to target resolution.
  
  Args:
    voxel: Input voxel grid of shape (num_bins, H, W)
    target_height: Target height
    target_width: Target width
    mode: Interpolation mode ('bilinear', 'bicubic', or 'nearest')
    preserve_density: If True, scale values to preserve event density
  
  Returns:
    Upscaled voxel grid of shape (num_bins, target_height, target_width)
  
  Implementation details:
  - Uses PyTorch's interpolate for efficient GPU acceleration
  - Bilinear interpolation maintains smoothness while preserving structure
  - Optionally preserves event density (events per pixel)
  """
  orig_h, orig_w = voxel.shape[1], voxel.shape[2]
  
  if orig_h == target_height and orig_w == target_width:
    return voxel
  
  # Add batch dimension for interpolation: (B, H, W) -> (1, B, H, W)
  voxel_batched = voxel.unsqueeze(0)
  
  # Upscale using interpolation
  voxel_upscaled = F.interpolate(
    voxel_batched,
    size=(target_height, target_width),
    mode=mode,
    align_corners=False if mode != 'nearest' else None,
  )
  
  # Remove batch dimension
  voxel_upscaled = voxel_upscaled.squeeze(0)
  
  if preserve_density:
    # Scale values to preserve event density
    # When upscaling, the same events are spread over more pixels
    # So we scale by the area ratio to maintain total "event mass"
    scale_factor = (target_height * target_width) / (orig_h * orig_w)
    voxel_upscaled = voxel_upscaled / scale_factor
    
  return voxel_upscaled

class EventRepresentation:
  """Factory class for different event representations."""
  
  @staticmethod
  def voxel_grid(
    events: Union[Dict, np.ndarray],
    num_bins: int,
    width: int,
    height: int,
  ) -> torch.Tensor:
    """Create voxel grid representation."""
    return events_to_voxel_grid(events, num_bins, width, height)
  
  @staticmethod
  def time_surface(
    events: Union[Dict, np.ndarray],
    width: int,
    height: int,
    tau: float = 0.1,
  ) -> torch.Tensor:
    """
    Create time surface representation.
    
    Args:
        events: Either dict or Nx4 array (x, y, t_ms, p)
        width: Image width
        height: Image height
        tau: Time decay constant
    
    Returns:
        Time surface of shape (2, height, width) for pos/neg polarities
    """
    time_surface = np.zeros((2, height, width), dtype=np.float32)
    
    # Handle dict format
    if isinstance(events, dict):
      if len(events['x']) == 0:
        return torch.from_numpy(time_surface)
      x = events['x'] if isinstance(events['x'], np.ndarray) else events['x'].numpy()
      y = events['y'] if isinstance(events['y'], np.ndarray) else events['y'].numpy()
      t_ms = events['t_ms'] if isinstance(events['t_ms'], np.ndarray) else events['t_ms'].numpy()
      p = events['p'] if isinstance(events['p'], np.ndarray) else events['p'].numpy()
    else:
      if len(events) == 0:
        return torch.from_numpy(time_surface)
      x = events[:, 0]
      y = events[:, 1]
      t_ms = events[:, 2]
      p = events[:, 3]
    
    # Normalize time to [0, 1]
    if len(t_ms) > 0:
      t_min, t_max = t_ms.min(), t_ms.max()
      if t_max > t_min:
        t_norm = (t_ms - t_min) / (t_max - t_min)
      else:
        t_norm = np.zeros_like(t_ms, dtype=np.float32)
    else:
      return torch.from_numpy(time_surface)
    
    x = x.astype(np.int32)
    y = y.astype(np.int32)
    
    # Clip to bounds
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x, y, t_norm, p = x[valid], y[valid], t_norm[valid], p[valid]
    
    # Update time surface with exponential decay
    t_ref = 1.0  # Reference time (most recent)
    
    for i in range(len(x)):
      pol_idx = 0 if p[i] > 0 else 1
      decay = np.exp(-(t_ref - t_norm[i]) / tau)
      time_surface[pol_idx, y[i], x[i]] = max(
        time_surface[pol_idx, y[i], x[i]],
        decay
      )
    
    return torch.from_numpy(time_surface)
