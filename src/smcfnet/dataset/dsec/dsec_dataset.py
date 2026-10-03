#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Concatenated dataset wrapper for DSEC sequences.
Enables training on multiple sequences while maintaining per-sequence caching.
"""

import bisect
from typing import List
from torch.utils.data import ConcatDataset as TorchConcatDataset


class DSECDataset(TorchConcatDataset):
  """
  Concatenates multiple DSEC sequence datasets for multi-sequence training.
  
  Features:
  - Efficient indexing across sequences using cumulative sizes
  - Maintains per-sequence caching for fast data loading
  - Supports pull_item for direct sample access
  - Preserves input dimensions and properties from first dataset
  
  Args:
    datasets: List of DSECDataset instances (one per sequence)
  """
  
  def __init__(self, datasets: List):
    super().__init__(datasets)
    
    # Inherit properties from first dataset
    if hasattr(self.datasets[0], "input_dim"):
      self._input_dim = self.datasets[0].input_dim
      self.input_dim = self.datasets[0].input_dim
    
    # Store dataset names for logging
    self.sequence_names = [
      getattr(ds, 'sequence', f'seq_{i}') 
      for i, ds in enumerate(self.datasets)
    ]
    
    # Cache cumulative sizes for efficiency
    self._cumulative_sizes = self.cumulative_sizes
  
  def pull_item(self, idx: int):
    """
    Pull a single item by absolute index across all sequences.
    
    This enables direct access to samples without triggering augmentation,
    useful for validation and visualization.
    
    Args:
      idx: Global index across all concatenated sequences
        
    Returns:
      Sample data from the appropriate sequence
    """
    if idx < 0:
      if -idx > len(self):
        raise ValueError(
          "Absolute value of index should not exceed dataset length"
        )
      idx = len(self) + idx
    
    # Find which dataset this index belongs to
    dataset_idx = bisect.bisect_right(self._cumulative_sizes, idx)
    
    # Calculate local index within that dataset
    if dataset_idx == 0:
      sample_idx = idx
    else:
      sample_idx = idx - self._cumulative_sizes[dataset_idx - 1]
    
    # Get item from appropriate dataset
    return self.datasets[dataset_idx].pull_item(sample_idx)
  
  def get_sequence_info(self) -> dict:
    """
    Get information about all sequences in this concatenated dataset.
    
    Returns:
        Dictionary with sequence statistics
    """
    info = {
      'num_sequences': len(self.datasets),
      'total_samples': len(self),
      'sequences': []
    }
    
    for i, (ds, name) in enumerate(zip(self.datasets, self.sequence_names)):
      seq_info = {
        'name': name,
        'samples': len(ds),
        'start_idx': 0 if i == 0 else self._cumulative_sizes[i - 1],
        'end_idx': self._cumulative_sizes[i],
      }
      
      # Add caching info if available
      if hasattr(ds, 'cache'):
        seq_info['cache_enabled'] = ds.cache
        seq_info['cache_type'] = getattr(ds, 'cache_type', 'ram')
      
      info['sequences'].append(seq_info)
    
    return info
  
  def __repr__(self) -> str:
    """String representation with sequence information."""
    info = self.get_sequence_info()
    return (
      f"DSECConcatDataset(\n"
      f"  num_sequences={info['num_sequences']},\n"
      f"  total_samples={info['total_samples']},\n"
      f"  sequences={self.sequence_names}\n"
      f")"
    )
