#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

from smcfnet.model.network_blocks import BaseConv, CSPLayer, DWConv

class YOLOPAFPN(nn.Module):
  """
  YOLOX PAFPN neck for multi-scale feature fusion.
  
  Takes backbone features at 3 scales and creates enhanced
  feature pyramid with both FPN (top-down) and PAN (bottom-up) paths.
  
  Architecture:
    Input: [P3/8, P4/16, P5/32] from backbone
    
    FPN (top-down):
      P5 → P4 → P3
    
    PAN (bottom-up):
      P3 → P4 → P5
    
  Output: [N3/8, N4/16, N5/32] enhanced features
  """
  def __init__(
    self,
    depth=1.0,
    width=1.0,
    in_features=("dark3", "dark4", "dark5"),
    in_channels=[256, 512, 1024],
    depthwise=False,
    act="silu",
  ):
    super().__init__()
          
    self.in_features = in_features
    Conv = DWConv if depthwise else BaseConv
    
    # Calculate actual channel numbers based on width multiplier
    # Standard: [256, 512, 1024] for dark3, dark4, dark5
    self.in_channels = [int(ch * width) for ch in in_channels]
    
    # ============================================================
    # FPN (Feature Pyramid Network) - Top-down pathway
    # ============================================================
    
    # Lateral connection for P5 -> P4
    # Reduces P5 channels: 1024 -> 512
    self.lateral_conv0 = BaseConv(
      self.in_channels[2],  # 1024
      self.in_channels[1],  # 512
      ksize=1,
      stride=1,
      act=act
    )
  
    # CSP block after concatenating upsampled P5 with P4
    # Input: 512 (P4) + 512 (upsampled P5) = 1024
    # Output: 512
    self.C3_p4 = CSPLayer(
      2 * self.in_channels[1],  # 1024
      self.in_channels[1],       # 512
      n=round(3 * depth),
      shortcut=False,
      depthwise=depthwise,
      act=act,
    )
        
    # Lateral connection for P4 -> P3
    # Reduces P4 channels: 512 -> 256
    self.reduce_conv1 = BaseConv(
      self.in_channels[1],  # 512
      self.in_channels[0],  # 256
      ksize=1,
      stride=1,
      act=act
    )
        
    # CSP block after concatenating upsampled P4 with P3
    # Input: 256 (P3) + 256 (upsampled P4) = 512
    # Output: 256
    self.C3_p3 = CSPLayer(
      2 * self.in_channels[0],  # 512
      self.in_channels[0],       # 256
      n=round(3 * depth),
      shortcut=False,
      depthwise=depthwise,
      act=act,
    )
        
    # ============================================================
    # PAN (Path Aggregation Network) - Bottom-up pathway
    # ============================================================
    
    # Downsample P3 -> P4
    # 256 channels, stride 2
    self.bu_conv2 = Conv(
      self.in_channels[0],  # 256
      self.in_channels[0],  # 256
      ksize=3,
      stride=2,
      act=act
    )
        
    # CSP block after concatenating downsampled P3 with P4
    # Input: 256 (downsampled P3) + 256 (FPN P4) = 512
    # Output: 512
    self.C3_n3 = CSPLayer(
      2 * self.in_channels[0],  # 512
      self.in_channels[1],       # 512
      n=round(3 * depth),
      shortcut=False,
      depthwise=depthwise,
      act=act,
    )
        
    # Downsample P4 -> P5
    # 512 channels, stride 2
    self.bu_conv1 = Conv(
      self.in_channels[1],  # 512
      self.in_channels[1],  # 512
      ksize=3,
      stride=2,
      act=act
    )
        
    # CSP block after concatenating downsampled P4 with P5
    # Input: 512 (downsampled P4) + 512 (FPN P5) = 1024
    # Output: 1024
    self.C3_n4 = CSPLayer(
      2 * self.in_channels[1],  # 1024
      self.in_channels[2],       # 1024
      n=round(3 * depth),
      shortcut=False,
      depthwise=depthwise,
      act=act,
    )
        
    # Upsampling layer (nearest neighbor interpolation)
    self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

  @staticmethod
  def _resize_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Resize `x` to match `ref` spatial size.

    This makes the neck robust to inputs whose height/width are not divisible by
    the backbone's total stride (e.g. H=1080), which can otherwise yield
    off-by-one feature map sizes and break concatenations.
    """
    if x.shape[-2:] == ref.shape[-2:]:
      return x
    return F.interpolate(x, size=ref.shape[-2:], mode="nearest")
    
  def forward(self, features):
    """
    Forward pass through PAFPN.
    
    Args:
        features: Can be either:
            - List [dark3, dark4, dark5] of tensors
            - Dict {'dark3': tensor, 'dark4': tensor, 'dark5': tensor}
    
    Returns:
        Tuple of enhanced features (P3, P4, P5) for detection head
            P3: (B, 256*width, H/8, W/8)
            P4: (B, 512*width, H/16, W/16)
            P5: (B, 1024*width, H/32, W/32)
    """
    
    # Handle both list and dict inputs
    if isinstance(features, dict):
      x2 = features['dark3']  # P3/8, 256 channels
      x1 = features['dark4']  # P4/16, 512 channels
      x0 = features['dark5']  # P5/32, 1024 channels
    else:
      # Assume list [dark3, dark4, dark5]
      x2, x1, x0 = features
        
    # ============================================================
    # FPN Top-down Pathway
    # ============================================================
    
    # Process P5 (largest receptive field)
    # 1024 -> 512 channels
    fpn_out0 = self.lateral_conv0(x0)
        
    # Upsample and fuse with P4
    f_out0 = self.upsample(fpn_out0)           # 512, ~H/16, ~W/16
    f_out0 = self._resize_like(f_out0, x1)
    f_out0 = torch.cat([f_out0, x1], dim=1)    # Concat: 512 + 512 = 1024
    f_out0 = self.C3_p4(f_out0)                # CSP: 1024 -> 512
    
    # Process P4
    # 512 -> 256 channels
    fpn_out1 = self.reduce_conv1(f_out0)
    
    # Upsample and fuse with P3
    f_out1 = self.upsample(fpn_out1)           # 256, ~H/8, ~W/8
    f_out1 = self._resize_like(f_out1, x2)
    f_out1 = torch.cat([f_out1, x2], dim=1)    # Concat: 256 + 256 = 512
    pan_out2 = self.C3_p3(f_out1)              # CSP: 512 -> 256 (final P3)
    
    # ============================================================
    # PAN Bottom-up Pathway
    # ============================================================
    
    # Downsample P3 and fuse with P4
    p_out1 = self.bu_conv2(pan_out2)           # Downsample: 256, H/16, W/16
    p_out1 = self._resize_like(p_out1, fpn_out1)
    p_out1 = torch.cat([p_out1, fpn_out1], dim=1)  # Concat: 256 + 256 = 512
    pan_out1 = self.C3_n3(p_out1)              # CSP: 512 -> 512 (final P4)
    
    # Downsample P4 and fuse with P5
    p_out0 = self.bu_conv1(pan_out1)           # Downsample: 512, H/32, W/32
    p_out0 = self._resize_like(p_out0, fpn_out0)
    p_out0 = torch.cat([p_out0, fpn_out0], dim=1)  # Concat: 512 + 512 = 1024
    pan_out0 = self.C3_n4(p_out0)              # CSP: 1024 -> 1024 (final P5)
    
    # ============================================================
    # Output multi-scale features for detection head
    # ============================================================
    # pan_out2: P3/8 - Small objects (256 channels)
    # pan_out1: P4/16 - Medium objects (512 channels)
    # pan_out0: P5/32 - Large objects (1024 channels)
    
    outputs = (pan_out2, pan_out1, pan_out0)
    return outputs