"""
CSPDarknet53 Backbone 
"""

import torch
import torch.nn as nn
from smcfnet.model.network_blocks import BaseConv, DWConv, Focus, CSPLayer, SPPBottleneck
from smcfnet.utils.profiling import CudaStageTimer

class CSPDarknetDualStream(nn.Module):
  """
  Dual-stream CSPDarknet backbone for RGB and Event data.
  
  Phase 1 Implementation:
  - Separate processing streams for RGB and Event data
  - Simple concatenation fusion at specified layers (dark3, dark4, dark5)
  - No advanced modules (ECM, EDUM, CMM) - these are for later phases
  
  Args:
      dep_mul (float): Depth multiplier for number of layers
      wid_mul (float): Width multiplier for number of channels
      out_features (tuple): Output feature layers, default ("dark3", "dark4", "dark5")
      depthwise (bool): Use depthwise convolutions
      act (str): Activation function type
  """
  def __init__(
    self,
    dep_mul=1.0,
    wid_mul=1.0,
    out_features=("dark3", "dark4", "dark5"),
    depthwise=False,
    act="silu",
    event_in_channels=5,
    use_mcm=False,
    mcm_instance=None,
    use_edum=False,
    edum_kwargs=None,
    use_cmm=False,
    cmm_kwargs=None,
    # Legacy kwargs — accepted for backward compat but ignored.
    # Use use_mcm + mcm_instance instead.
    use_ecm=False,
    use_smcm=False,
    ecm_kwargs=None,
    smcm_instance=None,
  ):
    super().__init__()
    assert out_features, "please provide output features of Darknet"
    self.out_features = out_features
    self.use_edum = use_edum
    self.use_cmm = use_cmm

    # Unified MCM flag.  Accept legacy use_ecm / use_smcm for backward compat.
    if not use_mcm and (use_ecm or use_smcm):
      use_mcm = True
    self.use_mcm = use_mcm
    
    Conv = DWConv if depthwise else BaseConv
    
    # Calculate channel dimensions based on width multiplier
    base_channels = int(wid_mul * 64)  # 64
    base_depth = max(round(dep_mul * 3), 1)  # 3
    
    # ============================================================
    # RGB Stream (Standard CSPDarknet)
    # ============================================================
    
    # RGB Stem: Focus layer (space-to-depth) + conv
    # Input: (B, 3, H, W) -> Output: (B, base_channels, H/2, W/2)
    self.stem_rgb = Focus(3, base_channels, ksize=3, act=act)
    
    # RGB Dark2: Downsample + CSP block
    self.dark2_rgb = nn.Sequential(
      Conv(base_channels, base_channels * 2, 3, 2, act=act),
      CSPLayer(
        base_channels * 2,
        base_channels * 2,
        n=base_depth,
        depthwise=depthwise,
        act=act,
      ),
    )
    
    # RGB Dark3: Downsample + CSP block
    self.dark3_rgb = nn.Sequential(
      Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
      CSPLayer(
        base_channels * 4,
        base_channels * 4,
        n=base_depth * 3,
        depthwise=depthwise,
        act=act,
      ),
    )   
    
    # RGB Dark4: Downsample + CSP block
    self.dark4_rgb = nn.Sequential(
      Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
      CSPLayer(
        base_channels * 8,
        base_channels * 8,
        n=base_depth * 3,
        depthwise=depthwise,
        act=act,
      ),
    ) 
    
    # RGB Dark5: Downsample + CSP block
    self.dark5_rgb = nn.Sequential(
      Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
      SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
      CSPLayer(
        base_channels * 16,
        base_channels * 16,
        n=base_depth,
        shortcut=False,
        depthwise=depthwise,
        act=act,
      ),
    )
    
    # ============================================================
    # Event Stream (Parallel CSPDarknet for event voxel grids)
    # ============================================================
    
    # Event Stem: Process configured event representation.
    self.event_in_channels = int(event_in_channels)
    self.stem_event = Focus(self.event_in_channels, base_channels, ksize=3, act=act)
    
    # Event Dark2
    self.dark2_event = nn.Sequential(
      Conv(base_channels, base_channels * 2, 3, 2, act=act),
      CSPLayer(
        base_channels * 2,
        base_channels * 2,
        n=base_depth,
        depthwise=depthwise,
        act=act,
      ),
    )
    
    # Event Dark3
    self.dark3_event = nn.Sequential(
      Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
      CSPLayer(
        base_channels * 4,
        base_channels * 4,
        n=base_depth * 3,
        depthwise=depthwise,
        act=act,
      ),
    )
    
    # Event Dark4
    self.dark4_event = nn.Sequential(
      Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
      CSPLayer(
        base_channels * 8,
        base_channels * 8,
        n=base_depth * 3,
        depthwise=depthwise,
        act=act,
      ),
    )
    
    # Event Dark5
    self.dark5_event = nn.Sequential(
      Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
      SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
      CSPLayer(
        base_channels * 16,
        base_channels * 16,
        n=base_depth,
        shortcut=False,
        depthwise=depthwise,
        act=act,
      ),
    )

    # Inter-stream collapse is a parameter-free elementwise sum in forward
    # (matches MCFNet's yolo_pafpn_concatadd, which sums in the neck;
    # mathematically identical, kept in the backbone here to avoid
    # changing the neck contract).

    # ============================================================
    # Motion Compensation Module (MCM) — generic pre-backbone event correction.
    # ECM and SMCM are both concrete MCM implementations.
    # ============================================================
    if self.use_mcm:
      _inst = mcm_instance or smcm_instance  # smcm_instance is legacy kwarg
      if _inst is None and ecm_kwargs is not None:
        # Legacy: construct ECM from kwargs
        from smcfnet.model.mcm.ecm import EventCorrectionModule
        _inst = EventCorrectionModule(**ecm_kwargs)
      if _inst is None:
        raise ValueError(
          "use_mcm=True but no mcm_instance (or legacy ecm_kwargs/smcm_instance) provided. "
          "Build the MCM externally via _build_mcm() in build.py."
        )
      self.mcm = _inst

    # Phase 2: Event Dynamic Upsampling Module (EDUM)
    if self.use_edum:
      from smcfnet.model.edum import EventDynamicUpsamplingModule
      self.edum = EventDynamicUpsamplingModule(**(edum_kwargs or {}))

    # Phase 3: Cross-Modal Mamba Fusion (CMM)
    self.cmm_layers = set((cmm_kwargs or {}).get("fusion_layers", [3, 4, 5]))
    if self.use_cmm:
      from smcfnet.model.cmm import CrossModalMambaFusionModule
      cmm_shared_kwargs = {
        "d_state": (cmm_kwargs or {}).get("d_state", 16),
        "d_conv": (cmm_kwargs or {}).get("d_conv", 3),
        "expand": (cmm_kwargs or {}).get("expand", 2.0),
        "dt_rank": (cmm_kwargs or {}).get("dt_rank", "auto"),
        "dropout": (cmm_kwargs or {}).get("dropout", 0.0),
      }
      self.cmm_dark3 = CrossModalMambaFusionModule(channels=base_channels * 4, **cmm_shared_kwargs)
      self.cmm_dark4 = CrossModalMambaFusionModule(channels=base_channels * 8, **cmm_shared_kwargs)
      self.cmm_dark5 = CrossModalMambaFusionModule(channels=base_channels * 16, **cmm_shared_kwargs)

    # Optional per-batch SynOps logging (set True externally, e.g. cfg.training.log_synops)
    self._log_synops = False

  def forward(self, x_rgb, x_event, **kwargs):
    """
    Forward pass through dual-stream backbone.
    
    Args:
      rgb (Tensor): RGB images, shape (B, 3, H, W)
      event (Tensor): Event voxel grids, shape (B, 5, H, W)
      **kwargs: Extra context forwarded to MCM (e.g. ``events``,
        ``events_offsets`` for EventWarpingLoss).
    
    Returns:
      dict: Fused output features at different scales
        {
          'dark3': (B, 256, H/8, W/8),
          'dark4': (B, 512, H/16, W/16),
          'dark5': (B, 1024, H/32, W/32)
        }
    """

    profile_stages = bool(kwargs.pop("profile_stages", False)) and bool(x_rgb.is_cuda)
    stage_timer = CudaStageTimer(enabled=profile_stages)

    # ============================================================
    # Pre-backbone motion compensation (MCM — covers ECM, SMCM, etc.)
    # ============================================================
    mcm_loss = None
    mcm_flow = None
    mcm_input = None
    mcm_synops_report = None
    if self.use_mcm:
      with stage_timer.section("mcm"):
        mcm_input = x_event.detach()  # pre-MCM voxel for viz
        input_dtype = x_event.dtype
        mcm_kwargs = dict(kwargs)
        if stage_timer.enabled:
          mcm_kwargs["_profile_mcm_stages"] = True
        # Run MCM in FP32 to prevent NaN from AMP FP16 precision loss.
        # Flow estimators operate on small values that can underflow in FP16.
        with torch.amp.autocast("cuda", enabled=False):
          x_event_f32 = x_event.float()
          if self._log_synops:
            from smcfnet.utils.synops import SynOpsProfiler
            _profiler = SynOpsProfiler(self.mcm)
            with _profiler:
              x_event_f32, mcm_loss = self.mcm(x_event_f32, **mcm_kwargs)
            mcm_synops_report = _profiler.report()
          else:
            x_event_f32, mcm_loss = self.mcm(x_event_f32, **mcm_kwargs)
        x_event = x_event_f32.to(input_dtype)
        mcm_flow = self.mcm._last_flow
        mcm_output = x_event.detach()  # post-MCM voxel for viz

    if self.use_edum:
      with stage_timer.section("edum"):
        x_event = self.edum(x_event, x_rgb)

    # ============================================================
    # RGB Stream Forward Pass
    # ============================================================
    with stage_timer.section("rgb_backbone"):
      rgb_stem = self.stem_rgb(x_rgb)         # (B, 64, H/2, W/2)
      rgb_dark2 = self.dark2_rgb(rgb_stem)    # (B, 128, H/4, W/4)
      rgb_dark3 = self.dark3_rgb(rgb_dark2)   # (B, 256, H/8, W/8)
      rgb_dark4 = self.dark4_rgb(rgb_dark3)   # (B, 512, H/16, W/16)
      rgb_dark5 = self.dark5_rgb(rgb_dark4)   # (B, 1024, H/32, W/32)

    # ============================================================
    # Event Stream Forward Pass
    # ============================================================
    with stage_timer.section("event_backbone"):
      event_stem = self.stem_event(x_event)       # (B, 64, H/2, W/2)
      event_dark2 = self.dark2_event(event_stem)  # (B, 128, H/4, W/4)
      event_dark3 = self.dark3_event(event_dark2) # (B, 256, H/8, W/8)
      event_dark4 = self.dark4_event(event_dark3) # (B, 512, H/16, W/16)
      event_dark5 = self.dark5_event(event_dark4) # (B, 1024, H/32, W/32)

    # ============================================================
    # Optional Cross-Modal Mamba Fusion (CMM)
    # ============================================================
    if self.use_cmm:
      with stage_timer.section("cmm"):
        if 3 in self.cmm_layers:
          event_dark3, rgb_dark3 = self.cmm_dark3(event_dark3, rgb_dark3)
        if 4 in self.cmm_layers:
          event_dark4, rgb_dark4 = self.cmm_dark4(event_dark4, rgb_dark4)
        if 5 in self.cmm_layers:
          event_dark5, rgb_dark5 = self.cmm_dark5(event_dark5, rgb_dark5)
    
    # ============================================================
    # Feature Fusion (elementwise sum, MCFNet-style)
    # ============================================================
    outputs = {}
    with stage_timer.section("fusion"):
      fused_dark3 = rgb_dark3 + event_dark3
      fused_dark4 = rgb_dark4 + event_dark4
      fused_dark5 = rgb_dark5 + event_dark5

    outputs["dark3"] = fused_dark3
    outputs["dark4"] = fused_dark4
    outputs["dark5"] = fused_dark5

    result = {k: v for k, v in outputs.items() if k in self.out_features}

    # Propagate auxiliary MCM data for the model wrapper to pick up
    if mcm_loss is not None:
      result["_mcm_loss"] = mcm_loss
    if mcm_flow is not None:
      result["_mcm_flow"] = mcm_flow
      result["_mcm_input"] = mcm_input
      result["_mcm_output"] = mcm_output

    # Per-term MCM loss components (for TensorBoard / W&B diagnostics)
    if self.use_mcm and hasattr(self.mcm, "_last_contrast_loss"):
      result["_mcm_contrast"] = self.mcm._last_contrast_loss
      result["_mcm_smooth"] = self.mcm._last_smooth_loss

    if mcm_synops_report is not None:
      result["_mcm_synops"] = mcm_synops_report["total_synops"]
      result["_mcm_spike_rate"] = mcm_synops_report["avg_spike_rate"]

    if stage_timer.enabled:
      profile_events = dict(stage_timer.events)
      if self.use_mcm:
        mcm_profile_events = getattr(self.mcm, "_last_profile_events", None)
        if isinstance(mcm_profile_events, dict):
          profile_events.update(mcm_profile_events)
      result["_profile_events"] = profile_events

    return result


class CSPDarknet(nn.Module):
  """Single-stream CSPDarknet backbone (RGB-only).

  This is the standard YOLOX-style CSPDarknet trunk used to produce the
  feature maps consumed by `YOLOPAFPN`.

  Output contract matches the dual-stream backbone:
    {"dark3": P3/8, "dark4": P4/16, "dark5": P5/32}
  """

  def __init__(
    self,
    dep_mul: float = 1.0,
    wid_mul: float = 1.0,
    out_features=("dark3", "dark4", "dark5"),
    depthwise: bool = False,
    act: str = "silu",
  ):
    super().__init__()
    assert out_features, "please provide output features of Darknet"
    self.out_features = out_features

    Conv = DWConv if depthwise else BaseConv

    base_channels = int(wid_mul * 64)
    base_depth = max(round(dep_mul * 3), 1)

    # Stem
    self.stem = Focus(3, base_channels, ksize=3, act=act)

    # Stages
    self.dark2 = nn.Sequential(
      Conv(base_channels, base_channels * 2, 3, 2, act=act),
      CSPLayer(
        base_channels * 2,
        base_channels * 2,
        n=base_depth,
        depthwise=depthwise,
        act=act,
      ),
    )

    self.dark3 = nn.Sequential(
      Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
      CSPLayer(
        base_channels * 4,
        base_channels * 4,
        n=base_depth * 3,
        depthwise=depthwise,
        act=act,
      ),
    )

    self.dark4 = nn.Sequential(
      Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
      CSPLayer(
        base_channels * 8,
        base_channels * 8,
        n=base_depth * 3,
        depthwise=depthwise,
        act=act,
      ),
    )

    self.dark5 = nn.Sequential(
      Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
      SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
      CSPLayer(
        base_channels * 16,
        base_channels * 16,
        n=base_depth,
        shortcut=False,
        depthwise=depthwise,
        act=act,
      ),
    )

  def forward(self, x: torch.Tensor):
    x = self.stem(x)
    x = self.dark2(x)
    x3 = self.dark3(x)
    x4 = self.dark4(x3)
    x5 = self.dark5(x4)

    outputs = {
      "dark3": x3,
      "dark4": x4,
      "dark5": x5,
    }
    return {k: v for k, v in outputs.items() if k in self.out_features}
