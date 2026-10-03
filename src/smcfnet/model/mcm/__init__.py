"""Motion compensation modules (MCM).

An MCM is a thin compositor:
  estimator input -> flow estimator -> warper/renderer -> corrected events -> loss

ECM/SMCM/etc are all concrete implementations of the generic
`MotionCompensationModule`.  The preferred entry point is
`MotionCompensationModule` configured with a `FlowEstimator` and loss
from the respective registries (see `build.py:_build_mcm`).

Backward-compat aliases:
  EventCorrectionModule        -> ECM-specialised MCM
  SpikingMotionCompensationModule -> alias for MotionCompensationModule
"""

from .base import BaseMotionCompensationModule
from .mcm import EventWarpingModule, MotionCompensationModule
from .ecm import EventCorrectionModule, PEODEventCorrectionModule
from .smcm import SpikingMotionCompensationModule, StreamingMotionCompensationModule
