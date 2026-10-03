from __future__ import annotations

from smcfnet.dataset.dsec.dsec_dataset import DSECDataset


class PEODDataset(DSECDataset):
    """Concat wrapper for PEOD sequence-index datasets.

    The inherited implementation is dataset-agnostic despite its historical
    DSEC name. Keeping this thin subclass gives PEOD a clear public type while
    avoiding a broad rename of the existing loader stack.
    """

