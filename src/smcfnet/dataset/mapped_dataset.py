from __future__ import annotations

from typing import Any, Callable

from torch.utils.data import Dataset


class MappedDataset(Dataset):
    """Map-style dataset wrapper that applies a mapper in __getitem__.

    This is the minimal Detectron2-like pattern:
      mapped[i] = mapper(dataset[i])

    The mapper will run inside DataLoader workers when num_workers>0.
    """

    def __init__(self, dataset: Dataset, mapper: Callable[[Any], Any]):
        self._dataset = dataset
        self._mapper = mapper

    @property
    def dataset(self) -> Dataset:
        """Wrapped underlying dataset (read-only accessor)."""
        return self._dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int):
        return self._mapper(self._dataset[idx])
