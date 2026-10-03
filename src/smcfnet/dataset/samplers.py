"""Sampling utilities.

This module contains samplers inspired by Detectron2/YOLOX training loops.

Important note (from Detectron2 design)
- A sampler passed to `torch.utils.data.DataLoader` is used only with map-style datasets
  and is executed in the *main* process.
- If the sampling logic is executed inside DataLoader workers (e.g. by wrapping indices
  into an IterableDataset), then the sampler output must additionally be sharded by
  worker id to avoid duplicated samples.
  Detectron2 addresses this with `ToIterableDataset`.

We keep the same invariant: `InfiniteSampler` below shards by distributed rank/world_size
only. Worker-id sharding is intentionally NOT handled here.
"""

from __future__ import annotations

import itertools
import os
import random
from typing import Iterator, Optional

import torch
from torch.utils.data.sampler import Sampler

from smcfnet.core.launch import get_rank, get_world_size


def _shared_random_seed() -> int:
    """Return a seed shared by all distributed ranks.

    If torch.distributed is initialized, broadcast a random seed from rank0.
    Otherwise return a process-local random seed.
    """

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        seed = torch.empty((), dtype=torch.int64)
        if get_rank() == 0:
            # Use python RNG to avoid coupling with torch global generator state.
            seed.fill_(random.randint(0, 2**31 - 1))
        torch.distributed.broadcast(seed, src=0)
        return int(seed.item())

    # Non-distributed: mix pid to reduce accidental collisions.
    return int((random.randint(0, 2**31 - 1) + os.getpid()) % (2**31 - 1))


class InfiniteSampler(Sampler[int]):
    """Yield an infinite stream of indices.

    This matches the Detectron2/YOLOX philosophy: training is iteration-based and consumes
    an infinite stream rather than defining strict epochs.

    Behavior
    - Produces indices: shuffle(range(size)) + shuffle(range(size)) + ... when shuffle=True.
    - Shards indices by distributed rank/world_size so that each rank sees a disjoint slice:
        indices[rank::world_size]

    Worker note
    - This sampler does NOT shard by DataLoader worker id.
      Only use it as the `sampler=` argument of a map-style DataLoader.
      If you later move sampling into workers (IterableDataset), add worker sharding.
    """

    def __init__(
        self,
        size: int,
        *,
        shuffle: bool = True,
        seed: Optional[int] = 0,
    ):
        if not isinstance(size, int):
            raise TypeError(f"InfiniteSampler(size=) expects int, got {type(size)}")
        if size <= 0:
            raise ValueError(f"InfiniteSampler(size=) expects positive size, got {size}")

        self._size = int(size)
        self._shuffle = bool(shuffle)
        if seed is None:
            seed = _shared_random_seed()
        self._seed = int(seed)
        self._epoch = 0

        self._rank = int(get_rank())
        self._world_size = int(get_world_size())

    def set_epoch(self, epoch: int) -> None:
        """Set the logical epoch for deterministic per-epoch reshuffling."""
        self._epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        start = self._rank
        yield from itertools.islice(self._infinite_indices(), start, None, self._world_size)

    def _infinite_indices(self) -> Iterator[int]:
        g = torch.Generator()
        g.manual_seed(self._seed + self._epoch)
        while True:
            if self._shuffle:
                # `.tolist()` is significantly faster than yielding scalar tensors.
                yield from torch.randperm(self._size, generator=g).tolist()
            else:
                yield from torch.arange(self._size).tolist()

    def __len__(self) -> int:
        # Length is not meaningful for infinite samplers, but PyTorch expects it.
        return self._size // max(self._world_size, 1)
