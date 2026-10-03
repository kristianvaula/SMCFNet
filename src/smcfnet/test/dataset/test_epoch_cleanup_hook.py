import unittest
from types import SimpleNamespace

from smcfnet.core.hooks import EpochCleanupHook


class _MapperWithReset:
  def __init__(self):
    self.calls = 0

  def reset_caches(self):
    self.calls += 1


class _MappedDatasetLike:
  def __init__(self, dataset, mapper):
    self._dataset = dataset
    self._mapper = mapper


class _DatasetWithPublicMapper:
  def __init__(self, mapper):
    self.mapper = mapper


class TestEpochCleanupHook(unittest.TestCase):
  def test_reset_mapper_cache_finds_private_mapper_on_wrapped_dataset(self):
    mapper = _MapperWithReset()
    wrapped = _MappedDatasetLike(dataset=object(), mapper=mapper)
    trainer = SimpleNamespace(train_loader=SimpleNamespace(dataset=wrapped), storage=None)

    hook = EpochCleanupHook(clear_track_cache=False, clear_latest_scalars=False)
    hook.trainer = trainer
    hook.after_epoch()

    self.assertEqual(mapper.calls, 1)

  def test_reset_mapper_cache_finds_public_mapper(self):
    mapper = _MapperWithReset()
    dataset = _DatasetWithPublicMapper(mapper=mapper)
    trainer = SimpleNamespace(train_loader=SimpleNamespace(dataset=dataset), storage=None)

    hook = EpochCleanupHook(clear_track_cache=False, clear_latest_scalars=False)
    hook.trainer = trainer
    hook.after_epoch()

    self.assertEqual(mapper.calls, 1)


if __name__ == "__main__":
  unittest.main()