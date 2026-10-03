import unittest

from smcfnet.dataset.mapped_dataset import MappedDataset


class _ListDataset:
    def __init__(self, items):
        self._items = list(items)

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        return self._items[idx]


class TestMappedDataset(unittest.TestCase):
    def test_len_delegates_to_underlying_dataset(self):
        base = _ListDataset([1, 2, 3])
        wrapped = MappedDataset(base, mapper=lambda x: x)
        self.assertEqual(len(wrapped), 3)

    def test_getitem_applies_mapper(self):
        base = _ListDataset([1, 2, 3])
        wrapped = MappedDataset(base, mapper=lambda x: {"value": x, "double": x * 2})
        self.assertEqual(wrapped[1], {"value": 2, "double": 4})


if __name__ == "__main__":
    unittest.main()