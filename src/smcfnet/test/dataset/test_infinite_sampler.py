import itertools
import unittest

from smcfnet.dataset.samplers import InfiniteSampler


class TestInfiniteSampler(unittest.TestCase):
    def test_non_shuffled_repeats_range(self):
        size = 10
        sampler = InfiniteSampler(size, shuffle=False, seed=123)
        it = iter(sampler)
        first = list(itertools.islice(it, size))
        second = list(itertools.islice(it, size))
        self.assertEqual(first, list(range(size)))
        self.assertEqual(second, list(range(size)))

    def test_shuffled_is_permutation(self):
        size = 17
        sampler = InfiniteSampler(size, shuffle=True, seed=1)
        it = iter(sampler)
        block = list(itertools.islice(it, size))
        self.assertEqual(sorted(block), list(range(size)))

    def test_deterministic_given_seed(self):
        size = 25
        s1 = InfiniteSampler(size, shuffle=True, seed=999)
        s2 = InfiniteSampler(size, shuffle=True, seed=999)
        b1 = list(itertools.islice(iter(s1), size * 2))
        b2 = list(itertools.islice(iter(s2), size * 2))
        self.assertEqual(b1, b2)

    def test_epoch_changes_shuffle_order(self):
        size = 25
        sampler = InfiniteSampler(size, shuffle=True, seed=999)

        sampler.set_epoch(0)
        epoch0 = list(itertools.islice(iter(sampler), size))

        sampler.set_epoch(1)
        epoch1 = list(itertools.islice(iter(sampler), size))

        self.assertEqual(sorted(epoch0), list(range(size)))
        self.assertEqual(sorted(epoch1), list(range(size)))
        self.assertNotEqual(epoch0, epoch1)


if __name__ == "__main__":
    unittest.main()