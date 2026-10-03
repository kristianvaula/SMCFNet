"""Regression tests for issue #109: EA failure classifier misclassified every
normal crash because ``"inf" in lowered`` matched the substring in ``[INFO]``.

``TrainerBackedEvaluator._classify_failure`` reads a run log and returns a
failure category used by the EA retry logic. The check must use word boundaries
so an ordinary ``[INFO]``-laden crash is classified ``trainer_failed``, not
``non_finite``.

These tests are pure-string / filesystem only (no CUDA, no dataset).
"""

import tempfile
import unittest
from pathlib import Path

from smcfnet.evolution.evaluator import TrainerBackedEvaluator


def _classify(log_text: str) -> str:
    # _classify_failure only uses self for the (static) _looks_like_* helpers,
    # so a __new__ instance avoids the heavy constructor.
    evaluator = TrainerBackedEvaluator.__new__(TrainerBackedEvaluator)
    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as fh:
        fh.write(log_text)
        path = Path(fh.name)
    try:
        return evaluator._classify_failure(path)
    finally:
        path.unlink(missing_ok=True)


class TestClassifyFailure(unittest.TestCase):
    def test_info_lines_are_not_non_finite(self):
        log = (
            "[INFO] epoch 3 starting\n"
            "[INFO] running inference on val\n"
            "RuntimeError: something unrelated broke\n"
        )
        self.assertEqual(_classify(log), "trainer_failed")

    def test_genuine_inf_loss_is_non_finite(self):
        self.assertEqual(_classify("[INFO] step 10\nloss is inf, aborting\n"), "non_finite")

    def test_genuine_nan_is_non_finite(self):
        self.assertEqual(_classify("[INFO] step 10\nNaN detected in gradients\n"), "non_finite")

    def test_non_finite_phrase_is_non_finite(self):
        self.assertEqual(_classify("error: non-finite values in flow\n"), "non_finite")

    def test_oom_takes_precedence(self):
        # An OOM log that also contains an [INFO] line must classify as oom.
        self.assertEqual(
            _classify("[INFO] step 5\nCUDA out of memory. Tried to allocate ...\n"),
            "oom",
        )

    def test_inference_word_alone_is_not_non_finite(self):
        self.assertEqual(_classify("[INFO] inference complete\nexit code 1\n"), "trainer_failed")


if __name__ == "__main__":
    unittest.main()
