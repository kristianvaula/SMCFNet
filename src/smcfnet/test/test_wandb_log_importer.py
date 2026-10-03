"""Regression tests for the WandB log importer (issue #112).

Issue #112 asked whether non-rank-0 workers could produce duplicate same-step
records via the importer. Verification outcome: NOT a bug. Eval and
"Final metrics:" lines are emitted on rank 0 only (the Trainer eval routine
returns early for non-main processes), the log lines carry no rank tag to filter
on, and within a step the parser suppresses the "Eval - ..." line once a
"Final metrics:" line has recorded that step. These tests lock that behavior.
"""

import tempfile
import unittest
from pathlib import Path

from smcfnet.utils.wandb_log_importer import parse_log_file


def _parse(log_text: str):
    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as fh:
        fh.write(log_text)
        path = Path(fh.name)
    try:
        return parse_log_file(path).records
    finally:
        path.unlink(missing_ok=True)


class TestWandbLogImporterDedup(unittest.TestCase):
    def test_final_metrics_then_eval_yields_single_eval_record(self):
        # Real rank-0 emission order: compute_metrics logs "Final metrics:" first,
        # then _log_validation logs "Eval - ...". The second must be suppressed.
        log = (
            "Epoch [1/2] Iter 100 LR: 0.01 Loss: 1.5\n"
            "Final metrics: mAP=0.3000, mAP50=0.5000, mAP75=0.2000\n"
            "Eval - mAP: 0.3000, mAP50: 0.5000\n"
        )
        records = _parse(log)
        eval_at_100 = [r for r in records if r.source == "eval" and r.step == 100]
        self.assertEqual(len(eval_at_100), 1, "expected exactly one eval record per step")
        self.assertIn("eval/mAP75", eval_at_100[0].metrics)  # the Final-metrics record

    def test_train_and_eval_parsed(self):
        log = (
            "Epoch [1/2] Iter 50 LR: 0.01 Loss: 2.0\n"
            "Final metrics: mAP=0.1000, mAP50=0.2000, mAP75=0.0500\n"
        )
        records = _parse(log)
        sources = {r.source for r in records}
        self.assertIn("train", sources)
        self.assertIn("eval", sources)
        train = next(r for r in records if r.source == "train")
        self.assertEqual(train.step, 50)
        self.assertEqual(train.metrics["train/loss"], 2.0)

    def test_monitor_validation_parsed_as_mon_eval(self):
        # The Trainer logs monitor eval as "Monitor Validation - mAP: ..."; it
        # must classify as mon_eval (not eval, and not be dropped).
        log = (
            "Epoch [1/2] Iter 10 LR: 0.01 Loss: 1.0\n"
            "Monitor Validation - mAP: 0.4000, mAP50: 0.6000\n"
        )
        records = _parse(log)
        mon = [r for r in records if r.source == "mon_eval"]
        self.assertEqual(len(mon), 1)
        self.assertEqual(mon[0].metrics["mon_eval/mAP50"], 0.6)
        self.assertNotIn("eval/mAP50", mon[0].metrics)

    def test_legacy_mon_eval_alias_still_parsed(self):
        log = (
            "Epoch [1/2] Iter 10 LR: 0.01 Loss: 1.0\n"
            "Mon Eval - mAP: 0.4000, mAP50: 0.6000\n"
        )
        records = _parse(log)
        self.assertEqual([r.source for r in records if "eval" in r.source].count("mon_eval"), 1)

    def test_no_duplicate_eval_records_across_steps(self):
        # Two eval points at distinct steps -> two records, one each.
        log = (
            "Epoch [1/2] Iter 100 LR: 0.01 Loss: 1.5\n"
            "Final metrics: mAP=0.30, mAP50=0.50, mAP75=0.20\n"
            "Epoch [2/2] Iter 200 LR: 0.01 Loss: 1.0\n"
            "Final metrics: mAP=0.40, mAP50=0.60, mAP75=0.30\n"
        )
        records = _parse(log)
        eval_steps = sorted(r.step for r in records if r.source == "eval")
        self.assertEqual(eval_steps, [100, 200])


if __name__ == "__main__":
    unittest.main()
