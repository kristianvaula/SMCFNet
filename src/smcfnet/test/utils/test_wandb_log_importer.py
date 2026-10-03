from __future__ import annotations

from pathlib import Path

from smcfnet.utils.wandb_log_importer import parse_log_file, summarize_records


def test_parse_detector_training_eval_and_epoch(tmp_path: Path) -> None:
    log = tmp_path / "train.log"
    log.write_text(
        "\n".join(
            [
                "[2026-05-17 02:11:51,279] [rank=0] [smcfnet.core.trainer] [INFO] "
                "Epoch [0/300] Iter 0 LR: 0.000010 Loss: 8.8553 Obj: 3.8015 "
                "Cls: 2.7815 Box: 2.2723 FG: 6.2744 IoU: 0.4545 Data: 13.7955s GPU: 1130/29528MiB",
                "[2026-05-17 02:25:35,293] [rank=0] [smcfnet.core.trainer] [INFO] "
                "Epoch [0/300] completed in 849.58s Avg Loss: 2.5212",
                "[2026-05-17 09:14:22,345] [rank=0] [smcfnet.utils.validation] [INFO] "
                "Final metrics: mAP=0.3886, mAP50=0.5722, mAP75=0.4419",
            ]
        ),
        encoding="utf-8",
    )

    parsed = parse_log_file(log)

    assert len(parsed.records) == 3
    assert parsed.records[0].step == 0
    assert parsed.records[0].metrics["train/loss"] == 8.8553
    assert parsed.records[0].metrics["train/loss_obj"] == 3.8015
    assert parsed.records[0].metrics["gpu/mem_reserved_mib"] == 29528.0
    assert parsed.records[1].metrics["epoch/total"] == 2.5212
    assert parsed.records[2].metrics["eval/mAP75"] == 0.4419


def test_parse_mcm_training_validation_and_epoch(tmp_path: Path) -> None:
    log = tmp_path / "train.log"
    log.write_text(
        "\n".join(
            [
                "[2026-05-18] [rank=0] [INFO] [MCM] iter=50  total=1.2500  mcm_epe=0.7500  lr=0.0001",
                "[2026-05-18] [rank=0] [INFO] [MCM][VAL] epoch=2 iter=50  total=1.1000  mcm_epe=0.7000",
                "[2026-05-18] [rank=0] [INFO] [MCM] Epoch 2 done in 12.5s  total=1.2000",
            ]
        ),
        encoding="utf-8",
    )

    parsed = parse_log_file(log)
    summary = summarize_records(parsed.records)

    assert summary["records"] == 3
    assert summary["last_step"] == 50
    assert parsed.records[0].metrics["mcm_train/total"] == 1.25
    assert parsed.records[0].metrics["mcm_train/lr"] == 0.0001
    assert parsed.records[1].metrics["mcm_val/mcm_epe"] == 0.7
    assert parsed.records[2].metrics["mcm_epoch/time_sec"] == 12.5
