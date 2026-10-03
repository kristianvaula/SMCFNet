import pytest

from smcfnet.model.losses import (
    ScheduledEventWarpingSupervisedLoss,
    create_mc_loss,
)


def test_scheduled_event_warping_supervised_loss_default_schedule():
    loss = ScheduledEventWarpingSupervisedLoss(
        schedule_total_epochs=30,
        self_supervised_only_epochs=3,
    )

    assert loss.schedule_weights(epoch=0) == pytest.approx((0.0, 1.0))
    assert loss.schedule_weights(epoch=27) == pytest.approx((1.0, 0.0))
    assert loss.schedule_weights(epoch=29) == pytest.approx((1.0, 0.0))

    ew_weight, sup_weight = loss.schedule_weights(epoch=14)
    assert ew_weight == pytest.approx(14.0 / 27.0)
    assert sup_weight == pytest.approx(13.0 / 27.0)


def test_scheduled_event_warping_supervised_loss_factory_key():
    loss = create_mc_loss(
        "event_warping_supervised_scheduled",
        schedule_total_epochs=30,
        self_supervised_only_epochs=3,
    )

    assert isinstance(loss, ScheduledEventWarpingSupervisedLoss)

    alias_loss = create_mc_loss("scheduled_event_warping_supervised")
    assert isinstance(alias_loss, ScheduledEventWarpingSupervisedLoss)
