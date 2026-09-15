from datetime import datetime, timedelta, timezone

import pytest

from forecaster.clock import Clock


def test_live_clock_tracks_real_time():
    clock = Clock.live()
    assert clock.is_live
    assert abs((clock.now() - datetime.now(timezone.utc)).total_seconds()) < 5


def test_pinned_clock_is_frozen_and_utc():
    moment = datetime(2026, 9, 7, 9, 30, tzinfo=timezone(timedelta(hours=-5)))
    clock = Clock.pinned(moment)
    assert not clock.is_live
    assert clock.now() == moment
    assert clock.now().utcoffset() == timedelta(0)


def test_pinned_clock_rejects_naive_datetimes():
    with pytest.raises(ValueError):
        Clock.pinned(datetime(2026, 9, 7))
