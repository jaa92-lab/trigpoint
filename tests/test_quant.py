import math

import pytest

from forecaster.quant import (
    lognormal_quantiles,
    prob_above_at,
    prob_touch_above,
    prob_touch_below,
    prob_touch_in_window,
    realized_sigma_per_day,
)

DAY = 86400.0


def test_at_the_money_is_just_under_half():
    assert 0.45 < prob_above_at(spot=100, strike=100, sigma_day=0.03, days=4) < 0.5


def test_far_strikes_are_near_certain():
    assert prob_above_at(100, 50, 0.02, 2) > 0.999
    assert prob_above_at(100, 200, 0.02, 2) < 0.001


def test_zero_horizon_is_deterministic():
    assert prob_above_at(101, 100, 0.05, 0) == 1.0
    assert prob_above_at(99, 100, 0.05, 0) == 0.0


def test_touching_is_at_least_as_likely_as_finishing_above():
    assert prob_touch_above(1000, 1100, 0.04, 3) >= prob_above_at(1000, 1100, 0.04, 3)
    assert prob_touch_above(1200, 1100, 0.04, 3) == 1.0
    assert prob_touch_below(1000, 900, 0.04, 3) == pytest.approx(
        prob_touch_above(1000, 1000 * 1000 / 900, 0.04, 3)
    )


def test_window_touch_starting_now_matches_plain_touch():
    assert prob_touch_in_window(1000, 1100, 0.04, 0.0, 3.0) == pytest.approx(prob_touch_above(1000, 1100, 0.04, 3.0))


def test_being_past_the_strike_before_the_window_opens_is_not_certainty():
    p = prob_touch_in_window(1163, 1100, 0.064, 2.0, 3.0, tail_factor=1.25)
    assert 0.6 < p < 0.999
    assert p > prob_above_at(1163, 1100, 0.064, 2.0, 1.25)


def test_window_probability_for_below_and_degenerate_cases():
    assert 0.0 < prob_touch_in_window(1000, 900, 0.04, 3.0, 4.0, direction="below") < 1.0
    assert prob_touch_in_window(1000, 900, 0.04, 3.0, 0.0, direction="below") == 0.0
    assert prob_touch_in_window(800, 900, 0.04, 3.0, 0.0, direction="below") == 1.0


def test_lognormal_quantiles_are_increasing_with_driftless_median():
    q = lognormal_quantiles(spot=3.0, sigma_day=0.02, days=4, levels=[0.1, 0.5, 0.9])
    assert q[0.1] < q[0.5] < q[0.9]
    s = 0.02 * math.sqrt(4)
    assert q[0.5] == pytest.approx(3.0 * math.exp(-0.5 * s * s))


def test_tail_factor_widens_the_distribution():
    narrow = lognormal_quantiles(100, 0.02, 4, [0.1, 0.9])
    wide = lognormal_quantiles(100, 0.02, 4, [0.1, 0.9], tail_factor=1.5)
    assert wide[0.9] - wide[0.1] > narrow[0.9] - narrow[0.1]


def _alternating_series(move: float, step_seconds: float, count: int = 20):
    price, points = 100.0, [(0.0, 100.0)]
    for i in range(1, count + 1):
        price *= math.exp(move if i % 2 else -move)
        points.append((i * step_seconds, price))
    return points


def test_realized_sigma_recovers_daily_moves():
    assert realized_sigma_per_day(_alternating_series(0.03, DAY)) == pytest.approx(0.03, rel=0.1)


def test_realized_sigma_scales_hourly_moves_to_daily():
    sigma = realized_sigma_per_day(_alternating_series(0.01, DAY / 24))
    assert sigma == pytest.approx(0.01 * math.sqrt(24), rel=0.1)


def test_realized_sigma_needs_enough_points():
    assert realized_sigma_per_day([(0, 1.0), (DAY, 1.1), (2 * DAY, 1.0)]) is None
