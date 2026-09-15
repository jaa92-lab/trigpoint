import pytest

from forecaster import aggregate as agg


def test_symmetric_binary_forecasts_pool_to_half():
    assert agg.pool_binary([0.2, 0.8]) == pytest.approx(0.5)


def test_single_forecast_is_unchanged():
    assert agg.pool_binary([0.9]) == pytest.approx(0.9)


def test_weights_pull_toward_the_heavier_forecast():
    assert agg.pool_binary([0.2, 0.8], weights=[3, 1]) < 0.5


def test_log_odds_pooling_of_agreeing_forecasts_beats_linear_average():
    assert agg.pool_binary([0.9, 0.99]) > (0.9 + 0.99) / 2


def test_extremizing_moves_away_from_half():
    assert agg.pool_binary([0.7, 0.7], extremize=1.5) > 0.7


def test_bad_inputs_are_rejected():
    with pytest.raises(ValueError):
        agg.pool_binary([0.5, 0.6], weights=[0, 0])
    with pytest.raises(ValueError):
        agg.pool_binary([0.5], weights=[1, 2])
    with pytest.raises(ValueError):
        agg.pool_binary([])


def test_logit_spread_measures_disagreement():
    assert agg.logit_spread([0.5]) == 0.0
    assert agg.logit_spread([0.1, 0.9]) == pytest.approx(2 * agg.logit(0.9))


def test_shrinking_interpolates_in_log_odds():
    assert agg.shrink_toward(0.3, None, 0.5) == 0.3
    assert agg.shrink_toward(0.3, 0.8, 0.0) == 0.3
    assert agg.shrink_toward(0.3, 0.8, 1.0) == pytest.approx(0.8)
    assert 0.3 < agg.shrink_toward(0.3, 0.8, 0.5) < 0.8


def test_clamping():
    assert agg.clamp_probability(0.001) == 0.02
    assert agg.clamp_probability(0.999) == 0.98
    assert agg.clamp_probability(0.5) == 0.5
    with pytest.raises(ValueError):
        agg.clamp_probability(0.5, floor=0.6, ceiling=0.4)


def test_multiple_choice_pooling_of_identical_forecasts_is_identity():
    dist = {"A": 0.5, "B": 0.3, "C": 0.2}
    assert agg.pool_multiple_choice([dist, dist], ["A", "B", "C"]) == pytest.approx(dist)


def test_multiple_choice_output_respects_floor_and_sums_to_one():
    options = ["A", "B", "C", "D"]
    pooled = agg.pool_multiple_choice(
        [{"A": 0.97, "B": 0.03, "C": 0.0}, {"A": 0.9, "B": 0.1}], options, floor=0.01
    )
    assert set(pooled) == set(options)
    assert sum(pooled.values()) == pytest.approx(1.0, abs=1e-12)
    assert min(pooled.values()) >= 0.01 - 1e-12
    assert pooled["A"] > pooled["B"] > pooled["C"]


def test_apply_floor_handles_cascading_small_options():
    probs = agg.apply_floor({"A": 0.94, "B": 0.03, "C": 0.02, "D": 0.005, "E": 0.005}, floor=0.02)
    assert min(probs.values()) >= 0.02 - 1e-12
    assert sum(probs.values()) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        agg.apply_floor({"A": 0.5, "B": 0.5}, floor=0.5)


def test_pool_quantiles_averages_shared_levels():
    a = {0.1: 10.0, 0.5: 20.0, 0.9: 30.0}
    b = {0.1: 14.0, 0.5: 24.0, 0.9: 40.0, 0.99: 99.0}
    assert agg.pool_quantiles([a, b]) == {0.1: 12.0, 0.5: 22.0, 0.9: 35.0}
    with pytest.raises(ValueError):
        agg.pool_quantiles([{0.5: 1.0}, {0.5: 2.0}])


def test_widening_keeps_the_median_and_stretches_the_tails():
    q = {0.1: 80.0, 0.4: 95.0, 0.6: 105.0, 0.9: 120.0}
    wide = agg.widen_tails(q, factor=1.5)
    assert agg.median_of(wide) == pytest.approx(agg.median_of(q))
    assert wide[0.1] == pytest.approx(70.0)
    assert wide[0.9] == pytest.approx(130.0)
    assert wide[0.4] == pytest.approx(100 - 5 * 1.125)


def test_fit_to_bounds_respects_closed_and_open_bounds():
    q = {0.1: -5.0, 0.5: 50.0, 0.9: 130.0}
    assert agg.fit_to_bounds(q, 0.0, 100.0, open_lower=False, open_upper=False) == {
        0.1: 0.0, 0.5: 50.0, 0.9: 100.0,
    }
    assert agg.fit_to_bounds(q, 0.0, 100.0, open_lower=True, open_upper=True) == {
        0.1: -5.0, 0.5: 50.0, 0.9: 120.0,
    }


def test_enforce_increasing_breaks_ties():
    fixed = agg.enforce_increasing({0.1: 5.0, 0.5: 5.0, 0.9: 4.0})
    values = [fixed[k] for k in sorted(fixed)]
    assert values[0] < values[1] < values[2]
