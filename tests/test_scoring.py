import math

import pytest

from forecaster.evaluation import scoring as s


def test_binary_baseline_score_landmarks():
    assert s.binary_baseline_score(0.5, True) == pytest.approx(0.0)
    assert s.binary_baseline_score(0.25, True) == pytest.approx(-100.0)
    assert s.binary_baseline_score(0.999999, True) == pytest.approx(100.0, abs=0.01)
    assert s.binary_baseline_score(0.25, False) == pytest.approx(100 * (math.log2(0.75) + 1))


def test_binary_relative_score_rewards_better_calls():
    assert s.binary_relative_score(0.7, 0.7, True) == 0.0
    assert s.binary_relative_score(0.9, 0.6, True) > 0
    assert s.binary_relative_score(0.9, 0.6, False) < 0


def test_multiple_choice_scores():
    uniform = {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3}
    assert s.mc_baseline_score(uniform, "B") == pytest.approx(0.0)
    sharp = {"A": 0.8, "B": 0.1, "C": 0.1}
    assert s.mc_relative_score(sharp, uniform, "A") > 0
    with pytest.raises(ValueError):
        s.mc_baseline_score({"A": 1.0}, "A")


def test_numeric_density_of_a_uniform_cdf():
    cdf = [i / 200 for i in range(201)]
    assert s.numeric_log_density(cdf, 0.0, 100.0, 37.0) == pytest.approx(math.log(1 / 100))


def test_numeric_relative_score_prefers_mass_near_the_outcome():
    uniform = [i / 200 for i in range(201)]
    sharp = [0.0] * 75 + [1.0] * 126  # all mass in the bin that contains 37
    assert s.numeric_relative_score(sharp, uniform, 0.0, 100.0, 37.0) > 0
    assert s.numeric_relative_score(sharp, uniform, 0.0, 100.0, 80.0) < 0


def test_summaries_and_significance():
    a = s.summarize([10.0, 12.0, 11.0, 13.0])
    b = s.summarize([1.0, 2.0, 1.5, 2.5])
    assert a.count == 4
    assert a.mean == pytest.approx(11.5)
    assert a.beats(b)
    assert not a.beats(b, noise_floor=20.0)
    assert math.isnan(s.summarize([]).mean)
