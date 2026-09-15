import pytest

from forecaster import panel
from forecaster.analysts import AnalystResult
from forecaster.config import BotConfig

CONFIG = BotConfig()
LEVELS = (0.1, 0.2, 0.4, 0.6, 0.8, 0.9)


def ok(name, value, weight=1.0, flags=None):
    return AnalystResult(name, f"openrouter/x/{name}", value, "why", weight, list(flags or []))


def failed(name, error="timed out"):
    return AnalystResult(name, "m", None, "", error=error)


def test_escalation_rules():
    assert not panel.should_escalate([ok("a", 0.30), ok("b", 0.32)], "binary", CONFIG)
    assert panel.should_escalate([ok("a", 0.10), ok("b", 0.60)], "binary", CONFIG)
    assert panel.should_escalate([ok("a", 0.30), failed("b")], "binary", CONFIG)
    assert panel.should_escalate([ok("a", 0.30), ok("b", 0.31, flags=["TOOL_CLAIM: x"])], "binary", CONFIG)
    assert panel.should_escalate([ok("a", {"A": 1.0}), ok("b", {"A": 1.0})], "multiple_choice", CONFIG)


def test_escalation_reasons_are_readable():
    assert panel.escalation_reason([ok("a", 0.1), ok("b", 0.6)], "binary", CONFIG) == "the analysts disagreed"
    assert panel.escalation_reason([ok("a", 0.3), failed("b")], "binary", CONFIG) == "an analyst failed"


def test_combine_binary_pools_anchors_and_clamps():
    final, pooled = panel.combine_binary([ok("a", 0.2), ok("b", 0.8)], prior=None, config=CONFIG)
    assert pooled == pytest.approx(0.5)
    assert final == pytest.approx(0.5)
    anchored, _ = panel.combine_binary([ok("a", 0.5)], prior=0.9, config=CONFIG)
    assert 0.5 < anchored < 0.9
    clamped, _ = panel.combine_binary([ok("a", 0.999)], prior=None, config=CONFIG)
    assert clamped == CONFIG.binary_ceiling


def test_combine_binary_falls_back_to_the_prior_then_raises():
    final, pooled = panel.combine_binary([failed("a")], prior=0.4, config=CONFIG)
    assert final == pytest.approx(0.4)
    assert pooled is None
    with pytest.raises(panel.NoForecastError, match="a: timed out"):
        panel.combine_binary([failed("a")], prior=None, config=CONFIG)


def test_lower_weights_shift_the_pool():
    pooled, _ = panel.combine_binary([ok("a", 0.2, weight=1.0), ok("b", 0.8, weight=0.25)], None, CONFIG)
    assert pooled < 0.5


def test_combine_numeric_blends_the_prior_and_fits_bounds():
    analyst = dict(zip(LEVELS, (40, 45, 50, 55, 60, 65)))
    prior = dict(zip(LEVELS, (60, 65, 70, 75, 80, 85)))
    without = panel.combine_numeric([ok("a", analyst)], None, CONFIG, 0, 100, False, False)
    with_prior = panel.combine_numeric([ok("a", analyst)], prior, CONFIG, 0, 100, False, False)
    assert with_prior[0.4] > without[0.4]
    values = [with_prior[level] for level in LEVELS]
    assert values == sorted(values)
    assert values[0] >= 0 and values[-1] <= 100


def test_combine_numeric_uses_the_prior_alone_when_every_analyst_fails():
    prior = dict(zip(LEVELS, (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)))
    result = panel.combine_numeric([failed("a")], prior, CONFIG, 0, 10, True, True)
    assert result[0.1] < result[0.9]
    with pytest.raises(panel.NoForecastError):
        panel.combine_numeric([failed("a")], None, CONFIG, 0, 10, True, True)


def test_combine_multiple_choice_respects_the_floor():
    result = panel.combine_multiple_choice(
        [ok("a", {"X": 0.999, "Y": 0.001}), ok("b", {"X": 0.95, "Y": 0.05})], ["X", "Y"], CONFIG
    )
    assert result["Y"] >= CONFIG.mc_floor
    assert sum(result.values()) == pytest.approx(1.0)


def test_report_lists_every_analyst_with_flags_and_errors():
    text = panel.render_panel_report(
        kind="event",
        question_type="binary",
        stage="Full panel: escalated because the analysts disagreed.",
        results=[ok("news", 0.3, 0.7, ["TOOL_CLAIM: searched"]), failed("data")],
        final_text="31.0%",
        anchor_text=None,
        notes=["Markets were skipped."],
        spend=0.12,
    )
    assert "Final forecast: 31.0%" in text
    assert "### news (news): 30.0% (weight 0.70)" in text
    assert "- Flag: TOOL_CLAIM: searched" in text
    assert "### data (m): no forecast (timed out)" in text
    assert "Markets were skipped." in text
    assert "Estimated analyst spend: $0.12" in text
