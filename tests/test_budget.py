from datetime import datetime, timedelta, timezone

import pytest

from forecaster.budget import (
    CONSERVATIVE_PRICE,
    CostLedger,
    ModelPrice,
    TimeBudget,
    estimate_tokens,
    should_skip,
)

NOW = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)


class FakeMonotonic:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def test_budget_leaves_a_safety_margin_before_close():
    budget = TimeBudget.for_question(
        NOW + timedelta(minutes=20),
        NOW,
        max_seconds=1500,
        safety_margin_seconds=300,
        monotonic=FakeMonotonic(),
    )
    assert budget.total_seconds == pytest.approx(900)


def test_budget_is_capped_and_handles_missing_close_time():
    assert TimeBudget.for_question(NOW + timedelta(hours=2), NOW, max_seconds=1500).total_seconds == 1500
    assert TimeBudget.for_question(None, NOW, max_seconds=600).total_seconds == 600


def test_budget_is_zero_when_close_falls_inside_the_margin():
    budget = TimeBudget.for_question(NOW + timedelta(minutes=3), NOW, safety_margin_seconds=300)
    assert budget.total_seconds == 0
    assert budget.expired()


def test_remaining_and_slices_follow_the_clock():
    clock = FakeMonotonic()
    budget = TimeBudget(total_seconds=100, monotonic=clock)
    clock.t += 40
    assert budget.remaining() == pytest.approx(60)
    assert budget.slice(0.5) == pytest.approx(30)
    assert budget.slice(0.01, minimum=10) == pytest.approx(10)
    assert budget.slice(0.9, maximum=20) == pytest.approx(20)
    clock.t += 100
    assert budget.expired()
    assert budget.slice(0.5) == 0.0


def test_should_skip_questions_closing_too_soon():
    assert should_skip(NOW + timedelta(seconds=60), NOW)
    assert not should_skip(NOW + timedelta(minutes=30), NOW)
    assert not should_skip(None, NOW)


def test_cost_ledger_estimates_records_and_caps():
    ledger = CostLedger(cap=1.0, prices={"claude": ModelPrice(5.0, 25.0)})
    assert ledger.estimate("claude", 10_000, 2_000) == pytest.approx(0.1)
    assert ledger.can_afford("claude", 10_000, 2_000)
    ledger.record("analyst", "claude", 10_000, 2_000)
    assert ledger.spent == pytest.approx(0.1)
    assert ledger.remaining == pytest.approx(0.9)
    assert not ledger.can_afford("claude", 200_000, 40_000)


def test_unknown_models_are_priced_conservatively():
    assert CostLedger(cap=1.0, prices={}).price_for("mystery") == CONSERVATIVE_PRICE


def test_token_estimate_is_about_four_characters_per_token():
    assert estimate_tokens("") == 1
    assert estimate_tokens("x" * 400) == 100
