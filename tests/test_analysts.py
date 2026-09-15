import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from forecaster.analysts import (
    QuestionView,
    build_prompt,
    parse_binary,
    parse_date_percentiles,
    parse_multiple_choice,
    parse_percentiles,
    run_analyst,
)
from forecaster.budget import CostLedger, ModelPrice
from forecaster.config import ANALYSTS

NOW = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)
SPEC = ANALYSTS[0]


def storm_question() -> QuestionView:
    return QuestionView(
        title="Will a new named storm form in the Atlantic basin between September 17 and September 18, 2026?",
        question_type="binary",
        resolution_criteria="Resolves Yes if the NHC names a new storm in that window.",
        close_time=NOW + timedelta(hours=1),
        resolve_time=NOW + timedelta(days=4),
    )


def test_prompt_contains_role_criteria_timing_evidence_and_format():
    prompt = build_prompt(SPEC, storm_question(), "## News coverage\nQuiet tropics.", NOW)
    assert SPEC.brief in prompt
    assert "Resolves Yes if the NHC names a new storm in that window." in prompt
    assert "automated reader" in prompt
    assert "Today is 2026-09-14 12:00 UTC." in prompt
    assert "about 4.0 days from now" in prompt
    assert "Quiet tropics." in prompt
    assert prompt.rstrip().endswith("Probability: NN%")


def test_multiple_choice_and_numeric_formats():
    mc = QuestionView(title="Who wins?", question_type="multiple_choice", options=("Alice", "Bob"))
    mc_prompt = build_prompt(SPEC, mc, "none", NOW)
    assert "Options: Alice; Bob" in mc_prompt
    assert mc_prompt.rstrip().endswith("Bob: NN%")

    numeric = QuestionView(
        title="How many points?",
        question_type="numeric",
        unit="points",
        lower_bound=0,
        upper_bound=100,
        open_lower_bound=False,
        open_upper_bound=True,
    )
    numeric_prompt = build_prompt(SPEC, numeric, "none", NOW)
    assert numeric_prompt.rstrip().endswith("Percentile 90: X")
    assert "The outcome cannot be lower than 0 points, and it is likely not higher than 100 points." in numeric_prompt


def test_parse_binary_takes_the_last_probability_line():
    assert parse_binary("First guess Probability: 80%\nthen\nProbability: 35%") == pytest.approx(0.35)
    assert parse_binary("Probability: 7.5 %") == pytest.approx(0.075)
    assert parse_binary("no number here") is None
    assert parse_binary("Probability: 150%") is None


def test_parse_multiple_choice_normalizes_and_handles_prefix_options():
    text = "Reasoning.\n- 1: 20%\n- 10: 50%\n**100**: 30%"
    assert parse_multiple_choice(text, ["1", "10", "100"]) == pytest.approx({"1": 0.2, "10": 0.5, "100": 0.3})
    assert parse_multiple_choice("1: 20%", ["1", "10"]) is None
    assert parse_multiple_choice("A: 10%\nB: 10%", ["A", "B"]) is None


def test_parse_percentiles_requires_every_level_in_order():
    values = ("1,000", "1,200", "1,500", "1,700", "2,000", "2,400")
    text = "\n".join(f"Percentile {label}: {value}" for label, value in zip((10, 20, 40, 60, 80, 90), values))
    assert parse_percentiles(text) == {0.1: 1000.0, 0.2: 1200.0, 0.4: 1500.0, 0.6: 1700.0, 0.8: 2000.0, 0.9: 2400.0}
    assert parse_percentiles("Percentile 10: 5\nPercentile 90: 9") is None
    assert parse_percentiles(text.replace("Percentile 90: 2,400", "Percentile 90: 100")) is None


async def test_run_analyst_parses_and_records_cost():
    async def call(model, prompt):
        return "Rationale.\nProbability: 30%"

    ledger = CostLedger(cap=1.0, prices={SPEC.model: ModelPrice(5.0, 25.0)})
    result = await run_analyst(SPEC, "prompt text", call, parse_binary, timeout=5, ledger=ledger)
    assert result.ok
    assert result.value == pytest.approx(0.3)
    assert ledger.spent > 0


async def test_run_analyst_uses_the_fallback_parser():
    async def call(model, prompt):
        return "I lean toward about a third."

    async def fallback(text):
        return 0.33

    result = await run_analyst(SPEC, "p", call, parse_binary, timeout=5, fallback=fallback)
    assert result.value == 0.33


async def test_run_analyst_reports_failures_without_raising():
    async def slow(model, prompt):
        await asyncio.sleep(1)
        return "Probability: 50%"

    timed_out = await run_analyst(SPEC, "p", slow, parse_binary, timeout=0.05)
    assert not timed_out.ok and "timed out" in timed_out.error

    async def broken(model, prompt):
        raise RuntimeError("rate limited")

    failed = await run_analyst(SPEC, "p", broken, parse_binary, timeout=5)
    assert not failed.ok and "rate limited" in failed.error

    async def unparseable(model, prompt):
        return "no idea"

    unparsed = await run_analyst(SPEC, "p", unparseable, parse_binary, timeout=5)
    assert not unparsed.ok and unparsed.rationale == "no idea"


def test_date_prompt_asks_for_dates_within_the_bounds():
    question = QuestionView(
        title="When will the report be published?",
        question_type="date",
        lower_bound=datetime(2026, 9, 20, tzinfo=timezone.utc).timestamp(),
        upper_bound=datetime(2026, 12, 31, tzinfo=timezone.utc).timestamp(),
        open_lower_bound=False,
        open_upper_bound=True,
    )
    prompt = build_prompt(SPEC, question, "none", NOW)
    assert "The outcome cannot be earlier than 2026-09-20, and it is likely not later than 2026-12-31." in prompt
    assert prompt.rstrip().endswith("Percentile 90: YYYY-MM-DD")


def test_parse_date_percentiles_reads_dates_and_times_as_utc_timestamps():
    text = (
        "Reasoning first.\nPercentile 10: 2026-10-01\nPercentile 20: 2026-10-05\n"
        "Percentile 40: 2026-10-10T12:00Z\nPercentile 60: 2026-10-15\nPercentile 80: 2026-10-22\n"
        "Percentile 90: 2026-10-30"
    )
    parsed = parse_date_percentiles(text)
    assert parsed[0.1] == datetime(2026, 10, 1, tzinfo=timezone.utc).timestamp()
    assert parsed[0.4] == datetime(2026, 10, 10, 12, tzinfo=timezone.utc).timestamp()
    assert parse_date_percentiles(text.replace("2026-10-30", "2026-09-01")) is None
    assert parse_date_percentiles("Percentile 10: 2026-10-01") is None
