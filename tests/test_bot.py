"""End-to-end runs of the bot with scripted models and no network."""

from datetime import datetime, timedelta, timezone

import httpx
import pytest
from forecasting_tools import (
    BinaryQuestion,
    DateQuestion,
    MetaculusClient,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    PredictedOptionList,
)

from forecaster.bot import PanelForecaster
from forecaster.clock import Clock
from forecaster.config import ANALYSTS, BotConfig
from forecaster.evidence.base import EvidenceItem

NOW = datetime.now(timezone.utc)
ALL_ANALYSTS = ("news", "outside_view", "data", "sources")


def offline_client() -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if "manifold" in request.url.host:
            return httpx.Response(200, json=[])
        if "polymarket" in request.url.host:
            return httpx.Response(200, json={"events": []})
        return httpx.Response(404)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


class FakeNews:
    available = True

    def __init__(self) -> None:
        self.queries = []

    async def search(self, query, clock):
        self.queries.append((query, clock.now()))
        return [EvidenceItem("news", "Officials quiet", "No announcement yet.", "https://news.example/a", clock.now())]


class ScriptedModels:
    """Answers each analyst by recognizing its role brief in the prompt."""

    def __init__(self, answers) -> None:
        self.answers = answers
        self.calls = []
        self.prompts = {}

    async def __call__(self, model, prompt):
        name = next(spec.name for spec in ANALYSTS if spec.brief in prompt)
        self.calls.append(name)
        self.prompts[name] = prompt
        answer = self.answers[name]
        if isinstance(answer, list):
            answer = answer.pop(0) if len(answer) > 1 else answer[0]
        if isinstance(answer, Exception):
            raise answer
        return answer


def make_bot(models, clock_policy=None, config=None, http_client_factory=offline_client):
    return PanelForecaster(
        config=config or BotConfig(),
        clock_policy=clock_policy,
        llm_call=models,
        news=FakeNews(),
        http_client_factory=http_client_factory,
        publish_reports_to_metaculus=False,
        metaculus_client=MetaculusClient(token="test-token"),
    )


def event_question(**overrides) -> BinaryQuestion:
    fields = dict(
        question_text="Will Trump have formally nominated a permanent Secretary of the Army by September 18, 2026?",
        id_of_question=101,
        id_of_post=101,
        page_url="https://www.metaculus.com/questions/101/",
        resolution_criteria="Resolves Yes if a nomination is formally sent to the Senate.",
        close_time=NOW + timedelta(hours=1),
        scheduled_resolution_time=NOW + timedelta(days=4),
        open_time=NOW - timedelta(minutes=10),
    )
    fields.update(overrides)
    return BinaryQuestion(**fields)


async def test_agreeing_stage_one_analysts_answer_twice_without_escalation():
    models = ScriptedModels({"news": "Probability: 30%", "outside_view": "Probability: 32%"})
    report = await make_bot(models).forecast_question(event_question())
    assert sorted(models.calls) == ["news", "news", "outside_view", "outside_view"]
    assert 0.30 <= report.prediction <= 0.32
    assert "asked twice" in report.explanation
    assert "### news, answer 2" in report.explanation


async def test_second_answers_that_disagree_still_escalate():
    models = ScriptedModels(
        {
            "news": ["Probability: 30%", "Probability: 75%"],
            "outside_view": "Probability: 32%",
            "data": "Probability: 40%",
            "sources": "Probability: 45%",
        }
    )
    report = await make_bot(models).forecast_question(event_question())
    assert sorted(models.calls) == ["data", "news", "news", "outside_view", "outside_view", "sources"]
    assert "after a second answer from each" in report.explanation


async def test_second_answers_can_be_switched_off():
    models = ScriptedModels({"news": "Probability: 30%", "outside_view": "Probability: 32%"})
    report = await make_bot(models, config=BotConfig(min_answers=0)).forecast_question(event_question())
    assert sorted(models.calls) == ["news", "outside_view"]
    assert "Stage one only" in report.explanation


async def test_disagreement_escalates_to_the_full_panel():
    models = ScriptedModels(
        {"news": "Probability: 10%", "outside_view": "Probability: 60%", "data": "Probability: 30%", "sources": "Probability: 25%"}
    )
    report = await make_bot(models).forecast_question(event_question())
    assert sorted(models.calls) == sorted(ALL_ANALYSTS)
    assert "Full panel" in report.explanation
    assert 0.1 < report.prediction < 0.6


async def test_a_failed_analyst_is_reported_and_the_rest_still_forecast():
    models = ScriptedModels(
        {"news": RuntimeError("rate limited"), "outside_view": "Probability: 20%", "data": "Probability: 22%", "sources": "Probability: 18%"}
    )
    report = await make_bot(models).forecast_question(event_question())
    assert "model call failed: rate limited" in report.explanation
    assert 0.15 < report.prediction < 0.25


async def test_every_analyst_failing_is_an_error_not_a_forecast():
    models = ScriptedModels({name: RuntimeError("provider down") for name in ALL_ANALYSTS})
    result = await make_bot(models).forecast_question(event_question(), return_exceptions=True)
    assert isinstance(result, BaseException)


async def test_fact_check_flags_tool_claims():
    models = ScriptedModels(
        {
            "news": "I searched the web and found a nominee.\nProbability: 90%",
            "outside_view": "Probability: 20%",
            "data": "Probability: 20%",
            "sources": "Probability: 20%",
        }
    )
    report = await make_bot(models).forecast_question(event_question())
    assert "TOOL_CLAIM" in report.explanation


async def test_a_pinned_clock_reaches_prompts_and_evidence():
    pinned = datetime(2026, 9, 7, 9, 0, tzinfo=timezone.utc)
    models = ScriptedModels({"news": "Probability: 30%", "outside_view": "Probability: 31%"})
    bot = make_bot(models, clock_policy=lambda question: Clock.pinned(pinned))
    question = event_question(
        close_time=datetime(2026, 9, 7, 10, 30, tzinfo=timezone.utc),
        scheduled_resolution_time=datetime(2026, 9, 18, tzinfo=timezone.utc),
    )
    report = await bot.forecast_question(question)
    assert "Today is 2026-09-07 09:00 UTC." in models.prompts["news"]
    assert bot._news.queries[0][1] == pinned
    assert "not available for pastcasting" in report.explanation


async def test_questions_closing_too_soon_are_skipped_before_any_model_call():
    models = ScriptedModels({})
    result = await make_bot(models).forecast_question(
        event_question(close_time=NOW + timedelta(seconds=30)), return_exceptions=True
    )
    assert isinstance(result, BaseException)
    assert models.calls == []


async def test_multiple_choice_panel_returns_a_valid_option_list():
    options = ["Alice", "Bob", "Neither"]
    models = ScriptedModels({name: "Alice: 60%\nBob: 35%\nNeither: 5%" for name in ALL_ANALYSTS})
    question = MultipleChoiceQuestion(
        question_text="Who will be nominated?",
        options=options,
        id_of_question=102,
        id_of_post=102,
        page_url="https://www.metaculus.com/questions/102/",
        close_time=NOW + timedelta(hours=1),
        scheduled_resolution_time=NOW + timedelta(days=3),
    )
    report = await make_bot(models).forecast_question(question)
    assert isinstance(report.prediction, PredictedOptionList)
    probabilities = {o.option_name: o.probability for o in report.prediction.predicted_options}
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert probabilities["Alice"] > probabilities["Bob"] > probabilities["Neither"]
    assert len(models.calls) == 4


async def test_numeric_panel_returns_a_valid_distribution():
    text = "\n".join(
        f"Percentile {label}: {value}" for label, value in zip((10, 20, 40, 60, 80, 90), (38, 42, 47, 51, 55, 60))
    )
    models = ScriptedModels({name: text for name in ALL_ANALYSTS})
    question = NumericQuestion(
        question_text="How many combined points will be scored in the Detroit Lions at Buffalo Bills game on September 17, 2026?",
        id_of_question=103,
        id_of_post=103,
        page_url="https://www.metaculus.com/questions/103/",
        close_time=NOW + timedelta(hours=1),
        scheduled_resolution_time=NOW + timedelta(days=3),
        lower_bound=0.0,
        upper_bound=100.0,
        open_lower_bound=False,
        open_upper_bound=True,
    )
    report = await make_bot(models).forecast_question(question)
    assert isinstance(report.prediction, NumericDistribution)
    cdf = [p.percentile for p in report.prediction.get_cdf()]
    assert len(cdf) == 201
    assert cdf == sorted(cdf)


def days_from_now(days: int) -> datetime:
    return NOW + timedelta(days=days)


def title_date(moment: datetime) -> str:
    return f"{moment:%B} {moment.day}, {moment.year}"


async def test_date_panel_returns_a_date_distribution():
    days = (10, 14, 20, 26, 34, 45)
    text = "\n".join(
        f"Percentile {label}: {days_from_now(d):%Y-%m-%d}" for label, d in zip((10, 20, 40, 60, 80, 90), days)
    )
    models = ScriptedModels({name: text for name in ALL_ANALYSTS})
    question = DateQuestion(
        question_text="When will the Senate confirm the next Secretary of the Army?",
        id_of_question=104,
        id_of_post=104,
        page_url="https://www.metaculus.com/questions/104/",
        close_time=NOW + timedelta(hours=1),
        scheduled_resolution_time=days_from_now(90),
        lower_bound=days_from_now(1),
        upper_bound=days_from_now(90),
        open_lower_bound=False,
        open_upper_bound=True,
    )
    report = await make_bot(models).forecast_question(question)
    assert isinstance(report.prediction, NumericDistribution)
    assert report.prediction.is_date
    cdf = [p.percentile for p in report.prediction.get_cdf()]
    assert len(cdf) == 201
    assert cdf == sorted(cdf)
    assert "Final forecast: P10 20" in report.explanation
    assert "Percentile 90: YYYY-MM-DD" in models.prompts["news"]


def weather_client() -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "geocoding-api.open-meteo.com":
            denver = {"name": "Denver", "latitude": 39.74, "longitude": -104.98, "country": "United States", "admin1": "Colorado"}
            return httpx.Response(200, json={"results": [denver]})
        if request.url.host == "api.open-meteo.com":
            day = request.url.params["start_date"]
            daily = {"time": [day], "temperature_2m_max": [93.5]}
            return httpx.Response(200, json={"timezone": "America/Denver", "daily_units": {"temperature_2m_max": "°F"}, "daily": daily})
        return httpx.Response(404)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def denver_heat_question() -> BinaryQuestion:
    return event_question(
        question_text=f"Will the high temperature in Denver, Colorado exceed 90°F on {title_date(days_from_now(2))}?",
        resolution_criteria="Resolves Yes if the National Weather Service reports a high above 90°F at Denver International Airport.",
    )


async def test_weather_questions_lead_with_the_forecast_for_the_data_analyst():
    models = ScriptedModels({"data": "Probability: 70%", "sources": "Probability: 66%"})
    await make_bot(models, http_client_factory=weather_client).forecast_question(denver_heat_question())
    assert sorted(models.calls) == ["data", "data", "sources", "sources"]
    assert "## Weather model forecasts" in models.prompts["data"]
    assert "Denver, Colorado, United States" in models.prompts["data"]
    assert "high 93.5°F" in models.prompts["data"]
    assert "## Weather model forecasts" not in models.prompts["sources"]


async def test_weather_questions_without_a_forecast_lead_with_sources_and_news():
    models = ScriptedModels({"sources": "Probability: 66%", "news": "Probability: 64%"})
    report = await make_bot(models).forecast_question(denver_heat_question())
    assert sorted(models.calls) == ["news", "news", "sources", "sources"]
    assert "weather geocoder" in report.explanation
