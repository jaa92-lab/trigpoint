from datetime import datetime, timedelta, timezone

import forecasting_tools
import httpx
import pytest
from forecasting_tools import (
    BinaryQuestion,
    DateQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    Percentile,
    PredictedOption,
    PredictedOptionList,
)

from forecaster.config import ANALYSTS, BotConfig
from forecaster.evaluation import dataset as dataset_module
from forecaster.evaluation import pastcast as pc
from forecaster.evaluation import scoring
from forecaster.evaluation.cli import main as cli_main
from forecaster.evaluation.dataset import FetchResult, fetch_resolved, is_scoreable, load_questions, save_questions
from forecaster.evidence.base import EvidenceItem

OPEN = datetime(2026, 9, 7, 9, 0, tzinfo=timezone.utc)
CLOSE = OPEN + timedelta(minutes=90)
ALL_ANALYSTS = ("news", "outside_view", "data", "sources")


def offline_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(404)))


class FakeNews:
    available = True

    async def search(self, query, clock, broad=True):
        return [EvidenceItem("news", "Quiet week", "Nothing new.", "https://news.example/q", clock.now())]


class ScriptedModels:
    def __init__(self, answers) -> None:
        self.answers = answers

    async def __call__(self, model, prompt):
        name = next(spec.name for spec in ANALYSTS if spec.brief in prompt)
        answer = self.answers[name]
        if isinstance(answer, Exception):
            raise answer
        return answer


def community_history(**entry) -> dict:
    return {"question": {"aggregations": {"recency_weighted": {"history": [{"start_time": OPEN.timestamp(), "end_time": None, **entry}]}}}}


def resolved_binary(resolution: str = "no") -> BinaryQuestion:
    return BinaryQuestion(
        question_text="Will Trump have formally nominated a permanent Secretary of the Army by September 18, 2026?",
        id_of_question=201,
        id_of_post=201,
        page_url="https://www.metaculus.com/questions/201/",
        resolution_criteria="Resolves Yes if a nomination is formally sent to the Senate.",
        open_time=OPEN,
        close_time=CLOSE,
        cp_reveal_time=CLOSE,
        scheduled_resolution_time=datetime(2026, 9, 18, tzinfo=timezone.utc),
        resolution_string=resolution,
        api_json=community_history(centers=[0.3], forecast_values=[0.7, 0.3]),
    )


def resolved_mc() -> MultipleChoiceQuestion:
    return MultipleChoiceQuestion(
        question_text="Who will be nominated?",
        options=["Alice", "Bob"],
        id_of_question=202,
        id_of_post=202,
        open_time=OPEN,
        close_time=CLOSE,
        resolution_string="Alice",
        api_json=community_history(forecast_values=[0.5, 0.5]),
    )


def resolved_numeric() -> NumericQuestion:
    return NumericQuestion(
        question_text="How many combined points will be scored in the game?",
        id_of_question=203,
        id_of_post=203,
        open_time=OPEN,
        close_time=CLOSE,
        resolution_string="51",
        lower_bound=0.0,
        upper_bound=100.0,
        open_lower_bound=False,
        open_upper_bound=True,
        api_json=community_history(forecast_values=[i / 200 for i in range(201)]),
    )


def test_pinned_clock_replays_from_shortly_after_opening():
    assert pc.pinned_clock_policy(10)(resolved_binary()).now() == OPEN + timedelta(minutes=10)
    no_open = resolved_binary().model_copy(update={"open_time": None, "published_time": None})
    with pytest.raises(ValueError):
        pc.pinned_clock_policy(10)(no_open)


def test_reference_forecasts_come_from_the_community_history():
    assert pc.reference_forecast(resolved_binary(), CLOSE) == pytest.approx(0.3)
    assert pc.reference_forecast(resolved_mc(), CLOSE) == {"Alice": 0.5, "Bob": 0.5}
    assert len(pc.reference_forecast(resolved_numeric(), CLOSE)) == 201
    assert pc.reference_forecast(resolved_binary().model_copy(update={"api_json": {}}), CLOSE) is None
    assert pc.reference_forecast(resolved_binary(), OPEN - timedelta(days=1)) is None


def test_outcomes_handle_every_type():
    assert pc.outcome_of(resolved_binary("yes")) is True
    assert pc.outcome_of(resolved_mc()) == "Alice"
    assert pc.outcome_of(resolved_numeric()) == 51.0
    above = resolved_numeric().model_copy(update={"resolution_string": "above_upper_bound"})
    assert pc.outcome_of(above) == 200.0
    assert pc.outcome_of(resolved_binary("annulled")) is None


def test_config_switches():
    config = pc.config_with_disabled(BotConfig(), ["markets", "prior", "escalation"])
    assert not config.use_markets
    assert config.data_prior_weight == 0.0
    assert not config.full_panel_for_non_binary
    with pytest.raises(ValueError, match="Unknown switch"):
        pc.config_with_disabled(BotConfig(), ["vibes"])


def test_scoring_multiple_choice_and_numeric_forecasts():
    mc = resolved_mc()
    prediction = PredictedOptionList(
        predicted_options=[PredictedOption(option_name="Alice", probability=0.8), PredictedOption(option_name="Bob", probability=0.2)]
    )
    baseline, relative = pc.score_forecast(mc, pc.forecast_json(mc, prediction))
    assert baseline > 0 and relative > 0

    numeric = resolved_numeric()
    percentiles = [Percentile(percentile=p, value=v) for p, v in zip((0.1, 0.2, 0.4, 0.6, 0.8, 0.9), (38, 42, 48, 53, 58, 63))]
    distribution = NumericDistribution.from_question(percentiles, numeric)
    baseline, relative = pc.score_forecast(numeric, pc.forecast_json(numeric, distribution))
    assert baseline > 0
    assert relative == pytest.approx(baseline)


async def test_pastcast_scores_a_resolved_binary_question():
    question = resolved_binary("no")
    models = ScriptedModels({"news": "Probability: 20%", "outside_view": "Probability: 22%"})
    [record] = await pc.run_pastcast(
        [question], llm_call=models, news=FakeNews(), http_client_factory=offline_client
    )
    assert record.error is None
    assert record.kind == "event"
    assert 0.2 <= record.forecast <= 0.22
    assert record.baseline_score == pytest.approx(scoring.binary_baseline_score(record.forecast, False))
    assert record.relative_score > 0


async def test_pastcast_records_failures_and_counts_them_as_zero():
    boom = RuntimeError("provider down")
    models = ScriptedModels({name: boom for name in ALL_ANALYSTS})
    records = await pc.run_pastcast(
        [resolved_binary("no")], llm_call=models, news=FakeNews(), http_client_factory=offline_client
    )
    summary = pc.summarize_run(records)
    assert records[0].error
    assert summary.failures == 1
    assert summary.effective_baseline.mean == 0.0
    assert summary.baseline.count == 0


def test_dataset_round_trip(tmp_path):
    questions = [resolved_binary("yes"), resolved_mc(), resolved_numeric()]
    assert all(is_scoreable(q) for q in questions)
    path = tmp_path / "holdout.jsonl"
    save_questions(questions, path)
    loaded = load_questions(path)
    assert [type(q) for q in loaded] == [type(q) for q in questions]
    assert [q.id_of_question for q in loaded] == [201, 202, 203]
    assert pc.outcome_of(loaded[0]) is True
    assert pc.reference_forecast(loaded[0], CLOSE) == pytest.approx(0.3)


def record(question_id, score):
    return pc.PastcastRecord(question_id, None, f"Q{question_id}", "binary", "event", 0.5, False, score, None)


def test_noise_floor_measures_spread_and_unstable_questions():
    runs = [[record(1, 10.0), record(2, 50.0)], [record(1, 12.0), record(2, -30.0)], [record(1, 11.0), record(2, 40.0)]]
    floor = pc.noise_floor(runs, instability_threshold=20.0)
    assert floor.run_means == [30.0, -9.0, 25.5]
    assert floor.spread > 0
    assert [title for title, _ in floor.unstable_questions] == ["Q2"]
    report = pc.render_report(runs, ["markets"])
    assert "Noise floor" in report and "Switched off: markets." in report


def test_pastcast_command_refuses_to_spend_without_confirmation(tmp_path, capsys):
    path = tmp_path / "holdout.jsonl"
    save_questions([resolved_binary()], path)
    assert cli_main(["pastcast", "--data", str(path), "--runs", "2"]) == 0
    output = capsys.readouterr().out
    assert "Worst-case model spend $1.20" in output
    assert "Nothing was run" in output


def test_resample_and_weather_switches():
    config = pc.config_with_disabled(BotConfig(), ["resample", "weather"])
    assert config.min_answers == 0
    assert config.use_weather is False


def test_date_questions_are_scored_on_a_timestamp_axis():
    question = DateQuestion(
        question_text="When will the report be published?",
        id_of_question=301,
        id_of_post=301,
        page_url="https://www.metaculus.com/questions/301/",
        open_time=OPEN,
        close_time=CLOSE,
        lower_bound=OPEN,
        upper_bound=OPEN + timedelta(days=100),
        open_lower_bound=False,
        open_upper_bound=True,
        resolution_string="2026-10-01",
    )
    assert pc.outcome_of(question) == datetime(2026, 10, 1, tzinfo=timezone.utc).timestamp()
    uniform = [i / 200 for i in range(201)]
    baseline, relative = pc.score_forecast(question, uniform)
    assert baseline == pytest.approx(0.0, abs=1e-9)
    assert relative is None


async def test_fetch_counts_resolutions_the_api_withheld():
    def binary(resolution):
        return BinaryQuestion(
            question_text="Will it happen?",
            id_of_question=1,
            id_of_post=1,
            page_url="https://www.metaculus.com/questions/1/",
            open_time=OPEN,
            resolution_string=resolution,
        )

    class FakeClient:
        async def get_questions_matching_filter(self, api_filter, num_questions, error_if_question_target_missed):
            return [binary("yes"), binary(None)]

    result = await fetch_resolved(FakeClient(), ["minibench-2026-08-24"])
    assert (len(result.questions), result.fetched, result.hidden) == (1, 2, 1)


def test_fetch_command_explains_withheld_resolutions(tmp_path, capsys, monkeypatch):
    async def fake_fetch(client, tournaments, max_questions):
        return FetchResult(questions=[], fetched=60, hidden=60)

    monkeypatch.setattr(dataset_module, "fetch_resolved", fake_fetch)
    monkeypatch.setattr(forecasting_tools, "MetaculusClient", lambda: None)
    code = cli_main(["fetch", "--tournament", "minibench-2026-08-24", "--out", str(tmp_path / "questions.jsonl")])
    output = capsys.readouterr().out
    assert code == 1
    assert "restricted API tier" in output
    assert "Bot Benchmarking Access Tier" in output


def test_news_queries_switch():
    assert pc.config_with_disabled(BotConfig(), ["news_queries"]).extra_news_queries == 0
