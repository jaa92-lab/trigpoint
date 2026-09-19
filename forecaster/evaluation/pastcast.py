"""Pastcasting: replay resolved questions as if the bot were seeing them fresh.

Each question's clock is pinned to shortly after it opened, so every evidence
fetcher sees only what existed then. Prediction markets are skipped because
their past prices are unavailable. Forecasts are scored against the actual
outcome and, where the API supplies it, against the community forecast at the
moment it was revealed, which is the bar a bot must clear to earn peer points.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone

from forecasting_tools import (
    BinaryQuestion,
    DateQuestion,
    MetaculusClient,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)
from forecasting_tools.data_models.questions import OutOfBoundsResolution

from forecaster.bot import PanelForecaster, bounds_of, question_type_of
from forecaster.clock import Clock
from forecaster.config import BotConfig
from forecaster.evaluation import scoring
from forecaster.evidence.cache import DiskCache
from forecaster.evidence.news import NewsSearcher
from forecaster.triage import triage

DISABLE_SWITCHES = {
    "news": "use_news",
    "markets": "use_markets",
    "prices": "use_prices",
    "sources": "use_sources",
    "weather": "use_weather",
    "grounding": "use_grounding",
}
SPECIAL_SWITCHES = ("prior", "escalation", "resample", "news_queries")


@dataclass
class PastcastRecord:
    question_id: int | None
    url: str | None
    title: str
    question_type: str
    kind: str
    forecast: object | None
    resolution: object | None
    baseline_score: float | None
    relative_score: float | None
    error: str | None = None

    def to_json(self) -> dict:
        return asdict(self)


def config_with_disabled(config: BotConfig, disabled: Sequence[str]) -> BotConfig:
    changes: dict[str, object] = {}
    for name in disabled:
        if name == "prior":
            changes["data_prior_weight"] = 0.0
        elif name == "escalation":
            changes["escalate_on_logit_spread"] = math.inf
            changes["full_panel_for_non_binary"] = False
        elif name == "resample":
            changes["min_answers"] = 0
        elif name == "news_queries":
            changes["extra_news_queries"] = 0
        elif name in DISABLE_SWITCHES:
            changes[DISABLE_SWITCHES[name]] = False
        else:
            options = ", ".join(sorted([*DISABLE_SWITCHES, *SPECIAL_SWITCHES]))
            raise ValueError(f"Unknown switch {name!r}. Choose from: {options}")
    return config.with_changes(**changes)


def pinned_clock_policy(delay_minutes: float = 10.0) -> Callable[[MetaculusQuestion], Clock]:
    def policy(question: MetaculusQuestion) -> Clock:
        start = question.open_time or question.published_time
        if start is None:
            raise ValueError(f"Question {question.page_url} has no open time to replay from")
        return Clock.pinned(start + timedelta(minutes=delay_minutes))

    return policy


def reference_time(question: MetaculusQuestion) -> datetime | None:
    return question.cp_reveal_time or question.close_time


def reference_forecast(question: MetaculusQuestion, at: datetime | None) -> object | None:
    """The community forecast (the bot aggregate, in bot tournaments) at a moment, if available."""
    if at is None:
        return None
    try:
        history = question.api_json["question"]["aggregations"]["recency_weighted"]["history"]
    except (KeyError, TypeError):
        return None
    moment = at.timestamp()
    entry = None
    for item in history or []:
        start = item.get("start_time") or 0
        end = item.get("end_time")
        if start <= moment and (not end or moment <= end):
            entry = item
    if entry is None:
        return None
    if isinstance(question, BinaryQuestion):
        centers = entry.get("centers") or []
        return float(centers[0]) if centers else None
    values = entry.get("forecast_values") or []
    if isinstance(question, MultipleChoiceQuestion):
        return dict(zip(question.options, values)) if len(values) == len(question.options) else None
    if isinstance(question, (NumericQuestion, DateQuestion)):
        return list(values) if len(values) == question.cdf_size else None
    return None


def outcome_of(question: MetaculusQuestion) -> object | None:
    try:
        if isinstance(question, BinaryQuestion):
            resolution = question.binary_resolution
            return resolution if isinstance(resolution, bool) else None
        if isinstance(question, MultipleChoiceQuestion):
            resolution = question.mc_resolution
            return resolution if isinstance(resolution, str) else None
        if isinstance(question, (NumericQuestion, DateQuestion)):
            is_date = isinstance(question, DateQuestion)
            resolution = question.date_resolution if is_date else question.numeric_resolution
            lower, upper = bounds_of(question)
            span = upper - lower
            if resolution == OutOfBoundsResolution.ABOVE_UPPER_BOUND:
                return upper + span
            if resolution == OutOfBoundsResolution.BELOW_LOWER_BOUND:
                return lower - span
            if isinstance(resolution, datetime):
                return resolution.timestamp()
            if isinstance(resolution, (int, float)) and not isinstance(resolution, bool):
                return float(resolution)
    except Exception:
        return None
    return None


def forecast_json(question: MetaculusQuestion, prediction: object) -> object:
    if isinstance(question, BinaryQuestion):
        return round(float(prediction), 6)
    if isinstance(question, MultipleChoiceQuestion):
        return {o.option_name: round(o.probability, 6) for o in prediction.predicted_options}
    if isinstance(question, (NumericQuestion, DateQuestion)):
        return [round(p.percentile, 6) for p in prediction.get_cdf()]
    return None


def score_forecast(question: MetaculusQuestion, forecast: object) -> tuple[float | None, float | None]:
    outcome = outcome_of(question)
    if outcome is None or forecast is None:
        return None, None
    reference = reference_forecast(question, reference_time(question))
    if isinstance(question, BinaryQuestion):
        baseline = scoring.binary_baseline_score(forecast, outcome)
        relative = scoring.binary_relative_score(forecast, reference, outcome) if reference is not None else None
        return baseline, relative
    if isinstance(question, MultipleChoiceQuestion):
        baseline = scoring.mc_baseline_score(forecast, outcome)
        relative = scoring.mc_relative_score(forecast, reference, outcome) if reference else None
        return baseline, relative
    if isinstance(question, (NumericQuestion, DateQuestion)):
        if question.zero_point is not None:
            return None, None  # log-scaled axes are not scored yet
        uniform = [i / (len(forecast) - 1) for i in range(len(forecast))]
        lower, upper = bounds_of(question)
        baseline = scoring.numeric_relative_score(forecast, uniform, lower, upper, outcome)
        relative = scoring.numeric_relative_score(forecast, reference, lower, upper, outcome) if reference else None
        return baseline, relative
    return None, None


def record_for(question: MetaculusQuestion, report: object) -> PastcastRecord:
    question_type = question_type_of(question)
    moment = question.open_time or question.published_time or question.close_time
    kind = (
        triage(question_text=question.question_text, question_type=question_type, now=moment).kind
        if moment
        else "unknown"
    )
    base = dict(
        question_id=question.id_of_question,
        url=question.page_url,
        title=question.question_text,
        question_type=question_type,
        kind=kind,
        resolution=outcome_of(question),
    )
    if isinstance(report, BaseException):
        return PastcastRecord(
            **base, forecast=None, baseline_score=None, relative_score=None,
            error=f"{type(report).__name__}: {report}",
        )
    forecast = forecast_json(question, report.prediction)
    baseline, relative = score_forecast(question, forecast)
    return PastcastRecord(**base, forecast=forecast, baseline_score=baseline, relative_score=relative)


async def run_pastcast(
    questions: Sequence[MetaculusQuestion],
    *,
    config: BotConfig | None = None,
    delay_minutes: float = 10.0,
    cache_dir: str | None = None,
    llm_call=None,
    news=None,
    http_client_factory=None,
) -> list[PastcastRecord]:
    bot = PanelForecaster(
        config=config or BotConfig(),
        clock_policy=pinned_clock_policy(delay_minutes),
        llm_call=llm_call,
        news=news if news is not None else NewsSearcher(cache=DiskCache(cache_dir)),
        http_client_factory=http_client_factory,
        publish_reports_to_metaculus=False,
        skip_previously_forecasted_questions=False,
        metaculus_client=MetaculusClient(),
    )
    reports = await bot.forecast_questions(list(questions), return_exceptions=True)
    return [record_for(question, report) for question, report in zip(questions, reports)]


# ------------------------------------------------------------ summaries


@dataclass
class RunSummary:
    count: int
    failures: int
    baseline: scoring.ScoreSummary
    effective_baseline: scoring.ScoreSummary
    relative: scoring.ScoreSummary
    baseline_by_kind: dict[str, scoring.ScoreSummary]


def summarize_run(records: Sequence[PastcastRecord]) -> RunSummary:
    scored = [r for r in records if r.baseline_score is not None]
    effective = [r.baseline_score if r.baseline_score is not None else 0.0 for r in records if r.resolution is not None]
    by_kind = {
        kind: scoring.summarize([r.baseline_score for r in scored if r.kind == kind])
        for kind in sorted({r.kind for r in scored})
    }
    return RunSummary(
        count=len(records),
        failures=sum(1 for r in records if r.error),
        baseline=scoring.summarize([r.baseline_score for r in scored]),
        effective_baseline=scoring.summarize(effective),
        relative=scoring.summarize([r.relative_score for r in records if r.relative_score is not None]),
        baseline_by_kind=by_kind,
    )


@dataclass
class NoiseFloor:
    run_means: list[float]
    spread: float
    unstable_questions: list[tuple[str, float]]


def noise_floor(runs: Sequence[Sequence[PastcastRecord]], instability_threshold: float = 20.0) -> NoiseFloor:
    means = [summarize_run(run).effective_baseline.mean for run in runs]
    spread = statistics.stdev(means) if len(means) > 1 else float("nan")
    per_question: dict[tuple[object, str], list[float]] = {}
    for run in runs:
        for record in run:
            if record.baseline_score is not None:
                per_question.setdefault((record.question_id, record.title), []).append(record.baseline_score)
    unstable = [
        (title, statistics.stdev(scores))
        for (_, title), scores in per_question.items()
        if len(scores) > 1 and statistics.stdev(scores) > instability_threshold
    ]
    unstable.sort(key=lambda pair: pair[1], reverse=True)
    return NoiseFloor(run_means=means, spread=spread, unstable_questions=unstable)


def estimate_max_cost(question_count: int, runs: int, config: BotConfig) -> float:
    return question_count * runs * config.max_cost_per_question


def _fmt(summary: scoring.ScoreSummary) -> str:
    if summary.count == 0:
        return "n/a"
    return f"{summary.mean:+.1f} (±{summary.stderr:.1f}, n={summary.count})"


def render_report(runs: Sequence[Sequence[PastcastRecord]], disabled: Sequence[str]) -> str:
    lines = [
        "# Pastcast results",
        f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC.",
        f"Switched off: {', '.join(disabled) if disabled else 'nothing (full bot)'}.",
        "",
        "Baseline score: 0 is a uniform guess, higher is better. Counting failures as 0 matches the tournament.",
        "Relative score: points against the community forecast at reveal time. Positive beats the field.",
        "",
    ]
    for index, run in enumerate(runs, start=1):
        summary = summarize_run(run)
        lines += [
            f"## Run {index}",
            f"- Questions: {summary.count}, failures: {summary.failures}",
            f"- Baseline, failures as 0: {_fmt(summary.effective_baseline)}",
            f"- Baseline, forecasts only: {_fmt(summary.baseline)}",
            f"- Relative to community: {_fmt(summary.relative)}",
        ]
        lines += [f"- {kind}: {_fmt(s)}" for kind, s in summary.baseline_by_kind.items()]
        lines.append("")
    if len(runs) > 1:
        floor = noise_floor(runs)
        lines += [
            "## Noise floor",
            f"- Run means: {', '.join(f'{m:+.1f}' for m in floor.run_means)}",
            f"- Spread between identical runs (standard deviation): {floor.spread:.1f}",
            "- A change must beat this spread to count as an improvement.",
        ]
        lines += [f"- Unstable: {title} (±{sd:.0f})" for title, sd in floor.unstable_questions[:10]]
    return "\n".join(lines).strip() + "\n"
