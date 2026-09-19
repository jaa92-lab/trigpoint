"""A scorecard from the bot's own resolved forecasts.

The restricted API tier hides most closed questions, but it does return the
resolution of every question this account forecast on, plus Metaculus's own
score once one exists. That is the feedback loop: after each MiniBench round,
score what the bot did, by question kind, and check its calibration. It costs
no model credit.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx

from forecaster.evaluation import scoring
from forecaster.evidence.http import get_json
from forecaster.triage import triage

API = "https://www.metaculus.com/api"
BACKOFF_SECONDS = 10.0  # Metaculus rate-limits bursts of reads; wait well before retrying
CANCELED = ("annulled", "ambiguous")
CALIBRATION_BUCKETS = ((0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01))


@dataclass
class ScoredForecast:
    post_id: int | None
    question_id: int | None
    title: str
    url: str
    question_type: str
    kind: str
    resolved_at: str | None
    forecast: str
    outcome: str
    p_yes: float | None
    correct_option_probability: float | None
    baseline_score: float | None
    covered_by_80pct_interval: bool | None
    metaculus_scores: dict[str, float] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------- fetching


async def fetch_user_id(client: httpx.AsyncClient) -> int:
    data = await get_json(client, f"{API}/users/me/", backoff_seconds=BACKOFF_SECONDS)
    return int(data["id"])


async def list_forecasted_posts(
    client: httpx.AsyncClient,
    user_id: int,
    tournaments: Sequence[str | int] = (),
    page_size: int = 100,
    pause_seconds: float = 2.0,
    sleep=asyncio.sleep,
) -> list[dict[str, Any]]:
    """Every post this account forecast on, in any status.

    The API sends a next-page link even past the last result, so paging stops at
    the first short page rather than at a missing link.
    """
    posts: list[dict[str, Any]] = []
    params: dict[str, Any] | None = {"forecaster_id": user_id, "limit": page_size}
    if tournaments:
        params["tournaments"] = ",".join(str(t) for t in tournaments)
    url: str | None = f"{API}/posts/"
    while url:
        page = await get_json(client, url, params=params, backoff_seconds=BACKOFF_SECONDS)
        params = None  # the next URL already carries them
        results = page.get("results") or []
        posts.extend(results)
        if len(results) < page_size or not page.get("next"):
            break
        url = page["next"]
        await sleep(pause_seconds)
    return posts


async def fetch_post(client: httpx.AsyncClient, post_id: int) -> dict[str, Any]:
    return await get_json(client, f"{API}/posts/{post_id}/", backoff_seconds=BACKOFF_SECONDS)


def questions_in_post(post: dict[str, Any]) -> list[dict[str, Any]]:
    """The scoreable question dicts in a post: the question itself, or each group member."""
    post_id = post.get("id")
    base_url = f"https://www.metaculus.com/questions/{post_id}/"
    found = []
    if post.get("question"):
        question = dict(post["question"])
        question.setdefault("title", post.get("title") or "")
        question["_post_id"], question["_url"] = post_id, base_url
        found.append(question)
    for member in (post.get("group_of_questions") or {}).get("questions") or []:
        question = dict(member)
        label = question.get("label")
        question["title"] = f"{post.get('title') or question.get('title') or ''}" + (f" [{label}]" if label else "")
        question["_post_id"], question["_url"] = post_id, base_url
        found.append(question)
    return found


# ---------------------------------------------------------------- scoring


def my_latest_forecast(question: dict[str, Any]) -> dict[str, Any] | None:
    latest = (question.get("my_forecasts") or {}).get("latest")
    return latest if latest and latest.get("forecast_values") else None


def metaculus_scores(question: dict[str, Any]) -> dict[str, float]:
    data = (question.get("my_forecasts") or {}).get("score_data") or {}
    return {
        key: float(value)
        for key, value in data.items()
        if "score" in key and isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def _parse_outcome(question: dict[str, Any]) -> float | str | bool | None:
    resolution = question.get("resolution")
    if resolution is None or str(resolution).lower() in CANCELED:
        return None
    qtype = question.get("type")
    if qtype == "binary":
        return {"yes": True, "no": False}.get(str(resolution).lower())
    if qtype == "multiple_choice":
        return str(resolution)
    scaling = question.get("scaling") or {}
    lower, upper = scaling.get("range_min"), scaling.get("range_max")
    if lower is None or upper is None:
        return None
    span = float(upper) - float(lower)
    text = str(resolution).lower()
    if text == "above_upper_bound":
        return float(upper) + span
    if text == "below_lower_bound":
        return float(lower) - span
    if qtype == "date":
        try:
            moment = datetime.fromisoformat(str(resolution).replace("Z", "+00:00"))
        except ValueError:
            return None
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        return moment.timestamp()
    try:
        return float(resolution)
    except (TypeError, ValueError):
        return None


def interval_from_cdf(cdf: Sequence[float], lower: float, upper: float, low: float = 0.1, high: float = 0.9) -> tuple[float, float]:
    n = len(cdf)
    step = (upper - lower) / (n - 1)

    def crossing(level: float) -> float:
        for i, value in enumerate(cdf):
            if value >= level:
                return lower + i * step
        return upper

    return crossing(low), crossing(high)


def _format_outcome(question_type: str, outcome: float | str | bool) -> str:
    if question_type == "binary":
        return "Yes" if outcome else "No"
    if question_type == "date" and isinstance(outcome, float):
        return f"{datetime.fromtimestamp(outcome, tz=timezone.utc):%Y-%m-%d}"
    if isinstance(outcome, float):
        return f"{outcome:,.6g}"
    return str(outcome)


def score_question(question: dict[str, Any]) -> ScoredForecast | None:
    """Score one resolved question the bot forecast on, or None if it can't be scored."""
    if question.get("status") != "resolved":
        return None
    latest = my_latest_forecast(question)
    outcome = _parse_outcome(question)
    if latest is None or outcome is None:
        return None
    qtype = str(question.get("type"))
    values = [float(v) for v in latest["forecast_values"]]
    title = question.get("title") or ""
    kind = triage(question_text=title, question_type=qtype, now=datetime.now(timezone.utc)).kind
    record = ScoredForecast(
        post_id=question.get("_post_id"),
        question_id=question.get("id"),
        title=title,
        url=question.get("_url") or "",
        question_type=qtype,
        kind=kind,
        resolved_at=question.get("actual_resolve_time"),
        forecast="",
        outcome=_format_outcome(qtype, outcome),
        p_yes=None,
        correct_option_probability=None,
        baseline_score=None,
        covered_by_80pct_interval=None,
        metaculus_scores=metaculus_scores(question),
    )
    if qtype == "binary" and len(values) == 2:
        p_yes = values[1]
        record.p_yes = p_yes
        record.forecast = f"{p_yes:.0%} Yes"
        record.baseline_score = scoring.binary_baseline_score(p_yes, bool(outcome))
        return record
    if qtype == "multiple_choice":
        options = question.get("options") or []
        if len(options) != len(values) or outcome not in options:
            return None
        probabilities = dict(zip(options, values))
        record.correct_option_probability = probabilities[str(outcome)]
        record.forecast = ", ".join(f"{o} {p:.0%}" for o, p in probabilities.items())
        record.baseline_score = scoring.mc_baseline_score(probabilities, str(outcome))
        return record
    scaling = question.get("scaling") or {}
    if scaling.get("zero_point") is not None or len(values) < 2:
        return None  # log-scaled axes are not scored yet
    lower, upper = float(scaling["range_min"]), float(scaling["range_max"])
    uniform = [i / (len(values) - 1) for i in range(len(values))]
    record.baseline_score = scoring.numeric_relative_score(values, uniform, lower, upper, float(outcome))
    p10, p90 = interval_from_cdf(values, lower, upper)
    record.covered_by_80pct_interval = p10 <= float(outcome) <= p90
    record.forecast = f"P10 {_format_outcome(qtype, p10)}, P90 {_format_outcome(qtype, p90)}"
    return record


async def build_scorecard(
    client: httpx.AsyncClient,
    tournaments: Sequence[str | int] = (),
    pause_seconds: float = 2.0,
    sleep=asyncio.sleep,
    page_size: int = 100,
) -> list[ScoredForecast]:
    user_id = await fetch_user_id(client)
    records: list[ScoredForecast] = []
    posts = await list_forecasted_posts(client, user_id, tournaments, page_size, pause_seconds, sleep)
    for post in posts:
        await sleep(pause_seconds)
        detail = await fetch_post(client, int(post["id"]))
        for question in questions_in_post(detail):
            record = score_question(question)
            if record is not None:
                records.append(record)
    return records


# ------------------------------------------------------------- reporting


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def calibration_table(records: Sequence[ScoredForecast]) -> list[tuple[str, int, float, float]]:
    """(bucket, count, mean forecast, observed yes rate) for binary forecasts."""
    rows = []
    binary = [r for r in records if r.p_yes is not None]
    for low, high in CALIBRATION_BUCKETS:
        inside = [r for r in binary if low <= r.p_yes < high]
        if not inside:
            continue
        observed = _mean([1.0 if r.outcome == "Yes" else 0.0 for r in inside])
        rows.append((f"{low:.0%}-{min(high, 1.0):.0%}", len(inside), _mean([r.p_yes for r in inside]), observed))
    return rows


def render_scorecard(records: Sequence[ScoredForecast], generated: datetime | None = None) -> str:
    generated = generated or datetime.now(timezone.utc)
    lines = [
        "# Trigpoint scorecard",
        f"Generated {generated:%Y-%m-%d %H:%M} UTC from {len(records)} resolved forecasts.",
        "",
        "Baseline score: 0 is a uniform guess, 100 is certainty on the right answer, negative is worse than guessing.",
        "Metaculus scores are the site's own numbers where it has published them; peer scores are what the tournament pays on.",
        "",
    ]
    if not records:
        lines.append("Nothing has resolved yet.")
        return "\n".join(lines) + "\n"

    scored = [r for r in records if r.baseline_score is not None]
    lines += ["## Overall", f"- Mean baseline score: {_mean([r.baseline_score for r in scored]):+.1f} (n={len(scored)})"]
    score_keys = sorted({key for r in records for key in r.metaculus_scores})
    for key in score_keys:
        values = [r.metaculus_scores[key] for r in records if key in r.metaculus_scores]
        lines.append(f"- Metaculus {key.replace('_', ' ')}: mean {_mean(values):+.1f}, sum {sum(values):+.1f} (n={len(values)})")
    lines.append("")

    lines += ["## By question kind", "| Kind | n | Mean baseline | Metaculus peer (mean) |", "|---|---|---|---|"]
    for kind in sorted({r.kind for r in scored}):
        inside = [r for r in scored if r.kind == kind]
        peers = [v for r in inside for k, v in r.metaculus_scores.items() if "peer" in k]
        peer = f"{_mean(peers):+.1f}" if peers else "n/a"
        lines.append(f"| {kind} | {len(inside)} | {_mean([r.baseline_score for r in inside]):+.1f} | {peer} |")
    lines.append("")

    rows = calibration_table(records)
    if rows:
        lines += [
            "## Calibration of yes/no forecasts",
            "If the bot is calibrated, the observed rate matches the mean forecast in each row.",
            "| Forecast | n | Mean forecast | Observed Yes |",
            "|---|---|---|---|",
        ]
        lines += [f"| {bucket} | {n} | {mean:.0%} | {observed:.0%} |" for bucket, n, mean, observed in rows]
        lines.append("")

    numeric = [r for r in records if r.covered_by_80pct_interval is not None]
    if numeric:
        covered = sum(1 for r in numeric if r.covered_by_80pct_interval)
        lines += [
            "## Numeric and date intervals",
            f"- Outcomes inside the bot's 10th-90th percentile range: {covered} of {len(numeric)} ({covered / len(numeric):.0%}). Well-calibrated is about 80%; lower means the ranges are too narrow.",
            "",
        ]

    worst = sorted(scored, key=lambda r: r.baseline_score)[:10]
    lines += ["## Worst misses", "| Score | Kind | Forecast | Outcome | Question |", "|---|---|---|---|---|"]
    for r in worst:
        lines.append(f"| {r.baseline_score:+.0f} | {r.kind} | {r.forecast} | {r.outcome} | [{r.title[:80]}]({r.url}) |")
    skipped = len(records) - len(scored)
    if skipped:
        lines += ["", f"{skipped} resolved forecasts could not be scored (log-scaled axes or unusual resolutions)."]
    return "\n".join(lines) + "\n"
