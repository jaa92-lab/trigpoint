"""Analyst prompts, forecast parsing, and a runner that never raises.

Each analyst is one model call over one slice of evidence. Parsing is strict
and cheap (regular expressions); a parser model is only a fallback. A failed
analyst returns a result with an error instead of an exception, so one bad
call can never sink a question that other analysts could still answer.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone

from forecaster.budget import CostLedger, estimate_tokens
from forecaster.config import AnalystSpec

PERCENTILE_LABELS = (10, 20, 40, 60, 80, 90)
REASONING_OUTPUT_MULTIPLIER = 3.0
ESTIMATED_OUTPUT_TOKENS = 3000  # for affordability checks before a call
MIN_FALLBACK_SECONDS = 20.0

PANEL_PREAMBLE = (
    "You are one member of a forecasting panel. Each panelist sees a different slice of "
    "evidence, and the panel's forecasts are combined afterward. Give your own honest estimate "
    "from what you can see. Do not drift toward 50% just because your evidence is partial: say "
    "what it implies, and allow for what you cannot see."
)
RESOLVER_NOTE = (
    "An automated reader will resolve this question by applying these criteria literally. "
    "Forecast what that reader will decide from the criteria as written."
)

LlmCall = Callable[[str, str], Awaitable[str]]
FallbackParse = Callable[[str], Awaitable[object | None]]


@dataclass(frozen=True)
class QuestionView:
    """The parts of a question the prompts need, independent of the framework.

    Date questions carry their bounds as UTC timestamps.
    """

    title: str
    question_type: str
    resolution_criteria: str = ""
    fine_print: str = ""
    background: str = ""
    close_time: datetime | None = None
    resolve_time: datetime | None = None
    options: tuple[str, ...] = ()
    unit: str | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None
    open_lower_bound: bool = True
    open_upper_bound: bool = True


@dataclass
class AnalystResult:
    name: str
    model: str
    value: object | None
    rationale: str
    weight: float = 1.0
    flags: list[str] = field(default_factory=list)
    error: str | None = None
    note: str | None = None  # e.g. a stand-in model answered; never affects weight or escalation

    @property
    def ok(self) -> bool:
        return self.value is not None and self.error is None


# ------------------------------------------------------------ prompts


def _fmt_time(value: datetime | None) -> str:
    return f"{value:%Y-%m-%d %H:%M} UTC" if value else "not stated"


def _fmt_date(timestamp: float) -> str:
    return f"{datetime.fromtimestamp(timestamp, tz=timezone.utc):%Y-%m-%d}"


def _timing(question: QuestionView, now: datetime) -> str:
    parts = [
        f"Today is {now:%Y-%m-%d %H:%M} UTC.",
        f"Forecasting closes {_fmt_time(question.close_time)}.",
        f"Resolution is expected {_fmt_time(question.resolve_time)}.",
    ]
    if question.resolve_time is not None:
        days = (question.resolve_time - now).total_seconds() / 86400.0
        parts.append(f"That is about {days:.1f} days from now.")
    return " ".join(parts)


def bounds_message(question: QuestionView) -> str:
    if question.lower_bound is None or question.upper_bound is None:
        return ""
    if question.question_type == "date":
        earliest, latest = _fmt_date(question.lower_bound), _fmt_date(question.upper_bound)
        low = (
            f"is likely not earlier than {earliest}"
            if question.open_lower_bound
            else f"cannot be earlier than {earliest}"
        )
        high = f"is likely not later than {latest}" if question.open_upper_bound else f"cannot be later than {latest}"
        return f"The outcome {low}, and it {high}."
    unit = f" {question.unit}" if question.unit else ""
    low = (
        f"is likely not lower than {question.lower_bound:,g}{unit}"
        if question.open_lower_bound
        else f"cannot be lower than {question.lower_bound:,g}{unit}"
    )
    high = (
        f"is likely not higher than {question.upper_bound:,g}{unit}"
        if question.open_upper_bound
        else f"cannot be higher than {question.upper_bound:,g}{unit}"
    )
    return f"The outcome {low}, and it {high}."


def answer_format(question: QuestionView) -> str:
    if question.question_type == "binary":
        return "End your answer with one line in exactly this form:\nProbability: NN%"
    if question.question_type == "multiple_choice":
        lines = "\n".join(f"{option}: NN%" for option in question.options)
        return (
            "End your answer with one line per option, in this order, in exactly this form. "
            "The percentages must add up to 100.\n" + lines
        )
    if question.question_type == "date":
        lines = "\n".join(f"Percentile {label}: YYYY-MM-DD" for label in PERCENTILE_LABELS)
        return (
            f"{bounds_message(question)}\n"
            "Give each value as a date, YYYY-MM-DD, adding THH:MMZ in UTC only if the hour matters. "
            "Dates must get later from Percentile 10 to Percentile 90.\n"
            "End your answer with exactly these lines:\n" + lines
        ).strip()
    unit = question.unit or "the units the question asks for"
    lines = "\n".join(f"Percentile {label}: X" for label in PERCENTILE_LABELS)
    return (
        f"{bounds_message(question)}\n"
        f"Give values in {unit}. Values must increase from Percentile 10 to Percentile 90. "
        "Never use scientific notation.\n"
        "End your answer with exactly these lines:\n" + lines
    ).strip()


def reasoning_steps(question: QuestionView) -> str:
    """A short checklist every analyst works through, drawn from top public bots' prompts
    (Panshul42's Q2 2025 winner and joy-void-joy's scheduled-catalyst rule)."""
    steps = [
        "Restate in one line exactly what the question requires or measures, by when, and according to which source.",
        "Status quo: what is the outcome if nothing changes before resolution?",
        "Scheduled events: is anything scheduled inside the window (a meeting, report, release, vote, game, "
        "or deadline) that could move the outcome? Say when.",
        "Evidence: what in your evidence moves you away from the status quo, how far, and is each piece "
        "strong, moderate, or weak?",
    ]
    if question.question_type == "binary":
        steps += [
            "Sanity check: out of 100 questions like this one, how many would resolve Yes? "
            "Your forecast should match that number, or you should say why it differs.",
            "Window check: would your forecast change if the window were half as long, or twice as long? "
            "If it would not change at all, you may be ignoring timing.",
        ]
    elif question.question_type == "multiple_choice":
        steps.append("Sanity check: in situations like this, how often does the option you rate highest actually win?")
    else:
        steps.append(
            "Sanity check: about 1 in 5 outcomes should land outside your 10th to 90th percentile range. "
            "Name a plausible outcome just outside it, and check the range is neither too narrow nor too wide."
        )
    return "Work through these briefly, in order:\n" + "\n".join(f"{i}. {step}" for i, step in enumerate(steps, 1))


def build_prompt(spec: AnalystSpec, question: QuestionView, evidence_text: str, now: datetime) -> str:
    sections: list[str] = [PANEL_PREAMBLE, "", f"Your role: {spec.brief}", "", "# Question", question.title]
    if question.question_type == "multiple_choice":
        sections += ["", "Options: " + "; ".join(question.options)]
    sections += ["", "## Resolution criteria", question.resolution_criteria or "Not provided.", "", RESOLVER_NOTE]
    if question.fine_print:
        sections += ["", "## Fine print", question.fine_print]
    if question.background:
        sections += ["", "## Background", question.background]
    sections += [
        "",
        "## Timing",
        _timing(question, now),
        "",
        "# Your evidence",
        evidence_text,
        "",
        "# How to answer",
        reasoning_steps(question),
        "Keep the rationale under 300 words. You have no tools and cannot search, so never say that "
        "you searched or checked anything.",
        "",
        answer_format(question),
    ]
    return "\n".join(sections)


# ------------------------------------------------------------ parsing

_PROBABILITY = re.compile(r"probability\s*[:=]\s*(\d{1,3}(?:\.\d+)?)\s*%", re.IGNORECASE)
_PERCENTILE = re.compile(r"percentile\s*(\d{1,2})\s*[:=]\s*(-?\d[\d,]*(?:\.\d+)?)", re.IGNORECASE)
_DATE_PERCENTILE = re.compile(
    r"percentile\s*(\d{1,2})\s*[:=]\s*(\d{4})-(\d{2})-(\d{2})(?:[T ](\d{2}):(\d{2})(?::(\d{2}))?)?",
    re.IGNORECASE,
)


def parse_binary(text: str) -> float | None:
    matches = _PROBABILITY.findall(text)
    if not matches:
        return None
    value = float(matches[-1])
    if not 0.0 <= value <= 100.0:
        return None
    return value / 100.0


def parse_multiple_choice(text: str, options: Sequence[str]) -> dict[str, float] | None:
    found: dict[str, float] = {}
    for option in options:
        pattern = re.compile(
            rf"^[ \t]*(?:[-*][ \t]*)?(?:option[ \t]*)?\**{re.escape(option)}\**[ \t]*[:=][ \t]*"
            rf"(\d{{1,3}}(?:\.\d+)?)[ \t]*%",
            re.IGNORECASE | re.MULTILINE,
        )
        matches = pattern.findall(text)
        if not matches:
            return None
        found[option] = float(matches[-1])
    total = sum(found.values())
    if not 90.0 <= total <= 110.0:
        return None
    return {option: value / total for option, value in found.items()}


def _complete_and_increasing(found: dict[float, float]) -> dict[float, float] | None:
    if len(found) != len(PERCENTILE_LABELS):
        return None
    values = [found[label / 100] for label in PERCENTILE_LABELS]
    if any(later < earlier for earlier, later in zip(values, values[1:])):
        return None
    return found


def parse_percentiles(text: str) -> dict[float, float] | None:
    found: dict[float, float] = {}
    for label, raw in _PERCENTILE.findall(text):
        level = int(label)
        if level in PERCENTILE_LABELS:
            found[level / 100] = float(raw.replace(",", ""))
    return _complete_and_increasing(found)


def parse_date_percentiles(text: str) -> dict[float, float] | None:
    """Percentile dates as UTC timestamps, the axis date questions are forecast on."""
    found: dict[float, float] = {}
    for label, year, month, day, hour, minute, second in _DATE_PERCENTILE.findall(text):
        level = int(label)
        if level not in PERCENTILE_LABELS:
            continue
        try:
            moment = datetime(
                int(year), int(month), int(day), int(hour or 0), int(minute or 0), int(second or 0), tzinfo=timezone.utc
            )
        except ValueError:
            return None
        found[level / 100] = moment.timestamp()
    return _complete_and_increasing(found)


# ------------------------------------------------------------- runner


def _short(model: str) -> str:
    return model.split("/")[-1]


def _affordable(ledger: CostLedger | None, model: str, prompt: str) -> bool:
    return ledger is None or ledger.can_afford(model, estimate_tokens(prompt), ESTIMATED_OUTPUT_TOKENS)


async def run_analyst(
    spec: AnalystSpec,
    prompt: str,
    call: LlmCall,
    parse: Callable[[str], object | None],
    timeout: float,
    ledger: CostLedger | None = None,
    fallback: FallbackParse | None = None,
    fallback_model: str | None = None,
) -> AnalystResult:
    """One analyst answer. A failed call is retried once on the stand-in model, if there is one;
    a timeout is not, because the time is gone."""
    started = time.monotonic()
    model = spec.model
    note = None
    try:
        text = await asyncio.wait_for(call(spec.model, prompt), timeout=timeout)
    except asyncio.TimeoutError:
        return AnalystResult(spec.name, spec.model, None, "", error=f"timed out after {timeout:.0f}s")
    except Exception as error:
        if fallback_model is None or not _affordable(ledger, fallback_model, prompt):
            return AnalystResult(spec.name, spec.model, None, "", error=f"model call failed: {error}")
        remaining = max(MIN_FALLBACK_SECONDS, timeout - (time.monotonic() - started))
        try:
            text = await asyncio.wait_for(call(fallback_model, prompt), timeout=remaining)
        except Exception as second:
            detail = str(second) or type(second).__name__
            return AnalystResult(
                spec.name, spec.model, None, "",
                error=f"model call failed: {error}; {_short(fallback_model)} also failed: {detail}",
            )
        model = fallback_model
        note = f"{_short(spec.model)} failed ({error}), so {_short(fallback_model)} answered instead"

    if ledger is not None:
        output_tokens = int(estimate_tokens(text) * REASONING_OUTPUT_MULTIPLIER)
        ledger.record(spec.name, model, estimate_tokens(prompt), output_tokens)

    value = parse(text)
    if value is None and fallback is not None:
        try:
            value = await fallback(text)
        except Exception as error:
            return AnalystResult(spec.name, model, None, text, error=f"could not parse forecast: {error}", note=note)
    if value is None:
        return AnalystResult(spec.name, model, None, text, error="could not parse forecast", note=note)
    return AnalystResult(spec.name, model, value, text, note=note)
