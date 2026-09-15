"""Pure panel logic: when to escalate, how to combine, and how to explain.

Kept free of network and framework code so every rule that decides a forecast
can be tested directly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from forecaster import aggregate as agg
from forecaster.analysts import AnalystResult
from forecaster.config import BotConfig

MAX_RATIONALE_CHARS = 1500


class NoForecastError(RuntimeError):
    """No analyst produced a usable forecast and no statistical fallback exists."""


def successful(results: Sequence[AnalystResult]) -> list[AnalystResult]:
    return [r for r in results if r.ok]


def error_summary(results: Sequence[AnalystResult]) -> str:
    errors = [f"{r.name}: {r.error}" for r in results if r.error]
    return "; ".join(errors) if errors else "no analysts ran"


def should_escalate(results: Sequence[AnalystResult], question_type: str, config: BotConfig) -> bool:
    if question_type != "binary" and config.full_panel_for_non_binary:
        return True
    ok = successful(results)
    if len(ok) < 2:
        return True
    if any(r.flags for r in ok):
        return True
    return agg.logit_spread([float(r.value) for r in ok]) >= config.escalate_on_logit_spread


def escalation_reason(results: Sequence[AnalystResult], question_type: str, config: BotConfig) -> str:
    if question_type != "binary" and config.full_panel_for_non_binary:
        return "non-binary questions always use the full panel"
    ok = successful(results)
    if len(ok) < 2:
        return "an analyst failed"
    if any(r.flags for r in ok):
        return "a fact check flagged an analyst"
    return "the analysts disagreed"


def combine_binary(
    results: Sequence[AnalystResult], prior: float | None, config: BotConfig
) -> tuple[float, float | None]:
    """Returns (final forecast, pooled analyst forecast or None if every analyst failed)."""
    ok = successful(results)
    if not ok:
        if prior is None:
            raise NoForecastError(error_summary(results))
        return agg.clamp_probability(prior, config.binary_floor, config.binary_ceiling), None
    pooled = agg.pool_binary([float(r.value) for r in ok], [r.weight for r in ok], config.extremize)
    anchored = agg.shrink_toward(pooled, prior, config.data_prior_weight)
    return agg.clamp_probability(anchored, config.binary_floor, config.binary_ceiling), pooled


def combine_multiple_choice(
    results: Sequence[AnalystResult], options: Sequence[str], config: BotConfig
) -> dict[str, float]:
    ok = successful(results)
    if not ok:
        raise NoForecastError(error_summary(results))
    return agg.pool_multiple_choice([r.value for r in ok], options, [r.weight for r in ok], config.mc_floor)


def _prior_weight(analyst_weight_total: float, share: float) -> float:
    share = min(max(share, 0.0), 0.95)
    if analyst_weight_total <= 0:
        return 1.0
    return share * analyst_weight_total / (1.0 - share)


def combine_numeric(
    results: Sequence[AnalystResult],
    prior_quantiles: Mapping[float, float] | None,
    config: BotConfig,
    lower: float,
    upper: float,
    open_lower: bool,
    open_upper: bool,
    zero_point: float | None = None,
) -> dict[float, float]:
    ok = successful(results)
    sets = [dict(r.value) for r in ok]
    weights = [r.weight for r in ok]
    if prior_quantiles and config.data_prior_weight > 0:
        sets.append(dict(prior_quantiles))
        weights.append(_prior_weight(sum(weights), config.data_prior_weight))
    if not sets:
        raise NoForecastError(error_summary(results))
    pooled = agg.pool_quantiles(sets, weights)
    widened = agg.widen_tails(pooled, config.numeric_tail_widening)
    fitted = agg.fit_to_bounds(widened, lower, upper, open_lower, open_upper, zero_point)
    return agg.enforce_increasing(fitted)


def format_value(value: object, question_type: str) -> str:
    if question_type == "binary":
        return f"{float(value):.1%}"
    if question_type == "multiple_choice":
        return ", ".join(f"{option} {p:.0%}" for option, p in value.items())
    return ", ".join(f"P{round(level * 100)} {v:,.4g}" for level, v in sorted(value.items()))


def _trim(text: str) -> str:
    text = text.strip()
    if len(text) <= MAX_RATIONALE_CHARS:
        return text
    return text[:MAX_RATIONALE_CHARS] + " [rationale truncated]"


def render_panel_report(
    *,
    kind: str,
    question_type: str,
    stage: str,
    results: Sequence[AnalystResult],
    final_text: str,
    anchor_text: str | None,
    notes: Sequence[str],
    spend: float,
) -> str:
    lines = [
        "# Panel forecast",
        f"Final forecast: {final_text}",
        f"Question kind: {kind}. {stage}",
    ]
    if anchor_text:
        lines.append(anchor_text)
    lines += [f"Estimated analyst spend: ${spend:.2f}", "", "## Analysts"]
    for r in results:
        header = f"### {r.name} ({r.model.split('/')[-1]})"
        if r.ok:
            header += f": {format_value(r.value, question_type)}"
            if r.weight < 1.0:
                header += f" (weight {r.weight:.2f})"
        else:
            header += f": no forecast ({r.error})"
        lines.append(header)
        lines += [f"- Flag: {flag}" for flag in r.flags]
        if r.rationale:
            lines.append(_trim(r.rationale))
        lines.append("")
    if notes:
        lines.append("## Evidence notes")
        lines += [f"- {note}" for note in notes]
    return "\n".join(lines).strip()
