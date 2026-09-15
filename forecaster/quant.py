"""Closed-form probability estimates for asset-price questions.

Short-horizon price questions ("will X trade above K on Friday?", "what will X
close at?") are dominated by one quantity: how volatile the asset is relative to
the distance to the threshold. A driftless lognormal model gets that right more
reliably than an LLM reading headlines, so it serves as a data prior that
analysts see and the pooled forecast can be shrunk toward.

tail_factor > 1 widens the distribution to allow for fat tails.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from statistics import NormalDist

_NORMAL = NormalDist()


def realized_sigma_per_day(points: Sequence[tuple[float, float]]) -> float | None:
    """Volatility of log returns per sqrt(day) from (unix_seconds, price) points."""
    clean = sorted((t, p) for t, p in points if p is not None and p > 0)
    scaled: list[float] = []
    for (t0, p0), (t1, p1) in zip(clean, clean[1:]):
        dt_days = (t1 - t0) / 86400.0
        if dt_days <= 0:
            continue
        scaled.append(math.log(p1 / p0) / math.sqrt(dt_days))
    if len(scaled) < 4:
        return None
    mean = sum(scaled) / len(scaled)
    variance = sum((x - mean) ** 2 for x in scaled) / (len(scaled) - 1)
    return math.sqrt(variance)


def _horizon_sigma(sigma_day: float, days: float, tail_factor: float) -> float:
    return sigma_day * tail_factor * math.sqrt(days)


def prob_above_at(
    spot: float, strike: float, sigma_day: float, days: float, tail_factor: float = 1.0
) -> float:
    """P(price at the horizon > strike) under a driftless lognormal model."""
    if strike <= 0:
        return 1.0
    if days <= 0 or sigma_day <= 0:
        return 1.0 if spot > strike else 0.0
    s = _horizon_sigma(sigma_day, days, tail_factor)
    return _NORMAL.cdf((math.log(spot / strike) - 0.5 * s * s) / s)


def prob_touch_above(
    spot: float, strike: float, sigma_day: float, days: float, tail_factor: float = 1.0
) -> float:
    """P(price reaches strike at any time within the horizon), reflection principle."""
    if spot >= strike:
        return 1.0
    if days <= 0 or sigma_day <= 0:
        return 0.0
    s = _horizon_sigma(sigma_day, days, tail_factor)
    return min(1.0, 2.0 * _NORMAL.cdf(math.log(spot / strike) / s))


def prob_touch_below(
    spot: float, strike: float, sigma_day: float, days: float, tail_factor: float = 1.0
) -> float:
    """P(price falls to strike at any time within the horizon)."""
    if spot <= strike:
        return 1.0
    if days <= 0 or sigma_day <= 0:
        return 0.0
    s = _horizon_sigma(sigma_day, days, tail_factor)
    return min(1.0, 2.0 * _NORMAL.cdf(math.log(strike / spot) / s))


def prob_touch_in_window(
    spot: float,
    strike: float,
    sigma_day: float,
    start_days: float,
    end_days: float,
    direction: str = "above",
    tail_factor: float = 1.0,
    grid: int = 400,
) -> float:
    """P(price is beyond strike at some moment between start_days and end_days from now).

    If the window has not started, today's price being beyond the strike proves
    nothing: the price at the window's start is uncertain, so this integrates the
    touch probability over the lognormal distribution of that starting price.
    """
    touch = prob_touch_above if direction == "above" else prob_touch_below
    beyond = spot >= strike if direction == "above" else spot <= strike
    if end_days <= 0:
        return 1.0 if beyond else 0.0
    start = min(max(start_days, 0.0), end_days)
    if start == 0.0:
        return touch(spot, strike, sigma_day, end_days, tail_factor)
    if sigma_day <= 0:
        return 1.0 if beyond else 0.0
    s1 = _horizon_sigma(sigma_day, start, tail_factor)
    window = end_days - start
    total = 0.0
    weight_sum = 0.0
    for i in range(grid):
        z = -6.0 + 12.0 * (i + 0.5) / grid
        weight = math.exp(-0.5 * z * z)
        level = spot * math.exp(-0.5 * s1 * s1 + s1 * z)
        total += weight * touch(level, strike, sigma_day, window, tail_factor)
        weight_sum += weight
    return total / weight_sum


def lognormal_quantiles(
    spot: float,
    sigma_day: float,
    days: float,
    levels: Iterable[float],
    tail_factor: float = 1.0,
) -> dict[float, float]:
    """Price quantiles at the horizon; the mean stays at spot (driftless)."""
    s = _horizon_sigma(sigma_day, max(days, 1e-6), tail_factor)
    return {
        level: spot * math.exp(-0.5 * s * s + s * _NORMAL.inv_cdf(level)) for level in levels
    }
