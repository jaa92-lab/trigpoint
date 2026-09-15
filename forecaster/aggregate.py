"""Combining analyst forecasts.

Binary forecasts are pooled as a weighted mean in log-odds space, which matches
how log scoring rewards and punishes confidence. Multiple choice uses log-linear
pooling. Numeric forecasts average quantiles level by level. Everything leaving
this module already satisfies the forecasting-tools validation rules, so the
framework never has to renormalize (it raises if its own adjustment is large).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

EPS = 1e-6


def logit(p: float) -> float:
    p = min(max(p, EPS), 1.0 - EPS)
    return math.log(p / (1.0 - p))


def expit(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def normalize_weights(count: int, weights: Sequence[float] | None) -> list[float]:
    if count <= 0:
        raise ValueError("Nothing to pool")
    if weights is None:
        return [1.0 / count] * count
    if len(weights) != count:
        raise ValueError("One weight per forecast is required")
    clipped = [max(w, 0.0) for w in weights]
    total = sum(clipped)
    if total <= 0:
        raise ValueError("Weights must have a positive sum")
    return [w / total for w in clipped]


# ---------------------------------------------------------------- binary


def pool_binary(
    probabilities: Sequence[float],
    weights: Sequence[float] | None = None,
    extremize: float = 1.0,
) -> float:
    w = normalize_weights(len(probabilities), weights)
    pooled = sum(wi * logit(p) for wi, p in zip(w, probabilities))
    return expit(extremize * pooled)


def logit_spread(probabilities: Sequence[float]) -> float:
    if len(probabilities) < 2:
        return 0.0
    logits = [logit(p) for p in probabilities]
    return max(logits) - min(logits)


def shrink_toward(probability: float, anchor: float | None, weight: float) -> float:
    if anchor is None or weight <= 0:
        return probability
    weight = min(weight, 1.0)
    return expit((1.0 - weight) * logit(probability) + weight * logit(anchor))


def clamp_probability(probability: float, floor: float = 0.02, ceiling: float = 0.98) -> float:
    if not 0.0 < floor < ceiling < 1.0:
        raise ValueError("Need 0 < floor < ceiling < 1")
    return min(max(probability, floor), ceiling)


# ------------------------------------------------------- multiple choice


def apply_floor(probabilities: Mapping[str, float], floor: float) -> dict[str, float]:
    """Lift options below the floor and take that mass proportionally from the rest."""
    if not probabilities:
        raise ValueError("No options")
    if floor * len(probabilities) >= 1.0:
        raise ValueError("Floor is too high for this many options")
    total = sum(max(v, 0.0) for v in probabilities.values())
    if total <= 0:
        probs = {k: 1.0 / len(probabilities) for k in probabilities}
    else:
        probs = {k: max(v, 0.0) / total for k, v in probabilities.items()}

    fixed: set[str] = set()
    for _ in range(len(probs)):
        newly_low = {k for k, v in probs.items() if k not in fixed and v < floor}
        if not newly_low:
            break
        fixed |= newly_low
        free = [k for k in probs if k not in fixed]
        free_mass = 1.0 - floor * len(fixed)
        free_total = sum(probs[k] for k in free)
        for k in fixed:
            probs[k] = floor
        for k in free:
            if free_total > 0:
                probs[k] = probs[k] * free_mass / free_total
            else:
                probs[k] = free_mass / len(free)

    drift = 1.0 - sum(probs.values())
    largest = max(probs, key=lambda k: probs[k])
    probs[largest] += drift
    return probs


def pool_multiple_choice(
    distributions: Sequence[Mapping[str, float]],
    options: Sequence[str],
    weights: Sequence[float] | None = None,
    floor: float = 0.01,
) -> dict[str, float]:
    w = normalize_weights(len(distributions), weights)
    log_pool: dict[str, float] = {}
    for option in options:
        acc = 0.0
        for wi, dist in zip(w, distributions):
            total = sum(max(dist.get(o, 0.0), 0.0) for o in options) or 1.0
            p = max(dist.get(option, 0.0), 0.0) / total
            acc += wi * math.log(max(p, EPS))
        log_pool[option] = acc
    top = max(log_pool.values())
    raw = {o: math.exp(v - top) for o, v in log_pool.items()}
    z = sum(raw.values())
    return apply_floor({o: v / z for o, v in raw.items()}, floor)


# --------------------------------------------------------------- numeric


def pool_quantiles(
    analyst_quantiles: Sequence[Mapping[float, float]],
    weights: Sequence[float] | None = None,
) -> dict[float, float]:
    """Average each shared percentile level across analysts (Vincentization)."""
    if not analyst_quantiles:
        raise ValueError("Nothing to pool")
    shared = set(analyst_quantiles[0])
    for q in analyst_quantiles[1:]:
        shared &= set(q)
    levels = sorted(shared)
    if len(levels) < 2:
        raise ValueError("Analysts must share at least two percentile levels")
    w = normalize_weights(len(analyst_quantiles), weights)
    return {lvl: sum(wi * q[lvl] for wi, q in zip(w, analyst_quantiles)) for lvl in levels}


def median_of(quantiles: Mapping[float, float]) -> float:
    if 0.5 in quantiles:
        return quantiles[0.5]
    levels = sorted(quantiles)
    below = [lvl for lvl in levels if lvl < 0.5]
    above = [lvl for lvl in levels if lvl > 0.5]
    if not below or not above:
        return quantiles[levels[len(levels) // 2]]
    lo, hi = below[-1], above[0]
    t = (0.5 - lo) / (hi - lo)
    return quantiles[lo] + t * (quantiles[hi] - quantiles[lo])


def widen_tails(quantiles: Mapping[float, float], factor: float = 1.15) -> dict[float, float]:
    """Stretch values away from the median, fully at the 10th and 90th percentiles."""
    if factor < 1.0:
        raise ValueError("Widening factor must be at least 1")
    med = median_of(quantiles)
    out = {}
    for lvl, value in quantiles.items():
        strength = min(abs(lvl - 0.5) / 0.4, 1.0)
        out[lvl] = med + (value - med) * (1.0 + (factor - 1.0) * strength)
    return out


def fit_to_bounds(
    quantiles: Mapping[float, float],
    lower: float,
    upper: float,
    open_lower: bool,
    open_upper: bool,
    zero_point: float | None = None,
) -> dict[float, float]:
    span = upper - lower
    lo = lower - 0.2 * span if open_lower else lower
    hi = upper + 0.2 * span if open_upper else upper
    if zero_point is not None:
        lo = max(lo, zero_point + abs(span) * 1e-6)
    return {lvl: min(max(v, lo), hi) for lvl, v in quantiles.items()}


def enforce_increasing(quantiles: Mapping[float, float]) -> dict[float, float]:
    levels = sorted(quantiles)
    values = [quantiles[lvl] for lvl in levels]
    span = max(values) - min(values)
    gap = max(abs(span) * 1e-6, 1e-9)
    for i in range(1, len(values)):
        if values[i] <= values[i - 1]:
            values[i] = values[i - 1] + gap
    return dict(zip(levels, values))
