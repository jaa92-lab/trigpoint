"""Scores for evaluating forecasts against resolutions.

Binary and multiple-choice formulas follow Metaculus's scoring FAQ: log score,
baseline score (100 = perfect, 0 = chance), and a peer-style comparison against
a reference forecast such as the community prediction (100 x log-score gap).

Numeric scoring is an approximation. It reads a density off a 201-point CDF on a
linear axis and mixes in a small uniform floor, as Metaculus does. Use it to
compare bot versions with each other, not to predict exact leaderboard points.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean, stdev


def _clip(p: float) -> float:
    return min(max(p, 1e-9), 1.0 - 1e-9)


def binary_log_score(p_yes: float, outcome: bool) -> float:
    return math.log(_clip(p_yes if outcome else 1.0 - p_yes))


def binary_baseline_score(p_yes: float, outcome: bool) -> float:
    return 100.0 * (math.log2(_clip(p_yes if outcome else 1.0 - p_yes)) + 1.0)


def binary_relative_score(p_yes: float, reference_yes: float, outcome: bool) -> float:
    return 100.0 * (binary_log_score(p_yes, outcome) - binary_log_score(reference_yes, outcome))


def mc_log_score(probabilities: Mapping[str, float], outcome: str) -> float:
    return math.log(_clip(probabilities.get(outcome, 0.0)))


def mc_baseline_score(probabilities: Mapping[str, float], outcome: str) -> float:
    n = len(probabilities)
    if n < 2:
        raise ValueError("Need at least two options")
    return 100.0 * (mc_log_score(probabilities, outcome) / math.log(n) + 1.0)


def mc_relative_score(
    probabilities: Mapping[str, float], reference: Mapping[str, float], outcome: str
) -> float:
    return 100.0 * (mc_log_score(probabilities, outcome) - mc_log_score(reference, outcome))


def numeric_log_density(cdf: Sequence[float], lower: float, upper: float, outcome: float) -> float:
    n = len(cdf)
    if n < 2 or upper <= lower:
        raise ValueError("Need a CDF with at least two points and upper > lower")
    width = (upper - lower) / (n - 1)
    if outcome < lower:
        mass = cdf[0]
    elif outcome > upper:
        mass = 1.0 - cdf[-1]
    else:
        i = min(int((outcome - lower) / width), n - 2)
        mass = cdf[i + 1] - cdf[i]
    density = max(mass, 0.0) / width
    uniform = 1.0 / (upper - lower)
    return math.log(max(0.99 * density + 0.01 * uniform, 1e-12))


def numeric_relative_score(
    cdf: Sequence[float], reference_cdf: Sequence[float], lower: float, upper: float, outcome: float
) -> float:
    return 100.0 * (
        numeric_log_density(cdf, lower, upper, outcome)
        - numeric_log_density(reference_cdf, lower, upper, outcome)
    )


@dataclass(frozen=True)
class ScoreSummary:
    count: int
    mean: float
    stderr: float

    def beats(self, other: ScoreSummary, noise_floor: float = 0.0) -> bool:
        """True only if the gap clears both sampling error and the measured noise floor."""
        combined = math.sqrt(self.stderr**2 + other.stderr**2)
        return (self.mean - other.mean) > max(2.0 * combined, noise_floor)


def summarize(values: Sequence[float]) -> ScoreSummary:
    if not values:
        return ScoreSummary(0, float("nan"), float("nan"))
    if len(values) == 1:
        return ScoreSummary(1, values[0], float("inf"))
    return ScoreSummary(len(values), fmean(values), stdev(values) / math.sqrt(len(values)))
