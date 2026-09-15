"""Time and money budgets for a single question.

Questions stay open about 90 minutes and each accepts one forecast. A question
the bot never answers scores zero, so the time budget protects submission first
and quality second. The cost ledger estimates spend from token counts with a
price table, because litellm does not know the prices of brand-new models.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class TimeBudget:
    total_seconds: float
    monotonic: Callable[[], float] = time.monotonic
    _start: float | None = None

    def __post_init__(self) -> None:
        if self._start is None:
            self._start = self.monotonic()

    @classmethod
    def for_question(
        cls,
        close_time: datetime | None,
        now: datetime,
        max_seconds: float = 1500.0,
        safety_margin_seconds: float = 300.0,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> TimeBudget:
        if close_time is None:
            return cls(total_seconds=max_seconds, monotonic=monotonic)
        until_close = (close_time - now).total_seconds() - safety_margin_seconds
        return cls(total_seconds=max(0.0, min(max_seconds, until_close)), monotonic=monotonic)

    def elapsed(self) -> float:
        return self.monotonic() - (self._start or 0.0)

    def remaining(self) -> float:
        return max(0.0, self.total_seconds - self.elapsed())

    def expired(self) -> bool:
        return self.remaining() <= 0.0

    def slice(self, fraction: float, minimum: float = 5.0, maximum: float | None = None) -> float:
        """A timeout for the next stage: a share of what is left, within limits."""
        left = self.remaining()
        wanted = max(minimum, left * fraction)
        if maximum is not None:
            wanted = min(wanted, maximum)
        return max(0.0, min(wanted, left))


def should_skip(close_time: datetime | None, now: datetime, min_seconds: float = 180.0) -> bool:
    if close_time is None:
        return False
    return (close_time - now).total_seconds() < min_seconds


@dataclass(frozen=True)
class ModelPrice:
    input_per_million: float
    output_per_million: float


CONSERVATIVE_PRICE = ModelPrice(10.0, 50.0)


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


@dataclass
class CostLedger:
    cap: float
    prices: dict[str, ModelPrice]
    spent: float = 0.0
    entries: list[tuple[str, str, float]] = field(default_factory=list)

    def price_for(self, model: str) -> ModelPrice:
        return self.prices.get(model, CONSERVATIVE_PRICE)

    def estimate(self, model: str, input_tokens: int, output_tokens: int) -> float:
        price = self.price_for(model)
        return (
            input_tokens * price.input_per_million + output_tokens * price.output_per_million
        ) / 1_000_000

    def can_afford(self, model: str, input_tokens: int, output_tokens: int) -> bool:
        return self.spent + self.estimate(model, input_tokens, output_tokens) <= self.cap

    def record(self, label: str, model: str, input_tokens: int, output_tokens: int) -> float:
        cost = self.estimate(model, input_tokens, output_tokens)
        self.spent += cost
        self.entries.append((label, model, cost))
        return cost

    @property
    def remaining(self) -> float:
        return max(0.0, self.cap - self.spent)
