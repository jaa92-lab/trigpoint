"""One source of "now" for the whole pipeline.

Live runs use the real clock. Pastcast runs pin the clock to the moment the bot
would have seen a question, and every evidence fetcher must respect it. Any code
that reads datetime.now() directly instead of Clock.now() is a leakage bug.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class Clock:
    as_of: datetime | None = None

    @classmethod
    def live(cls) -> Clock:
        return cls(None)

    @classmethod
    def pinned(cls, as_of: datetime) -> Clock:
        if as_of.tzinfo is None:
            raise ValueError("A pinned clock needs a timezone-aware datetime")
        return cls(as_of.astimezone(timezone.utc))

    @property
    def is_live(self) -> bool:
        return self.as_of is None

    def now(self) -> datetime:
        if self.as_of is None:
            return datetime.now(timezone.utc)
        return self.as_of
