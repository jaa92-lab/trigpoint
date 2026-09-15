"""Evidence containers and how they are rendered into analyst prompts.

Truncation is always announced in the rendered text. A prompt that silently
loses evidence is worse than a shorter prompt, because the analyst cannot know
what it did not see.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

MAX_ITEM_CHARS = 1500

SOURCE_HEADINGS = {
    "news": "News coverage",
    "markets": "Prediction markets on related questions",
    "sources": "Resolution source pages",
    "weather": "Weather model forecasts",
}
# Sources only some question kinds gather. When empty they are left out of the
# prompt, instead of printing "Nothing was found" on every other question.
OPTIONAL_SOURCES = frozenset({"weather"})


@dataclass(frozen=True)
class EvidenceItem:
    source: str
    title: str
    text: str
    url: str | None = None
    published_at: datetime | None = None

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["published_at"] = self.published_at.isoformat() if self.published_at else None
        return data

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> EvidenceItem:
        published = data.get("published_at")
        return cls(
            source=data["source"],
            title=data["title"],
            text=data["text"],
            url=data.get("url"),
            published_at=datetime.fromisoformat(published) if published else None,
        )


@dataclass(frozen=True)
class PriceSnapshot:
    symbol: str
    provider: str
    as_of: datetime
    last_price: float
    sigma_per_day: float | None
    change_1d: float | None = None
    change_7d: float | None = None
    currency: str | None = None
    note: str | None = None


@dataclass
class DataPrior:
    probability: float | None = None
    quantiles: dict[float, float] | None = None
    explanation: str = ""


@dataclass
class EvidenceBundle:
    items: list[EvidenceItem] = field(default_factory=list)
    prices: list[PriceSnapshot] = field(default_factory=list)
    prior: DataPrior | None = None
    notes: list[str] = field(default_factory=list)

    def add(self, *items: EvidenceItem) -> None:
        self.items.extend(items)

    def by_source(self, *sources: str) -> list[EvidenceItem]:
        return [item for item in self.items if item.source in sources]

    def urls(self, *sources: str) -> list[str]:
        return [item.url for item in self.by_source(*sources) if item.url]

    def reference_price(self) -> float | None:
        return self.prices[0].last_price if self.prices else None

    def render(
        self,
        sources: tuple[str, ...],
        include_prices: bool = False,
        include_prior: bool = False,
        max_chars: int = 14000,
    ) -> str:
        parts: list[str] = []
        if include_prices and self.prices:
            parts.append(render_prices(self.prices))
        if include_prior and self.prior and self.prior.explanation:
            parts.append("## Statistical model\n" + self.prior.explanation)
        for source in sources:
            heading = SOURCE_HEADINGS.get(source, source.title())
            items = self.by_source(source)
            if not items:
                if source not in OPTIONAL_SOURCES:
                    parts.append(f"## {heading}\nNothing was found.")
                continue
            parts.append("\n\n".join([f"## {heading}"] + [render_item(i) for i in items]))
        if not parts:
            return "No evidence is provided for this analysis."
        text = "\n\n".join(parts)
        if len(text) > max_chars:
            omitted = len(text) - max_chars
            text = text[:max_chars] + f"\n\n[Evidence truncated: {omitted} more characters were not shown.]"
        return text


def render_item(item: EvidenceItem) -> str:
    date = f" ({item.published_at:%Y-%m-%d})" if item.published_at else ""
    link = f" <{item.url}>" if item.url else ""
    body = item.text.strip()
    if len(body) > MAX_ITEM_CHARS:
        body = body[:MAX_ITEM_CHARS] + " [item truncated]"
    return f"### {item.title}{date}{link}\n{body}"


def render_prices(prices: list[PriceSnapshot]) -> str:
    lines = ["## Price data"]
    for p in prices:
        currency = f" {p.currency}" if p.currency else ""
        bits = [f"last price {p.last_price:,.6g}{currency}"]
        if p.change_1d is not None:
            bits.append(f"24-hour change {p.change_1d:+.1%}")
        if p.change_7d is not None:
            bits.append(f"7-day change {p.change_7d:+.1%}")
        if p.sigma_per_day is not None:
            bits.append(f"typical daily move {p.sigma_per_day:.1%}")
        lines.append(f"- {p.symbol} via {p.provider}, as of {p.as_of:%Y-%m-%d %H:%M} UTC: " + ", ".join(bits))
        if p.note:
            lines.append(f"  Note: {p.note}")
    return "\n".join(lines)
