"""Bot configuration: models, the analyst roster, budgets, and feature switches.

Every behavior the pastcast harness might switch off is a field here, so an
experiment is simply a different BotConfig.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from forecaster.budget import ModelPrice

LEAD_MODEL = "openrouter/anthropic/claude-opus-5"
SECOND_MODEL = "openrouter/openai/gpt-5.6-sol"
THIRD_MODEL = "openrouter/google/gemini-3.6-flash"
PARSER_MODEL = "openrouter/openai/gpt-5.6-luna"

# Dollars per million tokens, from OpenRouter's model list on 2026-09-14.
MODEL_PRICES: dict[str, ModelPrice] = {
    LEAD_MODEL: ModelPrice(5.00, 25.00),
    SECOND_MODEL: ModelPrice(2.00, 10.00),
    THIRD_MODEL: ModelPrice(0.75, 3.75),
    PARSER_MODEL: ModelPrice(0.20, 1.20),
}


@dataclass(frozen=True)
class AnalystSpec:
    name: str
    model: str
    evidence: tuple[str, ...]
    include_prices: bool
    include_prior: bool
    brief: str


ANALYSTS: tuple[AnalystSpec, ...] = (
    AnalystSpec(
        name="news",
        model=LEAD_MODEL,
        evidence=("news",),
        include_prices=False,
        include_prior=False,
        brief=(
            "You see only recent news coverage. Establish what has actually happened, what is "
            "scheduled, and how that bears on the resolution criteria."
        ),
    ),
    AnalystSpec(
        name="data",
        model=SECOND_MODEL,
        evidence=("markets", "weather"),
        include_prices=True,
        include_prior=True,
        brief=(
            "You see only quantitative evidence: price data, a statistical model where one "
            "applies, prediction-market prices for related questions, and weather model forecasts "
            "for weather questions. Anchor on the numbers, and note where a related market asks a "
            "slightly different question."
        ),
    ),
    AnalystSpec(
        name="sources",
        model=THIRD_MODEL,
        evidence=("sources",),
        include_prices=False,
        include_prior=False,
        brief=(
            "You see only the pages named in the resolution criteria. Establish what those "
            "sources show now and exactly what would have to change for the question to resolve Yes."
        ),
    ),
    AnalystSpec(
        name="outside_view",
        model=LEAD_MODEL,
        evidence=(),
        include_prices=False,
        include_prior=False,
        brief=(
            "You see no current evidence, on purpose. Reason from base rates: how often do events "
            "like this happen within a window this long, and what is the status quo outcome?"
        ),
    ),
)

# Which two analysts run first for each triage kind. The rest join on escalation.
STAGE_ONE: dict[str, tuple[str, str]] = {
    "price_threshold": ("data", "news"),
    "price_level": ("data", "news"),
    "weather": ("data", "sources"),
    "official_count": ("sources", "news"),
    "sports": ("data", "news"),
    "event": ("news", "outside_view"),
    "generic": ("news", "outside_view"),
}

# Weather questions lead with the data analyst only when a forecast was found.
STAGE_ONE_WITHOUT_WEATHER: tuple[str, str] = ("sources", "news")


@dataclass(frozen=True)
class BotConfig:
    analysts: tuple[AnalystSpec, ...] = ANALYSTS
    stage_one: dict[str, tuple[str, str]] = field(default_factory=lambda: dict(STAGE_ONE))
    stage_one_without_weather: tuple[str, str] = STAGE_ONE_WITHOUT_WEATHER
    parser_model: str = PARSER_MODEL
    model_prices: dict[str, ModelPrice] = field(default_factory=lambda: dict(MODEL_PRICES))

    # Escalation from two analysts to the full panel
    escalate_on_logit_spread: float = 0.8
    full_panel_for_non_binary: bool = True
    # When stage one agrees with fewer answers than this, each of its analysts answers
    # again, and disagreement among the answers still escalates. 0 switches this off.
    min_answers: int = 4

    # Combining
    binary_floor: float = 0.02
    binary_ceiling: float = 0.98
    mc_floor: float = 0.01
    extremize: float = 1.0
    numeric_tail_widening: float = 1.15
    data_prior_weight: float = 0.35
    price_tail_factor: float = 1.25

    # Evidence switches
    use_news: bool = True
    use_markets: bool = True
    use_prices: bool = True
    use_sources: bool = True
    use_weather: bool = True
    use_grounding: bool = True
    max_source_urls: int = 3

    # Budgets
    max_cost_per_question: float = 1.00
    max_seconds_per_question: float = 1500.0
    safety_margin_seconds: float = 300.0
    skip_if_closing_within_seconds: float = 180.0
    fast_path_below_seconds: float = 420.0

    def analyst(self, name: str) -> AnalystSpec:
        for spec in self.analysts:
            if spec.name == name:
                return spec
        raise KeyError(f"No analyst named {name!r}")

    def with_changes(self, **changes: object) -> BotConfig:
        return replace(self, **changes)


def register_model_prices(prices: dict[str, ModelPrice]) -> None:
    """Teach litellm the prices of new models so framework cost tracking works."""
    import litellm

    litellm.register_model(
        {
            model: {
                "input_cost_per_token": price.input_per_million / 1_000_000,
                "output_cost_per_token": price.output_per_million / 1_000_000,
                "litellm_provider": "openrouter",
                "mode": "chat",
            }
            for model, price in prices.items()
        }
    )
