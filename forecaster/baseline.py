"""The comparison bot: Metaculus's own template, run as a labeled secondary bot.

It uses the panel's lead model with the template's single-prompt scaffolding, so
the score gap between the two bots measures what the panel design adds. The
tournament rules require it to run under a separate bot account linked as
secondary, which keeps it out of prize calculations.
"""

from __future__ import annotations

import os

from forecasting_tools import GeneralLlm, MetaculusClient
from forecasting_tools.forecast_bots.official_bots.template_bot_2026_fall import FallTemplateBot2026

from forecaster.config import LEAD_MODEL, MODEL_PRICES, PARSER_MODEL, register_model_prices

BASELINE_TOKEN_ENV = "METACULUS_BASELINE_TOKEN"


def build_baseline_bot(publish: bool) -> FallTemplateBot2026:
    token = os.getenv(BASELINE_TOKEN_ENV)
    if not token:
        raise RuntimeError(
            f"{BASELINE_TOKEN_ENV} is not set. The comparison bot needs its own bot account, "
            "linked as a secondary bot in Metaculus settings."
        )
    register_model_prices(MODEL_PRICES)
    has_news = bool(
        (os.getenv("ASKNEWS_CLIENT_ID") and os.getenv("ASKNEWS_SECRET")) or os.getenv("ASKNEWS_API_KEY")
    )
    return FallTemplateBot2026(
        research_reports_per_question=1,
        predictions_per_research_report=3,
        publish_reports_to_metaculus=publish,
        skip_previously_forecasted_questions=True,
        extra_metadata_in_explanation=True,
        metaculus_client=MetaculusClient(token=token),
        llms={
            "default": GeneralLlm(model=LEAD_MODEL, temperature=None, timeout=300, allowed_tries=2),
            "summarizer": GeneralLlm(model=PARSER_MODEL, temperature=None),
            "researcher": "asknews/news-summaries" if has_news else "no_research",
            "parser": GeneralLlm(model=PARSER_MODEL, temperature=None),
        },
    )
