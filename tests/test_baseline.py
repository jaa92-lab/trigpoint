import pytest
from forecasting_tools.forecast_bots.official_bots.template_bot_2026_fall import FallTemplateBot2026

from forecaster.baseline import BASELINE_TOKEN_ENV, build_baseline_bot
from forecaster.config import LEAD_MODEL


def test_baseline_bot_needs_its_own_token(monkeypatch):
    monkeypatch.delenv(BASELINE_TOKEN_ENV, raising=False)
    with pytest.raises(RuntimeError, match="secondary bot"):
        build_baseline_bot(publish=False)


def test_baseline_bot_uses_the_template_with_the_lead_model(monkeypatch):
    monkeypatch.setenv(BASELINE_TOKEN_ENV, "secondary-token")
    for name in ("ASKNEWS_CLIENT_ID", "ASKNEWS_SECRET", "ASKNEWS_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    bot = build_baseline_bot(publish=False)
    assert isinstance(bot, FallTemplateBot2026)
    assert bot.metaculus_client.token == "secondary-token"
    assert bot.get_llm("default", "llm").model == LEAD_MODEL
    assert bot.get_llm("researcher") == "no_research"
    assert not bot.publish_reports_to_metaculus
