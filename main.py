"""Command-line entry point for the panel forecasting bot.

    python main.py --mode tournament
        Forecast new MiniBench and seasonal-tournament questions, and publish.
    python main.py --mode test_questions
        Forecast the bot-testing area and publish there. Use this to check setup.
    python main.py --mode dry_run --url https://www.metaculus.com/questions/12345/
        Forecast specific questions and print the reasoning without publishing.
    python main.py --mode baseline
        Run Metaculus's template as the secondary comparison bot (needs METACULUS_BASELINE_TOKEN).

Every mode first checks the OpenRouter credit balance (see forecaster/credits.py)
and logs what the run actually spent.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

import dotenv

EXPECTED_SKIPS = ("closes too soon", "not supported")


def configure_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("LiteLLM", "httpx", "httpcore", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the panel forecasting bot")
    parser.add_argument(
        "--mode", choices=["tournament", "test_questions", "dry_run", "baseline"], default="tournament"
    )
    parser.add_argument("--url", action="append", default=[], help="Question URL for dry_run (repeatable)")
    return parser


def is_expected_skip(error: BaseException) -> bool:
    return any(phrase in str(error) for phrase in EXPECTED_SKIPS)


async def credit_status():
    """The OpenRouter key's balance, or None when there is no key or it cannot be read."""
    from forecaster.credits import fetch_key_status
    from forecaster.evidence.http import make_client

    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not key:
        return None
    try:
        async with make_client() as http:
            return await fetch_key_status(http, key)
    except Exception as error:
        logging.warning("Could not read the OpenRouter credit balance: %r", error)
        return None


async def log_spend(before, reports) -> None:
    from forecaster.credits import spent_between

    after = await credit_status()
    if before is None or after is None:
        return
    spent = spent_between(before, after)
    if spent is None:
        return
    forecasts = sum(1 for report in reports if not isinstance(report, BaseException))
    per_forecast = f", about ${spent / forecasts:.2f} each" if forecasts else ""
    left = f" ${after.remaining:.2f} remains." if after.remaining is not None else ""
    logging.info(f"This run spent ${spent:.2f} of OpenRouter credit on {forecasts} forecasts{per_forecast}.{left}")


async def run(args: argparse.Namespace) -> int:
    from forecasting_tools import MetaculusClient

    from forecaster.config import BotConfig
    from forecaster.credits import plan_run

    config = BotConfig()
    client = MetaculusClient()
    before = await credit_status()
    plan = plan_run(
        before.remaining if before else None,
        client.CURRENT_MINIBENCH_ID,
        client.CURRENT_AI_COMPETITION_ID,
        config.credit_floor,
        config.minibench_reserve,
    )
    logging.info(plan.reason)
    if not plan.tournaments:
        return 0

    if args.mode == "baseline":
        from forecaster.baseline import build_baseline_bot

        bot = build_baseline_bot(publish=True)
    else:
        from forecaster.bot import PanelForecaster

        bot = PanelForecaster(
            config=config,
            publish_reports_to_metaculus=args.mode in ("tournament", "test_questions"),
            skip_previously_forecasted_questions=args.mode == "tournament",
            extra_metadata_in_explanation=True,
            metaculus_client=client,
        )

    if args.mode in ("tournament", "baseline"):
        reports = []
        for tournament in plan.tournaments:
            reports += await bot.forecast_on_tournament(tournament, return_exceptions=True)
    elif args.mode == "test_questions":
        reports = await bot.forecast_on_tournament("bot-testing-area", return_exceptions=True)
    else:
        if not args.url:
            raise SystemExit("dry_run needs at least one --url")
        questions = [client.get_question_by_url(url) for url in args.url]
        reports = await bot.forecast_questions(questions, return_exceptions=True)
        for report in reports:
            print(report if isinstance(report, BaseException) else report.explanation)

    if reports:
        bot.log_report_summary(reports, raise_errors=False)
    await log_spend(before, reports)
    failures = [r for r in reports if isinstance(r, BaseException) and not is_expected_skip(r)]
    if reports and len(failures) == len(reports):
        logging.error("Every question failed; exiting with an error so the workflow is marked failed.")
        return 1
    return 0


def main() -> None:
    from forecaster.env import clean_secret_env

    dotenv.load_dotenv()
    configure_output()
    cleaned = clean_secret_env()
    if cleaned:
        logging.warning(f"Stripped stray whitespace from {', '.join(cleaned)}. Re-save those secrets to fix them at the source.")
    sys.exit(asyncio.run(run(build_parser().parse_args())))


if __name__ == "__main__":
    main()
