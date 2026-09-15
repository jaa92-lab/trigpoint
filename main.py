"""Command-line entry point for the panel forecasting bot.

    python main.py --mode tournament
        Forecast new questions in the seasonal tournament and MiniBench, and publish.
    python main.py --mode test_questions
        Forecast the bot-testing area and publish there. Use this to check setup.
    python main.py --mode dry_run --url https://www.metaculus.com/questions/12345/
        Forecast specific questions and print the reasoning without publishing.
    python main.py --mode baseline
        Run Metaculus's template as the secondary comparison bot (needs METACULUS_BASELINE_TOKEN).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
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


async def run(args: argparse.Namespace) -> int:
    from forecasting_tools import MetaculusClient

    client = MetaculusClient()
    tournaments = (client.CURRENT_AI_COMPETITION_ID, client.CURRENT_MINIBENCH_ID)

    if args.mode == "baseline":
        from forecaster.baseline import build_baseline_bot

        bot = build_baseline_bot(publish=True)
    else:
        from forecaster.bot import PanelForecaster

        bot = PanelForecaster(
            publish_reports_to_metaculus=args.mode in ("tournament", "test_questions"),
            skip_previously_forecasted_questions=args.mode == "tournament",
            extra_metadata_in_explanation=True,
            metaculus_client=client,
        )

    if args.mode in ("tournament", "baseline"):
        reports = []
        for tournament in tournaments:
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
    failures = [r for r in reports if isinstance(r, BaseException) and not is_expected_skip(r)]
    if reports and len(failures) == len(reports):
        logging.error("Every question failed; exiting with an error so the workflow is marked failed.")
        return 1
    return 0


def main() -> None:
    dotenv.load_dotenv()
    configure_output()
    sys.exit(asyncio.run(run(build_parser().parse_args())))


if __name__ == "__main__":
    main()
