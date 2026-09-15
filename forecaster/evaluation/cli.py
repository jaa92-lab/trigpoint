"""Offline evaluation commands.

    uv run python -m forecaster.evaluation.cli fetch --tournament <slug-or-id> --out data/holdout.jsonl
        Save resolved questions from past tournaments. Needs METACULUS_TOKEN.

    uv run python -m forecaster.evaluation.cli pastcast --data data/holdout.jsonl --limit 40
        Print the worst-case spend and stop. Add --confirm-spend to run the models.
        Add --runs 3 to measure the noise floor, or --disable markets,grounding to test a change.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import dotenv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m forecaster.evaluation.cli")
    commands = parser.add_subparsers(dest="command", required=True)

    fetch = commands.add_parser("fetch", help="Save resolved questions from past tournaments")
    fetch.add_argument("--tournament", action="append", required=True, help="Tournament slug or ID (repeatable)")
    fetch.add_argument("--out", default="data/holdout.jsonl")
    fetch.add_argument("--max-questions", type=int, default=500)

    past = commands.add_parser("pastcast", help="Replay saved questions and score the forecasts")
    past.add_argument("--data", default="data/holdout.jsonl")
    past.add_argument("--limit", type=int, default=40)
    past.add_argument("--runs", type=int, default=1)
    past.add_argument("--disable", default="", help="Comma list: news, markets, prices, sources, grounding, prior, escalation")
    past.add_argument("--delay-minutes", type=float, default=10.0)
    past.add_argument("--cache-dir", default="data/cache")
    past.add_argument("--out-dir", default=None)
    past.add_argument("--confirm-spend", action="store_true", help="Actually call the models")
    return parser


async def fetch_command(args: argparse.Namespace) -> int:
    from forecasting_tools import MetaculusClient

    from forecaster.evaluation.dataset import fetch_resolved, save_questions

    questions = await fetch_resolved(MetaculusClient(), args.tournament, args.max_questions)
    save_questions(questions, args.out)
    print(f"Saved {len(questions)} scoreable resolved questions to {args.out}")
    return 0


async def pastcast_command(args: argparse.Namespace) -> int:
    from forecaster.config import BotConfig
    from forecaster.evaluation.dataset import load_questions
    from forecaster.evaluation.pastcast import config_with_disabled, estimate_max_cost, render_report, run_pastcast

    questions = load_questions(args.data)[: args.limit]
    disabled = [name.strip() for name in args.disable.split(",") if name.strip()]
    config = config_with_disabled(BotConfig(), disabled)
    ceiling = estimate_max_cost(len(questions), args.runs, config)
    print(
        f"{len(questions)} questions x {args.runs} run(s). Worst-case model spend ${ceiling:.2f} "
        f"at a ${config.max_cost_per_question:.2f} cap per question."
    )
    if not args.confirm_spend:
        print("Nothing was run. Add --confirm-spend to call the models.")
        return 0

    out_dir = Path(args.out_dir or f"runs/{datetime.now(timezone.utc):%Y%m%d-%H%M%S}")
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for index in range(args.runs):
        records = await run_pastcast(questions, config=config, delay_minutes=args.delay_minutes, cache_dir=args.cache_dir)
        runs.append(records)
        with (out_dir / f"run{index + 1}.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record.to_json(), ensure_ascii=False) + "\n")
    report = render_report(runs, disabled)
    (out_dir / "summary.md").write_text(report, encoding="utf-8")
    print(report)
    return 0


def main(argv: list[str] | None = None) -> int:
    dotenv.load_dotenv()
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    command = fetch_command if args.command == "fetch" else pastcast_command
    return asyncio.run(command(args))


if __name__ == "__main__":
    sys.exit(main())
