# Forecasting bot

A bot for Metaculus's FutureEval bot tournaments: the seasonal tournament and the
two-week MiniBench rounds. It runs unattended on GitHub Actions every 20 minutes
and forecasts each new question once.

## How it forecasts a question

1. **Triage.** The question title is classified as a price question, a price
   level, sports, weather, an official count, an event, or generic. That decides
   which evidence to gather and which analysts go first. Dates in the title
   become a time window.
2. **Evidence, gathered in parallel.**
   - Price history from Yahoo Finance and CoinGecko, plus a statistical model:
     a driftless lognormal estimate based on recent volatility.
   - Related prediction markets on Manifold and Polymarket.
   - The pages named in the resolution criteria.
   - News from AskNews.
3. **Independent analysts.** Each analyst sees a different slice of evidence, and
   they run on different models:
   - A news analyst on Claude Opus 5.
   - A data analyst on GPT, which sees prices, markets, and the statistical model.
   - A resolution-source analyst on Gemini.
   - A base-rate analyst on Claude, which sees no current evidence at all.
4. **Two stages.** The two analysts best suited to the question run first. The
   other two join when those two disagree, one fails, a fact check flags one, or
   the question is not yes/no.
5. **Fact checks.** Rules with no AI judge downweight an analyst who claims to
   have searched, cites a site it was never shown, or states a price that
   contradicts the price feed.
6. **Combining.**
   - Yes/no forecasts are averaged in log-odds, pulled toward the statistical
     model when one applies, and kept between 2% and 98%.
   - Multiple-choice forecasts give every option at least 1%.
   - Numeric forecasts average percentiles, with slightly widened tails.

Every step lands in the comment posted with the forecast, and prize winners must
explain how their bot works.

Ideas carried over from two earlier projects:

- **The Clearing:** the contradiction checker.
- **Silas:** the noise-floor measurement and the automatically scored prediction loop.

## Limits and safety rails

- **Time.** Questions stay open about 90 minutes. The bot budgets up to 25 minutes
  per question and stops 5 minutes before close. It skips any question closing
  within 3 minutes, and switches to a single analyst when time is short.
- **Money.** Each question has a cost cap, $1.00 by default. Analysts that would
  break it are skipped.
- **Tournament rules.**
  - No human in the loop.
  - One forecast per question.
  - Changes are tested only on questions that have already closed.
- **Unsupported question types.** Date and conditional questions are refused
  instead of guessed.
- **Credentials.** The bot never stores credentials in the repository. Keys live
  in GitHub secrets or a local `.env` file.

## Setup

1. Create a Metaculus account, add a bot account in settings, and copy its token.
2. Request donated model credits through Metaculus's participation form
   (https://forms.gle/aQdYMq9Pisrf1v7d8), which provides an OpenRouter key, and
   sign up for free AskNews access as described on the
   [resources page](https://www.metaculus.com/notebooks/38928/futureeval-resources-page/).
3. In the GitHub repository, add these under **Settings → Secrets and variables →
   Actions**: `METACULUS_TOKEN`, `OPENROUTER_API_KEY`, and either
   `ASKNEWS_API_KEY` or the older `ASKNEWS_CLIENT_ID` + `ASKNEWS_SECRET` pair.
4. Enable Actions, then run **Test bot on the bot-testing area** once by hand.

## Running locally

```bash
uv sync
cp .env.template .env   # then fill in the values
uv run pytest -q
uv run python main.py --mode dry_run --url https://www.metaculus.com/questions/12345/
```

`dry_run` prints the full reasoning and publishes nothing.

## Testing changes on past questions

Improvements are judged by pastcasting. The bot replays resolved questions as if
it were seeing each one ten minutes after it opened. It gets only the news,
prices, and archived pages that existed at that moment. Prediction markets are
skipped, because their past prices are unavailable.

```bash
# Save resolved questions from past rounds. Slugs are listed on the MiniBench page.
uv run python -m forecaster.evaluation.cli fetch --tournament <slug> --out data/holdout.jsonl

# Show the worst-case spend. Nothing runs without --confirm-spend.
uv run python -m forecaster.evaluation.cli pastcast --data data/holdout.jsonl --limit 40 --runs 3

# Test one change by switching a feature off
uv run python -m forecaster.evaluation.cli pastcast --limit 40 --disable grounding --confirm-spend
```

Run the same configuration three times first. The spread between those identical
runs is the noise floor, and a change only counts if it beats that spread.

## Comparison bot

`python main.py --mode baseline` runs Metaculus's own template bot with the same
lead model. It runs under a second bot account, linked as secondary so it never
counts for prizes. Its token goes in the `METACULUS_BASELINE_TOKEN` secret. The
score gap between the two bots shows what the panel design adds. Its workflow is
manual-only until the main bot is live.

## Layout

| Path | What it does |
|---|---|
| `main.py` | Command-line entry point |
| `forecaster/bot.py` | The bot class, which ties the pipeline into forecasting-tools |
| `forecaster/triage.py` | Question classification, thresholds, date windows |
| `forecaster/evidence/` | Price, market, source-page, and news fetchers |
| `forecaster/quant.py` | Probability math for price questions |
| `forecaster/analysts.py` | Prompts, forecast parsing, and a failure-safe runner |
| `forecaster/grounding.py` | Rule-based fact checks |
| `forecaster/panel.py`, `aggregate.py` | Escalation and combining |
| `forecaster/budget.py` | Time and cost budgets |
| `forecaster/clock.py` | One source of "now", so tests on past questions cannot leak later information |
| `forecaster/evaluation/` | Scoring for offline tests on resolved questions |
