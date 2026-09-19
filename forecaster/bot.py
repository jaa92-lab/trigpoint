"""The panel forecasting bot, built on forecasting-tools' ForecastBot.

The framework supplies question fetching, per-question error isolation,
publishing, and comment formatting. Each question gets one research pass and
one forecast call. The whole panel runs inside those two calls: triage,
evidence gathering, independent analysts, fact checks, escalation, and pooling.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import timezone

from forecasting_tools import (
    BinaryPrediction,
    BinaryQuestion,
    DateQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    Percentile,
    PredictedOption,
    PredictedOptionList,
    ReasonedPrediction,
    structure_output,
)
from forecasting_tools.data_models.numeric_report import DatePercentile

from forecaster import aggregate as agg
from forecaster import panel
from forecaster.analysts import (
    ESTIMATED_OUTPUT_TOKENS,
    PERCENTILE_LABELS,
    AnalystResult,
    LlmCall,
    QuestionView,
    build_prompt,
    parse_binary,
    parse_date_percentiles,
    parse_multiple_choice,
    parse_percentiles,
    run_analyst,
)
from forecaster.budget import CostLedger, TimeBudget, estimate_tokens, should_skip
from forecaster.clock import Clock
from forecaster.config import AnalystSpec, BotConfig, register_model_prices
from forecaster.evidence.base import EvidenceBundle, EvidenceItem
from forecaster.evidence.http import make_client
from forecaster.evidence.markets import fetch_markets
from forecaster.evidence.news import NEWS_QUERY_PROMPT, NewsSearcher, merge_news, parse_queries
from forecaster.evidence.prices import derive_price_prior, fetch_prices
from forecaster.evidence.sources import fetch_sources
from forecaster.evidence.weather import fetch_weather
from forecaster.grounding import check_rationale
from forecaster.triage import Triage, triage

logger = logging.getLogger(__name__)

ClockPolicy = Callable[[MetaculusQuestion], Clock]
SUPPORTED_TYPES = ("binary", "multiple_choice", "numeric", "date")


def live_clock(question: MetaculusQuestion) -> Clock:
    return Clock.live()


def question_type_of(question: MetaculusQuestion) -> str:
    if isinstance(question, BinaryQuestion):
        return "binary"
    if isinstance(question, MultipleChoiceQuestion):
        return "multiple_choice"
    if isinstance(question, NumericQuestion):
        return "numeric"
    if isinstance(question, DateQuestion):
        return "date"
    return "other"


def bounds_of(question: NumericQuestion | DateQuestion) -> tuple[float, float]:
    """Bounds on the axis the forecast uses. Date questions use timestamps, as the framework does."""
    if isinstance(question, DateQuestion):
        return question.lower_bound.timestamp(), question.upper_bound.timestamp()
    return question.lower_bound, question.upper_bound


def view_of(question: MetaculusQuestion) -> QuestionView:
    options: tuple[str, ...] = ()
    lower = upper = None
    open_lower = open_upper = True
    if isinstance(question, MultipleChoiceQuestion):
        options = tuple(question.options)
    if isinstance(question, NumericQuestion):
        lower = question.nominal_lower_bound if question.nominal_lower_bound is not None else question.lower_bound
        upper = question.nominal_upper_bound if question.nominal_upper_bound is not None else question.upper_bound
        open_lower = question.open_lower_bound
        open_upper = question.open_upper_bound
    elif isinstance(question, DateQuestion):
        lower, upper = bounds_of(question)
        open_lower = question.open_lower_bound
        open_upper = question.open_upper_bound
    return QuestionView(
        title=question.question_text,
        question_type=question_type_of(question),
        resolution_criteria=question.resolution_criteria or "",
        fine_print=question.fine_print or "",
        background=question.background_info or "",
        close_time=question.close_time,
        resolve_time=question.scheduled_resolution_time,
        options=options,
        unit=question.unit_of_measure,
        lower_bound=lower,
        upper_bound=upper,
        open_lower_bound=open_lower,
        open_upper_bound=open_upper,
    )


@dataclass
class CaseFile:
    view: QuestionView
    triage: Triage
    clock: Clock
    bundle: EvidenceBundle
    budget: TimeBudget
    ledger: CostLedger


def research_digest(case: CaseFile) -> str:
    bundle = case.bundle
    lines = [
        "## Evidence gathered",
        f"Question kind: {case.triage.kind}. Planned sources: {', '.join(case.triage.evidence_plan)}.",
        f"- {len(bundle.by_source('news'))} news articles",
        f"- {len(bundle.by_source('markets'))} related prediction markets",
        f"- {len(bundle.by_source('sources'))} resolution source pages",
    ]
    if "weather" in case.triage.evidence_plan:
        lines.append(f"- {len(bundle.by_source('weather'))} weather forecasts")
    lines += [f"- Price data for {p.symbol} from {p.provider}" for p in bundle.prices]
    if bundle.prior and bundle.prior.explanation:
        lines.append(f"- Statistical model: {bundle.prior.explanation}")
    window = case.triage.window
    if window is not None:
        lines.append(f"- Question window: {window.start:%Y-%m-%d %H:%M} to {window.end:%Y-%m-%d %H:%M} UTC")
    return "\n".join(lines)


class PanelForecaster(ForecastBot):
    _max_concurrent_questions = 2
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    def __init__(
        self,
        *,
        config: BotConfig | None = None,
        clock_policy: ClockPolicy | None = None,
        llm_call: LlmCall | None = None,
        news: NewsSearcher | None = None,
        http_client_factory: Callable[[], object] | None = None,
        **kwargs: object,
    ) -> None:
        self.config = config or BotConfig()
        self.clock_policy = clock_policy or live_clock
        self._news = news if news is not None else NewsSearcher()
        self._client_factory = http_client_factory or make_client
        self._llms_by_model: dict[str, GeneralLlm] = {}
        self._llm_call: LlmCall = llm_call or self._call_model
        kwargs.setdefault("research_reports_per_question", 1)
        kwargs.setdefault("predictions_per_research_report", 1)
        kwargs.setdefault("enable_summarize_research", False)
        kwargs.setdefault("required_successful_predictions", 1.0)
        super().__init__(**kwargs)
        register_model_prices(self.config.model_prices)

    def _llm_config_defaults(self) -> dict[str, str | GeneralLlm | None]:
        config = getattr(self, "config", None) or BotConfig()
        defaults: dict[str, str | GeneralLlm | None] = {
            "default": config.analysts[0].model,
            "summarizer": config.parser_model,
            "researcher": config.parser_model,
            "parser": config.parser_model,
        }
        for spec in config.analysts:
            defaults[f"analyst_{spec.name}"] = spec.model
        return defaults

    # ------------------------------------------------------------- models

    def _llm(self, model: str, timeout: int = 300) -> GeneralLlm:
        llm = self._llms_by_model.get(model)
        if llm is None:
            extra = {"reasoning_effort": "high"} if "/openai/" in model else {}
            llm = GeneralLlm(model=model, temperature=None, timeout=timeout, allowed_tries=2, **extra)
            self._llms_by_model[model] = llm
        return llm

    async def _call_model(self, model: str, prompt: str) -> str:
        return await self._llm(model).invoke(prompt)

    # ----------------------------------------------------------- research

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            clock = self.clock_policy(question)
            now = clock.now()
            view = view_of(question)
            if view.question_type not in SUPPORTED_TYPES:
                raise NotImplementedError(f"{view.question_type} questions are not supported")
            if clock.is_live and should_skip(question.close_time, now, self.config.skip_if_closing_within_seconds):
                raise RuntimeError("The question closes too soon to forecast safely.")
            triage_result = triage(
                question_text=view.title,
                question_type=view.question_type,
                now=now,
                resolution_criteria=view.resolution_criteria,
                fine_print=view.fine_print,
                background_info=view.background,
                resolve_time=view.resolve_time,
            )
            budget = TimeBudget.for_question(
                question.close_time if clock.is_live else None,
                now,
                self.config.max_seconds_per_question,
                self.config.safety_margin_seconds,
            )
            case = CaseFile(
                view=view,
                triage=triage_result,
                clock=clock,
                bundle=EvidenceBundle(),
                budget=budget,
                ledger=CostLedger(cap=self.config.max_cost_per_question, prices=self.config.model_prices),
            )
            case.bundle = await self.gather_evidence(case)
            notepad = await self._get_notepad(question)
            notepad.note_entries["case"] = case
            return research_digest(case)

    async def gather_evidence(self, case: CaseFile) -> EvidenceBundle:
        bundle = EvidenceBundle()
        plan = case.triage.evidence_plan
        timeout = case.budget.slice(0.35, minimum=15.0, maximum=240.0)
        jobs: dict[str, Awaitable[object]] = {}
        async with self._client_factory() as client:
            if self.config.use_prices and "prices" in plan:
                jobs["prices"] = fetch_prices(client, case.triage, case.clock)
            if self.config.use_markets and "markets" in plan:
                jobs["markets"] = fetch_markets(client, case.view.title, case.clock)
            if self.config.use_sources and "sources" in plan and case.triage.urls:
                compare_days = None
                if case.triage.kind == "official_count" and case.budget.remaining() > 600.0:
                    compare_days = self.config.source_compare_days or None
                jobs["sources"] = fetch_sources(
                    client,
                    case.triage.urls,
                    case.clock,
                    self.config.max_source_urls,
                    focus=f"{case.view.title} {case.view.resolution_criteria[:500]}",
                    compare_days=compare_days,
                )
            if self.config.use_weather and "weather" in plan:
                jobs["weather"] = fetch_weather(
                    client, case.view.title, case.view.resolution_criteria, case.triage, case.clock
                )
            if self.config.use_news and "news" in plan:
                if self._news.available:
                    jobs["news"] = self._gather_news(case)
                else:
                    bundle.notes.append("News search is not configured, so no news was gathered.")
            names = list(jobs)
            results = await asyncio.gather(
                *(asyncio.wait_for(jobs[name], timeout) for name in names), return_exceptions=True
            )
        for name, result in zip(names, results):
            if isinstance(result, BaseException):
                detail = str(result) or type(result).__name__
                bundle.notes.append(f"{name.capitalize()} evidence was unavailable: {detail}")
            elif name == "prices":
                bundle.prices.extend(result)
            else:
                items, notes = result
                bundle.add(*items)
                bundle.notes.extend(notes)
        if bundle.prices:
            bundle.prior = derive_price_prior(
                bundle.prices[0],
                case.triage,
                case.view.title,
                case.view.question_type,
                case.triage.horizon_days,
                self.config.price_tail_factor,
            )
        return bundle

    async def _gather_news(self, case: CaseFile) -> tuple[list[EvidenceItem], list[str]]:
        """The question as asked, plus targeted searches for the kinds where news is the main evidence."""
        main = self._news.search(case.view.title, case.clock)
        wants_more = self.config.extra_news_queries > 0 and case.triage.kind in self.config.extra_news_kinds
        if wants_more:
            first, queries = await asyncio.gather(main, self._news_queries(case))
        else:
            first, queries = await main, []
        batches = [first]
        notes: list[str] = []
        for query in queries:
            try:
                batches.append(await self._news.search(query, case.clock, broad=False))
            except Exception as error:  # one failed search should not lose the others
                notes.append(f"News search {query!r} failed: {error}")
        if queries:
            notes.append(f"Extra news searches: {'; '.join(queries)}")
        return merge_news(batches, self.config.max_news_items), notes

    async def _news_queries(self, case: CaseFile) -> list[str]:
        """Targeted searches from the parser model. Pastcasts cache them, so reruns hit the news cache."""
        cache = getattr(self._news, "cache", None)
        key = f"{case.view.title}|{case.clock.now().isoformat()}"
        if not case.clock.is_live and cache is not None:
            cached = cache.get("news_queries", key)
            if cached is not None:
                return list(cached)
        prompt = NEWS_QUERY_PROMPT.format(
            count=self.config.extra_news_queries,
            today=f"{case.clock.now():%Y-%m-%d}",
            title=case.view.title,
            criteria=(case.view.resolution_criteria or "Not provided.")[:600],
        )
        model = self.config.parser_model
        if not case.ledger.can_afford(model, estimate_tokens(prompt), 500):
            return []
        try:
            text = await asyncio.wait_for(self._llm_call(model, prompt), timeout=60)
        except Exception as error:
            logger.info("Writing news queries failed: %r", error)
            return []
        case.ledger.record("news_queries", model, estimate_tokens(prompt), estimate_tokens(text) * 3)
        queries = parse_queries(text, case.view.title, self.config.extra_news_queries)
        if not case.clock.is_live and cache is not None:
            cache.set("news_queries", key, queries)
        return queries

    # -------------------------------------------------------------- panel

    async def _case(self, question: MetaculusQuestion) -> CaseFile:
        notepad = await self._get_notepad(question)
        case = notepad.note_entries.get("case")
        if case is None:
            raise RuntimeError("Research did not run for this question")
        return case

    def _stage_one_names(self, case: CaseFile) -> tuple[str, ...]:
        if case.triage.kind == "weather" and not case.bundle.by_source("weather"):
            return self.config.stage_one_without_weather
        return self.config.stage_one.get(case.triage.kind, ("news", "outside_view"))

    async def _run_panel(
        self,
        case: CaseFile,
        parse: Callable[[str], object | None],
        fallback: Callable[[str], Awaitable[object | None]] | None,
    ) -> tuple[list[AnalystResult], str]:
        first_names = self._stage_one_names(case)
        first = [self.config.analyst(name) for name in first_names]
        rest = [spec for spec in self.config.analysts if spec.name not in first_names]
        question_type = case.view.question_type

        if case.budget.remaining() < self.config.fast_path_below_seconds:
            results = await self._run_stage(case, first[:1], parse, fallback, share=0.9)
            return results, "Fast path: little time remained before close, so one analyst forecast."

        results = await self._run_stage(case, first, parse, fallback, share=0.5)
        if rest and panel.should_escalate(results, question_type, self.config):
            return await self._escalate(case, results, rest, parse, fallback, "")

        # Two agreeing answers can agree by chance, so ask each analyst again.
        if len(panel.successful(results)) >= self.config.min_answers or case.budget.remaining() < 60.0:
            return results, "Stage one only: the first two analysts agreed closely."
        results += await self._run_stage(case, first, parse, fallback, share=0.5, answer=2)
        if rest and panel.should_escalate(results, question_type, self.config):
            return await self._escalate(case, results, rest, parse, fallback, " (after a second answer from each)")
        return results, "Stage one, asked twice: the first two analysts agreed closely, and so did their second answers."

    async def _escalate(
        self,
        case: CaseFile,
        results: list[AnalystResult],
        rest: list[AnalystSpec],
        parse: Callable[[str], object | None],
        fallback: Callable[[str], Awaitable[object | None]] | None,
        when: str,
    ) -> tuple[list[AnalystResult], str]:
        if case.budget.remaining() < 60.0:
            return results, "Stage one only: no time was left to escalate."
        reason = panel.escalation_reason(results, case.view.question_type, self.config)
        results = results + await self._run_stage(case, rest, parse, fallback, share=0.9)
        return results, f"Full panel: escalated because {reason}{when}."

    async def _run_stage(
        self,
        case: CaseFile,
        specs: list[AnalystSpec],
        parse: Callable[[str], object | None],
        fallback: Callable[[str], Awaitable[object | None]] | None,
        share: float,
        answer: int = 1,
    ) -> list[AnalystResult]:
        timeout = case.budget.slice(share, minimum=30.0, maximum=600.0)
        skipped: list[AnalystResult] = []
        runnable: list[tuple[AnalystSpec, str]] = []
        for spec in specs:
            evidence = case.bundle.render(spec.evidence, spec.include_prices, spec.include_prior)
            prompt = build_prompt(spec, case.view, evidence, case.clock.now())
            if not case.ledger.can_afford(spec.model, estimate_tokens(prompt), ESTIMATED_OUTPUT_TOKENS):
                skipped.append(
                    AnalystResult(spec.name, spec.model, None, "", error="skipped to stay under the per-question cost cap")
                )
                continue
            runnable.append((spec, prompt))
        outcomes = await asyncio.gather(
            *(
                run_analyst(
                    spec, prompt, self._llm_call, parse, timeout, case.ledger, fallback,
                    fallback_model=self.config.fallback_models.get(spec.model),
                )
                for spec, prompt in runnable
            )
        )
        for (spec, _), result in zip(runnable, outcomes):
            self._fact_check(case, spec, result)
        results = list(outcomes) + skipped
        if answer > 1:
            for result in results:
                result.name = f"{result.name}, answer {answer}"
        return results

    def _fact_check(self, case: CaseFile, spec: AnalystSpec, result: AnalystResult) -> None:
        if not self.config.use_grounding or not result.ok:
            return
        report = check_rationale(
            result.rationale,
            case.bundle.urls(*spec.evidence),
            reference_price=case.bundle.reference_price(),
            known_values=case.triage.thresholds,
        )
        result.flags.extend(report.flags)
        result.weight *= report.weight

    def _report(
        self,
        case: CaseFile,
        results: list[AnalystResult],
        stage: str,
        final_text: str,
        anchor_text: str | None,
    ) -> str:
        return panel.render_panel_report(
            kind=case.triage.kind,
            question_type=case.view.question_type,
            stage=stage,
            results=results,
            final_text=final_text,
            anchor_text=anchor_text,
            notes=case.bundle.notes,
            spend=case.ledger.spent,
        )

    # ---------------------------------------------------------- forecasts

    async def _run_forecast_on_binary(self, question: BinaryQuestion, research: str) -> ReasonedPrediction[float]:
        case = await self._case(question)
        results, stage = await self._run_panel(case, parse_binary, self._fallback_binary)
        prior = case.bundle.prior.probability if case.bundle.prior else None
        final, pooled = panel.combine_binary(results, prior, self.config)
        anchor = None
        if pooled is None:
            anchor = "Every analyst failed, so this is the statistical model's estimate alone."
        elif prior is not None and self.config.data_prior_weight > 0:
            anchor = (
                f"The analysts pooled to {pooled:.1%}, then moved toward the statistical model's "
                f"{prior:.1%} with weight {self.config.data_prior_weight:.2f}."
            )
        return ReasonedPrediction(
            prediction_value=final, reasoning=self._report(case, results, stage, f"{final:.1%}", anchor)
        )

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        case = await self._case(question)
        options = list(question.options)
        results, stage = await self._run_panel(
            case,
            lambda text: parse_multiple_choice(text, options),
            lambda text: self._fallback_multiple_choice(text, options),
        )
        pooled = panel.combine_multiple_choice(results, options, self.config)
        prediction = PredictedOptionList(
            predicted_options=[PredictedOption(option_name=o, probability=pooled[o]) for o in options]
        )
        text = ", ".join(f"{o} {pooled[o]:.0%}" for o in options)
        return ReasonedPrediction(prediction_value=prediction, reasoning=self._report(case, results, stage, text, None))

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        case = await self._case(question)
        results, stage = await self._run_panel(case, parse_percentiles, self._fallback_numeric)
        prior_quantiles = case.bundle.prior.quantiles if case.bundle.prior else None
        quantiles = panel.combine_numeric(
            results,
            prior_quantiles,
            self.config,
            question.lower_bound,
            question.upper_bound,
            question.open_lower_bound,
            question.open_upper_bound,
            question.zero_point,
        )
        distribution = self._distribution(quantiles, question)
        text = panel.format_value(quantiles, "numeric")
        anchor = None
        if prior_quantiles and self.config.data_prior_weight > 0:
            anchor = f"The statistical model's quantiles were blended in with a {self.config.data_prior_weight:.0%} share."
        return ReasonedPrediction(prediction_value=distribution, reasoning=self._report(case, results, stage, text, anchor))

    async def _run_forecast_on_date(
        self, question: DateQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        case = await self._case(question)
        results, stage = await self._run_panel(case, parse_date_percentiles, self._fallback_date)
        lower, upper = bounds_of(question)
        quantiles = panel.combine_numeric(
            results, None, self.config, lower, upper, question.open_lower_bound, question.open_upper_bound
        )
        distribution = self._distribution(quantiles, question)
        text = panel.format_value(quantiles, "date")
        return ReasonedPrediction(prediction_value=distribution, reasoning=self._report(case, results, stage, text, None))

    @staticmethod
    def _distribution(
        quantiles: dict[float, float], question: NumericQuestion | DateQuestion
    ) -> NumericDistribution:
        percentiles = [Percentile(percentile=level, value=value) for level, value in sorted(quantiles.items())]
        try:
            return NumericDistribution.from_question(percentiles, question)
        except ValueError:
            lower, upper = bounds_of(question)
            span = upper - lower
            lo = lower + span * 1e-3
            hi = upper - span * 1e-3
            inside = agg.enforce_increasing({level: min(max(v, lo), hi) for level, v in quantiles.items()})
            return NumericDistribution.from_question(
                [Percentile(percentile=level, value=value) for level, value in sorted(inside.items())], question
            )

    # ------------------------------------------------- fallback parsing

    async def _fallback_binary(self, text: str) -> float | None:
        prediction = await structure_output(text, BinaryPrediction, model=self._llm(self.config.parser_model, 120))
        return prediction.prediction_in_decimal

    async def _fallback_multiple_choice(self, text: str, options: list[str]) -> dict[str, float] | None:
        parsed = await structure_output(
            text,
            PredictedOptionList,
            model=self._llm(self.config.parser_model, 120),
            additional_instructions=f"Option names must be exactly these: {options}",
        )
        values = {o.option_name: o.probability for o in parsed.predicted_options}
        return values if set(values) == set(options) else None

    async def _fallback_numeric(self, text: str) -> dict[float, float] | None:
        parsed = await structure_output(text, list[Percentile], model=self._llm(self.config.parser_model, 120))
        found = {round(p.percentile, 2): p.value for p in parsed}
        wanted = [label / 100 for label in PERCENTILE_LABELS]
        if not all(level in found for level in wanted):
            return None
        return {level: found[level] for level in wanted}

    async def _fallback_date(self, text: str) -> dict[float, float] | None:
        parsed = await structure_output(text, list[DatePercentile], model=self._llm(self.config.parser_model, 120))
        found = {}
        for p in parsed:
            moment = p.value if p.value.tzinfo else p.value.replace(tzinfo=timezone.utc)
            found[round(p.percentile, 2)] = moment.timestamp()
        wanted = [label / 100 for label in PERCENTILE_LABELS]
        if not all(level in found for level in wanted):
            return None
        return {level: found[level] for level in wanted}
