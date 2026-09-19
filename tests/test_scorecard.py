import json
from datetime import datetime, timezone

import httpx
import pytest

from forecaster.evaluation import scorecard as sc

RESOLVED_AT = "2026-09-22T12:00:00Z"


def binary(question_id, p_yes, resolution, title="Will Zcash (ZEC) trade above $1,100 on September 25, 2026?"):
    return {
        "id": question_id,
        "type": "binary",
        "status": "resolved",
        "title": title,
        "resolution": resolution,
        "actual_resolve_time": RESOLVED_AT,
        "my_forecasts": {"latest": {"forecast_values": [1 - p_yes, p_yes]}, "score_data": {"spot_peer_score": 12.5, "coverage": 1.0}},
    }


def numeric_post(cdf_from, resolution, kind_title="How many combined points will be scored in the Lions at Bills game on September 25, 2026?"):
    n = 201
    cdf = [min(1.0, max(0.0, (i / 200 - cdf_from) / 0.2)) for i in range(n)]  # ramp from cdf_from to cdf_from + 0.2
    return {
        "id": 900,
        "title": kind_title,
        "question": {
            "id": 901,
            "type": "numeric",
            "status": "resolved",
            "resolution": resolution,
            "scaling": {"range_min": 0.0, "range_max": 100.0, "zero_point": None},
            "my_forecasts": {"latest": {"forecast_values": cdf}, "score_data": {}},
        },
    }


def test_binary_forecasts_are_scored_with_metaculus_scores_kept():
    post = {"id": 100, "title": "Will Zcash (ZEC) trade above $1,100 on September 25, 2026?", "question": binary(101, 0.8, "yes")}
    [question] = sc.questions_in_post(post)
    record = sc.score_question(question)
    assert record.kind == "price_threshold"
    assert record.forecast == "80% Yes"
    assert record.outcome == "Yes"
    assert record.baseline_score == pytest.approx(100 * (1 + __import__("math").log2(0.8)))
    assert record.metaculus_scores == {"spot_peer_score": 12.5}
    assert record.url == "https://www.metaculus.com/questions/100/"


def test_group_posts_score_only_the_members_the_bot_forecast():
    post = {
        "id": 200,
        "title": "Will the net worth reach 4x?",
        "group_of_questions": {
            "questions": [
                binary(201, 0.3, "no"),
                {**binary(202, 0.3, "no"), "my_forecasts": {"latest": None, "score_data": {}}},
                {**binary(203, 0.3, None), "status": "open", "resolution": None},
            ]
        },
    }
    members = sc.questions_in_post(post)
    records = [sc.score_question(q) for q in members]
    assert [r is not None for r in records] == [True, False, False]
    assert records[0].title == "Will the net worth reach 4x?"


def test_multiple_choice_numeric_and_date_questions():
    mc = {
        "id": 301, "type": "multiple_choice", "status": "resolved", "title": "Who wins?", "resolution": "Bob",
        "options": ["Alice", "Bob"], "my_forecasts": {"latest": {"forecast_values": [0.25, 0.75]}, "score_data": {}},
    }
    record = sc.score_question(mc)
    assert record.correct_option_probability == 0.75
    assert record.baseline_score > 0

    numeric = sc.questions_in_post(numeric_post(0.4, "50"))[0]
    record = sc.score_question(numeric)
    assert record.kind == "sports"
    assert record.covered_by_80pct_interval is True
    assert record.baseline_score > 0
    missed = sc.score_question(sc.questions_in_post(numeric_post(0.4, "95"))[0])
    assert missed.covered_by_80pct_interval is False
    assert missed.baseline_score < record.baseline_score

    lower = datetime(2026, 9, 1, tzinfo=timezone.utc).timestamp()
    upper = datetime(2026, 12, 31, tzinfo=timezone.utc).timestamp()
    date = {
        "id": 401, "type": "date", "status": "resolved", "title": "When will the report be published?",
        "resolution": "2026-10-01", "scaling": {"range_min": lower, "range_max": upper, "zero_point": None},
        "my_forecasts": {"latest": {"forecast_values": [i / 200 for i in range(201)]}, "score_data": {}},
    }
    record = sc.score_question(date)
    assert record.outcome == "2026-10-01"
    assert record.covered_by_80pct_interval is True


def test_unscoreable_questions_are_skipped():
    assert sc.score_question({**binary(1, 0.5, "annulled")}) is None
    assert sc.score_question({**binary(2, 0.5, "yes"), "status": "open"}) is None
    log_scaled = sc.questions_in_post(numeric_post(0.4, "50"))[0]
    log_scaled["scaling"]["zero_point"] = 0.5
    assert sc.score_question(log_scaled) is None


def test_report_has_calibration_and_worst_misses():
    records = [
        sc.score_question(binary(1, 0.9, "yes")),
        sc.score_question(binary(2, 0.9, "no", title="Will Trump have nominated a Secretary of the Army by September 25, 2026?")),
        sc.score_question(binary(3, 0.2, "no")),
        sc.score_question(sc.questions_in_post(numeric_post(0.4, "50"))[0]),
    ]
    report = sc.render_scorecard(records, generated=datetime(2026, 9, 26, tzinfo=timezone.utc))
    assert "from 4 resolved forecasts" in report
    assert "| 90%-100% | 2 | 90% | 50% |" in report
    assert "| 10%-30% | 1 | 20% | 0% |" in report
    assert "Outcomes inside the bot's 10th-90th percentile range: 1 of 1" in report
    assert "Metaculus spot peer score: mean +12.5, sum +37.5 (n=3)" in report
    assert "## Worst misses" in report
    assert "Secretary of the Army" in report.split("## Worst misses")[1].splitlines()[3]
    assert sc.render_scorecard([]).strip().endswith("Nothing has resolved yet.")


async def test_build_scorecard_walks_pages_and_details():
    page1 = {"results": [{"id": 100}], "next": "https://www.metaculus.com/api/posts/?forecaster_id=7&limit=1&offset=1"}
    page2 = {"results": [{"id": 200}], "next": "https://www.metaculus.com/api/posts/?forecaster_id=7&limit=1&offset=2"}
    empty = {"results": [], "next": "https://www.metaculus.com/api/posts/?forecaster_id=7&limit=1&offset=3"}
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        path = request.url.path
        if path == "/api/users/me/":
            return httpx.Response(200, json={"id": 7, "username": "Trigpoint"})
        if path == "/api/posts/":
            assert request.url.params["forecaster_id"] == "7"
            offset = request.url.params.get("offset")
            return httpx.Response(200, json={None: page1, "1": page2}.get(offset, empty))
        if path == "/api/posts/100/":
            return httpx.Response(200, json={"id": 100, "title": "Will Zcash (ZEC) trade above $1,100 on September 25, 2026?", "question": binary(101, 0.8, "yes")})
        if path == "/api/posts/200/":
            return httpx.Response(200, json={"id": 200, "title": "Open one", "question": {**binary(201, 0.5, None), "status": "open", "resolution": None}})
        return httpx.Response(404)

    async def no_sleep(seconds):
        return None

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        records = await sc.build_scorecard(client, tournaments=["minibench"], sleep=no_sleep, page_size=1)
    assert [r.question_id for r in records] == [101]
    assert any("tournaments=minibench" in url for url in seen)
    # Three list pages were read (two with results, one empty), and the empty page's next link was not followed.
    assert sum("/api/posts/?" in url for url in seen) == 3
    assert json.loads(json.dumps(records[0].to_json()))["p_yes"] == 0.8
