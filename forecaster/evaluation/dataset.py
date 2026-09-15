"""Resolved questions for pastcasting: fetch once, store locally, reload.

Fetching needs METACULUS_TOKEN, because Metaculus's API requires an account even
for reads. Stored files contain only public question data.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import forecasting_tools
from forecasting_tools import (
    ApiFilter,
    BinaryQuestion,
    MetaculusClient,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)
from forecasting_tools.data_models.questions import CanceledResolution

SCOREABLE_CLASSES = (BinaryQuestion, MultipleChoiceQuestion, NumericQuestion)


def is_scoreable(question: MetaculusQuestion) -> bool:
    if not isinstance(question, SCOREABLE_CLASSES):
        return False
    if question.open_time is None and question.published_time is None:
        return False
    try:
        resolution = question.typed_resolution
    except Exception:
        return False
    return resolution is not None and not isinstance(resolution, CanceledResolution)


async def fetch_resolved(
    client: MetaculusClient, tournaments: Sequence[str | int], max_questions: int = 500
) -> list[MetaculusQuestion]:
    api_filter = ApiFilter(
        allowed_tournaments=list(tournaments),
        allowed_statuses=["resolved"],
        allowed_types=["binary", "multiple_choice", "numeric", "discrete"],
        group_question_mode="unpack_subquestions",
    )
    questions = await client.get_questions_matching_filter(
        api_filter, num_questions=max_questions, error_if_question_target_missed=False
    )
    return [q for q in questions if is_scoreable(q)]


def save_questions(questions: Sequence[MetaculusQuestion], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for question in questions:
            record = {"class": type(question).__name__, "data": question.to_json()}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_questions(path: str | Path) -> list[MetaculusQuestion]:
    questions: list[MetaculusQuestion] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            question_class = getattr(forecasting_tools, record["class"])
            questions.append(question_class.from_json(record["data"]))
    return questions
