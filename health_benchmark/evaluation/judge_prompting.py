from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class RenderedJudgePrompt:
    system_message: str
    user_message: str


def render_answerable_judge_prompt(items: Sequence[dict[str, Any]]) -> RenderedJudgePrompt:
    system_lines = [
        "You are grading candidate answers for short-answer medical benchmark questions.",
        "Judge only from the provided question, gold_answer, and candidate_answer.",
        "Score 1 when the candidate answer is correct.",
        "Score 0 when the candidate answer is false, incorrect, unsupported, incomplete enough to be wrong, or only says it is not answerable.",
        'Return strict JSON with the schema {"judgments": [{"qa_id": "...", "score": 1}]}.',
        "Each score must be exactly one of: 0, 1.",
        "Return exactly one judgment per provided qa_id.",
    ]
    user_message = json.dumps({"items": list(items)}, ensure_ascii=False, indent=2)
    return RenderedJudgePrompt(
        system_message="\n".join(system_lines),
        user_message=user_message,
    )
