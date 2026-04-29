from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class RenderedAnswerPrompt:
    system_message: str
    user_message: str


def render_answer_prompt(
    *,
    context_text: str,
    questions: Sequence[dict[str, Any]],
    retry_count: int = 0,
) -> RenderedAnswerPrompt:
    system_lines = [
        "You are answering benchmark questions from a long patient conversation.",
        "Use only the provided conversation.",
        "Keep every answer short.",
        "Use wording from the conversation when possible.",
        'You must return strict JSON with the schema {"answers": [{"qa_id": "...", "prediction": "..."}]}.',
        "Return exactly one answer per provided qa_id.",
        "The response is invalid if any qa_id is omitted, repeated, or changed.",
        "Do not include explanations, citations, or extra keys.",
    ]
    if retry_count > 0:
        system_lines.append(
            "This is a retry because a previous response omitted required qa_ids. Answer every remaining qa_id exactly once."
        )
    user_message = json.dumps(
        {
            "patient_conversation": context_text,
            "questions": list(questions),
        },
        ensure_ascii=False,
        indent=2,
    )
    return RenderedAnswerPrompt(
        system_message="\n".join(system_lines),
        user_message=user_message,
    )
