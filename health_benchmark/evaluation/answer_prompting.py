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
    context_description: str = "conversation",
    context_payload_key: str = "patient_conversation",
) -> RenderedAnswerPrompt:
    resolved_context_description = str(context_description or "conversation").strip()
    resolved_context_payload_key = str(context_payload_key or "patient_conversation").strip()
    if resolved_context_description == "conversation":
        opening_line = "You are answering benchmark questions from a long patient conversation."
        grounding_line = "Use only the provided conversation."
        answer_policy_lines = [
            'If the answer is not supported by the conversation, answer exactly "the question is not answerable".',
        ]
    else:
        opening_line = "You are answering benchmark questions from compact patient context."
        grounding_line = f"Use only the provided {resolved_context_description}."
        if resolved_context_payload_key == "patient_memory_context":
            answer_policy_lines = [
                "Retrieved memories are partial patient facts.",
                "Use one or more memories only when they directly support the requested fact, relationship, comparison, or temporal change.",
                "For comparison, progression, and pattern questions, synthesize only when relevant memories provide specific evidence for the compared items or timepoints.",
                "Related clinical facts are not enough by themselves.",
                'Only answer exactly "the question is not answerable" when the retrieved memories do not support the requested fact or relationship.',
                "Prefer exact wording from the retrieved memories when possible.",
                "Do not invent details that are not supported by the retrieved memories.",
            ]
        elif resolved_context_payload_key == "retrieved_patient_context":
            answer_policy_lines = [
                "Retrieved excerpts are selected from patient admissions and may be partial.",
                "Use one or more excerpts only when they directly support the requested fact, relationship, comparison, or temporal change.",
                "For comparison, progression, and pattern questions, synthesize only when relevant excerpts provide specific evidence for the compared items or timepoints.",
                "Related clinical facts are not enough by themselves.",
                'Only answer exactly "the question is not answerable" when the retrieved excerpts do not support the requested fact or relationship.',
                "Prefer exact wording from the retrieved excerpts when possible.",
                "Do not invent details that are not supported by the retrieved excerpts.",
            ]
        else:
            answer_policy_lines = [
                f'If the answer is not supported by the {resolved_context_description}, answer exactly "the question is not answerable".',
            ]
    system_lines = [
        opening_line,
        grounding_line,
        *answer_policy_lines,
        "Keep every answer short.",
        f"Use wording from the {resolved_context_description} when possible.",
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
            resolved_context_payload_key: context_text,
            "questions": list(questions),
        },
        ensure_ascii=False,
        indent=2,
    )
    return RenderedAnswerPrompt(
        system_message="\n".join(system_lines),
        user_message=user_message,
    )
