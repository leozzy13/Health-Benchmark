from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .qa_validation import (
    CROSS_ADMISSION_QUESTION_TYPES,
    SINGLE_ADMISSION_QUESTION_TYPES,
)
from .utils import canonical_json_dumps


SINGLE_ADMISSION_SYSTEM_MESSAGE = """You are generating benchmark-quality hard question-answer pairs from one hospital-admission conversation between a doctor and a patient.

Rules:
1. Use the provided admission conversation as the evidence source.
2. Outside medical knowledge is allowed, but only if it is common, stable, and clinically basic.
3. Use outside medical knowledge to interpret the context, not to invent unsupported facts.
4. Every question must be hard.
5. A hard question must require synthesis, interpretation, temporal reasoning, plan reasoning, or multiple supporting turns.
6. Do not generate trivial lookup questions answerable from one explicit sentence unless medical reasoning is still required.
7. Questions must sound natural.
8. Questions are for benchmark evaluators, not for the patient.
9. Do not use second-person wording like "you" or "your" in the question text.
10. Use third-person phrasing such as "the patient", "the patient's symptoms", or "the doctor" "the doctor suggests" when needed.
11. Never mention subject_id, hadm_id, raw admission identifiers, note types, table names, file names, or internal benchmark structure in the question text.
12. Keep answers short and evaluation-friendly, usually 1 to 12 words, maximum 20 words.
13. Every item must include a question_type from the allowed list.
14. Evidence must contain the admission id in evidence only, plus supporting turn_ids.
15. Output valid JSON only."""


CROSS_ADMISSION_SYSTEM_MESSAGE = """You are generating benchmark-quality hard question-answer pairs from a chronological set of hospital admissions for the same patient.

The source context is a sequence of admission summaries. Each admission includes:
- admission_start
- admission_end
- one summary paragraph
- a problem list

Rules:
1. Every question must be hard.
2. Every question must require at least 2 admissions.
3. Prefer questions that require 3 or more admissions when possible.
4. Focus on longitudinal reasoning: progression, recurrence, comparison, order over time, first/last occurrence, and multi-admission medical interpretation.
5. Outside medical knowledge is allowed, but only if it is common, stable, and clinically basic.
6. Use outside medical knowledge to interpret the summaries, not to invent unsupported facts.
7. Because the source is summaries, avoid tiny wording-level details and focus on patterns preserved in summaries.
8. Questions must sound natural.
9. Never mention subject_id, hadm_id, raw admission identifiers, note types, table names, file names, or internal benchmark structure in the question text.
10. Keep answers short and evaluation-friendly, usually 1 to 12 words, maximum 20 words.
11. Every item must include a question_type from the allowed list.
12. Evidence must list only the admissions needed for the answer.
13. Output valid JSON only."""


@dataclass
class RenderedQAPrompt:
    system_message: str
    user_message: str
    question_count: int
    context_json: str


def render_single_admission_qa_prompt(
    conversation_payload: dict[str, Any],
    *,
    question_count: int,
) -> RenderedQAPrompt:
    context_json = canonical_json_dumps(conversation_payload)
    user_message = "\n".join(
        [
            f"Generate {int(question_count)} hard question-answer pairs for this single admission.",
            "",
            "Allowed question_type values:",
            *[f"- {question_type}" for question_type in SINGLE_ADMISSION_QUESTION_TYPES],
            "",
            "Requirements:",
            "- Every question must be hard.",
            "- Every question must be answerable from the conversation, possibly with common outside medical knowledge.",
            "- At least most questions should require multiple turns rather than one explicit statement.",
            "- Keep question wording natural.",
            "- Questions are written for benchmark users, not for the patient.",
            "- Do not use second-person wording like 'you' or 'your' in the question.",
            "- Do not use wording like 'he' or 'she' to refer to the patient or the doctor. Use third-person phrasing such as 'the patient', 'the patient's symptoms', or 'the doctor' when needed.",
            "- Do not mention raw identifiers in the question.",
            "- Every question must begin with a date-based admission prefix.",
            "- If the admission spans multiple dates, start with: During the hospitalization from YYYY-MM-DD to YYYY-MM-DD, ...",
            "- If the admission begins and ends on the same date, start with: During the hospitalization on YYYY-MM-DD, ...",
            "- The answer should be concise.",
            "- Evidence must list the admission used and the key turn_ids supporting the answer.",
            "",
            "Admission conversation:",
            context_json,
            "",
            'Return JSON with this exact shape:',
            '{',
            '  "qas": [',
            "    {",
            '      "qa_id": "...",',
            '      "scope": "single_admission",',
            '      "question_type": "...",',
            '      "question": "...",',
            '      "answer": "...",',
            '      "evidence": {',
            '        "admissions": ["..."],',
            '        "turn_ids": [1, 5, 9]',
            "      }",
            "    }",
            "  ]",
            "}",
        ]
    )
    return RenderedQAPrompt(
        system_message=SINGLE_ADMISSION_SYSTEM_MESSAGE,
        user_message=user_message,
        question_count=int(question_count),
        context_json=context_json,
    )


def render_cross_admission_qa_prompt(
    summary_contexts: list[dict[str, Any]],
    *,
    question_count: int = 50,
) -> RenderedQAPrompt:
    context_json = canonical_json_dumps(summary_contexts)
    user_message = "\n".join(
        [
            f"Generate {int(question_count)} hard cross-admission question-answer pairs from the chronological admission summaries below.",
            "",
            "Allowed question_type values:",
            *[f"- {question_type}" for question_type in CROSS_ADMISSION_QUESTION_TYPES],
            "",
            "Requirements:",
            "- Every question must be hard.",
            "- Every question must require at least 2 admissions.",
            "- Prefer questions that require synthesis across 3 or more admissions whenever possible.",
            "- No question should be answerable from one admission summary alone.",
            "- Favor progression, recurrence, first/last, comparison, and longitudinal timeline reasoning.",
            "- The question should sound natural and should not mention raw identifiers.",
            "- The answer should be concise.",
            "- Evidence must list only the admissions actually used.",
            "",
            "Chronological admission summaries:",
            context_json,
            "",
            'Return JSON with this exact shape:',
            '{',
            '  "qas": [',
            "    {",
            '      "qa_id": "...",',
            '      "scope": "cross_admission",',
            '      "question_type": "...",',
            '      "question": "...",',
            '      "answer": "...",',
            '      "evidence": {',
            '        "admissions": ["...", "...", "..."]',
            "      }",
            "    }",
            "  ]",
            "}",
        ]
    )
    return RenderedQAPrompt(
        system_message=CROSS_ADMISSION_SYSTEM_MESSAGE,
        user_message=user_message,
        question_count=int(question_count),
        context_json=context_json,
    )
