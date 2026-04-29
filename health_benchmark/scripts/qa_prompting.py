from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .qa_validation import (
    CANONICAL_ADVERSARIAL_ANSWER,
    CROSS_ADMISSION_QUESTION_TYPES,
    CROSS_ADMISSION_REGULAR_QUESTION_TYPES,
    SINGLE_ADMISSION_QUESTION_TYPES,
    SINGLE_ADMISSION_REGULAR_QUESTION_TYPES,
)
from .utils import canonical_json_dumps


SINGLE_ADMISSION_REGULAR_SYSTEM_MESSAGE = """You are generating benchmark-quality hard short-answer question-answer pairs from one hospital-admission conversation between a doctor and a patient.

Rules:
1. Use the provided admission conversation as the evidence source.
2. Generate only answerable questions.
3. Outside medical knowledge is allowed only if it is common, stable, and clinically basic.
4. Use outside knowledge to interpret the context, not to invent unsupported facts.
5. Every question must be hard and should usually require synthesis across multiple turns.
6. Questions must sound natural and be written for benchmark evaluators, not for the patient.
7. Do not use second-person wording like "you" or "your" in the question text.
8. Use third-person phrasing such as "the patient", "the patient's symptoms", or "the doctor" when needed.
9. Every answer must be a short open answer, not yes/no, not multiple choice, and not more than 10 words.
10. Evidence must cite the admission id plus supporting turn_ids.
11. Output valid JSON only."""

SINGLE_ADMISSION_ADVERSARIAL_SYSTEM_MESSAGE = """You are generating adversarial benchmark questions from one hospital-admission conversation plus the patient's chronological admission summaries.

Adversarial question rules:
1. Generate a question whose correct response is that the requested information is not answerable from the provided record.
2. The question must still sound natural, specific, and medically plausible.
3. Anchor the question to the target hospitalization and ask for a plausible but unsupported detail about that admission.
4. Before writing the final question, verify that the asked-for fact is absent from the full provided context, including the patient-level summaries.
5. Make the question hard by using plausible confusion, such as wrong admission, wrong timepoint, wrong causal interpretation, wrong episode, or unsupported progression claim.
6. Do not use trick wording, double negatives, yes/no questions, or vague references.
7. The expected answer must be exactly: "the question is not answerable".
8. Evidence must point to the closest tempting context that makes the question plausible, not to an answer span.
9. Evidence must cite the admission id plus supporting turn_ids.
10. Output valid JSON only."""

CROSS_ADMISSION_REGULAR_SYSTEM_MESSAGE = """You are generating benchmark-quality hard short-answer question-answer pairs from a chronological set of hospital admissions for the same patient.

The source context is a sequence of admission summaries. Each admission includes:
- admission_start
- admission_end
- one summary paragraph
- a problem list

Rules:
1. Generate only answerable questions.
2. Every question must require at least 2 admissions.
3. Prefer questions that require 3 or more admissions when possible.
4. Focus on longitudinal reasoning: progression, recurrence, comparison, order over time, first/last occurrence, and multi-admission medical interpretation.
5. Outside medical knowledge is allowed only if it is common, stable, and clinically basic.
6. Use outside knowledge to interpret the summaries, not to invent unsupported facts.
7. Because the source is summaries, avoid tiny wording-level details and focus on patterns preserved in summaries.
8. Questions must sound natural and should not mention raw identifiers.
9. Every answer must be a short open answer, not yes/no, and not more than 10 words.
10. Evidence must list only the admissions actually needed for the answer.
11. Output valid JSON only."""

CROSS_ADMISSION_ADVERSARIAL_SYSTEM_MESSAGE = """You are generating adversarial benchmark questions from a chronological set of hospital admissions for the same patient.

Adversarial question rules:
1. Generate a question whose correct response is that the requested information is not answerable from the provided summaries.
2. The question must still sound natural, specific, and medically plausible.
3. Ask about a plausible longitudinal pattern, recurrence, or comparison that is not explicitly supported anywhere in the summaries.
4. Before writing the final question, verify that the asked-for fact is absent from the full provided context.
5. Make the question hard by using plausible confusion, such as unsupported recurrence, unsupported progression, unsupported causal interpretation, or unsupported longitudinal comparison.
6. Do not use trick wording, double negatives, yes/no questions, or vague references.
7. The expected answer must be exactly: "the question is not answerable".
8. Evidence admissions should be the admissions that create the plausible confusion, not an answer span.
9. Output valid JSON only."""

QA_REPAIR_MESSAGE = """The previous response failed validation.
Return corrected JSON only. Keep the same schema and fix all count,
question_type, answer, admission-evidence, and turn-id issues.
If the prompt allows only one question_type, every item must use that exact question_type.
For cross-admission QA, every item must satisfy the exact requested item count
and cite at least 2 unique admissions.
For cross-admission QA, evidence.admissions must copy only the exact
admission_id_for_evidence_only values already present in the prompt context.
Do not invent, alter, extend, or reformat admission ids."""


@dataclass
class RenderedQAPrompt:
    system_message: str
    user_message: str
    question_count: int
    context_json: str


def render_single_admission_regular_qa_prompt(
    conversation_payload: dict[str, Any],
    *,
    question_count: int,
) -> RenderedQAPrompt:
    context_json = canonical_json_dumps(conversation_payload)
    user_message = "\n".join(
        [
            f"Generate exactly {int(question_count)} hard answerable short-answer question-answer pairs for this single admission.",
            "",
            "Allowed question_type values:",
            *[f"- {question_type}" for question_type in SINGLE_ADMISSION_REGULAR_QUESTION_TYPES],
            "",
            "Requirements:",
            "- Every question must be hard.",
            "- Every question must be answerable from the conversation, possibly with common outside medical knowledge.",
            "- At least most questions should require multiple turns rather than one explicit statement.",
            "- Keep question wording natural.",
            "- Questions are written for benchmark users, not for the patient.",
            "- Do not use second-person wording like 'you' or 'your' in the question.",
            "- Do not use wording like 'he' or 'she' to refer to the patient or the doctor. Use third-person phrasing such as 'the patient', 'the patient's symptoms', or 'the doctor' when needed.",
            f"- Generate exactly {int(question_count)} questions total.",
            "- Every item must use one of the allowed regular single-admission question_type values.",
            "- Every answer must be a short open answer, preferably 1 to 7 words, and never more than 10 words.",
            "- When possible, form the short answer using exact wording from the conversation rather than paraphrasing.",
            "- If exact wording is awkward, use only light normalization while keeping the wording as close as possible to the conversation.",
            "- Do not use yes/no questions or yes/no answers.",
            "- Do not mention raw identifiers in the question.",
            "- Every question must begin with a date-based admission prefix.",
            "- If the admission spans multiple dates, start with: During the hospitalization from YYYY-MM-DD to YYYY-MM-DD, ...",
            "- If the admission begins and ends on the same date, start with: During the hospitalization on YYYY-MM-DD, ...",
            "- Evidence must list the admission used and the key turn_ids supporting the answer.",
            "",
            "Admission conversation:",
            context_json,
            "",
            "Return JSON with this exact shape:",
            "{",
            '  "qas": [',
            "    {",
            '      "qa_id": "...",',
            '      "scope": "single_admission",',
            '      "question_type": "medical_reasoning",',
            '      "question": "During the hospitalization from YYYY-MM-DD to YYYY-MM-DD, which condition most likely drove the antibiotic escalation?",',
            '      "answer": "worsening pneumonia",',
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
        system_message=SINGLE_ADMISSION_REGULAR_SYSTEM_MESSAGE,
        user_message=user_message,
        question_count=int(question_count),
        context_json=context_json,
    )


def render_single_admission_adversarial_qa_prompt(
    conversation_payload: dict[str, Any],
    summary_contexts: list[dict[str, Any]],
    *,
    question_count: int,
) -> RenderedQAPrompt:
    context_payload = {
        "target_admission_conversation": conversation_payload,
        "patient_admission_summaries": summary_contexts,
    }
    context_json = canonical_json_dumps(context_payload)
    user_message = "\n".join(
        [
            f"Generate exactly {int(question_count)} adversarial short-answer question-answer pair for this admission.",
            "",
            "Allowed question_type values:",
            "- adversarial",
            "",
            "Requirements:",
            "- The question must be unsupported by the full provided context.",
            "- The question must still sound natural, specific, and medically plausible.",
            "- Anchor the question to the target hospitalization while asking for a plausible but unsupported detail.",
            "- The requested fact must not appear anywhere else in the patient's full record included below.",
            f"- Generate exactly {int(question_count)} questions total.",
            f"- The expected answer must be exactly: {CANONICAL_ADVERSARIAL_ANSWER!r}.",
            "- Questions are written for benchmark users, not for the patient.",
            "- Do not use second-person wording like 'you' or 'your' in the question.",
            "- Do not use wording like 'he' or 'she' to refer to the patient or the doctor. Use third-person phrasing such as 'the patient', 'the patient's symptoms', or 'the doctor' when needed.",
            "- Do not use trick wording, double negatives, yes/no questions, or vague references.",
            "- Keep the current date-based admission prefix style.",
            "- Evidence must list the target admission and supporting turn_ids for the tempting context only.",
            "",
            "Target admission conversation plus patient-level chronological summaries:",
            context_json,
            "",
            "Return JSON with this exact shape:",
            "{",
            '  "qas": [',
            "    {",
            '      "qa_id": "...",',
            '      "scope": "single_admission",',
            '      "question_type": "adversarial",',
            '      "question": "During the hospitalization from YYYY-MM-DD to YYYY-MM-DD, which culture result ultimately confirmed the infection source?",',
            f'      "answer": "{CANONICAL_ADVERSARIAL_ANSWER}",',
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
        system_message=SINGLE_ADMISSION_ADVERSARIAL_SYSTEM_MESSAGE,
        user_message=user_message,
        question_count=int(question_count),
        context_json=context_json,
    )


def render_cross_admission_regular_qa_prompt(
    summary_contexts: list[dict[str, Any]],
    *,
    question_count: int,
) -> RenderedQAPrompt:
    context_json = canonical_json_dumps(summary_contexts)
    user_message = "\n".join(
        [
            f"Generate exactly {int(question_count)} hard answerable cross-admission short-answer question-answer pairs from the chronological admission summaries below.",
            "",
            "Allowed question_type values:",
            *[f"- {question_type}" for question_type in CROSS_ADMISSION_REGULAR_QUESTION_TYPES],
            "",
            "Requirements:",
            "- Every question must be hard.",
            "- Every question must require at least 2 admissions.",
            "- Prefer questions that require synthesis across 3 or more admissions whenever possible.",
            "- No question should be answerable from one admission summary alone.",
            "- Favor progression, comparison, and longitudinal frequency reasoning.",
            f"- Generate exactly {int(question_count)} questions total.",
            "- Every item must use one of the allowed regular cross-admission question_type values.",
            "- Every answer must be a short open answer, preferably 1 to 7 words, and never more than 10 words.",
            "- When possible, form the short answer using exact wording from the summaries rather than paraphrasing.",
            "- If exact wording is awkward, use only light normalization while keeping the wording as close as possible to the summaries.",
            "- Do not use yes/no questions or yes/no answers.",
            "- The question should sound natural and should not mention raw identifiers.",
            "- Evidence must list only the admissions actually used.",
            "- Evidence must cite at least 2 unique admissions for every item.",
            "- evidence.admissions must use only the exact admission_id_for_evidence_only values already present in the provided context.",
            "- Do not invent, alter, extend, or reformat admission ids.",
            "",
            "Chronological admission summaries:",
            context_json,
            "",
            "Return JSON with this exact shape:",
            "{",
            '  "qas": [',
            "    {",
            '      "qa_id": "...",',
            '      "scope": "cross_admission",',
            '      "question_type": "longitudinal_progression",',
            '      "question": "Across the admissions, which problem most clearly showed recurrent respiratory decline?",',
            '      "answer": "respiratory failure",',
            '      "evidence": {',
            '        "admissions": ["...", "...", "..."]',
            "      }",
            "    }",
            "  ]",
            "}",
        ]
    )
    return RenderedQAPrompt(
        system_message=CROSS_ADMISSION_REGULAR_SYSTEM_MESSAGE,
        user_message=user_message,
        question_count=int(question_count),
        context_json=context_json,
    )


def render_cross_admission_adversarial_qa_prompt(
    summary_contexts: list[dict[str, Any]],
    *,
    question_count: int,
) -> RenderedQAPrompt:
    context_json = canonical_json_dumps(summary_contexts)
    user_message = "\n".join(
        [
            f"Generate exactly {int(question_count)} adversarial cross-admission short-answer question-answer pairs from the chronological admission summaries below.",
            "",
            "Allowed question_type values:",
            "- adversarial",
            "",
            "Requirements:",
            "- Every question must be unsupported by the full provided context.",
            "- Every item in this stream must use question_type 'adversarial'.",
            "- Every question must still sound natural, specific, and medically plausible.",
            "- Ask about a plausible longitudinal pattern, recurrence, or comparison that is not explicitly supported anywhere in the summaries.",
            f"- Generate exactly {int(question_count)} questions total.",
            f"- The expected answer for every item must be exactly: {CANONICAL_ADVERSARIAL_ANSWER!r}.",
            "- Evidence must cite at least 2 unique admissions for every item.",
            "- Questions are written for benchmark users, not for the patient.",
            "- Do not use second-person wording like 'you' or 'your' in the question.",
            "- Do not use wording like 'he' or 'she' to refer to the patient or the doctor. Use third-person phrasing such as 'the patient', 'the patient's symptoms', or 'the doctor' when needed.",
            "- Do not use trick wording, double negatives, yes/no questions, or vague references.",
            "- Evidence admissions should be the admissions that create the plausible confusion, not an answer span.",
            "- evidence.admissions must use only the exact admission_id_for_evidence_only values already present in the provided context.",
            "- Do not invent, alter, extend, or reformat admission ids.",
            "",
            "Chronological admission summaries:",
            context_json,
            "",
            "Return JSON with this exact shape:",
            "{",
            '  "qas": [',
            "    {",
            '      "qa_id": "...",',
            '      "scope": "cross_admission",',
            '      "question_type": "adversarial",',
            '      "question": "Across the admissions, which hospitalization first documented the recurrent resistant organism?",',
            f'      "answer": "{CANONICAL_ADVERSARIAL_ANSWER}",',
            '      "evidence": {',
            '        "admissions": ["...", "..."]',
            "      }",
            "    }",
            "  ]",
            "}",
        ]
    )
    return RenderedQAPrompt(
        system_message=CROSS_ADMISSION_ADVERSARIAL_SYSTEM_MESSAGE,
        user_message=user_message,
        question_count=int(question_count),
        context_json=context_json,
    )


def append_qa_repair_message(user_message: str, validation_error: str) -> str:
    return "\n\n".join([user_message, QA_REPAIR_MESSAGE, f"Validation errors: {validation_error}"])
