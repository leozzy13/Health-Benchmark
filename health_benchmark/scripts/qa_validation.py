from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError as PydanticValidationError

from .utils import parse_dt


SINGLE_ADMISSION_QUESTION_TYPES = (
    "medical_reasoning",
    "temporal_reasoning",
    "test_result_interpretation",
    "care_plan_rationale",
    "patient_concern_explanation",
    "knowledge_augmented_inference",
)

CROSS_ADMISSION_QUESTION_TYPES = (
    "longitudinal_progression",
    "recurrence_pattern",
    "cross_admission_comparison",
    "first_last_occurrence",
    "longitudinal_temporal_reasoning",
    "longitudinal_medical_inference",
    "frequency_pattern",
)


class QAValidationError(ValueError):
    pass


class SingleAdmissionEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    admissions: list[str]
    turn_ids: list[int]


class SingleAdmissionQAItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    qa_id: str
    scope: Literal["single_admission"]
    question_type: Literal[
        "medical_reasoning",
        "temporal_reasoning",
        "test_result_interpretation",
        "care_plan_rationale",
        "patient_concern_explanation",
        "knowledge_augmented_inference",
    ]
    question: str
    answer: str
    evidence: SingleAdmissionEvidence


class SingleAdmissionQAFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    qas: list[SingleAdmissionQAItem]


class CrossAdmissionEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    admissions: list[str]


class CrossAdmissionQAItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    qa_id: str
    scope: Literal["cross_admission"]
    question_type: Literal[
        "longitudinal_progression",
        "recurrence_pattern",
        "cross_admission_comparison",
        "first_last_occurrence",
        "longitudinal_temporal_reasoning",
        "longitudinal_medical_inference",
        "frequency_pattern",
    ]
    question: str
    answer: str
    evidence: CrossAdmissionEvidence


class CrossAdmissionQAFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    qas: list[CrossAdmissionQAItem]


LEADING_ADMISSION_ANCHOR_RE = re.compile(
    r"^\s*(?:during|in|for)\s+(?:this|the)\s+"
    r"(?:hospitalization|admission|hospital stay|stay)"
    r"(?:\s+(?:from|between)\s+[^,]+?(?:\s+to\s+[^,]+?)?|\s+on\s+[^,]+?)?"
    r"\s*,?\s*",
    re.IGNORECASE,
)


def validate_single_admission_qa(
    value: Any,
    *,
    subject_id: str,
    hadm_id: str,
    admission_start: str,
    admission_end: str,
    valid_turn_ids: set[int],
    expected_count: int,
) -> dict[str, Any]:
    try:
        parsed = SingleAdmissionQAFile.model_validate(value)
    except PydanticValidationError as exc:
        raise QAValidationError(str(exc)) from exc

    if len(parsed.qas) != int(expected_count):
        raise QAValidationError(
            f"single-admission QA count must equal {expected_count}; got {len(parsed.qas)}"
        )

    question_prefix = _build_single_admission_question_prefix(
        admission_start=admission_start,
        admission_end=admission_end,
    )
    normalized_items: list[dict[str, Any]] = []
    for index, item in enumerate(parsed.qas, start=1):
        question = _normalize_single_admission_question(item.question, question_prefix)
        answer = item.answer.strip()
        if not item.qa_id.strip():
            raise QAValidationError(f"qas[{index}].qa_id must be non-empty")
        if not question:
            raise QAValidationError(f"qas[{index}].question must be non-empty")
        if not answer:
            raise QAValidationError(f"qas[{index}].answer must be non-empty")
        if _word_count(answer) > 20:
            raise QAValidationError(f"qas[{index}].answer must be at most 20 words")
        if not item.evidence.admissions:
            raise QAValidationError(f"qas[{index}].evidence.admissions must be non-empty")
        if any(str(admission) != hadm_id for admission in item.evidence.admissions):
            raise QAValidationError(
                f"qas[{index}].evidence.admissions must reference only hadm_id={hadm_id}"
            )
        if not item.evidence.turn_ids:
            raise QAValidationError(f"qas[{index}].evidence.turn_ids must be non-empty")

        turn_ids = sorted(set(int(turn_id) for turn_id in item.evidence.turn_ids))
        if any(turn_id <= 0 for turn_id in turn_ids):
            raise QAValidationError(f"qas[{index}].evidence.turn_ids must be positive integers")
        invalid_turn_ids = [turn_id for turn_id in turn_ids if turn_id not in valid_turn_ids]
        if invalid_turn_ids:
            raise QAValidationError(
                f"qas[{index}].evidence.turn_ids contain unknown turn numbers: {invalid_turn_ids[:3]}"
            )

        normalized_items.append(
            {
                "qa_id": f"{subject_id}_{hadm_id}_q{index:02d}",
                "scope": "single_admission",
                "question_type": item.question_type,
                "question": question,
                "answer": answer,
                "evidence": {
                    "admissions": [hadm_id],
                    "turn_ids": turn_ids,
                },
            }
        )
    return {"qas": normalized_items}


def validate_cross_admission_qa(
    value: Any,
    *,
    subject_id: str,
    ordered_hadm_ids: list[str],
    expected_count: int,
    admission_aliases: dict[str, str] | None = None,
) -> dict[str, Any]:
    try:
        parsed = CrossAdmissionQAFile.model_validate(value)
    except PydanticValidationError as exc:
        raise QAValidationError(str(exc)) from exc

    if len(parsed.qas) != int(expected_count):
        raise QAValidationError(
            f"cross-admission QA count must equal {expected_count}; got {len(parsed.qas)}"
        )

    chronology = {hadm_id: index for index, hadm_id in enumerate(ordered_hadm_ids)}
    alias_map = {
        _normalize_admission_reference(alias): str(hadm_id)
        for alias, hadm_id in (admission_aliases or {}).items()
    }
    normalized_items: list[dict[str, Any]] = []
    for index, item in enumerate(parsed.qas, start=1):
        question = item.question.strip()
        answer = item.answer.strip()
        if not item.qa_id.strip():
            raise QAValidationError(f"qas[{index}].qa_id must be non-empty")
        if not question:
            raise QAValidationError(f"qas[{index}].question must be non-empty")
        if not answer:
            raise QAValidationError(f"qas[{index}].answer must be non-empty")
        if _word_count(answer) > 20:
            raise QAValidationError(f"qas[{index}].answer must be at most 20 words")

        resolved_admissions: list[str] = []
        unknown: list[str] = []
        for admission in item.evidence.admissions:
            resolved = _resolve_cross_admission_reference(
                str(admission),
                chronology=chronology,
                alias_map=alias_map,
            )
            if resolved is None:
                unknown.append(str(admission))
                continue
            resolved_admissions.append(resolved)
        if len(resolved_admissions) < 2:
            raise QAValidationError(f"qas[{index}].evidence.admissions must contain at least 2 admissions")
        if len(set(resolved_admissions)) != len(resolved_admissions):
            raise QAValidationError(f"qas[{index}].evidence.admissions must not contain duplicates")
        if unknown:
            raise QAValidationError(
                f"qas[{index}].evidence.admissions contain unknown hadm_ids: {unknown[:3]}"
            )
        admissions = sorted(resolved_admissions, key=lambda admission: chronology[admission])

        normalized_items.append(
            {
                "qa_id": f"{subject_id}_cross_q{index:02d}",
                "scope": "cross_admission",
                "question_type": item.question_type,
                "question": question,
                "answer": answer,
                "evidence": {
                    "admissions": admissions,
                },
            }
        )
    return {"qas": normalized_items}


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def _build_single_admission_question_prefix(*, admission_start: str, admission_end: str) -> str:
    start_dt = parse_dt(admission_start)
    end_dt = parse_dt(admission_end)
    if start_dt is None or end_dt is None:
        raise QAValidationError("single-admission QA normalization requires valid admission_start and admission_end")

    start_date = start_dt.date().isoformat()
    end_date = end_dt.date().isoformat()
    if start_date == end_date:
        return f"During the hospitalization on {start_date}, "
    return f"During the hospitalization from {start_date} to {end_date}, "


def _normalize_single_admission_question(question: str, prefix: str) -> str:
    stripped = question.strip()
    if not stripped:
        return stripped
    if stripped.startswith(prefix):
        return stripped

    matched = LEADING_ADMISSION_ANCHOR_RE.match(stripped)
    if matched is not None:
        stripped = stripped[matched.end() :].lstrip(" ,:;-")
    return prefix + stripped


def _resolve_cross_admission_reference(
    reference: str,
    *,
    chronology: dict[str, int],
    alias_map: dict[str, str],
) -> str | None:
    stripped = reference.strip()
    if stripped in chronology:
        return stripped
    return alias_map.get(_normalize_admission_reference(stripped))


def _normalize_admission_reference(reference: str) -> str:
    normalized = re.sub(r"\s+", " ", reference.strip())
    normalized = normalized.strip(" ,;:.")
    return normalized.lower()
