from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError as PydanticValidationError

from .utils import parse_dt


SINGLE_ADMISSION_QUESTION_TYPES = (
    "medical_reasoning",
    "temporal_reasoning",
    "care_plan_rationale",
    "adversarial",
)

CROSS_ADMISSION_QUESTION_TYPES = (
    "longitudinal_progression",
    "cross_admission_comparison",
    "frequency_pattern",
    "adversarial",
)

SINGLE_ADMISSION_REGULAR_QUESTION_TYPES = (
    "medical_reasoning",
    "temporal_reasoning",
    "care_plan_rationale",
)

CROSS_ADMISSION_REGULAR_QUESTION_TYPES = (
    "longitudinal_progression",
    "cross_admission_comparison",
    "frequency_pattern",
)

CANONICAL_ADVERSARIAL_ANSWER = "the question is not answerable"
ADVERSARIAL_ANSWER_ALIASES = frozenset(
    {
        "the question is not answerable",
        "not answerable from the record",
        "not answerable from the provided context",
        "cannot be determined from the record",
        "no information available",
    }
)
MAX_NON_ADVERSARIAL_ANSWER_WORDS = 10


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
        "care_plan_rationale",
        "adversarial",
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
        "cross_admission_comparison",
        "frequency_pattern",
        "adversarial",
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
    expected_adversarial_count: int,
    allowed_question_types: tuple[str, ...] | None = None,
    qa_index_start: int = 1,
) -> dict[str, Any]:
    try:
        parsed = SingleAdmissionQAFile.model_validate(value)
    except PydanticValidationError as exc:
        raise QAValidationError(str(exc)) from exc

    if len(parsed.qas) != int(expected_count):
        raise QAValidationError(
            f"single-admission QA count must equal {expected_count}; got {len(parsed.qas)}"
        )
    _validate_adversarial_mix(
        parsed.qas,
        expected_total=int(expected_count),
        expected_adversarial_count=int(expected_adversarial_count),
        label="single-admission",
    )
    _validate_allowed_question_types(
        parsed.qas,
        allowed_question_types=allowed_question_types,
        label="single-admission",
    )

    question_prefix = _build_single_admission_question_prefix(
        admission_start=admission_start,
        admission_end=admission_end,
    )
    normalized_items: list[dict[str, Any]] = []
    for offset, item in enumerate(parsed.qas):
        index = offset + 1
        question = _normalize_single_admission_question(item.question, question_prefix)
        answer = _normalize_answer(item.answer)
        if not item.qa_id.strip():
            raise QAValidationError(f"qas[{index}].qa_id must be non-empty")
        if not question:
            raise QAValidationError(f"qas[{index}].question must be non-empty")
        if not answer:
            raise QAValidationError(f"qas[{index}].answer must be non-empty")
        answer = _normalize_answer_for_question_type(
            question_type=item.question_type,
            answer=answer,
            index=index,
        )
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
                "qa_id": f"{subject_id}_{hadm_id}_q{int(qa_index_start) + offset:02d}",
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
    expected_adversarial_count: int,
    admission_aliases: dict[str, str] | None = None,
    allowed_question_types: tuple[str, ...] | None = None,
    qa_index_start: int = 1,
    coerce_adversarial_question_type_from_answer: bool = False,
) -> dict[str, Any]:
    try:
        parsed = CrossAdmissionQAFile.model_validate(value)
    except PydanticValidationError as exc:
        raise QAValidationError(str(exc)) from exc

    if len(parsed.qas) != int(expected_count):
        raise QAValidationError(
            f"cross-admission QA count must equal {expected_count}; got {len(parsed.qas)}"
        )
    prepared_items = [
        _prepare_cross_admission_item_for_validation(
            item,
            index=index,
            coerce_adversarial_question_type_from_answer=coerce_adversarial_question_type_from_answer,
        )
        for index, item in enumerate(parsed.qas, start=1)
    ]
    _validate_adversarial_mix(
        prepared_items,
        expected_total=int(expected_count),
        expected_adversarial_count=int(expected_adversarial_count),
        label="cross-admission",
    )
    _validate_allowed_question_types(
        prepared_items,
        allowed_question_types=allowed_question_types,
        label="cross-admission",
    )

    chronology = {hadm_id: index for index, hadm_id in enumerate(ordered_hadm_ids)}
    alias_map = {
        _normalize_admission_reference(alias): str(hadm_id)
        for alias, hadm_id in (admission_aliases or {}).items()
    }
    normalized_items: list[dict[str, Any]] = []
    for offset, prepared_item in enumerate(prepared_items):
        item = prepared_item["item"]
        index = int(prepared_item["index"])
        question = item.question.strip()
        answer = str(prepared_item["answer"])
        question_type = str(prepared_item["question_type"])
        if not item.qa_id.strip():
            raise QAValidationError(f"qas[{index}].qa_id must be non-empty")
        if not question:
            raise QAValidationError(f"qas[{index}].question must be non-empty")
        if not answer:
            raise QAValidationError(f"qas[{index}].answer must be non-empty")

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
        if unknown:
            raise QAValidationError(
                f"qas[{index}].evidence.admissions contain unknown hadm_ids: {unknown[:3]}"
            )
        unique_admissions: list[str] = []
        seen_admissions: set[str] = set()
        for admission in resolved_admissions:
            if admission in seen_admissions:
                continue
            seen_admissions.add(admission)
            unique_admissions.append(admission)
        if len(unique_admissions) < 2:
            raise QAValidationError(f"qas[{index}].evidence.admissions must contain at least 2 unique admissions")
        admissions = sorted(unique_admissions, key=lambda admission: chronology[admission])

        normalized_items.append(
            {
                "qa_id": f"{subject_id}_cross_q{int(qa_index_start) + offset:02d}",
                "scope": "cross_admission",
                "question_type": question_type,
                "question": question,
                "answer": answer,
                "evidence": {
                    "admissions": admissions,
                },
            }
        )
    return {"qas": normalized_items}


def _validate_allowed_question_types(
    items: list[Any],
    *,
    allowed_question_types: tuple[str, ...] | None,
    label: str,
) -> None:
    if allowed_question_types is None:
        return
    allowed = set(allowed_question_types)
    for index, item in enumerate(items, start=1):
        question_type = _item_question_type(item)
        if question_type not in allowed:
            allowed_text = ", ".join(sorted(allowed))
            raise QAValidationError(
                f"{label} qas[{index}].question_type must be one of: {allowed_text}"
            )


def _validate_adversarial_mix(
    items: list[Any],
    *,
    expected_total: int,
    expected_adversarial_count: int,
    label: str,
) -> None:
    if len(items) != int(expected_total):
        raise QAValidationError(f"{label} QA count must equal {expected_total}; got {len(items)}")
    adversarial_count = sum(1 for item in items if _item_question_type(item) == "adversarial")
    if adversarial_count != int(expected_adversarial_count):
        raise QAValidationError(
            f"{label} adversarial QA count must equal {expected_adversarial_count}; got {adversarial_count}"
        )
    non_adversarial_count = len(items) - adversarial_count
    expected_non_adversarial_count = int(expected_total) - int(expected_adversarial_count)
    if non_adversarial_count != expected_non_adversarial_count:
        raise QAValidationError(
            f"{label} non-adversarial QA count must equal {expected_non_adversarial_count}; got {non_adversarial_count}"
        )


def _normalize_answer_for_question_type(
    *,
    question_type: str,
    answer: str,
    index: int,
) -> str:
    normalized_key = _normalize_answer_key(answer)
    if question_type == "adversarial":
        if normalized_key not in ADVERSARIAL_ANSWER_ALIASES:
            raise QAValidationError(
                f"qas[{index}].answer must use the canonical adversarial abstention or an accepted alias"
            )
        return CANONICAL_ADVERSARIAL_ANSWER

    if normalized_key in ADVERSARIAL_ANSWER_ALIASES:
        raise QAValidationError(f"qas[{index}].answer must not use the adversarial abstention phrase")

    if len(answer.split()) > MAX_NON_ADVERSARIAL_ANSWER_WORDS:
        raise QAValidationError(
            f"qas[{index}].answer must contain at most {MAX_NON_ADVERSARIAL_ANSWER_WORDS} words for non-adversarial questions"
        )
    return answer


def _prepare_cross_admission_item_for_validation(
    item: CrossAdmissionQAItem,
    *,
    index: int,
    coerce_adversarial_question_type_from_answer: bool,
) -> dict[str, Any]:
    answer = _normalize_answer(item.answer)
    question_type = item.question_type
    if coerce_adversarial_question_type_from_answer and _normalize_answer_key(answer) in ADVERSARIAL_ANSWER_ALIASES:
        question_type = "adversarial"
    answer = _normalize_answer_for_question_type(
        question_type=question_type,
        answer=answer,
        index=index,
    )
    return {
        "item": item,
        "index": index,
        "question_type": question_type,
        "answer": answer,
    }


def _item_question_type(item: Any) -> str:
    if isinstance(item, dict):
        return str(item["question_type"])
    return str(item.question_type)


def _normalize_answer(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _normalize_answer_key(text: str) -> str:
    return _normalize_answer(text).lower()


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
    normalized = _normalize_admission_reference(reference)
    if normalized in chronology:
        return normalized
    if normalized in alias_map:
        return alias_map[normalized]
    return None


def _normalize_admission_reference(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip())
