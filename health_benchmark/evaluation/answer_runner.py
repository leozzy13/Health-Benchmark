from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

from health_benchmark.scripts.llm_client import StructuredResponseValidationError, structured_content_candidates

from .answer_prompting import render_answer_prompt
from .types import AnswerBatchResponse, AnswerItem, AnswerPrediction, EvalQuestion, QuestionBatch


@dataclass(frozen=True)
class BatchValidationResult:
    predictions_by_id: dict[str, str]
    missing_ids: list[str]


@dataclass(frozen=True)
class AnswerBatchRepair:
    parsed: AnswerBatchResponse
    method: str
    original_payload: Any


def run_answer_batches(
    llm_client: Any,
    batches: list[QuestionBatch],
    *,
    context_text: str | None = None,
    model_name: str,
    save_raw_response: bool,
    max_output_tokens: int,
    retry_limit: int = 3,
) -> tuple[list[AnswerPrediction], list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    predictions: list[AnswerPrediction] = []
    raw_records: list[dict[str, Any]] = []
    error_records: list[dict[str, Any]] = []
    failed_statuses: dict[str, str] = {}

    for batch in batches:
        batch_predictions, batch_raw_records, batch_errors, batch_failed_statuses = _run_batch_with_retries(
            llm_client,
            batch,
            context_text=batch.context_text if batch.context_text is not None else str(context_text or ""),
            model_name=model_name,
            save_raw_response=save_raw_response,
            max_output_tokens=max_output_tokens,
            retry_limit=retry_limit,
        )
        predictions.extend(batch_predictions)
        raw_records.extend(batch_raw_records)
        error_records.extend(batch_errors)
        failed_statuses.update(batch_failed_statuses)

    return predictions, raw_records, error_records, failed_statuses


def _run_batch_with_retries(
    llm_client: Any,
    batch: QuestionBatch,
    *,
    context_text: str,
    model_name: str,
    save_raw_response: bool,
    max_output_tokens: int,
    retry_limit: int,
) -> tuple[list[AnswerPrediction], list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    raw_records: list[dict[str, Any]] = []
    error_records: list[dict[str, Any]] = []
    collected_predictions: dict[str, str] = {}
    remaining_questions = list(batch.questions)
    attempts_used = 0
    max_attempts = max(1, int(retry_limit) + 1)

    while remaining_questions and attempts_used < max_attempts:
        attempts_used += 1
        context_description = str(batch.context_record.get("prompt_context_description", "conversation"))
        context_payload_key = str(batch.context_record.get("prompt_context_payload_key", "patient_conversation"))
        rendered = render_answer_prompt(
            context_text=context_text,
            questions=[question.model_question() for question in remaining_questions],
            retry_count=attempts_used - 1,
            context_description=context_description,
            context_payload_key=context_payload_key,
        )
        try:
            llm_result = llm_client.generate_structured_response(
                rendered.system_message,
                rendered.user_message,
                AnswerBatchResponse,
                max_output_tokens=max_output_tokens,
            )
            parsed = AnswerBatchResponse.model_validate(llm_result.parsed_output)
            validation = _validate_batch_predictions(remaining_questions, parsed)
            raw_records.append(
                _build_raw_record(
                    batch=batch,
                    requested_questions=remaining_questions,
                    model_name=model_name,
                    llm_result=llm_result,
                    parsed=parsed,
                    save_raw_response=save_raw_response,
                    attempt_index=attempts_used,
                    missing_ids=validation.missing_ids,
                )
            )
            collected_predictions.update(validation.predictions_by_id)
            if not validation.missing_ids:
                break
            remaining_questions = [
                question
                for question in remaining_questions
                if question.qa_id in validation.missing_ids
            ]
            if attempts_used >= max_attempts:
                exc = ValueError(f"Missing qa_ids in answer batch response: {validation.missing_ids[:3]}")
                error_records.append(
                    _build_error_record(
                        stage="answer_batch",
                        exc=exc,
                        batch=batch,
                        attempt_index=attempts_used,
                        pending_questions=remaining_questions,
                        max_attempts=max_attempts,
                        will_retry=False,
                    )
                )
                break
        except Exception as exc:
            repaired = _try_repair_structured_response_error(exc, remaining_questions)
            if repaired is not None:
                validation = _validate_batch_predictions(remaining_questions, repaired.parsed)
                raw_records.append(
                    _build_raw_record(
                        batch=batch,
                        requested_questions=remaining_questions,
                        model_name=model_name,
                        llm_result=_llm_result_from_validation_error(exc),
                        parsed=repaired.parsed,
                        save_raw_response=save_raw_response,
                        attempt_index=attempts_used,
                        missing_ids=validation.missing_ids,
                        schema_repair_applied=True,
                        schema_repair_method=repaired.method,
                        schema_repair_original_payload=repaired.original_payload,
                        original_schema_error=str(exc),
                    )
                )
                collected_predictions.update(validation.predictions_by_id)
                if not validation.missing_ids:
                    break
                remaining_questions = [
                    question
                    for question in remaining_questions
                    if question.qa_id in validation.missing_ids
                ]
                if attempts_used >= max_attempts:
                    exc = ValueError(f"Missing qa_ids in answer batch response: {validation.missing_ids[:3]}")
                    error_records.append(
                        _build_error_record(
                            stage="answer_batch",
                            exc=exc,
                            batch=batch,
                            attempt_index=attempts_used,
                            pending_questions=remaining_questions,
                            max_attempts=max_attempts,
                            will_retry=False,
                        )
                    )
                    break
                continue
            will_retry = attempts_used < max_attempts and not _is_context_length_error(exc)
            error_records.append(
                _build_error_record(
                    stage="answer_batch",
                    exc=exc,
                    batch=batch,
                    attempt_index=attempts_used,
                    pending_questions=remaining_questions,
                    max_attempts=max_attempts,
                    will_retry=will_retry,
                )
            )
            if will_retry:
                continue
            break

    failed_statuses = {
        question.qa_id: "answer_failed"
        for question in remaining_questions
        if question.qa_id not in collected_predictions
    }
    return _materialize_predictions(batch, collected_predictions), raw_records, error_records, failed_statuses


def _validate_batch_predictions(
    questions: list[EvalQuestion],
    parsed: AnswerBatchResponse,
) -> BatchValidationResult:
    expected_ids = [question.qa_id for question in questions]
    by_id: dict[str, str] = {}
    for answer in parsed.answers:
        qa_id = str(answer.qa_id).strip()
        if qa_id in by_id:
            raise ValueError(f"Duplicate qa_id in answer batch response: {qa_id}")
        if qa_id not in expected_ids:
            raise ValueError(f"Unknown qa_id in answer batch response: {qa_id}")
        by_id[qa_id] = str(answer.prediction).strip()
    missing_ids = [qa_id for qa_id in expected_ids if qa_id not in by_id]
    return BatchValidationResult(
        predictions_by_id=by_id,
        missing_ids=missing_ids,
    )


def summarize_schema_repair_metrics(raw_records: list[dict[str, Any]]) -> dict[str, int]:
    repaired_records = [record for record in raw_records if bool(record.get("schema_repair_applied"))]
    return {
        "schema_repaired_batch_count": len(repaired_records),
        "schema_repaired_prediction_count": sum(
            len(record.get("returned_qa_ids") or [])
            for record in repaired_records
        ),
        "schema_order_repaired_batch_count": sum(
            1
            for record in repaired_records
            if record.get("schema_repair_method") == "ordered_string_list"
        ),
    }


def _try_repair_structured_response_error(
    exc: Exception,
    pending_questions: list[EvalQuestion],
) -> AnswerBatchRepair | None:
    if not isinstance(exc, StructuredResponseValidationError):
        return None
    if exc.schema_name != "AnswerBatchResponse":
        return None
    last_error: Exception | None = None
    for candidate in structured_content_candidates(exc.content):
        try:
            payload = json.loads(candidate, strict=False)
        except json.JSONDecodeError as parse_exc:
            last_error = parse_exc
            continue
        try:
            return _repair_answer_payload(payload, pending_questions)
        except ValueError as repair_exc:
            last_error = repair_exc
            if candidate.strip() == exc.content.strip():
                return None
            continue
    if last_error is None:
        return None
    return None


def _repair_answer_payload(payload: Any, pending_questions: list[EvalQuestion]) -> AnswerBatchRepair:
    if isinstance(payload, dict) and isinstance(payload.get("answers"), list):
        return _repair_answer_items(
            payload["answers"],
            pending_questions,
            method="answers_alias_keys",
            original_payload=payload,
        )
    if isinstance(payload, list):
        if all(isinstance(item, str) for item in payload):
            if len(payload) != len(pending_questions):
                raise ValueError("Cannot repair string-list answer response with wrong length.")
            answers = [
                AnswerItem(qa_id=question.qa_id, prediction=str(prediction).strip())
                for question, prediction in zip(pending_questions, payload, strict=True)
            ]
            return AnswerBatchRepair(
                parsed=AnswerBatchResponse(answers=answers),
                method="ordered_string_list",
                original_payload=payload,
            )
        return _repair_answer_items(
            payload,
            pending_questions,
            method="top_level_item_list",
            original_payload=payload,
        )
    if isinstance(payload, dict):
        return _repair_answer_items(
            [payload],
            pending_questions,
            method="single_item_object",
            original_payload=payload,
        )
    raise ValueError(f"Unsupported answer response payload type: {type(payload).__name__}")


def _repair_answer_items(
    items: list[Any],
    pending_questions: list[EvalQuestion],
    *,
    method: str,
    original_payload: Any,
) -> AnswerBatchRepair:
    if not all(isinstance(item, dict) for item in items):
        raise ValueError("Cannot repair answer items that are not JSON objects.")
    question_by_id = {question.qa_id: question for question in pending_questions}
    question_text_to_ids: dict[str, list[str]] = {}
    for question in pending_questions:
        question_text_to_ids.setdefault(_normalize_question_text(question.question), []).append(question.qa_id)
    answers: list[AnswerItem] = []
    seen_ids: set[str] = set()
    used_question_text = False
    used_alias_keys = False
    for item in items:
        qa_id = _coerce_repaired_qa_id(item, question_by_id, question_text_to_ids)
        if qa_id not in question_by_id:
            raise ValueError(f"Unknown qa_id in repaired answer response: {qa_id}")
        if qa_id in seen_ids:
            raise ValueError(f"Duplicate qa_id in repaired answer response: {qa_id}")
        seen_ids.add(qa_id)
        if "qa_id" not in item or "prediction" not in item:
            used_alias_keys = True
        if "qa_id" not in item and "question_id" not in item:
            used_question_text = True
        prediction = _coerce_repaired_prediction(item)
        answers.append(AnswerItem(qa_id=qa_id, prediction=prediction))
    repair_method = method
    if used_question_text:
        repair_method = f"{method}_question_text"
    elif used_alias_keys:
        repair_method = f"{method}_alias_keys"
    return AnswerBatchRepair(
        parsed=AnswerBatchResponse(answers=answers),
        method=repair_method,
        original_payload=original_payload,
    )


def _coerce_repaired_qa_id(
    item: dict[str, Any],
    question_by_id: dict[str, EvalQuestion],
    question_text_to_ids: dict[str, list[str]],
) -> str:
    qa_id = str(item.get("qa_id") or item.get("question_id") or "").strip()
    if qa_id:
        return qa_id
    question_text = item.get("question")
    if not isinstance(question_text, str):
        raise ValueError("Repaired answer item is missing qa_id/question_id.")
    matching_ids = question_text_to_ids.get(_normalize_question_text(question_text), [])
    if len(matching_ids) != 1:
        raise ValueError("Question text did not match exactly one pending question.")
    matched_id = matching_ids[0]
    if matched_id not in question_by_id:
        raise ValueError(f"Unknown qa_id from question text: {matched_id}")
    return matched_id


def _coerce_repaired_prediction(item: dict[str, Any]) -> str:
    if "prediction" in item:
        prediction = item["prediction"]
    elif "answer" in item:
        prediction = item["answer"]
    else:
        raise ValueError("Repaired answer item is missing prediction/answer.")
    if prediction is None:
        raise ValueError("Repaired answer item has null prediction.")
    return str(prediction).strip()


def _normalize_question_text(value: str) -> str:
    return " ".join(str(value or "").split())


def _llm_result_from_validation_error(exc: Exception) -> Any:
    if isinstance(exc, StructuredResponseValidationError):
        return SimpleNamespace(
            parsed_output={},
            raw_response=exc.raw_response,
            usage=exc.usage,
            response_id=exc.response_id,
            latency_ms=exc.latency_ms,
        )
    return SimpleNamespace(
        parsed_output={},
        raw_response={},
        usage={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        response_id=None,
        latency_ms=0,
    )


def _materialize_predictions(
    batch: QuestionBatch,
    collected_predictions: dict[str, str],
) -> list[AnswerPrediction]:
    return [
        AnswerPrediction(
            qa_id=question.qa_id,
            prediction=collected_predictions[question.qa_id],
            batch_id=batch.batch_id,
        )
        for question in batch.questions
        if question.qa_id in collected_predictions
    ]


def _build_raw_record(
    *,
    batch: QuestionBatch,
    requested_questions: list[EvalQuestion],
    model_name: str,
    llm_result: Any,
    parsed: AnswerBatchResponse,
    save_raw_response: bool,
    attempt_index: int,
    missing_ids: list[str],
    schema_repair_applied: bool = False,
    schema_repair_method: str | None = None,
    schema_repair_original_payload: Any | None = None,
    original_schema_error: str | None = None,
) -> dict[str, Any]:
    raw_record: dict[str, Any] = {
        "batch_id": batch.batch_id,
        "attempt_index": int(attempt_index),
        "qa_ids": batch.qa_ids(),
        "requested_qa_ids": [question.qa_id for question in requested_questions],
        "returned_qa_ids": [str(answer.qa_id).strip() for answer in parsed.answers],
        "missing_qa_ids": list(missing_ids),
        "status": "completed" if not missing_ids else "partial_missing_qa_ids",
        "answer_context_strategy": batch.context_record.get("strategy"),
        "model": model_name,
        "api_usage": llm_result.usage,
        "response_id": llm_result.response_id,
        "latency_ms": llm_result.latency_ms,
        "raw_structured_response": parsed.model_dump(mode="json"),
    }
    if schema_repair_applied:
        raw_record.update(
            {
                "schema_repair_applied": True,
                "schema_repair_method": str(schema_repair_method or "unknown"),
                "schema_repair_original_payload": schema_repair_original_payload,
                "original_schema_error": str(original_schema_error or ""),
            }
        )
    if save_raw_response:
        raw_record["raw_response"] = llm_result.raw_response
    return raw_record


def _build_error_record(
    *,
    stage: str,
    exc: Exception,
    batch: QuestionBatch,
    attempt_index: int,
    pending_questions: list[EvalQuestion],
    max_attempts: int,
    will_retry: bool,
) -> dict[str, Any]:
    return {
        "timestamp": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "stage": stage,
        "error_type": type(exc).__name__,
        "error_kind": _classify_error(exc),
        "message": str(exc),
        "batch_id": batch.batch_id,
        "attempt_index": int(attempt_index),
        "max_attempts": int(max_attempts),
        "will_retry": bool(will_retry),
        "qa_ids": batch.qa_ids(),
        "pending_qa_ids": [question.qa_id for question in pending_questions],
    }


def _classify_error(exc: Exception) -> str:
    message = str(exc).lower()
    if _is_context_length_error(exc):
        return "context_length_error"
    if "json" in message or "schema" in message or "parsed" in message or "qa_id" in message:
        return "format_error"
    return "api_error"


def _is_context_length_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "maximum context length" in message
        or ("input_tokens" in message and "please reduce" in message)
        or ("context length" in message and "requested" in message)
    )
