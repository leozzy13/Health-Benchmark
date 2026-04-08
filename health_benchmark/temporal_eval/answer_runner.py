from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from .answer_prompting import render_answer_prompt
from .types import AnswerBatchResponse, AnswerPrediction, EvalQuestion, QuestionBatch


MAX_INCOMPLETE_BATCH_ATTEMPTS = 4


@dataclass(frozen=True)
class BatchValidationResult:
    predictions_by_id: dict[str, str]
    missing_ids: list[str]


def run_answer_batches(
    llm_client: Any,
    batches: list[QuestionBatch],
    *,
    context_text: str,
    model_name: str,
    save_raw_response: bool,
    max_output_tokens: int,
) -> tuple[list[AnswerPrediction], list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    predictions: list[AnswerPrediction] = []
    raw_records: list[dict[str, Any]] = []
    error_records: list[dict[str, Any]] = []
    failed_statuses: dict[str, str] = {}

    for batch in batches:
        batch_predictions, batch_raw_records, batch_errors, batch_failed_statuses = _run_batch_with_retries(
            llm_client,
            batch,
            context_text=context_text,
            model_name=model_name,
            save_raw_response=save_raw_response,
            max_output_tokens=max_output_tokens,
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
) -> tuple[list[AnswerPrediction], list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    raw_records: list[dict[str, Any]] = []
    collected_predictions: dict[str, str] = {}
    remaining_questions = list(batch.questions)
    attempts_used = 0

    while remaining_questions:
        attempts_used += 1
        rendered = render_answer_prompt(
            context_text=context_text,
            questions=[question.model_question() for question in remaining_questions],
            retry_count=attempts_used - 1,
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
            if attempts_used >= MAX_INCOMPLETE_BATCH_ATTEMPTS:
                raise ValueError(f"Missing qa_ids in answer batch response: {validation.missing_ids[:3]}")
            remaining_questions = [
                question
                for question in remaining_questions
                if question.qa_id in validation.missing_ids
            ]
        except Exception as exc:
            error_records = [
                _build_error_record(
                    stage="answer_batch",
                    exc=exc,
                    batch=batch,
                    attempt_index=attempts_used,
                    pending_questions=remaining_questions,
                )
            ]
            failed_statuses = {
                question.qa_id: "answer_failed"
                for question in remaining_questions
            }
            return _materialize_predictions(batch, collected_predictions), raw_records, error_records, failed_statuses

    return _materialize_predictions(batch, collected_predictions), raw_records, [], {}


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
) -> dict[str, Any]:
    raw_record: dict[str, Any] = {
        "batch_id": batch.batch_id,
        "attempt_index": int(attempt_index),
        "qa_ids": batch.qa_ids(),
        "requested_qa_ids": [question.qa_id for question in requested_questions],
        "returned_qa_ids": [str(answer.qa_id).strip() for answer in parsed.answers],
        "missing_qa_ids": list(missing_ids),
        "status": "completed" if not missing_ids else "partial_missing_qa_ids",
        "model": model_name,
        "api_usage": llm_result.usage,
        "response_id": llm_result.response_id,
        "latency_ms": llm_result.latency_ms,
        "raw_structured_response": parsed.model_dump(mode="json"),
    }
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
) -> dict[str, Any]:
    return {
        "timestamp": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "stage": stage,
        "error_type": type(exc).__name__,
        "error_kind": _classify_error(exc),
        "message": str(exc),
        "batch_id": batch.batch_id,
        "attempt_index": int(attempt_index),
        "qa_ids": batch.qa_ids(),
        "pending_qa_ids": [question.qa_id for question in pending_questions],
    }


def _classify_error(exc: Exception) -> str:
    message = str(exc).lower()
    if "json" in message or "schema" in message or "parsed" in message or "qa_id" in message:
        return "format_error"
    return "api_error"
