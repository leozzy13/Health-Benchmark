from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Sequence

from .judge_prompting import render_answerable_judge_prompt
from .types import AnswerableJudgeBatchResponse


def run_llm_judge_batches(
    llm_client: Any,
    scored_rows: list[dict[str, Any]],
    *,
    judge_model_name: str,
    save_raw_response: bool,
    max_output_tokens: int,
    batch_size: int,
    retry_limit: int = 3,
) -> list[dict[str, Any]]:
    row_map = {str(row["qa_id"]): row for row in scored_rows}
    answerable_items = [
        {
            "qa_id": str(row["qa_id"]),
            "question": str(row["question"]),
            "gold_answer": str(row["gold_answer"]),
            "candidate_answer": str(row["prediction"]),
        }
        for row in scored_rows
        if row.get("status") == "scored" and not bool(row.get("is_adversarial"))
    ]

    raw_records: list[dict[str, Any]] = []
    raw_records.extend(
        _run_answerable_batches(
            llm_client,
            answerable_items,
            row_map=row_map,
            judge_model_name=judge_model_name,
            save_raw_response=save_raw_response,
            max_output_tokens=max_output_tokens,
            batch_size=batch_size,
            retry_limit=retry_limit,
        )
    )

    missing_scores = [
        qa_id
        for qa_id, row in row_map.items()
        if row.get("status") == "scored"
        and not bool(row.get("is_adversarial"))
        and row.get("llm_judge_score") is None
    ]
    if missing_scores:
        raise ValueError(f"Missing llm_judge_score for qa_ids: {missing_scores[:3]}")
    return raw_records


def _run_answerable_batches(
    llm_client: Any,
    items: Sequence[dict[str, Any]],
    *,
    row_map: dict[str, dict[str, Any]],
    judge_model_name: str,
    save_raw_response: bool,
    max_output_tokens: int,
    batch_size: int,
    retry_limit: int,
) -> list[dict[str, Any]]:
    raw_records: list[dict[str, Any]] = []
    for batch in _chunk_items(items, batch_size):
        max_attempts = max(1, int(retry_limit) + 1)
        last_exc: Exception | None = None
        for attempt_index in range(1, max_attempts + 1):
            rendered = render_answerable_judge_prompt(batch)
            try:
                llm_result = llm_client.generate_structured_response(
                    rendered.system_message,
                    rendered.user_message,
                    AnswerableJudgeBatchResponse,
                    max_output_tokens=max_output_tokens,
                )
                parsed = AnswerableJudgeBatchResponse.model_validate(llm_result.parsed_output)
                returned_scores = _validate_returned_scores(batch, parsed.judgments)
            except Exception as exc:
                last_exc = exc
                if attempt_index < max_attempts:
                    continue
                break
            for qa_id, score in returned_scores.items():
                row_map[qa_id]["llm_judge_score"] = float(score)
            raw_records.append(
                _build_raw_record(
                    judge_kind="answerable",
                    requested_items=batch,
                    returned_scores=returned_scores,
                    judge_model_name=judge_model_name,
                    llm_result=llm_result,
                    parsed_output=parsed.model_dump(mode="json"),
                    save_raw_response=save_raw_response,
                    attempt_index=attempt_index,
                    max_attempts=max_attempts,
                )
            )
            last_exc = None
            break
        if last_exc is not None:
            requested_ids = [str(item["qa_id"]) for item in batch]
            raise RuntimeError(
                f"LLM judge batch failed after {max_attempts} attempts for qa_ids {requested_ids[:3]}: {last_exc}"
            ) from last_exc
    return raw_records


def _validate_returned_scores(
    requested_items: Sequence[dict[str, Any]],
    returned_items: Sequence[Any],
) -> dict[str, float]:
    expected_ids = [str(item["qa_id"]) for item in requested_items]
    returned_scores: dict[str, float] = {}
    for item in returned_items:
        qa_id = str(item.qa_id).strip()
        if qa_id in returned_scores:
            raise ValueError(f"Duplicate qa_id in LLM judge response: {qa_id}")
        if qa_id not in expected_ids:
            raise ValueError(f"Unknown qa_id in LLM judge response: {qa_id}")
        returned_scores[qa_id] = float(item.score)
    missing_ids = [qa_id for qa_id in expected_ids if qa_id not in returned_scores]
    if missing_ids:
        raise ValueError(f"Missing qa_ids in LLM judge response: {missing_ids[:3]}")
    return returned_scores


def _build_raw_record(
    *,
    judge_kind: str,
    requested_items: Sequence[dict[str, Any]],
    returned_scores: dict[str, float],
    judge_model_name: str,
    llm_result: Any,
    parsed_output: dict[str, Any],
    save_raw_response: bool,
    attempt_index: int,
    max_attempts: int,
) -> dict[str, Any]:
    raw_record: dict[str, Any] = {
        "timestamp": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "judge_kind": judge_kind,
        "judge_model": judge_model_name,
        "requested_qa_ids": [str(item["qa_id"]) for item in requested_items],
        "returned_qa_ids": list(returned_scores.keys()),
        "attempt_index": int(attempt_index),
        "max_attempts": int(max_attempts),
        "scores_by_qa_id": returned_scores,
        "api_usage": llm_result.usage,
        "response_id": llm_result.response_id,
        "latency_ms": llm_result.latency_ms,
        "raw_structured_response": parsed_output,
    }
    if save_raw_response:
        raw_record["raw_response"] = llm_result.raw_response
    return raw_record


def _chunk_items(items: Sequence[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive for LLM judging, got: {batch_size}")
    return [
        list(items[index : index + batch_size])
        for index in range(0, len(items), batch_size)
    ]
