from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .utils import parse_dt
from .writers import benchmark_qa_path, combined_conversation_path, cross_admission_qa_path, patient_summary_path, qa_path


def verify_patient_outputs(patient_root: Path, *, expect_qa: bool = True) -> dict[str, Any]:
    resolved_root = patient_root.expanduser().resolve()
    if not resolved_root.exists() or not resolved_root.is_dir():
        raise ValueError(f"Patient directory does not exist: {resolved_root}")

    combined = _load_json_object(combined_conversation_path(resolved_root), "combined_conversation.json")
    patient_summary = _load_json_object(patient_summary_path(resolved_root), "patient_summary.json")
    processed_hadm_ids = _extract_processed_hadm_ids(combined, patient_summary)
    if not processed_hadm_ids:
        raise ValueError(f"No processed_hadm_ids found in patient outputs: {resolved_root}")

    subject_id = str(combined.get("subject_id") or patient_summary.get("subject_id") or "").strip()
    if not subject_id:
        raise ValueError(f"Missing subject_id in patient outputs: {resolved_root}")
    if str(patient_summary.get("subject_id") or subject_id) != subject_id:
        raise ValueError(f"patient_summary.json subject_id does not match combined_conversation.json: {resolved_root}")

    admission_summaries: list[dict[str, Any]] = []
    total_single_admission_qas = 0

    for hadm_id in processed_hadm_ids:
        admission_dir = resolved_root / hadm_id
        if not admission_dir.exists() or not admission_dir.is_dir():
            raise ValueError(f"Missing admission directory for hadm_id={hadm_id}: {admission_dir}")

        _load_json_object(admission_dir / "formed_packet.json", f"{hadm_id}/formed_packet.json")
        _load_json_object(admission_dir / "prompt_record.json", f"{hadm_id}/prompt_record.json")
        conversation = _load_json_object(admission_dir / "conversation.json", f"{hadm_id}/conversation.json")
        summary = _load_json_object(admission_dir / "summary.json", f"{hadm_id}/summary.json")

        _validate_conversation_artifact(conversation, hadm_id=hadm_id)
        _validate_summary_artifact(summary, hadm_id=hadm_id)

        admission_summary: dict[str, Any] = {
            "hadm_id": hadm_id,
            "conversation_turns": len(conversation["conversation_lines"]),
        }
        if expect_qa:
            qa_payload = _load_json_object(qa_path(admission_dir), f"{hadm_id}/qa.json")
            qa_items = _require_qas(qa_payload, label=f"{hadm_id}/qa.json")
            total_single_admission_qas += len(qa_items)
            admission_summary["qa_count"] = len(qa_items)
        admission_summaries.append(admission_summary)

    result = {
        "patient_dir": str(resolved_root),
        "subject_id": subject_id,
        "processed_hadm_ids": processed_hadm_ids,
        "processed_admissions": len(processed_hadm_ids),
        "admissions": admission_summaries,
    }

    if expect_qa:
        cross_payload = _load_json_object(cross_admission_qa_path(resolved_root), "cross_admission_qa.json")
        benchmark_payload = _load_json_object(benchmark_qa_path(resolved_root), "benchmark_qa.json")
        cross_qas = _require_qas(cross_payload, label="cross_admission_qa.json")
        benchmark_qas = _require_qas(benchmark_payload, label="benchmark_qa.json")
        expected_total = total_single_admission_qas + len(cross_qas)
        if len(benchmark_qas) != expected_total:
            raise ValueError(
                "benchmark_qa.json count does not match the sum of admission qa.json files and cross_admission_qa.json"
            )
        result["qa_counts"] = {
            "single_admission_total": total_single_admission_qas,
            "cross_admission_total": len(cross_qas),
            "benchmark_total": len(benchmark_qas),
        }

    return result


def _extract_processed_hadm_ids(combined: dict[str, Any], patient_summary: dict[str, Any]) -> list[str]:
    combined_ids = _normalize_id_list(combined.get("processed_hadm_ids"))
    summary_ids = _normalize_id_list(patient_summary.get("processed_hadm_ids"))
    processed_hadm_ids = combined_ids or summary_ids
    if not processed_hadm_ids:
        return []
    if combined_ids and summary_ids and combined_ids != summary_ids:
        raise ValueError("processed_hadm_ids mismatch between combined_conversation.json and patient_summary.json")
    return processed_hadm_ids


def _normalize_id_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    for item in value:
        text = str(item).strip()
        if not text:
            continue
        normalized.append(text)
    return normalized


def _require_qas(payload: dict[str, Any], *, label: str) -> list[dict[str, Any]]:
    qas = payload.get("qas")
    if not isinstance(qas, list) or not qas:
        raise ValueError(f"{label} must contain a non-empty qas list")
    normalized_qas: list[dict[str, Any]] = []
    for index, qa in enumerate(qas, start=1):
        if not isinstance(qa, dict):
            raise ValueError(f"{label} qas[{index}] must be an object")
        normalized_qas.append(qa)
    return normalized_qas


def _validate_conversation_artifact(conversation: dict[str, Any], *, hadm_id: str) -> None:
    if str(conversation.get("hadm_id") or "").strip() != hadm_id:
        raise ValueError(f"conversation.json hadm_id mismatch for admission {hadm_id}")
    lines = conversation.get("conversation_lines")
    if not isinstance(lines, list) or not lines:
        raise ValueError(f"conversation.json must contain non-empty conversation_lines for admission {hadm_id}")
    for index, line in enumerate(lines, start=1):
        if not isinstance(line, dict):
            raise ValueError(f"conversation_lines[{index}] must be an object for admission {hadm_id}")
        turn_number = line.get("turn_number")
        if not isinstance(turn_number, int) or turn_number <= 0:
            raise ValueError(f"conversation_lines[{index}].turn_number must be a positive integer for admission {hadm_id}")


def _validate_summary_artifact(summary: dict[str, Any], *, hadm_id: str) -> None:
    required_fields = ("admission_start", "admission_end", "summary_paragraph", "problems")
    for field in required_fields:
        if field not in summary:
            raise ValueError(f"summary.json missing required field {field} for admission {hadm_id}")
    if parse_dt(summary["admission_start"]) is None or parse_dt(summary["admission_end"]) is None:
        raise ValueError(f"summary.json admission timestamps must be valid for admission {hadm_id}")
    if not isinstance(summary["summary_paragraph"], str) or not summary["summary_paragraph"].strip():
        raise ValueError(f"summary.json summary_paragraph must be non-empty for admission {hadm_id}")
    if not isinstance(summary["problems"], list):
        raise ValueError(f"summary.json problems must be a list for admission {hadm_id}")


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise ValueError(f"Missing required file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return payload
