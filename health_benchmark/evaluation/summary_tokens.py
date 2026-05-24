from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from .context_renderer import render_conversation_lines
from .hf_tokenizer import count_batch_text_tokens
from .io_utils import load_json
from .loader import read_patient_manifest
from ..scripts.utils import round_mean, write_json


def refresh_patient_summary_tokens(
    *,
    benchmark_root: Path,
    patient_manifest: Path | None,
    subject_ids: Sequence[int] | None,
    tokenizer_model: str,
) -> dict[str, Any]:
    target_ids = _resolve_target_ids(
        patient_manifest=patient_manifest,
        subject_ids=subject_ids,
    )
    rows: list[dict[str, Any]] = []
    for subject_id in target_ids:
        rows.append(
            refresh_one_patient_summary_tokens(
                patient_root=benchmark_root / str(subject_id),
                tokenizer_model=tokenizer_model,
            )
        )
    return {
        "benchmark_root": str(benchmark_root),
        "tokenizer_model": tokenizer_model,
        "updated_subject_ids": [row["subject_id"] for row in rows],
        "updated_count": len(rows),
        "patients": rows,
    }


def refresh_one_patient_summary_tokens(
    *,
    patient_root: Path,
    tokenizer_model: str,
) -> dict[str, Any]:
    combined_path = patient_root / "combined_conversation.json"
    summary_path = patient_root / "patient_summary.json"
    if not combined_path.exists():
        raise ValueError(f"Missing combined_conversation.json: {combined_path}")
    if not summary_path.exists():
        raise ValueError(f"Missing patient_summary.json: {summary_path}")

    combined = load_json(combined_path)
    summary = load_json(summary_path)
    admissions = combined.get("admissions")
    if not isinstance(admissions, list) or not admissions:
        raise ValueError(f"combined_conversation.json must contain admissions: {combined_path}")

    rendered_admissions = [
        render_conversation_lines(list(admission.get("conversation_lines") or []))
        for admission in admissions
    ]
    token_counts, tokenizer_name = count_batch_text_tokens(
        model_name=tokenizer_model,
        tokenizer_name=tokenizer_model,
        texts=rendered_admissions,
    )
    turn_counts = [
        len(admission.get("conversation_lines") or [])
        for admission in admissions
    ]
    summary["conversation_stats"] = {
        **dict(summary.get("conversation_stats") or {}),
        "mean_turns": round_mean(turn_counts),
        "total_turns": int(sum(turn_counts)),
        "mean_tokens": round_mean(token_counts),
        "total_tokens": int(sum(token_counts)),
        "tokenizer": tokenizer_name,
        "token_count_format": "flat_time_speaker_text",
    }
    write_json(summary_path, summary)
    return {
        "subject_id": str(summary.get("subject_id") or combined.get("subject_id") or patient_root.name),
        "patient_root": str(patient_root),
        "total_turns": int(sum(turn_counts)),
        "total_tokens": int(sum(token_counts)),
        "tokenizer": tokenizer_name,
    }


def _resolve_target_ids(
    *,
    patient_manifest: Path | None,
    subject_ids: Sequence[int] | None,
) -> list[int]:
    has_manifest = patient_manifest is not None
    has_subject_ids = bool(subject_ids)
    if has_manifest == has_subject_ids:
        raise ValueError("Provide exactly one of patient_manifest or subject_ids.")
    if patient_manifest is not None:
        return read_patient_manifest(patient_manifest)
    return [int(subject_id) for subject_id in subject_ids or []]
