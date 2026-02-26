from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import BenchmarkConfig
from .utils import ensure_dir, write_json, write_jsonl


@dataclass(slots=True)
class AdmissionArtifactPaths:
    admission_dir: Path
    packet_json: Path
    input_data_manifest_json: Path
    prompt_record_json: Path
    model_call_record_json: Path
    conversation_jsonl: Path
    summary_json: Path
    raw_model_output_json: Path
    unlinked_notes_json: Path


def benchmark_root(config: BenchmarkConfig) -> Path:
    assert config.paths is not None
    # Simplified output layout requested by user: everything directly under output root.
    return config.paths.output_root


def cohort_csv_path(config: BenchmarkConfig, cohort_name: str = "top1000_by_admission_count") -> Path:
    return benchmark_root(config) / f"{cohort_name}.csv"


def patient_dir(config: BenchmarkConfig, subject_id: int) -> Path:
    return benchmark_root(config) / str(subject_id)


def patient_manifest_path(config: BenchmarkConfig, subject_id: int) -> Path:
    return patient_dir(config, subject_id) / "patient_manifest.json"


def patient_conversation_details_path(config: BenchmarkConfig, subject_id: int) -> Path:
    return patient_dir(config, subject_id) / "conversation_details.jsonl"


def patient_conversation_only_path(config: BenchmarkConfig, subject_id: int) -> Path:
    return patient_dir(config, subject_id) / "conversation_only.json"


def admission_artifact_paths(config: BenchmarkConfig, subject_id: int, hadm_id: int) -> AdmissionArtifactPaths:
    base = patient_dir(config, subject_id) / "admissions" / str(hadm_id)
    return AdmissionArtifactPaths(
        admission_dir=base,
        packet_json=base / "packet.json",
        input_data_manifest_json=base / "input_data_manifest.json",
        prompt_record_json=base / "prompt_record.json",
        model_call_record_json=base / "model_call_record.json",
        conversation_jsonl=base / "conversation.jsonl",
        summary_json=base / "summary.json",
        raw_model_output_json=base / "raw_model_output.json",
        unlinked_notes_json=base / "unlinked_notes.json",
    )


def write_packet(paths: AdmissionArtifactPaths, packet: dict[str, Any]) -> None:
    ensure_dir(paths.admission_dir)
    write_json(paths.packet_json, packet)


def write_input_data_manifest(paths: AdmissionArtifactPaths, manifest: dict[str, Any]) -> None:
    write_json(paths.input_data_manifest_json, manifest)


def write_prompt_record(paths: AdmissionArtifactPaths, record: dict[str, Any]) -> None:
    write_json(paths.prompt_record_json, record)


def write_model_call_record(paths: AdmissionArtifactPaths, record: dict[str, Any]) -> None:
    write_json(paths.model_call_record_json, record)


def write_conversation(paths: AdmissionArtifactPaths, conversation: list[dict[str, Any]]) -> None:
    write_jsonl(paths.conversation_jsonl, conversation)


def write_patient_conversation_details(
    config: BenchmarkConfig,
    subject_id: int,
    rows: list[dict[str, Any]],
) -> None:
    write_jsonl(patient_conversation_details_path(config, subject_id), rows)


def write_patient_conversation_only(
    config: BenchmarkConfig,
    subject_id: int,
    rows: list[dict[str, Any]],
) -> None:
    write_json(patient_conversation_only_path(config, subject_id), rows)


def write_summary(paths: AdmissionArtifactPaths, summary: dict[str, Any]) -> None:
    write_json(paths.summary_json, summary)


def write_raw_output(paths: AdmissionArtifactPaths, raw_output: dict[str, Any]) -> None:
    write_json(paths.raw_model_output_json, raw_output)


def write_unlinked_notes(paths: AdmissionArtifactPaths, rows: list[dict[str, Any]]) -> None:
    if rows:
        write_json(paths.unlinked_notes_json, {"radiology_notes": rows})
