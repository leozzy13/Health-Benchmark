from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import BenchmarkConfig
from .utils import ensure_dir, write_json


@dataclass
class AdmissionArtifactPaths:
    admission_dir: Path
    formed_packet_json: Path
    prompt_record_json: Path
    conversation_json: Path
    summary_json: Path


def benchmark_root(config: BenchmarkConfig) -> Path:
    return config.output.root


def staging_root(config: BenchmarkConfig) -> Path:
    return benchmark_root(config) / config.runtime.staging_dir_name


def patient_dir(config: BenchmarkConfig, subject_id: int) -> Path:
    return benchmark_root(config) / str(subject_id)


def patient_staging_dir(config: BenchmarkConfig, subject_id: int) -> Path:
    return staging_root(config) / str(subject_id)


def runs_root(config: BenchmarkConfig) -> Path:
    return benchmark_root(config) / "runs"


def batch_run_dir(config: BenchmarkConfig, run_id: str) -> Path:
    return runs_root(config) / run_id


def combined_conversation_path(root: Path) -> Path:
    return root / "combined_conversation.json"


def patient_summary_path(root: Path) -> Path:
    return root / "patient_summary.json"


def qa_path(root: Path) -> Path:
    return root / "qa.json"


def cross_admission_qa_path(root: Path) -> Path:
    return root / "cross_admission_qa.json"


def benchmark_qa_path(root: Path) -> Path:
    return root / "benchmark_qa.json"


def batch_summary_path(root: Path) -> Path:
    return root / "batch_summary.json"


def admission_artifact_paths(root: Path, hadm_id: int) -> AdmissionArtifactPaths:
    admission_dir = root / str(hadm_id)
    return AdmissionArtifactPaths(
        admission_dir=admission_dir,
        formed_packet_json=admission_dir / "formed_packet.json",
        prompt_record_json=admission_dir / "prompt_record.json",
        conversation_json=admission_dir / "conversation.json",
        summary_json=admission_dir / "summary.json",
    )


def write_packet(paths: AdmissionArtifactPaths, packet: dict[str, Any]) -> None:
    ensure_dir(paths.admission_dir)
    write_json(paths.formed_packet_json, packet)


def write_prompt_record(paths: AdmissionArtifactPaths, prompt_record: dict[str, Any]) -> None:
    ensure_dir(paths.admission_dir)
    write_json(paths.prompt_record_json, prompt_record)


def write_conversation(paths: AdmissionArtifactPaths, conversation: dict[str, Any]) -> None:
    ensure_dir(paths.admission_dir)
    write_json(paths.conversation_json, conversation)


def write_summary(paths: AdmissionArtifactPaths, summary: dict[str, Any]) -> None:
    ensure_dir(paths.admission_dir)
    write_json(paths.summary_json, summary)


def write_combined_conversation(root: Path, payload: dict[str, Any]) -> None:
    ensure_dir(root)
    write_json(combined_conversation_path(root), payload)


def write_patient_summary(root: Path, payload: dict[str, Any]) -> None:
    ensure_dir(root)
    write_json(patient_summary_path(root), payload)


def write_qa(root: Path, payload: dict[str, Any]) -> None:
    ensure_dir(root)
    write_json(qa_path(root), payload)


def write_cross_admission_qa(root: Path, payload: dict[str, Any]) -> None:
    ensure_dir(root)
    write_json(cross_admission_qa_path(root), payload)


def write_benchmark_qa(root: Path, payload: dict[str, Any]) -> None:
    ensure_dir(root)
    write_json(benchmark_qa_path(root), payload)


def write_batch_summary(root: Path, payload: dict[str, Any]) -> None:
    ensure_dir(root)
    write_json(batch_summary_path(root), payload)
