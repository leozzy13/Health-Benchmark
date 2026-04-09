from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from .io_utils import load_json
from .types import LoadedPatientArtifacts


def resolve_patient_root(*, output_root: Path, subject_id: int | None, patient_dir: Path | None) -> tuple[Path, int]:
    if (subject_id is None) == (patient_dir is None):
        raise ValueError("Exactly one of subject_id or patient_dir must be provided.")
    if patient_dir is not None:
        resolved_root = patient_dir.expanduser().resolve()
        if not resolved_root.exists():
            raise ValueError(f"Patient directory does not exist: {resolved_root}")
        return resolved_root, derive_subject_id(resolved_root)
    resolved_subject_id = int(subject_id)
    resolved_root = (output_root / str(resolved_subject_id)).resolve()
    if not resolved_root.exists():
        raise ValueError(f"Patient directory does not exist: {resolved_root}")
    return resolved_root, resolved_subject_id


def resolve_patient_targets(
    *,
    output_root: Path,
    subject_id: int | None,
    subject_ids: Sequence[int] | None,
    patient_manifest: Path | None,
    patient_dir: Path | None,
) -> list[tuple[int, Path]]:
    options_selected = sum(
        choice is not None
        for choice in (subject_id, subject_ids if subject_ids else None, patient_manifest, patient_dir)
    )
    if options_selected != 1:
        raise ValueError("Provide exactly one of subject_id, subject_ids, patient_manifest, or patient_dir.")
    if patient_dir is not None:
        resolved_root, derived_subject_id = resolve_patient_root(
            output_root=output_root,
            subject_id=None,
            patient_dir=patient_dir,
        )
        return [(derived_subject_id, resolved_root)]
    if patient_manifest is not None:
        manifest_ids = read_patient_manifest(patient_manifest)
        return [
            _resolve_target_tuple(output_root=output_root, subject_id=value)
            for value in manifest_ids
        ]
    if subject_ids:
        return [
            _resolve_target_tuple(output_root=output_root, subject_id=int(value))
            for value in subject_ids
        ]
    resolved_root, resolved_subject_id = resolve_patient_root(
        output_root=output_root,
        subject_id=subject_id,
        patient_dir=None,
    )
    return [(resolved_subject_id, resolved_root)]


def _resolve_target_tuple(*, output_root: Path, subject_id: int) -> tuple[int, Path]:
    resolved_root, resolved_subject_id = resolve_patient_root(
        output_root=output_root,
        subject_id=subject_id,
        patient_dir=None,
    )
    return resolved_subject_id, resolved_root


def read_patient_manifest(path: Path) -> list[int]:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise ValueError(f"Patient manifest does not exist: {resolved}")
    subject_ids: list[int] = []
    seen: set[int] = set()
    for line_number, raw_line in enumerate(resolved.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            subject_id = int(stripped)
        except ValueError as exc:
            raise ValueError(f"Invalid subject_id in patient manifest line {line_number}: {stripped}") from exc
        if subject_id in seen:
            continue
        seen.add(subject_id)
        subject_ids.append(subject_id)
    if not subject_ids:
        raise ValueError(f"Patient manifest must contain at least one subject_id: {resolved}")
    return subject_ids


def derive_subject_id(patient_root: Path) -> int:
    if patient_root.name.isdigit():
        return int(patient_root.name)
    combined_path = patient_root / "combined_conversation.json"
    if combined_path.exists():
        payload = load_json(combined_path)
        subject_id = str(payload.get("subject_id") or "").strip()
        if subject_id.isdigit():
            return int(subject_id)
    raise ValueError(f"Could not derive subject_id from patient directory: {patient_root}")


def load_patient_artifacts(patient_root: Path) -> LoadedPatientArtifacts:
    combined_path = patient_root / "combined_conversation.json"
    benchmark_path = patient_root / "benchmark_qa.json"
    if not combined_path.exists():
        raise ValueError(f"Missing combined_conversation.json: {combined_path}")
    if not benchmark_path.exists():
        raise ValueError(f"Missing benchmark_qa.json: {benchmark_path}")
    combined_payload = load_json(combined_path)
    benchmark_payload = load_json(benchmark_path)
    _validate_combined_payload(combined_payload)
    _validate_benchmark_payload(benchmark_payload)
    return LoadedPatientArtifacts(
        subject_id=str(combined_payload["subject_id"]),
        patient_root=patient_root,
        combined_payload=combined_payload,
        benchmark_payload=benchmark_payload,
    )


def _validate_combined_payload(payload: Any) -> None:
    if not isinstance(payload, dict):
        raise ValueError("combined_conversation.json must contain a JSON object")
    if not str(payload.get("subject_id") or "").strip():
        raise ValueError("combined_conversation.json must contain a non-empty subject_id")
    admissions = payload.get("admissions")
    if not isinstance(admissions, list) or not admissions:
        raise ValueError("combined_conversation.json must contain a non-empty admissions list")
    for index, admission in enumerate(admissions, start=1):
        if not isinstance(admission, dict):
            raise ValueError(f"combined_conversation admissions[{index}] must be an object")
        if not str(admission.get("hadm_id") or "").strip():
            raise ValueError(f"combined_conversation admissions[{index}].hadm_id must be non-empty")
        lines = admission.get("conversation_lines")
        if not isinstance(lines, list) or not lines:
            raise ValueError(f"combined_conversation admissions[{index}].conversation_lines must be non-empty")


def _validate_benchmark_payload(payload: Any) -> None:
    if not isinstance(payload, dict):
        raise ValueError("benchmark_qa.json must contain a JSON object")
    qas = payload.get("qas")
    if not isinstance(qas, list) or not qas:
        raise ValueError("benchmark_qa.json must contain a non-empty qas list")
