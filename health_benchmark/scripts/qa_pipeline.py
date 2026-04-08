from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .config import BenchmarkConfig
from .qa_prompting import (
    append_qa_repair_message,
    render_cross_admission_adversarial_qa_prompt,
    render_cross_admission_regular_qa_prompt,
    render_single_admission_adversarial_qa_prompt,
    render_single_admission_regular_qa_prompt,
)
from .qa_validation import (
    CROSS_ADMISSION_REGULAR_QUESTION_TYPES,
    CrossAdmissionQAFile,
    QAValidationError,
    SINGLE_ADMISSION_REGULAR_QUESTION_TYPES,
    SingleAdmissionQAFile,
    validate_cross_admission_qa,
    validate_single_admission_qa,
)
from .utils import ensure_dir, parse_dt, utc_now_iso, write_json
from .writers import benchmark_qa_path, cross_admission_qa_path, qa_path

SINGLE_ADMISSION_REGULAR_COUNT = 2
SINGLE_ADMISSION_ADVERSARIAL_COUNT = 1
SINGLE_ADMISSION_QA_COUNT = SINGLE_ADMISSION_REGULAR_COUNT + SINGLE_ADMISSION_ADVERSARIAL_COUNT
DEFAULT_CROSS_ADMISSION_MAX_OUTPUT_TOKENS = 30000


@dataclass
class AdmissionQAContext:
    hadm_id: str
    admission_dir: Path
    conversation: dict[str, Any]
    summary: dict[str, Any]
    admission_start: str


def generate_patient_qa(
    config: BenchmarkConfig,
    llm_client: Any,
    *,
    subject_id: int | None = None,
    patient_dir: Path | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    patient_root, resolved_subject_id = _resolve_patient_root(config, subject_id=subject_id, patient_dir=patient_dir)
    admissions = _discover_patient_admissions(patient_root)
    if len(admissions) < 2:
        raise ValueError(
            f"Patient {resolved_subject_id} must have at least 2 admissions with conversation and summary artifacts for cross-admission QA."
        )
    admission_count = len(admissions)
    cross_admission_qa_count = compute_cross_admission_qa_count(admission_count)
    cross_adversarial_count = compute_cross_adversarial_count(cross_admission_qa_count)
    cross_regular_count = int(cross_admission_qa_count) - int(cross_adversarial_count)

    resolved_model = model_name or config.llm.model
    config.llm.model = resolved_model
    staging_root = _prepare_qa_staging_dir(config, patient_root, resolved_subject_id)

    summary_contexts = _build_summary_contexts(admissions)
    single_admission_outputs: list[tuple[AdmissionQAContext, dict[str, Any]]] = []
    llm_usage_rows: list[dict[str, int]] = []

    try:
        for admission in admissions:
            rendered_regular = render_single_admission_regular_qa_prompt(
                admission.conversation,
                question_count=SINGLE_ADMISSION_REGULAR_COUNT,
            )
            llm_result_regular, validated_regular = _generate_validated_qa_response(
                config=config,
                llm_client=llm_client,
                system_message=rendered_regular.system_message,
                user_message=rendered_regular.user_message,
                response_schema=SingleAdmissionQAFile,
                validator=lambda parsed_output, admission=admission: validate_single_admission_qa(
                    parsed_output,
                    subject_id=str(resolved_subject_id),
                    hadm_id=admission.hadm_id,
                    admission_start=str(admission.conversation["admission_start"]),
                    admission_end=str(admission.conversation["admission_end"]),
                    valid_turn_ids={int(line["turn_number"]) for line in admission.conversation["conversation_lines"]},
                    expected_count=SINGLE_ADMISSION_REGULAR_COUNT,
                    expected_adversarial_count=0,
                    allowed_question_types=SINGLE_ADMISSION_REGULAR_QUESTION_TYPES,
                    qa_index_start=1,
                ),
            )
            llm_usage_rows.append(llm_result_regular.usage)

            rendered_adversarial = render_single_admission_adversarial_qa_prompt(
                admission.conversation,
                summary_contexts,
                question_count=SINGLE_ADMISSION_ADVERSARIAL_COUNT,
            )
            llm_result_adversarial, validated_adversarial = _generate_validated_qa_response(
                config=config,
                llm_client=llm_client,
                system_message=rendered_adversarial.system_message,
                user_message=rendered_adversarial.user_message,
                response_schema=SingleAdmissionQAFile,
                validator=lambda parsed_output, admission=admission: validate_single_admission_qa(
                    parsed_output,
                    subject_id=str(resolved_subject_id),
                    hadm_id=admission.hadm_id,
                    admission_start=str(admission.conversation["admission_start"]),
                    admission_end=str(admission.conversation["admission_end"]),
                    valid_turn_ids={int(line["turn_number"]) for line in admission.conversation["conversation_lines"]},
                    expected_count=SINGLE_ADMISSION_ADVERSARIAL_COUNT,
                    expected_adversarial_count=SINGLE_ADMISSION_ADVERSARIAL_COUNT,
                    allowed_question_types=("adversarial",),
                    qa_index_start=SINGLE_ADMISSION_REGULAR_COUNT + 1,
                ),
            )
            llm_usage_rows.append(llm_result_adversarial.usage)

            combined_single_payload = {
                "qas": list(validated_regular["qas"]) + list(validated_adversarial["qas"])
            }
            _write_admission_qa(staging_root, admission.hadm_id, combined_single_payload)
            single_admission_outputs.append((admission, combined_single_payload))

        rendered_cross_regular = render_cross_admission_regular_qa_prompt(
            summary_contexts,
            question_count=int(cross_regular_count),
        )
        llm_result_cross_regular, validated_cross_regular = _generate_validated_qa_response(
            config=config,
            llm_client=llm_client,
            system_message=rendered_cross_regular.system_message,
            user_message=rendered_cross_regular.user_message,
            response_schema=CrossAdmissionQAFile,
            max_output_tokens=max(
                int(config.llm.max_output_tokens),
                DEFAULT_CROSS_ADMISSION_MAX_OUTPUT_TOKENS,
            ),
            validator=lambda parsed_output: validate_cross_admission_qa(
                parsed_output,
                subject_id=str(resolved_subject_id),
                ordered_hadm_ids=[admission.hadm_id for admission in admissions],
                expected_count=int(cross_regular_count),
                expected_adversarial_count=0,
                admission_aliases=_build_cross_admission_aliases(admissions),
                allowed_question_types=CROSS_ADMISSION_REGULAR_QUESTION_TYPES,
                qa_index_start=1,
            ),
        )
        llm_usage_rows.append(llm_result_cross_regular.usage)

        rendered_cross_adversarial = render_cross_admission_adversarial_qa_prompt(
            summary_contexts,
            question_count=int(cross_adversarial_count),
        )
        llm_result_cross_adversarial, validated_cross_adversarial = _generate_validated_qa_response(
            config=config,
            llm_client=llm_client,
            system_message=rendered_cross_adversarial.system_message,
            user_message=rendered_cross_adversarial.user_message,
            response_schema=CrossAdmissionQAFile,
            max_output_tokens=max(
                int(config.llm.max_output_tokens),
                DEFAULT_CROSS_ADMISSION_MAX_OUTPUT_TOKENS,
            ),
            validator=lambda parsed_output: validate_cross_admission_qa(
                parsed_output,
                subject_id=str(resolved_subject_id),
                ordered_hadm_ids=[admission.hadm_id for admission in admissions],
                expected_count=int(cross_adversarial_count),
                expected_adversarial_count=int(cross_adversarial_count),
                admission_aliases=_build_cross_admission_aliases(admissions),
                allowed_question_types=("adversarial",),
                qa_index_start=int(cross_regular_count) + 1,
                coerce_adversarial_question_type_from_answer=True,
            ),
        )
        llm_usage_rows.append(llm_result_cross_adversarial.usage)

        cross_admission_payload = {
            "qas": list(validated_cross_regular["qas"]) + list(validated_cross_adversarial["qas"])
        }
        write_json(cross_admission_qa_path(staging_root), cross_admission_payload)

        benchmark_items = [
            qa_item
            for _admission, payload in single_admission_outputs
            for qa_item in payload["qas"]
        ] + list(cross_admission_payload["qas"])
        random.Random(int(resolved_subject_id)).shuffle(benchmark_items)
        write_json(benchmark_qa_path(staging_root), {"qas": benchmark_items})
        _promote_qa_outputs(patient_root, staging_root, [admission.hadm_id for admission in admissions])
    except Exception:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise

    input_tokens = [row.get("input_tokens", 0) for row in llm_usage_rows]
    output_tokens = [row.get("output_tokens", 0) for row in llm_usage_rows]
    return {
        "subject_id": str(resolved_subject_id),
        "patient_dir": str(patient_root),
        "processed_admissions": len(admissions),
        "single_admission_qa_count": SINGLE_ADMISSION_QA_COUNT,
        "cross_admission_qa_count": int(cross_admission_qa_count),
        "cross_adversarial_count": int(cross_adversarial_count),
        "admission_count": int(admission_count),
        "total_qas": len(admissions) * SINGLE_ADMISSION_QA_COUNT + int(cross_admission_qa_count),
        "request_timestamp": utc_now_iso(),
        "provider": config.llm.provider,
        "model": config.llm.model,
        "llm_call_stats": {
            "calls": len(llm_usage_rows),
            "total_input_tokens": int(sum(input_tokens)),
            "total_output_tokens": int(sum(output_tokens)),
        },
    }


def compute_cross_admission_qa_count(admission_count: int) -> int:
    if int(admission_count) < 2:
        return 0
    return 3 * int(admission_count)


def compute_cross_adversarial_count(cross_count: int) -> int:
    return int(cross_count) // 3


def _build_summary_contexts(admissions: list[AdmissionQAContext]) -> list[dict[str, Any]]:
    return [
        {
            "admission_id_for_evidence_only": admission.hadm_id,
            "admission_start": admission.summary["admission_start"],
            "admission_end": admission.summary["admission_end"],
            "summary_paragraph": admission.summary["summary_paragraph"],
            "problems": admission.summary["problems"],
        }
        for admission in admissions
    ]


def _generate_validated_qa_response(
    *,
    config: BenchmarkConfig,
    llm_client: Any,
    system_message: str,
    user_message: str,
    response_schema: type[Any],
    validator: Callable[[dict[str, Any]], dict[str, Any]],
    max_output_tokens: int | None = None,
) -> tuple[Any, dict[str, Any]]:
    max_attempts = max(1, int(config.llm.max_retries) + 1)
    current_user_message = user_message
    latest_error: Exception | None = None

    for _attempt in range(1, max_attempts + 1):
        try:
            llm_result = llm_client.generate_structured_response(
                system_message,
                current_user_message,
                response_schema,
                max_output_tokens=max_output_tokens,
            )
            return llm_result, validator(llm_result.parsed_output)
        except (QAValidationError, RuntimeError) as exc:
            latest_error = exc
            current_user_message = append_qa_repair_message(user_message, str(exc))

    if isinstance(latest_error, QAValidationError):
        raise latest_error
    raise RuntimeError(f"QA generation failed after {max_attempts} attempts: {latest_error}")


def _resolve_patient_root(
    config: BenchmarkConfig,
    *,
    subject_id: int | None,
    patient_dir: Path | None,
) -> tuple[Path, int]:
    if (subject_id is None) == (patient_dir is None):
        raise ValueError("Exactly one of subject_id or patient_dir must be provided for QA generation.")

    if patient_dir is not None:
        resolved_root = patient_dir.expanduser().resolve()
        if not resolved_root.exists():
            raise ValueError(f"Patient directory does not exist: {resolved_root}")
        derived_subject_id = _derive_subject_id(resolved_root)
        return resolved_root, derived_subject_id

    resolved_subject_id = int(subject_id)
    resolved_root = (config.output.root / str(resolved_subject_id)).resolve()
    if not resolved_root.exists():
        raise ValueError(f"Patient directory does not exist: {resolved_root}")
    return resolved_root, resolved_subject_id


def _derive_subject_id(patient_root: Path) -> int:
    if patient_root.name.isdigit():
        return int(patient_root.name)

    combined_path = patient_root / "combined_conversation.json"
    if combined_path.exists():
        data = _load_json(combined_path, "combined_conversation.json")
        subject_id = str(data.get("subject_id") or "").strip()
        if subject_id.isdigit():
            return int(subject_id)
    raise ValueError(f"Could not derive subject_id from patient directory: {patient_root}")


def _discover_patient_admissions(patient_root: Path) -> list[AdmissionQAContext]:
    admission_contexts: list[AdmissionQAContext] = []
    admission_dirs = [path for path in patient_root.iterdir() if path.is_dir() and path.name.isdigit()]
    if not admission_dirs:
        raise ValueError(f"No admission folders found in patient directory: {patient_root}")

    for admission_dir in admission_dirs:
        conversation_path = admission_dir / "conversation.json"
        summary_path = admission_dir / "summary.json"
        if not conversation_path.exists():
            raise ValueError(f"Missing conversation.json for admission {admission_dir.name}: {conversation_path}")
        if not summary_path.exists():
            raise ValueError(f"Missing summary.json for admission {admission_dir.name}: {summary_path}")

        conversation = _load_json(conversation_path, "conversation.json")
        summary = _load_json(summary_path, "summary.json")
        _validate_conversation_artifact(conversation, conversation_path)
        _validate_summary_artifact(summary, summary_path)
        admission_contexts.append(
            AdmissionQAContext(
                hadm_id=str(admission_dir.name),
                admission_dir=admission_dir,
                conversation=conversation,
                summary=summary,
                admission_start=str(summary["admission_start"]),
            )
        )

    admission_contexts.sort(key=lambda item: (item.admission_start, item.hadm_id))
    return admission_contexts


def _prepare_qa_staging_dir(config: BenchmarkConfig, patient_root: Path, subject_id: int) -> Path:
    staging_parent = patient_root.parent / config.runtime.staging_dir_name
    staging_root = staging_parent / f"qa_{subject_id}"
    if staging_root.exists():
        shutil.rmtree(staging_root)
    ensure_dir(staging_root)
    return staging_root


def _build_cross_admission_aliases(admissions: list[AdmissionQAContext]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for admission in admissions:
        hadm_id = admission.hadm_id
        start = str(admission.summary["admission_start"])
        end = str(admission.summary["admission_end"])
        aliases[start] = hadm_id
        aliases[end] = hadm_id
        aliases[f"{start} to {end}"] = hadm_id
        aliases[f"from {start} to {end}"] = hadm_id

        start_dt = parse_dt(start)
        end_dt = parse_dt(end)
        if start_dt is None or end_dt is None:
            continue
        start_date = start_dt.date().isoformat()
        end_date = end_dt.date().isoformat()
        if start_date == end_date:
            aliases[start_date] = hadm_id
            aliases[f"on {start_date}"] = hadm_id
        else:
            aliases[f"{start_date} to {end_date}"] = hadm_id
            aliases[f"from {start_date} to {end_date}"] = hadm_id
    return aliases


def _write_admission_qa(staging_root: Path, hadm_id: str, payload: dict[str, Any]) -> None:
    write_json(qa_path(staging_root / hadm_id), payload)


def _promote_qa_outputs(patient_root: Path, staging_root: Path, hadm_ids: list[str]) -> None:
    if not patient_root.exists():
        raise RuntimeError(f"Refusing to promote QA outputs into missing patient directory: {patient_root}")

    for hadm_id in hadm_ids:
        staged = qa_path(staging_root / hadm_id)
        target_dir = patient_root / hadm_id
        ensure_dir(target_dir)
        staged.replace(qa_path(target_dir))

    cross_staged = cross_admission_qa_path(staging_root)
    benchmark_staged = benchmark_qa_path(staging_root)
    cross_staged.replace(cross_admission_qa_path(patient_root))
    benchmark_staged.replace(benchmark_qa_path(patient_root))
    shutil.rmtree(staging_root, ignore_errors=True)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {label}: {path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return data


def _validate_conversation_artifact(conversation: dict[str, Any], path: Path) -> None:
    if not isinstance(conversation.get("conversation_lines"), list) or not conversation["conversation_lines"]:
        raise ValueError(f"conversation.json must contain non-empty conversation_lines: {path}")
    for index, line in enumerate(conversation["conversation_lines"], start=1):
        if not isinstance(line, dict):
            raise ValueError(f"conversation_lines[{index}] must be an object: {path}")
        turn_number = line.get("turn_number")
        if not isinstance(turn_number, int) or turn_number <= 0:
            raise ValueError(f"conversation_lines[{index}].turn_number must be a positive integer: {path}")


def _validate_summary_artifact(summary: dict[str, Any], path: Path) -> None:
    required = ("admission_start", "admission_end", "summary_paragraph", "problems")
    for field in required:
        if field not in summary:
            raise ValueError(f"summary.json missing required field {field}: {path}")
    if parse_dt(summary["admission_start"]) is None or parse_dt(summary["admission_end"]) is None:
        raise ValueError(f"summary.json admission_start/admission_end must be exact timestamps: {path}")
    if not isinstance(summary["summary_paragraph"], str) or not summary["summary_paragraph"].strip():
        raise ValueError(f"summary.json summary_paragraph must be non-empty: {path}")
    if not isinstance(summary["problems"], list):
        raise ValueError(f"summary.json problems must be a list: {path}")
