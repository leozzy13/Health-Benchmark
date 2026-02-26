from __future__ import annotations

import csv
import shutil
from dataclasses import asdict
from typing import Any

from .config import BenchmarkConfig
from .duckdb_store import MimicDuckDBStore
from .extractor import PacketExtractor
from .llm_client import LLMCallResult, OpenAILLMClient
from .prompting import append_repair_block, render_prompt
from .utils import canonical_json_dumps, ensure_dir, sha256_hex, utc_now_iso, write_json
from .validation import ValidationError, enforce_evidence_references, parse_json_response, validate_output_schema
from .writers import (
    admission_artifact_paths,
    cohort_csv_path,
    patient_conversation_details_path,
    patient_conversation_only_path,
    patient_dir,
    patient_manifest_path,
    write_conversation,
    write_input_data_manifest,
    write_model_call_record,
    write_packet,
    write_patient_conversation_details,
    write_patient_conversation_only,
    write_prompt_record,
    write_raw_output,
    write_summary,
    write_unlinked_notes,
)


class BenchmarkPipeline:
    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        if config.paths is None:
            raise ValueError("BenchmarkConfig.paths must be configured.")
        self.store = MimicDuckDBStore(
            hosp_dir=config.paths.mimiciv_dir / "hosp",
            icu_dir=config.paths.mimiciv_dir / "icu",
            note_dir=config.paths.mimiciv_note_dir,
        )
        self.extractor = PacketExtractor(self.store, config)
        self.llm_client = OpenAILLMClient(config)

    def close(self) -> None:
        self.store.close()

    def build_top_cohort(self, *, limit: int = 1000) -> Path:
        rows = self.store.fetch_rows(
            f"""
            SELECT
              subject_id,
              COUNT(DISTINCT hadm_id) AS n_admissions
            FROM hosp_admissions
            GROUP BY subject_id
            ORDER BY n_admissions DESC, subject_id ASC
            LIMIT {int(limit)}
            """
        )
        path = cohort_csv_path(self.config, cohort_name=f"top{int(limit)}_by_admission_count")
        ensure_dir(path.parent)
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["subject_id", "n_admissions"])
            writer.writeheader()
            for row in rows:
                writer.writerow({"subject_id": int(row["subject_id"]), "n_admissions": int(row["n_admissions"])})
        return path

    def generate_patient_sample(
        self,
        *,
        subject_id: int,
        model_name: str,
        hadm_id: int | None = None,
        max_admissions: int | None = None,
        only_with_discharge: bool | None = None,
    ) -> dict[str, Any]:
        self.config.model.name = model_name

        admissions = self.extractor.list_admissions_for_subject(
            subject_id,
            only_with_discharge=only_with_discharge,
            max_admissions=max_admissions,
        )
        if hadm_id is not None:
            admissions = [a for a in admissions if int(a["hadm_id"]) == int(hadm_id)]
        if not admissions:
            raise ValueError(f"No admissions found for subject_id={subject_id} under current filters.")

        # Cache subject-scoped temp tables to avoid repeated full CSV scans for each query/admission.
        self.store.prepare_subject_cache(subject_id)
        self._reset_patient_output_dir(subject_id)

        patient_manifest = {
            "schema_version": self.config.manifest_schema_version,
            "benchmark_version": self.config.benchmark_version,
            "benchmark_name": self.config.benchmark_name,
            "generated_at_utc": utc_now_iso(),
            "ids": {"subject_id": int(subject_id)},
            "paths": {
                "patient_dir": str(patient_dir(self.config, subject_id)),
                "patient_manifest": str(patient_manifest_path(self.config, subject_id)),
                "conversation_details": str(patient_conversation_details_path(self.config, subject_id)),
                "conversation_only": str(patient_conversation_only_path(self.config, subject_id)),
            },
            "dataset_versions": {
                "mimiciv": self.config.dataset_versions.mimiciv,
                "mimiciv_note": self.config.dataset_versions.mimiciv_note,
            },
            "config_snapshot": {
                "packet_schema_version": self.config.packet_schema_version,
                "prompt_template_version": self.config.prompt.template_version,
                "model": {
                    "provider": self.config.model.provider,
                    "name": self.config.model.name,
                    "temperature": self.config.model.temperature,
                    "max_output_tokens": self.config.model.max_output_tokens,
                    "seed": self.config.model.seed,
                    "retry_limit": self.config.model.retry_limit,
                },
                "strict_validation": self.config.strict_validation,
            },
            "admissions": [],
        }

        prev_summary_text = ""
        patient_conversation_details_rows: list[dict[str, Any]] = []
        patient_conversation_only_rows: list[dict[str, Any]] = []
        for admission_row in admissions:
            target_hadm_id = int(admission_row["hadm_id"])
            artifact_paths = admission_artifact_paths(self.config, subject_id, target_hadm_id)
            admission_entry = {
                "hadm_id": target_hadm_id,
                "admittime": admission_row.get("admittime"),
                "dischtime": admission_row.get("dischtime"),
                "discharge_note_count": admission_row.get("discharge_note_count"),
                "status": "pending",
                "paths": {
                    "admission_dir": str(artifact_paths.admission_dir),
                    "packet": str(artifact_paths.packet_json),
                    "input_data_manifest": str(artifact_paths.input_data_manifest_json),
                    "prompt_record": str(artifact_paths.prompt_record_json),
                    "model_call_record": str(artifact_paths.model_call_record_json),
                    "conversation": str(artifact_paths.conversation_jsonl),
                    "summary": str(artifact_paths.summary_json),
                },
            }
            patient_manifest["admissions"].append(admission_entry)
            self._write_patient_manifest(subject_id, patient_manifest)

            try:
                extraction = self.extractor.extract_admission_packet(subject_id, target_hadm_id)
                write_packet(artifact_paths, extraction.packet)
                write_input_data_manifest(artifact_paths, extraction.input_data_manifest)
                write_unlinked_notes(artifact_paths, extraction.unlinked_radiology_notes)

                rendered = render_prompt(extraction.packet, prev_summary_text, self.config)
                prompt_record = {
                    "schema_version": self.config.manifest_schema_version,
                    "prompt_template_version": self.config.prompt.template_version,
                    "ids": {"subject_id": subject_id, "hadm_id": target_hadm_id},
                    "packet_path": "packet.json",
                    "previous_summary_included": bool(prev_summary_text),
                    "system_message": rendered.system_message,
                    "user_message": rendered.user_message,
                    "delimiters": self.config.prompt.delimiters,
                    "hashes": {
                        "packet_sha256": extraction.packet["packet_stats"]["sha256_canonical_packet"],
                        "system_sha256": sha256_hex(rendered.system_message),
                        "user_sha256": sha256_hex(rendered.user_message),
                    },
                }
                write_prompt_record(artifact_paths, prompt_record)

                llm_result, parsed = self._generate_validated_response(
                    extraction.packet,
                    rendered.system_message,
                    rendered.user_message,
                )
                write_raw_output(
                    artifact_paths,
                    {
                        "response_text": llm_result.text,
                        "raw_response": llm_result.raw_response,
                    },
                )
                write_conversation(artifact_paths, parsed["conversation"])
                patient_conversation_details_rows.extend(
                    self._tag_conversation_rows(
                        subject_id=subject_id,
                        hadm_id=target_hadm_id,
                        conversation=parsed["conversation"],
                    )
                )
                write_patient_conversation_details(self.config, subject_id, patient_conversation_details_rows)
                patient_conversation_only_rows.append(
                    {
                        "hadm_id": int(target_hadm_id),
                        "conversation": self._speaker_text_only(parsed["conversation"]),
                    }
                )
                write_patient_conversation_only(
                    self.config,
                    subject_id,
                    patient_conversation_only_rows,
                )
                write_summary(artifact_paths, parsed["end_of_admission_summary"])

                model_call_record = self._build_model_call_record(
                    subject_id=subject_id,
                    hadm_id=target_hadm_id,
                    llm_result=llm_result,
                )
                write_model_call_record(artifact_paths, model_call_record)

                prev_summary_text = canonical_json_dumps(parsed["end_of_admission_summary"])
                admission_entry["status"] = "completed"
                admission_entry["conversation_turns"] = len(parsed["conversation"])
                admission_entry["packet_sha256"] = extraction.packet["packet_stats"]["sha256_canonical_packet"]
                admission_entry["output_summary_relative_discharge_time"] = parsed["end_of_admission_summary"][
                    "relative_discharge_time"
                ]
                self._write_patient_manifest(subject_id, patient_manifest)
            except Exception as exc:
                admission_entry["status"] = "error"
                admission_entry["error"] = str(exc)
                self._write_patient_manifest(subject_id, patient_manifest)
                raise

        return patient_manifest

    @staticmethod
    def _tag_conversation_rows(
        *,
        subject_id: int,
        hadm_id: int,
        conversation: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        tagged_rows: list[dict[str, Any]] = []
        for row in conversation:
            tagged_rows.append(
                {
                    "subject_id": int(subject_id),
                    "hadm_id": int(hadm_id),
                    **row,
                }
            )
        return tagged_rows

    @staticmethod
    def _speaker_text_only(conversation: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for row in conversation:
            rows.append(
                {
                    "speaker": row.get("speaker"),
                    "text": row.get("text"),
                }
            )
        return rows

    def _generate_validated_response(
        self,
        packet: dict[str, Any],
        system_message: str,
        user_message: str,
    ) -> tuple[LLMCallResult, dict[str, Any]]:
        max_attempts = max(1, int(self.config.model.retry_limit))
        validation_errors: list[str] = []
        aggregated_attempts = []
        latest_result: LLMCallResult | None = None
        current_user_message = user_message

        for schema_attempt in range(1, max_attempts + 1):
            latest_result = self.llm_client.generate_with_retries(system_message, current_user_message)
            aggregated_attempts.extend(latest_result.attempts)
            try:
                parsed = parse_json_response(latest_result.text)
                validate_output_schema(parsed)
                enforce_evidence_references(parsed, packet, strict=self.config.strict_validation)
                # Renumber attempts across transport + validation retries for clean logs.
                for idx, attempt in enumerate(aggregated_attempts, start=1):
                    attempt.attempt_index = idx
                latest_result.attempts = aggregated_attempts
                return latest_result, parsed
            except ValidationError as exc:
                validation_errors.append(str(exc))
                if schema_attempt >= max_attempts:
                    break
                current_user_message = append_repair_block(user_message)

        for idx, attempt in enumerate(aggregated_attempts, start=1):
            attempt.attempt_index = idx
        if latest_result is not None:
            latest_result.attempts = aggregated_attempts
        raise ValidationError(
            f"Model response failed validation after {max_attempts} schema attempts: {' | '.join(validation_errors)}"
        )

    def _build_model_call_record(self, *, subject_id: int, hadm_id: int, llm_result: LLMCallResult) -> dict[str, Any]:
        return {
            "schema_version": self.config.manifest_schema_version,
            "ids": {"subject_id": int(subject_id), "hadm_id": int(hadm_id)},
            "model": {
                "provider": self.config.model.provider,
                "name": self.config.model.name,
                "reasoning_effort": self.config.model.reasoning_effort,
            },
            "params": {
                "temperature": self.config.model.temperature,
                "max_output_tokens": self.config.model.max_output_tokens,
                "seed": self.config.model.seed,
            },
            "attempts": [asdict(a) for a in llm_result.attempts],
            "output_paths": {
                "raw_model_output": "raw_model_output.json",
                "conversation": "conversation.jsonl",
                "summary": "summary.json",
            },
        }

    def _write_patient_manifest(self, subject_id: int, manifest: dict[str, Any]) -> None:
        path = patient_manifest_path(self.config, subject_id)
        ensure_dir(path.parent)
        write_json(path, manifest)

    def _reset_patient_output_dir(self, subject_id: int) -> None:
        path = patient_dir(self.config, subject_id)
        if not path.exists():
            return
        assert self.config.paths is not None
        if path.parent != self.config.paths.output_root:
            raise RuntimeError(f"Refusing to delete unexpected patient output path: {path}")
        shutil.rmtree(path)
