from __future__ import annotations

import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import BenchmarkConfig
from .duckdb_store import MimicDuckDBStore
from .extractor import AdmissionSkipError, PacketExtractor
from .llm_client import LLMCallResult, build_llm_client
from .prompting import append_repair_message, render_prompt
from .qa_pipeline import generate_patient_qa
from .verify_outputs import verify_patient_outputs
from .utils import conversation_token_render, ensure_dir, round_mean, utc_now_iso
from .validation import ValidationError, validate_generation
from .writers import (
    admission_artifact_paths,
    batch_run_dir,
    batch_summary_path,
    benchmark_root,
    patient_dir,
    patient_staging_dir,
    write_batch_summary,
    write_combined_conversation,
    write_conversation,
    write_packet,
    write_patient_summary,
    write_prompt_record,
    write_summary,
)


def _import_tiktoken():
    try:
        import tiktoken  # type: ignore
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("tiktoken is required. Install dependencies in requirements.txt") from exc
    return tiktoken


class BenchmarkPipeline:
    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self.store: MimicDuckDBStore | None = None
        self.extractor: PacketExtractor | None = None
        self.llm_client = build_llm_client(config)

    def close(self) -> None:
        if self.store is not None:
            self.store.close()
            self.store = None
        self.extractor = None

    def _ensure_data_backend(self) -> None:
        if self.store is None:
            self.store = MimicDuckDBStore(
                hosp_dir=self.config.dataset.mimiciv_hosp_path,
                note_dir=self.config.dataset.mimiciv_note_path,
            )
        if self.extractor is None:
            self.extractor = PacketExtractor(self.store, self.config)

    def generate_patient_sample(
        self,
        *,
        subject_id: int | None = None,
        model_name: str | None = None,
        hadm_id: int | None = None,
        max_admissions: int | None = None,
    ) -> dict[str, Any]:
        self._ensure_data_backend()
        assert self.store is not None
        assert self.extractor is not None

        resolved_subject_id = int(subject_id if subject_id is not None else self._require_subject_id())
        resolved_model = model_name or self.config.llm.model
        resolved_max_admissions = max_admissions if max_admissions is not None else self.config.selection.max_admissions
        self.config.llm.model = resolved_model

        self.store.prepare_subject_cache(resolved_subject_id)
        eligible_admissions = self.extractor.list_admissions_for_subject(resolved_subject_id)
        selected_admissions = list(eligible_admissions)
        if resolved_max_admissions is not None:
            selected_admissions = selected_admissions[: int(resolved_max_admissions)]
        if hadm_id is not None:
            selected_admissions = [row for row in selected_admissions if int(row["hadm_id"]) == int(hadm_id)]
        if not selected_admissions:
            raise ValueError(f"No eligible admissions found for subject_id={resolved_subject_id} under current filters.")

        staging_dir = self._prepare_staging_dir(resolved_subject_id)

        previous_summary: dict[str, Any] | None = None
        processed_admissions: list[dict[str, Any]] = []
        llm_usage_rows: list[dict[str, int]] = []
        conversation_turn_counts: list[int] = []
        conversation_token_counts: list[int] = []
        tokenizer_name: str | None = None
        skipped_admissions: list[dict[str, Any]] = []

        for admission_row in selected_admissions:
            current_hadm_id = int(admission_row["hadm_id"])
            artifact_paths = admission_artifact_paths(staging_dir, current_hadm_id)
            try:
                extraction = self.extractor.extract_admission_packet(
                    resolved_subject_id,
                    current_hadm_id,
                    previous_summary,
                )
            except AdmissionSkipError as exc:
                skipped_admissions.append({"hadm_id": str(current_hadm_id), "reason": str(exc)})
                continue

            packet = extraction.packet
            write_packet(artifact_paths, packet)

            rendered = render_prompt(packet, previous_summary)
            request_timestamp = utc_now_iso()
            llm_result, parsed_output = self._generate_validated_response(
                packet=packet,
                system_message=rendered.system_message,
                user_message=rendered.user_message,
            )

            conversation_payload = {
                "subject_id": str(resolved_subject_id),
                "hadm_id": str(current_hadm_id),
                "admission_start": packet["admission_start"],
                "admission_end": packet["admission_end"],
                "conversation_lines": parsed_output["conversation_lines"],
            }
            summary_payload = parsed_output["summary"]
            prompt_record = self._build_prompt_record(
                subject_id=resolved_subject_id,
                hadm_id=current_hadm_id,
                request_timestamp=request_timestamp,
                rendered_prompt=rendered,
                previous_summary=previous_summary,
                packet=packet,
                llm_result=llm_result,
                parsed_output=parsed_output,
            )

            write_conversation(artifact_paths, conversation_payload)
            write_summary(artifact_paths, summary_payload)
            write_prompt_record(artifact_paths, prompt_record)

            processed_admissions.append(
                {
                    "hadm_id": str(current_hadm_id),
                    "admission_start": packet["admission_start"],
                    "admission_end": packet["admission_end"],
                    "conversation_lines": parsed_output["conversation_lines"],
                }
            )
            llm_usage_rows.append(llm_result.usage)
            conversation_turn_counts.append(len(parsed_output["conversation_lines"]))
            token_count, tokenizer_name = self._count_conversation_tokens(
                resolved_model,
                parsed_output["conversation_lines"],
            )
            conversation_token_counts.append(token_count)
            previous_summary = summary_payload

        if not processed_admissions:
            raise ValueError(
                f"Generation produced no valid admissions for subject_id={resolved_subject_id}. "
                f"Skipped admissions: {skipped_admissions}"
            )

        processed_admissions.sort(key=lambda row: (row["admission_start"], row["hadm_id"]))
        combined_payload = {
            "subject_id": str(resolved_subject_id),
            "processed_hadm_ids": [row["hadm_id"] for row in processed_admissions],
            "admissions": processed_admissions,
        }
        patient_summary = self._build_patient_summary(
            subject_id=resolved_subject_id,
            eligible_count=len(eligible_admissions),
            processed_admissions=processed_admissions,
            max_admissions=resolved_max_admissions,
            llm_usage_rows=llm_usage_rows,
            conversation_turn_counts=conversation_turn_counts,
            conversation_token_counts=conversation_token_counts,
            tokenizer_name=tokenizer_name or "o200k_base",
        )
        if skipped_admissions:
            patient_summary["skipped_admissions"] = skipped_admissions

        write_combined_conversation(staging_dir, combined_payload)
        write_patient_summary(staging_dir, patient_summary)
        self._promote_staging_dir(resolved_subject_id, staging_dir)
        return patient_summary

    def generate_all(
        self,
        *,
        subject_ids: list[int],
        model_name: str | None = None,
        max_admissions: int | None = None,
        fail_fast: bool = False,
    ) -> dict[str, Any]:
        requested_subject_ids = [int(subject_id) for subject_id in subject_ids]
        if not requested_subject_ids:
            raise ValueError("generate_all requires at least one subject_id")

        resolved_model = model_name or self.config.llm.model
        self.config.llm.model = resolved_model
        resolved_max_admissions = max_admissions if max_admissions is not None else self.config.selection.max_admissions

        start_time = utc_now_iso()
        run_root = batch_run_dir(self.config, self._build_batch_run_id())
        results: list[dict[str, Any]] = []
        succeeded: list[int] = []
        failed: list[int] = []

        try:
            for subject_id in requested_subject_ids:
                try:
                    patient_summary = self.generate_patient_sample(
                        subject_id=subject_id,
                        model_name=resolved_model,
                        max_admissions=resolved_max_admissions,
                    )
                    qa_summary = self.generate_patient_qa(
                        subject_id=subject_id,
                        model_name=resolved_model,
                    )
                    verification_summary = verify_patient_outputs(
                        patient_dir(self.config, subject_id),
                        expect_qa=True,
                    )
                    succeeded.append(subject_id)
                    results.append(
                        {
                            "subject_id": subject_id,
                            "status": "success",
                            "patient_summary": patient_summary,
                            "qa_summary": qa_summary,
                            "verification_summary": verification_summary,
                        }
                    )
                except Exception as exc:
                    failed.append(subject_id)
                    results.append(
                        {
                            "subject_id": subject_id,
                            "status": "failed",
                            "error": str(exc),
                            "error_type": type(exc).__name__,
                        }
                    )
                    if fail_fast:
                        break
        finally:
            batch_summary = {
                "provider": self.config.llm.provider,
                "model": self.config.llm.model,
                "start_time": start_time,
                "end_time": utc_now_iso(),
                "requested_subject_ids": requested_subject_ids,
                "succeeded": succeeded,
                "failed": failed,
                "fail_fast": bool(fail_fast),
                "max_admissions": resolved_max_admissions,
                "results": results,
                "slurm_job_id": os.getenv("SLURM_JOB_ID"),
                "final_status": "success" if not failed else "failed",
            }
            write_batch_summary(run_root, batch_summary)
            batch_summary["batch_summary_path"] = str(batch_summary_path(run_root))

        return batch_summary

    def generate_patient_qa(
        self,
        *,
        subject_id: int | None = None,
        patient_dir: Path | None = None,
        model_name: str | None = None,
    ) -> dict[str, Any]:
        return generate_patient_qa(
            self.config,
            self.llm_client,
            subject_id=subject_id,
            patient_dir=patient_dir,
            model_name=model_name,
        )

    def _require_subject_id(self) -> int:
        if self.config.selection.subject_id is None:
            raise ValueError("subject_id must be provided via CLI or config.yaml")
        return int(self.config.selection.subject_id)

    def _prepare_staging_dir(self, subject_id: int) -> Path:
        staging_dir = patient_staging_dir(self.config, subject_id)
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        ensure_dir(staging_dir)
        return staging_dir

    def _promote_staging_dir(self, subject_id: int, staging_dir: Path) -> None:
        final_dir = patient_dir(self.config, subject_id)
        root = benchmark_root(self.config)
        if final_dir.parent != root:
            raise RuntimeError(f"Refusing to replace unexpected output path: {final_dir}")
        if final_dir.exists():
            if not self.config.runtime.replace_existing_patient_output:
                raise RuntimeError(f"Output already exists for subject_id={subject_id}: {final_dir}")
            shutil.rmtree(final_dir)
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        staging_dir.rename(final_dir)

    def _generate_validated_response(
        self,
        *,
        packet: dict[str, Any],
        system_message: str,
        user_message: str,
    ) -> tuple[LLMCallResult, dict[str, Any]]:
        max_attempts = max(1, int(self.config.llm.max_retries) + 1)
        current_user_message = user_message
        latest_error: Exception | None = None

        for _attempt in range(1, max_attempts + 1):
            try:
                llm_result = self.llm_client.generate_response(system_message, current_user_message)
                parsed_output = validate_generation(llm_result.parsed_output, packet)
                return llm_result, parsed_output
            except (ValidationError, RuntimeError) as exc:
                latest_error = exc
                current_user_message = append_repair_message(user_message, str(exc))
        raise RuntimeError(f"Generation failed after {max_attempts} attempts: {latest_error}")

    def _build_prompt_record(
        self,
        *,
        subject_id: int,
        hadm_id: int,
        request_timestamp: str,
        rendered_prompt: Any,
        previous_summary: dict[str, Any] | None,
        packet: dict[str, Any],
        llm_result: LLMCallResult,
        parsed_output: dict[str, Any],
    ) -> dict[str, Any]:
        record = {
            "subject_id": str(subject_id),
            "hadm_id": str(hadm_id),
            "provider": self.config.llm.provider,
            "model": self.config.llm.model,
            "temperature": self.config.llm.temperature,
            "base_url": self.config.llm.base_url,
            "request_timestamp": request_timestamp,
            "system_message": rendered_prompt.system_message,
            "task_message": rendered_prompt.task_message,
            "previous_admission_summary_used": previous_summary,
            "stay_days": rendered_prompt.stay_days,
            "recommended_turn_range": rendered_prompt.recommended_turn_range,
            "storyline_prompt_block": rendered_prompt.storyline_prompt_block,
            "problem_prompt_block": rendered_prompt.problem_prompt_block,
            "supporting_context_block": rendered_prompt.supporting_context_block,
            "formed_packet": packet,
            "api_usage": llm_result.usage,
            "response_id": llm_result.response_id,
            "latency_ms": llm_result.latency_ms,
            "raw_structured_response": parsed_output,
        }
        if self.config.runtime.save_raw_response:
            record["raw_response"] = llm_result.raw_response
        return record

    def _build_patient_summary(
        self,
        *,
        subject_id: int,
        eligible_count: int,
        processed_admissions: list[dict[str, Any]],
        max_admissions: int | None,
        llm_usage_rows: list[dict[str, int]],
        conversation_turn_counts: list[int],
        conversation_token_counts: list[int],
        tokenizer_name: str,
    ) -> dict[str, Any]:
        input_tokens = [row.get("input_tokens", 0) for row in llm_usage_rows]
        output_tokens = [row.get("output_tokens", 0) for row in llm_usage_rows]
        return {
            "subject_id": str(subject_id),
            "eligible_admissions": int(eligible_count),
            "processed_admissions": len(processed_admissions),
            "max_admissions": max_admissions,
            "processed_hadm_ids": [row["hadm_id"] for row in processed_admissions],
            "llm_call_stats": {
                "mean_input_tokens": round_mean(input_tokens),
                "mean_output_tokens": round_mean(output_tokens),
                "total_input_tokens": int(sum(input_tokens)),
                "total_output_tokens": int(sum(output_tokens)),
            },
            "conversation_stats": {
                "mean_turns": round_mean(conversation_turn_counts),
                "total_turns": int(sum(conversation_turn_counts)),
                "mean_tokens": round_mean(conversation_token_counts),
                "total_tokens": int(sum(conversation_token_counts)),
                "tokenizer": tokenizer_name,
            },
        }

    def _count_conversation_tokens(self, model_name: str, conversation_lines: list[dict[str, Any]]) -> tuple[int, str]:
        tiktoken = _import_tiktoken()
        configured_tokenizer = self.config.llm.tokenizer_name
        if configured_tokenizer:
            try:
                encoding = tiktoken.get_encoding(configured_tokenizer)
                rendered_text = conversation_token_render(conversation_lines)
                return len(encoding.encode(rendered_text)), configured_tokenizer
            except KeyError:
                pass
        try:
            encoding = tiktoken.encoding_for_model(model_name)
            encoding_name = getattr(encoding, "name", model_name)
        except KeyError:
            encoding = tiktoken.get_encoding("o200k_base")
            encoding_name = "o200k_base"
        rendered_text = conversation_token_render(conversation_lines)
        return len(encoding.encode(rendered_text)), encoding_name

    def _build_batch_run_id(self) -> str:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        slurm_job_id = os.getenv("SLURM_JOB_ID")
        if slurm_job_id:
            return f"{timestamp}_slurm_{slurm_job_id}"
        return timestamp
