from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..scripts.config import BenchmarkConfig
from ..scripts.utils import ensure_dir, utc_now_iso, write_json
from .answer_runner import run_answer_batches, summarize_schema_repair_metrics
from .config import (
    EvaluationSettings,
    MEMORY_EVALUATION_VARIANTS,
    build_evaluation_paths,
    memory_suffix_for_variant,
)
from .context_renderer import render_full_patient_context
from .judge_runner import run_llm_judge_batches
from .loader import load_patient_artifacts
from .memory import DenseMemoryRetriever, build_memory_question_batches, build_mem0_memory_store
from .pipeline import (
    EvaluationPipeline,
    build_context_stats,
    build_model_summary,
    build_patient_config_record,
    empty_memory_store_payload,
    normalize_benchmark,
    rebuild_comparison_summary,
)
from .scoring import score_predictions
from .token_budget import build_preflight_record
from .types import EvalQuestion, LoadedPatientArtifacts, ModelSpec


class MemoryEvaluationPipeline(EvaluationPipeline):
    def __init__(
        self,
        base_config: BenchmarkConfig,
        settings: EvaluationSettings,
        *,
        client_factory=None,
        client_overrides: dict[str, Any] | None = None,
        memory_retriever_factory=None,
    ) -> None:
        super().__init__(
            base_config,
            settings,
            client_factory=client_factory,
            client_overrides=client_overrides,
        )
        if settings.evaluation_variant not in MEMORY_EVALUATION_VARIANTS:
            raise ValueError("MemoryEvaluationPipeline requires a memory evaluation variant")
        self.memory_retriever_factory = memory_retriever_factory

    def evaluate_patient(self, *, subject_id: int, patient_root: Path) -> dict[str, Any]:
        loaded = load_patient_artifacts(patient_root)
        normalized_questions = normalize_benchmark(loaded.benchmark_payload)
        context_text = render_full_patient_context(loaded.combined_payload)
        paths = build_evaluation_paths(
            patient_root,
            evaluation_root=self.settings.evaluation_root,
            subject_id=loaded.subject_id,
        )
        ensure_dir(paths.evaluation_root)

        model_results: list[dict[str, Any]] = []
        model_prechecks: dict[str, Any] = {}
        for base_model_spec in self.settings.model_specs:
            memory_model_spec = memory_model_spec_for(
                base_model_spec,
                evaluation_variant=self.settings.evaluation_variant,
            )
            if self.settings.stage == "judge":
                model_result, precheck = self._run_model_judge_stage(
                    loaded=loaded,
                    normalized_questions=normalized_questions,
                    context_text=context_text,
                    paths=paths,
                    model_spec=memory_model_spec,
                )
            elif self.settings.stage == "answers":
                model_result, precheck = self._run_model_memory_answers_stage(
                    loaded=loaded,
                    normalized_questions=normalized_questions,
                    context_text=context_text,
                    paths=paths,
                    base_model_spec=base_model_spec,
                    memory_model_spec=memory_model_spec,
                    provisional_run_status="answers_completed",
                )
            else:
                model_result, precheck = self._run_model_memory_full_evaluation(
                    loaded=loaded,
                    normalized_questions=normalized_questions,
                    context_text=context_text,
                    paths=paths,
                    base_model_spec=base_model_spec,
                    memory_model_spec=memory_model_spec,
                )
            model_results.append(model_result)
            model_prechecks[memory_model_spec.slug] = precheck

        write_json(
            paths.config_json,
            build_patient_config_record(
                settings=self.settings,
                loaded=loaded,
                paths=paths,
            ),
        )
        write_json(
            paths.context_stats_json,
            build_context_stats(
                loaded=loaded,
                normalized_questions=normalized_questions,
                context_text=context_text,
                model_prechecks=model_prechecks,
            ),
        )
        write_json(paths.benchmark_snapshot_json, loaded.benchmark_payload)

        comparison = {"models": []}
        comparison_summary_path: str | None = None
        if self.settings.stage != "answers":
            comparison = rebuild_comparison_summary(paths, subject_id=str(loaded.subject_id))
            comparison_summary_path = str(paths.comparison_dir / "summary.md")

        successful_statuses = {"completed"}
        if self.settings.stage == "answers":
            successful_statuses.add("answers_completed")
        failed_models = [
            result["model_slug"]
            for result in model_results
            if result["run_status"] not in successful_statuses
        ]
        return {
            "subject_id": int(subject_id),
            "status": "completed" if not failed_models else "failed",
            "stage": self.settings.stage,
            "evaluation_variant": self.settings.evaluation_variant,
            "patient_dir": str(patient_root),
            "evaluation_dir": str(paths.evaluation_root),
            "completed_models": [
                result["model_slug"]
                for result in model_results
                if result["run_status"] in successful_statuses
            ],
            "failed_models": failed_models,
            "comparison_summary_path": comparison_summary_path,
            "comparison_models": comparison["models"],
        }

    def _run_model_memory_full_evaluation(
        self,
        *,
        loaded: LoadedPatientArtifacts,
        normalized_questions: list[EvalQuestion],
        context_text: str,
        paths,
        base_model_spec: ModelSpec,
        memory_model_spec: ModelSpec,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        started_at = datetime.now(UTC)
        (
            precheck,
            batches_payload,
            raw_predictions,
            scored_predictions,
            memory_store_payload,
            memory_event_records,
            error_records,
            summary_payload,
        ) = self._run_memory_answers_computation(
            loaded=loaded,
            normalized_questions=normalized_questions,
            context_text=context_text,
            base_model_spec=base_model_spec,
            memory_model_spec=memory_model_spec,
        )
        llm_judgments: list[dict[str, Any]] = []
        try:
            if summary_payload["run_status"] == "completed":
                llm_client = self._client_for_model(memory_model_spec)
                judge_client = self._judge_client_for_model(
                    memory_model_spec,
                    answer_client=llm_client,
                )
                llm_judgments = run_llm_judge_batches(
                    judge_client,
                    scored_predictions,
                    judge_model_name=self.settings.judge_model_spec.model_name,
                    save_raw_response=self.settings.save_raw_response,
                    max_output_tokens=self.settings.judge_max_output_tokens,
                    batch_size=self.settings.batch_size,
                    retry_limit=self.settings.retry_limit,
                )
                summary_payload = build_model_summary(
                    loaded=loaded,
                    settings=self.settings,
                    model_spec=memory_model_spec,
                    scored_predictions=scored_predictions,
                    error_records=error_records,
                    precheck=precheck,
                    started_at=started_at,
                    batch_count=len(batches_payload["batches"]),
                    run_status="completed",
                )
        except Exception as exc:
            error_records.append(
                {
                    "timestamp": utc_now_iso(),
                    "stage": "memory_pipeline",
                    "error_type": type(exc).__name__,
                    "error_kind": "input_error" if isinstance(exc, ValueError) else "internal_error",
                    "message": str(exc),
                }
            )
            summary_payload = build_model_summary(
                loaded=loaded,
                settings=self.settings,
                model_spec=memory_model_spec,
                scored_predictions=scored_predictions,
                error_records=error_records,
                precheck=precheck,
                started_at=started_at,
                batch_count=len(batches_payload["batches"]),
                run_status="failed_input_error" if isinstance(exc, ValueError) else "failed_internal_error",
            )

        return self._persist_model_outputs(
            paths=paths,
            model_spec=memory_model_spec,
            precheck=precheck,
            batches_payload=batches_payload,
            raw_predictions=raw_predictions,
            scored_predictions=scored_predictions,
            memory_store_payload=memory_store_payload,
            memory_event_records=memory_event_records,
            llm_judgments=llm_judgments,
            summary_payload=summary_payload,
            error_records=error_records,
        )

    def _run_model_memory_answers_stage(
        self,
        *,
        loaded: LoadedPatientArtifacts,
        normalized_questions: list[EvalQuestion],
        context_text: str,
        paths,
        base_model_spec: ModelSpec,
        memory_model_spec: ModelSpec,
        provisional_run_status: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        (
            precheck,
            batches_payload,
            raw_predictions,
            scored_predictions,
            memory_store_payload,
            memory_event_records,
            error_records,
            summary_payload,
        ) = self._run_memory_answers_computation(
            loaded=loaded,
            normalized_questions=normalized_questions,
            context_text=context_text,
            base_model_spec=base_model_spec,
            memory_model_spec=memory_model_spec,
            provisional_run_status=provisional_run_status,
        )
        return self._persist_model_outputs(
            paths=paths,
            model_spec=memory_model_spec,
            precheck=precheck,
            batches_payload=batches_payload,
            raw_predictions=raw_predictions,
            scored_predictions=scored_predictions,
            memory_store_payload=memory_store_payload,
            memory_event_records=memory_event_records,
            llm_judgments=[],
            summary_payload=summary_payload,
            error_records=error_records,
        )

    def _run_memory_answers_computation(
        self,
        *,
        loaded: LoadedPatientArtifacts,
        normalized_questions: list[EvalQuestion],
        context_text: str,
        base_model_spec: ModelSpec,
        memory_model_spec: ModelSpec,
        provisional_run_status: str = "completed",
    ) -> tuple[
        dict[str, Any],
        dict[str, Any],
        list[dict[str, Any]],
        list[dict[str, Any]],
        dict[str, Any],
        list[dict[str, Any]],
        list[dict[str, Any]],
        dict[str, Any],
    ]:
        started_at = datetime.now(UTC)
        error_records: list[dict[str, Any]] = []
        raw_predictions: list[dict[str, Any]] = []
        scored_predictions: list[dict[str, Any]] = []
        batches_payload = {"batches": []}
        memory_store_payload = empty_memory_store_payload(self.settings.evaluation_variant)
        memory_event_records: list[dict[str, Any]] = []
        full_precheck = build_preflight_record(
            model_name=memory_model_spec.model_name,
            tokenizer_name=self.settings.tokenizer_name,
            context_text=context_text,
            questions=normalized_questions,
            batch_size=self.settings.batch_size,
            max_model_len=memory_model_spec.max_model_len,
            max_output_tokens=self.settings.max_output_tokens,
            safe_margin_tokens=self.settings.safe_margin_tokens,
            token_estimate_safety_multiplier=self.settings.token_estimate_safety_multiplier,
            enable_thinking=self.settings.enable_thinking,
        )
        precheck: dict[str, Any] = {
            "status": f"{self.settings.evaluation_variant}_memory_context",
            "full_context_status": full_precheck.get("status", "not_run"),
            "evaluation_variant": self.settings.evaluation_variant,
            "base_model_slug": base_model_spec.slug,
            "answer_context_strategy": "memory",
        }

        try:
            llm_client = self._client_for_model(memory_model_spec)
            memory_result = build_mem0_memory_store(
                llm_client,
                combined_payload=loaded.combined_payload,
                settings=self.settings,
                model_name=memory_model_spec.model_name,
                retriever=self._build_memory_retriever(),
            )
            memory_store_payload = memory_result.store_payload
            memory_event_records = memory_result.event_records
            batches = build_memory_question_batches(
                normalized_questions,
                memory_store=memory_result.store,
                settings=self.settings,
                model_spec=memory_model_spec,
            )
            batches_payload = {"batches": [batch.to_record() for batch in batches]}
            precheck = {
                **precheck,
                "memory": memory_result.metrics,
                "batch_contexts": [batch.context_record for batch in batches],
                "truncated_batch_count": sum(
                    1 for batch in batches if bool(batch.context_record.get("was_truncated"))
                ),
                "max_estimated_batch_prompt_tokens": max(
                    (int(batch.estimated_prompt_tokens) for batch in batches),
                    default=0,
                ),
                "max_adjusted_estimated_batch_prompt_tokens": max(
                    (int(batch.adjusted_estimated_prompt_tokens) for batch in batches),
                    default=0,
                ),
            }
            answer_predictions, raw_predictions, answer_errors, answer_failures = run_answer_batches(
                llm_client,
                batches,
                model_name=memory_model_spec.model_name,
                save_raw_response=self.settings.save_raw_response,
                max_output_tokens=self.settings.max_output_tokens,
                retry_limit=self.settings.retry_limit,
            )
            precheck.update(summarize_schema_repair_metrics(raw_predictions))
            error_records.extend(answer_errors)
            scored_predictions = score_predictions(
                normalized_questions,
                answer_predictions,
                answer_failures,
            )
            summary_payload = build_model_summary(
                loaded=loaded,
                settings=self.settings,
                model_spec=memory_model_spec,
                scored_predictions=scored_predictions,
                error_records=error_records,
                precheck=precheck,
                started_at=started_at,
                batch_count=len(batches),
                run_status=provisional_run_status,
            )
        except Exception as exc:
            error_records.append(
                {
                    "timestamp": utc_now_iso(),
                    "stage": "memory_pipeline",
                    "error_type": type(exc).__name__,
                    "error_kind": "input_error" if isinstance(exc, ValueError) else "internal_error",
                    "message": str(exc),
                }
            )
            summary_payload = build_model_summary(
                loaded=loaded,
                settings=self.settings,
                model_spec=memory_model_spec,
                scored_predictions=scored_predictions,
                error_records=error_records,
                precheck=precheck,
                started_at=started_at,
                batch_count=len(batches_payload["batches"]),
                run_status="failed_input_error" if isinstance(exc, ValueError) else "failed_internal_error",
            )

        return (
            precheck,
            batches_payload,
            raw_predictions,
            scored_predictions,
            memory_store_payload,
            memory_event_records,
            error_records,
            summary_payload,
        )

    def _build_memory_retriever(self) -> DenseMemoryRetriever | None:
        if self.memory_retriever_factory is None:
            return None
        return self.memory_retriever_factory(self.settings)


def memory_model_spec_for(model_spec: ModelSpec, *, evaluation_variant: str = "mem0") -> ModelSpec:
    slug = model_spec.slug
    suffix = memory_suffix_for_variant(evaluation_variant)
    if not slug.endswith(suffix):
        slug = f"{slug}{suffix}"
    return ModelSpec(
        model_name=model_spec.model_name,
        slug=slug,
        tensor_parallel_size=model_spec.tensor_parallel_size,
        max_model_len=model_spec.max_model_len,
    )
