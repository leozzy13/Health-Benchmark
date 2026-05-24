from __future__ import annotations

import copy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..scripts.config import BenchmarkConfig
from ..scripts.utils import ensure_dir, utc_now_iso, write_json
from .answer_runner import run_answer_batches, summarize_schema_repair_metrics
from .config import (
    DEFAULT_MAX_MODEL_LEN,
    EvaluationSettings,
    RAG_EVALUATION_VARIANTS,
    build_evaluation_paths,
    build_model_artifact_paths,
    rag_suffix_for_variant,
)
from .context_renderer import render_full_patient_context
from .judge_runner import run_llm_judge_batches
from .loader import load_patient_artifacts
from .pipeline import (
    EvaluationPipeline,
    build_context_stats,
    build_model_summary,
    build_patient_config_record,
    empty_memory_store_payload,
    empty_retrieval_store_payload,
    normalize_benchmark,
    rebuild_comparison_summary,
)
from .rag import (
    build_admission_documents,
    build_rag_question_batches,
    build_rag_retriever,
    build_rag_store_payload,
)
from .scoring import score_predictions
from .token_budget import build_preflight_record
from .types import EvalQuestion, LoadedPatientArtifacts, ModelSpec


class RagEvaluationPipeline(EvaluationPipeline):
    def __init__(
        self,
        base_config: BenchmarkConfig,
        settings: EvaluationSettings,
        *,
        client_factory=None,
        client_overrides: dict[str, Any] | None = None,
        rag_retriever_factory=None,
    ) -> None:
        super().__init__(
            base_config,
            settings,
            client_factory=client_factory,
            client_overrides=client_overrides,
        )
        if settings.evaluation_variant not in RAG_EVALUATION_VARIANTS:
            raise ValueError("RagEvaluationPipeline requires a RAG evaluation variant")
        self.rag_retriever_factory = rag_retriever_factory

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
            rag_model_spec = rag_model_spec_for(
                base_model_spec,
                evaluation_variant=self.settings.evaluation_variant,
            )
            normal_truncated = self._normal_full_context_truncated(
                loaded=loaded,
                normalized_questions=normalized_questions,
                context_text=context_text,
                paths=paths,
                base_model_spec=base_model_spec,
            )
            if not normal_truncated:
                model_result, precheck = self._run_model_passthrough(
                    loaded=loaded,
                    paths=paths,
                    base_model_spec=base_model_spec,
                    rag_model_spec=rag_model_spec,
                )
            elif self.settings.stage == "judge":
                model_result, precheck = self._run_model_judge_stage(
                    loaded=loaded,
                    normalized_questions=normalized_questions,
                    context_text=context_text,
                    paths=paths,
                    model_spec=rag_model_spec,
                )
            elif self.settings.stage == "answers":
                model_result, precheck = self._run_model_rag_answers_stage(
                    loaded=loaded,
                    normalized_questions=normalized_questions,
                    context_text=context_text,
                    paths=paths,
                    base_model_spec=base_model_spec,
                    rag_model_spec=rag_model_spec,
                    provisional_run_status="answers_completed",
                )
            else:
                model_result, precheck = self._run_model_rag_full_evaluation(
                    loaded=loaded,
                    normalized_questions=normalized_questions,
                    context_text=context_text,
                    paths=paths,
                    base_model_spec=base_model_spec,
                    rag_model_spec=rag_model_spec,
                )
            model_results.append(model_result)
            model_prechecks[rag_model_spec.slug] = precheck

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

    def _run_model_rag_full_evaluation(
        self,
        *,
        loaded: LoadedPatientArtifacts,
        normalized_questions: list[EvalQuestion],
        context_text: str,
        paths,
        base_model_spec: ModelSpec,
        rag_model_spec: ModelSpec,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        started_at = datetime.now(UTC)
        (
            precheck,
            batches_payload,
            raw_predictions,
            scored_predictions,
            retrieval_store_payload,
            error_records,
            summary_payload,
        ) = self._run_rag_answers_computation(
            loaded=loaded,
            normalized_questions=normalized_questions,
            context_text=context_text,
            base_model_spec=base_model_spec,
            rag_model_spec=rag_model_spec,
        )
        llm_judgments: list[dict[str, Any]] = []
        try:
            if summary_payload["run_status"] == "completed":
                llm_client = self._client_for_model(rag_model_spec)
                judge_client = self._judge_client_for_model(
                    rag_model_spec,
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
                    model_spec=rag_model_spec,
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
                    "stage": "rag_pipeline",
                    "error_type": type(exc).__name__,
                    "error_kind": "input_error" if isinstance(exc, ValueError) else "internal_error",
                    "message": str(exc),
                }
            )
            summary_payload = build_model_summary(
                loaded=loaded,
                settings=self.settings,
                model_spec=rag_model_spec,
                scored_predictions=scored_predictions,
                error_records=error_records,
                precheck=precheck,
                started_at=started_at,
                batch_count=len(batches_payload["batches"]),
                run_status="failed_input_error" if isinstance(exc, ValueError) else "failed_internal_error",
            )

        return self._persist_model_outputs(
            paths=paths,
            model_spec=rag_model_spec,
            precheck=precheck,
            batches_payload=batches_payload,
            raw_predictions=raw_predictions,
            scored_predictions=scored_predictions,
            memory_store_payload=empty_memory_store_payload(self.settings.evaluation_variant),
            memory_event_records=[],
            retrieval_store_payload=retrieval_store_payload,
            llm_judgments=llm_judgments,
            summary_payload=summary_payload,
            error_records=error_records,
        )

    def _run_model_rag_answers_stage(
        self,
        *,
        loaded: LoadedPatientArtifacts,
        normalized_questions: list[EvalQuestion],
        context_text: str,
        paths,
        base_model_spec: ModelSpec,
        rag_model_spec: ModelSpec,
        provisional_run_status: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        (
            precheck,
            batches_payload,
            raw_predictions,
            scored_predictions,
            retrieval_store_payload,
            error_records,
            summary_payload,
        ) = self._run_rag_answers_computation(
            loaded=loaded,
            normalized_questions=normalized_questions,
            context_text=context_text,
            base_model_spec=base_model_spec,
            rag_model_spec=rag_model_spec,
            provisional_run_status=provisional_run_status,
        )
        return self._persist_model_outputs(
            paths=paths,
            model_spec=rag_model_spec,
            precheck=precheck,
            batches_payload=batches_payload,
            raw_predictions=raw_predictions,
            scored_predictions=scored_predictions,
            memory_store_payload=empty_memory_store_payload(self.settings.evaluation_variant),
            memory_event_records=[],
            retrieval_store_payload=retrieval_store_payload,
            llm_judgments=[],
            summary_payload=summary_payload,
            error_records=error_records,
        )

    def _run_rag_answers_computation(
        self,
        *,
        loaded: LoadedPatientArtifacts,
        normalized_questions: list[EvalQuestion],
        context_text: str,
        base_model_spec: ModelSpec,
        rag_model_spec: ModelSpec,
        provisional_run_status: str = "completed",
    ) -> tuple[
        dict[str, Any],
        dict[str, Any],
        list[dict[str, Any]],
        list[dict[str, Any]],
        dict[str, Any],
        list[dict[str, Any]],
        dict[str, Any],
    ]:
        started_at = datetime.now(UTC)
        error_records: list[dict[str, Any]] = []
        raw_predictions: list[dict[str, Any]] = []
        scored_predictions: list[dict[str, Any]] = []
        batches_payload = {"batches": []}
        retrieval_store_payload = empty_retrieval_store_payload(self.settings.evaluation_variant)
        full_precheck = build_preflight_record(
            model_name=rag_model_spec.model_name,
            tokenizer_name=self.settings.tokenizer_name,
            context_text=context_text,
            questions=normalized_questions,
            batch_size=self.settings.batch_size,
            max_model_len=rag_model_spec.max_model_len,
            max_output_tokens=self.settings.max_output_tokens,
            safe_margin_tokens=self.settings.safe_margin_tokens,
            token_estimate_safety_multiplier=self.settings.token_estimate_safety_multiplier,
            enable_thinking=self.settings.enable_thinking,
        )
        precheck: dict[str, Any] = {
            "status": f"{self.settings.evaluation_variant}_context",
            "full_context_status": full_precheck.get("status", "not_run"),
            "evaluation_variant": self.settings.evaluation_variant,
            "base_model_slug": base_model_spec.slug,
            "answer_context_strategy": "rag",
            "rag_passthrough": False,
        }

        try:
            llm_client = self._client_for_model(rag_model_spec)
            documents = build_admission_documents(loaded.combined_payload)
            retriever = self._build_rag_retriever()
            retrieval_store_payload = build_rag_store_payload(
                documents=documents,
                settings=self.settings,
                retriever=retriever,
                passthrough=False,
            )
            batches = build_rag_question_batches(
                normalized_questions,
                documents=documents,
                retriever=retriever,
                settings=self.settings,
                model_spec=rag_model_spec,
            )
            batches_payload = {"batches": [batch.to_record() for batch in batches]}
            precheck = {
                **precheck,
                "rag": {
                    **retrieval_store_payload.get("metrics", {}),
                    "rag_method": self.settings.rag_method,
                    "retrieval_backend": retrieval_store_payload.get("retriever", {}).get(
                        "retrieval_backend",
                        self.settings.rag_method,
                    ),
                    "document_unit": self.settings.rag_document_unit,
                    "selection_policy": self.settings.rag_selection_policy,
                    "render_order": self.settings.rag_render_order,
                    "passthrough_count": 0,
                    "retrieval_count": 1,
                },
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
                model_name=rag_model_spec.model_name,
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
                model_spec=rag_model_spec,
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
                    "stage": "rag_pipeline",
                    "error_type": type(exc).__name__,
                    "error_kind": "input_error" if isinstance(exc, ValueError) else "internal_error",
                    "message": str(exc),
                }
            )
            summary_payload = build_model_summary(
                loaded=loaded,
                settings=self.settings,
                model_spec=rag_model_spec,
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
            retrieval_store_payload,
            error_records,
            summary_payload,
        )

    def _run_model_passthrough(
        self,
        *,
        loaded: LoadedPatientArtifacts,
        paths,
        base_model_spec: ModelSpec,
        rag_model_spec: ModelSpec,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        started_at = datetime.now(UTC)
        normal_paths = build_model_artifact_paths(paths, base_model_spec)
        normal_artifacts = self._load_existing_model_outputs(normal_paths)
        include_judgments = self.settings.stage != "answers"
        llm_judgments = list(normal_artifacts["llm_judgments"]) if include_judgments else []
        scored_predictions = [dict(row) for row in normal_artifacts["scored_predictions"]]
        error_records = [dict(row) for row in normal_artifacts["error_records"]]
        batches_payload = copy.deepcopy(normal_artifacts["question_batches"])
        batches_payload["rag_passthrough"] = True
        batches_payload["passthrough_source_model_slug"] = base_model_spec.slug
        precheck = {
            "status": "full_context_passthrough",
            "full_context_status": "full_context_fits",
            "evaluation_variant": self.settings.evaluation_variant,
            "base_model_slug": base_model_spec.slug,
            "answer_context_strategy": "rag_passthrough",
            "rag_passthrough": True,
            "rag": {
                "enabled": True,
                "rag_method": self.settings.rag_method,
                "document_unit": self.settings.rag_document_unit,
                "selection_policy": self.settings.rag_selection_policy,
                "render_order": self.settings.rag_render_order,
                "passthrough_source_model_slug": base_model_spec.slug,
                "passthrough_count": 1,
                "retrieval_count": 0,
            },
            "batch_contexts": [
                {
                    **dict(batch.get("context", {})),
                    "rag_passthrough": True,
                    "passthrough_source_model_slug": base_model_spec.slug,
                }
                for batch in batches_payload.get("batches", [])
            ],
            "truncated_batch_count": 0,
            "max_estimated_batch_prompt_tokens": max(
                (int(batch.get("estimated_prompt_tokens", 0)) for batch in batches_payload.get("batches", [])),
                default=0,
            ),
            "max_adjusted_estimated_batch_prompt_tokens": max(
                (
                    int(batch.get("adjusted_estimated_prompt_tokens", 0))
                    for batch in batches_payload.get("batches", [])
                ),
                default=0,
            ),
        }
        summary_payload = build_model_summary(
            loaded=loaded,
            settings=self.settings,
            model_spec=rag_model_spec,
            scored_predictions=scored_predictions,
            error_records=error_records,
            precheck=precheck,
            started_at=started_at,
            batch_count=len(batches_payload.get("batches", [])),
            run_status="answers_completed" if self.settings.stage == "answers" else "completed",
        )
        retrieval_store_payload = {
            "mode": "rag_passthrough",
            "enabled": True,
            "evaluation_variant": self.settings.evaluation_variant,
            "rag_method": self.settings.rag_method,
            "rag_passthrough": True,
            "passthrough_source_model_slug": base_model_spec.slug,
            "metrics": {
                "enabled": True,
                "document_count": 0,
                "passthrough_count": 1,
                "retrieval_count": 0,
            },
        }
        return self._persist_model_outputs(
            paths=paths,
            model_spec=rag_model_spec,
            precheck=precheck,
            batches_payload=batches_payload,
            raw_predictions=[dict(row) for row in normal_artifacts["raw_predictions"]],
            scored_predictions=scored_predictions,
            memory_store_payload=empty_memory_store_payload(self.settings.evaluation_variant),
            memory_event_records=[],
            retrieval_store_payload=retrieval_store_payload,
            llm_judgments=llm_judgments,
            summary_payload=summary_payload,
            error_records=error_records,
        )

    def _normal_full_context_truncated(
        self,
        *,
        loaded: LoadedPatientArtifacts,
        normalized_questions: list[EvalQuestion],
        context_text: str,
        paths,
        base_model_spec: ModelSpec,
    ) -> bool:
        normal_paths = build_model_artifact_paths(paths, base_model_spec)
        if normal_paths.question_batches_json.exists():
            question_batches = self._load_existing_model_outputs(normal_paths)["question_batches"]
            return any(
                bool((batch.get("context") or {}).get("was_truncated"))
                for batch in question_batches.get("batches", [])
            )
        normal_model_spec = ModelSpec(
            model_name=base_model_spec.model_name,
            slug=base_model_spec.slug,
            tensor_parallel_size=base_model_spec.tensor_parallel_size,
            max_model_len=DEFAULT_MAX_MODEL_LEN,
        )
        precheck = build_preflight_record(
            model_name=normal_model_spec.model_name,
            tokenizer_name=self.settings.tokenizer_name,
            context_text=context_text,
            questions=normalized_questions,
            batch_size=self.settings.batch_size,
            max_model_len=normal_model_spec.max_model_len,
            max_output_tokens=self.settings.max_output_tokens,
            safe_margin_tokens=self.settings.safe_margin_tokens,
            token_estimate_safety_multiplier=self.settings.token_estimate_safety_multiplier,
            enable_thinking=self.settings.enable_thinking,
        )
        if precheck.get("status") == "full_context_fits":
            raise ValueError(
                f"RAG passthrough requires existing normal artifacts for {base_model_spec.slug}; "
                f"missing {normal_paths.question_batches_json}"
            )
        return True

    def _build_rag_retriever(self):
        if self.rag_retriever_factory is not None:
            return self.rag_retriever_factory(self.settings)
        return build_rag_retriever(self.settings)


def rag_model_spec_for(model_spec: ModelSpec, *, evaluation_variant: str) -> ModelSpec:
    slug = model_spec.slug
    suffix = rag_suffix_for_variant(evaluation_variant)
    if not slug.endswith(suffix):
        slug = f"{slug}{suffix}"
    return ModelSpec(
        model_name=model_spec.model_name,
        slug=slug,
        tensor_parallel_size=model_spec.tensor_parallel_size,
        max_model_len=model_spec.max_model_len,
    )
