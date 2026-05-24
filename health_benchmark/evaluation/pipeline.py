from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Sequence

from ..scripts.config import BenchmarkConfig
from ..scripts.llm_client import build_llm_client
from ..scripts.utils import ensure_dir, utc_now_iso, write_json, write_text
from .answer_runner import run_answer_batches, summarize_schema_repair_metrics
from .batch_builder import build_batches
from .judge_runner import run_llm_judge_batches
from .config import (
    EvaluationSettings,
    MEM0_MODEL_SLUG_SUFFIX,
    MEMORY_EVALUATION_VARIANTS,
    RAG_EVALUATION_VARIANTS,
    base_slug_for_memory_slug,
    base_slug_for_rag_slug,
    build_evaluation_paths,
    build_model_artifact_paths,
    clone_config_for_model,
)
from .context_renderer import count_total_turns, render_full_patient_context
from .io_utils import (
    list_existing_model_summary_paths,
    load_json,
    load_jsonl,
    write_csv,
    write_model_outputs,
)
from .loader import load_patient_artifacts
from .scoring import build_summary_breakdowns, build_top_level_metrics, score_predictions
from .summary_writer import build_comparison_markdown
from .token_budget import build_preflight_record
from .types import EvalQuestion, LoadedPatientArtifacts, ModelSpec


ClientFactory = Callable[[BenchmarkConfig], Any]
JUDGE_CLIENT_OVERRIDE_KEY = "__judge__"


class EvaluationPipeline:
    def __init__(
        self,
        base_config: BenchmarkConfig,
        settings: EvaluationSettings,
        *,
        client_factory: ClientFactory | None = None,
        client_overrides: dict[str, Any] | None = None,
    ) -> None:
        self.base_config = base_config
        self.settings = settings
        self.client_factory = client_factory or build_llm_client
        self.client_overrides = dict(client_overrides or {})

    def run(self, targets: Sequence[tuple[int, Path]]) -> dict[str, Any]:
        started_at = datetime.now(UTC)
        results: list[dict[str, Any]] = []
        succeeded: list[int] = []
        failed: list[int] = []

        for subject_id, patient_root in targets:
            result = self.evaluate_patient(subject_id=subject_id, patient_root=patient_root)
            results.append(result)
            if result["status"] == "completed":
                succeeded.append(subject_id)
            else:
                failed.append(subject_id)

        return {
            "stage": self.settings.stage,
            "evaluation_variant": self.settings.evaluation_variant,
            "provider": self.settings.provider,
            "requested_subject_ids": [int(subject_id) for subject_id, _patient_root in targets],
            "requested_models": [spec.model_name for spec in self.settings.model_specs],
            "start_time": started_at.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "end_time": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "succeeded": succeeded,
            "failed": failed,
            "results": results,
            "final_status": "completed" if not failed else "failed",
        }

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
        for model_spec in self.settings.model_specs:
            if self.settings.stage == "answers":
                model_result, precheck = self._run_model_answers_stage(
                    loaded=loaded,
                    normalized_questions=normalized_questions,
                    context_text=context_text,
                    paths=paths,
                    model_spec=model_spec,
                )
            elif self.settings.stage == "judge":
                model_result, precheck = self._run_model_judge_stage(
                    loaded=loaded,
                    normalized_questions=normalized_questions,
                    context_text=context_text,
                    paths=paths,
                    model_spec=model_spec,
                )
            else:
                model_result, precheck = self._run_model_full_evaluation(
                    loaded=loaded,
                    normalized_questions=normalized_questions,
                    context_text=context_text,
                    paths=paths,
                    model_spec=model_spec,
                )
            model_results.append(model_result)
            model_prechecks[model_spec.slug] = precheck

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

    def _run_model_full_evaluation(
        self,
        *,
        loaded: LoadedPatientArtifacts,
        normalized_questions: list[EvalQuestion],
        context_text: str,
        paths,
        model_spec: ModelSpec,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        started_at = datetime.now(UTC)
        (
            precheck,
            batches_payload,
            raw_predictions,
            scored_predictions,
            error_records,
            summary_payload,
        ) = self._run_answers_computation(
            loaded=loaded,
            normalized_questions=normalized_questions,
            context_text=context_text,
            model_spec=model_spec,
        )
        llm_judgments: list[dict[str, Any]] = []
        try:
            if summary_payload["run_status"] == "completed":
                llm_client = self._client_for_model(model_spec)
                judge_client = self._judge_client_for_model(
                    model_spec,
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
                    model_spec=model_spec,
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
                    "stage": "pipeline",
                    "error_type": type(exc).__name__,
                    "error_kind": "input_error" if isinstance(exc, ValueError) else "internal_error",
                    "message": str(exc),
                }
            )
            summary_payload = build_model_summary(
                loaded=loaded,
                settings=self.settings,
                model_spec=model_spec,
                scored_predictions=scored_predictions,
                error_records=error_records,
                precheck=precheck,
                started_at=started_at,
                batch_count=len(batches_payload["batches"]),
                run_status="failed_input_error" if isinstance(exc, ValueError) else "failed_internal_error",
            )

        return self._persist_model_outputs(
            paths=paths,
            model_spec=model_spec,
            precheck=precheck,
            batches_payload=batches_payload,
            raw_predictions=raw_predictions,
            scored_predictions=scored_predictions,
            memory_store_payload=empty_memory_store_payload(self.settings.evaluation_variant),
            memory_event_records=[],
            retrieval_store_payload=empty_retrieval_store_payload(self.settings.evaluation_variant),
            llm_judgments=llm_judgments,
            summary_payload=summary_payload,
            error_records=error_records,
        )

    def _run_model_answers_stage(
        self,
        *,
        loaded: LoadedPatientArtifacts,
        normalized_questions: list[EvalQuestion],
        context_text: str,
        paths,
        model_spec: ModelSpec,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        (
            precheck,
            batches_payload,
            raw_predictions,
            scored_predictions,
            error_records,
            summary_payload,
        ) = self._run_answers_computation(
            loaded=loaded,
            normalized_questions=normalized_questions,
            context_text=context_text,
            model_spec=model_spec,
            provisional_run_status="answers_completed",
        )
        return self._persist_model_outputs(
            paths=paths,
            model_spec=model_spec,
            precheck=precheck,
            batches_payload=batches_payload,
            raw_predictions=raw_predictions,
            scored_predictions=scored_predictions,
            memory_store_payload=empty_memory_store_payload(self.settings.evaluation_variant),
            memory_event_records=[],
            retrieval_store_payload=empty_retrieval_store_payload(self.settings.evaluation_variant),
            llm_judgments=[],
            summary_payload=summary_payload,
            error_records=error_records,
        )

    def _run_model_judge_stage(
        self,
        *,
        loaded: LoadedPatientArtifacts,
        normalized_questions: list[EvalQuestion],
        context_text: str,
        paths,
        model_spec: ModelSpec,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        del normalized_questions, context_text
        started_at = datetime.now(UTC)
        model_paths = build_model_artifact_paths(paths, model_spec)
        staged_artifacts = self._load_existing_model_outputs(model_paths)
        summary_payload = dict(staged_artifacts["summary_payload"])
        precheck = {
            "status": summary_payload.get("operational_metrics", {}).get(
                "token_precheck_status",
                "not_run",
            ),
            "evaluation_variant": staged_artifacts["run_config"].get(
                "evaluation_variant",
                summary_payload.get("evaluation_variant", self.settings.evaluation_variant),
            ),
            "base_model_slug": staged_artifacts["run_config"].get(
                "base_model_slug",
                summary_payload.get("base_model_slug"),
            ),
            "answer_context_strategy": staged_artifacts["run_config"].get(
                "answer_context_strategy",
                summary_payload.get("operational_metrics", {}).get(
                    "answer_context_strategy",
                    "full_context",
                ),
            ),
            "full_context_status": summary_payload.get("operational_metrics", {}).get(
                "full_context_precheck_status",
                "not_run",
            ),
            "memory": summary_payload.get("operational_metrics", {}).get("memory", {}),
        }
        scored_predictions = [dict(row) for row in staged_artifacts["scored_predictions"]]
        error_records = list(staged_artifacts["error_records"])
        llm_judgments: list[dict[str, Any]] = []
        prior_wall_time_seconds = float(
            staged_artifacts["summary_payload"].get("operational_metrics", {}).get(
                "total_wall_time_seconds",
                0.0,
            )
        )

        try:
            if summary_payload.get("run_status") in {"answers_completed", "completed"}:
                judge_client = self._judge_client_for_model(model_spec)
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
                    model_spec=model_spec,
                    scored_predictions=scored_predictions,
                    error_records=error_records,
                    precheck=precheck,
                    started_at=started_at,
                    batch_count=len(staged_artifacts["question_batches"].get("batches", [])),
                    run_status="completed",
                    prior_wall_time_seconds=prior_wall_time_seconds,
                )
        except Exception as exc:
            error_records.append(
                {
                    "timestamp": utc_now_iso(),
                    "stage": "judge_pipeline",
                    "error_type": type(exc).__name__,
                    "error_kind": "input_error" if isinstance(exc, ValueError) else "internal_error",
                    "message": str(exc),
                }
            )
            summary_payload = build_model_summary(
                loaded=loaded,
                settings=self.settings,
                model_spec=model_spec,
                scored_predictions=scored_predictions,
                error_records=error_records,
                precheck=precheck,
                started_at=started_at,
                batch_count=len(staged_artifacts["question_batches"].get("batches", [])),
                run_status="failed_input_error" if isinstance(exc, ValueError) else "failed_internal_error",
                prior_wall_time_seconds=prior_wall_time_seconds,
            )

        return self._persist_model_outputs(
            paths=paths,
            model_spec=model_spec,
            precheck=precheck,
            batches_payload=staged_artifacts["question_batches"],
            raw_predictions=staged_artifacts["raw_predictions"],
            scored_predictions=scored_predictions,
            memory_store_payload=staged_artifacts["memory_store_payload"],
            memory_event_records=staged_artifacts["memory_event_records"],
            retrieval_store_payload=staged_artifacts["retrieval_store_payload"],
            llm_judgments=llm_judgments,
            summary_payload=summary_payload,
            error_records=error_records,
        )

    def _run_answers_computation(
        self,
        *,
        loaded: LoadedPatientArtifacts,
        normalized_questions: list[EvalQuestion],
        context_text: str,
        model_spec: ModelSpec,
        provisional_run_status: str = "completed",
    ) -> tuple[
        dict[str, Any],
        dict[str, Any],
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
        dict[str, Any],
    ]:
        started_at = datetime.now(UTC)
        error_records: list[dict[str, Any]] = []
        raw_predictions: list[dict[str, Any]] = []
        scored_predictions: list[dict[str, Any]] = []
        batches_payload = {"batches": []}
        precheck = build_preflight_record(
            model_name=model_spec.model_name,
            tokenizer_name=self.settings.tokenizer_name,
            context_text=context_text,
            questions=normalized_questions,
            batch_size=self.settings.batch_size,
            max_model_len=model_spec.max_model_len,
            max_output_tokens=self.settings.max_output_tokens,
            safe_margin_tokens=self.settings.safe_margin_tokens,
            token_estimate_safety_multiplier=self.settings.token_estimate_safety_multiplier,
            enable_thinking=self.settings.enable_thinking,
        )

        try:
            llm_client = self._client_for_model(model_spec)
            batches = build_batches(
                normalized_questions,
                combined_payload=loaded.combined_payload,
                settings=self.settings,
                model_spec=model_spec,
            )
            batches_payload = {"batches": [batch.to_record() for batch in batches]}
            precheck = {
                **precheck,
                "evaluation_variant": self.settings.evaluation_variant,
                "answer_context_strategy": "full_context",
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
                model_name=model_spec.model_name,
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
                model_spec=model_spec,
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
                    "stage": "pipeline",
                    "error_type": type(exc).__name__,
                    "error_kind": "input_error" if isinstance(exc, ValueError) else "internal_error",
                    "message": str(exc),
                }
            )
            summary_payload = build_model_summary(
                loaded=loaded,
                settings=self.settings,
                model_spec=model_spec,
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
            error_records,
            summary_payload,
        )

    def _persist_model_outputs(
        self,
        *,
        paths,
        model_spec: ModelSpec,
        precheck: dict[str, Any],
        batches_payload: dict[str, Any],
        raw_predictions: list[dict[str, Any]],
        scored_predictions: list[dict[str, Any]],
        memory_store_payload: dict[str, Any],
        memory_event_records: list[dict[str, Any]],
        llm_judgments: list[dict[str, Any]],
        summary_payload: dict[str, Any],
        error_records: list[dict[str, Any]],
        retrieval_store_payload: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        model_paths = build_model_artifact_paths(paths, model_spec)
        write_model_outputs(
            model_paths,
            replace_existing=self.settings.replace_existing,
            run_config=build_model_run_config(self.settings, model_spec, model_paths),
            question_batches=batches_payload,
            memory_store=memory_store_payload,
            memory_events=memory_event_records,
            retrieval_store=retrieval_store_payload,
            raw_predictions=raw_predictions,
            scored_predictions=scored_predictions,
            llm_judgments=llm_judgments,
            summary_payload=summary_payload,
            error_records=error_records,
        )
        return summary_payload, precheck

    def _load_existing_model_outputs(self, model_paths) -> dict[str, Any]:
        missing_paths = [
            path
            for path in (
                model_paths.run_config_json,
                model_paths.question_batches_json,
                model_paths.raw_predictions_jsonl,
                model_paths.scored_predictions_jsonl,
                model_paths.summary_json,
                model_paths.errors_jsonl,
            )
            if not path.exists()
        ]
        if missing_paths:
            raise ValueError(
                "Judge stage requires existing answer artifacts. Missing: "
                + ", ".join(str(path) for path in missing_paths)
            )
        return {
            "run_config": load_json(model_paths.run_config_json),
            "question_batches": load_json(model_paths.question_batches_json),
            "memory_store_payload": (
                load_json(model_paths.memory_store_json)
                if model_paths.memory_store_json.exists()
                else {"mode": "unknown", "enabled": False}
            ),
            "memory_event_records": (
                load_jsonl(model_paths.memory_events_jsonl)
                if model_paths.memory_events_jsonl.exists()
                else []
            ),
            "retrieval_store_payload": (
                load_json(model_paths.retrieval_store_json)
                if model_paths.retrieval_store_json.exists()
                else {"mode": "unknown", "enabled": False}
            ),
            "raw_predictions": load_jsonl(model_paths.raw_predictions_jsonl),
            "scored_predictions": load_jsonl(model_paths.scored_predictions_jsonl),
            "llm_judgments": (
                load_jsonl(model_paths.llm_judgments_jsonl)
                if model_paths.llm_judgments_jsonl.exists()
                else []
            ),
            "summary_payload": load_json(model_paths.summary_json),
            "error_records": load_jsonl(model_paths.errors_jsonl),
        }

    def _client_for_model(self, model_spec: ModelSpec) -> Any:
        client = self.client_overrides.get(model_spec.model_name) or self.client_overrides.get(model_spec.slug)
        if client is not None:
            return client
        config = clone_config_for_model(
            self.base_config,
            provider=self.settings.provider,
            model_spec=model_spec,
            max_output_tokens=self.settings.max_output_tokens,
            base_url=self.settings.base_url,
            api_key_env=self.settings.api_key_env,
            timeout_seconds=self.settings.timeout_seconds,
        )
        return self.client_factory(config)

    def _judge_client_for_model(self, model_spec: ModelSpec, *, answer_client: Any | None = None) -> Any:
        if (
            answer_client is not None
            and model_spec.model_name == self.settings.judge_model_spec.model_name
            and self.settings.judge_base_url is None
        ):
            return answer_client
        for key in (
            JUDGE_CLIENT_OVERRIDE_KEY,
            self.settings.judge_model_spec.model_name,
            self.settings.judge_model_spec.slug,
        ):
            client = self.client_overrides.get(key)
            if client is not None:
                return client
        judge_base_url = self.settings.judge_base_url
        if (
            judge_base_url is None
            and model_spec.model_name == self.settings.judge_model_spec.model_name
        ):
            judge_base_url = self.settings.base_url
        if not judge_base_url:
            raise ValueError(
                "judge_base_url is required for non-27B evaluation models. Pass --judge-base-url for Quest-hosted Qwen/Qwen3.5-27B judging."
            )
        config = clone_config_for_model(
            self.base_config,
            provider=self.settings.provider,
            model_spec=self.settings.judge_model_spec,
            max_output_tokens=self.settings.judge_max_output_tokens,
            base_url=judge_base_url,
            api_key_env=self.settings.api_key_env,
            timeout_seconds=self.settings.timeout_seconds,
        )
        return self.client_factory(config)


def normalize_benchmark(payload: dict[str, Any]) -> list[EvalQuestion]:
    qas = payload.get("qas")
    if not isinstance(qas, list) or not qas:
        raise ValueError("benchmark_qa.json must contain a non-empty qas list")
    normalized_questions: list[EvalQuestion] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(qas, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"qas[{index}] must be an object")
        legacy_fields = [field for field in ("question_class", "options") if field in item]
        if legacy_fields:
            raise ValueError(f"qas[{index}] must not contain legacy fields: {legacy_fields}")
        qa_id = str(item.get("qa_id") or "").strip()
        scope = str(item.get("scope") or "").strip()
        question = str(item.get("question") or "").strip()
        answer = str(item.get("answer") or "").strip()
        question_type = str(item.get("question_type") or "").strip()
        evidence = item.get("evidence")
        if not qa_id:
            raise ValueError(f"qas[{index}].qa_id must be non-empty")
        if qa_id in seen_ids:
            raise ValueError(f"Duplicate qa_id in benchmark_qa.json: {qa_id}")
        seen_ids.add(qa_id)
        if not scope:
            raise ValueError(f"qas[{index}].scope must be non-empty")
        if not question_type:
            raise ValueError(f"qas[{index}].question_type must be non-empty")
        if not question:
            raise ValueError(f"qas[{index}].question must be non-empty")
        if not answer:
            raise ValueError(f"qas[{index}].answer must be non-empty")
        if not isinstance(evidence, dict):
            raise ValueError(f"qas[{index}].evidence must be an object")
        normalized_questions.append(
            EvalQuestion(
                qa_id=qa_id,
                scope=scope,
                question_type=question_type,
                question=question,
                answer=answer,
                evidence=evidence,
                is_adversarial=question_type == "adversarial",
            )
        )
    return normalized_questions


def build_patient_config_record(
    *,
    settings: EvaluationSettings,
    loaded: LoadedPatientArtifacts,
    paths,
) -> dict[str, Any]:
    return {
        "subject_id": str(loaded.subject_id),
        "patient_dir": str(loaded.patient_root),
        "evaluation_dir": str(paths.evaluation_root),
        "benchmark_root": str(loaded.patient_root.parent),
        "evaluation_root": str(settings.evaluation_root),
        "stage": settings.stage,
        "evaluation_variant": settings.evaluation_variant,
        "provider": settings.provider,
        "base_url": settings.base_url,
        "judge_base_url": settings.judge_base_url,
        "api_key_env": settings.api_key_env,
        "timeout_seconds": settings.timeout_seconds,
        "retry_limit": settings.retry_limit,
        "batch_size": settings.batch_size,
        "max_output_tokens": settings.max_output_tokens,
        "judge_max_output_tokens": settings.judge_max_output_tokens,
        "safe_margin_tokens": settings.safe_margin_tokens,
        "token_estimate_safety_multiplier": settings.token_estimate_safety_multiplier,
        "enable_thinking": settings.enable_thinking,
        **(
            {"memory": build_mem0_settings_record(settings)}
            if settings.evaluation_variant in MEMORY_EVALUATION_VARIANTS
            else {}
        ),
        **(
            {"rag": build_rag_settings_record(settings)}
            if settings.evaluation_variant in RAG_EVALUATION_VARIANTS
            else {}
        ),
        **({"mem0": build_mem0_settings_record(settings)} if settings.evaluation_variant == "mem0" else {}),
        "effective_prompt_budget_tokens": (
            settings.model_specs[0].max_model_len
            - settings.max_output_tokens
            - settings.safe_margin_tokens
            if settings.model_specs
            else 0
        ),
        "replace_existing": settings.replace_existing,
        "judge_model_spec": {
            "model_name": settings.judge_model_spec.model_name,
            "slug": settings.judge_model_spec.slug,
            "tensor_parallel_size": settings.judge_model_spec.tensor_parallel_size,
            "max_model_len": settings.judge_model_spec.max_model_len,
        },
        "model_specs": [
            {
                "model_name": spec.model_name,
                "slug": spec.slug,
                "tensor_parallel_size": spec.tensor_parallel_size,
                "max_model_len": spec.max_model_len,
            }
            for spec in settings.model_specs
        ],
    }


def build_model_run_config(
    settings: EvaluationSettings,
    model_spec: ModelSpec,
    model_paths,
) -> dict[str, Any]:
    return {
        "stage": settings.stage,
        "evaluation_variant": settings.evaluation_variant,
        "answer_context_strategy": (
            "memory"
            if settings.evaluation_variant in MEMORY_EVALUATION_VARIANTS
            else "rag"
            if settings.evaluation_variant in RAG_EVALUATION_VARIANTS
            else "full_context"
        ),
        "provider": settings.provider,
        "base_url": settings.base_url,
        "judge_base_url": settings.judge_base_url,
        "api_key_env": settings.api_key_env,
        "timeout_seconds": settings.timeout_seconds,
        "retry_limit": settings.retry_limit,
        "batch_size": settings.batch_size,
        "max_output_tokens": settings.max_output_tokens,
        "judge_max_output_tokens": settings.judge_max_output_tokens,
        "safe_margin_tokens": settings.safe_margin_tokens,
        "token_estimate_safety_multiplier": settings.token_estimate_safety_multiplier,
        "enable_thinking": settings.enable_thinking,
        **(
            {"memory": build_mem0_settings_record(settings)}
            if settings.evaluation_variant in MEMORY_EVALUATION_VARIANTS
            else {}
        ),
        **(
            {"rag": build_rag_settings_record(settings)}
            if settings.evaluation_variant in RAG_EVALUATION_VARIANTS
            else {}
        ),
        **({"mem0": build_mem0_settings_record(settings)} if settings.evaluation_variant == "mem0" else {}),
        "replace_existing": settings.replace_existing,
        "evaluation_root": str(settings.evaluation_root),
        "model_name": model_spec.model_name,
        "model_slug": model_spec.slug,
        **(
            {"base_model_slug": base_slug}
            if (base_slug := base_slug_for_memory_slug(model_spec.slug, settings.evaluation_variant))
            or (base_slug := base_slug_for_rag_slug(model_spec.slug, settings.evaluation_variant))
            else {}
        ),
        "tensor_parallel_size": model_spec.tensor_parallel_size,
        "max_model_len": model_spec.max_model_len,
        "judge_model_name": settings.judge_model_spec.model_name,
        "judge_model_slug": settings.judge_model_spec.slug,
        "artifact_paths": model_paths.artifact_paths(),
    }


def build_mem0_settings_record(settings: EvaluationSettings) -> dict[str, Any]:
    return {
        "memory_method": settings.evaluation_variant,
        "chunk_token_cap": int(settings.mem0_chunk_token_cap),
        "previous_chunk_summaries": int(settings.mem0_previous_chunk_summaries),
        "max_candidate_memories": int(settings.mem0_max_candidate_memories),
        "similar_memories": int(settings.mem0_similar_memories),
        "max_update_memories": int(settings.mem0_max_update_memories),
        "answer_retrieval_top_k": int(settings.mem0_answer_retrieval_top_k),
        "max_answer_memories": int(settings.mem0_max_answer_memories),
        "max_output_tokens": int(settings.mem0_max_output_tokens),
        "retrieval_backend": settings.mem0_retrieval_backend,
        "embedding_model": settings.mem0_embedding_model,
        "embedding_device": settings.mem0_embedding_device,
        "embedding_gpu_device_ids": settings.mem0_embedding_gpu_device_ids,
        "embedding_batch_size": int(settings.mem0_embedding_batch_size),
        "embedding_max_length": int(settings.mem0_embedding_max_length),
        "model_max_len": settings.mem0_model_max_len,
        "model_tensor_parallel_size": settings.mem0_model_tensor_parallel_size,
    }


def build_rag_settings_record(settings: EvaluationSettings) -> dict[str, Any]:
    return {
        "rag_method": settings.rag_method,
        "document_unit": settings.rag_document_unit,
        "selection_policy": settings.rag_selection_policy,
        "render_order": settings.rag_render_order,
        "embedding_model": settings.rag_embedding_model,
        "embedding_device": settings.rag_embedding_device,
        "embedding_gpu_device_ids": settings.rag_embedding_gpu_device_ids,
        "embedding_batch_size": int(settings.rag_embedding_batch_size),
        "embedding_max_length": int(settings.rag_embedding_max_length),
        "model_max_len": settings.rag_model_max_len,
        "model_tensor_parallel_size": settings.rag_model_tensor_parallel_size,
    }


def build_context_stats(
    *,
    loaded: LoadedPatientArtifacts,
    normalized_questions: list[EvalQuestion],
    context_text: str,
    model_prechecks: dict[str, Any],
) -> dict[str, Any]:
    return {
        "subject_id": str(loaded.subject_id),
        "admission_count": len(loaded.combined_payload["admissions"]),
        "total_turns": count_total_turns(loaded.combined_payload),
        "rendered_context_characters": len(context_text),
        "question_counts": {
            "total": len(normalized_questions),
            "answerable": sum(1 for question in normalized_questions if not question.is_adversarial),
            "adversarial": sum(1 for question in normalized_questions if question.is_adversarial),
        },
        "truncation": {
            "answer_context_strategy": next(
                (
                    str(precheck.get("batch_contexts", [{}])[0].get("strategy"))
                    for precheck in model_prechecks.values()
                    if precheck.get("batch_contexts")
                ),
                "not_run",
            ),
            "models_with_truncation": [
                slug
                for slug, precheck in sorted(model_prechecks.items())
                if int(precheck.get("truncated_batch_count", 0)) > 0
            ],
            "truncated_batch_count": sum(
                int(precheck.get("truncated_batch_count", 0))
                for precheck in model_prechecks.values()
            ),
        },
        "model_prechecks": model_prechecks,
    }


def build_model_summary(
    *,
    loaded: LoadedPatientArtifacts,
    settings: EvaluationSettings,
    model_spec: ModelSpec,
    scored_predictions: list[dict[str, Any]],
    error_records: list[dict[str, Any]],
    precheck: dict[str, Any],
    started_at: datetime,
    batch_count: int,
    run_status: str,
    prior_wall_time_seconds: float = 0.0,
) -> dict[str, Any]:
    top_level = build_top_level_metrics(scored_predictions)
    answer_failed_count = sum(1 for row in scored_predictions if row.get("status") == "answer_failed")
    answer_failed_percent = (
        round(100.0 * answer_failed_count / len(scored_predictions), 2)
        if scored_predictions
        else 0.0
    )
    return {
        "subject_id": str(loaded.subject_id),
        "patient_dir": str(loaded.patient_root),
        "provider": settings.provider,
        "model_name": model_spec.model_name,
        "model_slug": model_spec.slug,
        "evaluation_variant": precheck.get("evaluation_variant", settings.evaluation_variant),
        **(
            {"base_model_slug": precheck["base_model_slug"]}
            if precheck.get("base_model_slug")
            else {}
        ),
        "run_status": run_status,
        "answer_failed_percent": answer_failed_percent,
        **top_level,
        "breakdowns": build_summary_breakdowns(scored_predictions),
        "operational_metrics": {
            "answer_context_strategy": precheck.get("answer_context_strategy", "full_context"),
            "evaluation_variant": precheck.get("evaluation_variant", settings.evaluation_variant),
            "token_precheck_status": precheck.get("status", "not_run"),
            "full_context_precheck_status": precheck.get(
                "full_context_status",
                precheck.get("status", "not_run"),
            ),
            "num_batches": int(batch_count),
            "truncated_batch_count": int(precheck.get("truncated_batch_count", 0)),
            "max_estimated_batch_prompt_tokens": int(precheck.get("max_estimated_batch_prompt_tokens", 0)),
            "max_adjusted_estimated_batch_prompt_tokens": int(
                precheck.get("max_adjusted_estimated_batch_prompt_tokens", 0)
            ),
            "format_error_count": _count_errors(error_records, "format_error"),
            "api_error_count": _count_errors(error_records, "api_error"),
            "context_length_error_count": _count_errors(error_records, "context_length_error"),
            "failed_prediction_count": sum(1 for row in scored_predictions if row.get("status") != "scored"),
            "schema_repaired_batch_count": int(precheck.get("schema_repaired_batch_count", 0)),
            "schema_repaired_prediction_count": int(precheck.get("schema_repaired_prediction_count", 0)),
            "schema_order_repaired_batch_count": int(precheck.get("schema_order_repaired_batch_count", 0)),
            "memory": precheck.get("memory", {}),
            "rag": precheck.get("rag", {}),
            "rag_passthrough": bool(precheck.get("rag_passthrough", False)),
            "total_wall_time_seconds": round(
                float(prior_wall_time_seconds) + (datetime.now(UTC) - started_at).total_seconds(),
                3,
            ),
        },
    }


def rebuild_comparison_summary(paths, *, subject_id: str) -> dict[str, Any]:
    summary_paths = list_existing_model_summary_paths(paths.evaluation_root)
    leaderboard_rows: list[dict[str, Any]] = []
    model_summaries: list[dict[str, Any]] = []
    for summary_path in summary_paths:
        summary = load_json(summary_path)
        model_summaries.append(summary)
        leaderboard_rows.append(
            {
                "model_name": summary["model_name"],
                "model_slug": summary["model_slug"],
                "run_status": summary["run_status"],
                "overall_score": float(summary.get("overall_score", 0.0)),
                "llm_score": float(summary.get("llm_score", 0.0)),
                "macro_f1_answerable": float(summary.get("macro_f1_answerable", 0.0)),
                "adversarial_accuracy": float(summary.get("adversarial_accuracy", 0.0)),
                "num_questions_total": int(summary.get("num_questions_total", 0)),
            }
        )
    leaderboard_rows.sort(
        key=lambda row: (
            -float(row["overall_score"]),
            -float(row["llm_score"]),
            -float(row["macro_f1_answerable"]),
            -float(row["adversarial_accuracy"]),
            str(row["model_slug"]),
        )
    )
    breakdown_rows = build_comparison_breakdown_rows(model_summaries)
    ensure_dir(paths.comparison_dir)
    write_json(
        paths.comparison_dir / "leaderboard.json",
        {
            "subject_id": subject_id,
            "models": leaderboard_rows,
            "breakdowns": breakdown_rows,
        },
    )
    write_csv(
        paths.comparison_dir / "leaderboard.csv",
        leaderboard_rows,
        fieldnames=[
            "model_slug",
            "model_name",
            "run_status",
            "overall_score",
            "llm_score",
            "macro_f1_answerable",
            "adversarial_accuracy",
            "num_questions_total",
        ],
    )
    write_json(paths.comparison_dir / "breakdowns.json", {"subject_id": subject_id, "rows": breakdown_rows})
    write_csv(
        paths.comparison_dir / "breakdowns.csv",
        breakdown_rows,
        fieldnames=[
            "breakdown",
            "group",
            "model_slug",
            "model_name",
            "run_status",
            "count",
            "answerable_count",
            "adversarial_count",
            "overall_score",
            "llm_score",
            "macro_f1_answerable",
            "adversarial_accuracy",
        ],
    )
    write_text(
        paths.comparison_dir / "summary.md",
        build_comparison_markdown(subject_id, leaderboard_rows, breakdown_rows),
    )
    return {"models": leaderboard_rows, "breakdowns": breakdown_rows}


def build_comparison_breakdown_rows(model_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary in model_summaries:
        breakdowns = summary.get("breakdowns", {})
        if not isinstance(breakdowns, dict):
            continue
        for breakdown_name in (
            "by_answerability",
            "by_scope",
            "by_adversarial_scope",
            "by_answerable_scope",
            "by_question_type",
        ):
            groups = breakdowns.get(breakdown_name, {})
            if not isinstance(groups, dict):
                continue
            for group_name, metrics in sorted(groups.items()):
                if not isinstance(metrics, dict):
                    continue
                rows.append(
                    {
                        "breakdown": breakdown_name,
                        "group": str(group_name),
                        "model_slug": summary.get("model_slug"),
                        "model_name": summary.get("model_name"),
                        "run_status": summary.get("run_status"),
                        "count": int(metrics.get("count", 0)),
                        "answerable_count": int(metrics.get("answerable_count", 0)),
                        "adversarial_count": int(metrics.get("adversarial_count", 0)),
                        "overall_score": float(metrics.get("overall_score", 0.0)),
                        "llm_score": float(metrics.get("llm_score", 0.0)),
                        "macro_f1_answerable": float(metrics.get("macro_f1_answerable", 0.0)),
                        "adversarial_accuracy": float(metrics.get("adversarial_accuracy", 0.0)),
                    }
                )
    rows.sort(
        key=lambda row: (
            str(row["breakdown"]),
            str(row["group"]),
            -float(row["overall_score"]),
            str(row["model_slug"]),
        )
    )
    return rows


def _count_errors(error_records: list[dict[str, Any]], error_kind: str) -> int:
    return sum(1 for record in error_records if record.get("error_kind") == error_kind)


def empty_memory_store_payload(evaluation_variant: str = "normal") -> dict[str, Any]:
    return {
        "mode": "none",
        "enabled": False,
        "evaluation_variant": evaluation_variant,
        "metrics": {
            "enabled": False,
            "total_memories": 0,
            "active_memories": 0,
            "deleted_memories": 0,
            "extraction_call_count": 0,
            "update_call_count": 0,
            "summary_call_count": 0,
        },
    }


def empty_retrieval_store_payload(evaluation_variant: str = "normal") -> dict[str, Any]:
    return {
        "mode": "none",
        "enabled": False,
        "evaluation_variant": evaluation_variant,
        "metrics": {
            "enabled": False,
            "document_count": 0,
        },
    }
