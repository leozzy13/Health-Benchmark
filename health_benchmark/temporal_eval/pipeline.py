from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Sequence

from ..scripts.config import BenchmarkConfig
from ..scripts.llm_client import build_llm_client
from ..scripts.utils import ensure_dir, utc_now_iso, write_json, write_text
from .answer_runner import run_answer_batches
from .batch_builder import build_batches
from .config import (
    EvaluationSettings,
    build_evaluation_paths,
    build_model_artifact_paths,
    clone_config_for_model,
)
from .context_renderer import count_total_turns, render_full_patient_context
from .io_utils import list_existing_model_summary_paths, load_json, write_csv, write_model_outputs
from .loader import load_patient_artifacts
from .scoring import build_summary_breakdowns, build_top_level_metrics, score_predictions
from .summary_writer import build_comparison_markdown
from .token_budget import build_preflight_record
from .types import EvalQuestion, LoadedPatientArtifacts, ModelSpec


ClientFactory = Callable[[BenchmarkConfig], Any]


class TemporalEvaluationPipeline:
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
        paths = build_evaluation_paths(patient_root)
        ensure_dir(paths.evaluation_root)

        model_results: list[dict[str, Any]] = []
        model_prechecks: dict[str, Any] = {}
        for model_spec in self.settings.model_specs:
            model_result, precheck = self._run_model_evaluation(
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

        comparison = rebuild_comparison_summary(paths, subject_id=str(loaded.subject_id))
        failed_models = [result["model_slug"] for result in model_results if result["run_status"] != "completed"]
        return {
            "subject_id": int(subject_id),
            "status": "completed" if not failed_models else "failed",
            "patient_dir": str(patient_root),
            "evaluation_dir": str(paths.evaluation_root),
            "completed_models": [result["model_slug"] for result in model_results if result["run_status"] == "completed"],
            "failed_models": failed_models,
            "comparison_summary_path": str(paths.comparison_dir / "summary.md"),
            "comparison_models": comparison["models"],
        }

    def _run_model_evaluation(
        self,
        *,
        loaded: LoadedPatientArtifacts,
        normalized_questions: list[EvalQuestion],
        context_text: str,
        paths,
        model_spec: ModelSpec,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
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
        )

        try:
            if precheck["status"] == "context_too_long_for_fixed_batch_10":
                summary_payload = build_model_summary(
                    loaded=loaded,
                    settings=self.settings,
                    model_spec=model_spec,
                    scored_predictions=[],
                    error_records=[],
                    precheck=precheck,
                    started_at=started_at,
                    batch_count=0,
                    run_status="context_too_long_for_fixed_batch_10",
                )
            else:
                batches = build_batches(
                    normalized_questions,
                    context_text=context_text,
                    settings=self.settings,
                    model_name=model_spec.model_name,
                )
                batches_payload = {"batches": [batch.to_record() for batch in batches]}
                llm_client = self._client_for_model(model_spec)
                answer_predictions, raw_predictions, answer_errors, answer_failures = run_answer_batches(
                    llm_client,
                    batches,
                    context_text=context_text,
                    model_name=model_spec.model_name,
                    save_raw_response=self.settings.save_raw_response,
                    max_output_tokens=self.settings.max_output_tokens,
                )
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

        model_paths = build_model_artifact_paths(paths, model_spec)
        write_model_outputs(
            model_paths,
            replace_existing=self.settings.replace_existing,
            run_config=build_model_run_config(self.settings, model_spec, model_paths),
            question_batches=batches_payload,
            raw_predictions=raw_predictions,
            scored_predictions=scored_predictions,
            summary_payload=summary_payload,
            error_records=error_records,
        )
        return summary_payload, precheck

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
        "provider": settings.provider,
        "base_url": settings.base_url,
        "api_key_env": settings.api_key_env,
        "timeout_seconds": settings.timeout_seconds,
        "batch_size": settings.batch_size,
        "max_output_tokens": settings.max_output_tokens,
        "safe_margin_tokens": settings.safe_margin_tokens,
        "replace_existing": settings.replace_existing,
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
        "provider": settings.provider,
        "base_url": settings.base_url,
        "api_key_env": settings.api_key_env,
        "timeout_seconds": settings.timeout_seconds,
        "batch_size": settings.batch_size,
        "max_output_tokens": settings.max_output_tokens,
        "safe_margin_tokens": settings.safe_margin_tokens,
        "replace_existing": settings.replace_existing,
        "model_name": model_spec.model_name,
        "model_slug": model_spec.slug,
        "tensor_parallel_size": model_spec.tensor_parallel_size,
        "max_model_len": model_spec.max_model_len,
        "artifact_paths": model_paths.artifact_paths(),
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
) -> dict[str, Any]:
    top_level = build_top_level_metrics(scored_predictions)
    return {
        "subject_id": str(loaded.subject_id),
        "patient_dir": str(loaded.patient_root),
        "provider": settings.provider,
        "model_name": model_spec.model_name,
        "model_slug": model_spec.slug,
        "run_status": run_status,
        **top_level,
        "breakdowns": build_summary_breakdowns(scored_predictions),
        "operational_metrics": {
            "token_precheck_status": precheck.get("status", "not_run"),
            "num_batches": int(batch_count),
            "format_error_count": _count_errors(error_records, "format_error"),
            "api_error_count": _count_errors(error_records, "api_error"),
            "failed_prediction_count": sum(1 for row in scored_predictions if row.get("status") != "scored"),
            "total_wall_time_seconds": round((datetime.now(UTC) - started_at).total_seconds(), 3),
        },
    }


def rebuild_comparison_summary(paths, *, subject_id: str) -> dict[str, Any]:
    summary_paths = list_existing_model_summary_paths(paths.evaluation_root)
    leaderboard_rows: list[dict[str, Any]] = []
    for summary_path in summary_paths:
        summary = load_json(summary_path)
        leaderboard_rows.append(
            {
                "model_name": summary["model_name"],
                "model_slug": summary["model_slug"],
                "run_status": summary["run_status"],
                "overall_score": float(summary.get("overall_score", 0.0)),
                "macro_f1_answerable": float(summary.get("macro_f1_answerable", 0.0)),
                "adversarial_accuracy": float(summary.get("adversarial_accuracy", 0.0)),
                "num_questions_total": int(summary.get("num_questions_total", 0)),
            }
        )
    leaderboard_rows.sort(
        key=lambda row: (
            -float(row["overall_score"]),
            -float(row["macro_f1_answerable"]),
            -float(row["adversarial_accuracy"]),
            str(row["model_slug"]),
        )
    )
    ensure_dir(paths.comparison_dir)
    write_json(paths.comparison_dir / "leaderboard.json", {"subject_id": subject_id, "models": leaderboard_rows})
    write_csv(
        paths.comparison_dir / "leaderboard.csv",
        leaderboard_rows,
        fieldnames=[
            "model_slug",
            "model_name",
            "run_status",
            "overall_score",
            "macro_f1_answerable",
            "adversarial_accuracy",
            "num_questions_total",
        ],
    )
    write_text(
        paths.comparison_dir / "summary.md",
        build_comparison_markdown(subject_id, leaderboard_rows),
    )
    return {"models": leaderboard_rows}


def _count_errors(error_records: list[dict[str, Any]], error_kind: str) -> int:
    return sum(1 for record in error_records if record.get("error_kind") == error_kind)
