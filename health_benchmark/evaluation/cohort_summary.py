from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io_utils import load_json, write_csv, write_json
from .loader import read_patient_manifest


COHORT_LEADERBOARD_CSV = "cohort_leaderboard_detailed.csv"
COHORT_LEADERBOARD_JSON = "cohort_leaderboard_detailed.json"
COHORT_QUESTION_TYPE_LEADERBOARD_CSV = "cohort_leaderboard_question_types.csv"

COHORT_REGULAR_QUESTION_TYPES = (
    "medical_reasoning",
    "care_plan_rationale",
    "longitudinal_progression",
    "cross_admission_comparison",
    "frequency_pattern",
)

COHORT_LEADERBOARD_FIELDNAMES = [
    "model_slug",
    "model_name",
    "evaluation_variant",
    "cohort_status",
    "num_patients",
    "num_questions_total",
    "num_answerable",
    "num_adversarial",
    "overall_score",
    "overall_normal_f1",
    "overall_normal_llm_score",
    "overall_adversarial_accuracy",
    "single_admission_score",
    "single_admission_normal_f1",
    "single_admission_normal_llm_score",
    "single_admission_adversarial_accuracy",
    "cross_admission_score",
    "cross_admission_normal_f1",
    "cross_admission_normal_llm_score",
    "cross_admission_adversarial_accuracy",
]

COHORT_QUESTION_TYPE_LEADERBOARD_FIELDNAMES = [
    "model_slug",
    "model_name",
    "evaluation_variant",
    "cohort_status",
    "num_adversarial",
    "medical_reasoning_f1",
    "medical_reasoning_llm_score",
    "care_plan_rationale_f1",
    "care_plan_rationale_llm_score",
    "longitudinal_progression_f1",
    "longitudinal_progression_llm_score",
    "cross_admission_comparison_f1",
    "cross_admission_comparison_llm_score",
    "frequency_pattern_f1",
    "frequency_pattern_llm_score",
    "single_admission_adversarial_accuracy",
    "cross_admission_adversarial_accuracy",
]


@dataclass(frozen=True)
class WeightedMetric:
    value_field: str
    count_field: str
    breakdown_name: str | None = None
    group_name: str | None = None


@dataclass(frozen=True)
class ModelCoverage:
    model_slug: str
    subject_ids: list[str]
    summaries: list[dict[str, Any]]
    missing_subject_ids: list[str]
    malformed_subjects: list[dict[str, str]]

    @property
    def available_count(self) -> int:
        return len(self.summaries)

    @property
    def completed_count(self) -> int:
        return sum(1 for summary in self.summaries if str(summary.get("run_status") or "") == "completed")

    @property
    def answers_completed_count(self) -> int:
        return sum(1 for summary in self.summaries if str(summary.get("run_status") or "") == "answers_completed")

    @property
    def failed_count(self) -> int:
        return sum(1 for summary in self.summaries if str(summary.get("run_status") or "").startswith("failed"))

    @property
    def incomplete_subjects(self) -> list[dict[str, str]]:
        return [
            {
                "subject_id": str(summary.get("subject_id") or ""),
                "run_status": str(summary.get("run_status") or ""),
            }
            for summary in self.summaries
            if str(summary.get("run_status") or "") != "completed"
        ]

    @property
    def cohort_status(self) -> str:
        if self.available_count == 0:
            return "malformed_only"
        if (
            self.completed_count == len(self.subject_ids)
            and not self.missing_subject_ids
            and not self.malformed_subjects
        ):
            return "completed"
        return "partial"

    def metadata(self) -> dict[str, Any]:
        statuses: dict[str, int] = {}
        for summary in self.summaries:
            status = str(summary.get("run_status") or "")
            statuses[status] = statuses.get(status, 0) + 1
        return {
            "model_slug": self.model_slug,
            "cohort_status": self.cohort_status,
            "num_patients_expected": len(self.subject_ids),
            "num_patients_available": self.available_count,
            "num_patients_completed": self.completed_count,
            "num_patients_answers_completed": self.answers_completed_count,
            "num_patients_failed": self.failed_count,
            "num_patients_missing": len(self.missing_subject_ids),
            "run_status_counts": statuses,
            "missing_subject_ids": self.missing_subject_ids,
            "malformed_subjects": self.malformed_subjects,
            "incomplete_subjects": self.incomplete_subjects,
        }


def summarize_evaluation_cohort(
    *,
    evaluation_root: Path,
    patient_manifest: Path,
) -> dict[str, Any]:
    subject_ids = [str(subject_id) for subject_id in read_patient_manifest(patient_manifest)]
    resolved_evaluation_root = evaluation_root.expanduser().resolve()
    rows: list[dict[str, Any]] = []
    question_type_rows_by_model_slug: dict[str, dict[str, Any]] = {}
    model_coverage: list[dict[str, Any]] = []
    excluded_models: list[dict[str, Any]] = []
    for model_slug in _discover_model_slugs(resolved_evaluation_root, subject_ids):
        coverage = _load_model_coverage(
            resolved_evaluation_root,
            subject_ids,
            model_slug,
        )
        metadata = coverage.metadata()
        model_coverage.append(metadata)
        if coverage.available_count == 0:
            excluded_models.append(metadata)
            continue
        row = _build_model_row(coverage)
        rows.append(row)
        question_type_rows_by_model_slug[coverage.model_slug] = _build_question_type_model_row(
            coverage,
            row,
        )
    rows.sort(
        key=lambda row: (
            -float(row["overall_score"]),
            -float(row["overall_normal_llm_score"]),
            -float(row["overall_normal_f1"]),
            -float(row["overall_adversarial_accuracy"]),
            str(row["model_slug"]),
        )
    )
    question_type_rows = [
        question_type_rows_by_model_slug[str(row["model_slug"])]
        for row in rows
    ]
    csv_path = resolved_evaluation_root / COHORT_LEADERBOARD_CSV
    question_type_csv_path = resolved_evaluation_root / COHORT_QUESTION_TYPE_LEADERBOARD_CSV
    json_path = resolved_evaluation_root / COHORT_LEADERBOARD_JSON
    write_csv(csv_path, rows, fieldnames=COHORT_LEADERBOARD_FIELDNAMES)
    write_csv(
        question_type_csv_path,
        question_type_rows,
        fieldnames=COHORT_QUESTION_TYPE_LEADERBOARD_FIELDNAMES,
    )
    payload = {
        "evaluation_root": str(resolved_evaluation_root),
        "patient_manifest": str(patient_manifest.expanduser().resolve()),
        "subject_ids": subject_ids,
        "num_patients": len(subject_ids),
        "models": rows,
        "model_coverage": sorted(model_coverage, key=lambda item: str(item.get("model_slug", ""))),
        "excluded_models": sorted(excluded_models, key=lambda item: str(item.get("model_slug", ""))),
        "outputs": {
            "csv": str(csv_path),
            "question_types_csv": str(question_type_csv_path),
            "json": str(json_path),
        },
    }
    write_json(json_path, payload)
    return payload


def _discover_model_slugs(evaluation_root: Path, subject_ids: list[str]) -> list[str]:
    slugs: set[str] = set()
    for subject_id in subject_ids:
        patient_root = evaluation_root / subject_id
        if not patient_root.exists():
            continue
        for child in patient_root.iterdir():
            if not child.is_dir() or child.name.startswith(".") or child.name == "comparison":
                continue
            if (child / "summary.json").exists():
                slugs.add(child.name)
    return sorted(slugs)


def _load_model_coverage(
    evaluation_root: Path,
    subject_ids: list[str],
    model_slug: str,
) -> ModelCoverage:
    summaries: list[dict[str, Any]] = []
    missing_subject_ids: list[str] = []
    malformed_subjects: list[dict[str, str]] = []
    for subject_id in subject_ids:
        summary_path = evaluation_root / subject_id / model_slug / "summary.json"
        if not summary_path.exists():
            missing_subject_ids.append(subject_id)
            continue
        try:
            summary = load_json(summary_path)
        except Exception as exc:  # pragma: no cover - defensive against corrupted artifacts
            malformed_subjects.append(
                {
                    "subject_id": subject_id,
                    "summary_path": str(summary_path),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        if not isinstance(summary, dict):
            malformed_subjects.append(
                {
                    "subject_id": subject_id,
                    "summary_path": str(summary_path),
                    "error": "summary payload is not a JSON object",
                }
            )
            continue
        summary.setdefault("subject_id", subject_id)
        summaries.append(summary)
    return ModelCoverage(
        model_slug=model_slug,
        subject_ids=subject_ids,
        summaries=summaries,
        missing_subject_ids=missing_subject_ids,
        malformed_subjects=malformed_subjects,
    )


def _build_model_row(coverage: ModelCoverage) -> dict[str, Any]:
    summaries = coverage.summaries
    model_names = [str(summary.get("model_name") or "") for summary in summaries if summary.get("model_name")]
    variants = [
        str(summary.get("evaluation_variant") or "")
        for summary in summaries
        if summary.get("evaluation_variant")
    ]
    return {
        "model_slug": coverage.model_slug,
        "model_name": _most_common(model_names),
        "evaluation_variant": _most_common(variants),
        "cohort_status": coverage.cohort_status,
        "num_patients": coverage.available_count,
        "num_questions_total": sum(_int_metric(summary, "num_questions_total") for summary in summaries),
        "num_answerable": sum(_int_metric(summary, "num_answerable") for summary in summaries),
        "num_adversarial": sum(_int_metric(summary, "num_adversarial") for summary in summaries),
        "overall_score": _composite_metric(
            summaries,
            [
                WeightedMetric("llm_score", "num_answerable"),
                WeightedMetric("adversarial_accuracy", "num_adversarial"),
            ],
            denominator_count=sum(_int_metric(summary, "num_questions_total") for summary in summaries),
        ),
        "overall_normal_f1": _weighted_metric(
            summaries,
            WeightedMetric("macro_f1_answerable", "num_answerable"),
        ),
        "overall_normal_llm_score": _weighted_metric(
            summaries,
            WeightedMetric("llm_score", "num_answerable"),
        ),
        "overall_adversarial_accuracy": _weighted_metric(
            summaries,
            WeightedMetric("adversarial_accuracy", "num_adversarial"),
        ),
        "single_admission_score": _composite_metric(
            summaries,
            [
                WeightedMetric("llm_score", "answerable_count", "by_answerable_scope", "single_admission"),
                WeightedMetric("adversarial_accuracy", "adversarial_count", "by_adversarial_scope", "single_admission"),
            ],
        ),
        "single_admission_normal_f1": _weighted_metric(
            summaries,
            WeightedMetric("macro_f1_answerable", "answerable_count", "by_answerable_scope", "single_admission"),
        ),
        "single_admission_normal_llm_score": _weighted_metric(
            summaries,
            WeightedMetric("llm_score", "answerable_count", "by_answerable_scope", "single_admission"),
        ),
        "single_admission_adversarial_accuracy": _weighted_metric(
            summaries,
            WeightedMetric("adversarial_accuracy", "adversarial_count", "by_adversarial_scope", "single_admission"),
        ),
        "cross_admission_score": _composite_metric(
            summaries,
            [
                WeightedMetric("llm_score", "answerable_count", "by_answerable_scope", "cross_admission"),
                WeightedMetric("adversarial_accuracy", "adversarial_count", "by_adversarial_scope", "cross_admission"),
            ],
        ),
        "cross_admission_normal_f1": _weighted_metric(
            summaries,
            WeightedMetric("macro_f1_answerable", "answerable_count", "by_answerable_scope", "cross_admission"),
        ),
        "cross_admission_normal_llm_score": _weighted_metric(
            summaries,
            WeightedMetric("llm_score", "answerable_count", "by_answerable_scope", "cross_admission"),
        ),
        "cross_admission_adversarial_accuracy": _weighted_metric(
            summaries,
            WeightedMetric("adversarial_accuracy", "adversarial_count", "by_adversarial_scope", "cross_admission"),
        ),
    }


def _build_question_type_model_row(
    coverage: ModelCoverage,
    model_row: dict[str, Any],
) -> dict[str, Any]:
    summaries = coverage.summaries
    row = {
        "model_slug": model_row["model_slug"],
        "model_name": model_row["model_name"],
        "evaluation_variant": model_row["evaluation_variant"],
        "cohort_status": model_row["cohort_status"],
        "num_adversarial": model_row["num_adversarial"],
    }
    for question_type in COHORT_REGULAR_QUESTION_TYPES:
        row[f"{question_type}_f1"] = _weighted_metric(
            summaries,
            WeightedMetric(
                "macro_f1_answerable",
                "answerable_count",
                "by_question_type",
                question_type,
            ),
        )
        row[f"{question_type}_llm_score"] = _weighted_metric(
            summaries,
            WeightedMetric(
                "llm_score",
                "answerable_count",
                "by_question_type",
                question_type,
            ),
        )
    row["single_admission_adversarial_accuracy"] = _weighted_metric(
        summaries,
        WeightedMetric("adversarial_accuracy", "adversarial_count", "by_adversarial_scope", "single_admission"),
    )
    row["cross_admission_adversarial_accuracy"] = _weighted_metric(
        summaries,
        WeightedMetric("adversarial_accuracy", "adversarial_count", "by_adversarial_scope", "cross_admission"),
    )
    return row


def _composite_metric(
    summaries: list[dict[str, Any]],
    components: list[WeightedMetric],
    *,
    denominator_count: int | None = None,
) -> float:
    numerator = 0.0
    component_denominator = 0
    for metric in components:
        for summary in summaries:
            source = _metric_source(summary, metric)
            if source is None:
                continue
            count = _int_metric(source, metric.count_field)
            component_denominator += count
            numerator += float(source.get(metric.value_field) or 0.0) * count
    denominator = component_denominator if denominator_count is None else denominator_count
    return round(numerator / denominator, 4) if denominator else 0.0


def _weighted_metric(summaries: list[dict[str, Any]], metric: WeightedMetric) -> float:
    numerator = 0.0
    denominator = 0
    for summary in summaries:
        source = _metric_source(summary, metric)
        if source is None:
            continue
        count = _int_metric(source, metric.count_field)
        denominator += count
        numerator += float(source.get(metric.value_field) or 0.0) * count
    return round(numerator / denominator, 4) if denominator else 0.0


def _metric_source(summary: dict[str, Any], metric: WeightedMetric) -> dict[str, Any] | None:
    if metric.breakdown_name is None:
        return summary
    breakdowns = summary.get("breakdowns", {})
    if not isinstance(breakdowns, dict):
        return None
    breakdown = breakdowns.get(metric.breakdown_name, {})
    if not isinstance(breakdown, dict):
        return None
    group = breakdown.get(metric.group_name or "", {})
    return group if isinstance(group, dict) else None


def _int_metric(payload: dict[str, Any], field: str) -> int:
    return int(payload.get(field) or 0)


def _most_common(values: list[str]) -> str:
    if not values:
        return ""
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
