from __future__ import annotations

import re
import string
from collections import Counter, defaultdict
from statistics import mean
from typing import Any

from .types import AnswerPrediction, CANONICAL_ABSTENTION_ANSWER, EvalQuestion


ADVERSARIAL_ANSWER_ALIASES = frozenset(
    {
        CANONICAL_ABSTENTION_ANSWER,
        "not answerable from the record",
        "not answerable from the provided context",
        "cannot be determined from the record",
        "no information available",
        "not mentioned",
    }
)
_ARTICLES_RE = re.compile(r"\b(a|an|the|and)\b")
_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_TABLE = str.maketrans({character: " " for character in string.punctuation})


def normalize_answer(value: str) -> str:
    normalized = str(value or "").replace(",", " ").lower()
    normalized = normalized.translate(_PUNCT_TABLE)
    normalized = _ARTICLES_RE.sub(" ", normalized)
    return _WHITESPACE_RE.sub(" ", normalized).strip()


NORMALIZED_ADVERSARIAL_ALIASES = frozenset(normalize_answer(alias) for alias in ADVERSARIAL_ANSWER_ALIASES)


def score_adversarial(prediction: str) -> float:
    return 1.0 if normalize_answer(prediction) in NORMALIZED_ADVERSARIAL_ALIASES else 0.0


def score_answerable(prediction: str, gold_answer: str) -> dict[str, float | bool]:
    whole_metrics = _score_token_overlap(
        normalize_answer(prediction).split(),
        normalize_answer(gold_answer).split(),
    )
    comma_metrics = _score_comma_fallback(prediction, gold_answer)
    if comma_metrics is None or float(whole_metrics["f1"]) > float(comma_metrics["f1"]):
        return {
            **whole_metrics,
            "used_comma_fallback": False,
        }
    return {
        **comma_metrics,
        "used_comma_fallback": True,
    }


def score_predictions(
    questions: list[EvalQuestion],
    answer_predictions: list[AnswerPrediction],
    answer_failures: dict[str, str],
) -> list[dict[str, Any]]:
    prediction_map = {prediction.qa_id: prediction for prediction in answer_predictions}
    scored_rows: list[dict[str, Any]] = []
    for question in questions:
        prediction = prediction_map.get(question.qa_id)
        row: dict[str, Any] = question.to_record()
        row.update(
            {
                "gold_answer": question.answer,
                "prediction": prediction.prediction if prediction is not None else None,
                "batch_id": prediction.batch_id if prediction is not None else None,
                "status": "scored",
                "precision": None,
                "recall": None,
                "f1": None,
                "abstention_accuracy": None,
                "per_question_score": 0.0,
                "used_comma_fallback": False,
            }
        )
        if prediction is None:
            row["status"] = answer_failures.get(question.qa_id, "missing_prediction")
            scored_rows.append(row)
            continue
        if question.is_adversarial:
            abstention_accuracy = score_adversarial(prediction.prediction)
            row["abstention_accuracy"] = abstention_accuracy
            row["per_question_score"] = abstention_accuracy
        else:
            metrics = score_answerable(prediction.prediction, question.answer)
            row["precision"] = metrics["precision"]
            row["recall"] = metrics["recall"]
            row["f1"] = metrics["f1"]
            row["per_question_score"] = metrics["f1"]
            row["used_comma_fallback"] = bool(metrics["used_comma_fallback"])
        scored_rows.append(row)
    return scored_rows


def build_summary_breakdowns(scored_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "by_scope": _metric_breakdown(scored_rows, "scope"),
        "by_question_type": _metric_breakdown(scored_rows, "question_type"),
        "by_answerability": _metric_breakdown(
            scored_rows,
            lambda row: "adversarial" if row.get("is_adversarial") else "answerable",
        ),
    }


def build_top_level_metrics(scored_rows: list[dict[str, Any]]) -> dict[str, Any]:
    answerable_rows = [row for row in scored_rows if not bool(row.get("is_adversarial"))]
    adversarial_rows = [row for row in scored_rows if bool(row.get("is_adversarial"))]
    return {
        "num_questions_total": len(scored_rows),
        "num_answerable": len(answerable_rows),
        "num_adversarial": len(adversarial_rows),
        "macro_precision_answerable": _macro_mean(answerable_rows, "precision"),
        "macro_recall_answerable": _macro_mean(answerable_rows, "recall"),
        "macro_f1_answerable": _macro_mean(answerable_rows, "f1"),
        "adversarial_accuracy": _macro_mean(adversarial_rows, "abstention_accuracy"),
        "overall_score": round(
            mean(float(row.get("per_question_score") or 0.0) for row in scored_rows),
            4,
        )
        if scored_rows
        else 0.0,
    }


def _metric_breakdown(
    rows: list[dict[str, Any]],
    key: str | Any,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if callable(key):
        for row in rows:
            grouped[str(key(row))].append(row)
    else:
        for row in rows:
            grouped[str(row.get(key))].append(row)
    return {
        group_key: _aggregate_rows(group_rows)
        for group_key, group_rows in sorted(grouped.items())
    }


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answerable_rows = [row for row in rows if not bool(row.get("is_adversarial"))]
    adversarial_rows = [row for row in rows if bool(row.get("is_adversarial"))]
    return {
        "count": len(rows),
        "answerable_count": len(answerable_rows),
        "adversarial_count": len(adversarial_rows),
        "macro_precision_answerable": _macro_mean(answerable_rows, "precision"),
        "macro_recall_answerable": _macro_mean(answerable_rows, "recall"),
        "macro_f1_answerable": _macro_mean(answerable_rows, "f1"),
        "adversarial_accuracy": _macro_mean(adversarial_rows, "abstention_accuracy"),
        "overall_score": round(
            mean(float(row.get("per_question_score") or 0.0) for row in rows),
            4,
        )
        if rows
        else 0.0,
    }


def _macro_mean(rows: list[dict[str, Any]], field: str) -> float:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    if not values:
        return 0.0
    return round(mean(values), 4)


def _score_token_overlap(pred_tokens: list[str], gold_tokens: list[str]) -> dict[str, float]:
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    precision = num_same / len(pred_tokens) if pred_tokens else 0.0
    recall = num_same / len(gold_tokens) if gold_tokens else 0.0
    f1 = 0.0 if num_same == 0 else (2 * precision * recall / (precision + recall))
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def _score_comma_fallback(prediction: str, gold_answer: str) -> dict[str, float] | None:
    if "," not in prediction and "," not in gold_answer:
        return None
    pred_parts = [normalize_answer(part) for part in prediction.split(",")]
    gold_parts = [normalize_answer(part) for part in gold_answer.split(",")]
    pred_parts = [part for part in pred_parts if part]
    gold_parts = [part for part in gold_parts if part]
    if not pred_parts or not gold_parts:
        return None
    precision_scores = [
        max(_score_token_overlap(pred_part.split(), gold_part.split())["f1"] for gold_part in gold_parts)
        for pred_part in pred_parts
    ]
    recall_scores = [
        max(_score_token_overlap(pred_part.split(), gold_part.split())["f1"] for pred_part in pred_parts)
        for gold_part in gold_parts
    ]
    precision = mean(precision_scores) if precision_scores else 0.0
    recall = mean(recall_scores) if recall_scores else 0.0
    f1 = 0.0 if precision == 0.0 or recall == 0.0 else (2 * precision * recall / (precision + recall))
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }
