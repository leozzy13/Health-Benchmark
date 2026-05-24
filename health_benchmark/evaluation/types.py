from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


CANONICAL_ABSTENTION_ANSWER = "the question is not answerable"


@dataclass(frozen=True)
class ModelSpec:
    model_name: str
    slug: str
    tensor_parallel_size: int
    max_model_len: int


@dataclass(frozen=True)
class LoadedPatientArtifacts:
    subject_id: str
    patient_root: Path
    combined_payload: dict[str, Any]
    benchmark_payload: dict[str, Any]


@dataclass(frozen=True)
class EvalQuestion:
    qa_id: str
    scope: str
    question_type: str
    question: str
    answer: str
    evidence: dict[str, Any]
    is_adversarial: bool

    def model_question(self) -> dict[str, str]:
        return {
            "qa_id": self.qa_id,
            "question": self.question,
        }

    def to_record(self) -> dict[str, Any]:
        return {
            "qa_id": self.qa_id,
            "scope": self.scope,
            "question_type": self.question_type,
            "question": self.question,
            "answer": self.answer,
            "evidence": self.evidence,
            "is_adversarial": self.is_adversarial,
        }


@dataclass(frozen=True)
class QuestionBatch:
    batch_id: str
    questions: list[EvalQuestion]
    estimated_prompt_tokens: int
    adjusted_estimated_prompt_tokens: int
    context_text: str
    context_record: dict[str, Any]

    def qa_ids(self) -> list[str]:
        return [question.qa_id for question in self.questions]

    def model_payload(self) -> dict[str, Any]:
        return {
            "questions": [question.model_question() for question in self.questions],
        }

    def to_record(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "qa_ids": self.qa_ids(),
            "estimated_prompt_tokens": int(self.estimated_prompt_tokens),
            "adjusted_estimated_prompt_tokens": int(self.adjusted_estimated_prompt_tokens),
            "context": self.context_record,
            "questions": [question.to_record() for question in self.questions],
            "model_payload": self.model_payload(),
        }


@dataclass(frozen=True)
class AnswerPrediction:
    qa_id: str
    prediction: str
    batch_id: str


@dataclass(frozen=True)
class PromptTokenEstimate:
    total_tokens: int
    encoding_name: str


@dataclass(frozen=True)
class EvaluationPaths:
    patient_root: Path
    evaluation_root: Path
    comparison_dir: Path
    config_json: Path
    context_stats_json: Path
    benchmark_snapshot_json: Path

    def model_dir(self, model_slug: str) -> Path:
        return self.evaluation_root / model_slug


@dataclass(frozen=True)
class ModelArtifactPaths:
    model_dir: Path
    run_config_json: Path
    question_batches_json: Path
    memory_store_json: Path
    memory_events_jsonl: Path
    retrieval_store_json: Path
    raw_predictions_jsonl: Path
    scored_predictions_jsonl: Path
    llm_judgments_jsonl: Path
    summary_json: Path
    errors_jsonl: Path

    def artifact_paths(self) -> dict[str, str]:
        return {
            "run_config_json": str(self.run_config_json),
            "question_batches_json": str(self.question_batches_json),
            "memory_store_json": str(self.memory_store_json),
            "memory_events_jsonl": str(self.memory_events_jsonl),
            "retrieval_store_json": str(self.retrieval_store_json),
            "raw_predictions_jsonl": str(self.raw_predictions_jsonl),
            "scored_predictions_jsonl": str(self.scored_predictions_jsonl),
            "llm_judgments_jsonl": str(self.llm_judgments_jsonl),
            "summary_json": str(self.summary_json),
            "errors_jsonl": str(self.errors_jsonl),
        }


class AnswerItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    qa_id: str
    prediction: str


class AnswerBatchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answers: list[AnswerItem]


class AnswerableJudgeItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    qa_id: str
    score: Literal[0, 1]


class AnswerableJudgeBatchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    judgments: list[AnswerableJudgeItem]
