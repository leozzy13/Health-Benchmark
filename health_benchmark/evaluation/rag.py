from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol, Sequence

from .answer_prompting import render_answer_prompt
from .config import EvaluationSettings
from .context_renderer import render_conversation_line
from .memory import LocalHFDenseEmbedder, Mem0EmbeddingError, _dense_dot, _normalize_dense_vector
from .token_budget import adjusted_prompt_tokens, effective_raw_prompt_budget, estimate_prompt_tokens
from .types import EvalQuestion, ModelSpec, QuestionBatch


RAG_CONTEXT_DESCRIPTION = "retrieved patient conversation excerpts"
RAG_CONTEXT_PAYLOAD_KEY = "retrieved_patient_context"
RAG_DENSE_BACKEND = "local_dense_hf"
RAG_BM25_BACKEND = "local_bm25"
_VISIBLE_TIMESTAMP_PATTERN = re.compile(
    r"\b(\d{4}-\d{2}-\d{2})(?:[ T](\d{2}:\d{2}(?::\d{2})?))?\b"
)
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


@dataclass(frozen=True)
class AdmissionDocument:
    doc_id: str
    admission_index: int
    hadm_id: str
    first_timestamp: str
    turn_count: int
    text: str

    def to_record(self, *, include_text: bool = False) -> dict[str, Any]:
        record: dict[str, Any] = {
            "doc_id": self.doc_id,
            "admission_index": int(self.admission_index),
            "hadm_id": self.hadm_id,
            "first_timestamp": self.first_timestamp,
            "turn_count": int(self.turn_count),
        }
        if include_text:
            record["text"] = self.text
        return record


class AdmissionRetriever(Protocol):
    backend_name: str

    def score_all(self, documents: Sequence[AdmissionDocument], *, query: str) -> list[dict[str, Any]]:
        ...

    def config_payload(self) -> dict[str, Any]:
        ...


class DenseAdmissionRetriever:
    backend_name = RAG_DENSE_BACKEND

    def __init__(self, embedder) -> None:
        self.embedder = embedder
        self.embedding_model = str(getattr(embedder, "model_name", ""))
        self.embedding_device = str(getattr(embedder, "device", ""))
        self.embedding_batch_size = int(getattr(embedder, "batch_size", 0) or 0)
        self.embedding_max_length = int(getattr(embedder, "max_length", 0) or 0)
        self._cache: dict[str, tuple[str, list[float]]] = {}

    @property
    def embedding_dimension(self) -> int | None:
        dimension = getattr(self.embedder, "embedding_dimension", None)
        if dimension:
            return int(dimension)
        for _doc_id, (_text, embedding) in self._cache.items():
            if embedding:
                return len(embedding)
        return None

    def ensure_index(self, documents: Sequence[AdmissionDocument]) -> None:
        missing_docs: list[AdmissionDocument] = []
        missing_texts: list[str] = []
        for document in documents:
            cached = self._cache.get(document.doc_id)
            if cached is None or cached[0] != document.text:
                missing_docs.append(document)
                missing_texts.append(document.text)
        if not missing_docs:
            return
        embeddings = self.embedder.embed(missing_texts)
        if len(embeddings) != len(missing_docs):
            raise Mem0EmbeddingError(
                f"Dense embedder returned {len(embeddings)} embeddings for {len(missing_docs)} admissions."
            )
        for document, embedding in zip(missing_docs, embeddings, strict=True):
            self._cache[document.doc_id] = (document.text, _normalize_dense_vector(embedding))

    def score_all(self, documents: Sequence[AdmissionDocument], *, query: str) -> list[dict[str, Any]]:
        if not str(query or "").strip():
            return [_score_record(document, 0.0, 0.0) for document in documents]
        self.ensure_index(documents)
        query_embeddings = self.embedder.embed([str(query)])
        if not query_embeddings:
            return [_score_record(document, 0.0, 0.0) for document in documents]
        query_embedding = _normalize_dense_vector(query_embeddings[0])
        scored: list[dict[str, Any]] = []
        for document in documents:
            cached = self._cache.get(document.doc_id)
            score = _dense_dot(query_embedding, cached[1]) if cached is not None else 0.0
            scored.append(_score_record(document, score, score, dense_score=score))
        scored.sort(key=lambda record: (-float(record["retrieval_score"]), str(record["doc_id"])))
        return _with_selected_ranks(scored)

    def config_payload(self) -> dict[str, Any]:
        return {
            "retrieval_backend": self.backend_name,
            "embedding_model": self.embedding_model,
            "embedding_device": self.embedding_device,
            "embedding_batch_size": self.embedding_batch_size,
            "embedding_max_length": self.embedding_max_length,
            "embedding_dimension": self.embedding_dimension,
        }


class BM25AdmissionRetriever:
    backend_name = RAG_BM25_BACKEND

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = float(k1)
        self.b = float(b)
        self._indexed_ids: tuple[str, ...] = ()
        self._term_freqs: list[dict[str, int]] = []
        self._doc_lengths: list[int] = []
        self._idf: dict[str, float] = {}
        self._avg_doc_length = 0.0

    def ensure_index(self, documents: Sequence[AdmissionDocument]) -> None:
        doc_ids = tuple(document.doc_id for document in documents)
        if doc_ids == self._indexed_ids:
            return
        tokenized = [_tokenize(document.text) for document in documents]
        self._term_freqs = []
        self._doc_lengths = []
        document_frequency: dict[str, int] = {}
        for tokens in tokenized:
            term_freq: dict[str, int] = {}
            for token in tokens:
                term_freq[token] = term_freq.get(token, 0) + 1
            self._term_freqs.append(term_freq)
            self._doc_lengths.append(len(tokens))
            for token in term_freq:
                document_frequency[token] = document_frequency.get(token, 0) + 1
        doc_count = max(1, len(documents))
        self._avg_doc_length = sum(self._doc_lengths) / doc_count
        self._idf = {
            token: math.log(1.0 + (doc_count - freq + 0.5) / (freq + 0.5))
            for token, freq in document_frequency.items()
        }
        self._indexed_ids = doc_ids

    def score_all(self, documents: Sequence[AdmissionDocument], *, query: str) -> list[dict[str, Any]]:
        self.ensure_index(documents)
        query_tokens = _tokenize(query)
        scored: list[dict[str, Any]] = []
        for index, document in enumerate(documents):
            score = self._score_document(index, query_tokens)
            scored.append(_score_record(document, score, score, bm25_score=score))
        scored.sort(key=lambda record: (-float(record["retrieval_score"]), str(record["doc_id"])))
        return _with_selected_ranks(scored)

    def _score_document(self, index: int, query_tokens: list[str]) -> float:
        if not query_tokens:
            return 0.0
        term_freq = self._term_freqs[index]
        doc_length = self._doc_lengths[index] if index < len(self._doc_lengths) else 0
        if doc_length <= 0:
            return 0.0
        score = 0.0
        for token in query_tokens:
            freq = term_freq.get(token, 0)
            if freq <= 0:
                continue
            idf = self._idf.get(token, 0.0)
            denominator = freq + self.k1 * (
                1.0 - self.b + self.b * doc_length / max(self._avg_doc_length, 1.0)
            )
            score += idf * (freq * (self.k1 + 1.0)) / denominator
        return float(score)

    def config_payload(self) -> dict[str, Any]:
        return {
            "retrieval_backend": self.backend_name,
            "k1": self.k1,
            "b": self.b,
        }


def build_rag_retriever(settings: EvaluationSettings) -> AdmissionRetriever:
    if settings.evaluation_variant == "bm25_rag":
        return BM25AdmissionRetriever()
    if settings.evaluation_variant != "embedding_rag":
        raise ValueError(f"Unsupported RAG evaluation variant: {settings.evaluation_variant}")
    embedder = LocalHFDenseEmbedder(
        model_name=settings.rag_embedding_model,
        device=settings.rag_embedding_device,
        gpu_device_ids=settings.rag_embedding_gpu_device_ids,
        batch_size=settings.rag_embedding_batch_size,
        max_length=settings.rag_embedding_max_length,
    )
    return DenseAdmissionRetriever(embedder)


def build_admission_documents(combined_payload: dict[str, Any]) -> list[AdmissionDocument]:
    documents: list[AdmissionDocument] = []
    for admission_index, admission in enumerate(combined_payload.get("admissions", []), start=1):
        lines = list(admission.get("conversation_lines") or [])
        text = "\n".join(render_conversation_line(line) for line in lines)
        first_timestamp = ""
        for line in lines:
            candidate = str(line.get("time") or "").strip()
            if candidate:
                first_timestamp = candidate
                break
        documents.append(
            AdmissionDocument(
                doc_id=f"admission_{admission_index:03d}",
                admission_index=admission_index,
                hadm_id=str(admission.get("hadm_id") or ""),
                first_timestamp=first_timestamp,
                turn_count=len(lines),
                text=text,
            )
        )
    return documents


def build_rag_store_payload(
    *,
    documents: Sequence[AdmissionDocument],
    settings: EvaluationSettings,
    retriever: AdmissionRetriever,
    passthrough: bool,
) -> dict[str, Any]:
    return {
        "mode": "rag",
        "enabled": True,
        "evaluation_variant": settings.evaluation_variant,
        "rag_method": settings.rag_method,
        "rag_passthrough": bool(passthrough),
        "document_unit": settings.rag_document_unit,
        "selection_policy": settings.rag_selection_policy,
        "render_order": settings.rag_render_order,
        "retriever": retriever.config_payload(),
        "metrics": {
            "enabled": True,
            "document_count": len(documents),
            "admission_count": len(documents),
        },
        "documents": [document.to_record(include_text=False) for document in documents],
    }


def build_rag_question_batches(
    questions: list[EvalQuestion],
    *,
    documents: Sequence[AdmissionDocument],
    retriever: AdmissionRetriever,
    settings: EvaluationSettings,
    model_spec: ModelSpec,
) -> list[QuestionBatch]:
    batches: list[QuestionBatch] = []
    for index, question in enumerate(questions, start=1):
        selection = select_rag_context_for_question(
            documents=documents,
            retriever=retriever,
            question=question,
            evaluation_variant=settings.evaluation_variant,
            rag_method=settings.rag_method,
            model_name=model_spec.model_name,
            tokenizer_name=settings.tokenizer_name,
            max_model_len=model_spec.max_model_len,
            max_output_tokens=settings.max_output_tokens,
            safe_margin_tokens=settings.safe_margin_tokens,
            token_estimate_safety_multiplier=settings.token_estimate_safety_multiplier,
        )
        batches.append(
            QuestionBatch(
                batch_id=f"question_{index:03d}",
                questions=[question],
                estimated_prompt_tokens=int(selection["estimated_prompt_tokens"]),
                adjusted_estimated_prompt_tokens=int(selection["adjusted_estimated_prompt_tokens"]),
                context_text=str(selection["context_text"]),
                context_record=dict(selection["context_record"]),
            )
        )
    return batches


def select_rag_context_for_question(
    *,
    documents: Sequence[AdmissionDocument],
    retriever: AdmissionRetriever,
    question: EvalQuestion,
    evaluation_variant: str,
    rag_method: str,
    model_name: str,
    tokenizer_name: str | None,
    max_model_len: int,
    max_output_tokens: int,
    safe_margin_tokens: int,
    token_estimate_safety_multiplier: float,
) -> dict[str, Any]:
    effective_budget = int(max_model_len - max_output_tokens - safe_margin_tokens)
    scored_documents = retriever.score_all(documents, query=question.question)
    selected_by_score = list(scored_documents)
    rendered_documents = order_documents_chronologically(selected_by_score)
    context_text = render_rag_context(rendered_documents)
    estimate = _estimate_rag_context(
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        context_text=context_text,
        questions=[question],
    )
    adjusted_estimate = adjusted_prompt_tokens(estimate.total_tokens, token_estimate_safety_multiplier)
    was_pruned = False

    while selected_by_score and adjusted_estimate > effective_budget:
        was_pruned = True
        selected_by_score = selected_by_score[:-1]
        rendered_documents = order_documents_chronologically(selected_by_score)
        context_text = render_rag_context(rendered_documents)
        estimate = _estimate_rag_context(
            model_name=model_name,
            tokenizer_name=tokenizer_name,
            context_text=context_text,
            questions=[question],
        )
        adjusted_estimate = adjusted_prompt_tokens(estimate.total_tokens, token_estimate_safety_multiplier)

    if adjusted_estimate > effective_budget:
        was_pruned = True
        selected_by_score = []
        rendered_documents = []
        context_text = render_rag_context([])
        estimate = _estimate_rag_context(
            model_name=model_name,
            tokenizer_name=tokenizer_name,
            context_text=context_text,
            questions=[question],
        )
        adjusted_estimate = adjusted_prompt_tokens(estimate.total_tokens, token_estimate_safety_multiplier)

    context_record = {
        "strategy": "rag",
        "evaluation_variant": str(evaluation_variant),
        "rag_method": str(rag_method),
        "retrieval_backend": retriever.backend_name,
        "rag_passthrough": False,
        "was_truncated": bool(was_pruned),
        "all_admissions_scored": True,
        "selected_admissions": len(selected_by_score),
        "scored_admissions": len(scored_documents),
        "omitted_admissions": max(0, len(scored_documents) - len(selected_by_score)),
        "selected_admission_records": [_diagnostic_record(record, index) for index, record in enumerate(selected_by_score, start=1)],
        "rendered_admission_records": [
            {
                **_diagnostic_record(record, int(record.get("selected_rank", index))),
                "rendered_order": index,
            }
            for index, record in enumerate(rendered_documents, start=1)
        ],
        "estimated_prompt_tokens": int(estimate.total_tokens),
        "adjusted_estimated_prompt_tokens": int(adjusted_estimate),
        "tokenizer": estimate.encoding_name,
        "max_model_len": int(max_model_len),
        "max_output_tokens": int(max_output_tokens),
        "safe_margin_tokens": int(safe_margin_tokens),
        "token_estimate_safety_multiplier": float(token_estimate_safety_multiplier),
        "effective_prompt_budget_tokens": int(effective_budget),
        "effective_raw_prompt_budget_tokens": effective_raw_prompt_budget(
            effective_budget,
            token_estimate_safety_multiplier,
        ),
        "prompt_context_description": RAG_CONTEXT_DESCRIPTION,
        "prompt_context_payload_key": RAG_CONTEXT_PAYLOAD_KEY,
    }
    return {
        "context_text": context_text,
        "estimated_prompt_tokens": int(estimate.total_tokens),
        "adjusted_estimated_prompt_tokens": int(adjusted_estimate),
        "context_record": context_record,
    }


def order_documents_chronologically(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked: list[tuple[datetime | None, int, dict[str, Any]]] = []
    for index, record in enumerate(records, start=1):
        selected_rank = int(record.get("selected_rank", index))
        timestamp = _first_visible_timestamp(str(record.get("text", ""))) or _first_visible_timestamp(
            str(record.get("first_timestamp", ""))
        )
        rendered_record = dict(record)
        ranked.append((timestamp, selected_rank, rendered_record))
    ranked.sort(
        key=lambda item: (
            item[0] is None,
            item[0] or datetime.max,
            item[1],
        )
    )
    return [record for _timestamp, _selected_rank, record in ranked]


def render_rag_context(records: Sequence[dict[str, Any]]) -> str:
    lines = ["=== Retrieved Patient Conversation Excerpts ==="]
    lines.append("Excerpts are shown in chronological order when timestamps are available.")
    lines.append("Use only excerpts relevant to the question; some answers may require more than one excerpt.")
    if not records:
        lines.append("None.")
        return "\n".join(lines)
    for index, record in enumerate(records, start=1):
        text = _sanitize_visible_text(str(record.get("text", "")))
        lines.append(f"Excerpt {index}:")
        lines.append(text if text else "None.")
    return "\n".join(lines)


def _estimate_rag_context(
    *,
    model_name: str,
    tokenizer_name: str | None,
    context_text: str,
    questions: Sequence[EvalQuestion],
):
    rendered = render_answer_prompt(
        context_text=context_text,
        questions=[question.model_question() for question in questions],
        context_description=RAG_CONTEXT_DESCRIPTION,
        context_payload_key=RAG_CONTEXT_PAYLOAD_KEY,
    )
    return estimate_prompt_tokens(
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        system_message=rendered.system_message,
        user_message=rendered.user_message,
    )


def _score_record(
    document: AdmissionDocument,
    retrieval_score: float,
    score: float,
    **extra_scores: float,
) -> dict[str, Any]:
    return {
        "doc_id": document.doc_id,
        "admission_index": int(document.admission_index),
        "hadm_id": document.hadm_id,
        "first_timestamp": document.first_timestamp,
        "turn_count": int(document.turn_count),
        "text": document.text,
        "score": float(score),
        "retrieval_score": float(retrieval_score),
        **{key: float(value) for key, value in extra_scores.items()},
    }


def _with_selected_ranks(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        ranked_record = dict(record)
        ranked_record["selected_rank"] = index
        ranked.append(ranked_record)
    return ranked


def _diagnostic_record(record: dict[str, Any], selected_rank: int) -> dict[str, Any]:
    payload = {
        "doc_id": str(record.get("doc_id", "")),
        "admission_index": int(record.get("admission_index", 0) or 0),
        "hadm_id": str(record.get("hadm_id", "")),
        "first_timestamp": str(record.get("first_timestamp", "")),
        "turn_count": int(record.get("turn_count", 0) or 0),
        "text": str(record.get("text", "")),
        "score": float(record.get("score", record.get("retrieval_score", 0.0))),
        "retrieval_score": float(record.get("retrieval_score", 0.0)),
        "selected_rank": int(selected_rank),
    }
    if "dense_score" in record:
        payload["dense_score"] = float(record.get("dense_score", 0.0))
    if "bm25_score" in record:
        payload["bm25_score"] = float(record.get("bm25_score", 0.0))
    return payload


def _first_visible_timestamp(value: str) -> datetime | None:
    match = _VISIBLE_TIMESTAMP_PATTERN.search(str(value or ""))
    if not match:
        return None
    date_part = match.group(1)
    time_part = match.group(2) or "00:00"
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(f"{date_part} {time_part}", fmt)
        except ValueError:
            continue
    return None


def _sanitize_visible_text(value: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", str(value or "").strip())


def _tokenize(value: str) -> list[str]:
    return [match.group(0).lower() for match in _TOKEN_PATTERN.finditer(str(value or ""))]
