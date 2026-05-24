from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field
from health_benchmark.scripts.llm_client import StructuredResponseValidationError, structured_content_candidates

from .answer_prompting import render_answer_prompt
from .config import EvaluationSettings
from .token_budget import adjusted_prompt_tokens, effective_raw_prompt_budget, estimate_prompt_tokens
from .types import EvalQuestion, ModelSpec, PromptTokenEstimate, QuestionBatch


MEM0_CONTEXT_DESCRIPTION = "patient memory context"
MEM0_CONTEXT_PAYLOAD_KEY = "patient_memory_context"
MEM0_RETRIEVAL_BACKEND = "local_dense_hf"
MEM0_SUMMARY_CHAR_LIMIT = 1200
MEM0_SUMMARY_MAX_OUTPUT_TOKENS = 1024
_VISIBLE_MEMORY_TIMESTAMP_PATTERN = re.compile(
    r"\b(\d{4}-\d{2}-\d{2})(?:[ T](\d{2}:\d{2}(?::\d{2})?))?\b"
)


class Mem0CandidateMemory(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    memory: str


class Mem0ExtractionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memories: list[Mem0CandidateMemory]


class MemAlphaExtractionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    core: list[Mem0CandidateMemory] = Field(default_factory=list)
    episodic: list[Mem0CandidateMemory] = Field(default_factory=list)
    semantic: list[Mem0CandidateMemory] = Field(default_factory=list)


class Mem0UpdateAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    operation: str
    target_memory_id: str | None = None
    memory: str | None = None


class Mem0UpdateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    actions: list[Mem0UpdateAction]


class Mem0SummaryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str


class Mem0EmbeddingError(RuntimeError):
    """Raised when the dense memory embedding backend cannot be initialized."""


class Mem0EmbeddingModel(Protocol):
    model_name: str
    device: str
    batch_size: int
    max_length: int
    embedding_dimension: int | None

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        ...


class LocalHFDenseEmbedder:
    def __init__(
        self,
        *,
        model_name: str,
        device: str,
        gpu_device_ids: str,
        batch_size: int,
        max_length: int,
    ) -> None:
        self.model_name = str(model_name)
        self.requested_device = str(device)
        self.gpu_device_ids = str(gpu_device_ids or "").strip()
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.embedding_dimension: int | None = None
        self.device = self._resolve_device()
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except Exception as exc:  # pragma: no cover - exercised in environments without torch.
            raise Mem0EmbeddingError(
                "Dense memory retrieval requires torch and transformers. "
                "Install the embedding dependencies or run on an environment with the local HF embedding stack."
            ) from exc

        self._torch = torch
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")
            dtype = torch.float16 if str(self.device).startswith("cuda") else None
            model_kwargs: dict[str, Any] = {}
            if dtype is not None:
                model_kwargs["torch_dtype"] = dtype
            self.model = AutoModel.from_pretrained(self.model_name, **model_kwargs)
            self.model.to(self.device)
            self.model.eval()
            self.embedding_dimension = int(getattr(self.model.config, "hidden_size", 0)) or None
        except Exception as exc:  # pragma: no cover - depends on runtime model cache.
            raise Mem0EmbeddingError(
                f"Failed to initialize dense memory embedding model {self.model_name!r} on {self.device}: {exc}"
            ) from exc

    def _resolve_device(self) -> str:
        requested = str(self.requested_device or "cuda").strip()
        if self.gpu_device_ids and requested.startswith("cuda"):
            os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_device_ids
            return "cuda:0"
        return requested

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        cleaned = [str(text or "").strip() for text in texts]
        if not cleaned:
            return []
        embeddings: list[list[float]] = []
        torch = self._torch
        for start in range(0, len(cleaned), self.batch_size):
            batch_texts = cleaned[start : start + self.batch_size]
            tokenized = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            tokenized = {key: value.to(self.device) for key, value in tokenized.items()}
            with torch.inference_mode():
                outputs = self.model(**tokenized)
                hidden = outputs.last_hidden_state
                attention_mask = tokenized["attention_mask"]
                if bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item()):
                    pooled = hidden[:, -1]
                else:
                    sequence_lengths = attention_mask.sum(dim=1) - 1
                    row_indices = torch.arange(hidden.shape[0], device=hidden.device)
                    pooled = hidden[row_indices, sequence_lengths]
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            batch_embeddings = pooled.detach().cpu().float().tolist()
            embeddings.extend([_normalize_dense_vector(vector) for vector in batch_embeddings])
        return embeddings


class DenseMemoryRetriever:
    backend_name = MEM0_RETRIEVAL_BACKEND

    def __init__(self, embedder: Mem0EmbeddingModel) -> None:
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
        for _memory_id, (_memory, embedding) in self._cache.items():
            if embedding:
                return len(embedding)
        return None

    def ensure_index(self, records: Sequence[dict[str, Any]]) -> None:
        missing_records: list[dict[str, Any]] = []
        missing_texts: list[str] = []
        for record in records:
            memory_id = str(record.get("memory_id", ""))
            memory = str(record.get("memory", ""))
            if not memory_id or not memory:
                continue
            cached = self._cache.get(memory_id)
            if cached is None or cached[0] != memory:
                missing_records.append(record)
                missing_texts.append(memory)
        if not missing_records:
            return
        embeddings = self.embedder.embed(missing_texts)
        if len(embeddings) != len(missing_records):
            raise Mem0EmbeddingError(
                f"Dense embedder returned {len(embeddings)} embeddings for {len(missing_records)} memories."
            )
        for record, embedding in zip(missing_records, embeddings):
            self._cache[str(record["memory_id"])] = (
                str(record.get("memory", "")),
                _normalize_dense_vector(embedding),
            )

    def search(
        self,
        records: Sequence[dict[str, Any]],
        *,
        query: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        if int(top_k) <= 0 or not str(query or "").strip():
            return []
        active_records = [
            record
            for record in records
            if str(record.get("memory_id", "")) and str(record.get("memory", ""))
        ]
        if not active_records:
            return []
        self.ensure_index(active_records)
        query_embeddings = self.embedder.embed([str(query)])
        if not query_embeddings:
            return []
        query_embedding = _normalize_dense_vector(query_embeddings[0])
        scored: list[tuple[float, dict[str, Any]]] = []
        for record in active_records:
            cached = self._cache.get(str(record["memory_id"]))
            if cached is None:
                continue
            score = _dense_dot(query_embedding, cached[1])
            scored.append((score, record))
        scored.sort(
            key=lambda item: (
                -item[0],
                str(item[1].get("memory_id", "")),
            )
        )
        return [
            {
                **record,
                "dense_score": float(score),
                "retrieval_score": float(score),
            }
            for score, record in scored[: int(top_k)]
        ]

    def config_payload(self) -> dict[str, Any]:
        return {
            "retrieval_backend": self.backend_name,
            "embedding_model": self.embedding_model,
            "embedding_device": self.embedding_device,
            "embedding_batch_size": self.embedding_batch_size,
            "embedding_max_length": self.embedding_max_length,
            "embedding_dimension": self.embedding_dimension,
        }


def build_mem0_retriever(settings: EvaluationSettings) -> DenseMemoryRetriever:
    if settings.mem0_retrieval_backend != MEM0_RETRIEVAL_BACKEND:
        raise ValueError(
            f"Memory evaluation supports dense retrieval only; got {settings.mem0_retrieval_backend!r}."
        )
    embedder = LocalHFDenseEmbedder(
        model_name=settings.mem0_embedding_model,
        device=settings.mem0_embedding_device,
        gpu_device_ids=settings.mem0_embedding_gpu_device_ids,
        batch_size=settings.mem0_embedding_batch_size,
        max_length=settings.mem0_embedding_max_length,
    )
    return DenseMemoryRetriever(embedder)


@dataclass(frozen=True)
class ConversationTurn:
    global_index: int
    admission_index: int
    hadm_id: str
    admission_start: str
    admission_end: str
    turn_number: str
    time: str
    speaker: str
    text: str

    @property
    def turn_id(self) -> str:
        return f"hadm={self.hadm_id}:turn={self.turn_number}:global={self.global_index}"

    def render_visible(self) -> str:
        return f"{self.time} | {self.speaker} | {self.text}"

    def to_record(self) -> dict[str, Any]:
        return {
            "global_index": int(self.global_index),
            "admission_index": int(self.admission_index),
            "hadm_id": self.hadm_id,
            "admission_start": self.admission_start,
            "admission_end": self.admission_end,
            "turn_number": self.turn_number,
            "time": self.time,
            "speaker": self.speaker,
            "text": self.text,
            "turn_id": self.turn_id,
        }


@dataclass(frozen=True)
class AdmissionChunk:
    chunk_id: str
    admission_index: int
    chunk_index: int
    chunk_count_for_admission: int
    hadm_id: str
    admission_start: str
    admission_end: str
    turns: list[ConversationTurn]

    @property
    def visible_label(self) -> str:
        label = "Current chunk"
        if self.chunk_count_for_admission > 1:
            label += f" part {self.chunk_index}"
        return label

    def render_visible(self) -> str:
        return "\n".join(turn.render_visible() for turn in self.turns)

    def to_record(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "admission_index": int(self.admission_index),
            "chunk_index": int(self.chunk_index),
            "chunk_count_for_admission": int(self.chunk_count_for_admission),
            "turn_count": len(self.turns),
        }


@dataclass
class Mem0MemoryStore:
    subject_id: str
    summary: str
    records: list[dict[str, Any]]
    evaluation_variant: str = "mem0"
    embedding_backend: str = MEM0_RETRIEVAL_BACKEND
    embedding_model: str = ""
    embedding_device: str = ""
    embedding_batch_size: int = 0
    embedding_max_length: int = 0
    retriever: DenseMemoryRetriever | None = None
    next_memory_index: int = 1

    def active_records(self) -> list[dict[str, Any]]:
        return [record for record in self.records if not bool(record.get("deleted", False))]

    def search(self, query: str, *, top_k: int) -> list[dict[str, Any]]:
        if self.retriever is None:
            raise Mem0EmbeddingError(
                "Memory store has no dense retriever. Memory evaluation requires dense retrieval."
            )
        return self.retriever.search(self.active_records(), query=query, top_k=top_k)

    def ensure_index(self) -> None:
        if self.retriever is None:
            raise Mem0EmbeddingError(
                "Memory store has no dense retriever. Memory evaluation requires dense retrieval."
            )
        self.retriever.ensure_index(self.active_records())

    def embedding_dimension(self) -> int | None:
        if self.retriever is None:
            return None
        return self.retriever.embedding_dimension

    def to_payload(self, *, settings: EvaluationSettings, metrics: dict[str, Any]) -> dict[str, Any]:
        return {
            "mode": self.evaluation_variant,
            "enabled": True,
            "evaluation_variant": self.evaluation_variant,
            "subject_id": self.subject_id,
            "summary": self.summary,
            "embedding_backend": self.embedding_backend,
            "embedding_model": self.embedding_model,
            "embedding_device": self.embedding_device,
            "embedding_batch_size": int(self.embedding_batch_size),
            "embedding_max_length": int(self.embedding_max_length),
            "embedding_dimension": self.embedding_dimension(),
            "settings": build_mem0_settings_payload(settings),
            "metrics": metrics,
            "memories": [
                _stored_memory_payload(record, evaluation_variant=self.evaluation_variant)
                for record in self.active_records()
            ],
        }


@dataclass(frozen=True)
class Mem0BuildResult:
    store: Mem0MemoryStore
    store_payload: dict[str, Any]
    event_records: list[dict[str, Any]]
    metrics: dict[str, Any]


@dataclass(frozen=True)
class _MemoryLLMResult:
    parsed: BaseModel
    raw_response: dict[str, Any]
    usage: dict[str, Any]
    response_id: str | None
    latency_ms: int


def build_mem0_settings_payload(settings: EvaluationSettings) -> dict[str, Any]:
    return {
        "memory_method": settings.evaluation_variant,
        "chunk_token_cap": int(settings.mem0_chunk_token_cap),
        "previous_chunk_summaries": int(settings.mem0_previous_chunk_summaries),
        "max_candidate_memories": int(settings.mem0_max_candidate_memories),
        "similar_memories_per_candidate": int(settings.mem0_similar_memories),
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


def _stored_memory_payload(record: dict[str, Any], *, evaluation_variant: str) -> dict[str, Any]:
    payload = {
        "memory_id": str(record.get("memory_id", "")),
        "memory": str(record.get("memory", "")),
    }
    if evaluation_variant == "memalpha":
        payload["memory_type"] = str(record.get("memory_type") or "semantic")
    return payload


def build_mem0_memory_store(
    llm_client: Any,
    *,
    combined_payload: dict[str, Any],
    settings: EvaluationSettings,
    model_name: str,
    retriever: DenseMemoryRetriever | None = None,
) -> Mem0BuildResult:
    chunks = build_admission_chunks(
        combined_payload,
        model_name=model_name,
        tokenizer_name=settings.tokenizer_name,
        chunk_token_cap=settings.mem0_chunk_token_cap,
    )
    subject_id = str(combined_payload.get("subject_id") or "")
    store = Mem0MemoryStore(
        subject_id=subject_id,
        summary=render_initial_summary(combined_payload),
        records=[],
        evaluation_variant=settings.evaluation_variant,
        retriever=retriever or build_mem0_retriever(settings),
    )
    if store.retriever is not None:
        retriever_payload = store.retriever.config_payload()
        store.embedding_backend = str(retriever_payload.get("retrieval_backend") or MEM0_RETRIEVAL_BACKEND)
        store.embedding_model = str(retriever_payload.get("embedding_model") or "")
        store.embedding_device = str(retriever_payload.get("embedding_device") or "")
        store.embedding_batch_size = int(retriever_payload.get("embedding_batch_size") or 0)
        store.embedding_max_length = int(retriever_payload.get("embedding_max_length") or 0)
    events: list[dict[str, Any]] = []
    chunk_summaries: list[str] = []
    metrics: dict[str, Any] = {
        "enabled": True,
        "memory_method": settings.evaluation_variant,
        "construction_strategy": (
            "admission_chunks" if settings.evaluation_variant == "mem0" else "add_only_admission_chunks"
        ),
        "turn_count": sum(len(chunk.turns) for chunk in chunks),
        "admission_count": len(combined_payload.get("admissions", [])),
        "chunk_count": len(chunks),
        "total_memories": 0,
        "active_memories": 0,
        "deleted_memories": 0,
        "extraction_call_count": 0,
        "update_call_count": 0,
        "summary_call_count": 0,
        "summary_error_count": 0,
        "add_count": 0,
        "update_count": 0,
        "delete_count": 0,
        "noop_count": 0,
        "fallback_add_count": 0,
        "duplicate_skip_count": 0,
        "candidate_count": 0,
        "max_candidate_memories_per_chunk": int(settings.mem0_max_candidate_memories),
        "similar_memories_per_candidate": int(settings.mem0_similar_memories),
        "max_update_memories": int(settings.mem0_max_update_memories),
        "embedding_backend": store.embedding_backend,
        "embedding_model": store.embedding_model,
        "embedding_device": store.embedding_device,
        "embedding_batch_size": store.embedding_batch_size,
        "embedding_max_length": store.embedding_max_length,
    }

    for chunk in chunks:
        previous_summaries = chunk_summaries[-settings.mem0_previous_chunk_summaries :]
        existing_for_update: list[dict[str, Any]] = []
        if settings.evaluation_variant == "mem0":
            extraction_result = _extract_candidate_memories(
                llm_client,
                summary=store.summary,
                previous_chunk_summaries=previous_summaries,
                chunk=chunk,
                settings=settings,
                model_name=model_name,
            )
        else:
            existing_for_update = retrieve_existing_for_chunk(
                store,
                chunk,
                top_k=settings.mem0_max_update_memories,
            )
            extraction_result = _extract_add_only_memories(
                llm_client,
                summary=store.summary,
                previous_chunk_summaries=previous_summaries,
                chunk=chunk,
                existing_memories=existing_for_update,
                settings=settings,
                model_name=model_name,
            )
        metrics["extraction_call_count"] += 1
        candidates = _normalize_extracted_candidates(
            extraction_result.parsed,
            evaluation_variant=settings.evaluation_variant,
            limit=settings.mem0_max_candidate_memories,
        )
        metrics["candidate_count"] += len(candidates)
        events.append(
            _event_record(
                event_type="extract",
                chunk=chunk,
                llm_result=extraction_result,
                settings=settings,
                extra={
                    "candidate_count": len(candidates),
                    "raw_candidate_count": _count_extracted_candidates(extraction_result.parsed),
                    "candidate_ids": [candidate["candidate_id"] for candidate in candidates],
                    "existing_memory_ids": [record["memory_id"] for record in existing_for_update],
                },
            )
        )

        update_result: _MemoryLLMResult | None = None
        applied: list[dict[str, Any]] = []
        if settings.evaluation_variant == "mem0" and candidates:
            existing_for_update = retrieve_existing_for_candidates(
                store,
                candidates,
                similar_per_candidate=settings.mem0_similar_memories,
                max_existing=settings.mem0_max_update_memories,
            )
            update_result = _choose_memory_updates(
                llm_client,
                candidates=candidates,
                existing_memories=existing_for_update,
                settings=settings,
                model_name=model_name,
            )
            metrics["update_call_count"] += 1
            applied = _apply_update_actions(
                store,
                candidates=candidates,
                actions=update_result.parsed.actions,
                chunk=chunk,
            )
            for operation_record in applied:
                operation_key = f"{operation_record['operation'].lower()}_count"
                if operation_key in metrics:
                    metrics[operation_key] += 1
                if operation_record.get("used_fallback"):
                    metrics["fallback_add_count"] += 1
            events.append(
                _event_record(
                    event_type="update",
                    chunk=chunk,
                    llm_result=update_result,
                    settings=settings,
                    extra={
                        "candidate_ids": [candidate["candidate_id"] for candidate in candidates],
                        "existing_memory_ids": [record["memory_id"] for record in existing_for_update],
                        "existing_memory_count": len(existing_for_update),
                        "applied_operations": applied,
                    },
                )
            )
        elif candidates:
            applied = _add_only_memory_candidates(store, candidates=candidates, chunk=chunk)
            metrics["add_count"] += sum(1 for item in applied if item["operation"] == "ADD")
            metrics["duplicate_skip_count"] += sum(
                1 for item in applied if item["operation"] == "SKIP_DUPLICATE"
            )
            events.append(
                _event_record(
                    event_type="add",
                    chunk=chunk,
                    llm_result=extraction_result,
                    settings=settings,
                    extra={
                        "candidate_ids": [candidate["candidate_id"] for candidate in candidates],
                        "existing_memory_ids": [record["memory_id"] for record in existing_for_update],
                        "existing_memory_count": len(existing_for_update),
                        "applied_operations": applied,
                    },
                )
            )

        try:
            summary_result = _refresh_summary(
                llm_client,
                current_summary=store.summary,
                previous_chunk_summaries=previous_summaries,
                chunk=chunk,
                applied_operations=applied,
                settings=settings,
                model_name=model_name,
            )
            store.summary = _clamp_summary(summary_result.parsed.summary.strip() or store.summary)
            chunk_summaries.append(store.summary)
            metrics["summary_call_count"] += 1
            events.append(
                _event_record(
                    event_type="summary",
                    chunk=chunk,
                    llm_result=summary_result,
                    settings=settings,
                    extra={
                        "active_memory_count": len(store.active_records()),
                        "summary_char_limit": MEM0_SUMMARY_CHAR_LIMIT,
                    },
                )
            )
        except Exception as exc:
            metrics["summary_error_count"] += 1
            chunk_summaries.append(store.summary)
            events.append(
                _error_event_record(
                    event_type="summary_error",
                    chunk=chunk,
                    error=exc,
                    extra={
                        "active_memory_count": len(store.active_records()),
                        "kept_previous_summary": True,
                        "summary_char_limit": MEM0_SUMMARY_CHAR_LIMIT,
                    },
                )
            )

    metrics["total_memories"] = len(store.records)
    metrics["active_memories"] = len(store.active_records())
    metrics["deleted_memories"] = int(metrics.get("delete_count", 0))
    store.ensure_index()
    metrics["embedding_dimension"] = store.embedding_dimension()
    store_payload = store.to_payload(settings=settings, metrics=metrics)
    return Mem0BuildResult(store=store, store_payload=store_payload, event_records=events, metrics=metrics)


def build_memory_question_batches(
    questions: list[EvalQuestion],
    *,
    memory_store: Mem0MemoryStore,
    settings: EvaluationSettings,
    model_spec: ModelSpec,
) -> list[QuestionBatch]:
    batches: list[QuestionBatch] = []
    for index, question in enumerate(questions, start=1):
        selection = select_mem0_context_for_question(
            memory_store=memory_store,
            question=question,
            model_name=model_spec.model_name,
            tokenizer_name=settings.tokenizer_name,
            max_model_len=model_spec.max_model_len,
            max_output_tokens=settings.max_output_tokens,
            safe_margin_tokens=settings.safe_margin_tokens,
            token_estimate_safety_multiplier=settings.token_estimate_safety_multiplier,
            retrieval_top_k=settings.mem0_answer_retrieval_top_k,
            max_answer_memories=settings.mem0_max_answer_memories,
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


def select_mem0_context_for_question(
    *,
    memory_store: Mem0MemoryStore,
    question: EvalQuestion,
    model_name: str,
    tokenizer_name: str | None,
    max_model_len: int,
    max_output_tokens: int,
    safe_margin_tokens: int,
    token_estimate_safety_multiplier: float,
    retrieval_top_k: int,
    max_answer_memories: int,
) -> dict[str, Any]:
    effective_budget = int(max_model_len - max_output_tokens - safe_margin_tokens)
    retrieved = memory_store.search(question.question, top_k=retrieval_top_k)
    selected_memories = retrieved[:max_answer_memories]
    rendered_memories = order_memories_for_answer_context(selected_memories)
    context_text = render_mem0_context(memory_store.summary, rendered_memories)
    estimate = _estimate_mem0_context(
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        context_text=context_text,
        questions=[question],
    )
    adjusted_estimate = adjusted_prompt_tokens(estimate.total_tokens, token_estimate_safety_multiplier)
    was_pruned = False

    while selected_memories and adjusted_estimate > effective_budget:
        was_pruned = True
        selected_memories = selected_memories[:-1]
        rendered_memories = order_memories_for_answer_context(selected_memories)
        context_text = render_mem0_context(memory_store.summary, rendered_memories)
        estimate = _estimate_mem0_context(
            model_name=model_name,
            tokenizer_name=tokenizer_name,
            context_text=context_text,
            questions=[question],
        )
        adjusted_estimate = adjusted_prompt_tokens(estimate.total_tokens, token_estimate_safety_multiplier)

    if adjusted_estimate > effective_budget:
        was_pruned = True
        rendered_memories = []
        selected_memories = []
        context_text = render_mem0_context("", [])
        estimate = _estimate_mem0_context(
            model_name=model_name,
            tokenizer_name=tokenizer_name,
            context_text=context_text,
            questions=[question],
        )
        adjusted_estimate = adjusted_prompt_tokens(estimate.total_tokens, token_estimate_safety_multiplier)

    context_record = {
        "strategy": memory_store.evaluation_variant,
        "evaluation_variant": memory_store.evaluation_variant,
        "was_truncated": bool(was_pruned),
        "selected_memories": len(selected_memories),
        "retrieved_memories": len(retrieved),
        "omitted_memories": max(0, len(retrieved) - len(selected_memories)),
        "total_active_memories": len(memory_store.active_records()),
        "retrieval_top_k": int(retrieval_top_k),
        "max_answer_memories": int(max_answer_memories),
        "retrieved_memory_ids": [record["memory_id"] for record in selected_memories],
        "selected_memory_records": [
            {
                "memory_id": str(record.get("memory_id", "")),
                **(
                    {"memory_type": str(record.get("memory_type"))}
                    if record.get("memory_type") is not None
                    else {}
                ),
                "memory": str(record.get("memory", "")),
                "dense_score": float(record.get("dense_score", record.get("retrieval_score", 0.0))),
                "retrieval_score": float(record.get("retrieval_score", 0.0)),
                "selected_rank": index,
            }
            for index, record in enumerate(selected_memories, start=1)
        ],
        "rendered_memory_records": [
            {
                "memory_id": str(record.get("memory_id", "")),
                **(
                    {"memory_type": str(record.get("memory_type"))}
                    if record.get("memory_type") is not None
                    else {}
                ),
                "memory": str(record.get("memory", "")),
                "dense_score": float(record.get("dense_score", record.get("retrieval_score", 0.0))),
                "retrieval_score": float(record.get("retrieval_score", 0.0)),
                "selected_rank": int(record.get("_selected_rank", index)),
                "rendered_order": index,
            }
            for index, record in enumerate(rendered_memories, start=1)
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
        "prompt_context_description": MEM0_CONTEXT_DESCRIPTION,
        "prompt_context_payload_key": MEM0_CONTEXT_PAYLOAD_KEY,
    }
    return {
        "context_text": context_text,
        "estimated_prompt_tokens": int(estimate.total_tokens),
        "adjusted_estimated_prompt_tokens": int(adjusted_estimate),
        "context_record": context_record,
    }


def build_admission_chunks(
    combined_payload: dict[str, Any],
    *,
    model_name: str,
    tokenizer_name: str | None,
    chunk_token_cap: int,
) -> list[AdmissionChunk]:
    turns_by_admission: list[list[ConversationTurn]] = []
    global_index = 0
    for admission_index, admission in enumerate(combined_payload.get("admissions", []), start=1):
        admission_turns: list[ConversationTurn] = []
        for line in admission.get("conversation_lines", []):
            global_index += 1
            admission_turns.append(
                ConversationTurn(
                    global_index=global_index,
                    admission_index=admission_index,
                    hadm_id=str(admission.get("hadm_id") or ""),
                    admission_start=str(admission.get("admission_start") or ""),
                    admission_end=str(admission.get("admission_end") or ""),
                    turn_number=str(line.get("turn_number") or ""),
                    time=str(line.get("time") or ""),
                    speaker=str(line.get("speaker") or ""),
                    text=str(line.get("text") or ""),
                )
            )
        turns_by_admission.append(admission_turns)

    chunks: list[AdmissionChunk] = []
    for admission_index, turns in enumerate(turns_by_admission, start=1):
        if not turns:
            continue
        split_turn_groups = _split_turns_by_token_cap(
            turns,
            model_name=model_name,
            tokenizer_name=tokenizer_name,
            chunk_token_cap=chunk_token_cap,
        )
        chunk_count = len(split_turn_groups)
        for chunk_index, chunk_turns in enumerate(split_turn_groups, start=1):
            first_turn = chunk_turns[0]
            chunks.append(
                AdmissionChunk(
                    chunk_id=f"adm_{admission_index:03d}_chunk_{chunk_index:03d}",
                    admission_index=admission_index,
                    chunk_index=chunk_index,
                    chunk_count_for_admission=chunk_count,
                    hadm_id=first_turn.hadm_id,
                    admission_start=first_turn.admission_start,
                    admission_end=first_turn.admission_end,
                    turns=chunk_turns,
                )
            )
    return chunks


def render_initial_summary(combined_payload: dict[str, Any]) -> str:
    admissions = combined_payload.get("admissions", [])
    lines = [f"Patient has {len(admissions)} admissions in chronological order."]
    for index, admission in enumerate(admissions, start=1):
        lines.append(f"Chunk {index}: turns={len(admission.get('conversation_lines', []))}.")
    return "\n".join(lines)


def order_memories_for_answer_context(memories: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked_records: list[tuple[datetime | None, int, dict[str, Any]]] = []
    for selected_rank, record in enumerate(memories, start=1):
        rendered_record = dict(record)
        rendered_record["_selected_rank"] = selected_rank
        memory_timestamp = _first_visible_memory_timestamp(str(record.get("memory", "")))
        ranked_records.append((memory_timestamp, selected_rank, rendered_record))
    ranked_records.sort(
        key=lambda item: (
            item[0] is None,
            item[0] or datetime.max,
            item[1],
        )
    )
    return [record for _timestamp, _selected_rank, record in ranked_records]


def render_mem0_context(summary: str, memories: Sequence[dict[str, Any]]) -> str:
    del summary
    lines = ["=== Patient Memory ==="]
    lines.append("Retrieved memories are shown in chronological order when timestamps are available.")
    lines.append("Use only memories relevant to the question; some answers may require more than one memory.")
    lines.append("Retrieved memories:")
    if not memories:
        lines.append("None.")
    if any(str(record.get("memory_type") or "").strip() for record in memories):
        grouped: dict[str, list[dict[str, Any]]] = {"core": [], "episodic": [], "semantic": []}
        for record in memories:
            memory_type = str(record.get("memory_type") or "semantic").strip().lower()
            if memory_type not in grouped:
                memory_type = "semantic"
            grouped[memory_type].append(record)
        rendered_any = False
        for memory_type, label in (
            ("core", "Core memory"),
            ("episodic", "Episodic memory"),
            ("semantic", "Semantic memory"),
        ):
            records = grouped[memory_type]
            if not records:
                continue
            rendered_any = True
            lines.append(f"{label}:")
            ordered_records = order_memories_for_answer_context(records) if memory_type == "episodic" else records
            for index, record in enumerate(ordered_records, start=1):
                memory = _sanitize_visible_text(str(record.get("memory", "")))
                lines.append(f"{index}. {memory}")
        if not rendered_any:
            lines.append("None.")
    else:
        for index, record in enumerate(memories, start=1):
            memory = _sanitize_visible_text(str(record.get("memory", "")))
            lines.append(f"{index}. {memory}")
    return "\n".join(lines)


def retrieve_existing_for_candidates(
    store: Mem0MemoryStore,
    candidates: Sequence[dict[str, Any]],
    *,
    similar_per_candidate: int,
    max_existing: int,
) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        for result in store.search(str(candidate.get("memory", "")), top_k=similar_per_candidate):
            memory_id = str(result["memory_id"])
            previous = by_id.get(memory_id)
            if previous is None or float(result.get("retrieval_score", 0.0)) > float(previous.get("retrieval_score", 0.0)):
                by_id[memory_id] = result
    selected = sorted(
        by_id.values(),
        key=lambda record: (
            -float(record.get("retrieval_score", 0.0)),
            str(record.get("memory_id", "")),
        ),
    )
    return selected[: int(max_existing)]


def retrieve_existing_for_chunk(
    store: Mem0MemoryStore,
    chunk: AdmissionChunk,
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    return store.search(chunk.render_visible(), top_k=top_k) if store.active_records() else []


def _extract_add_only_memories(
    llm_client: Any,
    *,
    summary: str,
    previous_chunk_summaries: Sequence[str],
    chunk: AdmissionChunk,
    existing_memories: Sequence[dict[str, Any]],
    settings: EvaluationSettings,
    model_name: str,
) -> _MemoryLLMResult:
    if settings.evaluation_variant == "memalpha":
        return _extract_memalpha_candidate_memories(
            llm_client,
            summary=summary,
            previous_chunk_summaries=previous_chunk_summaries,
            chunk=chunk,
            existing_memories=existing_memories,
            settings=settings,
            model_name=model_name,
        )
    return _extract_mem0_star_candidate_memories(
        llm_client,
        summary=summary,
        previous_chunk_summaries=previous_chunk_summaries,
        chunk=chunk,
        existing_memories=existing_memories,
        settings=settings,
        model_name=model_name,
    )


def _extract_candidate_memories(
    llm_client: Any,
    *,
    summary: str,
    previous_chunk_summaries: Sequence[str],
    chunk: AdmissionChunk,
    settings: EvaluationSettings,
    model_name: str,
) -> _MemoryLLMResult:
    system_message = "\n".join(
        [
            "You are the extraction phase of a long-term patient memory module for a healthcare benchmark.",
            "Extract concise, atomic memories from only the current admission chunk.",
            "Use the patient memory summary and previous chunk summaries only to interpret context.",
            "Each memory must be standalone and include explicit temporal context from the visible conversation-line timestamps, such as 'On 2172-02-24 13:40, ...' or 'Between 2172-02-24 and 2172-02-26, ...'.",
            "Do not mention internal database identifiers, admission IDs, turn numbers, turn IDs, or global turn indices.",
            "Capture clinically relevant patient facts, events, decisions, and temporal changes.",
            "Preserve exact clinically meaningful wording from the conversation lines when concise, including test names, abbreviations, diagnoses, symptoms, procedures, medication names, and stated reasons.",
            "Do not over-paraphrase specific terms into broad categories; keep source phrases such as V/Q scan, PE, VAC, shortness of breath, new drainage, and INR when present.",
            "Avoid exact duplicates, unsupported inferences, small talk, and generic conversation mechanics.",
            'Return strict JSON with the schema {"memories": [{"candidate_id": "c001", "memory": "..."}]}.',
        ]
    )
    user_message = json.dumps(
        {
            "patient_memory_summary": _sanitize_visible_text(summary),
            "previous_chunk_summaries": [_sanitize_visible_text(item) for item in previous_chunk_summaries],
            "current_chunk": {
                "conversation": chunk.render_visible(),
            },
        },
        ensure_ascii=False,
        indent=2,
    )
    return _generate_memory_response(
        llm_client,
        system_message=system_message,
        user_message=user_message,
        response_schema=Mem0ExtractionResponse,
        max_output_tokens=settings.mem0_max_output_tokens,
        retry_limit=settings.retry_limit,
        model_name=model_name,
    )


def _extract_mem0_star_candidate_memories(
    llm_client: Any,
    *,
    summary: str,
    previous_chunk_summaries: Sequence[str],
    chunk: AdmissionChunk,
    existing_memories: Sequence[dict[str, Any]],
    settings: EvaluationSettings,
    model_name: str,
) -> _MemoryLLMResult:
    system_message = "\n".join(
        [
            "You are the extraction phase of an add-only long-term patient memory module for a healthcare benchmark.",
            "Extract concise, atomic memories from only the current conversation chunk.",
            "Use the patient memory summary, previous chunk summaries, and related existing memories only to interpret context and avoid exact duplicates.",
            "Each memory must be standalone and include explicit temporal context from the visible conversation-line timestamps, such as 'On 2172-02-24 13:40, ...' or 'Between 2172-02-24 and 2172-02-26, ...'.",
            "Do not mention internal database identifiers, admission IDs, turn numbers, turn IDs, or global turn indices.",
            "Capture clinically relevant patient facts, events, decisions, and temporal changes.",
            "Preserve exact clinically meaningful wording from the conversation lines when concise, including abbreviations, test names, medication names, procedure names, diagnosis names, and stated reasons.",
            "Do not over-paraphrase specific terms into broad categories.",
            "Avoid exact duplicates, unsupported inferences, small talk, and generic conversation mechanics.",
            "Only add new memories; do not update, delete, or return no-op decisions.",
            'Return strict JSON with the schema {"memories": [{"candidate_id": "c001", "memory": "..."}]}.',
        ]
    )
    user_message = json.dumps(
        {
            "patient_memory_summary": _sanitize_visible_text(summary),
            "previous_chunk_summaries": [_sanitize_visible_text(item) for item in previous_chunk_summaries],
            "related_existing_memories": [
                {"memory": _sanitize_visible_text(str(record.get("memory", "")))}
                for record in existing_memories
            ],
            "current_chunk": {
                "conversation": chunk.render_visible(),
            },
        },
        ensure_ascii=False,
        indent=2,
    )
    return _generate_memory_response(
        llm_client,
        system_message=system_message,
        user_message=user_message,
        response_schema=Mem0ExtractionResponse,
        max_output_tokens=settings.mem0_max_output_tokens,
        retry_limit=settings.retry_limit,
        model_name=model_name,
    )


def _extract_memalpha_candidate_memories(
    llm_client: Any,
    *,
    summary: str,
    previous_chunk_summaries: Sequence[str],
    chunk: AdmissionChunk,
    existing_memories: Sequence[dict[str, Any]],
    settings: EvaluationSettings,
    model_name: str,
) -> _MemoryLLMResult:
    system_message = "\n".join(
        [
            "You are the extraction phase of an add-only structured patient memory module for a healthcare benchmark.",
            "Extract memories from only the current conversation chunk into three groups: core, episodic, and semantic.",
            "Core memories are stable patient facts or durable clinical context.",
            "Episodic memories are timestamp-grounded clinical events, decisions, or changes.",
            "Semantic memories are reusable patient-specific clinical facts or relationships that are not best represented as one event.",
            "Use the patient memory summary, previous chunk summaries, and related existing memories only to interpret context and avoid exact duplicates.",
            "Each memory must be standalone and include explicit temporal context from the visible conversation-line timestamps when time is relevant.",
            "Do not mention internal database identifiers, admission IDs, turn numbers, turn IDs, or global turn indices.",
            "Capture clinically relevant patient facts, events, decisions, relationships, and temporal changes.",
            "Preserve exact clinically meaningful wording from the conversation lines when concise.",
            "Avoid exact duplicates, unsupported inferences, small talk, and generic conversation mechanics.",
            "Only add new memories; do not update, delete, or return no-op decisions.",
            'Return strict JSON with the schema {"core": [{"candidate_id": "c001", "memory": "..."}], "episodic": [{"candidate_id": "e001", "memory": "..."}], "semantic": [{"candidate_id": "s001", "memory": "..."}]}.',
        ]
    )
    user_message = json.dumps(
        {
            "patient_memory_summary": _sanitize_visible_text(summary),
            "previous_chunk_summaries": [_sanitize_visible_text(item) for item in previous_chunk_summaries],
            "related_existing_memories": [
                {
                    "memory_type": str(record.get("memory_type") or "semantic"),
                    "memory": _sanitize_visible_text(str(record.get("memory", ""))),
                }
                for record in existing_memories
            ],
            "current_chunk": {
                "conversation": chunk.render_visible(),
            },
        },
        ensure_ascii=False,
        indent=2,
    )
    return _generate_memory_response(
        llm_client,
        system_message=system_message,
        user_message=user_message,
        response_schema=MemAlphaExtractionResponse,
        max_output_tokens=settings.mem0_max_output_tokens,
        retry_limit=settings.retry_limit,
        model_name=model_name,
    )


def _choose_memory_updates(
    llm_client: Any,
    *,
    candidates: Sequence[dict[str, Any]],
    existing_memories: Sequence[dict[str, Any]],
    settings: EvaluationSettings,
    model_name: str,
) -> _MemoryLLMResult:
    system_message = "\n".join(
        [
            "You are the update phase of a long-term patient memory module.",
            "Choose how each candidate memory should change the memory store.",
            "Operations: ADD creates a new memory; UPDATE merges complementary information into one existing memory; DELETE removes a directly contradicted memory; NOOP keeps the store unchanged for duplicates or irrelevant candidates.",
            "In healthcare timelines, preserve time-specific changes. Do not delete an older fact just because a newer value or symptom status is different at a later time.",
            "Do not mention internal database identifiers, admission IDs, turn numbers, turn IDs, or global turn indices in memory text.",
            "For ADD or UPDATE, provide the final standalone memory text in the memory field, preserving explicit temporal context from conversation-line timestamps.",
            "Preserve exact candidate wording for clinically specific terms unless merging truly requires shortening.",
            "Do not generalize specific test names, symptoms, procedures, medications, diagnoses, or reasons into broader categories.",
            "For UPDATE or DELETE, target_memory_id must match an existing memory_id.",
            'Return strict JSON with the schema {"actions": [{"candidate_id": "c001", "operation": "ADD|UPDATE|DELETE|NOOP", "target_memory_id": null, "memory": "..."}]}.',
        ]
    )
    user_message = json.dumps(
        {
            "candidate_memories": [
                {
                    "candidate_id": candidate["candidate_id"],
                    "memory": _sanitize_visible_text(candidate["memory"]),
                }
                for candidate in candidates
            ],
            "existing_memories": [
                {
                    "memory_id": record.get("memory_id"),
                    "memory": _sanitize_visible_text(str(record.get("memory", ""))),
                    "score": record.get("retrieval_score"),
                }
                for record in existing_memories
            ],
        },
        ensure_ascii=False,
        indent=2,
    )
    return _generate_memory_response(
        llm_client,
        system_message=system_message,
        user_message=user_message,
        response_schema=Mem0UpdateResponse,
        max_output_tokens=settings.mem0_max_output_tokens,
        retry_limit=settings.retry_limit,
        model_name=model_name,
    )


def _refresh_summary(
    llm_client: Any,
    *,
    current_summary: str,
    previous_chunk_summaries: Sequence[str],
    chunk: AdmissionChunk,
    applied_operations: Sequence[dict[str, Any]],
    settings: EvaluationSettings,
    model_name: str,
) -> _MemoryLLMResult:
    system_message = "\n".join(
        [
            "Refresh the compact patient memory summary used by the memory extractor.",
            "Keep a chronological clinical timeline that helps interpret later chunks.",
            "Use only temporal context visible in conversation-line timestamps.",
            "Do not mention internal database identifiers, admission IDs, turn numbers, turn IDs, or global turn indices.",
            "Preserve dates, times, major clinical problems, treatments, and temporal changes.",
            f"Keep the summary under {MEM0_SUMMARY_CHAR_LIMIT} characters.",
            'Return strict JSON with the schema {"summary": "..."}.',
        ]
    )
    user_message = json.dumps(
        {
            "current_summary": _sanitize_visible_text(current_summary),
            "previous_chunk_summaries": [_sanitize_visible_text(item) for item in previous_chunk_summaries],
            "current_chunk": {
                "conversation": chunk.render_visible(),
            },
            "applied_memory_changes": [
                {
                    "operation": item.get("operation"),
                    "memory": _sanitize_visible_text(str(item.get("memory", ""))),
                }
                for item in applied_operations
                if item.get("memory")
            ],
        },
        ensure_ascii=False,
        indent=2,
    )
    return _generate_memory_response(
        llm_client,
        system_message=system_message,
        user_message=user_message,
        response_schema=Mem0SummaryResponse,
        max_output_tokens=min(int(settings.mem0_max_output_tokens), MEM0_SUMMARY_MAX_OUTPUT_TOKENS),
        retry_limit=settings.retry_limit,
        model_name=model_name,
    )


def _generate_memory_response(
    llm_client: Any,
    *,
    system_message: str,
    user_message: str,
    response_schema: type[BaseModel],
    max_output_tokens: int,
    retry_limit: int,
    model_name: str,
) -> _MemoryLLMResult:
    max_attempts = max(1, int(retry_limit) + 1)
    last_exc: Exception | None = None
    for _attempt in range(1, max_attempts + 1):
        try:
            llm_result = llm_client.generate_structured_response(
                system_message,
                user_message,
                response_schema,
                max_output_tokens=max_output_tokens,
            )
            parsed = response_schema.model_validate(
                _normalize_memory_payload(response_schema, llm_result.parsed_output)
            )
            return _MemoryLLMResult(
                parsed=parsed,
                raw_response=llm_result.raw_response,
                usage=llm_result.usage,
                response_id=llm_result.response_id,
                latency_ms=llm_result.latency_ms,
            )
        except StructuredResponseValidationError as exc:
            recovered = _recover_memory_response_from_structured_error(exc, response_schema)
            if recovered is not None:
                return recovered
            last_exc = exc
        except Exception as exc:  # pragma: no cover - pipeline error path
            last_exc = exc
    raise RuntimeError(
        f"Mem0 memory call failed for {model_name} after {max_attempts} attempt(s): {last_exc}"
    ) from last_exc


def _recover_memory_response_from_structured_error(
    exc: StructuredResponseValidationError,
    response_schema: type[BaseModel],
) -> _MemoryLLMResult | None:
    last_exc: Exception | None = None
    for candidate in structured_content_candidates(exc.content):
        try:
            payload = json.loads(candidate, strict=False)
        except json.JSONDecodeError as parse_exc:
            last_exc = parse_exc
            continue
        try:
            parsed = response_schema.model_validate(_normalize_memory_payload(response_schema, payload))
        except Exception as validation_exc:
            last_exc = validation_exc
            continue
        return _MemoryLLMResult(
            parsed=parsed,
            raw_response=exc.raw_response,
            usage=exc.usage,
            response_id=exc.response_id,
            latency_ms=exc.latency_ms,
        )
    del last_exc
    return None


def _normalize_memory_payload(response_schema: type[BaseModel], payload: Any) -> Any:
    if response_schema is Mem0ExtractionResponse:
        if isinstance(payload, dict) and "memories" in payload:
            return payload
        if _looks_like_candidate_memory(payload):
            return {"memories": [payload]}
        if isinstance(payload, list):
            return {"memories": payload}
    if response_schema is MemAlphaExtractionResponse:
        if isinstance(payload, dict) and any(key in payload for key in ("core", "episodic", "semantic")):
            return payload
        if isinstance(payload, dict) and isinstance(payload.get("memories"), list):
            return _group_memalpha_candidates(payload["memories"])
        if _looks_like_typed_candidate_memory(payload) or _looks_like_candidate_memory(payload):
            return _group_memalpha_candidates([payload])
        if isinstance(payload, list):
            return _group_memalpha_candidates(payload)
    if response_schema is Mem0UpdateResponse:
        if isinstance(payload, dict) and "actions" in payload:
            return payload
        if _looks_like_update_action(payload):
            return {"actions": [payload]}
        if isinstance(payload, list):
            return {"actions": payload}
    return payload


def _looks_like_candidate_memory(payload: Any) -> bool:
    return isinstance(payload, dict) and "candidate_id" in payload and "memory" in payload


def _looks_like_typed_candidate_memory(payload: Any) -> bool:
    return (
        isinstance(payload, dict)
        and "candidate_id" in payload
        and "memory" in payload
        and "memory_type" in payload
    )


def _group_memalpha_candidates(payloads: Sequence[Any]) -> dict[str, list[Any]]:
    grouped: dict[str, list[Any]] = {"core": [], "episodic": [], "semantic": []}
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        memory_type = str(payload.get("memory_type") or "semantic").strip().lower()
        if memory_type not in grouped:
            memory_type = "semantic"
        grouped[memory_type].append(
            {
                "candidate_id": payload.get("candidate_id"),
                "memory": payload.get("memory"),
            }
        )
    return grouped


def _looks_like_update_action(payload: Any) -> bool:
    return isinstance(payload, dict) and "candidate_id" in payload and "operation" in payload


def _normalize_extracted_candidates(
    parsed: BaseModel,
    *,
    evaluation_variant: str,
    limit: int,
) -> list[dict[str, Any]]:
    if isinstance(parsed, MemAlphaExtractionResponse) or evaluation_variant == "memalpha":
        flattened: list[dict[str, Any]] = []
        for memory_type in ("core", "episodic", "semantic"):
            candidates = getattr(parsed, memory_type, [])
            for candidate in candidates:
                flattened.append(
                    {
                        "candidate_id": str(candidate.candidate_id),
                        "memory": candidate.memory,
                        "memory_type": memory_type,
                    }
                )
        return _normalize_candidates(flattened, limit=limit)
    if isinstance(parsed, Mem0ExtractionResponse):
        return _normalize_candidates(parsed.memories, limit=limit)
    return []


def _count_extracted_candidates(parsed: BaseModel) -> int:
    if isinstance(parsed, MemAlphaExtractionResponse):
        return len(parsed.core) + len(parsed.episodic) + len(parsed.semantic)
    if isinstance(parsed, Mem0ExtractionResponse):
        return len(parsed.memories)
    return 0


def _add_only_memory_candidates(
    store: Mem0MemoryStore,
    *,
    candidates: Sequence[dict[str, Any]],
    chunk: AdmissionChunk,
) -> list[dict[str, Any]]:
    seen_memory_texts = {
        _memory_dedup_key(str(record.get("memory", "")))
        for record in store.active_records()
        if str(record.get("memory", "")).strip()
    }
    applied: list[dict[str, Any]] = []
    for candidate in candidates:
        memory = _sanitize_visible_text(str(candidate.get("memory", "")))
        dedup_key = _memory_dedup_key(memory)
        if not memory:
            continue
        if dedup_key in seen_memory_texts:
            applied.append(
                {
                    "operation": "SKIP_DUPLICATE",
                    "candidate_id": candidate.get("candidate_id"),
                    "target_memory_id": None,
                    "memory_id": None,
                    "memory": memory,
                    "used_fallback": False,
                    "requested_operation": "ADD",
                    **(
                        {"memory_type": str(candidate.get("memory_type"))}
                        if candidate.get("memory_type") is not None
                        else {}
                    ),
                }
            )
            continue
        seen_memory_texts.add(dedup_key)
        applied.append(
            _add_memory(
                store,
                candidate=candidate,
                memory=memory,
                chunk=chunk,
                used_fallback=False,
                requested_operation="ADD",
                memory_type=(
                    str(candidate.get("memory_type"))
                    if candidate.get("memory_type") is not None
                    else None
                ),
            )
        )
    return applied


def _memory_dedup_key(memory: str) -> str:
    return re.sub(r"\s+", " ", str(memory or "").strip().lower())


def _apply_update_actions(
    store: Mem0MemoryStore,
    *,
    candidates: Sequence[dict[str, Any]],
    actions: Sequence[Mem0UpdateAction],
    chunk: AdmissionChunk,
) -> list[dict[str, Any]]:
    candidates_by_id = {str(candidate["candidate_id"]): candidate for candidate in candidates}
    actions_by_candidate = {str(action.candidate_id): action for action in actions}
    applied: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        action = actions_by_candidate.get(candidate_id)
        if action is None:
            applied.append(
                _add_memory(
                    store,
                    candidate=candidate,
                    memory=candidate["memory"],
                    chunk=chunk,
                    used_fallback=True,
                    requested_operation="ADD",
                )
            )
            continue
        operation = str(action.operation or "").strip().upper()
        if operation == "NOOP":
            applied.append(
                {
                    "operation": "NOOP",
                    "candidate_id": candidate_id,
                    "target_memory_id": action.target_memory_id,
                    "memory_id": None,
                    "memory": None,
                    "used_fallback": False,
                }
            )
        elif operation == "ADD":
            applied.append(
                _add_memory(
                    store,
                    candidate=candidates_by_id[candidate_id],
                    memory=action.memory or candidate["memory"],
                    chunk=chunk,
                    used_fallback=False,
                    requested_operation="ADD",
                )
            )
        elif operation == "UPDATE":
            target = _find_active_record(store, action.target_memory_id)
            if target is None:
                applied.append(
                    _add_memory(
                        store,
                        candidate=candidates_by_id[candidate_id],
                        memory=action.memory or candidate["memory"],
                        chunk=chunk,
                        used_fallback=True,
                        requested_operation="UPDATE",
                    )
                )
            else:
                final_memory = _sanitize_visible_text(action.memory or candidate["memory"])
                target["memory"] = final_memory
                applied.append(
                    {
                        "operation": "UPDATE",
                        "candidate_id": candidate_id,
                        "target_memory_id": target["memory_id"],
                        "memory_id": target["memory_id"],
                        "memory": final_memory,
                        "used_fallback": False,
                    }
                )
        elif operation == "DELETE":
            target = _find_active_record(store, action.target_memory_id)
            if target is None:
                applied.append(
                    {
                        "operation": "NOOP",
                        "candidate_id": candidate_id,
                        "target_memory_id": action.target_memory_id,
                        "memory_id": None,
                        "memory": None,
                        "used_fallback": True,
                        "requested_operation": "DELETE",
                    }
                )
            else:
                store.records = [
                    record
                    for record in store.records
                    if record.get("memory_id") != target["memory_id"]
                ]
                applied.append(
                    {
                        "operation": "DELETE",
                        "candidate_id": candidate_id,
                        "target_memory_id": target["memory_id"],
                        "memory_id": target["memory_id"],
                        "memory": None,
                        "used_fallback": False,
                    }
                )
        else:
            applied.append(
                _add_memory(
                    store,
                    candidate=candidate,
                    memory=candidate["memory"],
                    chunk=chunk,
                    used_fallback=True,
                    requested_operation=operation or "UNKNOWN",
                )
            )
    return applied


def _add_memory(
    store: Mem0MemoryStore,
    *,
    candidate: dict[str, Any],
    memory: str,
    chunk: AdmissionChunk,
    used_fallback: bool,
    requested_operation: str,
    memory_type: str | None = None,
) -> dict[str, Any]:
    del chunk
    memory_id = f"mem_{store.next_memory_index:05d}"
    store.next_memory_index += 1
    final_memory = _sanitize_visible_text(memory)
    record = {
        "memory_id": memory_id,
        "memory": final_memory,
    }
    if memory_type is not None:
        record["memory_type"] = str(memory_type)
    store.records.append(record)
    applied = {
        "operation": "ADD",
        "candidate_id": candidate.get("candidate_id"),
        "target_memory_id": None,
        "memory_id": memory_id,
        "memory": final_memory,
        "used_fallback": bool(used_fallback),
        "requested_operation": requested_operation,
    }
    if memory_type is not None:
        applied["memory_type"] = str(memory_type)
    return applied


def _normalize_candidates(candidates: Sequence[Any], *, limit: int) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, candidate in enumerate(candidates[: int(limit)], start=1):
        if isinstance(candidate, dict):
            raw_candidate_id = candidate.get("candidate_id")
            raw_memory = candidate.get("memory")
            raw_memory_type = candidate.get("memory_type")
        else:
            raw_candidate_id = getattr(candidate, "candidate_id", None)
            raw_memory = getattr(candidate, "memory", "")
            raw_memory_type = getattr(candidate, "memory_type", None)
        memory = _sanitize_visible_text(str(raw_memory or ""))
        if not memory:
            continue
        candidate_id = str(raw_candidate_id or f"c{index:03d}").strip() or f"c{index:03d}"
        if candidate_id in seen_ids:
            candidate_id = f"c{index:03d}"
        seen_ids.add(candidate_id)
        record = {
            "candidate_id": candidate_id,
            "memory": memory,
        }
        if raw_memory_type is not None:
            memory_type = str(raw_memory_type).strip().lower()
            record["memory_type"] = memory_type if memory_type in {"core", "episodic", "semantic"} else "semantic"
        normalized.append(record)
    return normalized


def _find_active_record(store: Mem0MemoryStore, memory_id: str | None) -> dict[str, Any] | None:
    if not memory_id:
        return None
    for record in store.records:
        if record.get("memory_id") == memory_id and not bool(record.get("deleted", False)):
            return record
    return None


def _event_record(
    *,
    event_type: str,
    chunk: AdmissionChunk,
    llm_result: _MemoryLLMResult,
    settings: EvaluationSettings,
    extra: dict[str, Any],
) -> dict[str, Any]:
    record = {
        "timestamp": _utc_now_iso(),
        "event_type": event_type,
        "chunk": chunk.to_record(),
        "api_usage": llm_result.usage,
        "latency_ms": llm_result.latency_ms,
        "response_id": llm_result.response_id,
        "raw_structured_response": llm_result.parsed.model_dump(mode="json"),
        **extra,
    }
    if settings.save_raw_response:
        record["raw_response"] = llm_result.raw_response
    return record


def _error_event_record(
    *,
    event_type: str,
    chunk: AdmissionChunk,
    error: Exception,
    extra: dict[str, Any],
) -> dict[str, Any]:
    return {
        "timestamp": _utc_now_iso(),
        "event_type": event_type,
        "chunk": chunk.to_record(),
        "error_type": type(error).__name__,
        "error_kind": "internal_error",
        "message": str(error),
        **extra,
    }


def _estimate_mem0_context(
    *,
    model_name: str,
    tokenizer_name: str | None,
    context_text: str,
    questions: Sequence[EvalQuestion],
) -> PromptTokenEstimate:
    rendered = render_answer_prompt(
        context_text=context_text,
        questions=[question.model_question() for question in questions],
        context_description=MEM0_CONTEXT_DESCRIPTION,
        context_payload_key=MEM0_CONTEXT_PAYLOAD_KEY,
    )
    return estimate_prompt_tokens(
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        system_message=rendered.system_message,
        user_message=rendered.user_message,
    )


def _split_turns_by_token_cap(
    turns: list[ConversationTurn],
    *,
    model_name: str,
    tokenizer_name: str | None,
    chunk_token_cap: int,
) -> list[list[ConversationTurn]]:
    chunks: list[list[ConversationTurn]] = []
    current: list[ConversationTurn] = []
    for turn in turns:
        candidate = current + [turn]
        if current and _estimate_visible_turn_tokens(
            candidate,
            model_name=model_name,
            tokenizer_name=tokenizer_name,
        ) > int(chunk_token_cap):
            chunks.append(current)
            current = [turn]
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _estimate_visible_turn_tokens(
    turns: Sequence[ConversationTurn],
    *,
    model_name: str,
    tokenizer_name: str | None,
) -> int:
    estimate = estimate_prompt_tokens(
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        system_message="",
        user_message="\n".join(turn.render_visible() for turn in turns),
    )
    return int(estimate.total_tokens)


def _sanitize_visible_text(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    text = re.sub(r"\bhadm_id\s*=?\s*\d+\b", "the admission", text, flags=re.IGNORECASE)
    text = re.sub(r"\bhadm\s*=\s*\d+\b", "the admission", text, flags=re.IGNORECASE)
    text = re.sub(r"\bturn_number\s*=?\s*\d+\b", "the turn", text, flags=re.IGNORECASE)
    text = re.sub(r"\bturn\s*=\s*\d+\b", "the turn", text, flags=re.IGNORECASE)
    text = re.sub(r"\bglobal\s*=\s*\d+\b", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def _first_visible_memory_timestamp(value: str) -> datetime | None:
    match = _VISIBLE_MEMORY_TIMESTAMP_PATTERN.search(str(value or ""))
    if match is None:
        return None
    date_part = match.group(1)
    time_part = match.group(2)
    timestamp_text = f"{date_part} {time_part}" if time_part else date_part
    try:
        return datetime.fromisoformat(timestamp_text)
    except ValueError:
        return None


def _clamp_summary(value: str) -> str:
    text = _sanitize_visible_text(value)
    if len(text) <= MEM0_SUMMARY_CHAR_LIMIT:
        return text
    truncated = text[:MEM0_SUMMARY_CHAR_LIMIT].rsplit(" ", 1)[0].rstrip(" ,;:")
    return truncated or text[:MEM0_SUMMARY_CHAR_LIMIT]


def _normalize_dense_vector(vector: Sequence[float]) -> list[float]:
    values = [float(value) for value in vector]
    norm = sum(value * value for value in values) ** 0.5
    if norm <= 0:
        return [0.0 for _value in values]
    return [value / norm for value in values]


def _dense_dot(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right:
        return 0.0
    return sum(float(left_value) * float(right_value) for left_value, right_value in zip(left, right))


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
