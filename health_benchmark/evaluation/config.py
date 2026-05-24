from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

from ..scripts.config import BenchmarkConfig, resolve_llm_base_url
from .types import EvaluationPaths, ModelArtifactPaths, ModelSpec


DEFAULT_EVALUATION_DIR_NAME = "evaluation"
DEFAULT_BATCH_SIZE = 10
DEFAULT_MAX_MODEL_LEN = 131072
DEFAULT_MAX_OUTPUT_TOKENS = 4096
DEFAULT_JUDGE_MAX_OUTPUT_TOKENS = 1024
DEFAULT_SAFE_MARGIN_TOKENS = 8192
DEFAULT_TOKEN_ESTIMATE_SAFETY_MULTIPLIER = 1.0
DEFAULT_PROVIDER = "vllm"
DEFAULT_EVALUATION_VARIANT = "normal"
LINGSHU_MAX_MODEL_LEN = 128000
MEM0_MODEL_SLUG_SUFFIX = "-mem0"
MEM0_STAR_MODEL_SLUG_SUFFIX = "-mem0-star"
MEMALPHA_MODEL_SLUG_SUFFIX = "-memalpha"
EMBEDDING_RAG_MODEL_SLUG_SUFFIX = "-embedding-rag"
BM25_RAG_MODEL_SLUG_SUFFIX = "-bm25-rag"
MEMORY_EVALUATION_VARIANTS = {"mem0", "mem0_star", "memalpha"}
MEMORY_METHOD_CHOICES: tuple[str, ...] = ("mem0", "mem0-star", "memalpha")
MEMORY_METHOD_TO_VARIANT = {
    "mem0": "mem0",
    "mem0-star": "mem0_star",
    "mem0_star": "mem0_star",
    "memalpha": "memalpha",
}
MEMORY_VARIANT_TO_SUFFIX = {
    "mem0": MEM0_MODEL_SLUG_SUFFIX,
    "mem0_star": MEM0_STAR_MODEL_SLUG_SUFFIX,
    "memalpha": MEMALPHA_MODEL_SLUG_SUFFIX,
}
RAG_EVALUATION_VARIANTS = {"embedding_rag", "bm25_rag"}
RAG_METHOD_CHOICES: tuple[str, ...] = ("embedding-rag", "bm25-rag")
RAG_METHOD_TO_VARIANT = {
    "embedding-rag": "embedding_rag",
    "embedding_rag": "embedding_rag",
    "bm25-rag": "bm25_rag",
    "bm25_rag": "bm25_rag",
}
RAG_VARIANT_TO_SUFFIX = {
    "embedding_rag": EMBEDDING_RAG_MODEL_SLUG_SUFFIX,
    "bm25_rag": BM25_RAG_MODEL_SLUG_SUFFIX,
}
DEFAULT_MEM0_CHUNK_TOKEN_CAP = 12000
DEFAULT_MEM0_PREVIOUS_CHUNK_SUMMARIES = 1
DEFAULT_MEM0_MAX_CANDIDATE_MEMORIES = 32
DEFAULT_MEM0_SIMILAR_MEMORIES = 10
DEFAULT_MEM0_MAX_UPDATE_MEMORIES = 40
DEFAULT_MEM0_ANSWER_RETRIEVAL_TOP_K = 64
DEFAULT_MEM0_MAX_ANSWER_MEMORIES = 32
DEFAULT_MEM0_MAX_OUTPUT_TOKENS = 4096
DEFAULT_MEM0_RETRIEVAL_BACKEND = "local_dense_hf"
DEFAULT_MEM0_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
DEFAULT_MEM0_EMBEDDING_DEVICE = "cuda"
DEFAULT_MEM0_EMBEDDING_GPU_DEVICE_IDS = ""
DEFAULT_MEM0_EMBEDDING_BATCH_SIZE = 8
DEFAULT_MEM0_EMBEDDING_MAX_LENGTH = 1024
DEFAULT_RAG_DOCUMENT_UNIT = "admission"
DEFAULT_RAG_SELECTION_POLICY = "score_until_budget"
DEFAULT_RAG_RENDER_ORDER = "chronological"
DEFAULT_RAG_EMBEDDING_MODEL = DEFAULT_MEM0_EMBEDDING_MODEL
DEFAULT_RAG_EMBEDDING_DEVICE = DEFAULT_MEM0_EMBEDDING_DEVICE
DEFAULT_RAG_EMBEDDING_GPU_DEVICE_IDS = ""
DEFAULT_RAG_EMBEDDING_BATCH_SIZE = DEFAULT_MEM0_EMBEDDING_BATCH_SIZE
DEFAULT_RAG_EMBEDDING_MAX_LENGTH = DEFAULT_MEM0_EMBEDDING_MAX_LENGTH
QWEN_MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        model_name="Qwen/Qwen3.5-4B",
        slug="qwen3.5-4b",
        tensor_parallel_size=1,
        max_model_len=DEFAULT_MAX_MODEL_LEN,
    ),
    ModelSpec(
        model_name="Qwen/Qwen3.5-9B",
        slug="qwen3.5-9b",
        tensor_parallel_size=1,
        max_model_len=DEFAULT_MAX_MODEL_LEN,
    ),
    ModelSpec(
        model_name="Qwen/Qwen3.5-27B",
        slug="qwen3.5-27b",
        tensor_parallel_size=2,
        max_model_len=DEFAULT_MAX_MODEL_LEN,
    ),
)
GEMMA3_MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        model_name="google/gemma-3-4b-it",
        slug="gemma-3-4b-it",
        tensor_parallel_size=1,
        max_model_len=DEFAULT_MAX_MODEL_LEN,
    ),
    ModelSpec(
        model_name="google/gemma-3-12b-it",
        slug="gemma-3-12b-it",
        tensor_parallel_size=1,
        max_model_len=DEFAULT_MAX_MODEL_LEN,
    ),
    ModelSpec(
        model_name="google/gemma-3-27b-it",
        slug="gemma-3-27b-it",
        tensor_parallel_size=2,
        max_model_len=DEFAULT_MAX_MODEL_LEN,
    ),
)
MEDGEMMA_MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        model_name="google/medgemma-4b-it",
        slug="medgemma-4b-it",
        tensor_parallel_size=1,
        max_model_len=DEFAULT_MAX_MODEL_LEN,
    ),
    ModelSpec(
        model_name="google/medgemma-27b-it",
        slug="medgemma-27b-it",
        tensor_parallel_size=2,
        max_model_len=DEFAULT_MAX_MODEL_LEN,
    ),
)
MEDICAL_NEXT_MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        model_name="MBZUAI/MedMO-4B-Next",
        slug="medmo-4b-next",
        tensor_parallel_size=1,
        max_model_len=DEFAULT_MAX_MODEL_LEN,
    ),
    ModelSpec(
        model_name="MBZUAI/MedMO-8B-Next",
        slug="medmo-8b-next",
        tensor_parallel_size=1,
        max_model_len=DEFAULT_MAX_MODEL_LEN,
    ),
    ModelSpec(
        model_name="microsoft/MediPhi-Instruct",
        slug="mediphi-instruct",
        tensor_parallel_size=1,
        max_model_len=DEFAULT_MAX_MODEL_LEN,
    ),
    ModelSpec(
        model_name="lingshu-medical-mllm/Lingshu-32B",
        slug="lingshu-32b",
        tensor_parallel_size=2,
        max_model_len=LINGSHU_MAX_MODEL_LEN,
    ),
)
DEFAULT_MODEL_SPECS: tuple[ModelSpec, ...] = QWEN_MODEL_SPECS
KNOWN_MODEL_SPECS: tuple[ModelSpec, ...] = (
    QWEN_MODEL_SPECS
    + GEMMA3_MODEL_SPECS
    + MEDGEMMA_MODEL_SPECS
    + MEDICAL_NEXT_MODEL_SPECS
)


def _model_aliases(spec: ModelSpec) -> tuple[str, ...]:
    aliases = {
        spec.model_name.lower(),
        spec.slug.lower(),
        spec.model_name.split("/")[-1].lower(),
    }
    if spec.slug.startswith("qwen3.5-"):
        aliases.add(spec.slug.replace("qwen3.5-", ""))
        aliases.add(spec.slug.replace(".", ""))
    if spec.slug.startswith("gemma-3-"):
        aliases.add(spec.slug.replace("gemma-3-", ""))
        aliases.add(spec.slug.replace("gemma-3-", "gemma3-"))
        aliases.add(spec.slug.replace("-", ""))
    if spec.slug.startswith("medgemma-"):
        aliases.add(spec.slug.replace("medgemma-", ""))
        aliases.add(spec.slug.replace("medgemma-", "medgemma"))
        aliases.add(spec.slug.replace("-", ""))
    if spec.slug.startswith("medmo-"):
        aliases.add(spec.slug.replace("medmo-", ""))
        aliases.add(spec.slug.replace("medmo-", "medmo"))
        aliases.add(spec.slug.replace("-", ""))
    if spec.slug == "mediphi-instruct":
        aliases.add("mediphi")
    if spec.slug == "lingshu-32b":
        aliases.add("lingshu")
    return tuple(sorted(aliases))


MODEL_ALIASES: dict[str, ModelSpec] = {
    alias: spec
    for spec in KNOWN_MODEL_SPECS
    for alias in _model_aliases(spec)
}
JUDGE_MODEL_SPEC = ModelSpec(
    model_name="Qwen/Qwen3.5-27B",
    slug="qwen3.5-27b",
    tensor_parallel_size=2,
    max_model_len=DEFAULT_MAX_MODEL_LEN,
)
EVALUATION_STAGE_CHOICES: tuple[Literal["full", "answers", "judge"], ...] = (
    "full",
    "answers",
    "judge",
)

@dataclass(frozen=True)
class EvaluationSettings:
    stage: Literal["full", "answers", "judge"]
    evaluation_variant: str
    provider: str
    base_url: str | None
    judge_base_url: str | None
    api_key_env: str
    timeout_seconds: int
    retry_limit: int
    max_output_tokens: int
    judge_max_output_tokens: int
    safe_margin_tokens: int
    token_estimate_safety_multiplier: float
    batch_size: int
    replace_existing: bool
    save_raw_response: bool
    tokenizer_name: str | None
    enable_thinking: bool
    mem0_chunk_token_cap: int
    mem0_previous_chunk_summaries: int
    mem0_max_candidate_memories: int
    mem0_similar_memories: int
    mem0_max_update_memories: int
    mem0_answer_retrieval_top_k: int
    mem0_max_answer_memories: int
    mem0_max_output_tokens: int
    mem0_retrieval_backend: str
    mem0_embedding_model: str
    mem0_embedding_device: str
    mem0_embedding_gpu_device_ids: str
    mem0_embedding_batch_size: int
    mem0_embedding_max_length: int
    mem0_model_max_len: int | None
    mem0_model_tensor_parallel_size: int | None
    rag_method: str
    rag_document_unit: str
    rag_selection_policy: str
    rag_render_order: str
    rag_embedding_model: str
    rag_embedding_device: str
    rag_embedding_gpu_device_ids: str
    rag_embedding_batch_size: int
    rag_embedding_max_length: int
    rag_model_max_len: int | None
    rag_model_tensor_parallel_size: int | None
    evaluation_root: Path
    model_specs: tuple[ModelSpec, ...]
    judge_model_spec: ModelSpec


def slugify_model_name(model_name: str) -> str:
    normalized = re.sub(r"[^a-z0-9.-]+", "-", str(model_name or "").strip().lower())
    normalized = re.sub(r"-+", "-", normalized).strip("-")
    if not normalized:
        raise ValueError("Model name must be non-empty")
    return normalized


def resolve_model_specs(models: Sequence[str] | None) -> tuple[ModelSpec, ...]:
    if not models:
        return DEFAULT_MODEL_SPECS
    resolved: list[ModelSpec] = []
    seen_slugs: set[str] = set()
    for raw in models:
        candidate = str(raw or "").strip()
        if not candidate:
            raise ValueError("Model names must be non-empty")
        spec = MODEL_ALIASES.get(candidate.lower())
        if spec is None:
            spec = ModelSpec(
                model_name=candidate,
                slug=slugify_model_name(candidate),
                tensor_parallel_size=1,
                max_model_len=DEFAULT_MAX_MODEL_LEN,
            )
        if spec.slug in seen_slugs:
            continue
        seen_slugs.add(spec.slug)
        resolved.append(spec)
    return tuple(resolved)


def build_settings(
    base_config: BenchmarkConfig,
    *,
    provider: str | None,
    base_url: str | None,
    api_key_env: str | None,
    models: Sequence[str] | None,
    replace_existing: bool | None,
    judge_base_url: str | None = None,
    stage: Literal["full", "answers", "judge"] = "full",
    evaluation_variant: str | None = None,
    timeout_seconds: int | None = None,
    retry_limit: int | None = None,
    max_output_tokens: int | None = None,
    safe_margin_tokens: int | None = None,
    token_estimate_safety_multiplier: float | None = None,
    mem0_chunk_token_cap: int | None = None,
    mem0_previous_chunk_summaries: int | None = None,
    mem0_max_candidate_memories: int | None = None,
    mem0_similar_memories: int | None = None,
    mem0_max_update_memories: int | None = None,
    mem0_answer_retrieval_top_k: int | None = None,
    mem0_max_answer_memories: int | None = None,
    mem0_max_output_tokens: int | None = None,
    mem0_embedding_model: str | None = None,
    mem0_embedding_device: str | None = None,
    mem0_embedding_gpu_device_ids: str | None = None,
    mem0_embedding_batch_size: int | None = None,
    mem0_embedding_max_length: int | None = None,
    mem0_model_max_len: int | None = None,
    mem0_model_tensor_parallel_size: int | None = None,
    rag_method: str | None = None,
    rag_document_unit: str | None = None,
    rag_selection_policy: str | None = None,
    rag_render_order: str | None = None,
    rag_embedding_model: str | None = None,
    rag_embedding_device: str | None = None,
    rag_embedding_gpu_device_ids: str | None = None,
    rag_embedding_batch_size: int | None = None,
    rag_embedding_max_length: int | None = None,
    rag_model_max_len: int | None = None,
    rag_model_tensor_parallel_size: int | None = None,
    evaluation_root: Path | None = None,
) -> EvaluationSettings:
    resolved_stage = str(stage or "full").strip().lower()
    if resolved_stage not in EVALUATION_STAGE_CHOICES:
        raise ValueError(
            f"stage must be one of {list(EVALUATION_STAGE_CHOICES)}, got: {stage}"
        )
    resolved_evaluation_variant = str(evaluation_variant or DEFAULT_EVALUATION_VARIANT).strip().lower()
    if resolved_evaluation_variant not in {"normal", *MEMORY_EVALUATION_VARIANTS, *RAG_EVALUATION_VARIANTS}:
        raise ValueError(
            "evaluation_variant must be one of ['normal', 'mem0', 'mem0_star', 'memalpha', 'embedding_rag', 'bm25_rag'], "
            f"got: {evaluation_variant}"
        )
    resolved_provider = str(provider or DEFAULT_PROVIDER).strip().lower()
    resolved_base_url = resolve_llm_base_url(
        resolved_provider,
        base_url if base_url is not None else base_config.llm.base_url,
    )
    resolved_timeout_seconds = int(
        base_config.llm.timeout_seconds if timeout_seconds is None else timeout_seconds
    )
    if resolved_timeout_seconds <= 0:
        raise ValueError(f"timeout_seconds must be positive, got: {resolved_timeout_seconds}")
    resolved_retry_limit = int(
        base_config.llm.max_retries if retry_limit is None else retry_limit
    )
    if resolved_retry_limit < 0:
        raise ValueError(f"retry_limit must be non-negative, got: {resolved_retry_limit}")
    resolved_token_estimate_safety_multiplier = float(
        DEFAULT_TOKEN_ESTIMATE_SAFETY_MULTIPLIER
        if token_estimate_safety_multiplier is None
        else token_estimate_safety_multiplier
    )
    if resolved_token_estimate_safety_multiplier < 1.0:
        raise ValueError(
            "token_estimate_safety_multiplier must be at least 1.0, "
            f"got: {resolved_token_estimate_safety_multiplier}"
        )
    resolved_mem0_chunk_token_cap = _positive_setting(
        "mem0_chunk_token_cap",
        DEFAULT_MEM0_CHUNK_TOKEN_CAP
        if mem0_chunk_token_cap is None
        else mem0_chunk_token_cap,
    )
    resolved_mem0_previous_chunk_summaries = _non_negative_setting(
        "mem0_previous_chunk_summaries",
        DEFAULT_MEM0_PREVIOUS_CHUNK_SUMMARIES
        if mem0_previous_chunk_summaries is None
        else mem0_previous_chunk_summaries,
    )
    resolved_mem0_max_candidate_memories = _positive_setting(
        "mem0_max_candidate_memories",
        DEFAULT_MEM0_MAX_CANDIDATE_MEMORIES
        if mem0_max_candidate_memories is None
        else mem0_max_candidate_memories,
    )
    resolved_mem0_similar_memories = _positive_setting(
        "mem0_similar_memories",
        DEFAULT_MEM0_SIMILAR_MEMORIES if mem0_similar_memories is None else mem0_similar_memories,
    )
    resolved_mem0_max_update_memories = _positive_setting(
        "mem0_max_update_memories",
        DEFAULT_MEM0_MAX_UPDATE_MEMORIES
        if mem0_max_update_memories is None
        else mem0_max_update_memories,
    )
    resolved_mem0_answer_retrieval_top_k = _positive_setting(
        "mem0_answer_retrieval_top_k",
        DEFAULT_MEM0_ANSWER_RETRIEVAL_TOP_K
        if mem0_answer_retrieval_top_k is None
        else mem0_answer_retrieval_top_k,
    )
    resolved_mem0_max_answer_memories = _positive_setting(
        "mem0_max_answer_memories",
        DEFAULT_MEM0_MAX_ANSWER_MEMORIES
        if mem0_max_answer_memories is None
        else mem0_max_answer_memories,
    )
    resolved_mem0_max_output_tokens = _positive_setting(
        "mem0_max_output_tokens",
        DEFAULT_MEM0_MAX_OUTPUT_TOKENS if mem0_max_output_tokens is None else mem0_max_output_tokens,
    )
    resolved_mem0_embedding_model = str(
        DEFAULT_MEM0_EMBEDDING_MODEL if mem0_embedding_model is None else mem0_embedding_model
    ).strip()
    if not resolved_mem0_embedding_model:
        raise ValueError("mem0_embedding_model must be non-empty")
    resolved_mem0_embedding_device = str(
        DEFAULT_MEM0_EMBEDDING_DEVICE if mem0_embedding_device is None else mem0_embedding_device
    ).strip()
    if not resolved_mem0_embedding_device:
        raise ValueError("mem0_embedding_device must be non-empty")
    resolved_mem0_embedding_gpu_device_ids = str(
        DEFAULT_MEM0_EMBEDDING_GPU_DEVICE_IDS
        if mem0_embedding_gpu_device_ids is None
        else mem0_embedding_gpu_device_ids
    ).strip()
    resolved_mem0_embedding_batch_size = _positive_setting(
        "mem0_embedding_batch_size",
        DEFAULT_MEM0_EMBEDDING_BATCH_SIZE
        if mem0_embedding_batch_size is None
        else mem0_embedding_batch_size,
    )
    resolved_mem0_embedding_max_length = _positive_setting(
        "mem0_embedding_max_length",
        DEFAULT_MEM0_EMBEDDING_MAX_LENGTH
        if mem0_embedding_max_length is None
        else mem0_embedding_max_length,
    )
    resolved_mem0_model_max_len = (
        None
        if mem0_model_max_len is None
        else _positive_setting("mem0_model_max_len", mem0_model_max_len)
    )
    resolved_mem0_model_tensor_parallel_size = (
        None
        if mem0_model_tensor_parallel_size is None
        else _positive_setting("mem0_model_tensor_parallel_size", mem0_model_tensor_parallel_size)
    )
    resolved_rag_method = normalize_rag_method(
        rag_method
        if rag_method is not None
        else (
            resolved_evaluation_variant
            if resolved_evaluation_variant in RAG_EVALUATION_VARIANTS
            else "embedding-rag"
        )
    )
    resolved_rag_document_unit = str(
        DEFAULT_RAG_DOCUMENT_UNIT if rag_document_unit is None else rag_document_unit
    ).strip().lower()
    if resolved_rag_document_unit != "admission":
        raise ValueError(f"rag_document_unit must be 'admission', got: {rag_document_unit}")
    resolved_rag_selection_policy = str(
        DEFAULT_RAG_SELECTION_POLICY if rag_selection_policy is None else rag_selection_policy
    ).strip().lower()
    if resolved_rag_selection_policy != "score_until_budget":
        raise ValueError(
            f"rag_selection_policy must be 'score_until_budget', got: {rag_selection_policy}"
        )
    resolved_rag_render_order = str(
        DEFAULT_RAG_RENDER_ORDER if rag_render_order is None else rag_render_order
    ).strip().lower()
    if resolved_rag_render_order != "chronological":
        raise ValueError(f"rag_render_order must be 'chronological', got: {rag_render_order}")
    resolved_rag_embedding_model = str(
        DEFAULT_RAG_EMBEDDING_MODEL if rag_embedding_model is None else rag_embedding_model
    ).strip()
    if not resolved_rag_embedding_model:
        raise ValueError("rag_embedding_model must be non-empty")
    resolved_rag_embedding_device = str(
        DEFAULT_RAG_EMBEDDING_DEVICE if rag_embedding_device is None else rag_embedding_device
    ).strip()
    if not resolved_rag_embedding_device:
        raise ValueError("rag_embedding_device must be non-empty")
    resolved_rag_embedding_gpu_device_ids = str(
        DEFAULT_RAG_EMBEDDING_GPU_DEVICE_IDS
        if rag_embedding_gpu_device_ids is None
        else rag_embedding_gpu_device_ids
    ).strip()
    resolved_rag_embedding_batch_size = _positive_setting(
        "rag_embedding_batch_size",
        DEFAULT_RAG_EMBEDDING_BATCH_SIZE
        if rag_embedding_batch_size is None
        else rag_embedding_batch_size,
    )
    resolved_rag_embedding_max_length = _positive_setting(
        "rag_embedding_max_length",
        DEFAULT_RAG_EMBEDDING_MAX_LENGTH
        if rag_embedding_max_length is None
        else rag_embedding_max_length,
    )
    resolved_rag_model_max_len = (
        None
        if rag_model_max_len is None
        else _positive_setting("rag_model_max_len", rag_model_max_len)
    )
    resolved_rag_model_tensor_parallel_size = (
        None
        if rag_model_tensor_parallel_size is None
        else _positive_setting("rag_model_tensor_parallel_size", rag_model_tensor_parallel_size)
    )
    resolved_model_specs = resolve_model_specs(models)
    if resolved_evaluation_variant in MEMORY_EVALUATION_VARIANTS and (
        resolved_mem0_model_max_len is not None
        or resolved_mem0_model_tensor_parallel_size is not None
    ):
        resolved_model_specs = tuple(
            ModelSpec(
                model_name=spec.model_name,
                slug=spec.slug,
                tensor_parallel_size=(
                    spec.tensor_parallel_size
                    if resolved_mem0_model_tensor_parallel_size is None
                    else resolved_mem0_model_tensor_parallel_size
                ),
                max_model_len=(
                    spec.max_model_len
                    if resolved_mem0_model_max_len is None
                    else resolved_mem0_model_max_len
                ),
            )
            for spec in resolved_model_specs
        )
    if resolved_evaluation_variant in RAG_EVALUATION_VARIANTS and (
        resolved_rag_model_max_len is not None
        or resolved_rag_model_tensor_parallel_size is not None
    ):
        resolved_model_specs = tuple(
            ModelSpec(
                model_name=spec.model_name,
                slug=spec.slug,
                tensor_parallel_size=(
                    spec.tensor_parallel_size
                    if resolved_rag_model_tensor_parallel_size is None
                    else resolved_rag_model_tensor_parallel_size
                ),
                max_model_len=(
                    spec.max_model_len
                    if resolved_rag_model_max_len is None
                    else resolved_rag_model_max_len
                ),
            )
            for spec in resolved_model_specs
        )
    return EvaluationSettings(
        stage=resolved_stage,
        evaluation_variant=resolved_evaluation_variant,
        provider=resolved_provider,
        base_url=resolved_base_url,
        judge_base_url=(
            None
            if judge_base_url is None
            else resolve_llm_base_url(resolved_provider, judge_base_url)
        ),
        api_key_env=str(api_key_env or base_config.llm.api_key_env),
        timeout_seconds=resolved_timeout_seconds,
        retry_limit=resolved_retry_limit,
        max_output_tokens=int(
            DEFAULT_MAX_OUTPUT_TOKENS if max_output_tokens is None else max_output_tokens
        ),
        judge_max_output_tokens=int(DEFAULT_JUDGE_MAX_OUTPUT_TOKENS),
        safe_margin_tokens=int(
            DEFAULT_SAFE_MARGIN_TOKENS if safe_margin_tokens is None else safe_margin_tokens
        ),
        token_estimate_safety_multiplier=resolved_token_estimate_safety_multiplier,
        batch_size=DEFAULT_BATCH_SIZE,
        replace_existing=(
            bool(base_config.runtime.replace_existing_patient_output)
            if replace_existing is None
            else bool(replace_existing)
        ),
        save_raw_response=bool(base_config.runtime.save_raw_response),
        tokenizer_name=base_config.llm.tokenizer_name,
        enable_thinking=bool(base_config.vllm.enable_thinking),
        mem0_chunk_token_cap=resolved_mem0_chunk_token_cap,
        mem0_previous_chunk_summaries=resolved_mem0_previous_chunk_summaries,
        mem0_max_candidate_memories=resolved_mem0_max_candidate_memories,
        mem0_similar_memories=resolved_mem0_similar_memories,
        mem0_max_update_memories=resolved_mem0_max_update_memories,
        mem0_answer_retrieval_top_k=resolved_mem0_answer_retrieval_top_k,
        mem0_max_answer_memories=resolved_mem0_max_answer_memories,
        mem0_max_output_tokens=resolved_mem0_max_output_tokens,
        mem0_retrieval_backend=DEFAULT_MEM0_RETRIEVAL_BACKEND,
        mem0_embedding_model=resolved_mem0_embedding_model,
        mem0_embedding_device=resolved_mem0_embedding_device,
        mem0_embedding_gpu_device_ids=resolved_mem0_embedding_gpu_device_ids,
        mem0_embedding_batch_size=resolved_mem0_embedding_batch_size,
        mem0_embedding_max_length=resolved_mem0_embedding_max_length,
        mem0_model_max_len=resolved_mem0_model_max_len,
        mem0_model_tensor_parallel_size=resolved_mem0_model_tensor_parallel_size,
        rag_method=resolved_rag_method,
        rag_document_unit=resolved_rag_document_unit,
        rag_selection_policy=resolved_rag_selection_policy,
        rag_render_order=resolved_rag_render_order,
        rag_embedding_model=resolved_rag_embedding_model,
        rag_embedding_device=resolved_rag_embedding_device,
        rag_embedding_gpu_device_ids=resolved_rag_embedding_gpu_device_ids,
        rag_embedding_batch_size=resolved_rag_embedding_batch_size,
        rag_embedding_max_length=resolved_rag_embedding_max_length,
        rag_model_max_len=resolved_rag_model_max_len,
        rag_model_tensor_parallel_size=resolved_rag_model_tensor_parallel_size,
        evaluation_root=resolve_evaluation_root(
            base_config.output.root,
            evaluation_root=evaluation_root,
        ),
        model_specs=resolved_model_specs,
        judge_model_spec=JUDGE_MODEL_SPEC,
    )


def build_memory_settings(
    base_config: BenchmarkConfig,
    *,
    provider: str | None,
    base_url: str | None,
    api_key_env: str | None,
    models: Sequence[str] | None,
    replace_existing: bool | None,
    judge_base_url: str | None = None,
    stage: Literal["full", "answers", "judge"] = "full",
    timeout_seconds: int | None = None,
    retry_limit: int | None = None,
    max_output_tokens: int | None = None,
    safe_margin_tokens: int | None = None,
    token_estimate_safety_multiplier: float | None = None,
    mem0_chunk_token_cap: int | None = None,
    mem0_previous_chunk_summaries: int | None = None,
    mem0_max_candidate_memories: int | None = None,
    mem0_similar_memories: int | None = None,
    mem0_max_update_memories: int | None = None,
    mem0_answer_retrieval_top_k: int | None = None,
    mem0_max_answer_memories: int | None = None,
    mem0_max_output_tokens: int | None = None,
    mem0_embedding_model: str | None = None,
    mem0_embedding_device: str | None = None,
    mem0_embedding_gpu_device_ids: str | None = None,
    mem0_embedding_batch_size: int | None = None,
    mem0_embedding_max_length: int | None = None,
    mem0_model_max_len: int | None = None,
    mem0_model_tensor_parallel_size: int | None = None,
    memory_method: str | None = None,
    evaluation_root: Path | None = None,
) -> EvaluationSettings:
    resolved_memory_method = normalize_memory_method(memory_method)
    return build_settings(
        base_config,
        provider=provider,
        base_url=base_url,
        judge_base_url=judge_base_url,
        api_key_env=api_key_env,
        models=models,
        stage=stage,
        evaluation_variant=resolved_memory_method,
        replace_existing=replace_existing,
        timeout_seconds=timeout_seconds,
        retry_limit=retry_limit,
        max_output_tokens=max_output_tokens,
        safe_margin_tokens=safe_margin_tokens,
        token_estimate_safety_multiplier=token_estimate_safety_multiplier,
        mem0_chunk_token_cap=mem0_chunk_token_cap,
        mem0_previous_chunk_summaries=mem0_previous_chunk_summaries,
        mem0_max_candidate_memories=mem0_max_candidate_memories,
        mem0_similar_memories=mem0_similar_memories,
        mem0_max_update_memories=mem0_max_update_memories,
        mem0_answer_retrieval_top_k=mem0_answer_retrieval_top_k,
        mem0_max_answer_memories=mem0_max_answer_memories,
        mem0_max_output_tokens=mem0_max_output_tokens,
        mem0_embedding_model=mem0_embedding_model,
        mem0_embedding_device=mem0_embedding_device,
        mem0_embedding_gpu_device_ids=mem0_embedding_gpu_device_ids,
        mem0_embedding_batch_size=mem0_embedding_batch_size,
        mem0_embedding_max_length=mem0_embedding_max_length,
        mem0_model_max_len=mem0_model_max_len,
        mem0_model_tensor_parallel_size=mem0_model_tensor_parallel_size,
        evaluation_root=evaluation_root,
    )


def build_rag_settings(
    base_config: BenchmarkConfig,
    *,
    provider: str | None,
    base_url: str | None,
    api_key_env: str | None,
    models: Sequence[str] | None,
    replace_existing: bool | None,
    judge_base_url: str | None = None,
    stage: Literal["full", "answers", "judge"] = "full",
    timeout_seconds: int | None = None,
    retry_limit: int | None = None,
    max_output_tokens: int | None = None,
    safe_margin_tokens: int | None = None,
    token_estimate_safety_multiplier: float | None = None,
    rag_method: str | None = None,
    rag_document_unit: str | None = None,
    rag_selection_policy: str | None = None,
    rag_render_order: str | None = None,
    rag_embedding_model: str | None = None,
    rag_embedding_device: str | None = None,
    rag_embedding_gpu_device_ids: str | None = None,
    rag_embedding_batch_size: int | None = None,
    rag_embedding_max_length: int | None = None,
    rag_model_max_len: int | None = None,
    rag_model_tensor_parallel_size: int | None = None,
    evaluation_root: Path | None = None,
) -> EvaluationSettings:
    resolved_rag_method = normalize_rag_method(rag_method)
    return build_settings(
        base_config,
        provider=provider,
        base_url=base_url,
        judge_base_url=judge_base_url,
        api_key_env=api_key_env,
        models=models,
        stage=stage,
        evaluation_variant=resolved_rag_method,
        replace_existing=replace_existing,
        timeout_seconds=timeout_seconds,
        retry_limit=retry_limit,
        max_output_tokens=max_output_tokens,
        safe_margin_tokens=safe_margin_tokens,
        token_estimate_safety_multiplier=token_estimate_safety_multiplier,
        rag_method=rag_method,
        rag_document_unit=rag_document_unit,
        rag_selection_policy=rag_selection_policy,
        rag_render_order=rag_render_order,
        rag_embedding_model=rag_embedding_model,
        rag_embedding_device=rag_embedding_device,
        rag_embedding_gpu_device_ids=rag_embedding_gpu_device_ids,
        rag_embedding_batch_size=rag_embedding_batch_size,
        rag_embedding_max_length=rag_embedding_max_length,
        rag_model_max_len=rag_model_max_len,
        rag_model_tensor_parallel_size=rag_model_tensor_parallel_size,
        evaluation_root=evaluation_root,
    )


def normalize_memory_method(memory_method: str | None) -> str:
    key = str(memory_method or "mem0").strip().lower().replace("_", "-")
    if key not in MEMORY_METHOD_TO_VARIANT:
        raise ValueError(
            f"memory_method must be one of {list(MEMORY_METHOD_CHOICES)}, got: {memory_method}"
        )
    return MEMORY_METHOD_TO_VARIANT[key]


def normalize_rag_method(rag_method: str | None) -> str:
    key = str(rag_method or "embedding-rag").strip().lower().replace("_", "-")
    if key not in RAG_METHOD_TO_VARIANT:
        raise ValueError(
            f"rag_method must be one of {list(RAG_METHOD_CHOICES)}, got: {rag_method}"
        )
    return RAG_METHOD_TO_VARIANT[key]


def memory_suffix_for_variant(evaluation_variant: str) -> str:
    return MEMORY_VARIANT_TO_SUFFIX.get(str(evaluation_variant), MEM0_MODEL_SLUG_SUFFIX)


def rag_suffix_for_variant(evaluation_variant: str) -> str:
    return RAG_VARIANT_TO_SUFFIX.get(str(evaluation_variant), EMBEDDING_RAG_MODEL_SLUG_SUFFIX)


def base_slug_for_memory_slug(model_slug: str, evaluation_variant: str) -> str | None:
    suffix = MEMORY_VARIANT_TO_SUFFIX.get(str(evaluation_variant))
    if suffix and str(model_slug).endswith(suffix):
        return str(model_slug)[: -len(suffix)]
    for candidate_suffix in sorted(MEMORY_VARIANT_TO_SUFFIX.values(), key=len, reverse=True):
        if str(model_slug).endswith(candidate_suffix):
            return str(model_slug)[: -len(candidate_suffix)]
    return None


def base_slug_for_rag_slug(model_slug: str, evaluation_variant: str) -> str | None:
    suffix = RAG_VARIANT_TO_SUFFIX.get(str(evaluation_variant))
    if suffix and str(model_slug).endswith(suffix):
        return str(model_slug)[: -len(suffix)]
    for candidate_suffix in sorted(RAG_VARIANT_TO_SUFFIX.values(), key=len, reverse=True):
        if str(model_slug).endswith(candidate_suffix):
            return str(model_slug)[: -len(candidate_suffix)]
    return None


def resolve_evaluation_root(
    benchmark_root: Path,
    *,
    evaluation_root: Path | None = None,
) -> Path:
    if evaluation_root is not None:
        return evaluation_root.expanduser().resolve()
    return (benchmark_root.expanduser().resolve().parent / DEFAULT_EVALUATION_DIR_NAME).resolve()


def build_evaluation_paths(
    patient_root: Path,
    *,
    evaluation_root: Path,
    subject_id: str | int,
) -> EvaluationPaths:
    patient_evaluation_root = evaluation_root / str(subject_id)
    return EvaluationPaths(
        patient_root=patient_root,
        evaluation_root=patient_evaluation_root,
        comparison_dir=patient_evaluation_root / "comparison",
        config_json=patient_evaluation_root / "config.json",
        context_stats_json=patient_evaluation_root / "context_stats.json",
        benchmark_snapshot_json=patient_evaluation_root / "benchmark_snapshot.json",
    )


def build_model_artifact_paths(paths: EvaluationPaths, model_spec: ModelSpec) -> ModelArtifactPaths:
    model_dir = paths.model_dir(model_spec.slug)
    return ModelArtifactPaths(
        model_dir=model_dir,
        run_config_json=model_dir / "run_config.json",
        question_batches_json=model_dir / "question_batches.json",
        memory_store_json=model_dir / "memory_store.json",
        memory_events_jsonl=model_dir / "memory_events.jsonl",
        retrieval_store_json=model_dir / "retrieval_store.json",
        raw_predictions_jsonl=model_dir / "raw_predictions.jsonl",
        scored_predictions_jsonl=model_dir / "scored_predictions.jsonl",
        llm_judgments_jsonl=model_dir / "llm_judgments.jsonl",
        summary_json=model_dir / "summary.json",
        errors_jsonl=model_dir / "errors.jsonl",
    )


def clone_config_for_model(
    base_config: BenchmarkConfig,
    *,
    provider: str,
    model_spec: ModelSpec,
    max_output_tokens: int,
    base_url: str | None = None,
    api_key_env: str | None = None,
    timeout_seconds: int | None = None,
) -> BenchmarkConfig:
    cloned = copy.deepcopy(base_config)
    cloned.llm.provider = str(provider)
    cloned.llm.model = str(model_spec.model_name)
    cloned.llm.max_output_tokens = int(max_output_tokens)
    cloned.llm.temperature = 0.0
    cloned.llm.api_key_env = str(api_key_env or cloned.llm.api_key_env)
    cloned.llm.timeout_seconds = int(
        cloned.llm.timeout_seconds if timeout_seconds is None else timeout_seconds
    )
    cloned.llm.base_url = resolve_llm_base_url(
        cloned.llm.provider,
        base_url if base_url is not None else cloned.llm.base_url,
    )
    cloned.vllm.tensor_parallel_size = int(model_spec.tensor_parallel_size)
    cloned.vllm.max_model_len = int(model_spec.max_model_len)
    cloned.vllm.enable_thinking = False
    return cloned


def _positive_setting(name: str, value: int | str) -> int:
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"{name} must be positive, got: {resolved}")
    return resolved


def _non_negative_setting(name: str, value: int | str) -> int:
    resolved = int(value)
    if resolved < 0:
        raise ValueError(f"{name} must be non-negative, got: {resolved}")
    return resolved
