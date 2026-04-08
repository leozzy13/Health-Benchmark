from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ..scripts.config import BenchmarkConfig, resolve_llm_base_url
from .types import EvaluationPaths, ModelArtifactPaths, ModelSpec


DEFAULT_OUTPUT_DIR_NAME = "evaluation"
DEFAULT_BATCH_SIZE = 10
DEFAULT_MAX_OUTPUT_TOKENS = 1024
DEFAULT_SAFE_MARGIN_TOKENS = 1024
DEFAULT_PROVIDER = "vllm"
DEFAULT_MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        model_name="Qwen/Qwen3.5-4B",
        slug="qwen3.5-4b",
        tensor_parallel_size=1,
        max_model_len=262144,
    ),
    ModelSpec(
        model_name="Qwen/Qwen3.5-9B",
        slug="qwen3.5-9b",
        tensor_parallel_size=1,
        max_model_len=262144,
    ),
    ModelSpec(
        model_name="Qwen/Qwen3.5-27B",
        slug="qwen3.5-27b",
        tensor_parallel_size=8,
        max_model_len=262144,
    ),
)
MODEL_ALIASES: dict[str, ModelSpec] = {
    alias: spec
    for spec in DEFAULT_MODEL_SPECS
    for alias in (
        spec.model_name.lower(),
        spec.slug.lower(),
        spec.slug.replace("qwen3.5-", ""),
        spec.slug.replace(".", ""),
        spec.model_name.split("/")[-1].lower(),
    )
}


@dataclass(frozen=True)
class EvaluationSettings:
    provider: str
    base_url: str | None
    api_key_env: str
    timeout_seconds: int
    max_output_tokens: int
    safe_margin_tokens: int
    batch_size: int
    replace_existing: bool
    save_raw_response: bool
    tokenizer_name: str | None
    model_specs: tuple[ModelSpec, ...]


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
                max_model_len=262144,
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
    timeout_seconds: int | None = None,
    max_output_tokens: int | None = None,
    safe_margin_tokens: int | None = None,
) -> EvaluationSettings:
    resolved_provider = str(provider or DEFAULT_PROVIDER).strip().lower()
    resolved_base_url = resolve_llm_base_url(
        resolved_provider,
        base_url if base_url is not None else base_config.llm.base_url,
    )
    return EvaluationSettings(
        provider=resolved_provider,
        base_url=resolved_base_url,
        api_key_env=str(api_key_env or base_config.llm.api_key_env),
        timeout_seconds=int(
            base_config.llm.timeout_seconds if timeout_seconds is None else timeout_seconds
        ),
        max_output_tokens=int(
            DEFAULT_MAX_OUTPUT_TOKENS if max_output_tokens is None else max_output_tokens
        ),
        safe_margin_tokens=int(
            DEFAULT_SAFE_MARGIN_TOKENS if safe_margin_tokens is None else safe_margin_tokens
        ),
        batch_size=DEFAULT_BATCH_SIZE,
        replace_existing=(
            bool(base_config.runtime.replace_existing_patient_output)
            if replace_existing is None
            else bool(replace_existing)
        ),
        save_raw_response=bool(base_config.runtime.save_raw_response),
        tokenizer_name=base_config.llm.tokenizer_name,
        model_specs=resolve_model_specs(models),
    )


def build_evaluation_paths(patient_root: Path) -> EvaluationPaths:
    evaluation_root = patient_root / DEFAULT_OUTPUT_DIR_NAME
    return EvaluationPaths(
        patient_root=patient_root,
        evaluation_root=evaluation_root,
        comparison_dir=evaluation_root / "comparison",
        config_json=evaluation_root / "config.json",
        context_stats_json=evaluation_root / "context_stats.json",
        benchmark_snapshot_json=evaluation_root / "benchmark_snapshot.json",
    )


def build_model_artifact_paths(paths: EvaluationPaths, model_spec: ModelSpec) -> ModelArtifactPaths:
    model_dir = paths.model_dir(model_spec.slug)
    return ModelArtifactPaths(
        model_dir=model_dir,
        run_config_json=model_dir / "run_config.json",
        question_batches_json=model_dir / "question_batches.json",
        raw_predictions_jsonl=model_dir / "raw_predictions.jsonl",
        scored_predictions_jsonl=model_dir / "scored_predictions.jsonl",
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
