from __future__ import annotations

import os
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence


def count_chat_prompt_tokens(
    *,
    model_name: str,
    tokenizer_name: str | None,
    system_message: str,
    user_message: str,
    enable_thinking: bool,
) -> tuple[int, str]:
    tokenizer_model = _resolve_tokenizer_model(model_name, tokenizer_name)
    tokenizer = _load_hf_tokenizer(tokenizer_model)
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]
    rendered = _apply_chat_template(
        tokenizer,
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=bool(enable_thinking),
    )
    token_ids = _encode_text(tokenizer, str(rendered))
    return int(len(token_ids)), tokenizer_model


def count_text_tokens(
    *,
    model_name: str,
    tokenizer_name: str | None,
    text: str,
) -> tuple[int, str]:
    tokenizer_model = _resolve_tokenizer_model(model_name, tokenizer_name)
    tokenizer = _load_hf_tokenizer(tokenizer_model)
    token_ids = _encode_text(tokenizer, str(text or ""))
    return int(len(token_ids)), tokenizer_model


def count_batch_text_tokens(
    *,
    model_name: str,
    tokenizer_name: str | None,
    texts: Sequence[str],
) -> tuple[list[int], str]:
    tokenizer_model = _resolve_tokenizer_model(model_name, tokenizer_name)
    tokenizer = _load_hf_tokenizer(tokenizer_model)
    counts = [
        int(len(_encode_text(tokenizer, str(text or ""))))
        for text in texts
    ]
    return counts, tokenizer_model


def _resolve_tokenizer_model(model_name: str, tokenizer_name: str | None) -> str:
    candidate = str(tokenizer_name or model_name or "").strip()
    if not candidate:
        raise ValueError("A model_name or tokenizer_name is required for HF token counting.")
    return candidate


@lru_cache(maxsize=16)
def _load_hf_tokenizer(tokenizer_model: str) -> Any:
    try:
        from transformers import AutoProcessor, AutoTokenizer  # type: ignore
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "transformers is required for evaluation token counting. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    cache_dir = _resolve_hf_cache_dir()
    load_target = _resolve_local_snapshot(tokenizer_model, cache_dir) or tokenizer_model
    kwargs: dict[str, Any] = {
        "local_files_only": True,
        "trust_remote_code": True,
    }
    if cache_dir is not None and not Path(str(load_target)).exists():
        kwargs["cache_dir"] = str(cache_dir)
    try:
        return AutoTokenizer.from_pretrained(load_target, **kwargs)
    except Exception as tokenizer_exc:  # pragma: no cover - depends on local HF cache state
        try:
            return AutoProcessor.from_pretrained(load_target, **kwargs)
        except Exception as processor_exc:
            cache_hint = f" in cache_dir={cache_dir}" if cache_dir is not None else ""
            raise RuntimeError(
                f"Could not load local Hugging Face tokenizer or processor for {tokenizer_model!r}{cache_hint}. "
                "Make sure the model tokenizer is available in the configured HF cache."
            ) from processor_exc


def _apply_chat_template(
    tokenizer_or_processor: Any,
    messages: list[dict[str, str]],
    *,
    tokenize: bool,
    add_generation_prompt: bool,
    enable_thinking: bool,
) -> Any:
    template = getattr(tokenizer_or_processor, "apply_chat_template", None)
    if not callable(template):
        inner = getattr(tokenizer_or_processor, "tokenizer", None)
        template = getattr(inner, "apply_chat_template", None)
    if not callable(template):
        raise RuntimeError("Loaded Hugging Face tokenizer/processor does not expose apply_chat_template().")
    kwargs = {
        "tokenize": tokenize,
        "add_generation_prompt": add_generation_prompt,
        "enable_thinking": enable_thinking,
    }
    try:
        return template(messages, **kwargs)
    except TypeError as exc:
        if "enable_thinking" not in str(exc):
            raise
        kwargs.pop("enable_thinking", None)
        return template(messages, **kwargs)


def _encode_text(tokenizer_or_processor: Any, text: str) -> list[int]:
    encoder = getattr(tokenizer_or_processor, "encode", None)
    if callable(encoder):
        return list(encoder(text, add_special_tokens=False))
    inner = getattr(tokenizer_or_processor, "tokenizer", None)
    inner_encoder = getattr(inner, "encode", None)
    if callable(inner_encoder):
        return list(inner_encoder(text, add_special_tokens=False))
    caller = tokenizer_or_processor
    if not callable(caller) and callable(inner):
        caller = inner
    if callable(caller):
        encoded = caller(text, add_special_tokens=False)
        if isinstance(encoded, Mapping) and "input_ids" in encoded:
            return list(encoded["input_ids"])
    raise RuntimeError("Loaded Hugging Face tokenizer/processor cannot encode text for token counting.")


def _resolve_hf_cache_dir() -> Path | None:
    for env_name in ("HUGGINGFACE_HUB_CACHE", "HF_HUB_CACHE"):
        value = os.getenv(env_name)
        if value:
            return Path(value).expanduser().resolve()
    hf_home = os.getenv("HF_HOME")
    if hf_home:
        return (Path(hf_home).expanduser().resolve() / "hub").resolve()
    repo_cache = Path(__file__).resolve().parents[2] / "runtime" / "hf_cache" / "hub"
    if repo_cache.exists():
        return repo_cache.resolve()
    return None


def _resolve_local_snapshot(tokenizer_model: str, cache_dir: Path | None) -> str | None:
    candidate = Path(str(tokenizer_model)).expanduser()
    if candidate.exists():
        return str(candidate.resolve())
    if cache_dir is None or "/" not in str(tokenizer_model):
        return None
    model_cache = cache_dir / ("models--" + str(tokenizer_model).replace("/", "--"))
    snapshots_dir = model_cache / "snapshots"
    ref_path = model_cache / "refs" / "main"
    try:
        revision = ref_path.read_text(encoding="utf-8").strip()
    except OSError:
        revision = ""
    if revision:
        snapshot = snapshots_dir / revision
        if _snapshot_has_tokenizer_files(snapshot):
            return str(snapshot.resolve())
    try:
        snapshots = sorted(path for path in snapshots_dir.iterdir() if path.is_dir())
    except OSError:
        return None
    snapshots = [path for path in snapshots if _snapshot_has_tokenizer_files(path)]
    if len(snapshots) == 1:
        return str(snapshots[0].resolve())
    return None


def _snapshot_has_tokenizer_files(snapshot: Path) -> bool:
    if not snapshot.is_dir():
        return False
    tokenizer_files = (
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "spiece.model",
        "vocab.json",
        "vocab.txt",
        "merges.txt",
    )
    return any((snapshot / filename).is_file() for filename in tokenizer_files)
