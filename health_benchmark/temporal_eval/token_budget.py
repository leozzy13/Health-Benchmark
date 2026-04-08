from __future__ import annotations

from typing import Any, Sequence

from .answer_prompting import render_answer_prompt
from .types import EvalQuestion, PromptTokenEstimate


CHAT_MESSAGE_OVERHEAD_TOKENS = 256


def _import_tiktoken():
    try:
        import tiktoken  # type: ignore
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("tiktoken is required. Install dependencies in requirements.txt") from exc
    return tiktoken


def estimate_prompt_tokens(
    *,
    model_name: str,
    tokenizer_name: str | None,
    system_message: str,
    user_message: str,
) -> PromptTokenEstimate:
    tiktoken = _import_tiktoken()
    if tokenizer_name:
        try:
            encoding = tiktoken.get_encoding(tokenizer_name)
            encoded_length = len(encoding.encode(system_message)) + len(encoding.encode(user_message))
            return PromptTokenEstimate(
                total_tokens=int(encoded_length + CHAT_MESSAGE_OVERHEAD_TOKENS),
                encoding_name=str(tokenizer_name),
            )
        except KeyError:
            pass
    try:
        encoding = tiktoken.encoding_for_model(model_name)
        encoding_name = getattr(encoding, "name", model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("o200k_base")
        encoding_name = "o200k_base"
    encoded_length = len(encoding.encode(system_message)) + len(encoding.encode(user_message))
    return PromptTokenEstimate(
        total_tokens=int(encoded_length + CHAT_MESSAGE_OVERHEAD_TOKENS),
        encoding_name=str(encoding_name),
    )


def build_preflight_record(
    *,
    model_name: str,
    tokenizer_name: str | None,
    context_text: str,
    questions: Sequence[EvalQuestion],
    batch_size: int,
    max_model_len: int,
    max_output_tokens: int,
    safe_margin_tokens: int,
) -> dict[str, Any]:
    longest_questions = sorted(
        questions,
        key=lambda item: (len(item.question), item.qa_id),
        reverse=True,
    )[:batch_size]
    rendered = render_answer_prompt(
        context_text=context_text,
        questions=[question.model_question() for question in longest_questions],
    )
    estimate = estimate_prompt_tokens(
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        system_message=rendered.system_message,
        user_message=rendered.user_message,
    )
    effective_budget = int(max_model_len - max_output_tokens - safe_margin_tokens)
    return {
        "status": "ok" if estimate.total_tokens <= effective_budget else "context_too_long_for_fixed_batch_10",
        "batch_size": int(batch_size),
        "sample_qa_ids": [question.qa_id for question in longest_questions],
        "estimated_prompt_tokens": int(estimate.total_tokens),
        "tokenizer": estimate.encoding_name,
        "max_model_len": int(max_model_len),
        "max_output_tokens": int(max_output_tokens),
        "safe_margin_tokens": int(safe_margin_tokens),
        "effective_budget_tokens": int(effective_budget),
    }
