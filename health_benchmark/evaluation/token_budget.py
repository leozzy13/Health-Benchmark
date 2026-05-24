from __future__ import annotations

from math import ceil, floor
from typing import Any, Sequence

from .answer_prompting import render_answer_prompt
from .context_renderer import count_total_turns, render_full_patient_context, render_patient_context_recent_turns
from .hf_tokenizer import count_chat_prompt_tokens
from .types import EvalQuestion, PromptTokenEstimate


def adjusted_prompt_tokens(raw_token_count: int, safety_multiplier: float) -> int:
    return int(ceil(int(raw_token_count) * float(safety_multiplier)))


def effective_raw_prompt_budget(effective_budget: int, safety_multiplier: float) -> int:
    return int(floor(int(effective_budget) / float(safety_multiplier)))


def estimate_prompt_tokens(
    *,
    model_name: str,
    tokenizer_name: str | None,
    system_message: str,
    user_message: str,
    enable_thinking: bool = False,
) -> PromptTokenEstimate:
    token_count, tokenizer_model = count_chat_prompt_tokens(
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        system_message=system_message,
        user_message=user_message,
        enable_thinking=enable_thinking,
    )
    return PromptTokenEstimate(
        total_tokens=int(token_count),
        encoding_name=str(tokenizer_model),
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
    token_estimate_safety_multiplier: float,
    enable_thinking: bool = False,
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
        enable_thinking=enable_thinking,
    )
    effective_budget = int(max_model_len - max_output_tokens - safe_margin_tokens)
    adjusted_estimate = adjusted_prompt_tokens(
        estimate.total_tokens,
        token_estimate_safety_multiplier,
    )
    status = "full_context_fits" if adjusted_estimate <= effective_budget else "truncation_required"
    return {
        "status": status,
        "batch_size": int(batch_size),
        "sample_qa_ids": [question.qa_id for question in longest_questions],
        "estimated_full_prompt_tokens": int(estimate.total_tokens),
        "adjusted_estimated_full_prompt_tokens": int(adjusted_estimate),
        "tokenizer": estimate.encoding_name,
        "max_model_len": int(max_model_len),
        "max_output_tokens": int(max_output_tokens),
        "safe_margin_tokens": int(safe_margin_tokens),
        "token_estimate_safety_multiplier": float(token_estimate_safety_multiplier),
        "effective_budget_tokens": int(effective_budget),
        "effective_raw_prompt_budget_tokens": effective_raw_prompt_budget(
            effective_budget,
            token_estimate_safety_multiplier,
        ),
    }


def select_context_for_batch(
    *,
    combined_payload: dict[str, Any],
    batch_questions: Sequence[EvalQuestion],
    model_name: str,
    tokenizer_name: str | None,
    max_model_len: int,
    max_output_tokens: int,
    safe_margin_tokens: int,
    token_estimate_safety_multiplier: float,
    enable_thinking: bool = False,
) -> dict[str, Any]:
    effective_budget = int(max_model_len - max_output_tokens - safe_margin_tokens)
    full_context = render_full_patient_context(combined_payload)
    full_estimate = _estimate_for_questions(
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        context_text=full_context,
        questions=batch_questions,
        enable_thinking=enable_thinking,
    )
    adjusted_full_estimate = adjusted_prompt_tokens(
        full_estimate.total_tokens,
        token_estimate_safety_multiplier,
    )
    total_turns = count_total_turns(combined_payload)
    base_record: dict[str, Any] = {
        "strategy": "full_context",
        "was_truncated": False,
        "selected_turns": int(total_turns),
        "omitted_turns": 0,
        "total_turns": int(total_turns),
        "estimated_full_prompt_tokens": int(full_estimate.total_tokens),
        "adjusted_estimated_full_prompt_tokens": int(adjusted_full_estimate),
        "estimated_prompt_tokens": int(full_estimate.total_tokens),
        "adjusted_estimated_prompt_tokens": int(adjusted_full_estimate),
        "tokenizer": full_estimate.encoding_name,
        "max_model_len": int(max_model_len),
        "max_output_tokens": int(max_output_tokens),
        "safe_margin_tokens": int(safe_margin_tokens),
        "token_estimate_safety_multiplier": float(token_estimate_safety_multiplier),
        "effective_prompt_budget_tokens": int(effective_budget),
        "effective_raw_prompt_budget_tokens": effective_raw_prompt_budget(
            effective_budget,
            token_estimate_safety_multiplier,
        ),
    }
    if adjusted_full_estimate <= effective_budget:
        return {
            "context_text": full_context,
            "estimated_prompt_tokens": int(full_estimate.total_tokens),
            "adjusted_estimated_prompt_tokens": int(adjusted_full_estimate),
            "context_record": base_record,
        }

    empty_estimate = _estimate_for_questions(
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        context_text="",
        questions=batch_questions,
        enable_thinking=enable_thinking,
    )
    adjusted_empty_estimate = adjusted_prompt_tokens(
        empty_estimate.total_tokens,
        token_estimate_safety_multiplier,
    )
    if adjusted_empty_estimate > effective_budget:
        record = {
            **base_record,
            "strategy": "no_context_prompt_too_large",
            "was_truncated": True,
            "selected_turns": 0,
            "omitted_turns": int(total_turns),
            "estimated_prompt_tokens": int(empty_estimate.total_tokens),
            "adjusted_estimated_prompt_tokens": int(adjusted_empty_estimate),
        }
        return {
            "context_text": "",
            "estimated_prompt_tokens": int(empty_estimate.total_tokens),
            "adjusted_estimated_prompt_tokens": int(adjusted_empty_estimate),
            "context_record": record,
        }

    low = 0
    high = int(total_turns)
    best_turn_count = 0
    best_context = ""
    best_estimate = empty_estimate
    best_adjusted_estimate = adjusted_empty_estimate
    while low <= high:
        midpoint = (low + high) // 2
        candidate_context = render_patient_context_recent_turns(
            combined_payload,
            max_recent_turns=midpoint,
        )
        candidate_estimate = _estimate_for_questions(
            model_name=model_name,
            tokenizer_name=tokenizer_name,
            context_text=candidate_context,
            questions=batch_questions,
            enable_thinking=enable_thinking,
        )
        adjusted_candidate_estimate = adjusted_prompt_tokens(
            candidate_estimate.total_tokens,
            token_estimate_safety_multiplier,
        )
        if adjusted_candidate_estimate <= effective_budget:
            best_turn_count = int(midpoint)
            best_context = candidate_context
            best_estimate = candidate_estimate
            best_adjusted_estimate = adjusted_candidate_estimate
            low = midpoint + 1
        else:
            high = midpoint - 1

    record = {
        **base_record,
        "strategy": "recent_first",
        "was_truncated": True,
        "selected_turns": int(best_turn_count),
        "omitted_turns": int(max(0, total_turns - best_turn_count)),
        "estimated_prompt_tokens": int(best_estimate.total_tokens),
        "adjusted_estimated_prompt_tokens": int(best_adjusted_estimate),
    }
    return {
        "context_text": best_context,
        "estimated_prompt_tokens": int(best_estimate.total_tokens),
        "adjusted_estimated_prompt_tokens": int(best_adjusted_estimate),
        "context_record": record,
    }


def _estimate_for_questions(
    *,
    model_name: str,
    tokenizer_name: str | None,
    context_text: str,
    questions: Sequence[EvalQuestion],
    enable_thinking: bool = False,
) -> PromptTokenEstimate:
    rendered = render_answer_prompt(
        context_text=context_text,
        questions=[question.model_question() for question in questions],
    )
    return estimate_prompt_tokens(
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        system_message=rendered.system_message,
        user_message=rendered.user_message,
        enable_thinking=enable_thinking,
    )
