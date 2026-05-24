from __future__ import annotations

from .config import EvaluationSettings
from .token_budget import select_context_for_batch
from .types import EvalQuestion, ModelSpec, QuestionBatch


def build_batches(
    questions: list[EvalQuestion],
    *,
    combined_payload: dict[str, object],
    settings: EvaluationSettings,
    model_spec: ModelSpec,
) -> list[QuestionBatch]:
    if not questions:
        return []
    batches: list[QuestionBatch] = []
    for start in range(0, len(questions), settings.batch_size):
        batch_questions = questions[start : start + settings.batch_size]
        context_selection = select_context_for_batch(
            combined_payload=combined_payload,
            batch_questions=batch_questions,
            model_name=model_spec.model_name,
            tokenizer_name=settings.tokenizer_name,
            max_model_len=model_spec.max_model_len,
            max_output_tokens=settings.max_output_tokens,
            safe_margin_tokens=settings.safe_margin_tokens,
            token_estimate_safety_multiplier=settings.token_estimate_safety_multiplier,
            enable_thinking=settings.enable_thinking,
        )
        batches.append(
            QuestionBatch(
                batch_id=f"batch_{(start // settings.batch_size) + 1:03d}",
                questions=list(batch_questions),
                estimated_prompt_tokens=int(context_selection["estimated_prompt_tokens"]),
                adjusted_estimated_prompt_tokens=int(context_selection["adjusted_estimated_prompt_tokens"]),
                context_text=str(context_selection["context_text"]),
                context_record=dict(context_selection["context_record"]),
            )
        )
    return batches
