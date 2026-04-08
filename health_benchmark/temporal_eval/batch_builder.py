from __future__ import annotations

from .answer_prompting import render_answer_prompt
from .config import EvaluationSettings
from .token_budget import estimate_prompt_tokens
from .types import EvalQuestion, QuestionBatch


def build_batches(
    questions: list[EvalQuestion],
    *,
    context_text: str,
    settings: EvaluationSettings,
    model_name: str,
) -> list[QuestionBatch]:
    if not questions:
        return []
    batches: list[QuestionBatch] = []
    for start in range(0, len(questions), settings.batch_size):
        batch_questions = questions[start : start + settings.batch_size]
        rendered = render_answer_prompt(
            context_text=context_text,
            questions=[question.model_question() for question in batch_questions],
        )
        estimate = estimate_prompt_tokens(
            model_name=model_name,
            tokenizer_name=settings.tokenizer_name,
            system_message=rendered.system_message,
            user_message=rendered.user_message,
        )
        batches.append(
            QuestionBatch(
                batch_id=f"batch_{(start // settings.batch_size) + 1:03d}",
                questions=list(batch_questions),
                estimated_prompt_tokens=int(estimate.total_tokens),
            )
        )
    return batches
