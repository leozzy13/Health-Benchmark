from __future__ import annotations

from collections import defaultdict
from typing import Any


def build_comparison_markdown(
    subject_id: str,
    leaderboard_rows: list[dict[str, Any]],
    breakdown_rows: list[dict[str, Any]] | None = None,
) -> str:
    lines = [
        f"# Evaluation Summary for {subject_id}",
        "",
        "| Model | Status | Overall | LLM Score | Answerable F1 | Adversarial Acc |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in leaderboard_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["model_slug"]),
                    str(row["run_status"]),
                    f"{float(row['overall_score']):.4f}",
                    f"{float(row['llm_score']):.4f}",
                    f"{float(row['macro_f1_answerable']):.4f}",
                    f"{float(row['adversarial_accuracy']):.4f}",
                ]
            )
            + " |"
        )
    grouped_breakdowns: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in breakdown_rows or []:
        grouped_breakdowns[str(row["breakdown"])].append(row)
    for breakdown_name in (
        "by_answerability",
        "by_scope",
        "by_adversarial_scope",
        "by_answerable_scope",
        "by_question_type",
    ):
        rows = grouped_breakdowns.get(breakdown_name, [])
        if not rows:
            continue
        lines.extend(
            [
                "",
                f"## {breakdown_name}",
                "",
                "| Group | Model | Count | Overall | LLM Score | Answerable F1 | Adversarial Acc |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["group"]),
                        str(row["model_slug"]),
                        str(row["count"]),
                        f"{float(row['overall_score']):.4f}",
                        f"{float(row['llm_score']):.4f}",
                        f"{float(row['macro_f1_answerable']):.4f}",
                        f"{float(row['adversarial_accuracy']):.4f}",
                    ]
                )
                + " |"
            )
    return "\n".join(lines) + "\n"
