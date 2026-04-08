from __future__ import annotations

from typing import Any


def build_comparison_markdown(subject_id: str, leaderboard_rows: list[dict[str, Any]]) -> str:
    lines = [
        f"# Evaluation Summary for {subject_id}",
        "",
        "| Model | Status | Overall | Answerable F1 | Adversarial Acc |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in leaderboard_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["model_slug"]),
                    str(row["run_status"]),
                    f"{float(row['overall_score']):.4f}",
                    f"{float(row['macro_f1_answerable']):.4f}",
                    f"{float(row['adversarial_accuracy']):.4f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"
