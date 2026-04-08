from __future__ import annotations

from typing import Any


def render_full_patient_context(combined_payload: dict[str, Any]) -> str:
    admissions = combined_payload["admissions"]
    rendered: list[str] = []
    for index, admission in enumerate(admissions, start=1):
        rendered.append(
            (
                f"=== Admission {index} | hadm_id={admission['hadm_id']} | "
                f"start={admission['admission_start']} | end={admission['admission_end']} ==="
            )
        )
        for line in admission["conversation_lines"]:
            rendered.append(
                f"{line['turn_number']} | {line['time']} | {line['speaker']} | {line['text']}"
            )
    return "\n".join(rendered)


def count_total_turns(combined_payload: dict[str, Any]) -> int:
    return sum(len(admission["conversation_lines"]) for admission in combined_payload["admissions"])
