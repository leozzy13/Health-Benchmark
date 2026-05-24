from __future__ import annotations

from typing import Any


def render_full_patient_context(combined_payload: dict[str, Any]) -> str:
    return render_patient_context_recent_turns(combined_payload, max_recent_turns=None)


def render_patient_context_recent_turns(
    combined_payload: dict[str, Any],
    *,
    max_recent_turns: int | None,
) -> str:
    admissions = combined_payload["admissions"]
    total_turns = count_total_turns(combined_payload)
    first_selected_turn_index = 0
    if max_recent_turns is not None:
        first_selected_turn_index = max(0, total_turns - max(0, int(max_recent_turns)))
    rendered: list[str] = []
    global_turn_index = 0
    for admission in admissions:
        for line in admission["conversation_lines"]:
            if global_turn_index >= first_selected_turn_index:
                rendered.append(render_conversation_line(line))
            global_turn_index += 1
    return "\n".join(rendered)


def count_total_turns(combined_payload: dict[str, Any]) -> int:
    return sum(len(admission["conversation_lines"]) for admission in combined_payload["admissions"])


def render_conversation_line(line: dict[str, Any]) -> str:
    return f"{line['time']} | {line['speaker']} | {line['text']}"


def render_conversation_lines(lines: list[dict[str, Any]]) -> str:
    return "\n".join(render_conversation_line(line) for line in lines)
