from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .utils import canonical_json_dumps, parse_dt


SYSTEM_MESSAGE = """You are generating a synthetic but clinically grounded
conversation for exactly one hospital admission.

Rules:
1. Use only facts supported by the provided admission context.
2. The discharge notes are the backbone of the hospital-course
   narrative. Use them first.
3. The ordered problem list should shape the main clinical focus.
4. Radiology, procedures, microbiology, and the previous-admission
   summary are supporting context, not a checklist.
5. Not every packet item needs to appear in the conversation.
6. Speakers must be exactly: Doctor and Patient.
7. Every conversation line must contain:
   - turn number
   - exact timestamp
   - speaker
   - text
8. Timestamps must be monotonically increasing and stay within the
   admission start and admission end times.
9. Keep the conversation medically plausible and understandable to
   a patient.
10. Use short natural exchanges, patient questions, clarifications,
    explanations, reassurance, and follow-up discussion.
11. Do not invent unsupported diagnoses, procedures, test findings,
    or outcomes.
12. Do not mention the dataset, packet, note types, or that the
    conversation is synthetic.
13. Generate the conversation first, then write the summary strictly from the generated conversation_lines.
14. Every claim in summary.summary_paragraph and every item in summary.problems must be explicitly supported by the conversation_lines.
15. Return output that exactly matches the required JSON schema. """


OUTPUT_SCHEMA_REQUIREMENT = """Return JSON with this shape:
{
  "conversation_lines": [
    {
      "turn_number": 1,
      "time": "YYYY-MM-DD HH:MM:SS",
      "speaker": "Doctor",
      "text": "..."
    }
  ],
  "summary": {
    "admission_start": "YYYY-MM-DD HH:MM:SS",
    "admission_end": "YYYY-MM-DD HH:MM:SS",
    "summary_paragraph": "...",
    "problems": ["...", "..."]
  }
}"""


REPAIR_MESSAGE = """The previous response failed validation.
Return corrected JSON only. Keep the same schema and fix all speaker,
timestamp, chronology, and summary issues."""


@dataclass
class RenderedPrompt:
    system_message: str
    task_message: str
    user_message: str
    previous_summary_json: str
    packet_json: str
    stay_days: int
    recommended_turn_range: list[int]
    storyline_prompt_block: str
    problem_prompt_block: str
    supporting_context_block: str


def render_prompt(packet: dict[str, Any], previous_admission_summary: dict[str, Any] | None) -> RenderedPrompt:
    previous_summary_json = canonical_json_dumps(previous_admission_summary) if previous_admission_summary else "null"
    packet_json = canonical_json_dumps(packet)
    stay_days = _compute_stay_days(packet)
    recommended_turn_range = _recommended_turn_range(packet)
    storyline_prompt_block = _build_storyline_prompt_block(packet)
    problem_prompt_block = _build_problem_prompt_block(packet)
    supporting_context_block = _build_supporting_context_block(packet)
    task_message = _build_task_message(stay_days, recommended_turn_range)
    user_message = "\n\n".join(
        [
            task_message,
            "Previous admission summary:",
            previous_summary_json,
            "Admission storyline from discharge notes:",
            storyline_prompt_block,
            "Active admission problems:",
            problem_prompt_block,
            "Supporting context:",
            supporting_context_block,
            "Full admission packet for backup context only:",
            packet_json,
            OUTPUT_SCHEMA_REQUIREMENT,
        ]
    )
    return RenderedPrompt(
        system_message=SYSTEM_MESSAGE,
        task_message=task_message,
        user_message=user_message,
        previous_summary_json=previous_summary_json,
        packet_json=packet_json,
        stay_days=stay_days,
        recommended_turn_range=recommended_turn_range,
        storyline_prompt_block=storyline_prompt_block,
        problem_prompt_block=problem_prompt_block,
        supporting_context_block=supporting_context_block,
    )


def append_repair_message(user_message: str, validation_error: str) -> str:
    return "\n\n".join([user_message, REPAIR_MESSAGE, f"Validation errors: {validation_error}"])


def _build_task_message(stay_days: int, recommended_turn_range: list[int]) -> str:
    return "\n".join(
        [
            "Generate one rich patient-doctor conversation for this admission and one admission summary.",
            "",
            "Generation guidance:",
            f"- Approximate stay length: {stay_days} day(s).",
            f"- Recommended turn range: {recommended_turn_range[0]}-{recommended_turn_range[1]} turns.",
            "- Focus first on the discharge-note storyline and the core admission problems.",
            "- Use radiology, procedures, and microbiology only as supporting detail when they help the story.",
            "- Do not try to mention every packet item.",
            "- Let the conversation unfold across admission, mid-course updates, and discharge planning.",
            "- Prefer many short natural exchanges over a few long summary turns.",
            "- The patient should ask questions, express concerns, seek clarification, and respond to updates throughout the stay.",
            "- The doctor should explain results, treatment changes, progress, and follow-up in multiple conversational beats over time.",
            "- You may paraphrase and add natural conversational glue, but do not invent unsupported diagnoses, procedures, findings, or outcomes.",
            "- Generate the conversation first, then derive the final summary only from the conversation_lines you wrote.",
            "- The final summary must contain admission start time, admission end time, one paragraph summary, and a list of problems.",
        ]
    )


def _compute_stay_days(packet: dict[str, Any]) -> int:
    admission_start = parse_dt(packet.get("admission_start"))
    admission_end = parse_dt(packet.get("admission_end"))
    if admission_start is None or admission_end is None:
        return 1
    duration_seconds = max(0.0, (admission_end - admission_start).total_seconds())
    return max(1, math.ceil(duration_seconds / 86400.0))


def _recommended_turn_range(packet: dict[str, Any]) -> list[int]:
    admission_start = parse_dt(packet.get("admission_start"))
    admission_end = parse_dt(packet.get("admission_end"))
    if admission_start is None or admission_end is None:
        return [18, 28]
    duration_days = max(0.0, (admission_end - admission_start).total_seconds()) / 86400.0
    if duration_days < 1.0:
        return [18, 28]
    if duration_days <= 3.0:
        return [40, 60]
    if duration_days <= 7.0:
        return [50, 70]
    return [60, 85]


def _build_storyline_prompt_block(packet: dict[str, Any]) -> str:
    notes = packet.get("discharge_notes", [])
    if not notes:
        return "No discharge-note storyline is available."
    lines: list[str] = []
    for index, note in enumerate(notes, start=1):
        timestamp = note.get("charttime") or note.get("storetime") or "unknown time"
        text = str(note.get("text") or "").strip()
        if not text:
            continue
        lines.append(f"Discharge note {index} ({timestamp}): {text}")
    return "\n".join(lines) if lines else "No discharge-note storyline is available."


def _build_problem_prompt_block(packet: dict[str, Any]) -> str:
    titles = _ordered_unique_strings(item.get("long_title") for item in packet.get("diagnoses", []))
    if not titles:
        return "- No diagnoses recorded."
    return "\n".join(f"- {title}" for title in titles)


def _build_supporting_context_block(packet: dict[str, Any]) -> str:
    sections = [
        ("Radiology", _build_radiology_lines(packet)),
        ("Procedures", _build_procedure_lines(packet)),
        ("Microbiology", _build_microbiology_lines(packet)),
    ]
    rendered_sections: list[str] = []
    for title, lines in sections:
        section_body = "\n".join(lines) if lines else "- None recorded."
        rendered_sections.append(f"{title}:\n{section_body}")
    return "\n\n".join(rendered_sections)


def _build_radiology_lines(packet: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for note in packet.get("radiology_notes", [])[:8]:
        timestamp = note.get("charttime") or note.get("storetime") or "unknown time"
        text = _trim_text(str(note.get("text") or ""), 220)
        if text:
            lines.append(f"- {timestamp}: {text}")
    return lines


def _build_procedure_lines(packet: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for item in packet.get("procedures", [])[:10]:
        title = str(item.get("long_title") or "").strip()
        if not title:
            continue
        timestamp = item.get("normalized_time") or item.get("chartdate") or "unknown time"
        lines.append(f"- {timestamp}: {title}")
    return lines


def _build_microbiology_lines(packet: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for item in packet.get("microbiology", [])[:10]:
        timestamp = item.get("normalized_time") or item.get("charttime") or item.get("chartdate") or "unknown time"
        specimen = str(item.get("spec_type_desc") or "").strip()
        organism = str(item.get("org_name") or "").strip()
        interpretation = str(item.get("interpretation") or "").strip()
        comments = str(item.get("comments") or "").strip()
        details = []
        if specimen:
            details.append(specimen)
        if organism:
            details.append(organism)
        detail_text = " | ".join(details) if details else "microbiology result"
        suffix_parts = [part for part in [interpretation, comments] if part]
        suffix = f" ({'; '.join(suffix_parts)})" if suffix_parts else ""
        lines.append(f"- {timestamp}: {detail_text}{suffix}")
    return lines


def _ordered_unique_strings(values: Any) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        normalized = text.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(text)
    return ordered


def _trim_text(text: str, limit: int) -> str:
    stripped = " ".join((text or "").split())
    if len(stripped) <= limit:
        return stripped
    return stripped[: limit - 3].rstrip() + "..."
