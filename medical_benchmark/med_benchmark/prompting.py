from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import BenchmarkConfig
from .utils import canonical_json_dumps


SYSTEM_MESSAGE = """You are simulating a realistic inpatient doctor–patient conversation for a single hospital admission.
You MUST follow these rules:

1) Use only the facts provided in the EHR packet. Do NOT invent diagnoses, symptoms, test results, medications, procedures, timelines, or outcomes.
2) Every clinically factual statement MUST be supported by an explicit evidence citation using the provided EIDs (e.g., "Evidence: [LAB#000123, RAD#000004]").
3) If the EHR packet does not contain a fact, say you do not know or that it is not documented.
4) Produce a chronological conversation spanning the full admission, from admission through discharge.
5) Use RELATIVE time only (e.g., "H+03:15", "HospitalDay2 09:40"). Do not mention calendar dates or years.
6) Discharge summaries are provided; treat them as post-hoc summaries that help you maintain coherence, but do not reveal discharge outcomes before they occur in the chronology.
7) Output must be valid JSON, conforming exactly to the output schema described in the user message. No extra keys."""


TASK_BLOCK = """Generate a multi-turn inpatient conversation for this admission session.

Hard requirements:
- Discharge note text(s) from notes.discharge MUST influence the conversation and MUST be cited when used.
- The conversation must be medically and temporally coherent with the structured data (labs, orders, service changes, radiology).
- Use a mix of speakers typical for an inpatient stay: PATIENT, ATTENDING, RESIDENT, NURSE (optional), CONSULT (optional).
- Do not copy the discharge summary verbatim as dialogue. Use it to guide what happened.

Evidence rules:
- For each turn, include a list of EIDs supporting that turn’s medical content.
- If a turn contains only social talk (e.g., greeting), evidence may be [].

Output JSON schema (must match exactly):
{
  "conversation": [
    {
      "turn_id": integer (starts at 1),
      "speaker": "PATIENT" | "ATTENDING" | "RESIDENT" | "NURSE" | "CONSULT",
      "relative_time": string,
      "text": string,
      "evidence_eids": [string, ...]
    }
  ],
  "end_of_admission_summary": {
    "relative_discharge_time": string,
    "one_paragraph_summary": string,
    "problem_list": [
      {
        "problem": string,
        "status_at_discharge": string,
        "supporting_eids": [string, ...]
      }
    ],
    "key_tests_and_results": [
      {
        "test": string,
        "result": string,
        "relative_time": string,
        "supporting_eids": [string, ...]
      }
    ],
    "treatments_and_meds": [
      {
        "treatment_or_med": string,
        "details": string,
        "supporting_eids": [string, ...]
      }
    ],
    "disposition": {
      "discharge_location": string,
      "supporting_eids": [string, ...]
    }
  }
}"""


REPAIR_BLOCK = """<<REPAIR>>
Your previous response was invalid JSON or violated the required schema.
Return valid JSON exactly matching the schema. No prose, no markdown, no comments.
<<END_REPAIR>>"""


@dataclass(slots=True)
class RenderedPrompt:
    system_message: str
    user_message: str
    packet_canonical_json: str


def render_prompt(
    packet: dict[str, Any],
    previous_admission_summary: str,
    config: BenchmarkConfig,
) -> RenderedPrompt:
    d = config.prompt.delimiters
    packet_canonical = canonical_json_dumps(packet)
    meta_lines = [
        f"benchmark_name: {config.benchmark_name}",
        f"benchmark_version: {config.benchmark_version}",
        f"packet_schema_version: {config.packet_schema_version}",
        f"prompt_template_version: {config.prompt.template_version}",
        f"subject_id: {packet['ids']['subject_id']}",
        f"hadm_id: {packet['ids']['hadm_id']}",
    ]
    user_parts = [
        d["metadata"],
        *meta_lines,
        d["metadata_end"],
        "",
        d["prev_summary"],
        previous_admission_summary or "",
        d["prev_summary_end"],
        "",
        d["ehr_json"],
        packet_canonical,
        d["ehr_json_end"],
        "",
        d["task"],
        TASK_BLOCK,
        d["task_end"],
    ]
    user_message = "\n".join(user_parts)
    return RenderedPrompt(
        system_message=SYSTEM_MESSAGE,
        user_message=user_message,
        packet_canonical_json=packet_canonical,
    )


def append_repair_block(user_message: str) -> str:
    return f"{user_message}\n\n{REPAIR_BLOCK}"
