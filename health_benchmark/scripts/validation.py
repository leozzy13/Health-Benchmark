from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError as PydanticValidationError

from .utils import EXACT_TS_FORMAT, parse_dt


class ValidationError(ValueError):
    pass


class ConversationLine(BaseModel):
    model_config = ConfigDict(extra="forbid")

    turn_number: int
    time: str
    speaker: Literal["Doctor", "Patient"]
    text: str


class AdmissionSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    admission_start: str
    admission_end: str
    summary_paragraph: str
    problems: list[str] = Field(default_factory=list)


class GenerationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conversation_lines: list[ConversationLine]
    summary: AdmissionSummary


def validate_generation(output: GenerationOutput | dict[str, Any], packet: dict[str, Any]) -> dict[str, Any]:
    try:
        data = output.model_dump(mode="json") if isinstance(output, GenerationOutput) else GenerationOutput.model_validate(output).model_dump(mode="json")
    except PydanticValidationError as exc:
        raise ValidationError(str(exc)) from exc

    conversation_lines = data["conversation_lines"]
    if not conversation_lines:
        raise ValidationError("conversation_lines must be non-empty")

    admission_start = _parse_exact_timestamp(packet["admission_start"], "packet.admission_start")
    admission_end = _parse_exact_timestamp(packet["admission_end"], "packet.admission_end")

    expected_turn = 1
    previous_time: datetime | None = None
    for line in conversation_lines:
        if line["turn_number"] != expected_turn:
            raise ValidationError(
                f"turn_number must be contiguous starting at 1; expected {expected_turn}, got {line['turn_number']}"
            )
        expected_turn += 1
        if not str(line["text"]).strip():
            raise ValidationError(f"conversation line {line['turn_number']} has empty text")
        current_time = _parse_exact_timestamp(line["time"], f"conversation_lines[{line['turn_number']}].time")
        if previous_time is not None and current_time < previous_time:
            raise ValidationError("conversation timestamps must be monotonically increasing")
        if current_time < admission_start or current_time > admission_end:
            raise ValidationError("conversation timestamp is outside the admission window")
        previous_time = current_time

    summary = data["summary"]
    if summary["admission_start"] != packet["admission_start"]:
        raise ValidationError("summary.admission_start must match the packet admission_start exactly")
    if summary["admission_end"] != packet["admission_end"]:
        raise ValidationError("summary.admission_end must match the packet admission_end exactly")
    if not str(summary["summary_paragraph"]).strip():
        raise ValidationError("summary.summary_paragraph must be non-empty")
    if "\n" in str(summary["summary_paragraph"]).strip():
        raise ValidationError("summary.summary_paragraph must be exactly one paragraph")
    if not isinstance(summary["problems"], list):
        raise ValidationError("summary.problems must be a list")
    for index, problem in enumerate(summary["problems"], start=1):
        if not isinstance(problem, str) or not problem.strip():
            raise ValidationError(f"summary.problems[{index}] must be a non-empty string")

    return data


def _parse_exact_timestamp(value: str, path: str) -> datetime:
    try:
        return datetime.strptime(value, EXACT_TS_FORMAT)
    except ValueError as exc:
        raise ValidationError(f"{path} must match {EXACT_TS_FORMAT}") from exc


def parse_generation_response(value: Any) -> dict[str, Any]:
    if isinstance(value, GenerationOutput):
        return value.model_dump(mode="json")
    try:
        return GenerationOutput.model_validate(value).model_dump(mode="json")
    except PydanticValidationError as exc:
        raise ValidationError(str(exc)) from exc


def parse_packet_time(value: str) -> datetime | None:
    return parse_dt(value)
