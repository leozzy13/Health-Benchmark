from __future__ import annotations

import json
from typing import Any

from .utils import flatten_eids


ALLOWED_SPEAKERS = {"PATIENT", "ATTENDING", "RESIDENT", "NURSE", "CONSULT"}


class ValidationError(ValueError):
    pass


def parse_json_response(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"Invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValidationError("Top-level response must be a JSON object.")
    return data


def validate_output_schema(data: dict[str, Any]) -> None:
    expected_top = {"conversation", "end_of_admission_summary"}
    extra = set(data.keys()) - expected_top
    missing = expected_top - set(data.keys())
    if extra:
        raise ValidationError(f"Unexpected top-level keys: {sorted(extra)}")
    if missing:
        raise ValidationError(f"Missing top-level keys: {sorted(missing)}")

    conversation = data["conversation"]
    if not isinstance(conversation, list) or not conversation:
        raise ValidationError("conversation must be a non-empty list.")
    expected_turn_id = 1
    for idx, turn in enumerate(conversation, start=1):
        if not isinstance(turn, dict):
            raise ValidationError(f"conversation[{idx}] must be an object.")
        required_keys = {"turn_id", "speaker", "relative_time", "text", "evidence_eids"}
        if set(turn.keys()) != required_keys:
            raise ValidationError(
                f"conversation[{idx}] keys mismatch. Expected {sorted(required_keys)}, got {sorted(turn.keys())}."
            )
        if not isinstance(turn["turn_id"], int):
            raise ValidationError(f"conversation[{idx}].turn_id must be int.")
        if turn["turn_id"] != expected_turn_id:
            raise ValidationError(
                f"conversation[{idx}].turn_id must be sequential starting at 1 (expected {expected_turn_id})."
            )
        expected_turn_id += 1
        if turn["speaker"] not in ALLOWED_SPEAKERS:
            raise ValidationError(f"conversation[{idx}].speaker invalid: {turn['speaker']}")
        if not isinstance(turn["relative_time"], str):
            raise ValidationError(f"conversation[{idx}].relative_time must be string.")
        if not isinstance(turn["text"], str):
            raise ValidationError(f"conversation[{idx}].text must be string.")
        if not isinstance(turn["evidence_eids"], list) or any(not isinstance(x, str) for x in turn["evidence_eids"]):
            raise ValidationError(f"conversation[{idx}].evidence_eids must be list[str].")

    summary = data["end_of_admission_summary"]
    if not isinstance(summary, dict):
        raise ValidationError("end_of_admission_summary must be an object.")
    required_summary_keys = {
        "relative_discharge_time",
        "one_paragraph_summary",
        "problem_list",
        "key_tests_and_results",
        "treatments_and_meds",
        "disposition",
    }
    if set(summary.keys()) != required_summary_keys:
        raise ValidationError(
            f"end_of_admission_summary keys mismatch. Expected {sorted(required_summary_keys)}, got {sorted(summary.keys())}."
        )
    if not isinstance(summary["relative_discharge_time"], str):
        raise ValidationError("end_of_admission_summary.relative_discharge_time must be string.")
    if not isinstance(summary["one_paragraph_summary"], str):
        raise ValidationError("end_of_admission_summary.one_paragraph_summary must be string.")

    _validate_obj_list(
        summary["problem_list"],
        required_keys={"problem", "status_at_discharge", "supporting_eids"},
        path="end_of_admission_summary.problem_list",
    )
    _validate_obj_list(
        summary["key_tests_and_results"],
        required_keys={"test", "result", "relative_time", "supporting_eids"},
        path="end_of_admission_summary.key_tests_and_results",
    )
    _validate_obj_list(
        summary["treatments_and_meds"],
        required_keys={"treatment_or_med", "details", "supporting_eids"},
        path="end_of_admission_summary.treatments_and_meds",
    )

    disp = summary["disposition"]
    if not isinstance(disp, dict):
        raise ValidationError("end_of_admission_summary.disposition must be an object.")
    if set(disp.keys()) != {"discharge_location", "supporting_eids"}:
        raise ValidationError("end_of_admission_summary.disposition keys mismatch.")
    if not isinstance(disp["discharge_location"], str):
        raise ValidationError("end_of_admission_summary.disposition.discharge_location must be string.")
    if not _is_list_of_strings(disp["supporting_eids"]):
        raise ValidationError("end_of_admission_summary.disposition.supporting_eids must be list[str].")


def enforce_evidence_references(data: dict[str, Any], packet: dict[str, Any], *, strict: bool = True) -> list[str]:
    alias_map = _build_eid_alias_map(packet)
    _normalize_output_eid_aliases_in_place(data, alias_map)
    valid_eids = flatten_eids(packet)
    issues: list[str] = []

    for idx, turn in enumerate(data["conversation"], start=1):
        for eid in turn["evidence_eids"]:
            if eid not in valid_eids:
                issues.append(f"conversation[{idx}] references unknown EID: {eid}")

    summary = data["end_of_admission_summary"]
    for idx, obj in enumerate(summary["problem_list"], start=1):
        for eid in obj["supporting_eids"]:
            if eid not in valid_eids:
                issues.append(f"problem_list[{idx}] references unknown EID: {eid}")
    for idx, obj in enumerate(summary["key_tests_and_results"], start=1):
        for eid in obj["supporting_eids"]:
            if eid not in valid_eids:
                issues.append(f"key_tests_and_results[{idx}] references unknown EID: {eid}")
    for idx, obj in enumerate(summary["treatments_and_meds"], start=1):
        for eid in obj["supporting_eids"]:
            if eid not in valid_eids:
                issues.append(f"treatments_and_meds[{idx}] references unknown EID: {eid}")
    for eid in summary["disposition"]["supporting_eids"]:
        if eid not in valid_eids:
            issues.append(f"disposition references unknown EID: {eid}")

    if issues and strict:
        raise ValidationError("; ".join(issues))
    return issues


def _build_eid_alias_map(packet: dict[str, Any]) -> dict[str, str]:
    alias_map: dict[str, str] = {}
    patient_eid = packet.get("patient", {}).get("eid")
    admission_eid = packet.get("admission", {}).get("eid")
    if isinstance(patient_eid, str):
        alias_map["patient"] = patient_eid
        alias_map["pt"] = patient_eid
    if isinstance(admission_eid, str):
        alias_map["admission"] = admission_eid
        alias_map["adm"] = admission_eid
    return alias_map


def _normalize_output_eid_aliases_in_place(data: dict[str, Any], alias_map: dict[str, str]) -> None:
    if not alias_map:
        return

    for turn in data.get("conversation", []):
        if isinstance(turn, dict) and "evidence_eids" in turn:
            turn["evidence_eids"] = _normalize_eid_list(turn["evidence_eids"], alias_map)

    summary = data.get("end_of_admission_summary", {})
    if not isinstance(summary, dict):
        return
    for key in ("problem_list", "key_tests_and_results", "treatments_and_meds"):
        for item in summary.get(key, []):
            if isinstance(item, dict) and "supporting_eids" in item:
                item["supporting_eids"] = _normalize_eid_list(item["supporting_eids"], alias_map)
    disposition = summary.get("disposition")
    if isinstance(disposition, dict) and "supporting_eids" in disposition:
        disposition["supporting_eids"] = _normalize_eid_list(disposition["supporting_eids"], alias_map)


def _normalize_eid_list(values: Any, alias_map: dict[str, str]) -> Any:
    if not isinstance(values, list):
        return values
    out: list[str] = []
    for value in values:
        if not isinstance(value, str):
            out.append(value)
            continue
        normalized = alias_map.get(value.strip().lower(), value)
        out.append(normalized)
    return out


def _validate_obj_list(value: Any, *, required_keys: set[str], path: str) -> None:
    if not isinstance(value, list):
        raise ValidationError(f"{path} must be a list.")
    for idx, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            raise ValidationError(f"{path}[{idx}] must be an object.")
        if set(item.keys()) != required_keys:
            raise ValidationError(f"{path}[{idx}] keys mismatch.")
        for key, item_value in item.items():
            if key == "supporting_eids":
                if not _is_list_of_strings(item_value):
                    raise ValidationError(f"{path}[{idx}].supporting_eids must be list[str].")
            elif not isinstance(item_value, str):
                raise ValidationError(f"{path}[{idx}].{key} must be string.")


def _is_list_of_strings(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(x, str) for x in value)
