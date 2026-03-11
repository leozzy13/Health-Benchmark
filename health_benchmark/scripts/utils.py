from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Iterable


EXACT_TS_FORMAT = "%Y-%m-%d %H:%M:%S"
DATE_ONLY_FORMAT = "%Y-%m-%d"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def canonical_json_dumps(data: Any) -> str:
    return json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def pretty_json_dumps(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, data: Any) -> None:
    write_text(path, pretty_json_dumps(data) + "\n")


def format_dt(value: datetime | date | str | None) -> str | None:
    dt = parse_dt(value)
    if dt is None:
        return None
    return dt.strftime(EXACT_TS_FORMAT)


def parse_dt(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.replace(microsecond=0)
    if isinstance(value, date):
        return datetime.combine(value, time.min)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        for fmt in (EXACT_TS_FORMAT, DATE_ONLY_FORMAT, "%Y-%m-%dT%H:%M:%S"):
            try:
                parsed = datetime.strptime(stripped, fmt)
                if fmt == DATE_ONLY_FORMAT:
                    return datetime.combine(parsed.date(), time.min)
                return parsed
            except ValueError:
                continue
    return None


def normalize_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.replace(microsecond=0).strftime(EXACT_TS_FORMAT)
    if isinstance(value, date):
        return value.strftime(DATE_ONLY_FORMAT)
    if isinstance(value, timedelta):
        return int(value.total_seconds())
    return value


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: normalize_scalar(value) for key, value in row.items()}


def normalize_date_to_noon(value: Any) -> str | None:
    dt = parse_dt(value)
    if dt is None:
        return None
    noon_dt = datetime.combine(dt.date(), time(hour=12))
    return noon_dt.strftime(EXACT_TS_FORMAT)


def dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for row in rows:
        key = canonical_json_dumps(row)
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def split_sentences(text: str) -> list[str]:
    raw_parts = re.split(r"(?<=[.!?])\s+|\n+", text or "")
    sentences: list[str] = []
    for part in raw_parts:
        normalized = normalize_grounding_phrase(part)
        if len(normalized) < 12:
            continue
        sentences.append(normalized)
    return sentences


def normalize_grounding_phrase(text: str) -> str:
    compact = re.sub(r"\s+", " ", (text or "").strip())
    compact = compact.strip(" ,;:.")
    return compact.lower()


def round_mean(values: list[int | float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 1)


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def conversation_token_render(lines: list[dict[str, Any]]) -> str:
    rendered: list[str] = []
    for line in lines:
        rendered.append(
            f"{line['turn_number']} | {line['time']} | {line['speaker']} | {line['text']}"
        )
    return "\n".join(rendered)
