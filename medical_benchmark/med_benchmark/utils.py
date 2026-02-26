from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable


DATE_ONLY_FIELDS = {"chartdate", "storedate", "expirationdate"}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def canonical_json_dumps(data: Any) -> str:
    return json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def pretty_json_dumps(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=False, ensure_ascii=False)


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, data: Any) -> None:
    write_text(path, pretty_json_dumps(data) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(canonical_json_dumps(row))
            f.write("\n")


def parse_dt(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"):
            try:
                dt = datetime.strptime(value, fmt)
                if fmt == "%Y-%m-%d":
                    return datetime.combine(dt.date(), datetime.min.time())
                return dt
            except ValueError:
                continue
    return None


def dt_to_iso(value: Any, field_name: str | None = None) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        dt = parse_dt(value)
        if dt is None:
            return value
        return dt.strftime("%Y-%m-%dT%H:%M:%S")
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%dT%H:%M:%S")
    if isinstance(value, date):
        if field_name in DATE_ONLY_FIELDS:
            return datetime.combine(value, datetime.min.time()).strftime("%Y-%m-%dT%H:%M:%S")
        return value.isoformat()
    return str(value)


def normalize_scalar(value: Any, field_name: str | None = None) -> Any:
    if value is None:
        return None
    if isinstance(value, (datetime, date)):
        return dt_to_iso(value, field_name=field_name)
    if isinstance(value, timedelta):
        return value.total_seconds()
    return value


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    return {k: normalize_scalar(v, field_name=k) for k, v in row.items()}


def minutes_since(start_dt: datetime | None, event_dt: Any) -> int | None:
    if start_dt is None:
        return None
    dt = parse_dt(event_dt)
    if dt is None:
        return None
    return int((dt - start_dt).total_seconds() // 60)


def format_relative_time_from_minutes(minutes: int | None) -> str | None:
    if minutes is None:
        return None
    sign = "-" if minutes < 0 else ""
    abs_min = abs(minutes)
    hours, mins = divmod(abs_min, 60)
    return f"{sign}H+{hours:02d}:{mins:02d}"


def hospital_day_label(minutes: int | None) -> str | None:
    if minutes is None:
        return None
    if minutes < 0:
        return None
    day = (minutes // (24 * 60)) + 1
    hh = (minutes % (24 * 60)) // 60
    mm = minutes % 60
    return f"HospitalDay{day} {hh:02d}:{mm:02d}"


def clip_rows(rows: list[dict[str, Any]], cap: int | None) -> tuple[list[dict[str, Any]], bool]:
    if cap is None or cap < 0:
        return rows, False
    if len(rows) <= cap:
        return rows, False
    return rows[:cap], True


def flatten_eids(obj: Any) -> set[str]:
    eids: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "eid" and isinstance(v, str):
                eids.add(v)
            else:
                eids.update(flatten_eids(v))
    elif isinstance(obj, list):
        for item in obj:
            eids.update(flatten_eids(item))
    return eids


def utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
