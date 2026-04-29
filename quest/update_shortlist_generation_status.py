#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple


DEFAULT_CSV = Path(__file__).resolve().parents[1] / "output" / "top_100_eligible_patients.csv"
DEFAULT_QUEST_OUTPUT_ROOT = Path("/projects/p33194/medbench-output")
STATUS_FIELDNAME = "quest_generation_complete"
REQUIRED_OUTPUT_FILES = (
    "combined_conversation.json",
    "patient_summary.json",
    "cross_admission_qa.json",
    "benchmark_qa.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update the shortlist CSV with yes/no Quest generation completion status."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Path to top_100_eligible_patients.csv")
    parser.add_argument(
        "--quest-output-root",
        type=Path,
        default=DEFAULT_QUEST_OUTPUT_ROOT,
        help="Quest medbench output root used as the source of truth.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        required_fieldnames = {"subject_id", "eligible_admissions"}
        missing_fieldnames = sorted(required_fieldnames.difference(fieldnames))
        if missing_fieldnames:
            raise ValueError(
                "Expected CSV headers to include 'subject_id' and 'eligible_admissions'; "
                f"got {reader.fieldnames!r}"
            )
        rows = [dict(row) for row in reader]
    if STATUS_FIELDNAME not in fieldnames:
        fieldnames.append(STATUS_FIELDNAME)
    return fieldnames, rows


def generation_complete(quest_output_root: Path, subject_id: str) -> bool:
    patient_root = quest_output_root / str(subject_id)
    return all((patient_root / filename).is_file() for filename in REQUIRED_OUTPUT_FILES)


def write_rows(path: Path, fieldnames: List[str], rows: List[Dict[str, str]]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(path)


def main() -> int:
    args = parse_args()
    fieldnames, rows = load_rows(args.csv)
    quest_output_root = args.quest_output_root.expanduser().resolve()

    for row in rows:
        row[STATUS_FIELDNAME] = "yes" if generation_complete(quest_output_root, row["subject_id"]) else "no"

    write_rows(args.csv, fieldnames, rows)
    print(f"Updated {args.csv} using Quest output root {quest_output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
