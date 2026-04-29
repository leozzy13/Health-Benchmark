#!/usr/bin/env python3

import argparse
import csv
from collections import namedtuple
from pathlib import Path
from typing import List


DEFAULT_LAUNCHER = "/projects/p33194/health-benchmark/quest/run_generate_all_qwen.slurm"
DEFAULT_CSV = Path(__file__).resolve().parents[1] / "output" / "top_100_eligible_patients.csv"


PatientRow = namedtuple("PatientRow", ["csv_index", "subject_id", "eligible_admissions"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Emit Quest generation submission commands for one smoke-test patient plus "
            "the patients with the lowest eligible-admission counts from the shortlist CSV."
        )
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Path to top_100_eligible_patients.csv")
    parser.add_argument("--account", default="p33194", help="Slurm account for sbatch commands")
    parser.add_argument("--launcher", default=DEFAULT_LAUNCHER, help="Quest Slurm launcher path")
    parser.add_argument(
        "--smoke-test-subject",
        type=int,
        default=11826927,
        help="Subject id for the single-patient smoke test job",
    )
    parser.add_argument(
        "--lowest-count",
        type=int,
        default=50,
        help="Number of lowest eligible-admission patients to include after the smoke test",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Number of patients per follow-up batch job",
    )
    return parser.parse_args()


def load_rows(path: Path) -> List[PatientRow]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        required_fieldnames = {"subject_id", "eligible_admissions"}
        missing_fieldnames = sorted(required_fieldnames.difference(fieldnames))
        if missing_fieldnames:
            raise ValueError(
                "Expected CSV headers to include 'subject_id' and 'eligible_admissions'; "
                f"got {reader.fieldnames!r}"
            )
        rows = [
            PatientRow(
                csv_index=index,
                subject_id=int(row["subject_id"]),
                eligible_admissions=int(row["eligible_admissions"]),
            )
            for index, row in enumerate(reader)
        ]
    if not rows:
        raise ValueError(f"No patient rows found in {path}")
    return rows


def select_lowest_rows(
    rows: List[PatientRow],
    *,
    smoke_test_subject: int,
    lowest_count: int,
) -> List[PatientRow]:
    if lowest_count <= 0:
        raise ValueError("--lowest-count must be positive")
    filtered = [row for row in rows if row.subject_id != smoke_test_subject]
    if len(filtered) < lowest_count:
        raise ValueError(
            f"Requested {lowest_count} low-eligibility patients after excluding smoke-test "
            f"subject {smoke_test_subject}, but only found {len(filtered)}"
        )
    return sorted(filtered, key=lambda row: (row.eligible_admissions, row.csv_index))[:lowest_count]


def chunk_rows(rows: List[PatientRow], *, batch_size: int) -> List[List[PatientRow]]:
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    return [rows[index : index + batch_size] for index in range(0, len(rows), batch_size)]


def render_commands(
    *,
    account: str,
    launcher: str,
    smoke_test_subject: int,
    batches: List[List[PatientRow]],
) -> str:
    lines = [
        f"SCRIPT={launcher}",
        f"ACCOUNT={account}",
        "",
        "# Job 1: single-patient smoke test",
        f"sbatch --account=$ACCOUNT $SCRIPT {smoke_test_subject}",
    ]
    for batch_index, batch in enumerate(batches, start=2):
        counts = " ".join(str(row.eligible_admissions) for row in batch)
        subject_ids = " ".join(str(row.subject_id) for row in batch)
        lines.extend(
            [
                "",
                f"# Job {batch_index}",
                f"# counts: {counts}",
                f"sbatch --account=$ACCOUNT $SCRIPT {subject_ids}",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    rows = load_rows(args.csv)
    selected_rows = select_lowest_rows(
        rows,
        smoke_test_subject=args.smoke_test_subject,
        lowest_count=args.lowest_count,
    )
    batches = chunk_rows(selected_rows, batch_size=args.batch_size)
    print(
        render_commands(
            account=args.account,
            launcher=args.launcher,
            smoke_test_subject=args.smoke_test_subject,
            batches=batches,
        ),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
