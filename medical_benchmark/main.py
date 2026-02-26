#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from med_benchmark import BenchmarkPipeline, build_default_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate long-context healthcare benchmark samples from MIMIC-IV + MIMIC-IV-Note."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    cohort = sub.add_parser("build-cohort", help="Build top-N cohort CSV by admission count.")
    cohort.add_argument("--limit", type=int, default=1000, help="Number of patients to include (default: 1000).")

    gen = sub.add_parser("generate-patient", help="Generate benchmark artifacts for one patient (all admissions by default).")
    gen.add_argument("--subject-id", type=int, required=True, help="MIMIC subject_id to generate.")
    gen.add_argument("--model", required=True, help="OpenAI model name (e.g., gpt-4.1-mini).")
    gen.add_argument("--hadm-id", type=int, help="Optional single admission hadm_id to generate.")
    gen.add_argument("--max-admissions", type=int, help="Optional cap on admissions processed for the patient.")
    gen.add_argument(
        "--include-admissions-without-discharge",
        action="store_true",
        help="Include admissions without discharge notes (not recommended; prompt policy may fail).",
    )
    gen.add_argument("--retry-limit", type=int, help="Override model retry limit.")
    gen.add_argument("--max-output-tokens", type=int, help="Override model max output tokens.")
    gen.add_argument("--seed", type=int, help="Optional model seed if supported by provider/model.")
    gen.add_argument("--row-cap-labs", type=int, help="Optional deterministic row cap override for labs.")
    gen.add_argument("--row-cap-radiology", type=int, help="Optional deterministic row cap override for radiology notes.")
    gen.add_argument("--row-cap-emar", type=int, help="Optional deterministic row cap override for eMAR rows.")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_dir = Path(__file__).resolve().parent
    config = build_default_config(project_dir)

    if getattr(args, "retry_limit", None) is not None:
        config.model.retry_limit = int(args.retry_limit)
    if getattr(args, "max_output_tokens", None) is not None:
        config.model.max_output_tokens = int(args.max_output_tokens)
    if getattr(args, "seed", None) is not None:
        config.model.seed = int(args.seed)

    if getattr(args, "row_cap_labs", None) is not None:
        config.truncation.per_section_row_caps["labs"] = int(args.row_cap_labs)
    if getattr(args, "row_cap_radiology", None) is not None:
        config.truncation.per_section_row_caps["radiology"] = int(args.row_cap_radiology)
    if getattr(args, "row_cap_emar", None) is not None:
        config.truncation.per_section_row_caps["emar"] = int(args.row_cap_emar)

    pipeline = BenchmarkPipeline(config)
    try:
        if args.command == "build-cohort":
            path = pipeline.build_top_cohort(limit=args.limit)
            print(f"Wrote cohort CSV: {path}")
            return 0

        if args.command == "generate-patient":
            manifest = pipeline.generate_patient_sample(
                subject_id=args.subject_id,
                model_name=args.model,
                hadm_id=args.hadm_id,
                max_admissions=args.max_admissions,
                only_with_discharge=not bool(args.include_admissions_without_discharge),
            )
            print(
                "Completed patient generation:",
                f"subject_id={args.subject_id}",
                f"admissions={len(manifest['admissions'])}",
                f"output_root={config.paths.output_root if config.paths else 'N/A'}",
            )
            return 0

        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 2
    finally:
        pipeline.close()


if __name__ == "__main__":
    raise SystemExit(main())
