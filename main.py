#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from health_benchmark.scripts import BenchmarkPipeline, build_default_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate long-context healthcare benchmark samples from MIMIC-IV + MIMIC-IV-Note."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    cohort = sub.add_parser("build-cohort", help="Build top-N cohort CSV by admission count.")
    cohort.add_argument(
        "--mimiciv-dir",
        help="Override MIMIC-IV directory (expects hosp/ and icu/ under this path).",
    )
    cohort.add_argument(
        "--mimiciv-note-dir",
        help="Override MIMIC-IV-Note directory (expects discharge/radiology CSVs under this path).",
    )
    cohort.add_argument(
        "--output-root",
        help="Override output root directory for generated artifacts.",
    )
    cohort.add_argument("--limit", type=int, default=1000, help="Number of patients to include (default: 1000).")

    gen = sub.add_parser("generate-patient", help="Generate benchmark artifacts for one patient (all admissions by default).")
    gen.add_argument(
        "--mimiciv-dir",
        help="Override MIMIC-IV directory (expects hosp/ and icu/ under this path).",
    )
    gen.add_argument(
        "--mimiciv-note-dir",
        help="Override MIMIC-IV-Note directory (expects discharge/radiology CSVs under this path).",
    )
    gen.add_argument(
        "--output-root",
        help="Override output root directory for generated artifacts.",
    )
    gen.add_argument("--subject-id", type=int, required=True, help="MIMIC subject_id to generate.")
    gen.add_argument("--model", required=True, help="OpenAI model identifier.")
    gen.add_argument("--hadm-id", type=int, help="Optional single admission hadm_id to generate.")
    gen.add_argument("--max-admissions", type=int, help="Optional cap on admissions processed for the patient.")
    gen.add_argument(
        "--include-admissions-without-discharge",
        action="store_true",
        help="Include admissions without discharge notes (not recommended; prompt policy may fail).",
    )
    gen.add_argument("--retry-limit", type=int, help="Override model retry limit.")
    gen.add_argument("--max-output-tokens", type=int, help="Override model max output tokens.")
    gen.add_argument("--seed", type=int, help="Optional OpenAI model seed if supported.")
    gen.add_argument("--api-key-env", help="Environment variable name containing the OpenAI API key.")
    gen.add_argument("--openai-base-url", help="Optional OpenAI-compatible base URL override.")
    gen.add_argument("--row-cap-labs", type=int, help="Optional deterministic row cap override for labs.")
    gen.add_argument("--row-cap-radiology", type=int, help="Optional deterministic row cap override for radiology notes.")
    gen.add_argument("--row-cap-emar", type=int, help="Optional deterministic row cap override for eMAR rows.")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parent
    project_dir = repo_root / "health_benchmark"
    config = build_default_config(project_dir)
    if config.paths is None:
        raise RuntimeError("Default config did not configure paths.")

    if getattr(args, "mimiciv_dir", None):
        config.paths.mimiciv_dir = Path(args.mimiciv_dir).expanduser().resolve()
    if getattr(args, "mimiciv_note_dir", None):
        config.paths.mimiciv_note_dir = Path(args.mimiciv_note_dir).expanduser().resolve()
    if getattr(args, "output_root", None):
        config.paths.output_root = Path(args.output_root).expanduser().resolve()

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
    if getattr(args, "api_key_env", None):
        config.model.api_key_env = str(args.api_key_env)
    if getattr(args, "openai_base_url", None):
        config.model.openai_base_url = str(args.openai_base_url)

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
