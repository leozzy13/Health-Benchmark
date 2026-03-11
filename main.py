#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from health_benchmark.scripts import BenchmarkPipeline, build_default_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate simplified long-context healthcare benchmark samples from MIMIC-IV + MIMIC-IV-Note."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate-patient", help="Generate benchmark artifacts for one patient.")
    generate.add_argument("--mimiciv-dir", help="Override the MIMIC-IV root directory or hosp directory.")
    generate.add_argument("--mimiciv-note-dir", help="Override the MIMIC-IV-Note directory.")
    generate.add_argument("--output-root", help="Override the output root directory.")
    generate.add_argument("--subject-id", type=int, help="MIMIC subject_id to generate. Falls back to config.yaml.")
    generate.add_argument("--model", help="OpenAI model identifier. Falls back to config.yaml.")
    generate.add_argument("--hadm-id", type=int, help="Optional single admission hadm_id to generate.")
    generate.add_argument("--max-admissions", type=int, help="Optional cap on admissions processed for the patient.")
    generate.add_argument("--retry-limit", type=int, help="Override OpenAI validation retry count.")
    generate.add_argument("--max-output-tokens", type=int, help="Override the OpenAI output token cap.")
    generate.add_argument("--api-key-env", help="Environment variable name containing the OpenAI API key.")
    generate.add_argument("--openai-base-url", help="Optional OpenAI-compatible base URL override.")

    generate_qa = subparsers.add_parser("generate-qa", help="Generate QA benchmark artifacts for one patient.")
    generate_qa.add_argument("--output-root", help="Override the output root directory.")
    qa_target = generate_qa.add_mutually_exclusive_group(required=True)
    qa_target.add_argument("--subject-id", type=int, help="Patient subject_id whose benchmark artifacts already exist.")
    qa_target.add_argument("--patient-dir", help="Existing patient output directory containing conversation artifacts.")
    generate_qa.add_argument("--model", help="OpenAI model identifier. Falls back to config.yaml.")
    generate_qa.add_argument(
        "--single-admission-qa-count",
        type=int,
        default=12,
        help="Number of hard QA items to generate per admission (default: 12).",
    )
    generate_qa.add_argument("--max-output-tokens", type=int, help="Override the OpenAI output token cap.")
    generate_qa.add_argument("--api-key-env", help="Environment variable name containing the OpenAI API key.")
    generate_qa.add_argument("--openai-base-url", help="Optional OpenAI-compatible base URL override.")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parent
    project_dir = repo_root / "health_benchmark"
    config = build_default_config(project_dir)

    if getattr(args, "mimiciv_dir", None):
        config.dataset.mimiciv_hosp_path = _resolve_hosp_dir(Path(args.mimiciv_dir))
    if getattr(args, "mimiciv_note_dir", None):
        config.dataset.mimiciv_note_path = Path(args.mimiciv_note_dir).expanduser().resolve()
    if getattr(args, "output_root", None):
        config.output.root = Path(args.output_root).expanduser().resolve()
    if getattr(args, "retry_limit", None) is not None:
        config.openai.max_retries = int(args.retry_limit)
    if getattr(args, "max_output_tokens", None) is not None:
        config.openai.max_output_tokens = int(args.max_output_tokens)
    if getattr(args, "api_key_env", None):
        config.openai.api_key_env = str(args.api_key_env)
    if getattr(args, "openai_base_url", None):
        config.openai.base_url = str(args.openai_base_url)

    pipeline = BenchmarkPipeline(config)
    try:
        if args.command == "generate-patient":
            summary = pipeline.generate_patient_sample(
                subject_id=args.subject_id,
                model_name=args.model,
                hadm_id=args.hadm_id,
                max_admissions=args.max_admissions,
            )
            print(
                "Completed patient generation:",
                f"subject_id={summary['subject_id']}",
                f"processed_admissions={summary['processed_admissions']}",
                f"output_dir={config.output.root / str(summary['subject_id'])}",
            )
            return 0

        if args.command == "generate-qa":
            summary = pipeline.generate_patient_qa(
                subject_id=args.subject_id,
                patient_dir=Path(args.patient_dir).expanduser().resolve() if args.patient_dir else None,
                model_name=args.model,
                single_admission_qa_count=args.single_admission_qa_count,
            )
            print(
                "Completed QA generation:",
                f"subject_id={summary['subject_id']}",
                f"processed_admissions={summary['processed_admissions']}",
                f"total_qas={summary['total_qas']}",
                f"patient_dir={summary['patient_dir']}",
            )
            return 0

        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 2
    finally:
        pipeline.close()


def _resolve_hosp_dir(path: Path) -> Path:
    expanded = path.expanduser().resolve()
    if expanded.name == "hosp":
        return expanded
    candidate = expanded / "hosp"
    if candidate.exists():
        return candidate.resolve()
    return expanded


if __name__ == "__main__":
    raise SystemExit(main())
