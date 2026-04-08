#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from health_benchmark.scripts import BenchmarkPipeline, build_default_config
from health_benchmark.scripts.config import resolve_llm_base_url
from health_benchmark.temporal_eval.config import build_settings as build_evaluation_settings
from health_benchmark.temporal_eval.loader import resolve_patient_targets
from health_benchmark.temporal_eval.pipeline import TemporalEvaluationPipeline


def _add_common_llm_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provider", choices=["openai", "vllm"], default=None, help="LLM provider override.")
    parser.add_argument("--model", help="Model identifier. Falls back to config.yaml.")
    parser.add_argument("--max-output-tokens", type=int, help="Override the output token cap.")
    parser.add_argument("--api-key-env", help="Environment variable name containing the provider API key.")
    parser.add_argument(
        "--base-url",
        "--openai-base-url",
        dest="base_url",
        help="Optional OpenAI-compatible base URL override.",
    )


def _add_common_eval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provider", choices=["openai", "vllm"], default=None, help="Evaluation provider override.")
    parser.add_argument("--api-key-env", help="Environment variable name containing the provider API key.")
    parser.add_argument(
        "--base-url",
        "--openai-base-url",
        dest="base_url",
        help="Optional OpenAI-compatible base URL override.",
    )


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
    generate.add_argument("--hadm-id", type=int, help="Optional single admission hadm_id to generate.")
    generate.add_argument("--max-admissions", type=int, help="Optional cap on admissions processed for the patient.")
    generate.add_argument("--retry-limit", type=int, help="Override validation retry count.")
    _add_common_llm_args(generate)

    generate_qa = subparsers.add_parser("generate-qa", help="Generate QA benchmark artifacts for one patient.")
    generate_qa.add_argument("--output-root", help="Override the output root directory.")
    qa_target = generate_qa.add_mutually_exclusive_group(required=True)
    qa_target.add_argument("--subject-id", type=int, help="Patient subject_id whose benchmark artifacts already exist.")
    qa_target.add_argument("--patient-dir", help="Existing patient output directory containing conversation artifacts.")
    _add_common_llm_args(generate_qa)

    generate_all = subparsers.add_parser(
        "generate-all",
        help="Generate conversation and QA benchmark artifacts in one run.",
    )
    generate_all.add_argument("--mimiciv-dir", help="Override the MIMIC-IV root directory or hosp directory.")
    generate_all.add_argument("--mimiciv-note-dir", help="Override the MIMIC-IV-Note directory.")
    generate_all.add_argument("--output-root", help="Override the output root directory.")
    all_target = generate_all.add_mutually_exclusive_group(required=True)
    all_target.add_argument("--subject-id", type=int, help="One MIMIC subject_id to generate.")
    all_target.add_argument("--subject-ids", type=int, nargs="+", help="One or more MIMIC subject_ids to generate.")
    generate_all.add_argument("--max-admissions", type=int, help="Optional cap on admissions processed per patient.")
    generate_all.add_argument("--retry-limit", type=int, help="Override validation retry count.")
    generate_all.add_argument("--fail-fast", action="store_true", help="Stop the batch on the first patient failure.")
    _add_common_llm_args(generate_all)

    evaluate = subparsers.add_parser(
        "evaluate",
        help="Evaluate existing patient benchmark artifacts with open-answer scoring.",
    )
    evaluate.add_argument("--output-root", help="Override the output root directory.")
    eval_target = evaluate.add_mutually_exclusive_group(required=True)
    eval_target.add_argument("--subject-id", type=int, help="One patient subject_id to evaluate.")
    eval_target.add_argument("--subject-ids", type=int, nargs="+", help="One or more patient subject_ids to evaluate.")
    eval_target.add_argument("--patient-manifest", help="Text file containing one subject_id per line.")
    eval_target.add_argument("--patient-dir", help="Existing patient directory to evaluate.")
    evaluate.add_argument("--models", nargs="+", help="Optional model override. Defaults to the Qwen3.5 trio.")
    evaluate.add_argument("--replace-existing", action="store_true", help="Replace existing evaluation outputs for requested models.")
    _add_common_eval_args(evaluate)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parent
    project_dir = repo_root / "health_benchmark"
    config = build_default_config(project_dir)
    original_provider = config.llm.provider

    if getattr(args, "mimiciv_dir", None):
        config.dataset.mimiciv_hosp_path = _resolve_hosp_dir(Path(args.mimiciv_dir))
    if getattr(args, "mimiciv_note_dir", None):
        config.dataset.mimiciv_note_path = Path(args.mimiciv_note_dir).expanduser().resolve()
    if getattr(args, "output_root", None):
        config.output.root = Path(args.output_root).expanduser().resolve()
    if getattr(args, "provider", None):
        config.llm.provider = str(args.provider)
    if getattr(args, "retry_limit", None) is not None:
        config.llm.max_retries = int(args.retry_limit)
    if getattr(args, "model", None):
        config.llm.model = str(args.model)
    if getattr(args, "max_output_tokens", None) is not None:
        config.llm.max_output_tokens = int(args.max_output_tokens)
    if getattr(args, "api_key_env", None):
        config.llm.api_key_env = str(args.api_key_env)
    provider_changed = getattr(args, "provider", None) is not None and str(args.provider) != str(original_provider)
    if provider_changed and getattr(args, "base_url", None) is None:
        config.llm.base_url = None
    if getattr(args, "base_url", None):
        config.llm.base_url = str(args.base_url)
    config.llm.base_url = resolve_llm_base_url(config.llm.provider, config.llm.base_url)

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
            )
            print(
                "Completed QA generation:",
                f"subject_id={summary['subject_id']}",
                f"processed_admissions={summary['processed_admissions']}",
                f"total_qas={summary['total_qas']}",
                f"patient_dir={summary['patient_dir']}",
            )
            return 0

        if args.command == "generate-all":
            subject_ids = [int(args.subject_id)] if args.subject_id is not None else [int(subject_id) for subject_id in args.subject_ids]
            summary = pipeline.generate_all(
                subject_ids=subject_ids,
                model_name=args.model,
                max_admissions=args.max_admissions,
                fail_fast=bool(args.fail_fast),
            )
            print(json.dumps(summary, indent=2))
            return 0 if not summary["failed"] else 1

        if args.command == "evaluate":
            targets = resolve_patient_targets(
                output_root=config.output.root,
                subject_id=args.subject_id,
                subject_ids=args.subject_ids,
                patient_manifest=Path(args.patient_manifest).expanduser().resolve() if args.patient_manifest else None,
                patient_dir=Path(args.patient_dir).expanduser().resolve() if args.patient_dir else None,
            )
            settings = build_evaluation_settings(
                config,
                provider=args.provider,
                base_url=args.base_url,
                api_key_env=args.api_key_env,
                models=args.models,
                replace_existing=True if args.replace_existing else None,
            )
            eval_pipeline = TemporalEvaluationPipeline(config, settings)
            summary = eval_pipeline.run(targets)
            print(json.dumps(summary, indent=2))
            return 0 if not summary["failed"] else 1

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
