#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..scripts.config import build_default_config
from .config import EVALUATION_STAGE_CHOICES, build_settings
from .loader import resolve_patient_root
from .pipeline import TemporalEvaluationPipeline


PROVIDER_CHOICES = ("openai", "vllm")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the open-answer evaluation pipeline for one patient."
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--subject-id", type=int, help="Patient subject_id whose artifacts already exist.")
    target.add_argument("--patient-dir", help="Existing patient output directory containing conversation and QA artifacts.")
    parser.add_argument("--output-root", help="Override the output root directory.")
    parser.add_argument("--provider", choices=PROVIDER_CHOICES, help="Evaluation provider override. Defaults to vllm.")
    parser.add_argument("--base-url", help="OpenAI-compatible base URL override.")
    parser.add_argument("--judge-base-url", help="Optional OpenAI-compatible base URL for the fixed Qwen/Qwen3.5-27B judge.")
    parser.add_argument(
        "--stage",
        choices=EVALUATION_STAGE_CHOICES,
        default="full",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--api-key-env", help="Environment variable name for the evaluation API key.")
    parser.add_argument("--models", nargs="+", help="Optional model override. Defaults to the Qwen3.5 trio.")
    parser.add_argument("--replace-existing", action="store_true", help="Replace existing evaluation outputs for requested models.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    project_dir = repo_root / "health_benchmark"
    base_config = build_default_config(project_dir)
    if args.output_root:
        base_config.output.root = Path(args.output_root).expanduser().resolve()

    patient_root, subject_id = resolve_patient_root(
        output_root=base_config.output.root,
        subject_id=args.subject_id,
        patient_dir=Path(args.patient_dir).expanduser().resolve() if args.patient_dir else None,
    )
    settings = build_settings(
        base_config,
        provider=args.provider,
        base_url=args.base_url,
        judge_base_url=args.judge_base_url,
        api_key_env=args.api_key_env,
        models=args.models,
        stage=args.stage,
        replace_existing=True if args.replace_existing else None,
    )
    pipeline = TemporalEvaluationPipeline(base_config, settings)
    summary = pipeline.run([(subject_id, patient_root)])
    print(json.dumps(summary, indent=2))
    return 0 if summary["final_status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
