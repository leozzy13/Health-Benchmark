#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from health_benchmark.scripts import BenchmarkPipeline, build_default_config
from health_benchmark.scripts.config import resolve_llm_base_url
from health_benchmark.evaluation.config import (
    EVALUATION_STAGE_CHOICES,
    MEMORY_METHOD_CHOICES,
    RAG_METHOD_CHOICES,
    build_memory_settings,
    build_rag_settings,
    build_settings as build_evaluation_settings,
)
from health_benchmark.evaluation.loader import resolve_patient_targets
from health_benchmark.evaluation.memory_pipeline import MemoryEvaluationPipeline
from health_benchmark.evaluation.pipeline import EvaluationPipeline
from health_benchmark.evaluation.rag_pipeline import RagEvaluationPipeline
from health_benchmark.evaluation.cohort_summary import summarize_evaluation_cohort
from health_benchmark.evaluation.summary_tokens import refresh_patient_summary_tokens


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"must be a non-negative integer, got: {value}")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got: {value}")
    return parsed


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
    parser.add_argument("--judge-base-url", help="Optional OpenAI-compatible base URL for the fixed Qwen/Qwen3.5-27B judge.")
    parser.add_argument("--retry-limit", type=_non_negative_int, help="Retry limit per evaluation batch after the first attempt.")
    parser.add_argument("--timeout-seconds", type=_positive_int, help="OpenAI-compatible request timeout in seconds.")


def _add_memory_eval_args(parser: argparse.ArgumentParser) -> None:
    _add_common_eval_args(parser)
    parser.add_argument(
        "--memory-method",
        choices=MEMORY_METHOD_CHOICES,
        default="mem0",
        help="Memory construction method. Defaults to mem0.",
    )
    parser.add_argument("--mem0-chunk-token-cap", type=_positive_int, help="Admission chunk token cap. Defaults to 12000.")
    parser.add_argument("--mem0-previous-chunk-summaries", type=_non_negative_int, help="Previous chunk summaries included during construction. Defaults to 1.")
    parser.add_argument("--mem0-max-candidate-memories", type=_positive_int, help="Maximum extracted candidate memories per chunk. Defaults to 32.")
    parser.add_argument("--mem0-similar-memories", type=_positive_int, help="Similar existing memories retrieved per candidate. Defaults to 10.")
    parser.add_argument("--mem0-max-update-memories", type=_positive_int, help="Maximum unique existing memories shown to each update prompt. Defaults to 40.")
    parser.add_argument("--mem0-answer-retrieval-top-k", type=_positive_int, help="Question-time memory retrieval top-k. Defaults to 64.")
    parser.add_argument("--mem0-max-answer-memories", type=_positive_int, help="Maximum memories shown in each answer prompt. Defaults to 32.")
    parser.add_argument("--mem0-max-output-tokens", type=_positive_int, help="Output token cap for Mem0 extraction/update/summary calls. Defaults to 4096.")
    parser.add_argument("--mem0-embedding-model", help="Dense embedding model for memory retrieval. Defaults to Qwen/Qwen3-Embedding-8B.")
    parser.add_argument("--mem0-embedding-device", help="Dense embedding device. Defaults to cuda.")
    parser.add_argument("--mem0-embedding-gpu-device-ids", help="Physical GPU ids reserved for memory embeddings, e.g. 1.")
    parser.add_argument("--mem0-embedding-batch-size", type=_positive_int, help="Dense embedding batch size. Defaults to 8.")
    parser.add_argument("--mem0-embedding-max-length", type=_positive_int, help="Dense embedding tokenizer max length. Defaults to 1024.")
    parser.add_argument("--mem0-model-max-len", type=_positive_int, help="Memory answer-model max length override, used to match the vLLM server.")
    parser.add_argument("--mem0-model-tensor-parallel-size", type=_positive_int, help="Memory answer-model tensor parallel size override, used to match the vLLM server.")


def _add_rag_eval_args(parser: argparse.ArgumentParser) -> None:
    _add_common_eval_args(parser)
    parser.add_argument(
        "--rag-method",
        choices=RAG_METHOD_CHOICES,
        default="embedding-rag",
        help="RAG retrieval method. Defaults to embedding-rag.",
    )
    parser.add_argument("--rag-document-unit", default=None, help="RAG document unit. Defaults to admission.")
    parser.add_argument("--rag-selection-policy", default=None, help="RAG context selection policy. Defaults to score_until_budget.")
    parser.add_argument("--rag-render-order", default=None, help="RAG context render order. Defaults to chronological.")
    parser.add_argument("--rag-embedding-model", help="Dense embedding model for Embedding-RAG. Defaults to Qwen/Qwen3-Embedding-8B.")
    parser.add_argument("--rag-embedding-device", help="Dense embedding device. Defaults to cuda.")
    parser.add_argument("--rag-embedding-gpu-device-ids", help="Physical GPU ids reserved for RAG embeddings, e.g. 1.")
    parser.add_argument("--rag-embedding-batch-size", type=_positive_int, help="Dense embedding batch size. Defaults to 8.")
    parser.add_argument("--rag-embedding-max-length", type=_positive_int, help="Dense embedding tokenizer max length. Defaults to 1024.")
    parser.add_argument("--rag-model-max-len", type=_positive_int, help="RAG answer-model max length override, used to match the vLLM server.")
    parser.add_argument("--rag-model-tensor-parallel-size", type=_positive_int, help="RAG answer-model tensor parallel size override, used to match the vLLM server.")


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
    evaluate.add_argument("--benchmark-root", help="Benchmark input root. Defaults to output/benchmark.")
    evaluate.add_argument("--evaluation-root", help="Evaluation artifact root. Defaults to output/evaluation.")
    evaluate.add_argument("--output-root", help="Alias for --benchmark-root.")
    eval_target = evaluate.add_mutually_exclusive_group(required=True)
    eval_target.add_argument("--subject-id", type=int, help="One patient subject_id to evaluate.")
    eval_target.add_argument("--subject-ids", type=int, nargs="+", help="One or more patient subject_ids to evaluate.")
    eval_target.add_argument("--patient-manifest", help="Text file containing one subject_id per line.")
    eval_target.add_argument("--patient-dir", help="Existing patient directory to evaluate.")
    evaluate.add_argument("--models", nargs="+", help="Optional model override. Defaults to the Qwen3.5 trio.")
    evaluate.add_argument("--replace-existing", action="store_true", help="Replace existing evaluation outputs for requested models.")
    evaluate.add_argument(
        "--stage",
        choices=EVALUATION_STAGE_CHOICES,
        default="full",
        help=argparse.SUPPRESS,
    )
    _add_common_eval_args(evaluate)

    evaluate_memory = subparsers.add_parser(
        "evaluate-memory",
        help="Evaluate existing patient benchmark artifacts with admission-chunk Mem0 memory.",
    )
    evaluate_memory.add_argument("--benchmark-root", help="Benchmark input root. Defaults to output/benchmark.")
    evaluate_memory.add_argument("--evaluation-root", help="Evaluation artifact root. Defaults to output/evaluation.")
    evaluate_memory.add_argument("--output-root", help="Alias for --benchmark-root.")
    memory_target = evaluate_memory.add_mutually_exclusive_group(required=True)
    memory_target.add_argument("--subject-id", type=int, help="One patient subject_id to evaluate.")
    memory_target.add_argument("--subject-ids", type=int, nargs="+", help="One or more patient subject_ids to evaluate.")
    memory_target.add_argument("--patient-manifest", help="Text file containing one subject_id per line.")
    memory_target.add_argument("--patient-dir", help="Existing patient directory to evaluate.")
    evaluate_memory.add_argument("--models", nargs="+", help="Optional model override. Defaults to the Qwen3.5 trio.")
    evaluate_memory.add_argument("--replace-existing", action="store_true", help="Replace existing memory evaluation outputs for requested models.")
    evaluate_memory.add_argument(
        "--stage",
        choices=EVALUATION_STAGE_CHOICES,
        default="full",
        help=argparse.SUPPRESS,
    )
    _add_memory_eval_args(evaluate_memory)

    evaluate_rag = subparsers.add_parser(
        "evaluate-rag",
        help="Evaluate existing patient benchmark artifacts with admission-level RAG.",
    )
    evaluate_rag.add_argument("--benchmark-root", help="Benchmark input root. Defaults to output/benchmark.")
    evaluate_rag.add_argument("--evaluation-root", help="Evaluation artifact root. Defaults to output/evaluation.")
    evaluate_rag.add_argument("--output-root", help="Alias for --benchmark-root.")
    rag_target = evaluate_rag.add_mutually_exclusive_group(required=True)
    rag_target.add_argument("--subject-id", type=int, help="One patient subject_id to evaluate.")
    rag_target.add_argument("--subject-ids", type=int, nargs="+", help="One or more patient subject_ids to evaluate.")
    rag_target.add_argument("--patient-manifest", help="Text file containing one subject_id per line.")
    rag_target.add_argument("--patient-dir", help="Existing patient directory to evaluate.")
    evaluate_rag.add_argument("--models", nargs="+", help="Optional model override. Defaults to the Qwen3.5 trio.")
    evaluate_rag.add_argument("--replace-existing", action="store_true", help="Replace existing RAG evaluation outputs for requested models.")
    evaluate_rag.add_argument(
        "--stage",
        choices=EVALUATION_STAGE_CHOICES,
        default="full",
        help=argparse.SUPPRESS,
    )
    _add_rag_eval_args(evaluate_rag)

    refresh_summary_tokens = subparsers.add_parser(
        "refresh-summary-tokens",
        help="Recompute patient_summary.json conversation token counts with a Hugging Face tokenizer.",
    )
    refresh_summary_tokens.add_argument("--benchmark-root", help="Benchmark input root. Defaults to output/benchmark.")
    refresh_summary_tokens.add_argument("--output-root", help="Alias for --benchmark-root.")
    refresh_target = refresh_summary_tokens.add_mutually_exclusive_group(required=True)
    refresh_target.add_argument("--subject-id", type=int, help="One patient subject_id to refresh.")
    refresh_target.add_argument("--subject-ids", type=int, nargs="+", help="One or more patient subject_ids to refresh.")
    refresh_target.add_argument("--patient-manifest", help="Text file containing one subject_id per line.")
    refresh_summary_tokens.add_argument(
        "--tokenizer-model",
        default="Qwen/Qwen3.5-4B",
        help="HF tokenizer model or local tokenizer path. Defaults to Qwen/Qwen3.5-4B.",
    )

    summarize_cohort = subparsers.add_parser(
        "summarize-evaluation-cohort",
        help="Build a whole-cohort detailed leaderboard from completed evaluation summaries.",
    )
    summarize_cohort.add_argument("--evaluation-root", help="Evaluation artifact root. Defaults to output/evaluation.")
    summarize_cohort.add_argument(
        "--patient-manifest",
        required=True,
        help="Text file containing the cohort subject_ids, one per line.",
    )

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
    if getattr(args, "benchmark_root", None):
        config.output.root = Path(args.benchmark_root).expanduser().resolve()
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
                judge_base_url=args.judge_base_url,
                api_key_env=args.api_key_env,
                models=args.models,
                stage=args.stage,
                replace_existing=True if args.replace_existing else None,
                timeout_seconds=args.timeout_seconds,
                retry_limit=args.retry_limit,
                evaluation_root=Path(args.evaluation_root).expanduser().resolve() if args.evaluation_root else None,
            )
            eval_pipeline = EvaluationPipeline(config, settings)
            summary = eval_pipeline.run(targets)
            print(json.dumps(summary, indent=2))
            return 0 if not summary["failed"] else 1

        if args.command == "evaluate-memory":
            targets = resolve_patient_targets(
                output_root=config.output.root,
                subject_id=args.subject_id,
                subject_ids=args.subject_ids,
                patient_manifest=Path(args.patient_manifest).expanduser().resolve() if args.patient_manifest else None,
                patient_dir=Path(args.patient_dir).expanduser().resolve() if args.patient_dir else None,
            )
            settings = build_memory_settings(
                config,
                provider=args.provider,
                base_url=args.base_url,
                judge_base_url=args.judge_base_url,
                api_key_env=args.api_key_env,
                models=args.models,
                stage=args.stage,
                replace_existing=True if args.replace_existing else None,
                timeout_seconds=args.timeout_seconds,
                retry_limit=args.retry_limit,
                mem0_chunk_token_cap=args.mem0_chunk_token_cap,
                mem0_previous_chunk_summaries=args.mem0_previous_chunk_summaries,
                mem0_max_candidate_memories=args.mem0_max_candidate_memories,
                mem0_similar_memories=args.mem0_similar_memories,
                mem0_max_update_memories=args.mem0_max_update_memories,
                mem0_answer_retrieval_top_k=args.mem0_answer_retrieval_top_k,
                mem0_max_answer_memories=args.mem0_max_answer_memories,
                mem0_max_output_tokens=args.mem0_max_output_tokens,
                mem0_embedding_model=args.mem0_embedding_model,
                mem0_embedding_device=args.mem0_embedding_device,
                mem0_embedding_gpu_device_ids=args.mem0_embedding_gpu_device_ids,
                mem0_embedding_batch_size=args.mem0_embedding_batch_size,
                mem0_embedding_max_length=args.mem0_embedding_max_length,
                mem0_model_max_len=args.mem0_model_max_len,
                mem0_model_tensor_parallel_size=args.mem0_model_tensor_parallel_size,
                memory_method=args.memory_method,
                evaluation_root=Path(args.evaluation_root).expanduser().resolve() if args.evaluation_root else None,
            )
            memory_pipeline = MemoryEvaluationPipeline(config, settings)
            summary = memory_pipeline.run(targets)
            print(json.dumps(summary, indent=2))
            return 0 if not summary["failed"] else 1

        if args.command == "evaluate-rag":
            targets = resolve_patient_targets(
                output_root=config.output.root,
                subject_id=args.subject_id,
                subject_ids=args.subject_ids,
                patient_manifest=Path(args.patient_manifest).expanduser().resolve() if args.patient_manifest else None,
                patient_dir=Path(args.patient_dir).expanduser().resolve() if args.patient_dir else None,
            )
            settings = build_rag_settings(
                config,
                provider=args.provider,
                base_url=args.base_url,
                judge_base_url=args.judge_base_url,
                api_key_env=args.api_key_env,
                models=args.models,
                stage=args.stage,
                replace_existing=True if args.replace_existing else None,
                timeout_seconds=args.timeout_seconds,
                retry_limit=args.retry_limit,
                rag_method=args.rag_method,
                rag_document_unit=args.rag_document_unit,
                rag_selection_policy=args.rag_selection_policy,
                rag_render_order=args.rag_render_order,
                rag_embedding_model=args.rag_embedding_model,
                rag_embedding_device=args.rag_embedding_device,
                rag_embedding_gpu_device_ids=args.rag_embedding_gpu_device_ids,
                rag_embedding_batch_size=args.rag_embedding_batch_size,
                rag_embedding_max_length=args.rag_embedding_max_length,
                rag_model_max_len=args.rag_model_max_len,
                rag_model_tensor_parallel_size=args.rag_model_tensor_parallel_size,
                evaluation_root=Path(args.evaluation_root).expanduser().resolve() if args.evaluation_root else None,
            )
            rag_pipeline = RagEvaluationPipeline(config, settings)
            summary = rag_pipeline.run(targets)
            print(json.dumps(summary, indent=2))
            return 0 if not summary["failed"] else 1

        if args.command == "refresh-summary-tokens":
            subject_ids = None
            if args.subject_id is not None:
                subject_ids = [int(args.subject_id)]
            elif args.subject_ids:
                subject_ids = [int(subject_id) for subject_id in args.subject_ids]
            summary = refresh_patient_summary_tokens(
                benchmark_root=config.output.root,
                patient_manifest=Path(args.patient_manifest).expanduser().resolve() if args.patient_manifest else None,
                subject_ids=subject_ids,
                tokenizer_model=str(args.tokenizer_model),
            )
            print(json.dumps(summary, indent=2))
            return 0

        if args.command == "summarize-evaluation-cohort":
            evaluation_root = (
                Path(args.evaluation_root).expanduser().resolve()
                if args.evaluation_root
                else config.output.root.parent / "evaluation"
            )
            summary = summarize_evaluation_cohort(
                evaluation_root=evaluation_root,
                patient_manifest=Path(args.patient_manifest).expanduser().resolve(),
            )
            print(json.dumps(summary, indent=2))
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
