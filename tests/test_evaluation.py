from __future__ import annotations

import csv
import json
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import main as main_cli

from health_benchmark.scripts.config import build_default_config
from health_benchmark.scripts.llm_client import StructuredResponseValidationError, structured_content_candidates
from health_benchmark.evaluation.answer_prompting import render_answer_prompt
from health_benchmark.evaluation.answer_runner import run_answer_batches, summarize_schema_repair_metrics
from health_benchmark.evaluation.batch_builder import build_batches
from health_benchmark.evaluation.config import MODEL_ALIASES, build_rag_settings, build_settings
from health_benchmark.evaluation.judge_runner import run_llm_judge_batches
from health_benchmark.evaluation.judge_prompting import render_answerable_judge_prompt
from health_benchmark.evaluation.loader import resolve_patient_targets
from health_benchmark.evaluation.memory import (
    DenseMemoryRetriever,
    Mem0MemoryStore,
    Mem0ExtractionResponse,
    MemAlphaExtractionResponse,
    Mem0UpdateResponse,
    _generate_memory_response,
    build_mem0_memory_store,
    order_memories_for_answer_context,
    render_mem0_context,
    retrieve_existing_for_candidates,
    select_mem0_context_for_question,
)
from health_benchmark.evaluation.memory_pipeline import MemoryEvaluationPipeline, memory_model_spec_for
from health_benchmark.evaluation.pipeline import EvaluationPipeline, normalize_benchmark
from health_benchmark.evaluation.rag import (
    BM25AdmissionRetriever,
    DenseAdmissionRetriever,
    build_admission_documents,
    build_rag_question_batches,
    render_rag_context,
    select_rag_context_for_question,
)
from health_benchmark.evaluation.rag_pipeline import RagEvaluationPipeline, rag_model_spec_for
from health_benchmark.evaluation.scoring import normalize_answer, score_adversarial, score_answerable, score_predictions
from health_benchmark.evaluation.token_budget import build_preflight_record, select_context_for_batch
from health_benchmark.evaluation.types import AnswerableJudgeBatchResponse, CANONICAL_ABSTENTION_ANSWER, ModelSpec


class FakeHFTokenizer:
    def __init__(self) -> None:
        self.chat_template_calls: list[dict[str, object]] = []

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    ):
        self.chat_template_calls.append(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
                "enable_thinking": enable_thinking,
            }
        )
        rendered = "\n".join(f"<|{message['role']}|>{message['content']}" for message in messages)
        if add_generation_prompt:
            rendered += "\n<|assistant|>"
        if not enable_thinking:
            rendered += "<think></think>"
        return rendered

    def encode(self, text, *, add_special_tokens=False):
        del add_special_tokens
        return list(range(len(re.findall(r"\S+", str(text or "")))))


class FakeLLMClient:
    def __init__(self, payloads: list[dict[str, object]]) -> None:
        self.payloads = list(payloads)
        self.calls: list[dict[str, object]] = []

    def generate_structured_response(
        self,
        system_message: str,
        user_message: str,
        response_schema,
        *,
        max_output_tokens=None,
    ):
        self.calls.append(
            {
                "system_message": system_message,
                "user_message": user_message,
                "schema": getattr(response_schema, "__name__", str(response_schema)),
                "max_output_tokens": max_output_tokens,
            }
        )
        payload = self.payloads.pop(0)
        parsed = response_schema.model_validate(payload["parsed_output"])
        return SimpleNamespace(
            parsed_output=parsed.model_dump(mode="json"),
            raw_response=payload.get("raw_response", {"fake": True}),
            usage=payload.get(
                "usage",
                {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18},
            ),
            response_id=payload.get("response_id", "resp_fake"),
            latency_ms=payload.get("latency_ms", 5),
        )


class FakeDenseEmbedder:
    model_name = "fake-dense-embedder"
    device = "cpu"
    batch_size = 8
    max_length = 128
    embedding_dimension = 4

    def embed(self, texts):
        return [self._embed_one(text) for text in texts]

    @staticmethod
    def _embed_one(text):
        lowered = str(text or "").lower()
        vector = [0.0, 0.0, 0.0, 0.0]
        groups = [
            ("fever", "dialysis", "cellulitis", "diagnosis"),
            ("oxygen", "diuresis", "respiratory"),
            ("catheter", "line", "drainage"),
            ("walk", "discharge", "medication"),
        ]
        for index, terms in enumerate(groups):
            vector[index] = float(sum(lowered.count(term) for term in terms))
        if not any(vector):
            vector[3] = 1.0
        return vector


def fake_dense_retriever() -> DenseMemoryRetriever:
    return DenseMemoryRetriever(FakeDenseEmbedder())


class RaisingLLMClient:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc
        self.calls: list[dict[str, object]] = []

    def generate_structured_response(
        self,
        system_message: str,
        user_message: str,
        response_schema,
        *,
        max_output_tokens=None,
    ):
        self.calls.append(
            {
                "system_message": system_message,
                "user_message": user_message,
                "schema": getattr(response_schema, "__name__", str(response_schema)),
                "max_output_tokens": max_output_tokens,
            }
        )
        raise self.exc


class RawContentLLMClient:
    def __init__(self, payloads: list[dict[str, object]]) -> None:
        self.payloads = list(payloads)
        self.calls: list[dict[str, object]] = []

    def generate_structured_response(
        self,
        system_message: str,
        user_message: str,
        response_schema,
        *,
        max_output_tokens=None,
    ):
        self.calls.append(
            {
                "system_message": system_message,
                "user_message": user_message,
                "schema": getattr(response_schema, "__name__", str(response_schema)),
                "max_output_tokens": max_output_tokens,
            }
        )
        payload = self.payloads.pop(0)
        if "content" in payload:
            content = str(payload["content"])
            raise StructuredResponseValidationError(
                "vLLM response did not match expected schema: fake validation error",
                content=content,
                raw_response={"choices": [{"message": {"content": content}}]},
                usage=payload.get(
                    "usage",
                    {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18},
                ),
                response_id=str(payload.get("response_id", "resp_raw")),
                latency_ms=int(payload.get("latency_ms", 5)),
                schema_name=getattr(response_schema, "__name__", str(response_schema)),
            )
        parsed = response_schema.model_validate(payload["parsed_output"])
        return SimpleNamespace(
            parsed_output=parsed.model_dump(mode="json"),
            raw_response=payload.get("raw_response", {"fake": True}),
            usage=payload.get(
                "usage",
                {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18},
            ),
            response_id=payload.get("response_id", "resp_fake"),
            latency_ms=payload.get("latency_ms", 5),
        )


class EvaluationTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.output_root = self.root / "output" / "benchmark"
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.fake_tokenizer = FakeHFTokenizer()
        self.tokenizer_patch = patch(
            "health_benchmark.evaluation.hf_tokenizer._load_hf_tokenizer",
            return_value=self.fake_tokenizer,
        )
        self.tokenizer_patch.start()

    def tearDown(self) -> None:
        self.tokenizer_patch.stop()
        self.temp_dir.cleanup()

    def _config(self):
        project_dir = Path(__file__).resolve().parents[1] / "health_benchmark"
        config = build_default_config(project_dir)
        config.output.root = self.output_root
        return config

    def _write_json(self, path: Path, payload: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def _patient_root(self, subject_id: int = 11826927) -> Path:
        return self.output_root / str(subject_id)

    def _evaluation_root(self, subject_id: int = 11826927) -> Path:
        return self.output_root.parent / "evaluation" / str(subject_id)

    def _benchmark_payload(self) -> dict[str, object]:
        qas: list[dict[str, object]] = []
        for index in range(1, 13):
            qa_id = f"q{index:02d}"
            if index in {4, 11}:
                qas.append(
                    {
                        "qa_id": qa_id,
                        "scope": "single_admission" if index == 4 else "cross_admission",
                        "question_type": "adversarial",
                        "question": f"Unsupported question {index}?",
                        "answer": CANONICAL_ABSTENTION_ANSWER,
                        "evidence": {"admissions": ["2001", "2002"] if index == 11 else ["2001"], "turn_ids": [1, 2] if index == 4 else None},
                    }
                )
                if index == 11:
                    qas[-1]["evidence"] = {"admissions": ["2001", "2002"]}
                continue
            qas.append(
                {
                    "qa_id": qa_id,
                    "scope": "single_admission" if index <= 6 else "cross_admission",
                    "question_type": "medical_reasoning" if index <= 6 else "longitudinal_progression",
                    "question": f"Answerable question {index}?",
                    "answer": f"answer {index}",
                    "evidence": {"admissions": ["2001"], "turn_ids": [1, 2]} if index <= 6 else {"admissions": ["2001", "2002"]},
                }
            )
        return {"qas": qas}

    def _write_patient_artifacts(self, subject_id: int = 11826927) -> Path:
        patient_root = self._patient_root(subject_id)
        combined_payload = {
            "subject_id": str(subject_id),
            "processed_hadm_ids": ["2001", "2002"],
            "admissions": [
                {
                    "hadm_id": "2001",
                    "admission_start": "2020-01-01 08:00:00",
                    "admission_end": "2020-01-02 09:00:00",
                    "conversation_lines": [
                        {
                            "turn_number": 1,
                            "time": "2020-01-01 08:00:00",
                            "speaker": "Doctor",
                            "text": "We think the fever improved with dialysis.",
                        },
                        {
                            "turn_number": 2,
                            "time": "2020-01-01 08:05:00",
                            "speaker": "Patient",
                            "text": "The cough is better today.",
                        },
                    ],
                },
                {
                    "hadm_id": "2002",
                    "admission_start": "2020-02-01 10:00:00",
                    "admission_end": "2020-02-03 14:00:00",
                    "conversation_lines": [
                        {
                            "turn_number": 1,
                            "time": "2020-02-01 10:00:00",
                            "speaker": "Doctor",
                            "text": "Your breathing improved after dialysis.",
                        },
                        {
                            "turn_number": 2,
                            "time": "2020-02-01 10:10:00",
                            "speaker": "Patient",
                            "text": "I can walk farther now.",
                        },
                    ],
                },
            ],
        }
        self._write_json(patient_root / "combined_conversation.json", combined_payload)
        self._write_json(patient_root / "benchmark_qa.json", self._benchmark_payload())
        return patient_root

    def _predicted_answers_payload(self, start_index: int, end_index: int) -> dict[str, object]:
        answers = []
        for index in range(start_index, end_index + 1):
            qa_id = f"q{index:02d}"
            prediction = CANONICAL_ABSTENTION_ANSWER if index in {4, 11} else f"answer {index}"
            answers.append({"qa_id": qa_id, "prediction": prediction})
        return {"answers": answers}

    def _two_question_answer_batch(self):
        patient_root = self._write_patient_artifacts()
        combined_payload = json.loads((patient_root / "combined_conversation.json").read_text(encoding="utf-8"))
        questions = normalize_benchmark(
            {
                "qas": [
                    {
                        "qa_id": "q1",
                        "scope": "single_admission",
                        "question_type": "medical_reasoning",
                        "question": "What helped first?",
                        "answer": "dialysis",
                        "evidence": {"admissions": ["1"], "turn_ids": [1]},
                    },
                    {
                        "qa_id": "q2",
                        "scope": "single_admission",
                        "question_type": "medical_reasoning",
                        "question": "What helped second?",
                        "answer": "antibiotics",
                        "evidence": {"admissions": ["1"], "turn_ids": [2]},
                    },
                ]
            }
        )
        settings = build_settings(
            self._config(),
            provider="vllm",
            base_url=None,
            judge_base_url=None,
            api_key_env=None,
            models=["Qwen/Qwen3.5-4B"],
            replace_existing=None,
            retry_limit=0,
        )
        batches = build_batches(
            questions,
            combined_payload=combined_payload,
            settings=settings,
            model_spec=settings.model_specs[0],
        )
        return questions, settings, batches[:1]

    def _judge_answerable_payload(self, qa_ids: list[str], *, score: int = 1) -> dict[str, object]:
        return {
            "judgments": [{"qa_id": qa_id, "score": score} for qa_id in qa_ids]
        }

    def test_render_answer_prompt_uses_open_answer_contract(self) -> None:
        rendered = render_answer_prompt(
            context_text="context",
            questions=[{"qa_id": "q1", "question": "What happened?"}],
        )
        payload = json.loads(rendered.user_message)
        self.assertEqual(payload["questions"], [{"qa_id": "q1", "question": "What happened?"}])
        self.assertEqual(payload["patient_conversation"], "context")
        self.assertIn("Use only the provided conversation.", rendered.system_message)
        self.assertIn(
            'answer exactly "the question is not answerable"',
            rendered.system_message,
        )
        self.assertNotIn("partial patient facts", rendered.system_message)
        self.assertNotIn("compare or synthesize across relevant memories", rendered.system_message)
        self.assertNotIn("directly support the requested fact", rendered.system_message)
        self.assertNotIn("multiple_choice", rendered.system_message)

    def test_render_memory_answer_prompt_uses_less_abstention_heavy_contract(self) -> None:
        rendered = render_answer_prompt(
            context_text="=== Patient Memory ===\nRetrieved memories:\n1. On 2020-01-01, dialysis helped fever.",
            questions=[{"qa_id": "q1", "question": "What helped the fever?"}],
            context_description="patient memory context",
            context_payload_key="patient_memory_context",
        )
        payload = json.loads(rendered.user_message)
        self.assertEqual(payload["patient_memory_context"].splitlines()[0], "=== Patient Memory ===")
        self.assertIn("partial patient facts", rendered.system_message)
        self.assertIn("directly support the requested fact, relationship, comparison, or temporal change", rendered.system_message)
        self.assertIn("specific evidence for the compared items or timepoints", rendered.system_message)
        self.assertIn("Related clinical facts are not enough by themselves", rendered.system_message)
        self.assertIn("do not support the requested fact or relationship", rendered.system_message)
        self.assertIn("Only answer exactly", rendered.system_message)
        self.assertIn("Prefer exact wording from the retrieved memories", rendered.system_message)
        self.assertIn("Do not invent details", rendered.system_message)
        self.assertNotIn("Do not abstain just because no single memory uses the exact wording", rendered.system_message)
        self.assertNotIn("compare or synthesize across relevant memories", rendered.system_message)
        self.assertIn('return strict JSON with the schema {"answers"', rendered.system_message)
        self.assertNotIn(
            'If the answer is not supported by the patient memory context',
            rendered.system_message,
        )

    def test_render_llm_judge_prompt_uses_minimal_answerable_inputs(self) -> None:
        rendered_answerable = render_answerable_judge_prompt(
            [
                {
                    "qa_id": "q1",
                    "question": "What happened?",
                    "gold_answer": "dialysis",
                    "candidate_answer": "dialysis",
                }
            ]
        )
        answerable_payload = json.loads(rendered_answerable.user_message)
        self.assertEqual(
            answerable_payload["items"],
            [
                {
                    "qa_id": "q1",
                    "question": "What happened?",
                    "gold_answer": "dialysis",
                    "candidate_answer": "dialysis",
                }
            ],
        )
        self.assertIn("Each score must be exactly one of: 0, 1.", rendered_answerable.system_message)
        self.assertIn("Score 1 when the candidate answer is correct.", rendered_answerable.system_message)
        self.assertNotIn("0.5", rendered_answerable.system_message)
        with self.assertRaises(Exception):
            AnswerableJudgeBatchResponse.model_validate(
                {"judgments": [{"qa_id": "q1", "score": 0.5}]}
            )
        AnswerableJudgeBatchResponse.model_validate(
            {"judgments": [{"qa_id": "q1", "score": 1}]}
        )

    def test_build_settings_uses_fair_context_and_output_defaults(self) -> None:
        settings = build_settings(
            self._config(),
            provider="vllm",
            base_url="http://127.0.0.1:8000/v1",
            judge_base_url="http://127.0.0.1:8001/v1",
            api_key_env=None,
            models=["Qwen/Qwen3.5-4B"],
            replace_existing=True,
        )
        self.assertEqual(settings.judge_max_output_tokens, 1024)
        self.assertEqual(settings.max_output_tokens, 4096)
        self.assertEqual(settings.safe_margin_tokens, 8192)
        self.assertEqual(settings.token_estimate_safety_multiplier, 1.0)
        self.assertEqual(settings.evaluation_root, self.output_root.parent / "evaluation")
        self.assertEqual(settings.model_specs[0].max_model_len, 131072)
        self.assertEqual(settings.evaluation_variant, "normal")
        self.assertEqual(settings.mem0_chunk_token_cap, 12000)
        self.assertEqual(settings.mem0_previous_chunk_summaries, 1)
        self.assertEqual(settings.mem0_max_candidate_memories, 32)
        self.assertEqual(settings.mem0_similar_memories, 10)
        self.assertEqual(settings.mem0_max_update_memories, 40)
        self.assertEqual(settings.mem0_answer_retrieval_top_k, 64)
        self.assertEqual(settings.mem0_max_answer_memories, 32)
        self.assertEqual(settings.mem0_max_output_tokens, 4096)
        self.assertEqual(settings.mem0_retrieval_backend, "local_dense_hf")
        self.assertEqual(settings.mem0_embedding_model, "Qwen/Qwen3-Embedding-8B")
        self.assertEqual(settings.mem0_embedding_device, "cuda")
        self.assertEqual(settings.mem0_embedding_batch_size, 8)
        self.assertEqual(settings.mem0_embedding_max_length, 1024)
        self.assertEqual(settings.judge_model_spec.model_name, "Qwen/Qwen3.5-27B")
        self.assertEqual(settings.judge_model_spec.slug, "qwen3.5-27b")

    def test_build_settings_accepts_openai_api_judge_model(self) -> None:
        settings = build_settings(
            self._config(),
            provider="openai",
            base_url=None,
            judge_base_url=None,
            judge_model="gpt-5.1",
            api_key_env="OPENAI_API_KEY",
            models=["gpt-5.1"],
            replace_existing=True,
        )

        self.assertEqual(settings.provider, "openai")
        self.assertIsNone(settings.base_url)
        self.assertIsNone(settings.judge_base_url)
        self.assertEqual(settings.model_specs[0].model_name, "gpt-5.1")
        self.assertEqual(settings.model_specs[0].slug, "gpt-5.1")
        self.assertEqual(settings.judge_model_spec.model_name, "gpt-5.1")
        self.assertEqual(settings.judge_model_spec.slug, "gpt-5.1")
        self.assertEqual(settings.judge_model_spec.tensor_parallel_size, 1)

    def test_hf_tokenizer_snapshot_resolver_ignores_config_only_cache(self) -> None:
        from health_benchmark.evaluation.hf_tokenizer import _resolve_local_snapshot

        cache_dir = self.root / "hf_cache" / "hub"
        model_cache = cache_dir / "models--google--medgemma-4b-it"
        snapshot = model_cache / "snapshots" / "revision-a"
        snapshot.mkdir(parents=True)
        (model_cache / "refs").mkdir()
        (model_cache / "refs" / "main").write_text("revision-a\n", encoding="utf-8")
        (snapshot / "config.json").write_text("{}", encoding="utf-8")

        self.assertIsNone(_resolve_local_snapshot("google/medgemma-4b-it", cache_dir))

        (snapshot / "tokenizer_config.json").write_text("{}", encoding="utf-8")

        self.assertEqual(
            _resolve_local_snapshot("google/medgemma-4b-it", cache_dir),
            str(snapshot.resolve()),
        )

    def test_build_settings_resolves_gemma3_trio_metadata(self) -> None:
        settings = build_settings(
            self._config(),
            provider="vllm",
            base_url="http://127.0.0.1:8000/v1",
            judge_base_url="http://127.0.0.1:8001/v1",
            api_key_env=None,
            models=[
                "google/gemma-3-4b-it",
                "gemma-3-12b-it",
                "gemma3-27b-it",
            ],
            replace_existing=True,
        )

        self.assertEqual(
            [spec.model_name for spec in settings.model_specs],
            [
                "google/gemma-3-4b-it",
                "google/gemma-3-12b-it",
                "google/gemma-3-27b-it",
            ],
        )
        self.assertEqual(
            [spec.slug for spec in settings.model_specs],
            ["gemma-3-4b-it", "gemma-3-12b-it", "gemma-3-27b-it"],
        )
        self.assertEqual([spec.tensor_parallel_size for spec in settings.model_specs], [1, 1, 2])
        self.assertEqual([spec.max_model_len for spec in settings.model_specs], [131072, 131072, 131072])
        self.assertEqual(settings.max_output_tokens, 4096)
        self.assertEqual(settings.safe_margin_tokens, 8192)

    def test_build_settings_resolves_medgemma_pair_metadata(self) -> None:
        settings = build_settings(
            self._config(),
            provider="vllm",
            base_url="http://127.0.0.1:8000/v1",
            judge_base_url="http://127.0.0.1:8001/v1",
            api_key_env=None,
            models=["google/medgemma-4b-it", "medgemma27b-it"],
            replace_existing=True,
        )

        self.assertEqual(
            [spec.model_name for spec in settings.model_specs],
            ["google/medgemma-4b-it", "google/medgemma-27b-it"],
        )
        self.assertEqual(
            [spec.slug for spec in settings.model_specs],
            ["medgemma-4b-it", "medgemma-27b-it"],
        )
        self.assertEqual([spec.tensor_parallel_size for spec in settings.model_specs], [1, 2])
        self.assertEqual([spec.max_model_len for spec in settings.model_specs], [131072, 131072])
        self.assertEqual(settings.max_output_tokens, 4096)
        self.assertEqual(settings.safe_margin_tokens, 8192)

    def test_build_settings_resolves_medical_next_metadata(self) -> None:
        settings = build_settings(
            self._config(),
            provider="vllm",
            base_url="http://127.0.0.1:8000/v1",
            judge_base_url="http://127.0.0.1:8001/v1",
            api_key_env=None,
            models=[
                "MBZUAI/MedMO-4B-Next",
                "medmo8b-next",
                "mediphi",
                "lingshu",
            ],
            replace_existing=True,
        )

        self.assertNotIn("baichuan-m2", MODEL_ALIASES)
        self.assertNotIn("baichuanm2", MODEL_ALIASES)
        self.assertEqual(
            [spec.model_name for spec in settings.model_specs],
            [
                "MBZUAI/MedMO-4B-Next",
                "MBZUAI/MedMO-8B-Next",
                "microsoft/MediPhi-Instruct",
                "lingshu-medical-mllm/Lingshu-32B",
            ],
        )
        self.assertEqual(
            [spec.slug for spec in settings.model_specs],
            [
                "medmo-4b-next",
                "medmo-8b-next",
                "mediphi-instruct",
                "lingshu-32b",
            ],
        )
        self.assertEqual([spec.tensor_parallel_size for spec in settings.model_specs], [1, 1, 1, 2])
        self.assertEqual(
            [spec.max_model_len for spec in settings.model_specs],
            [131072, 131072, 131072, 128000],
        )
        self.assertEqual(settings.max_output_tokens, 4096)
        self.assertEqual(settings.safe_margin_tokens, 8192)

    def test_memory_settings_can_override_model_runtime_shape(self) -> None:
        settings = build_settings(
            self._config(),
            provider="vllm",
            base_url="http://127.0.0.1:8000/v1",
            judge_base_url=None,
            api_key_env=None,
            models=["Qwen/Qwen3.5-27B"],
            evaluation_variant="mem0",
            replace_existing=True,
            mem0_model_max_len=32768,
            mem0_model_tensor_parallel_size=1,
        )
        self.assertEqual(settings.model_specs[0].max_model_len, 32768)
        self.assertEqual(settings.model_specs[0].tensor_parallel_size, 1)
        self.assertEqual(settings.mem0_model_max_len, 32768)
        self.assertEqual(settings.mem0_model_tensor_parallel_size, 1)

    def test_rag_settings_can_override_model_runtime_shape(self) -> None:
        settings = build_rag_settings(
            self._config(),
            provider="vllm",
            base_url="http://127.0.0.1:8000/v1",
            judge_base_url=None,
            api_key_env=None,
            models=["google/gemma-3-27b-it"],
            replace_existing=True,
            rag_method="embedding-rag",
            rag_model_max_len=32768,
            rag_model_tensor_parallel_size=1,
        )

        self.assertEqual(settings.evaluation_variant, "embedding_rag")
        self.assertEqual(settings.model_specs[0].max_model_len, 32768)
        self.assertEqual(settings.model_specs[0].tensor_parallel_size, 1)
        self.assertEqual(settings.rag_embedding_model, "Qwen/Qwen3-Embedding-8B")

    def test_normalize_benchmark_rejects_legacy_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "legacy fields"):
            normalize_benchmark(
                {
                    "qas": [
                        {
                            "qa_id": "q1",
                            "scope": "single_admission",
                            "question_type": "medical_reasoning",
                            "question": "What happened?",
                            "answer": "dialysis",
                            "options": ["A", "B", "C", "D"],
                            "evidence": {"admissions": ["1"], "turn_ids": [1]},
                        }
                    ]
                }
            )

    def test_build_batches_uses_fixed_ten_with_remainder(self) -> None:
        patient_root = self._write_patient_artifacts()
        combined_payload = json.loads((patient_root / "combined_conversation.json").read_text(encoding="utf-8"))
        questions = normalize_benchmark(self._benchmark_payload())
        settings = build_settings(
            self._config(),
            provider="vllm",
            base_url=None,
            judge_base_url=None,
            api_key_env=None,
            models=["Qwen/Qwen3.5-4B"],
            replace_existing=None,
        )
        batches = build_batches(
            questions,
            combined_payload=combined_payload,
            settings=settings,
            model_spec=settings.model_specs[0],
        )
        self.assertEqual([len(batch.questions) for batch in batches], [10, 2])
        self.assertEqual(batches[0].qa_ids()[0], "q01")
        self.assertEqual(batches[1].qa_ids(), ["q11", "q12"])
        self.assertEqual(batches[0].model_payload()["questions"][0], {"qa_id": "q01", "question": "Answerable question 1?"})
        self.assertNotIn("Admission 1", batches[0].context_text)
        self.assertNotIn("admission_start", batches[0].context_text)
        self.assertNotIn("admission_end", batches[0].context_text)
        self.assertNotIn("hadm_id", batches[0].context_text)
        self.assertNotIn("1 | 2020-01-01", batches[0].context_text)
        self.assertIn("2020-01-01 08:00:00 | Doctor |", batches[0].context_text)

    def test_preflight_marks_truncation_required_without_failing_fixed_batch_ten(self) -> None:
        questions = normalize_benchmark(self._benchmark_payload())
        record = build_preflight_record(
            model_name="Qwen/Qwen3.5-4B",
            tokenizer_name=None,
            context_text="tiny context",
            questions=questions,
            batch_size=10,
            max_model_len=20,
            max_output_tokens=5,
            safe_margin_tokens=0,
            token_estimate_safety_multiplier=1.0,
        )
        self.assertEqual(record["status"], "truncation_required")
        self.assertEqual(record["batch_size"], 10)

    def test_preflight_uses_hf_chat_template_and_no_default_multiplier(self) -> None:
        questions = normalize_benchmark(self._benchmark_payload())
        record = build_preflight_record(
            model_name="Qwen/Qwen3.5-4B",
            tokenizer_name=None,
            context_text="clinical detail " * 120,
            questions=questions,
            batch_size=10,
            max_model_len=1_000_000,
            max_output_tokens=0,
            safe_margin_tokens=0,
            token_estimate_safety_multiplier=1.0,
        )
        raw_estimate = int(record["estimated_full_prompt_tokens"])
        adjusted_estimate = int(record["adjusted_estimated_full_prompt_tokens"])
        self.assertEqual(adjusted_estimate, raw_estimate)
        self.assertEqual(record["tokenizer"], "Qwen/Qwen3.5-4B")
        self.assertEqual(record["effective_raw_prompt_budget_tokens"], 1_000_000)
        self.assertTrue(self.fake_tokenizer.chat_template_calls)
        call = self.fake_tokenizer.chat_template_calls[-1]
        self.assertTrue(call["add_generation_prompt"])
        self.assertFalse(call["enable_thinking"])
        self.assertEqual([message["role"] for message in call["messages"]], ["system", "user"])

        fitting_record = build_preflight_record(
            model_name="Qwen/Qwen3.5-4B",
            tokenizer_name=None,
            context_text="clinical detail " * 120,
            questions=questions,
            batch_size=10,
            max_model_len=raw_estimate + 1,
            max_output_tokens=0,
            safe_margin_tokens=0,
            token_estimate_safety_multiplier=1.0,
        )
        self.assertEqual(fitting_record["status"], "full_context_fits")

    def test_preflight_uses_tiktoken_for_openai_api_models(self) -> None:
        questions = normalize_benchmark(self._benchmark_payload())
        record = build_preflight_record(
            model_name="gpt-5.1",
            tokenizer_name=None,
            context_text="clinical detail " * 20,
            questions=questions,
            batch_size=10,
            max_model_len=131072,
            max_output_tokens=4096,
            safe_margin_tokens=8192,
            token_estimate_safety_multiplier=1.0,
        )

        self.assertEqual(record["status"], "full_context_fits")
        self.assertTrue(str(record["tokenizer"]).startswith("tiktoken:"))
        self.assertFalse(self.fake_tokenizer.chat_template_calls)

    def test_build_batches_truncates_recent_first_under_prompt_budget(self) -> None:
        questions = normalize_benchmark(self._benchmark_payload())[:10]
        combined_payload = {
            "subject_id": "11826927",
            "processed_hadm_ids": ["2001"],
            "admissions": [
                {
                    "hadm_id": "2001",
                    "admission_start": "2020-01-01 08:00:00",
                    "admission_end": "2020-01-02 09:00:00",
                    "conversation_lines": [
                        {
                            "turn_number": index,
                            "time": f"2020-01-01 08:{index:02d}:00",
                            "speaker": "Doctor",
                            "text": ("oldest " if index == 1 else "recent ") + ("clinical detail " * 40),
                        }
                        for index in range(1, 31)
                    ],
                }
            ],
        }
        settings = build_settings(
            self._config(),
            provider="vllm",
            base_url=None,
            judge_base_url=None,
            api_key_env=None,
            models=["Qwen/Qwen3.5-4B"],
            replace_existing=None,
            max_output_tokens=16,
            safe_margin_tokens=0,
        )
        batches = build_batches(
            questions,
            combined_payload=combined_payload,
            settings=settings,
            model_spec=ModelSpec("Qwen/Qwen3.5-4B", "qwen3.5-4b", 1, 1800),
        )
        self.assertTrue(batches[0].context_record["was_truncated"])
        self.assertEqual(batches[0].context_record["strategy"], "recent_first")
        self.assertLess(batches[0].context_record["selected_turns"], 30)
        self.assertNotIn("oldest", batches[0].context_text)
        self.assertIn("recent", batches[0].context_text)
        self.assertLessEqual(
            batches[0].adjusted_estimated_prompt_tokens,
            batches[0].context_record["effective_prompt_budget_tokens"],
        )

    def test_context_selection_uses_raw_hf_estimate_without_extra_multiplier(self) -> None:
        questions = normalize_benchmark(self._benchmark_payload())[:10]
        combined_payload = {
            "subject_id": "11826927",
            "processed_hadm_ids": ["2001"],
            "admissions": [
                {
                    "hadm_id": "2001",
                    "admission_start": "2020-01-01 08:00:00",
                    "admission_end": "2020-01-02 09:00:00",
                    "conversation_lines": [
                        {
                            "turn_number": index,
                            "time": f"2020-01-01 08:{index:02d}:00",
                            "speaker": "Doctor",
                            "text": ("oldest " if index == 1 else "recent ") + ("clinical detail " * 25),
                        }
                        for index in range(1, 31)
                    ],
                }
            ],
        }
        loose_selection = select_context_for_batch(
            combined_payload=combined_payload,
            batch_questions=questions,
            model_name="Qwen/Qwen3.5-4B",
            tokenizer_name=None,
            max_model_len=1_000_000,
            max_output_tokens=0,
            safe_margin_tokens=0,
            token_estimate_safety_multiplier=1.0,
        )
        raw_full_estimate = int(loose_selection["estimated_prompt_tokens"])
        strict_selection = select_context_for_batch(
            combined_payload=combined_payload,
            batch_questions=questions,
            model_name="Qwen/Qwen3.5-4B",
            tokenizer_name=None,
            max_model_len=raw_full_estimate,
            max_output_tokens=0,
            safe_margin_tokens=0,
            token_estimate_safety_multiplier=1.0,
        )

        self.assertEqual(int(loose_selection["adjusted_estimated_prompt_tokens"]), raw_full_estimate)
        self.assertFalse(strict_selection["context_record"]["was_truncated"])
        self.assertEqual(strict_selection["context_record"]["strategy"], "full_context")
        self.assertLessEqual(int(strict_selection["adjusted_estimated_prompt_tokens"]), raw_full_estimate)

    def test_mem0_context_selection_retrieves_question_memory_without_source_ids(self) -> None:
        questions = normalize_benchmark(
            {
                "qas": [
                    {
                        "qa_id": "q1",
                        "scope": "single_admission",
                        "question_type": "medical_reasoning",
                        "question": "What helped the fever?",
                        "answer": "dialysis",
                        "evidence": {"admissions": ["2001"], "turn_ids": [1]},
                    }
                ]
            }
        )
        memory_store = Mem0MemoryStore(
            subject_id="11826927",
            summary="The first admission discussed fever and dialysis.",
            retriever=fake_dense_retriever(),
            records=[
                {
                    "memory_id": "mem_00001",
                    "memory": "During the first admission, the fever improved with dialysis.",
                    "deleted": False,
                    "created_at_chunk_index": 1,
                    "updated_at_chunk_index": 1,
                    "source_turn_ids": ["hadm=2001:turn=1:global=1"],
                },
                {
                    "memory_id": "mem_00002",
                    "memory": "During the second admission, the patient could walk farther.",
                    "deleted": False,
                    "created_at_chunk_index": 2,
                    "updated_at_chunk_index": 2,
                    "source_turn_ids": ["hadm=2002:turn=2:global=4"],
                },
            ],
        )

        selection = select_mem0_context_for_question(
            memory_store=memory_store,
            question=questions[0],
            model_name="Qwen/Qwen3.5-4B",
            tokenizer_name=None,
            max_model_len=4096,
            max_output_tokens=128,
            safe_margin_tokens=0,
            token_estimate_safety_multiplier=1.0,
            retrieval_top_k=10,
            max_answer_memories=10,
        )

        self.assertEqual(selection["context_record"]["strategy"], "mem0")
        self.assertEqual(selection["context_record"]["prompt_context_payload_key"], "patient_memory_context")
        self.assertIn("fever improved with dialysis", selection["context_text"])
        self.assertNotIn("Conversation summary:", selection["context_text"])
        self.assertNotIn("The first admission discussed fever and dialysis", selection["context_text"])
        self.assertEqual(selection["context_record"]["retrieved_memory_ids"][0], "mem_00001")
        selected_record = selection["context_record"]["selected_memory_records"][0]
        self.assertEqual(selected_record["memory_id"], "mem_00001")
        self.assertIn("fever improved with dialysis", selected_record["memory"])
        self.assertIn("dense_score", selected_record)
        self.assertIn("retrieval_score", selected_record)
        self.assertIn("selected_rank", selected_record)
        rendered_record = selection["context_record"]["rendered_memory_records"][0]
        self.assertEqual(rendered_record["memory_id"], "mem_00001")
        self.assertEqual(rendered_record["rendered_order"], 1)
        self.assertIn("selected_rank", rendered_record)
        self.assertNotIn("mem_00001", selection["context_text"])
        self.assertNotIn("dense_score", selection["context_text"])
        self.assertNotIn("retrieval_score", selection["context_text"])
        self.assertNotIn("selected_rank", selection["context_text"])
        self.assertNotIn("rendered_order", selection["context_text"])
        self.assertNotIn("hadm=", selection["context_text"])
        self.assertNotIn("turn=", selection["context_text"])

    def test_mem0_answer_context_renders_selected_memories_chronologically(self) -> None:
        selected_memories = [
            {
                "memory_id": "mem_late",
                "memory": "On 2020-02-03 09:00, fever improved after dialysis.",
                "dense_score": 0.99,
                "retrieval_score": 0.99,
            },
            {
                "memory_id": "mem_undated_first",
                "memory": "Fever was discussed without a visible timestamp.",
                "dense_score": 0.95,
                "retrieval_score": 0.95,
            },
            {
                "memory_id": "mem_early",
                "memory": "On 2020-02-01, fever worsened before dialysis.",
                "dense_score": 0.90,
                "retrieval_score": 0.90,
            },
            {
                "memory_id": "mem_undated_second",
                "memory": "Dialysis was discussed without a visible timestamp.",
                "dense_score": 0.80,
                "retrieval_score": 0.80,
            },
        ]

        rendered_memories = order_memories_for_answer_context(selected_memories)
        self.assertEqual(
            [record["memory_id"] for record in rendered_memories],
            ["mem_early", "mem_late", "mem_undated_first", "mem_undated_second"],
        )
        self.assertEqual(rendered_memories[0]["_selected_rank"], 3)
        self.assertEqual(rendered_memories[2]["_selected_rank"], 2)

        context_text = render_mem0_context("", rendered_memories)
        self.assertIn("chronological order when timestamps are available", context_text)
        self.assertIn("Use only memories relevant to the question", context_text)
        self.assertIn("some answers may require more than one memory", context_text)
        self.assertNotIn("Multiple memories may jointly support the answer", context_text)
        self.assertLess(
            context_text.index("On 2020-02-01"),
            context_text.index("On 2020-02-03"),
        )
        self.assertLess(
            context_text.index("On 2020-02-03"),
            context_text.index("Fever was discussed without a visible timestamp"),
        )
        self.assertLess(
            context_text.index("Fever was discussed without a visible timestamp"),
            context_text.index("Dialysis was discussed without a visible timestamp"),
        )
        self.assertNotIn("mem_late", context_text)
        self.assertNotIn("dense_score", context_text)
        self.assertNotIn("retrieval_score", context_text)

    def test_mem0_answer_retrieval_uses_dense_similarity_without_date_boost(self) -> None:
        question = normalize_benchmark(
            {
                "qas": [
                    {
                        "qa_id": "q1",
                        "scope": "single_admission",
                        "question_type": "medical_reasoning",
                        "question": "On 2020-02-01, what diagnosis was discussed?",
                        "answer": "cellulitis",
                        "evidence": {"admissions": ["1"], "turn_ids": [1]},
                    }
                ]
            }
        )[0]
        memory_store = Mem0MemoryStore(
            subject_id="11826927",
            summary="",
            retriever=fake_dense_retriever(),
            records=[
                {
                    "memory_id": "mem_dense",
                    "memory": "The diagnosis was discussed repeatedly as cellulitis without a visible date.",
                },
                {
                    "memory_id": "mem_date",
                    "memory": "On 2020-02-01, clinicians reviewed discharge paperwork.",
                },
            ],
        )

        selection = select_mem0_context_for_question(
            memory_store=memory_store,
            question=question,
            model_name="Qwen/Qwen3.5-4B",
            tokenizer_name=None,
            max_model_len=4096,
            max_output_tokens=128,
            safe_margin_tokens=0,
            token_estimate_safety_multiplier=1.0,
            retrieval_top_k=2,
            max_answer_memories=2,
        )

        selected_records = selection["context_record"]["selected_memory_records"]
        self.assertEqual(selected_records[0]["memory_id"], "mem_dense")
        self.assertIn("dense_score", selected_records[0])
        self.assertNotIn("date_boost", selected_records[0])

    def test_mem0_answer_retrieval_dense_similarity_ranks_range_question(self) -> None:
        question = normalize_benchmark(
            {
                "qas": [
                    {
                        "qa_id": "q1",
                        "scope": "single_admission",
                        "question_type": "medical_reasoning",
                        "question": "During the hospitalization from 2020-02-01 to 2020-02-03, what changed?",
                        "answer": "oxygen improved",
                        "evidence": {"admissions": ["1"], "turn_ids": [1]},
                    }
                ]
            }
        )[0]
        memory_store = Mem0MemoryStore(
            subject_id="11826927",
            summary="",
            retriever=fake_dense_retriever(),
            records=[
                {
                    "memory_id": "mem_inside",
                    "memory": "On 2020-02-02, oxygen improved after diuresis.",
                },
                {
                    "memory_id": "mem_outside",
                    "memory": "On 2020-03-10, oxygen improved after diuresis.",
                },
            ],
        )

        selection = select_mem0_context_for_question(
            memory_store=memory_store,
            question=question,
            model_name="Qwen/Qwen3.5-4B",
            tokenizer_name=None,
            max_model_len=4096,
            max_output_tokens=128,
            safe_margin_tokens=0,
            token_estimate_safety_multiplier=1.0,
            retrieval_top_k=2,
            max_answer_memories=2,
        )

        selected_records = selection["context_record"]["selected_memory_records"]
        self.assertEqual(selected_records[0]["memory_id"], "mem_inside")
        self.assertIn("dense_score", selected_records[0])

    def test_mem0_answer_retrieval_preserves_dense_order_without_question_dates(self) -> None:
        question = normalize_benchmark(
            {
                "qas": [
                    {
                        "qa_id": "q1",
                        "scope": "single_admission",
                        "question_type": "medical_reasoning",
                        "question": "What helped fever?",
                        "answer": "dialysis",
                        "evidence": {"admissions": ["1"], "turn_ids": [1]},
                    }
                ]
            }
        )[0]
        memory_store = Mem0MemoryStore(
            subject_id="11826927",
            summary="",
            retriever=fake_dense_retriever(),
            records=[
                {
                    "memory_id": "mem_best",
                    "memory": "On 2020-02-01, fever improved after dialysis.",
                },
                {
                    "memory_id": "mem_other",
                    "memory": "On 2020-01-01, discharge medications were reviewed.",
                },
            ],
        )

        selection = select_mem0_context_for_question(
            memory_store=memory_store,
            question=question,
            model_name="Qwen/Qwen3.5-4B",
            tokenizer_name=None,
            max_model_len=4096,
            max_output_tokens=128,
            safe_margin_tokens=0,
            token_estimate_safety_multiplier=1.0,
            retrieval_top_k=2,
            max_answer_memories=2,
        )

        selected_records = selection["context_record"]["selected_memory_records"]
        self.assertEqual(selected_records[0]["memory_id"], "mem_best")
        self.assertTrue(all("dense_score" in record for record in selected_records))

    def test_mem0_update_retrieval_uses_dense_similarity(self) -> None:
        memory_store = Mem0MemoryStore(
            subject_id="11826927",
            summary="",
            retriever=fake_dense_retriever(),
            records=[
                {
                    "memory_id": "mem_00001",
                    "memory": "On 2020-02-01, unrelated catheter detail.",
                },
                {
                    "memory_id": "mem_00002",
                    "memory": "Fever improved after dialysis without a visible date.",
                },
            ],
        )

        retrieved = retrieve_existing_for_candidates(
            memory_store,
            [{"candidate_id": "c001", "memory": "Fever improved after dialysis."}],
            similar_per_candidate=2,
            max_existing=2,
        )

        self.assertEqual(retrieved[0]["memory_id"], "mem_00002")
        self.assertNotIn("date_boost", retrieved[0])
        self.assertNotIn("sparse_score", retrieved[0])
        self.assertIn("dense_score", retrieved[0])

    def test_resolve_patient_targets_keeps_subject_id_path_order_for_multi_target_inputs(self) -> None:
        patient_a = self._write_patient_artifacts(11826927)
        patient_b = self._write_patient_artifacts(17207245)
        manifest_path = self.root / "patients.txt"
        manifest_path.write_text("11826927\n17207245\n", encoding="utf-8")

        manifest_targets = resolve_patient_targets(
            output_root=self.output_root,
            subject_id=None,
            subject_ids=None,
            patient_manifest=manifest_path,
            patient_dir=None,
        )
        self.assertEqual(
            [(subject_id, patient_root.resolve()) for subject_id, patient_root in manifest_targets],
            [(11826927, patient_a.resolve()), (17207245, patient_b.resolve())],
        )

        subject_id_targets = resolve_patient_targets(
            output_root=self.output_root,
            subject_id=None,
            subject_ids=[11826927, 17207245],
            patient_manifest=None,
            patient_dir=None,
        )
        self.assertEqual(
            [(subject_id, patient_root.resolve()) for subject_id, patient_root in subject_id_targets],
            [(11826927, patient_a.resolve()), (17207245, patient_b.resolve())],
        )

    def test_tracked_top10_manifest_matches_first_ten_shortlist_rows(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        manifest_path = repo_root / "health_benchmark" / "evaluation" / "cohorts" / "top10_patients.txt"
        csv_path = repo_root / "output" / "top_100_eligible_patients.csv"
        manifest_ids = [
            line.strip()
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        csv_ids = [
            line.split(",", 1)[0].strip()
            for line in csv_path.read_text(encoding="utf-8").splitlines()[1:11]
            if line.strip()
        ]

        self.assertEqual(manifest_ids, csv_ids)
        self.assertEqual(manifest_ids[0], "13813803")
        self.assertEqual(manifest_ids[-1], "15573773")

    def test_main_refresh_summary_tokens_updates_flat_hf_counts(self) -> None:
        patient_root = self._write_patient_artifacts()
        self._write_json(
            patient_root / "patient_summary.json",
            {
                "subject_id": "11826927",
                "eligible_admissions": 2,
                "processed_admissions": 2,
                "conversation_stats": {
                    "mean_turns": 0,
                    "total_turns": 0,
                    "mean_tokens": 0,
                    "total_tokens": 0,
                    "tokenizer": "old",
                },
            },
        )
        manifest_path = self.root / "patients.txt"
        manifest_path.write_text("11826927\n", encoding="utf-8")

        exit_code = main_cli.main(
            [
                "refresh-summary-tokens",
                "--output-root",
                str(self.output_root),
                "--patient-manifest",
                str(manifest_path),
                "--tokenizer-model",
                "Qwen/Qwen3.5-4B",
            ]
        )

        self.assertEqual(exit_code, 0)
        patient_summary = json.loads((patient_root / "patient_summary.json").read_text(encoding="utf-8"))
        stats = patient_summary["conversation_stats"]
        self.assertEqual(stats["total_turns"], 4)
        self.assertGreater(stats["total_tokens"], 0)
        self.assertEqual(stats["tokenizer"], "Qwen/Qwen3.5-4B")
        self.assertEqual(stats["token_count_format"], "flat_time_speaker_text")

    def test_scoring_uses_normalized_exact_adversarial_answer_and_keeps_comma_fallback(self) -> None:
        self.assertEqual(normalize_answer("The, Fever!"), "fever")
        metrics = score_answerable("vancomycin, cefepime", "cefepime, vancomycin")
        self.assertGreater(metrics["f1"], 0.9)
        self.assertTrue(metrics["used_comma_fallback"])
        self.assertEqual(score_adversarial(CANONICAL_ABSTENTION_ANSWER), 1.0)
        self.assertEqual(score_adversarial(f" {CANONICAL_ABSTENTION_ANSWER} "), 1.0)
        self.assertEqual(score_adversarial("The question is not answerable"), 1.0)
        self.assertEqual(score_adversarial("The question is not answerable."), 1.0)
        self.assertEqual(score_adversarial("THE   QUESTION IS NOT ANSWERABLE!!!"), 1.0)
        self.assertEqual(score_adversarial("This is not answerable from the provided conversation."), 0.0)
        self.assertEqual(score_adversarial("The conversation does not provide information about this."), 0.0)
        self.assertEqual(score_adversarial("Not mentioned"), 0.0)

    def test_run_answer_batches_recovers_missing_qa_ids_with_retry(self) -> None:
        patient_root = self._write_patient_artifacts()
        combined_payload = json.loads((patient_root / "combined_conversation.json").read_text(encoding="utf-8"))
        questions = normalize_benchmark(
            {
                "qas": [
                    {
                        "qa_id": "q1",
                        "scope": "single_admission",
                        "question_type": "medical_reasoning",
                        "question": "What helped first?",
                        "answer": "dialysis",
                        "evidence": {"admissions": ["1"], "turn_ids": [1]},
                    },
                    {
                        "qa_id": "q2",
                        "scope": "single_admission",
                        "question_type": "medical_reasoning",
                        "question": "What helped second?",
                        "answer": "antibiotics",
                        "evidence": {"admissions": ["1"], "turn_ids": [2]},
                    },
                ]
            }
        )
        settings = build_settings(
            self._config(),
            provider="vllm",
            base_url=None,
            judge_base_url=None,
            api_key_env=None,
            models=["Qwen/Qwen3.5-4B"],
            replace_existing=None,
        )
        batches = build_batches(
            questions,
            combined_payload=combined_payload,
            settings=settings,
            model_spec=settings.model_specs[0],
        )
        answer_client = FakeLLMClient(
            [
                {"parsed_output": {"answers": [{"qa_id": "q1", "prediction": "dialysis"}]}},
                {"parsed_output": {"answers": [{"qa_id": "q2", "prediction": "antibiotics"}]}},
            ]
        )
        predictions, raw_records, errors, failed_statuses = run_answer_batches(
            answer_client,
            batches,
            model_name="Qwen/Qwen3.5-4B",
            save_raw_response=True,
            max_output_tokens=settings.max_output_tokens,
        )
        self.assertEqual([prediction.qa_id for prediction in predictions], ["q1", "q2"])
        self.assertEqual(errors, [])
        self.assertEqual(failed_statuses, {})
        self.assertEqual(raw_records[0]["missing_qa_ids"], ["q2"])
        self.assertIn("retry", answer_client.calls[1]["system_message"].lower())

    def test_answer_batch_tolerant_parser_repairs_alias_keys_and_top_level_list(self) -> None:
        _, settings, batches = self._two_question_answer_batch()
        answer_client = RawContentLLMClient(
            [
                {
                    "content": json.dumps(
                        [
                            {"question_id": "q1", "answer": "dialysis"},
                            {"qa_id": "q2", "answer": "antibiotics"},
                        ]
                    )
                }
            ]
        )

        predictions, raw_records, errors, failed_statuses = run_answer_batches(
            answer_client,
            batches,
            model_name="microsoft/MediPhi-Instruct",
            save_raw_response=True,
            max_output_tokens=settings.max_output_tokens,
            retry_limit=settings.retry_limit,
        )

        self.assertEqual([(prediction.qa_id, prediction.prediction) for prediction in predictions], [("q1", "dialysis"), ("q2", "antibiotics")])
        self.assertEqual(errors, [])
        self.assertEqual(failed_statuses, {})
        self.assertTrue(raw_records[0]["schema_repair_applied"])
        self.assertIn("top_level_item_list", raw_records[0]["schema_repair_method"])
        self.assertEqual(raw_records[0]["returned_qa_ids"], ["q1", "q2"])

    def test_structured_content_candidates_extract_raw_control_character_json_blocks(self) -> None:
        embedded = (
            'The response is:\n{"memories": ['
            '{"candidate_id": "c001", "memory": "On 2020-01-01, fever\\nimproved."}'
            "]}\nDone."
        ).replace("\\n", "\n")
        candidates = structured_content_candidates(embedded)

        self.assertIn(
            '{"memories": [{"candidate_id": "c001", "memory": "On 2020-01-01, fever\nimproved."}]}',
            candidates,
        )

    def test_structured_content_candidates_extract_fenced_raw_control_character_json_blocks(self) -> None:
        fenced = (
            '```json\n{"memories": ['
            '{"candidate_id": "c001", "memory": "On 2020-01-01, fever\\nimproved."}'
            "]}\n```"
        ).replace("\\n", "\n")
        candidates = structured_content_candidates(fenced)

        self.assertIn(
            '{"memories": [{"candidate_id": "c001", "memory": "On 2020-01-01, fever\nimproved."}]}',
            candidates,
        )

    def test_structured_content_candidates_normalizes_smart_json_delimiter_quotes(self) -> None:
        embedded = (
            'The response is:\n{"memories": ['
            '{"candidate_id": "c001", "memory": "Family reported safety concerns.”\n    }'
            "]}\nDone."
        )
        candidates = structured_content_candidates(embedded)

        self.assertIn(
            '{"memories": [{"candidate_id": "c001", "memory": "Family reported safety concerns."\n    }]}',
            candidates,
        )

    def test_answer_batch_tolerant_parser_recovers_raw_control_characters(self) -> None:
        _, settings, batches = self._two_question_answer_batch()
        answer_client = RawContentLLMClient(
            [
                {
                    "content": (
                        '{"answers": ['
                        '{"qa_id": "q1", "prediction": "dialysis\\ncontinued"},'
                        '{"qa_id": "q2", "prediction": "antibiotics"}'
                        "]}"
                    ).replace("\\n", "\n")
                }
            ]
        )

        predictions, raw_records, errors, failed_statuses = run_answer_batches(
            answer_client,
            batches,
            model_name="google/gemma-3-4b-it",
            save_raw_response=True,
            max_output_tokens=settings.max_output_tokens,
            retry_limit=settings.retry_limit,
        )

        self.assertEqual([(prediction.qa_id, prediction.prediction) for prediction in predictions], [("q1", "dialysis\ncontinued"), ("q2", "antibiotics")])
        self.assertEqual(errors, [])
        self.assertEqual(failed_statuses, {})
        self.assertTrue(raw_records[0]["schema_repair_applied"])
        self.assertEqual(raw_records[0]["schema_repair_method"], "answers_alias_keys")

    def test_answer_batch_tolerant_parser_repairs_single_object_and_question_text(self) -> None:
        _, settings, batches = self._two_question_answer_batch()
        answer_client = RawContentLLMClient(
            [
                {"content": json.dumps({"question_id": "q1", "answer": "dialysis"})},
                {"content": json.dumps({"question": "What helped second?", "answer": "antibiotics"})},
            ]
        )

        predictions, raw_records, errors, failed_statuses = run_answer_batches(
            answer_client,
            batches,
            model_name="microsoft/MediPhi-Instruct",
            save_raw_response=True,
            max_output_tokens=settings.max_output_tokens,
            retry_limit=1,
        )

        self.assertEqual([(prediction.qa_id, prediction.prediction) for prediction in predictions], [("q1", "dialysis"), ("q2", "antibiotics")])
        self.assertEqual(errors, [])
        self.assertEqual(failed_statuses, {})
        self.assertEqual(raw_records[0]["missing_qa_ids"], ["q2"])
        self.assertIn("single_item_object", raw_records[0]["schema_repair_method"])
        self.assertIn("question_text", raw_records[1]["schema_repair_method"])

    def test_answer_batch_tolerant_parser_repairs_exact_length_string_list_by_order(self) -> None:
        _, settings, batches = self._two_question_answer_batch()
        answer_client = RawContentLLMClient(
            [
                {"content": json.dumps(["dialysis", "antibiotics"])},
            ]
        )

        predictions, raw_records, errors, failed_statuses = run_answer_batches(
            answer_client,
            batches,
            model_name="lingshu-medical-mllm/Lingshu-32B",
            save_raw_response=True,
            max_output_tokens=settings.max_output_tokens,
            retry_limit=settings.retry_limit,
        )

        self.assertEqual([(prediction.qa_id, prediction.prediction) for prediction in predictions], [("q1", "dialysis"), ("q2", "antibiotics")])
        self.assertEqual(errors, [])
        self.assertEqual(failed_statuses, {})
        self.assertEqual(raw_records[0]["schema_repair_method"], "ordered_string_list")

    def test_answer_batch_tolerant_parser_rejects_unsafe_repairs(self) -> None:
        _, settings, batches = self._two_question_answer_batch()
        unsafe_payloads = [
            ["dialysis"],
            [{"qa_id": "bad", "answer": "dialysis"}],
            [{"qa_id": "q1", "answer": "dialysis"}, {"qa_id": "q1", "answer": "duplicate"}],
            [{"answer": "dialysis"}],
        ]
        for payload in unsafe_payloads:
            with self.subTest(payload=payload):
                answer_client = RawContentLLMClient(
                    [
                        {"content": json.dumps(payload)},
                    ]
                )

                predictions, raw_records, errors, failed_statuses = run_answer_batches(
                    answer_client,
                    batches,
                    model_name="lingshu-medical-mllm/Lingshu-32B",
                    save_raw_response=True,
                    max_output_tokens=settings.max_output_tokens,
                    retry_limit=settings.retry_limit,
                )

                self.assertEqual(predictions, [])
                self.assertEqual(raw_records, [])
                self.assertEqual(len(errors), 1)
                self.assertEqual(errors[0]["error_kind"], "format_error")
                self.assertEqual(failed_statuses, {"q1": "answer_failed", "q2": "answer_failed"})

    def test_answer_batch_tolerant_parser_rejects_ambiguous_question_text_mapping(self) -> None:
        patient_root = self._write_patient_artifacts()
        combined_payload = json.loads((patient_root / "combined_conversation.json").read_text(encoding="utf-8"))
        questions = normalize_benchmark(
            {
                "qas": [
                    {
                        "qa_id": "q1",
                        "scope": "single_admission",
                        "question_type": "medical_reasoning",
                        "question": "Same question?",
                        "answer": "dialysis",
                        "evidence": {"admissions": ["1"], "turn_ids": [1]},
                    },
                    {
                        "qa_id": "q2",
                        "scope": "single_admission",
                        "question_type": "medical_reasoning",
                        "question": "Same question?",
                        "answer": "antibiotics",
                        "evidence": {"admissions": ["1"], "turn_ids": [2]},
                    },
                ]
            }
        )
        settings = build_settings(
            self._config(),
            provider="vllm",
            base_url=None,
            judge_base_url=None,
            api_key_env=None,
            models=["Qwen/Qwen3.5-4B"],
            replace_existing=None,
            retry_limit=0,
        )
        batches = build_batches(
            questions,
            combined_payload=combined_payload,
            settings=settings,
            model_spec=settings.model_specs[0],
        )
        answer_client = RawContentLLMClient(
            [
                {"content": json.dumps({"question": "Same question?", "answer": "dialysis"})},
            ]
        )

        predictions, raw_records, errors, failed_statuses = run_answer_batches(
            answer_client,
            batches[:1],
            model_name="microsoft/MediPhi-Instruct",
            save_raw_response=True,
            max_output_tokens=settings.max_output_tokens,
            retry_limit=settings.retry_limit,
        )

        self.assertEqual(predictions, [])
        self.assertEqual(raw_records, [])
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["error_kind"], "format_error")
        self.assertEqual(failed_statuses, {"q1": "answer_failed", "q2": "answer_failed"})

    def test_repaired_malformed_answer_batch_scores_instead_of_all_answer_failed(self) -> None:
        questions, settings, batches = self._two_question_answer_batch()
        answer_client = RawContentLLMClient(
            [
                {
                    "content": json.dumps(
                        {
                            "answers": [
                                {"question_id": "q1", "answer": "dialysis"},
                                {"question_id": "q2", "answer": "antibiotics"},
                            ]
                        }
                    )
                }
            ]
        )

        predictions, raw_records, errors, failed_statuses = run_answer_batches(
            answer_client,
            batches,
            model_name="microsoft/MediPhi-Instruct",
            save_raw_response=True,
            max_output_tokens=settings.max_output_tokens,
            retry_limit=settings.retry_limit,
        )
        scored_rows = score_predictions(questions, predictions, failed_statuses)

        self.assertEqual(errors, [])
        self.assertEqual([row["status"] for row in scored_rows], ["scored", "scored"])
        self.assertTrue(all(float(row["f1"]) == 1.0 for row in scored_rows))
        self.assertTrue(raw_records[0]["schema_repair_applied"])
        self.assertEqual(
            summarize_schema_repair_metrics(raw_records),
            {
                "schema_repaired_batch_count": 1,
                "schema_repaired_prediction_count": 2,
                "schema_order_repaired_batch_count": 0,
            },
        )

    def test_memory_response_tolerant_parser_wraps_extraction_shapes(self) -> None:
        payloads = [
            (
                '{"candidate_id": "c001", "memory": "On 2020-01-01, fever improved."}',
                ["c001"],
            ),
            (
                '[{"candidate_id": "c002", "memory": "On 2020-01-02, dialysis continued."}]',
                ["c002"],
            ),
        ]
        for content, expected_ids in payloads:
            with self.subTest(content=content):
                result = _generate_memory_response(
                    RawContentLLMClient([{"content": content}]),
                    system_message="extract",
                    user_message="{}",
                    response_schema=Mem0ExtractionResponse,
                    max_output_tokens=128,
                    retry_limit=0,
                    model_name="google/gemma-3-4b-it",
                )

                self.assertEqual([item.candidate_id for item in result.parsed.memories], expected_ids)

    def test_memory_response_tolerant_parser_wraps_memalpha_shapes(self) -> None:
        payloads = [
            (
                '{"candidate_id": "c001", "memory_type": "core", "memory": "Chronic dialysis."}',
                ("core", ["c001"]),
            ),
            (
                '[{"candidate_id": "s001", "memory_type": "semantic", "memory": "Dialysis improves volume status."}]',
                ("semantic", ["s001"]),
            ),
            (
                '{"memories": [{"candidate_id": "e001", "memory_type": "episodic", "memory": "On 2020-01-01, dialysis continued."}]}',
                ("episodic", ["e001"]),
            ),
        ]
        for content, (memory_type, expected_ids) in payloads:
            with self.subTest(content=content):
                result = _generate_memory_response(
                    RawContentLLMClient([{"content": content}]),
                    system_message="extract structured memory",
                    user_message="{}",
                    response_schema=MemAlphaExtractionResponse,
                    max_output_tokens=128,
                    retry_limit=0,
                    model_name="google/gemma-3-4b-it",
                )

                self.assertEqual(
                    [item.candidate_id for item in getattr(result.parsed, memory_type)],
                    expected_ids,
                )

    def test_memory_response_tolerant_parser_wraps_update_shapes(self) -> None:
        payloads = [
            (
                '{"candidate_id": "c001", "operation": "ADD", "target_memory_id": null, "memory": "On 2020-01-01, fever improved."}',
                ["c001"],
            ),
            (
                '[{"candidate_id": "c002", "operation": "NOOP", "target_memory_id": null, "memory": null}]',
                ["c002"],
            ),
        ]
        for content, expected_ids in payloads:
            with self.subTest(content=content):
                result = _generate_memory_response(
                    RawContentLLMClient([{"content": content}]),
                    system_message="update",
                    user_message="{}",
                    response_schema=Mem0UpdateResponse,
                    max_output_tokens=128,
                    retry_limit=0,
                    model_name="google/gemma-3-4b-it",
                )

                self.assertEqual([item.candidate_id for item in result.parsed.actions], expected_ids)

    def test_memory_response_tolerant_parser_recovers_control_characters(self) -> None:
        content = (
            '{"memories": ['
            '{"candidate_id": "c001", "memory": "On 2020-01-01, fever\\nimproved."}'
            "]}"
        ).replace("\\n", "\n")

        result = _generate_memory_response(
            RawContentLLMClient([{"content": content}]),
            system_message="extract",
            user_message="{}",
            response_schema=Mem0ExtractionResponse,
            max_output_tokens=128,
            retry_limit=0,
            model_name="google/gemma-3-4b-it",
        )

        self.assertEqual(result.parsed.memories[0].memory, "On 2020-01-01, fever\nimproved.")

    def test_memory_response_tolerant_parser_recovers_smart_closing_quote(self) -> None:
        content = (
            'Here is the JSON:\n{"memories": ['
            '{"candidate_id": "c001", "memory": "Family reported safety concerns.”\n    }'
            "]}\n"
        )

        result = _generate_memory_response(
            RawContentLLMClient([{"content": content}]),
            system_message="extract",
            user_message="{}",
            response_schema=Mem0ExtractionResponse,
            max_output_tokens=128,
            retry_limit=0,
            model_name="google/gemma-3-4b-it",
        )

        self.assertEqual(result.parsed.memories[0].memory, "Family reported safety concerns.")

    def test_memory_response_tolerant_parser_rejects_unsupported_shapes(self) -> None:
        with self.assertRaises(RuntimeError):
            _generate_memory_response(
                RawContentLLMClient([{"content": '{"candidate_id": "c001", "text": "missing memory"}'}]),
                system_message="extract",
                user_message="{}",
                response_schema=Mem0ExtractionResponse,
                max_output_tokens=128,
                retry_limit=0,
                model_name="google/gemma-3-4b-it",
            )

    def test_answer_batch_retry_limit_is_per_batch_and_marks_exhausted_failures(self) -> None:
        patient_root = self._write_patient_artifacts()
        combined_payload = json.loads((patient_root / "combined_conversation.json").read_text(encoding="utf-8"))
        questions = normalize_benchmark(self._benchmark_payload())
        settings = build_settings(
            self._config(),
            provider="vllm",
            base_url=None,
            judge_base_url=None,
            api_key_env=None,
            models=["Qwen/Qwen3.5-4B"],
            replace_existing=None,
            retry_limit=1,
        )
        batches = build_batches(
            questions,
            combined_payload=combined_payload,
            settings=settings,
            model_spec=settings.model_specs[0],
        )
        answer_client = FakeLLMClient(
            [
                {"parsed_output": {"answers": [{"qa_id": "bad", "prediction": "bad"}]}},
                {"parsed_output": self._predicted_answers_payload(1, 10)},
                {"parsed_output": {"answers": [{"qa_id": "also_bad", "prediction": "bad"}]}},
                {"parsed_output": {"answers": [{"qa_id": "still_bad", "prediction": "bad"}]}},
            ]
        )

        predictions, raw_records, errors, failed_statuses = run_answer_batches(
            answer_client,
            batches,
            model_name="Qwen/Qwen3.5-4B",
            save_raw_response=True,
            max_output_tokens=settings.max_output_tokens,
            retry_limit=settings.retry_limit,
        )

        self.assertEqual(len(answer_client.calls), 4)
        self.assertEqual([prediction.qa_id for prediction in predictions], [f"q{index:02d}" for index in range(1, 11)])
        self.assertEqual(raw_records[0]["attempt_index"], 2)
        self.assertTrue(errors[0]["will_retry"])
        self.assertFalse(errors[-1]["will_retry"])
        self.assertEqual(errors[-1]["max_attempts"], 2)
        self.assertEqual(failed_statuses, {"q11": "answer_failed", "q12": "answer_failed"})

    def test_answer_batch_context_length_errors_are_not_retried(self) -> None:
        patient_root = self._write_patient_artifacts()
        combined_payload = json.loads((patient_root / "combined_conversation.json").read_text(encoding="utf-8"))
        questions = normalize_benchmark(self._benchmark_payload())[:10]
        settings = build_settings(
            self._config(),
            provider="vllm",
            base_url=None,
            judge_base_url=None,
            api_key_env=None,
            models=["Qwen/Qwen3.5-4B"],
            replace_existing=None,
            retry_limit=3,
        )
        batches = build_batches(
            questions,
            combined_payload=combined_payload,
            settings=settings,
            model_spec=settings.model_specs[0],
        )
        context_error = RuntimeError(
            "This model's maximum context length is 131072 tokens. "
            "However, you requested 4096 output tokens and your prompt contains "
            "at least 126977 input tokens. Please reduce the length of the input prompt."
        )
        answer_client = RaisingLLMClient(context_error)

        predictions, raw_records, errors, failed_statuses = run_answer_batches(
            answer_client,
            batches[:1],
            model_name="Qwen/Qwen3.5-4B",
            save_raw_response=True,
            max_output_tokens=settings.max_output_tokens,
            retry_limit=settings.retry_limit,
        )

        self.assertEqual(len(answer_client.calls), 1)
        self.assertEqual(predictions, [])
        self.assertEqual(raw_records, [])
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["error_kind"], "context_length_error")
        self.assertFalse(errors[0]["will_retry"])
        self.assertEqual(set(failed_statuses), set(batches[0].qa_ids()))

    def test_failed_answers_score_as_zero_without_llm_judge(self) -> None:
        questions = normalize_benchmark(
            {
                "qas": [
                    {
                        "qa_id": "q1",
                        "scope": "single_admission",
                        "question_type": "medical_reasoning",
                        "question": "What helped?",
                        "answer": "dialysis",
                        "evidence": {"admissions": ["1"], "turn_ids": [1]},
                    },
                    {
                        "qa_id": "q2",
                        "scope": "cross_admission",
                        "question_type": "adversarial",
                        "question": "Unsupported?",
                        "answer": CANONICAL_ABSTENTION_ANSWER,
                        "evidence": {"admissions": ["1", "2"]},
                    },
                ]
            }
        )

        rows = score_predictions(questions, [], {"q1": "answer_failed", "q2": "answer_failed"})

        self.assertEqual(rows[0]["status"], "answer_failed")
        self.assertEqual(rows[0]["precision"], 0.0)
        self.assertEqual(rows[0]["recall"], 0.0)
        self.assertEqual(rows[0]["f1"], 0.0)
        self.assertEqual(rows[0]["llm_judge_score"], 0.0)
        self.assertEqual(rows[0]["per_question_score"], 0.0)
        self.assertEqual(rows[1]["status"], "answer_failed")
        self.assertEqual(rows[1]["abstention_accuracy"], 0.0)
        self.assertIsNone(rows[1]["llm_judge_score"])
        self.assertEqual(rows[1]["per_question_score"], 0.0)

    def test_llm_judge_batches_retry_before_failing(self) -> None:
        scored_rows = [
            {
                "qa_id": "q1",
                "question": "What helped?",
                "gold_answer": "dialysis",
                "prediction": "dialysis",
                "status": "scored",
                "is_adversarial": False,
                "llm_judge_score": None,
            }
        ]
        judge_client = FakeLLMClient(
            [
                {"parsed_output": {"judgments": [{"qa_id": "wrong", "score": 1}]}},
                {"parsed_output": {"judgments": [{"qa_id": "q1", "score": 1}]}},
            ]
        )

        raw_records = run_llm_judge_batches(
            judge_client,
            scored_rows,
            judge_model_name="Qwen/Qwen3.5-27B",
            save_raw_response=True,
            max_output_tokens=1024,
            batch_size=10,
            retry_limit=1,
        )

        self.assertEqual(len(judge_client.calls), 2)
        self.assertEqual(scored_rows[0]["llm_judge_score"], 1.0)
        self.assertEqual(raw_records[0]["attempt_index"], 2)
        self.assertEqual(raw_records[0]["max_attempts"], 2)

    def test_pipeline_runs_end_to_end_and_writes_comparison_outputs(self) -> None:
        patient_root = self._write_patient_artifacts()
        base_config = self._config()
        settings = build_settings(
            base_config,
            provider="vllm",
            base_url="http://127.0.0.1:8000/v1",
            judge_base_url="http://127.0.0.1:8001/v1",
            api_key_env=None,
            models=["Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B"],
            replace_existing=True,
        )
        answerable_ids = [f"q{index:02d}" for index in range(1, 13) if index not in {4, 11}]
        client_overrides = {
            "Qwen/Qwen3.5-4B": FakeLLMClient(
                [
                    {"parsed_output": self._predicted_answers_payload(1, 10)},
                    {"parsed_output": self._predicted_answers_payload(11, 12)},
                ]
            ),
            "Qwen/Qwen3.5-9B": FakeLLMClient(
                [
                    {"parsed_output": self._predicted_answers_payload(1, 10)},
                    {"parsed_output": self._predicted_answers_payload(11, 12)},
                ]
            ),
            "__judge__": FakeLLMClient(
                [
                    {"parsed_output": self._judge_answerable_payload(answerable_ids)},
                    {"parsed_output": self._judge_answerable_payload(answerable_ids)},
                ]
            ),
        }
        pipeline = EvaluationPipeline(
            base_config,
            settings,
            client_overrides=client_overrides,
        )

        summary = pipeline.run([(11826927, patient_root)])

        self.assertEqual(summary["final_status"], "completed")
        result = summary["results"][0]
        self.assertEqual(result["status"], "completed")
        evaluation_root = self._evaluation_root()
        self.assertTrue((evaluation_root / "config.json").exists())
        self.assertTrue((evaluation_root / "context_stats.json").exists())
        self.assertTrue((evaluation_root / "benchmark_snapshot.json").exists())
        self.assertTrue((evaluation_root / "qwen3.5-4b" / "summary.json").exists())
        self.assertTrue((evaluation_root / "qwen3.5-9b" / "summary.json").exists())
        self.assertTrue((evaluation_root / "qwen3.5-4b" / "llm_judgments.jsonl").exists())
        self.assertTrue((evaluation_root / "comparison" / "leaderboard.json").exists())
        leaderboard = json.loads((evaluation_root / "comparison" / "leaderboard.json").read_text(encoding="utf-8"))
        self.assertEqual([row["model_slug"] for row in leaderboard["models"]], ["qwen3.5-4b", "qwen3.5-9b"])
        self.assertEqual(leaderboard["models"][0]["llm_score"], 1.0)
        model_summary = json.loads((evaluation_root / "qwen3.5-4b" / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(model_summary["run_status"], "completed")
        self.assertEqual(model_summary["num_questions_total"], 12)
        self.assertEqual(model_summary["macro_f1_answerable"], 1.0)
        self.assertEqual(model_summary["adversarial_accuracy"], 1.0)
        self.assertEqual(model_summary["llm_score"], 1.0)
        self.assertEqual(model_summary["answer_failed_percent"], 0.0)
        self.assertEqual(model_summary["breakdowns"]["by_scope"]["cross_admission"]["answerable_count"], 5)
        self.assertIn("by_adversarial_scope", model_summary["breakdowns"])
        self.assertTrue((evaluation_root / "comparison" / "breakdowns.json").exists())
        self.assertNotIn("answer_failed_percent", leaderboard["models"][0])
        breakdowns = json.loads((evaluation_root / "comparison" / "breakdowns.json").read_text(encoding="utf-8"))
        self.assertTrue(all("answer_failed_percent" not in row for row in breakdowns["rows"]))
        summary_markdown = (evaluation_root / "comparison" / "summary.md").read_text(encoding="utf-8")
        self.assertNotIn("answer_failed", summary_markdown)

    def test_memory_pipeline_answers_stage_writes_sibling_mem0_artifacts(self) -> None:
        patient_root = self._write_patient_artifacts()
        base_config = self._config()
        settings = build_settings(
            base_config,
            provider="vllm",
            base_url="http://127.0.0.1:8000/v1",
            judge_base_url=None,
            api_key_env=None,
            models=["Qwen/Qwen3.5-4B"],
            stage="answers",
            evaluation_variant="mem0",
            replace_existing=True,
        )
        memory_payloads = [
            {
                "parsed_output": {
                    "memories": [
                        {
                            "candidate_id": "c001",
                            "memory": "On 2020-01-01 08:05:00, fever improved with dialysis.",
                        },
                        {
                            "candidate_id": "c002",
                            "memory": "On 2020-01-01 08:10:00, the cough was better.",
                        },
                    ]
                }
            },
            {
                "parsed_output": {
                    "actions": [
                        {
                            "candidate_id": "c001",
                            "operation": "ADD",
                            "target_memory_id": None,
                            "memory": "On 2020-01-01 08:05:00, fever improved with dialysis.",
                        },
                        {
                            "candidate_id": "c002",
                            "operation": "ADD",
                            "target_memory_id": None,
                            "memory": "On 2020-01-01 08:10:00, the cough was better.",
                        },
                    ]
                }
            },
            {"parsed_output": {"summary": "First admission: fever improved with dialysis and cough improved."}},
            {
                "parsed_output": {
                    "memories": [
                        {
                            "candidate_id": "c001",
                            "memory": "On 2020-02-01 10:05:00, breathing improved after dialysis.",
                        }
                    ]
                }
            },
            {
                "parsed_output": {
                    "actions": [
                        {
                            "candidate_id": "c001",
                            "operation": "ADD",
                            "target_memory_id": None,
                            "memory": "On 2020-02-01 10:05:00, breathing improved after dialysis.",
                        }
                    ]
                }
            },
            {"parsed_output": {"summary": "Second admission: breathing improved after dialysis."}},
        ]
        answer_payloads = [
            {"parsed_output": {"answers": [{"qa_id": f"q{index:02d}", "prediction": CANONICAL_ABSTENTION_ANSWER if index in {4, 11} else f"answer {index}"}]}}
            for index in range(1, 13)
        ]
        answer_client = FakeLLMClient(
            memory_payloads
            + answer_payloads
        )
        pipeline = MemoryEvaluationPipeline(
            base_config,
            settings,
            client_overrides={"Qwen/Qwen3.5-4B": answer_client},
            memory_retriever_factory=lambda _settings: fake_dense_retriever(),
        )

        summary = pipeline.run([(11826927, patient_root)])

        self.assertEqual(summary["final_status"], "completed")
        evaluation_root = self._evaluation_root()
        memory_dir = evaluation_root / "qwen3.5-4b-mem0"
        self.assertFalse((evaluation_root / "qwen3.5-4b").exists())
        memory_store = json.loads((memory_dir / "memory_store.json").read_text(encoding="utf-8"))
        question_batches = json.loads((memory_dir / "question_batches.json").read_text(encoding="utf-8"))
        model_summary = json.loads((memory_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(memory_store["mode"], "mem0")
        self.assertEqual(memory_store["settings"]["chunk_token_cap"], 12000)
        self.assertEqual(memory_store["settings"]["previous_chunk_summaries"], 1)
        self.assertEqual(memory_store["settings"]["max_candidate_memories"], 32)
        self.assertEqual(memory_store["settings"]["similar_memories_per_candidate"], 10)
        self.assertEqual(memory_store["settings"]["max_update_memories"], 40)
        self.assertEqual(memory_store["settings"]["answer_retrieval_top_k"], 64)
        self.assertEqual(memory_store["settings"]["max_answer_memories"], 32)
        self.assertEqual(memory_store["settings"]["max_output_tokens"], 4096)
        self.assertEqual(memory_store["settings"]["retrieval_backend"], "local_dense_hf")
        self.assertEqual(memory_store["settings"]["embedding_model"], "Qwen/Qwen3-Embedding-8B")
        self.assertEqual(memory_store["settings"]["embedding_batch_size"], 8)
        self.assertEqual(memory_store["settings"]["embedding_max_length"], 1024)
        self.assertEqual(memory_store["embedding_backend"], "local_dense_hf")
        self.assertEqual(memory_store["embedding_model"], "fake-dense-embedder")
        self.assertEqual(memory_store["embedding_dimension"], 4)
        self.assertEqual(memory_store["metrics"]["active_memories"], 3)
        self.assertEqual(memory_store["metrics"]["chunk_count"], 2)
        self.assertEqual(memory_store["metrics"]["extraction_call_count"], 2)
        self.assertEqual(memory_store["metrics"]["update_call_count"], 2)
        self.assertEqual(memory_store["metrics"]["summary_call_count"], 2)
        self.assertEqual(memory_store["metrics"]["summary_error_count"], 0)
        self.assertTrue(
            all(set(record) == {"memory_id", "memory"} for record in memory_store["memories"])
        )
        self.assertEqual(len(question_batches["batches"]), 12)
        self.assertTrue(all(len(batch["questions"]) == 1 for batch in question_batches["batches"]))
        self.assertEqual(question_batches["batches"][0]["context"]["strategy"], "mem0")
        selected_memory_records = next(
            batch["context"]["selected_memory_records"]
            for batch in question_batches["batches"]
            if batch["context"]["selected_memory_records"]
        )
        self.assertTrue(selected_memory_records)
        self.assertIn("memory_id", selected_memory_records[0])
        self.assertIn("memory", selected_memory_records[0])
        self.assertIn("dense_score", selected_memory_records[0])
        self.assertIn("retrieval_score", selected_memory_records[0])
        self.assertIn("selected_rank", selected_memory_records[0])
        rendered_memory_records = next(
            batch["context"]["rendered_memory_records"]
            for batch in question_batches["batches"]
            if batch["context"]["rendered_memory_records"]
        )
        self.assertTrue(rendered_memory_records)
        self.assertIn("rendered_order", rendered_memory_records[0])
        self.assertEqual(model_summary["evaluation_variant"], "mem0")
        self.assertEqual(model_summary["base_model_slug"], "qwen3.5-4b")
        self.assertEqual(model_summary["operational_metrics"]["answer_context_strategy"], "memory")
        self.assertEqual(model_summary["operational_metrics"]["memory"]["active_memories"], 3)
        self.assertEqual(answer_client.calls[-1]["schema"], "AnswerBatchResponse")
        self.assertIn("partial patient facts", answer_client.calls[-1]["system_message"])
        self.assertIn("directly support the requested fact, relationship, comparison, or temporal change", answer_client.calls[-1]["system_message"])
        self.assertIn("specific evidence for the compared items or timepoints", answer_client.calls[-1]["system_message"])
        self.assertIn("Related clinical facts are not enough by themselves", answer_client.calls[-1]["system_message"])
        self.assertNotIn("Do not abstain just because no single memory uses the exact wording", answer_client.calls[-1]["system_message"])
        self.assertNotIn("compare or synthesize across relevant memories", answer_client.calls[-1]["system_message"])
        self.assertIn("Prefer exact wording from the retrieved memories", answer_client.calls[-1]["system_message"])
        self.assertIn("patient_memory_context", answer_client.calls[-1]["user_message"])
        self.assertIn("chronological order when timestamps are available", answer_client.calls[-1]["user_message"])
        self.assertIn("Use only memories relevant to the question", answer_client.calls[-1]["user_message"])
        self.assertIn("some answers may require more than one memory", answer_client.calls[-1]["user_message"])
        self.assertNotIn("Multiple memories may jointly support the answer", answer_client.calls[-1]["user_message"])
        self.assertNotIn("Conversation summary:", answer_client.calls[-1]["user_message"])
        self.assertNotIn("mem_000", answer_client.calls[-1]["user_message"])
        self.assertNotIn("dense_score", answer_client.calls[-1]["user_message"])
        self.assertNotIn("retrieval_score", answer_client.calls[-1]["user_message"])
        self.assertNotIn("selected_rank", answer_client.calls[-1]["user_message"])
        self.assertNotIn("rendered_order", answer_client.calls[-1]["user_message"])
        self.assertTrue(all("hadm_id" not in call["user_message"] for call in answer_client.calls))
        self.assertTrue(all("turn=" not in call["user_message"] for call in answer_client.calls))
        memory_prompt_calls = [
            call
            for call in answer_client.calls
            if call["schema"] in {"Mem0ExtractionResponse", "Mem0UpdateResponse", "Mem0SummaryResponse"}
        ]
        self.assertEqual(len(memory_prompt_calls), 6)
        for call in memory_prompt_calls:
            visible_prompt = f"{call['system_message']}\n{call['user_message']}"
            self.assertNotIn("Mem0", visible_prompt)
            self.assertNotIn("mem0", visible_prompt)
            self.assertNotIn("Admission 1", visible_prompt)
            self.assertNotIn("admission_start", visible_prompt)
            self.assertNotIn("admission_end", visible_prompt)
            self.assertNotIn("start=", visible_prompt)
            self.assertNotIn("end=", visible_prompt)
            self.assertNotIn("hadm_id", visible_prompt)
            self.assertNotIn("turn=", visible_prompt)
            self.assertNotIn("global=", visible_prompt)
        extraction_system_messages = [
            str(call["system_message"])
            for call in memory_prompt_calls
            if call["schema"] == "Mem0ExtractionResponse"
        ]
        self.assertEqual(len(extraction_system_messages), 2)
        for system_message in extraction_system_messages:
            self.assertIn("conversation-line timestamps", system_message)
            self.assertIn("clinically relevant patient facts, events, decisions, and temporal changes.", system_message)
            self.assertIn("Preserve exact clinically meaningful wording", system_message)
            self.assertIn("V/Q scan", system_message)
            self.assertNotIn("that may help answer future questions", system_message)
            self.assertNotIn("Extract only high-value", system_message)
            self.assertNotIn("Do not use admission start/end metadata", system_message)
            self.assertNotIn("Preserve clinically useful facts", system_message)
            self.assertNotIn("symptoms, diagnoses, treatments", system_message)
            self.assertNotIn("Return at most", system_message)
            self.assertNotIn(str(settings.mem0_max_candidate_memories), system_message)
            self.assertNotIn("importance", system_message)
            self.assertNotIn("category", system_message)
            self.assertNotIn("time_range", system_message)
        summary_system_messages = [
            str(call["system_message"])
            for call in memory_prompt_calls
            if call["schema"] == "Mem0SummaryResponse"
        ]
        self.assertEqual(len(summary_system_messages), 2)
        for system_message in summary_system_messages:
            self.assertIn("conversation-line timestamps", system_message)
            self.assertIn("under 1200 characters", system_message)
            self.assertNotIn("admission start/end times", system_message)
        update_calls = [call for call in memory_prompt_calls if call["schema"] == "Mem0UpdateResponse"]
        self.assertTrue(any('"score"' in str(call["user_message"]) for call in update_calls))
        self.assertTrue(all("Preserve exact candidate wording" in str(call["system_message"]) for call in update_calls))
        self.assertTrue(all('"importance"' not in str(call["user_message"]) for call in update_calls))
        self.assertTrue(all('"category"' not in str(call["user_message"]) for call in update_calls))
        self.assertTrue(all('"time_range"' not in str(call["user_message"]) for call in update_calls))
        summary_calls = [call for call in memory_prompt_calls if call["schema"] == "Mem0SummaryResponse"]
        self.assertTrue(all(call["max_output_tokens"] == 1024 for call in summary_calls))

    def test_rag_selection_scores_all_admissions_and_renders_chronologically(self) -> None:
        patient_root = self._write_patient_artifacts()
        combined_payload = json.loads((patient_root / "combined_conversation.json").read_text(encoding="utf-8"))
        documents = build_admission_documents(combined_payload)
        questions = normalize_benchmark(self._benchmark_payload())
        question = next(item for item in questions if item.qa_id == "q07")
        retriever = DenseAdmissionRetriever(FakeDenseEmbedder())

        selection = select_rag_context_for_question(
            documents=documents,
            retriever=retriever,
            question=question,
            evaluation_variant="embedding_rag",
            rag_method="embedding-rag",
            model_name="Qwen/Qwen3.5-4B",
            tokenizer_name=None,
            max_model_len=4096,
            max_output_tokens=256,
            safe_margin_tokens=128,
            token_estimate_safety_multiplier=1.0,
        )

        context = selection["context_record"]
        self.assertTrue(context["all_admissions_scored"])
        self.assertEqual(context["scored_admissions"], 2)
        self.assertEqual(context["selected_admissions"], 2)
        self.assertEqual(context["selected_admission_records"][0]["doc_id"], "admission_002")
        self.assertEqual(context["rendered_admission_records"][0]["doc_id"], "admission_001")
        self.assertNotIn("hadm_id", selection["context_text"])
        self.assertNotIn("2001", selection["context_text"])
        self.assertNotIn("retrieval_score", selection["context_text"])

    def test_bm25_rag_scores_every_admission(self) -> None:
        patient_root = self._write_patient_artifacts()
        combined_payload = json.loads((patient_root / "combined_conversation.json").read_text(encoding="utf-8"))
        documents = build_admission_documents(combined_payload)
        scored = BM25AdmissionRetriever().score_all(documents, query="breathing dialysis")

        self.assertEqual(len(scored), 2)
        self.assertEqual({record["doc_id"] for record in scored}, {"admission_001", "admission_002"})
        self.assertIn("bm25_score", scored[0])

    def test_rag_answer_prompt_uses_retrieved_excerpt_policy(self) -> None:
        rendered = render_answer_prompt(
            context_text=render_rag_context(
                [
                    {
                        "text": "2020-01-01 08:00:00 | Doctor | Fever improved.",
                    }
                ]
            ),
            questions=[{"qa_id": "q1", "question": "What improved?"}],
            context_description="retrieved patient conversation excerpts",
            context_payload_key="retrieved_patient_context",
        )

        self.assertIn("Retrieved excerpts are selected from patient admissions", rendered.system_message)
        self.assertIn("directly support the requested fact, relationship, comparison, or temporal change", rendered.system_message)
        self.assertIn("retrieved_patient_context", rendered.user_message)
        self.assertNotIn("patient_memory_context", rendered.user_message)

    def test_rag_pipeline_passthrough_copies_normal_outputs(self) -> None:
        patient_root = self._write_patient_artifacts()
        base_config = self._config()
        normal_settings = build_settings(
            base_config,
            provider="vllm",
            base_url="http://127.0.0.1:8000/v1",
            judge_base_url=None,
            api_key_env=None,
            models=["Qwen/Qwen3.5-4B"],
            stage="answers",
            replace_existing=True,
        )
        normal_client = FakeLLMClient(
            [
                {"parsed_output": self._predicted_answers_payload(1, 10)},
                {"parsed_output": self._predicted_answers_payload(11, 12)},
            ]
        )
        EvaluationPipeline(
            base_config,
            normal_settings,
            client_overrides={"Qwen/Qwen3.5-4B": normal_client},
        ).run([(11826927, patient_root)])

        rag_settings = build_rag_settings(
            base_config,
            provider="vllm",
            base_url="http://127.0.0.1:8000/v1",
            judge_base_url=None,
            api_key_env=None,
            models=["Qwen/Qwen3.5-4B"],
            stage="answers",
            replace_existing=True,
            rag_method="embedding-rag",
            rag_model_max_len=32768,
            rag_model_tensor_parallel_size=1,
        )
        rag_client = FakeLLMClient([])
        summary = RagEvaluationPipeline(
            base_config,
            rag_settings,
            client_overrides={"Qwen/Qwen3.5-4B": rag_client},
        ).run([(11826927, patient_root)])

        self.assertEqual(summary["final_status"], "completed")
        self.assertEqual(rag_client.calls, [])
        evaluation_root = self._evaluation_root()
        rag_dir = evaluation_root / "qwen3.5-4b-embedding-rag"
        self.assertTrue(rag_dir.exists())
        model_summary = json.loads((rag_dir / "summary.json").read_text(encoding="utf-8"))
        question_batches = json.loads((rag_dir / "question_batches.json").read_text(encoding="utf-8"))
        retrieval_store = json.loads((rag_dir / "retrieval_store.json").read_text(encoding="utf-8"))
        self.assertEqual(model_summary["evaluation_variant"], "embedding_rag")
        self.assertEqual(model_summary["base_model_slug"], "qwen3.5-4b")
        self.assertTrue(model_summary["operational_metrics"]["rag_passthrough"])
        self.assertTrue(question_batches["rag_passthrough"])
        self.assertTrue(retrieval_store["rag_passthrough"])
        self.assertEqual(retrieval_store["passthrough_source_model_slug"], "qwen3.5-4b")

    def test_mem0_star_build_store_is_add_only_and_deduplicates_exact_memories(self) -> None:
        patient_root = self._write_patient_artifacts()
        combined_payload = json.loads((patient_root / "combined_conversation.json").read_text(encoding="utf-8"))
        settings = build_settings(
            self._config(),
            provider="vllm",
            base_url="http://127.0.0.1:8000/v1",
            judge_base_url=None,
            api_key_env=None,
            models=["Qwen/Qwen3.5-4B"],
            stage="answers",
            evaluation_variant="mem0_star",
            replace_existing=True,
        )
        memory_client = FakeLLMClient(
            [
                {
                    "parsed_output": {
                        "memories": [
                            {
                                "candidate_id": "c001",
                                "memory": "On 2020-01-01 08:00:00, fever improved with dialysis.",
                            }
                        ]
                    }
                },
                {"parsed_output": {"summary": "Fever improved with dialysis."}},
                {
                    "parsed_output": {
                        "memories": [
                            {
                                "candidate_id": "c001",
                                "memory": "On 2020-01-01 08:00:00, fever improved with dialysis.",
                            },
                            {
                                "candidate_id": "c002",
                                "memory": "On 2020-02-01 10:10:00, the patient could walk farther.",
                            },
                        ]
                    }
                },
                {"parsed_output": {"summary": "The patient could walk farther."}},
            ]
        )

        result = build_mem0_memory_store(
            memory_client,
            combined_payload=combined_payload,
            settings=settings,
            model_name="Qwen/Qwen3.5-4B",
            retriever=fake_dense_retriever(),
        )

        self.assertEqual(result.store.evaluation_variant, "mem0_star")
        self.assertEqual(result.metrics["update_call_count"], 0)
        self.assertEqual(result.metrics["add_count"], 2)
        self.assertEqual(result.metrics["duplicate_skip_count"], 1)
        self.assertEqual(result.store_payload["mode"], "mem0_star")
        self.assertEqual(result.store_payload["evaluation_variant"], "mem0_star")
        self.assertEqual(result.store_payload["settings"]["memory_method"], "mem0_star")
        self.assertTrue(
            all(set(record) == {"memory_id", "memory"} for record in result.store_payload["memories"])
        )
        self.assertEqual([call["schema"] for call in memory_client.calls], [
            "Mem0ExtractionResponse",
            "Mem0SummaryResponse",
            "Mem0ExtractionResponse",
            "Mem0SummaryResponse",
        ])
        extraction_calls = [call for call in memory_client.calls if call["schema"] == "Mem0ExtractionResponse"]
        self.assertIn("related_existing_memories", extraction_calls[1]["user_message"])
        for call in extraction_calls:
            visible_prompt = f"{call['system_message']}\n{call['user_message']}"
            self.assertNotIn("Mem0", visible_prompt)
            self.assertNotIn("mem0", visible_prompt)
            self.assertNotIn("memory_id", visible_prompt)
            self.assertNotIn('"score"', visible_prompt)
            self.assertNotIn("hadm_id", visible_prompt)
            self.assertNotIn("turn=", visible_prompt)

    def test_memalpha_build_store_uses_typed_add_only_memories_and_grouped_context(self) -> None:
        patient_root = self._write_patient_artifacts()
        combined_payload = json.loads((patient_root / "combined_conversation.json").read_text(encoding="utf-8"))
        settings = build_settings(
            self._config(),
            provider="vllm",
            base_url="http://127.0.0.1:8000/v1",
            judge_base_url=None,
            api_key_env=None,
            models=["Qwen/Qwen3.5-4B"],
            stage="answers",
            evaluation_variant="memalpha",
            replace_existing=True,
        )
        memory_client = FakeLLMClient(
            [
                {
                    "parsed_output": {
                        "core": [
                            {
                                "candidate_id": "core001",
                                "memory": "The patient has chronic dialysis needs.",
                            }
                        ],
                        "episodic": [
                            {
                                "candidate_id": "epi001",
                                "memory": "On 2020-01-01 08:00:00, fever improved with dialysis.",
                            }
                        ],
                        "semantic": [
                            {
                                "candidate_id": "sem001",
                                "memory": "Dialysis was associated with improvement in patient-specific symptoms.",
                            }
                        ],
                    }
                },
                {"parsed_output": {"summary": "Dialysis-related improvement was discussed."}},
                {"parsed_output": {"core": [], "episodic": [], "semantic": []}},
                {"parsed_output": {"summary": "No additional structured memories."}},
            ]
        )

        result = build_mem0_memory_store(
            memory_client,
            combined_payload=combined_payload,
            settings=settings,
            model_name="Qwen/Qwen3.5-4B",
            retriever=fake_dense_retriever(),
        )

        self.assertEqual(result.store.evaluation_variant, "memalpha")
        self.assertEqual(result.metrics["update_call_count"], 0)
        self.assertEqual(result.metrics["add_count"], 3)
        self.assertEqual(result.store_payload["mode"], "memalpha")
        self.assertEqual(result.store_payload["settings"]["memory_method"], "memalpha")
        self.assertTrue(
            all(set(record) == {"memory_id", "memory_type", "memory"} for record in result.store_payload["memories"])
        )
        self.assertEqual(
            sorted(record["memory_type"] for record in result.store_payload["memories"]),
            ["core", "episodic", "semantic"],
        )
        question = normalize_benchmark(
            {
                "qas": [
                    {
                        "qa_id": "q1",
                        "scope": "single_admission",
                        "question_type": "medical_reasoning",
                        "question": "What improved with dialysis?",
                        "answer": "fever",
                        "evidence": {"admissions": ["1"], "turn_ids": [1]},
                    }
                ]
            }
        )[0]
        selection = select_mem0_context_for_question(
            memory_store=result.store,
            question=question,
            model_name="Qwen/Qwen3.5-4B",
            tokenizer_name=None,
            max_model_len=4096,
            max_output_tokens=128,
            safe_margin_tokens=0,
            token_estimate_safety_multiplier=1.0,
            retrieval_top_k=10,
            max_answer_memories=10,
        )

        self.assertEqual(selection["context_record"]["strategy"], "memalpha")
        self.assertEqual(selection["context_record"]["evaluation_variant"], "memalpha")
        self.assertTrue(
            any("memory_type" in record for record in selection["context_record"]["selected_memory_records"])
        )
        self.assertIn("Core memory:", selection["context_text"])
        self.assertIn("Episodic memory:", selection["context_text"])
        self.assertIn("Semantic memory:", selection["context_text"])
        self.assertNotIn("memory_id", selection["context_text"])
        self.assertNotIn("dense_score", selection["context_text"])
        extraction_calls = [call for call in memory_client.calls if call["schema"] == "MemAlphaExtractionResponse"]
        self.assertEqual(len(extraction_calls), 2)
        for call in extraction_calls:
            visible_prompt = f"{call['system_message']}\n{call['user_message']}"
            self.assertNotIn("Mem0", visible_prompt)
            self.assertNotIn("mem0", visible_prompt)
            self.assertNotIn("memory_id", visible_prompt)
            self.assertNotIn('"score"', visible_prompt)
            self.assertNotIn("hadm_id", visible_prompt)
            self.assertNotIn("turn=", visible_prompt)

    def test_memory_model_spec_suffixes_follow_memory_variant(self) -> None:
        base_spec = ModelSpec("google/gemma-3-4b-it", "gemma-3-4b-it", 1, 32768)

        self.assertEqual(memory_model_spec_for(base_spec, evaluation_variant="mem0").slug, "gemma-3-4b-it-mem0")
        self.assertEqual(
            memory_model_spec_for(base_spec, evaluation_variant="mem0_star").slug,
            "gemma-3-4b-it-mem0-star",
        )
        self.assertEqual(
            memory_model_spec_for(base_spec, evaluation_variant="memalpha").slug,
            "gemma-3-4b-it-memalpha",
        )

    def test_quest_scripts_support_memory_method_and_one_gpu_controls(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        slurm_text = (repo_root / "quest" / "evaluate_models.slurm").read_text(encoding="utf-8")
        runner_text = (repo_root / "quest" / "run_multi_patient_eval_job.sh").read_text(encoding="utf-8")
        launcher_text = (repo_root / "quest" / "launch_vllm_server.sh").read_text(encoding="utf-8")

        self.assertIn("--memory-method", slurm_text)
        self.assertIn("MEMORY_METHOD", slurm_text)
        self.assertIn("MEMORY_METHOD", runner_text)
        self.assertIn('--memory-method "$MEMORY_METHOD"', runner_text)
        self.assertIn("MEM0_MODEL_GPU_DEVICE_IDS", slurm_text)
        self.assertIn("MEM0_MODEL_GPU_DEVICE_IDS", runner_text)
        self.assertIn("VLLM_GPU_MEMORY_UTILIZATION", slurm_text)
        self.assertIn("VLLM_GPU_MEMORY_UTILIZATION", runner_text)
        self.assertIn("--gpu-memory-utilization", launcher_text)

    def test_mem0_summary_failure_is_nonfatal_and_records_error(self) -> None:
        settings = build_settings(
            self._config(),
            provider="vllm",
            base_url="http://127.0.0.1:8000/v1",
            judge_base_url=None,
            api_key_env=None,
            models=["Qwen/Qwen3.5-4B"],
            evaluation_variant="mem0",
            replace_existing=True,
            retry_limit=0,
        )
        combined_payload = {
            "subject_id": "11826927",
            "admissions": [
                {
                    "hadm_id": "2001",
                    "conversation_lines": [
                        {
                            "turn_number": 1,
                            "time": "2020-01-01 08:00:00",
                            "speaker": "doctor",
                            "text": "The V/Q scan ruled out PE.",
                        }
                    ],
                }
            ],
        }
        client = FakeLLMClient(
            [
                {
                    "parsed_output": {
                        "memories": [
                            {
                                "candidate_id": "c001",
                                "memory": "On 2020-01-01 08:00:00, the V/Q scan ruled out PE.",
                            }
                        ]
                    }
                },
                {
                    "parsed_output": {
                        "actions": [
                            {
                                "candidate_id": "c001",
                                "operation": "ADD",
                                "target_memory_id": None,
                                "memory": "On 2020-01-01 08:00:00, the V/Q scan ruled out PE.",
                            }
                        ]
                    }
                },
                {"parsed_output": {"unexpected": "summary is missing"}},
            ]
        )

        result = build_mem0_memory_store(
            client,
            combined_payload=combined_payload,
            settings=settings,
            model_name="Qwen/Qwen3.5-4B",
            retriever=fake_dense_retriever(),
        )

        self.assertEqual(result.metrics["active_memories"], 1)
        self.assertEqual(result.metrics["summary_call_count"], 0)
        self.assertEqual(result.metrics["summary_error_count"], 1)
        self.assertEqual(result.event_records[-1]["event_type"], "summary_error")
        self.assertTrue(result.event_records[-1]["kept_previous_summary"])
        self.assertEqual(client.calls[-1]["schema"], "Mem0SummaryResponse")
        self.assertEqual(client.calls[-1]["max_output_tokens"], 1024)

    def test_mem0_summary_is_clamped_after_success(self) -> None:
        settings = build_settings(
            self._config(),
            provider="vllm",
            base_url="http://127.0.0.1:8000/v1",
            judge_base_url=None,
            api_key_env=None,
            models=["Qwen/Qwen3.5-4B"],
            evaluation_variant="mem0",
            replace_existing=True,
        )
        combined_payload = {
            "subject_id": "11826927",
            "admissions": [
                {
                    "hadm_id": "2001",
                    "conversation_lines": [
                        {
                            "turn_number": 1,
                            "time": "2020-01-01 08:00:00",
                            "speaker": "doctor",
                            "text": "INR was monitored.",
                        }
                    ],
                }
            ],
        }
        client = FakeLLMClient(
            [
                {
                    "parsed_output": {
                        "memories": [
                            {
                                "candidate_id": "c001",
                                "memory": "On 2020-01-01 08:00:00, INR was monitored.",
                            }
                        ]
                    }
                },
                {
                    "parsed_output": {
                        "actions": [
                            {
                                "candidate_id": "c001",
                                "operation": "ADD",
                                "target_memory_id": None,
                                "memory": "On 2020-01-01 08:00:00, INR was monitored.",
                            }
                        ]
                    }
                },
                {"parsed_output": {"summary": "word " * 400}},
            ]
        )

        result = build_mem0_memory_store(
            client,
            combined_payload=combined_payload,
            settings=settings,
            model_name="Qwen/Qwen3.5-4B",
            retriever=fake_dense_retriever(),
        )

        self.assertLessEqual(len(result.store.summary), 1200)
        self.assertEqual(result.metrics["summary_call_count"], 1)
        self.assertEqual(result.metrics["summary_error_count"], 0)
        self.assertEqual(client.calls[-1]["max_output_tokens"], 1024)

    def test_model_summary_records_answer_failed_percent_without_comparison_columns(self) -> None:
        patient_root = self._write_patient_artifacts()
        base_config = self._config()
        settings = build_settings(
            base_config,
            provider="vllm",
            base_url="http://127.0.0.1:8000/v1",
            judge_base_url=None,
            api_key_env=None,
            models=["Qwen/Qwen3.5-4B"],
            stage="answers",
            replace_existing=True,
            retry_limit=0,
        )
        pipeline = EvaluationPipeline(
            base_config,
            settings,
            client_overrides={
                "Qwen/Qwen3.5-4B": FakeLLMClient(
                    [
                        {"parsed_output": self._predicted_answers_payload(1, 10)},
                        {"parsed_output": {"answers": [{"qa_id": "unknown", "prediction": "bad"}]}},
                    ]
                )
            },
        )

        summary = pipeline.run([(11826927, patient_root)])

        self.assertEqual(summary["final_status"], "completed")
        model_summary = json.loads(
            (self._evaluation_root() / "qwen3.5-4b" / "summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(model_summary["run_status"], "answers_completed")
        self.assertEqual(model_summary["answer_failed_percent"], 16.67)
        self.assertEqual(model_summary["operational_metrics"]["failed_prediction_count"], 2)
        self.assertFalse((self._evaluation_root() / "comparison" / "leaderboard.json").exists())

    def test_pipeline_supports_serial_answers_then_judge_stages(self) -> None:
        patient_root = self._write_patient_artifacts()
        base_config = self._config()
        answer_settings = build_settings(
            base_config,
            provider="vllm",
            base_url="http://127.0.0.1:8000/v1",
            judge_base_url=None,
            api_key_env=None,
            models=["Qwen/Qwen3.5-4B"],
            stage="answers",
            replace_existing=True,
        )
        answer_pipeline = EvaluationPipeline(
            base_config,
            answer_settings,
            client_overrides={
                "Qwen/Qwen3.5-4B": FakeLLMClient(
                    [
                        {"parsed_output": self._predicted_answers_payload(1, 10)},
                        {"parsed_output": self._predicted_answers_payload(11, 12)},
                    ]
                )
            },
        )

        answer_summary = answer_pipeline.run([(11826927, patient_root)])

        self.assertEqual(answer_summary["final_status"], "completed")
        self.assertEqual(answer_summary["results"][0]["stage"], "answers")
        self.assertIsNone(answer_summary["results"][0]["comparison_summary_path"])
        answer_model_summary = json.loads(
            (self._evaluation_root() / "qwen3.5-4b" / "summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(answer_model_summary["run_status"], "answers_completed")
        self.assertEqual(answer_model_summary["llm_score"], 0.0)
        self.assertEqual(
            (self._evaluation_root() / "qwen3.5-4b" / "llm_judgments.jsonl").read_text(encoding="utf-8"),
            "",
        )
        self.assertFalse((self._evaluation_root() / "comparison" / "leaderboard.json").exists())

        answerable_ids = [f"q{index:02d}" for index in range(1, 13) if index not in {4, 11}]
        judge_settings = build_settings(
            base_config,
            provider="vllm",
            base_url=None,
            judge_base_url="http://127.0.0.1:8001/v1",
            api_key_env=None,
            models=["Qwen/Qwen3.5-4B"],
            stage="judge",
            replace_existing=True,
        )
        judge_pipeline = EvaluationPipeline(
            base_config,
            judge_settings,
            client_overrides={
                "__judge__": FakeLLMClient(
                    [
                        {"parsed_output": self._judge_answerable_payload(answerable_ids)},
                    ]
                )
            },
        )

        judge_summary = judge_pipeline.run([(11826927, patient_root)])

        self.assertEqual(judge_summary["final_status"], "completed")
        self.assertEqual(judge_summary["results"][0]["stage"], "judge")
        self.assertTrue((self._evaluation_root() / "comparison" / "leaderboard.json").exists())
        judged_model_summary = json.loads(
            (self._evaluation_root() / "qwen3.5-4b" / "summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(judged_model_summary["run_status"], "completed")
        self.assertEqual(judged_model_summary["llm_score"], 1.0)
        self.assertGreaterEqual(
            judged_model_summary["operational_metrics"]["total_wall_time_seconds"],
            answer_model_summary["operational_metrics"]["total_wall_time_seconds"],
        )

    def test_pipeline_reuses_27b_answer_server_for_judging_when_no_judge_base_url_is_given(self) -> None:
        patient_root = self._write_patient_artifacts()
        base_config = self._config()
        settings = build_settings(
            base_config,
            provider="vllm",
            base_url="http://127.0.0.1:8000/v1",
            judge_base_url=None,
            api_key_env=None,
            models=["Qwen/Qwen3.5-27B"],
            replace_existing=True,
        )
        answerable_ids = [f"q{index:02d}" for index in range(1, 13) if index not in {4, 11}]
        shared_client = FakeLLMClient(
            [
                {"parsed_output": self._predicted_answers_payload(1, 10)},
                {"parsed_output": self._predicted_answers_payload(11, 12)},
                {"parsed_output": self._judge_answerable_payload(answerable_ids)},
            ]
        )
        pipeline = EvaluationPipeline(
            base_config,
            settings,
            client_overrides={"Qwen/Qwen3.5-27B": shared_client},
        )

        summary = pipeline.run([(11826927, patient_root)])

        self.assertEqual(summary["final_status"], "completed")
        self.assertEqual(len(shared_client.calls), 3)
        model_summary = json.loads(
            (self._evaluation_root() / "qwen3.5-27b" / "summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(model_summary["llm_score"], 1.0)

    def test_pipeline_reuses_openai_answer_client_for_same_api_judge_model(self) -> None:
        patient_root = self._write_patient_artifacts()
        base_config = self._config()
        settings = build_settings(
            base_config,
            provider="openai",
            base_url=None,
            judge_base_url=None,
            judge_model="gpt-5.1",
            api_key_env="OPENAI_API_KEY",
            models=["gpt-5.1"],
            replace_existing=True,
        )
        answerable_ids = [f"q{index:02d}" for index in range(1, 13) if index not in {4, 11}]
        shared_client = FakeLLMClient(
            [
                {"parsed_output": self._predicted_answers_payload(1, 10)},
                {"parsed_output": self._predicted_answers_payload(11, 12)},
                {"parsed_output": self._judge_answerable_payload(answerable_ids)},
            ]
        )
        pipeline = EvaluationPipeline(
            base_config,
            settings,
            client_overrides={"gpt-5.1": shared_client},
        )

        summary = pipeline.run([(11826927, patient_root)])

        self.assertEqual(summary["final_status"], "completed")
        self.assertEqual(len(shared_client.calls), 3)
        run_config = json.loads(
            (self._evaluation_root() / "gpt-5.1" / "run_config.json").read_text(encoding="utf-8")
        )
        self.assertEqual(run_config["judge_model_name"], "gpt-5.1")
        self.assertEqual(run_config["judge_model_slug"], "gpt-5.1")
        model_summary = json.loads(
            (self._evaluation_root() / "gpt-5.1" / "summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(model_summary["llm_score"], 1.0)

    def test_pipeline_fails_non_27b_eval_without_judge_base_url(self) -> None:
        patient_root = self._write_patient_artifacts()
        base_config = self._config()
        settings = build_settings(
            base_config,
            provider="vllm",
            base_url="http://127.0.0.1:8000/v1",
            judge_base_url=None,
            api_key_env=None,
            models=["Qwen/Qwen3.5-4B"],
            replace_existing=True,
        )
        pipeline = EvaluationPipeline(
            base_config,
            settings,
            client_overrides={
                "Qwen/Qwen3.5-4B": FakeLLMClient(
                    [
                        {"parsed_output": self._predicted_answers_payload(1, 10)},
                        {"parsed_output": self._predicted_answers_payload(11, 12)},
                    ]
                )
            },
        )

        summary = pipeline.run([(11826927, patient_root)])

        self.assertEqual(summary["final_status"], "failed")
        self.assertEqual(summary["failed"], [11826927])
        model_summary = json.loads(
            (self._evaluation_root() / "qwen3.5-4b" / "summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(model_summary["run_status"], "failed_input_error")

    def test_main_evaluate_accepts_patient_manifest(self) -> None:
        manifest = self.root / "patients.txt"
        manifest.write_text("11826927\n17207245\n", encoding="utf-8")
        fake_pipeline = Mock()
        fake_pipeline.run.return_value = {
            "requested_subject_ids": [11826927, 17207245],
            "failed": [],
            "final_status": "completed",
            "results": [],
        }
        fake_config = self._config()

        with patch.object(main_cli, "build_default_config", return_value=fake_config), patch.object(
            main_cli,
            "EvaluationPipeline",
            return_value=fake_pipeline,
        ), patch.object(
            main_cli,
            "build_evaluation_settings",
            return_value=Mock(),
        ) as build_settings_mock, patch.object(
            main_cli,
            "resolve_patient_targets",
            return_value=[(11826927, self.output_root / "11826927"), (17207245, self.output_root / "17207245")],
        ):
            exit_code = main_cli.main(
                [
                    "evaluate",
                    "--output-root",
                    str(self.output_root),
                    "--evaluation-root",
                    str(self.root / "custom_eval"),
                    "--stage",
                    "judge",
                    "--patient-manifest",
                    str(manifest),
                    "--models",
                    "Qwen/Qwen3.5-4B",
                    "--retry-limit",
                    "3",
                    "--timeout-seconds",
                    "600",
                    "--judge-model",
                    "gpt-5.1",
                ]
            )

        self.assertEqual(exit_code, 0)
        build_settings_mock.assert_called_once()
        self.assertEqual(build_settings_mock.call_args.kwargs["stage"], "judge")
        self.assertEqual(build_settings_mock.call_args.kwargs["evaluation_root"], self.root / "custom_eval")
        self.assertEqual(build_settings_mock.call_args.kwargs["retry_limit"], 3)
        self.assertEqual(build_settings_mock.call_args.kwargs["timeout_seconds"], 600)
        self.assertEqual(build_settings_mock.call_args.kwargs["judge_model"], "gpt-5.1")
        fake_pipeline.run.assert_called_once()

    def test_main_evaluate_memory_wires_memory_pipeline_and_settings(self) -> None:
        fake_pipeline = Mock()
        fake_pipeline.run.return_value = {
            "requested_subject_ids": [11826927],
            "failed": [],
            "final_status": "completed",
            "results": [],
        }
        fake_config = self._config()

        with patch.object(main_cli, "build_default_config", return_value=fake_config), patch.object(
            main_cli,
            "MemoryEvaluationPipeline",
            return_value=fake_pipeline,
        ), patch.object(
            main_cli,
            "build_memory_settings",
            return_value=Mock(),
        ) as build_settings_mock, patch.object(
            main_cli,
            "resolve_patient_targets",
            return_value=[(11826927, self.output_root / "11826927")],
        ):
            exit_code = main_cli.main(
                [
                    "evaluate-memory",
                    "--output-root",
                    str(self.output_root),
                    "--subject-id",
                    "11826927",
                    "--models",
                    "Qwen/Qwen3.5-4B",
                    "--memory-method",
                    "memalpha",
                    "--mem0-chunk-token-cap",
                    "16000",
                    "--mem0-max-candidate-memories",
                    "80",
                    "--mem0-answer-retrieval-top-k",
                    "50",
                    "--mem0-max-answer-memories",
                    "35",
                    "--mem0-embedding-model",
                    "Qwen/Qwen3-Embedding-4B",
                    "--mem0-embedding-device",
                    "cuda",
                    "--mem0-embedding-gpu-device-ids",
                    "1",
                    "--mem0-embedding-batch-size",
                    "4",
                    "--mem0-embedding-max-length",
                    "512",
                    "--mem0-model-max-len",
                    "32768",
                    "--mem0-model-tensor-parallel-size",
                    "1",
                ]
            )

        self.assertEqual(exit_code, 0)
        build_settings_mock.assert_called_once()
        self.assertEqual(build_settings_mock.call_args.kwargs["stage"], "full")
        self.assertEqual(build_settings_mock.call_args.kwargs["memory_method"], "memalpha")
        self.assertEqual(build_settings_mock.call_args.kwargs["mem0_chunk_token_cap"], 16000)
        self.assertEqual(build_settings_mock.call_args.kwargs["mem0_max_candidate_memories"], 80)
        self.assertEqual(build_settings_mock.call_args.kwargs["mem0_answer_retrieval_top_k"], 50)
        self.assertEqual(build_settings_mock.call_args.kwargs["mem0_max_answer_memories"], 35)
        self.assertEqual(build_settings_mock.call_args.kwargs["mem0_embedding_model"], "Qwen/Qwen3-Embedding-4B")
        self.assertEqual(build_settings_mock.call_args.kwargs["mem0_embedding_device"], "cuda")
        self.assertEqual(build_settings_mock.call_args.kwargs["mem0_embedding_gpu_device_ids"], "1")
        self.assertEqual(build_settings_mock.call_args.kwargs["mem0_embedding_batch_size"], 4)
        self.assertEqual(build_settings_mock.call_args.kwargs["mem0_embedding_max_length"], 512)
        self.assertEqual(build_settings_mock.call_args.kwargs["mem0_model_max_len"], 32768)
        self.assertEqual(build_settings_mock.call_args.kwargs["mem0_model_tensor_parallel_size"], 1)
        fake_pipeline.run.assert_called_once()

    def test_cohort_summary_includes_partial_models_with_weighted_metrics(self) -> None:
        from health_benchmark.evaluation.cohort_summary import (
            COHORT_LEADERBOARD_FIELDNAMES,
            COHORT_QUESTION_TYPE_LEADERBOARD_FIELDNAMES,
            summarize_evaluation_cohort,
        )

        evaluation_root = self.root / "output" / "evaluation"
        manifest = self.root / "cohort.txt"
        manifest.write_text("101\n102\n", encoding="utf-8")
        self._write_cohort_summary(
            evaluation_root,
            "101",
            "model-a",
            model_name="Model/A",
            total=10,
            answerable=6,
            adversarial=4,
            overall=0.5,
            f1=0.2,
            llm=0.4,
            adv_acc=0.8,
            single_answerable=(2, 0.1, 0.2),
            cross_answerable=(4, 0.3, 0.5),
            single_adversarial=(1, 0.0),
            cross_adversarial=(3, 1.0),
            question_type_breakdowns={
                "medical_reasoning": (2, 0.2, 0.4),
                "care_plan_rationale": (1, 0.3, 0.1),
                "longitudinal_progression": (1, 0.4, 0.2),
                "cross_admission_comparison": (1, 0.5, 0.3),
                "frequency_pattern": (1, 0.6, 0.4),
            },
        )
        self._write_cohort_summary(
            evaluation_root,
            "102",
            "model-a",
            model_name="Model/A",
            total=20,
            answerable=10,
            adversarial=10,
            overall=0.8,
            f1=0.6,
            llm=0.7,
            adv_acc=0.9,
            single_answerable=(4, 0.5, 0.6),
            cross_answerable=(6, 0.7, 0.8),
            single_adversarial=(5, 0.8),
            cross_adversarial=(5, 1.0),
            question_type_breakdowns={
                "medical_reasoning": (3, 0.6, 0.8),
                "care_plan_rationale": (2, 0.5, 0.7),
                "longitudinal_progression": (2, 0.8, 0.6),
                "cross_admission_comparison": (2, 0.7, 0.9),
                "frequency_pattern": (1, 0.4, 1.0),
            },
        )
        for subject_id in ("101", "102"):
            self._write_cohort_summary(
                evaluation_root,
                subject_id,
                "model-a-mem0",
                model_name="Model/A",
                evaluation_variant="mem0",
                total=5,
                answerable=4,
                adversarial=1,
                overall=0.1,
                f1=0.1,
                llm=0.1,
                adv_acc=0.1,
                single_answerable=(2, 0.1, 0.1),
                cross_answerable=(2, 0.1, 0.1),
                single_adversarial=(1, 0.1),
                cross_adversarial=(0, 0.0),
            )
        self._write_cohort_summary(evaluation_root, "101", "missing-model", model_name="Missing", total=1, answerable=1, adversarial=0)
        self._write_cohort_summary(evaluation_root, "101", "incomplete-model", model_name="Incomplete", total=1, answerable=1, adversarial=0)
        incomplete_path = evaluation_root / "101" / "incomplete-model" / "summary.json"
        incomplete_payload = json.loads(incomplete_path.read_text(encoding="utf-8"))
        incomplete_payload["run_status"] = "answers_completed"
        incomplete_payload["overall_score"] = 0.9
        incomplete_payload["macro_f1_answerable"] = 0.9
        incomplete_payload["llm_score"] = 0.0
        incomplete_path.write_text(json.dumps(incomplete_payload), encoding="utf-8")
        self._write_cohort_summary(evaluation_root, "102", "failed-model", model_name="Failed", total=1, answerable=1, adversarial=0)
        failed_path = evaluation_root / "102" / "failed-model" / "summary.json"
        failed_payload = json.loads(failed_path.read_text(encoding="utf-8"))
        failed_payload["run_status"] = "failed_internal_error"
        failed_path.write_text(json.dumps(failed_payload), encoding="utf-8")
        bad_dir = evaluation_root / "101" / "bad-model"
        bad_dir.mkdir(parents=True)
        (bad_dir / "summary.json").write_text("{not json", encoding="utf-8")

        summary = summarize_evaluation_cohort(
            evaluation_root=evaluation_root,
            patient_manifest=manifest,
        )

        rows = summary["models"]
        self.assertEqual(
            [row["model_slug"] for row in rows],
            ["model-a", "model-a-mem0", "incomplete-model", "failed-model", "missing-model"],
        )
        model_row = rows[0]
        self.assertEqual(model_row["cohort_status"], "completed")
        self.assertNotIn("num_patients_expected", model_row)
        self.assertNotIn("num_patients_available", model_row)
        self.assertNotIn("num_patients_completed", model_row)
        self.assertNotIn("num_patients_answers_completed", model_row)
        self.assertNotIn("num_patients_failed", model_row)
        self.assertNotIn("num_patients_missing", model_row)
        self.assertEqual(model_row["num_patients"], 2)
        self.assertEqual(model_row["num_questions_total"], 30)
        self.assertEqual(model_row["num_answerable"], 16)
        self.assertEqual(model_row["num_adversarial"], 14)
        self.assertEqual(model_row["overall_score"], 0.72)
        self.assertEqual(model_row["overall_normal_f1"], 0.45)
        self.assertEqual(model_row["overall_normal_llm_score"], 0.5875)
        self.assertEqual(model_row["overall_adversarial_accuracy"], 0.8714)
        self.assertEqual(model_row["single_admission_score"], 0.5667)
        self.assertEqual(model_row["single_admission_normal_f1"], 0.3667)
        self.assertEqual(model_row["single_admission_normal_llm_score"], 0.4667)
        self.assertEqual(model_row["single_admission_adversarial_accuracy"], 0.6667)
        self.assertEqual(model_row["cross_admission_score"], 0.8222)
        self.assertEqual(model_row["cross_admission_normal_f1"], 0.54)
        self.assertEqual(model_row["cross_admission_normal_llm_score"], 0.68)
        self.assertEqual(model_row["cross_admission_adversarial_accuracy"], 1.0)
        partial_row = rows[2]
        self.assertEqual(partial_row["model_slug"], "incomplete-model")
        self.assertEqual(partial_row["cohort_status"], "partial")
        self.assertEqual(partial_row["num_patients"], 1)
        self.assertEqual(partial_row["overall_score"], 0.0)
        self.assertEqual(partial_row["overall_normal_f1"], 0.9)
        self.assertEqual(partial_row["overall_normal_llm_score"], 0.0)
        missing_row = next(row for row in rows if row["model_slug"] == "missing-model")
        self.assertEqual(missing_row["single_admission_score"], 0.0)
        self.assertEqual(missing_row["cross_admission_score"], 0.0)
        failed_row = next(row for row in rows if row["model_slug"] == "failed-model")
        self.assertEqual(failed_row["cohort_status"], "partial")
        coverage = {item["model_slug"]: item for item in summary["model_coverage"]}
        self.assertEqual(coverage["missing-model"]["num_patients_missing"], 1)
        self.assertEqual(coverage["incomplete-model"]["num_patients_answers_completed"], 1)
        self.assertEqual(coverage["failed-model"]["num_patients_failed"], 1)
        self.assertEqual(coverage["bad-model"]["cohort_status"], "malformed_only")
        self.assertEqual([item["model_slug"] for item in summary["excluded_models"]], ["bad-model"])
        csv_path = evaluation_root / "cohort_leaderboard_detailed.csv"
        header = csv_path.read_text(encoding="utf-8").splitlines()[0].split(",")
        self.assertEqual(header, COHORT_LEADERBOARD_FIELDNAMES)
        question_type_csv_path = evaluation_root / "cohort_leaderboard_question_types.csv"
        header = question_type_csv_path.read_text(encoding="utf-8").splitlines()[0].split(",")
        self.assertEqual(header, COHORT_QUESTION_TYPE_LEADERBOARD_FIELDNAMES)
        with question_type_csv_path.open("r", encoding="utf-8", newline="") as handle:
            question_type_rows = list(csv.DictReader(handle))
        self.assertEqual(
            [row["model_slug"] for row in question_type_rows],
            [row["model_slug"] for row in rows],
        )
        question_type_model_row = question_type_rows[0]
        self.assertEqual(question_type_model_row["num_adversarial"], "14")
        self.assertEqual(float(question_type_model_row["medical_reasoning_f1"]), 0.44)
        self.assertEqual(float(question_type_model_row["medical_reasoning_llm_score"]), 0.64)
        self.assertEqual(float(question_type_model_row["care_plan_rationale_f1"]), 0.4333)
        self.assertEqual(float(question_type_model_row["care_plan_rationale_llm_score"]), 0.5)
        self.assertEqual(float(question_type_model_row["longitudinal_progression_f1"]), 0.6667)
        self.assertEqual(float(question_type_model_row["longitudinal_progression_llm_score"]), 0.4667)
        self.assertEqual(float(question_type_model_row["cross_admission_comparison_f1"]), 0.6333)
        self.assertEqual(float(question_type_model_row["cross_admission_comparison_llm_score"]), 0.7)
        self.assertEqual(float(question_type_model_row["frequency_pattern_f1"]), 0.5)
        self.assertEqual(float(question_type_model_row["frequency_pattern_llm_score"]), 0.7)
        self.assertEqual(float(question_type_model_row["single_admission_adversarial_accuracy"]), 0.6667)
        self.assertEqual(float(question_type_model_row["cross_admission_adversarial_accuracy"]), 1.0)
        self.assertEqual(summary["outputs"]["question_types_csv"], str(question_type_csv_path))
        self.assertTrue((evaluation_root / "cohort_leaderboard_detailed.json").exists())

    def test_main_summarize_evaluation_cohort_writes_outputs(self) -> None:
        evaluation_root = self.root / "output" / "evaluation"
        manifest = self.root / "cohort.txt"
        manifest.write_text("101\n", encoding="utf-8")
        self._write_cohort_summary(evaluation_root, "101", "model-a", model_name="Model/A", total=1, answerable=1, adversarial=0)

        exit_code = main_cli.main(
            [
                "summarize-evaluation-cohort",
                "--evaluation-root",
                str(evaluation_root),
                "--patient-manifest",
                str(manifest),
            ]
        )

        self.assertEqual(exit_code, 0)
        self.assertTrue((evaluation_root / "cohort_leaderboard_detailed.csv").exists())
        self.assertTrue((evaluation_root / "cohort_leaderboard_question_types.csv").exists())
        payload = json.loads((evaluation_root / "cohort_leaderboard_detailed.json").read_text(encoding="utf-8"))
        self.assertEqual(
            payload["outputs"]["question_types_csv"],
            str(evaluation_root / "cohort_leaderboard_question_types.csv"),
        )
        self.assertEqual([row["model_slug"] for row in payload["models"]], ["model-a"])

    def _write_cohort_summary(
        self,
        evaluation_root: Path,
        subject_id: str,
        model_slug: str,
        *,
        model_name: str,
        evaluation_variant: str = "normal",
        total: int,
        answerable: int,
        adversarial: int,
        overall: float = 0.0,
        f1: float = 0.0,
        llm: float = 0.0,
        adv_acc: float = 0.0,
        single_answerable: tuple[int, float, float] = (0, 0.0, 0.0),
        cross_answerable: tuple[int, float, float] = (0, 0.0, 0.0),
        single_adversarial: tuple[int, float] = (0, 0.0),
        cross_adversarial: tuple[int, float] = (0, 0.0),
        question_type_breakdowns: dict[str, tuple[int, float, float]] | None = None,
    ) -> None:
        by_question_type = {
            question_type: {
                "count": metrics[0],
                "answerable_count": metrics[0],
                "adversarial_count": 0,
                "macro_f1_answerable": metrics[1],
                "llm_score": metrics[2],
            }
            for question_type, metrics in (question_type_breakdowns or {}).items()
        }
        if adversarial:
            by_question_type["adversarial"] = {
                "count": adversarial,
                "answerable_count": 0,
                "adversarial_count": adversarial,
                "adversarial_accuracy": adv_acc,
            }
        payload = {
            "subject_id": subject_id,
            "model_name": model_name,
            "model_slug": model_slug,
            "evaluation_variant": evaluation_variant,
            "run_status": "completed",
            "num_questions_total": total,
            "num_answerable": answerable,
            "num_adversarial": adversarial,
            "overall_score": overall,
            "macro_f1_answerable": f1,
            "llm_score": llm,
            "adversarial_accuracy": adv_acc,
            "breakdowns": {
                "by_answerable_scope": {
                    "single_admission": {
                        "answerable_count": single_answerable[0],
                        "macro_f1_answerable": single_answerable[1],
                        "llm_score": single_answerable[2],
                    },
                    "cross_admission": {
                        "answerable_count": cross_answerable[0],
                        "macro_f1_answerable": cross_answerable[1],
                        "llm_score": cross_answerable[2],
                    },
                },
                "by_adversarial_scope": {
                    "single_admission": {
                        "adversarial_count": single_adversarial[0],
                        "adversarial_accuracy": single_adversarial[1],
                    },
                    "cross_admission": {
                        "adversarial_count": cross_adversarial[0],
                        "adversarial_accuracy": cross_adversarial[1],
                    },
                },
                "by_question_type": by_question_type,
            },
        }
        summary_path = evaluation_root / subject_id / model_slug / "summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(payload), encoding="utf-8")
