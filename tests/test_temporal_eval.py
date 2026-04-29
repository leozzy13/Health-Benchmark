from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import main as main_cli

from health_benchmark.scripts.config import build_default_config
from health_benchmark.temporal_eval.answer_prompting import render_answer_prompt
from health_benchmark.temporal_eval.answer_runner import run_answer_batches
from health_benchmark.temporal_eval.batch_builder import build_batches
from health_benchmark.temporal_eval.config import build_settings
from health_benchmark.temporal_eval.judge_prompting import (
    render_adversarial_judge_prompt,
    render_answerable_judge_prompt,
)
from health_benchmark.temporal_eval.loader import resolve_patient_targets
from health_benchmark.temporal_eval.pipeline import TemporalEvaluationPipeline, normalize_benchmark
from health_benchmark.temporal_eval.scoring import normalize_answer, score_adversarial, score_answerable
from health_benchmark.temporal_eval.token_budget import build_preflight_record
from health_benchmark.temporal_eval.types import CANONICAL_ABSTENTION_ANSWER


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


class TemporalEvalTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.output_root = self.root / "output"
        self.output_root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
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

    def _judge_answerable_payload(self, qa_ids: list[str], *, score: float = 1.0) -> dict[str, object]:
        return {
            "judgments": [{"qa_id": qa_id, "score": score} for qa_id in qa_ids]
        }

    def _judge_adversarial_payload(self, qa_ids: list[str], *, score: int = 1) -> dict[str, object]:
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
        self.assertNotIn("If the answer is not supported by the conversation", rendered.system_message)
        self.assertNotIn("multiple_choice", rendered.system_message)

    def test_render_llm_judge_prompts_use_minimal_inputs(self) -> None:
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
        self.assertIn("Each score must be exactly one of: 0, 0.5, 1.", rendered_answerable.system_message)

        rendered_adversarial = render_adversarial_judge_prompt(
            [{"qa_id": "q4", "candidate_answer": "not mentioned in the record"}]
        )
        adversarial_payload = json.loads(rendered_adversarial.user_message)
        self.assertEqual(
            adversarial_payload["items"],
            [{"qa_id": "q4", "candidate_answer": "not mentioned in the record"}],
        )
        self.assertIn("Each score must be exactly one of: 0, 1.", rendered_adversarial.system_message)

    def test_build_settings_uses_larger_default_judge_output_budget(self) -> None:
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
            context_text="small context",
            settings=settings,
            model_name="Qwen/Qwen3.5-4B",
        )
        self.assertEqual([len(batch.questions) for batch in batches], [10, 2])
        self.assertEqual(batches[0].qa_ids()[0], "q01")
        self.assertEqual(batches[1].qa_ids(), ["q11", "q12"])

    def test_preflight_returns_context_too_long_for_fixed_batch_ten(self) -> None:
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
        )
        self.assertEqual(record["status"], "context_too_long_for_fixed_batch_10")
        self.assertEqual(record["batch_size"], 10)

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

    def test_scoring_supports_normalization_adversarial_alias_and_comma_fallback(self) -> None:
        self.assertEqual(normalize_answer("The, Fever!"), "fever")
        metrics = score_answerable("vancomycin, cefepime", "cefepime, vancomycin")
        self.assertGreater(metrics["f1"], 0.9)
        self.assertTrue(metrics["used_comma_fallback"])
        self.assertEqual(score_adversarial("Not mentioned"), 1.0)

    def test_run_answer_batches_recovers_missing_qa_ids_with_retry(self) -> None:
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
            context_text="small context",
            settings=settings,
            model_name="Qwen/Qwen3.5-4B",
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
            context_text="small context",
            model_name="Qwen/Qwen3.5-4B",
            save_raw_response=True,
            max_output_tokens=settings.max_output_tokens,
        )
        self.assertEqual([prediction.qa_id for prediction in predictions], ["q1", "q2"])
        self.assertEqual(errors, [])
        self.assertEqual(failed_statuses, {})
        self.assertEqual(raw_records[0]["missing_qa_ids"], ["q2"])
        self.assertIn("retry", answer_client.calls[1]["system_message"].lower())

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
        adversarial_ids = ["q04", "q11"]
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
                    {"parsed_output": self._judge_adversarial_payload(adversarial_ids)},
                    {"parsed_output": self._judge_answerable_payload(answerable_ids)},
                    {"parsed_output": self._judge_adversarial_payload(adversarial_ids)},
                ]
            ),
        }
        pipeline = TemporalEvaluationPipeline(
            base_config,
            settings,
            client_overrides=client_overrides,
        )

        summary = pipeline.run([(11826927, patient_root)])

        self.assertEqual(summary["final_status"], "completed")
        result = summary["results"][0]
        self.assertEqual(result["status"], "completed")
        evaluation_root = patient_root / "evaluation"
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
        self.assertNotIn("answerable_count", model_summary["breakdowns"]["by_scope"]["cross_admission"])

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
        answer_pipeline = TemporalEvaluationPipeline(
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
            (patient_root / "evaluation" / "qwen3.5-4b" / "summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(answer_model_summary["run_status"], "answers_completed")
        self.assertEqual(answer_model_summary["llm_score"], 0.0)
        self.assertEqual(
            (patient_root / "evaluation" / "qwen3.5-4b" / "llm_judgments.jsonl").read_text(encoding="utf-8"),
            "",
        )
        self.assertFalse((patient_root / "evaluation" / "comparison" / "leaderboard.json").exists())

        answerable_ids = [f"q{index:02d}" for index in range(1, 13) if index not in {4, 11}]
        adversarial_ids = ["q04", "q11"]
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
        judge_pipeline = TemporalEvaluationPipeline(
            base_config,
            judge_settings,
            client_overrides={
                "__judge__": FakeLLMClient(
                    [
                        {"parsed_output": self._judge_answerable_payload(answerable_ids)},
                        {"parsed_output": self._judge_adversarial_payload(adversarial_ids)},
                    ]
                )
            },
        )

        judge_summary = judge_pipeline.run([(11826927, patient_root)])

        self.assertEqual(judge_summary["final_status"], "completed")
        self.assertEqual(judge_summary["results"][0]["stage"], "judge")
        self.assertTrue((patient_root / "evaluation" / "comparison" / "leaderboard.json").exists())
        judged_model_summary = json.loads(
            (patient_root / "evaluation" / "qwen3.5-4b" / "summary.json").read_text(encoding="utf-8")
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
        adversarial_ids = ["q04", "q11"]
        shared_client = FakeLLMClient(
            [
                {"parsed_output": self._predicted_answers_payload(1, 10)},
                {"parsed_output": self._predicted_answers_payload(11, 12)},
                {"parsed_output": self._judge_answerable_payload(answerable_ids)},
                {"parsed_output": self._judge_adversarial_payload(adversarial_ids)},
            ]
        )
        pipeline = TemporalEvaluationPipeline(
            base_config,
            settings,
            client_overrides={"Qwen/Qwen3.5-27B": shared_client},
        )

        summary = pipeline.run([(11826927, patient_root)])

        self.assertEqual(summary["final_status"], "completed")
        self.assertEqual(len(shared_client.calls), 4)
        model_summary = json.loads(
            (patient_root / "evaluation" / "qwen3.5-27b" / "summary.json").read_text(encoding="utf-8")
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
        pipeline = TemporalEvaluationPipeline(
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
            (patient_root / "evaluation" / "qwen3.5-4b" / "summary.json").read_text(encoding="utf-8")
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
            "TemporalEvaluationPipeline",
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
                    "--stage",
                    "judge",
                    "--patient-manifest",
                    str(manifest),
                    "--models",
                    "Qwen/Qwen3.5-4B",
                ]
            )

        self.assertEqual(exit_code, 0)
        build_settings_mock.assert_called_once()
        self.assertEqual(build_settings_mock.call_args.kwargs["stage"], "judge")
        fake_pipeline.run.assert_called_once()
