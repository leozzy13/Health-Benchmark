from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import main as main_cli
from health_benchmark.scripts import BenchmarkPipeline, build_default_config, verify_patient_outputs
from health_benchmark.scripts.config import DEFAULT_VLLM_BASE_URL
from health_benchmark.scripts.llm_client import (
    LLMCallResult,
    OpenAILLMClient,
    VLLMLLMClient,
    build_llm_client,
)
from health_benchmark.scripts.prompting import append_repair_message, render_prompt
from health_benchmark.scripts.qa_pipeline import (
    SINGLE_ADMISSION_ADVERSARIAL_COUNT,
    SINGLE_ADMISSION_REGULAR_COUNT,
    SINGLE_ADMISSION_QA_COUNT,
    compute_cross_admission_qa_count,
    compute_cross_adversarial_count,
)
from health_benchmark.scripts.qa_prompting import (
    append_qa_repair_message,
    render_cross_admission_adversarial_qa_prompt,
    render_cross_admission_regular_qa_prompt,
    render_single_admission_adversarial_qa_prompt,
    render_single_admission_regular_qa_prompt,
)
from health_benchmark.scripts.qa_validation import (
    CANONICAL_ADVERSARIAL_ANSWER,
    SingleAdmissionQAFile,
    QAValidationError,
    validate_cross_admission_qa,
    validate_single_admission_qa,
)
from health_benchmark.scripts.validation import ValidationError, validate_generation


class FakeEncoding:
    def __init__(self, name: str) -> None:
        self.name = name

    def encode(self, text: str) -> list[str]:
        return text.split()


class FakeTikToken:
    def encoding_for_model(self, _model: str):
        raise KeyError("unknown model")

    def get_encoding(self, name: str) -> FakeEncoding:
        return FakeEncoding(name)


class FakeLLMClient:
    def __init__(self, payloads: list[dict[str, object]]) -> None:
        self.payloads = list(payloads)
        self.structured_calls: list[dict[str, object]] = []

    def _pop_result(self) -> LLMCallResult:
        payload = self.payloads.pop(0)
        return LLMCallResult(
            parsed_output=payload["parsed_output"],  # type: ignore[arg-type]
            raw_response={"fake": True},
            usage=payload["usage"],  # type: ignore[arg-type]
            response_id="resp_fake",
            latency_ms=5,
        )

    def generate_response(self, _system_message: str, _user_message: str) -> LLMCallResult:
        return self._pop_result()

    def generate_structured_response(
        self,
        _system_message: str,
        _user_message: str,
        _response_schema,
        *,
        max_output_tokens=None,
    ) -> LLMCallResult:
        self.structured_calls.append(
            {
                "schema": getattr(_response_schema, "__name__", str(_response_schema)),
                "max_output_tokens": max_output_tokens,
            }
        )
        return self._pop_result()


class FakeChatCompletionResponse:
    def __init__(self, content: str) -> None:
        self.id = "resp_vllm"
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=content))]
        self.usage = SimpleNamespace(prompt_tokens=11, completion_tokens=7, total_tokens=18)

    def model_dump(self, warnings: bool = False) -> dict[str, object]:
        del warnings
        return {"content": self.choices[0].message.content}


class FakeOpenAIClientFactory:
    def __init__(
        self,
        *,
        chat_content: str | None = None,
        parse_payload: dict[str, object] | None = None,
    ) -> None:
        self.chat_content = chat_content
        self.parse_payload = parse_payload or {}
        self.init_kwargs: list[dict[str, object]] = []
        self.chat_requests: list[dict[str, object]] = []
        self.parse_requests: list[dict[str, object]] = []

    def __call__(self, **kwargs):
        self.init_kwargs.append(kwargs)
        return SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=self._create_chat_completion)),
            responses=SimpleNamespace(parse=self._parse_response),
        )

    def _create_chat_completion(self, **kwargs):
        self.chat_requests.append(kwargs)
        return FakeChatCompletionResponse(self.chat_content or "{}")

    def _parse_response(self, **kwargs):
        self.parse_requests.append(kwargs)
        schema = kwargs["text_format"]
        parsed = schema.model_validate(self.parse_payload)
        return SimpleNamespace(
            id="resp_openai",
            output_parsed=parsed,
            usage=SimpleNamespace(input_tokens=13, output_tokens=5, total_tokens=18),
            model_dump=lambda warnings=False: {"warnings": warnings, "payload": self.parse_payload},
        )


class BenchmarkTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.hosp_dir = self.root / "mimic-iv" / "hosp"
        self.note_dir = self.root / "mimic-iv-notes"
        self.output_dir = self.root / "output"
        self.hosp_dir.mkdir(parents=True, exist_ok=True)
        self.note_dir.mkdir(parents=True, exist_ok=True)
        self._write_fixture_csvs()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _config(self):
        project_dir = Path(__file__).resolve().parents[1] / "health_benchmark"
        config = build_default_config(project_dir)
        config.dataset.mimiciv_hosp_path = self.hosp_dir
        config.dataset.mimiciv_note_path = self.note_dir
        config.output.root = self.output_dir
        config.selection.subject_id = 100
        config.selection.max_admissions = None
        config.openai.model = "fake-model"
        config.openai.max_retries = 1
        return config

    def _write_csv(self, path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def _write_json_file(self, path: Path, payload: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def _write_fixture_csvs(self) -> None:
        self._write_csv(
            self.hosp_dir / "admissions.csv",
            ["subject_id", "hadm_id", "admittime", "dischtime"],
            [
                {"subject_id": 100, "hadm_id": 10, "admittime": "2020-01-01 08:00:00", "dischtime": ""},
                {"subject_id": 100, "hadm_id": 11, "admittime": "2020-02-01 08:00:00", "dischtime": "2020-02-03 10:00:00"},
                {"subject_id": 100, "hadm_id": 12, "admittime": "2020-03-01 08:00:00", "dischtime": "2020-03-02 09:00:00"},
                {"subject_id": 200, "hadm_id": 20, "admittime": "2020-01-01 08:00:00", "dischtime": "2020-01-02 08:00:00"},
                {"subject_id": 200, "hadm_id": 21, "admittime": "2020-02-01 08:00:00", "dischtime": "2020-02-02 08:00:00"},
                {"subject_id": 300, "hadm_id": 30, "admittime": "2020-01-01 08:00:00", "dischtime": "2020-01-02 08:00:00"},
            ],
        )
        self._write_csv(
            self.hosp_dir / "diagnoses_icd.csv",
            ["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"],
            [
                {"subject_id": 100, "hadm_id": 10, "seq_num": 1, "icd_code": "A1", "icd_version": 10},
                {"subject_id": 100, "hadm_id": 10, "seq_num": 2, "icd_code": "A2", "icd_version": 10},
                {"subject_id": 100, "hadm_id": 12, "seq_num": 1, "icd_code": "A1", "icd_version": 10},
                {"subject_id": 100, "hadm_id": 12, "seq_num": 2, "icd_code": "A2", "icd_version": 10},
            ],
        )
        self._write_csv(
            self.hosp_dir / "d_icd_diagnoses.csv",
            ["icd_code", "icd_version", "long_title"],
            [
                {"icd_code": "A1", "icd_version": 10, "long_title": "Pneumonia"},
                {"icd_code": "A2", "icd_version": 10, "long_title": "Respiratory failure"},
                {"icd_code": "B1", "icd_version": 10, "long_title": "Appendicitis"},
            ],
        )
        self._write_csv(
            self.hosp_dir / "procedures_icd.csv",
            ["subject_id", "hadm_id", "seq_num", "chartdate", "icd_code", "icd_version"],
            [
                {"subject_id": 100, "hadm_id": 10, "seq_num": 1, "chartdate": "2020-01-01", "icd_code": "P1", "icd_version": 10},
                {"subject_id": 100, "hadm_id": 10, "seq_num": 1, "chartdate": "2020-01-01", "icd_code": "P1", "icd_version": 10},
                {"subject_id": 100, "hadm_id": 12, "seq_num": 1, "chartdate": "2020-03-01", "icd_code": "P1", "icd_version": 10},
            ],
        )
        self._write_csv(
            self.hosp_dir / "d_icd_procedures.csv",
            ["icd_code", "icd_version", "long_title"],
            [
                {"icd_code": "P1", "icd_version": 10, "long_title": "Bronchoscopy"},
                {"icd_code": "P2", "icd_version": 10, "long_title": "Appendectomy"},
            ],
        )
        self._write_csv(
            self.hosp_dir / "microbiologyevents.csv",
            ["subject_id", "hadm_id", "chartdate", "charttime", "spec_type_desc", "test_name", "org_name", "ab_name", "interpretation", "comments"],
            [
                {
                    "subject_id": 100,
                    "hadm_id": 10,
                    "chartdate": "2020-01-01",
                    "charttime": "",
                    "spec_type_desc": "Blood culture",
                    "test_name": "Culture",
                    "org_name": "Staphylococcus aureus",
                    "ab_name": "Vancomycin",
                    "interpretation": "S",
                    "comments": "Positive",
                },
                {
                    "subject_id": 100,
                    "hadm_id": 10,
                    "chartdate": "2020-01-01",
                    "charttime": "",
                    "spec_type_desc": "Blood culture",
                    "test_name": "Culture",
                    "org_name": "Staphylococcus aureus",
                    "ab_name": "Vancomycin",
                    "interpretation": "S",
                    "comments": "Positive",
                },
                {
                    "subject_id": 100,
                    "hadm_id": 12,
                    "chartdate": "2020-03-01",
                    "charttime": "2020-03-01 10:00:00",
                    "spec_type_desc": "Sputum",
                    "test_name": "Culture",
                    "org_name": "Klebsiella pneumoniae",
                    "ab_name": "Ceftriaxone",
                    "interpretation": "S",
                    "comments": "Positive",
                },
            ],
        )
        self._write_csv(
            self.note_dir / "discharge.csv",
            ["note_id", "subject_id", "hadm_id", "note_seq", "charttime", "storetime", "text"],
            [
                {
                    "note_id": 1,
                    "subject_id": 100,
                    "hadm_id": 10,
                    "note_seq": 1,
                    "charttime": "2020-01-02 10:00:00",
                    "storetime": "2020-01-02 11:00:00",
                    "text": "Hospital course: pneumonia treated with antibiotics. Chest x-ray improved.",
                },
                {
                    "note_id": 2,
                    "subject_id": 100,
                    "hadm_id": 12,
                    "note_seq": 1,
                    "charttime": "2020-03-02 09:00:00",
                    "storetime": "2020-03-02 09:30:00",
                    "text": "Hospital course: respiratory symptoms improved after bronchoscopy.",
                },
                {
                    "note_id": 3,
                    "subject_id": 200,
                    "hadm_id": 20,
                    "note_seq": 1,
                    "charttime": "2020-01-02 08:00:00",
                    "storetime": "2020-01-02 08:30:00",
                    "text": "Discharge note",
                },
                {
                    "note_id": 4,
                    "subject_id": 200,
                    "hadm_id": 21,
                    "note_seq": 1,
                    "charttime": "2020-02-02 08:00:00",
                    "storetime": "2020-02-02 08:30:00",
                    "text": "Discharge note",
                },
                {
                    "note_id": 5,
                    "subject_id": 300,
                    "hadm_id": 30,
                    "note_seq": 1,
                    "charttime": "2020-01-02 08:00:00",
                    "storetime": "2020-01-02 08:30:00",
                    "text": "Discharge note",
                },
            ],
        )
        self._write_csv(
            self.note_dir / "radiology.csv",
            ["note_id", "subject_id", "hadm_id", "charttime", "storetime", "text"],
            [
                {
                    "note_id": 10,
                    "subject_id": 100,
                    "hadm_id": 10,
                    "charttime": "2020-01-01 12:00:00",
                    "storetime": "2020-01-01 12:15:00",
                    "text": "Chest x-ray showed left lower lobe opacity.",
                },
                {
                    "note_id": 11,
                    "subject_id": 100,
                    "hadm_id": "",
                    "charttime": "2020-01-01 13:00:00",
                    "storetime": "2020-01-01 13:15:00",
                    "text": "Unlinked radiology row.",
                },
                {
                    "note_id": 12,
                    "subject_id": 100,
                    "hadm_id": 12,
                    "charttime": "2020-03-01 12:00:00",
                    "storetime": "2020-03-01 12:10:00",
                    "text": "Chest CT showed bibasilar infiltrates.",
                },
            ],
        )

    def _write_patient_artifacts(self, subject_id: int, admissions: list[dict[str, object]]) -> Path:
        patient_root = self.output_dir / str(subject_id)
        patient_root.mkdir(parents=True, exist_ok=True)
        combined_admissions: list[dict[str, object]] = []
        for admission in admissions:
            hadm_id = str(admission["hadm_id"])
            admission_start = str(admission["admission_start"])
            admission_end = str(admission["admission_end"])
            conversation_lines = admission.get("conversation_lines")
            if not isinstance(conversation_lines, list):
                conversation_lines = [
                    {
                        "turn_number": 1,
                        "time": admission_start,
                        "speaker": "Doctor",
                        "text": f"Discussing admission {hadm_id}.",
                    },
                    {
                        "turn_number": 2,
                        "time": admission_end,
                        "speaker": "Patient",
                        "text": "Understood.",
                    },
                ]
            summary_payload = {
                "admission_start": admission_start,
                "admission_end": admission_end,
                "summary_paragraph": str(admission.get("summary_paragraph") or f"Summary for {hadm_id}."),
                "problems": list(admission.get("problems") or ["Problem"]),
            }
            conversation_payload = {
                "subject_id": str(subject_id),
                "hadm_id": hadm_id,
                "admission_start": admission_start,
                "admission_end": admission_end,
                "conversation_lines": conversation_lines,
            }
            admission_dir = patient_root / hadm_id
            self._write_json_file(admission_dir / "conversation.json", conversation_payload)
            self._write_json_file(admission_dir / "summary.json", summary_payload)
            self._write_json_file(admission_dir / "formed_packet.json", {"hadm_id": hadm_id})
            self._write_json_file(admission_dir / "prompt_record.json", {"hadm_id": hadm_id, "status": "existing"})
            combined_admissions.append(
                {
                    "hadm_id": hadm_id,
                    "admission_start": admission_start,
                    "admission_end": admission_end,
                    "conversation_lines": conversation_lines,
                }
            )

        combined_admissions.sort(key=lambda item: (item["admission_start"], item["hadm_id"]))
        self._write_json_file(
            patient_root / "combined_conversation.json",
            {
                "subject_id": str(subject_id),
                "processed_hadm_ids": [item["hadm_id"] for item in combined_admissions],
                "admissions": combined_admissions,
            },
        )
        self._write_json_file(
            patient_root / "patient_summary.json",
            {
                "subject_id": str(subject_id),
                "eligible_admissions": len(combined_admissions),
                "processed_admissions": len(combined_admissions),
                "processed_hadm_ids": [item["hadm_id"] for item in combined_admissions],
            },
        )
        return patient_root

    def _write_patient_qa_outputs(self, patient_root: Path, admissions: list[dict[str, object]]) -> None:
        hadm_ids = [str(admission["hadm_id"]) for admission in admissions]
        single_payloads: list[dict[str, object]] = []
        for hadm_id in hadm_ids:
            payload = self._make_single_qa_payload(hadm_id)
            single_payloads.append(payload)
            self._write_json_file(patient_root / hadm_id / "qa.json", payload)

        cross_payload = self._make_cross_qa_payload(hadm_ids)
        benchmark_items = [
            qa_item
            for payload in single_payloads
            for qa_item in payload["qas"]  # type: ignore[index]
        ] + list(cross_payload["qas"])  # type: ignore[index]
        import random

        random.Random(int(patient_root.name)).shuffle(benchmark_items)
        benchmark_payload = {"qas": benchmark_items}
        self._write_json_file(patient_root / "cross_admission_qa.json", cross_payload)
        self._write_json_file(patient_root / "benchmark_qa.json", benchmark_payload)

    def _make_single_qa_payload(
        self,
        hadm_id: str,
        count: int = SINGLE_ADMISSION_QA_COUNT,
        *,
        adversarial_count: int = SINGLE_ADMISSION_ADVERSARIAL_COUNT,
    ) -> dict[str, object]:
        regular_count = count - adversarial_count
        qas = list(self._make_single_regular_payload(hadm_id, count=regular_count)["qas"])
        qas.extend(self._make_single_adversarial_payload(hadm_id, count=adversarial_count)["qas"])
        return {"qas": qas}

    def _make_single_regular_payload(
        self,
        hadm_id: str,
        *,
        count: int = SINGLE_ADMISSION_REGULAR_COUNT,
    ) -> dict[str, object]:
        qas: list[dict[str, object]] = []
        regular_types = [
            "medical_reasoning",
            "care_plan_rationale",
        ]
        for index in range(1, count + 1):
            qas.append(
                {
                    "qa_id": f"raw_regular_{index}",
                    "scope": "single_admission",
                    "question_type": regular_types[(index - 1) % len(regular_types)],
                    "question": f"Why did the team make decision {index}?",
                    "answer": f"reason {index}",
                    "evidence": {
                        "admissions": [hadm_id],
                        "turn_ids": [2, 1, 2],
                    },
                }
            )
        return {"qas": qas}

    def _make_single_adversarial_payload(
        self,
        hadm_id: str,
        *,
        count: int = SINGLE_ADMISSION_ADVERSARIAL_COUNT,
    ) -> dict[str, object]:
        qas: list[dict[str, object]] = []
        for index in range(1, count + 1):
            qas.append(
                {
                    "qa_id": f"raw_adv_{index}",
                    "scope": "single_admission",
                    "question_type": "adversarial",
                    "question": f"Which unsupported detail explains decision {index}?",
                    "answer": CANONICAL_ADVERSARIAL_ANSWER,
                    "evidence": {
                        "admissions": [hadm_id],
                        "turn_ids": [2, 1, 2],
                    },
                }
            )
        return {"qas": qas}

    def _make_cross_qa_payload(
        self,
        admissions: list[str],
        count: int | None = None,
        *,
        adversarial_count: int | None = None,
    ) -> dict[str, object]:
        if count is None:
            count = compute_cross_admission_qa_count(len(admissions))
        if adversarial_count is None:
            adversarial_count = compute_cross_adversarial_count(count)
        regular_count = int(count) - int(adversarial_count)
        qas = list(self._make_cross_regular_payload(admissions, count=regular_count)["qas"])
        qas.extend(self._make_cross_adversarial_payload(admissions, count=int(adversarial_count))["qas"])
        return {"qas": qas}

    def _make_cross_regular_payload(
        self,
        admissions: list[str],
        *,
        count: int | None = None,
    ) -> dict[str, object]:
        if count is None:
            count = compute_cross_admission_qa_count(len(admissions)) - compute_cross_adversarial_count(
                compute_cross_admission_qa_count(len(admissions))
            )
        evidence_admissions = admissions[: min(3, len(admissions))]
        qas: list[dict[str, object]] = []
        regular_types = [
            "longitudinal_progression",
            "cross_admission_comparison",
            "frequency_pattern",
        ]
        for index in range(1, int(count) + 1):
            qas.append(
                {
                    "qa_id": f"cross_regular_{index}",
                    "scope": "cross_admission",
                    "question_type": regular_types[(index - 1) % len(regular_types)],
                    "question": f"Which pattern recurred over admissions {index}?",
                    "answer": f"pattern {index}",
                    "evidence": {
                        "admissions": evidence_admissions,
                    },
                }
            )
        return {"qas": qas}

    def _make_cross_adversarial_payload(
        self,
        admissions: list[str],
        *,
        count: int | None = None,
    ) -> dict[str, object]:
        if count is None:
            count = compute_cross_adversarial_count(compute_cross_admission_qa_count(len(admissions)))
        evidence_admissions = admissions[: min(3, len(admissions))]
        qas: list[dict[str, object]] = []
        for index in range(1, int(count) + 1):
            qas.append(
                {
                    "qa_id": f"cross_adv_{index}",
                    "scope": "cross_admission",
                    "question_type": "adversarial",
                    "question": f"Which unsupported longitudinal pattern explains admission set {index}?",
                    "answer": CANONICAL_ADVERSARIAL_ANSWER,
                    "evidence": {
                        "admissions": evidence_admissions,
                    },
                }
            )
        return {"qas": qas}

    def test_extract_admission_packet_applies_filters_and_normalization(self) -> None:
        pipeline = BenchmarkPipeline(self._config())
        self.addCleanup(pipeline.close)
        pipeline._ensure_data_backend()
        assert pipeline.store is not None
        assert pipeline.extractor is not None
        pipeline.store.prepare_subject_cache(100)

        packet = pipeline.extractor.extract_admission_packet(100, 10, None).packet

        self.assertEqual(packet["admission_end"], "2020-01-02 10:00:00")
        self.assertEqual(len(packet["radiology_notes"]), 1)
        self.assertEqual(packet["radiology_notes"][0]["note_id"], "10")
        self.assertEqual(
            [item["normalized_time"] for item in packet["diagnoses"]],
            ["2020-01-02 09:00:00", "2020-01-02 10:00:00"],
        )
        self.assertEqual(packet["procedures"][0]["normalized_time"], "2020-01-01 12:00:00")
        self.assertEqual(len(packet["procedures"]), 1)
        self.assertEqual(packet["microbiology"][0]["normalized_time"], "2020-01-01 12:00:00")
        self.assertEqual(len(packet["microbiology"]), 1)

    def test_prepare_subject_cache_supports_sequential_subject_switches(self) -> None:
        pipeline = BenchmarkPipeline(self._config())
        self.addCleanup(pipeline.close)
        pipeline._ensure_data_backend()
        assert pipeline.store is not None
        assert pipeline.extractor is not None

        observed_counts: list[tuple[int, int]] = []
        for subject_id in (100, 200, 300, 100):
            pipeline.store.prepare_subject_cache(subject_id)
            observed_counts.append((subject_id, len(pipeline.extractor.list_admissions_for_subject(subject_id))))

        self.assertEqual(
            observed_counts,
            [
                (100, 2),
                (200, 2),
                (300, 1),
                (100, 2),
            ],
        )

    def test_validate_generation_rejects_invalid_speaker(self) -> None:
        packet = {
            "admission_start": "2020-01-01 08:00:00",
            "admission_end": "2020-01-01 10:00:00",
            "diagnoses": [{"long_title": "Pneumonia"}],
            "procedures": [{"long_title": "Bronchoscopy"}],
            "microbiology": [{"org_name": "Staphylococcus aureus", "spec_type_desc": "Blood culture"}],
            "radiology_notes": [{"text": "Chest x-ray showed left lower lobe opacity."}],
        }
        bad_output = {
            "conversation_lines": [
                {"turn_number": 1, "time": "2020-01-01 08:05:00", "speaker": "Nurse", "text": "Hello"},
            ],
            "summary": {
                "admission_start": "2020-01-01 08:00:00",
                "admission_end": "2020-01-01 10:00:00",
                "summary_paragraph": "Summary.",
                "problems": ["Pneumonia"],
            },
        }

        with self.assertRaises(ValidationError):
            validate_generation(bad_output, packet)

    def test_validate_generation_accepts_short_well_formed_conversation_without_grounding_gate(self) -> None:
        packet = {
            "admission_start": "2020-01-01 08:00:00",
            "admission_end": "2020-01-01 10:00:00",
            "diagnoses": [{"long_title": "Pneumonia"}],
            "procedures": [{"long_title": "Bronchoscopy"}],
            "microbiology": [{"org_name": "Staphylococcus aureus", "spec_type_desc": "Blood culture"}],
            "radiology_notes": [{"text": "Chest x-ray showed left lower lobe opacity."}],
        }
        output = {
            "conversation_lines": [
                {"turn_number": 1, "time": "2020-01-01 08:05:00", "speaker": "Doctor", "text": "We are treating appendicitis today."},
                {"turn_number": 2, "time": "2020-01-01 08:07:00", "speaker": "Patient", "text": "Thank you for the update."},
            ],
            "summary": {
                "admission_start": "2020-01-01 08:00:00",
                "admission_end": "2020-01-01 10:00:00",
                "summary_paragraph": "Appendicitis was the main issue.",
                "problems": ["Appendicitis"],
            },
        }

        validated = validate_generation(output, packet)
        self.assertEqual(len(validated["conversation_lines"]), 2)
        self.assertEqual(validated["summary"]["problems"], ["Appendicitis"])

    def test_validate_generation_rejects_summary_window_mismatch(self) -> None:
        packet = {
            "admission_start": "2020-01-01 08:00:00",
            "admission_end": "2020-01-01 10:00:00",
            "diagnoses": [{"long_title": "Pneumonia"}],
            "procedures": [],
            "microbiology": [],
            "radiology_notes": [],
        }
        bad_output = {
            "conversation_lines": [
                {"turn_number": 1, "time": "2020-01-01 08:05:00", "speaker": "Doctor", "text": "Your pneumonia is improving."},
            ],
            "summary": {
                "admission_start": "2020-01-01 08:01:00",
                "admission_end": "2020-01-01 10:00:00",
                "summary_paragraph": "Pneumonia improved.",
                "problems": ["Pneumonia"],
            },
        }

        with self.assertRaises(ValidationError):
            validate_generation(bad_output, packet)

    def test_render_prompt_prioritizes_storyline_problems_and_support_context(self) -> None:
        pipeline = BenchmarkPipeline(self._config())
        self.addCleanup(pipeline.close)
        pipeline._ensure_data_backend()
        assert pipeline.store is not None
        assert pipeline.extractor is not None
        pipeline.store.prepare_subject_cache(100)

        packet = pipeline.extractor.extract_admission_packet(100, 10, None).packet
        rendered = render_prompt(packet, {"summary_paragraph": "Prior stay summary.", "problems": ["Prior problem"]})

        self.assertEqual(rendered.recommended_turn_range, [40, 60])
        self.assertIn("Approximate stay length: 2 day(s).", rendered.task_message)
        self.assertIn("Recommended turn range: 40-60 turns.", rendered.task_message)
        self.assertIn("Exact admission window: 2020-01-01 08:00:00 to 2020-01-02 10:00:00.", rendered.task_message)
        self.assertIn("Final self-check before returning JSON:", rendered.task_message)
        self.assertIn("conversation_lines.time nondecreasing and inside the exact admission window", rendered.task_message)
        self.assertIn("summary.admission_start and summary.admission_end copied exactly from the packet", rendered.task_message)
        self.assertIn("no conversation line has empty text", rendered.task_message)
        self.assertIn("Hospital course: pneumonia treated with antibiotics.", rendered.storyline_prompt_block)
        self.assertIn("- Pneumonia", rendered.problem_prompt_block)
        self.assertIn("- Respiratory failure", rendered.problem_prompt_block)
        self.assertIn("Radiology:", rendered.supporting_context_block)
        self.assertIn("Chest x-ray showed left lower lobe opacity.", rendered.supporting_context_block)
        self.assertIn("Procedures:", rendered.supporting_context_block)
        self.assertIn("Microbiology:", rendered.supporting_context_block)
        self.assertIn("Full admission packet for backup context only:", rendered.user_message)

    def test_render_prompt_uses_higher_soft_ranges_for_staged_admissions(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        staged_root = repo_root / "output" / "benchmark" / "_tmp" / "11826927"
        if not staged_root.exists():
            staged_root = repo_root / "output" / "benchmark" / "11826927"
        cases = {
            "22736328": [50, 70],
            "25650592": [60, 85],
            "25681662": [60, 85],
        }

        for hadm_id, expected_range in cases.items():
            packet_path = staged_root / hadm_id / "formed_packet.json"
            if not packet_path.exists():
                self.skipTest(f"staged packet not available: {packet_path}")
            packet = json.loads(packet_path.read_text(encoding="utf-8"))
            rendered = render_prompt(packet, None)
            self.assertEqual(rendered.recommended_turn_range, expected_range)
            self.assertIn("Discharge note 1", rendered.storyline_prompt_block)
            self.assertIn("Radiology:", rendered.supporting_context_block)

    def test_append_repair_message_includes_specific_generation_guidance(self) -> None:
        repaired = append_repair_message(
            "original prompt body",
            "conversation timestamps must be monotonically increasing",
        )

        self.assertIn("original prompt body", repaired)
        self.assertIn("timestamps are nondecreasing inside the exact packet admission window", repaired)
        self.assertIn("turn_number is contiguous starting at 1", repaired)
        self.assertIn("summary.admission_start and summary.admission_end exactly copy the packet", repaired)
        self.assertIn("no line has empty text", repaired)
        self.assertIn("Validation errors: conversation timestamps must be monotonically increasing", repaired)

    def test_append_qa_repair_message_includes_cross_admission_id_guidance(self) -> None:
        repaired = append_qa_repair_message(
            "original qa prompt body",
            "qas[39].evidence.admissions contain unknown hadm_ids: ['211485835']",
        )

        self.assertIn("original qa prompt body", repaired)
        self.assertIn("evidence.admissions must copy only the exact", repaired)
        self.assertIn("admission_id_for_evidence_only values already present in the prompt context", repaired)
        self.assertIn("Do not invent, alter, extend, or reformat admission ids.", repaired)
        self.assertIn("Validation errors: qas[39].evidence.admissions contain unknown hadm_ids", repaired)

    def test_validate_single_admission_qa_rejects_missing_turn_ids(self) -> None:
        payload = {
            "qas": [
                {
                    "qa_id": "x",
                    "scope": "single_admission",
                    "question_type": "medical_reasoning",
                    "question": "Why was treatment escalated?",
                    "answer": "clinical deterioration",
                    "evidence": {"admissions": ["10"], "turn_ids": []},
                }
            ]
        }

        with self.assertRaises(QAValidationError):
            validate_single_admission_qa(
                payload,
                subject_id="100",
                hadm_id="10",
                admission_start="2020-01-01 08:00:00",
                admission_end="2020-01-02 10:00:00",
                valid_turn_ids={1, 2},
                expected_count=1,
                expected_adversarial_count=0,
            )

    def test_validate_single_admission_qa_rejects_removed_question_class_field(self) -> None:
        payload = {
            "qas": [
                {
                    "qa_id": "x",
                    "scope": "single_admission",
                    "question_class": "open_ended",
                    "question_type": "medical_reasoning",
                    "question": "Why was treatment escalated?",
                    "answer": "clinical deterioration",
                    "evidence": {"admissions": ["10"], "turn_ids": [2, 1, 2]},
                }
            ]
        }

        with self.assertRaises(QAValidationError):
            validate_single_admission_qa(
                payload,
                subject_id="100",
                hadm_id="10",
                admission_start="2020-01-01 08:00:00",
                admission_end="2020-01-02 10:00:00",
                valid_turn_ids={1, 2},
                expected_count=1,
                expected_adversarial_count=0,
            )

    def test_validate_single_admission_qa_rejects_removed_options_field(self) -> None:
        payload = {
            "qas": [
                {
                    "qa_id": "x",
                    "scope": "single_admission",
                    "question_type": "medical_reasoning",
                    "question": "Why was treatment escalated?",
                    "options": ["should not be here"],
                    "answer": "clinical deterioration",
                    "evidence": {"admissions": ["10"], "turn_ids": [2, 1, 2]},
                }
            ]
        }

        with self.assertRaises(QAValidationError):
            validate_single_admission_qa(
                payload,
                subject_id="100",
                hadm_id="10",
                admission_start="2020-01-01 08:00:00",
                admission_end="2020-01-02 10:00:00",
                valid_turn_ids={1, 2},
                expected_count=1,
                expected_adversarial_count=0,
            )

    def test_validate_single_admission_qa_normalizes_adversarial_alias_to_canonical_answer(self) -> None:
        payload = {
            "qas": [
                {
                    "qa_id": "x",
                    "scope": "single_admission",
                    "question_type": "adversarial",
                    "question": "Which definitive organism was eventually confirmed?",
                    "answer": "cannot be determined from the record",
                    "evidence": {"admissions": ["10"], "turn_ids": [2, 1, 2]},
                }
            ]
        }

        validated = validate_single_admission_qa(
            payload,
            subject_id="100",
            hadm_id="10",
            admission_start="2020-01-01 08:00:00",
            admission_end="2020-01-02 10:00:00",
            valid_turn_ids={1, 2},
            expected_count=1,
            expected_adversarial_count=1,
            allowed_question_types=("adversarial",),
        )
        self.assertEqual(validated["qas"][0]["answer"], CANONICAL_ADVERSARIAL_ANSWER)

    def test_validate_single_admission_qa_rejects_long_non_adversarial_answer(self) -> None:
        payload = {
            "qas": [
                {
                    "qa_id": "x",
                    "scope": "single_admission",
                    "question_type": "medical_reasoning",
                    "question": "Why was treatment escalated?",
                    "answer": (
                        "Clinical deterioration with persistent fever worsening blood pressure positive blood cultures "
                        "despite initial treatment suggested ongoing sepsis requiring escalation"
                    ),
                    "evidence": {"admissions": ["10"], "turn_ids": [1, 2]},
                }
            ]
        }

        with self.assertRaises(QAValidationError):
            validate_single_admission_qa(
                payload,
                subject_id="100",
                hadm_id="10",
                admission_start="2020-01-01 08:00:00",
                admission_end="2020-01-02 10:00:00",
                valid_turn_ids={1, 2},
                expected_count=1,
                expected_adversarial_count=0,
            )

    def test_validate_single_admission_qa_accepts_10_word_non_adversarial_answer(self) -> None:
        payload = {
            "qas": [
                {
                    "qa_id": "x",
                    "scope": "single_admission",
                    "question_type": "medical_reasoning",
                    "question": "Why was treatment escalated?",
                    "answer": "fever persisted despite antibiotics and blood pressure remained low overnight",
                    "evidence": {"admissions": ["10"], "turn_ids": [1, 2]},
                }
            ]
        }

        validated = validate_single_admission_qa(
            payload,
            subject_id="100",
            hadm_id="10",
            admission_start="2020-01-01 08:00:00",
            admission_end="2020-01-02 10:00:00",
            valid_turn_ids={1, 2},
            expected_count=1,
            expected_adversarial_count=0,
        )
        self.assertEqual(
            validated["qas"][0]["answer"],
            "fever persisted despite antibiotics and blood pressure remained low overnight",
        )

    def test_validate_single_admission_qa_rejects_supporting_evidence_question_type(self) -> None:
        payload = {
            "qas": [
                {
                    "qa_id": "x",
                    "scope": "single_admission",
                    "question_type": "supporting_evidence",
                    "question": "Which finding supported the diagnosis?",
                    "answer": "positive blood cultures",
                    "evidence": {"admissions": ["10"], "turn_ids": [1, 2]},
                }
            ]
        }

        with self.assertRaises(QAValidationError):
            validate_single_admission_qa(
                payload,
                subject_id="100",
                hadm_id="10",
                admission_start="2020-01-01 08:00:00",
                admission_end="2020-01-02 10:00:00",
                valid_turn_ids={1, 2},
                expected_count=1,
                expected_adversarial_count=0,
            )

    def test_validate_single_admission_qa_rejects_wrong_adversarial_mix(self) -> None:
        payload = self._make_single_qa_payload("10", adversarial_count=0)

        with self.assertRaises(QAValidationError):
            validate_single_admission_qa(
                payload,
                subject_id="100",
                hadm_id="10",
                admission_start="2020-01-01 08:00:00",
                admission_end="2020-01-02 10:00:00",
                valid_turn_ids={1, 2},
                expected_count=3,
                expected_adversarial_count=1,
            )

    def test_validate_single_admission_qa_rejects_regular_question_type_in_adversarial_stream(self) -> None:
        payload = self._make_single_qa_payload("10")
        payload["qas"][-1]["question_type"] = "medical_reasoning"

        with self.assertRaises(QAValidationError):
            validate_single_admission_qa(
                payload,
                subject_id="100",
                hadm_id="10",
                admission_start="2020-01-01 08:00:00",
                admission_end="2020-01-02 10:00:00",
                valid_turn_ids={1, 2},
                expected_count=3,
                expected_adversarial_count=1,
            )

    def test_validate_single_admission_qa_does_not_coerce_mistyped_adversarial_answers(self) -> None:
        payload = {
            "qas": [
                {
                    "qa_id": "x",
                    "scope": "single_admission",
                    "question_type": "medical_reasoning",
                    "question": "Which unsupported detail explains the decision?",
                    "answer": CANONICAL_ADVERSARIAL_ANSWER,
                    "evidence": {"admissions": ["10"], "turn_ids": [1, 2]},
                }
            ]
        }

        with self.assertRaises(QAValidationError):
            validate_single_admission_qa(
                payload,
                subject_id="100",
                hadm_id="10",
                admission_start="2020-01-01 08:00:00",
                admission_end="2020-01-02 10:00:00",
                valid_turn_ids={1, 2},
                expected_count=1,
                expected_adversarial_count=0,
            )

    def test_validate_single_admission_qa_adds_multi_day_question_prefix(self) -> None:
        validated = validate_single_admission_qa(
            self._make_single_regular_payload("10", count=1),
            subject_id="100",
            hadm_id="10",
            admission_start="2020-01-01 08:00:00",
            admission_end="2020-01-02 10:00:00",
            valid_turn_ids={1, 2},
            expected_count=1,
            expected_adversarial_count=0,
        )

        self.assertEqual(
            validated["qas"][0]["question"],
            "During the hospitalization from 2020-01-01 to 2020-01-02, Why did the team make decision 1?",
        )

    def test_validate_single_admission_qa_uses_same_day_question_prefix(self) -> None:
        payload = self._make_single_regular_payload("10", count=1)
        payload["qas"][0]["question"] = "Why was the patient monitored closely?"

        validated = validate_single_admission_qa(
            payload,
            subject_id="100",
            hadm_id="10",
            admission_start="2020-01-01 08:00:00",
            admission_end="2020-01-01 18:00:00",
            valid_turn_ids={1, 2},
            expected_count=1,
            expected_adversarial_count=0,
        )

        self.assertEqual(
            validated["qas"][0]["question"],
            "During the hospitalization on 2020-01-01, Why was the patient monitored closely?",
        )

    def test_validate_single_admission_qa_normalizes_existing_leading_anchor(self) -> None:
        payload = self._make_single_regular_payload("10", count=1)
        payload["qas"][0]["question"] = "During this hospitalization, why was oxygen continued?"

        validated = validate_single_admission_qa(
            payload,
            subject_id="100",
            hadm_id="10",
            admission_start="2020-01-01 08:00:00",
            admission_end="2020-01-02 10:00:00",
            valid_turn_ids={1, 2},
            expected_count=1,
            expected_adversarial_count=0,
        )

        self.assertEqual(
            validated["qas"][0]["question"],
            "During the hospitalization from 2020-01-01 to 2020-01-02, why was oxygen continued?",
        )

    def test_validate_single_admission_qa_preserves_canonical_prefix(self) -> None:
        payload = self._make_single_regular_payload("10", count=1)
        payload["qas"][0]["question"] = (
            "During the hospitalization from 2020-01-01 to 2020-01-02, Why was telemetry continued?"
        )

        validated = validate_single_admission_qa(
            payload,
            subject_id="100",
            hadm_id="10",
            admission_start="2020-01-01 08:00:00",
            admission_end="2020-01-02 10:00:00",
            valid_turn_ids={1, 2},
            expected_count=1,
            expected_adversarial_count=0,
        )

        self.assertEqual(
            validated["qas"][0]["question"],
            "During the hospitalization from 2020-01-01 to 2020-01-02, Why was telemetry continued?",
        )

    def test_validate_cross_admission_qa_rejects_single_admission_evidence(self) -> None:
        payload = {
            "qas": [
                {
                    "qa_id": "x",
                    "scope": "cross_admission",
                    "question_type": "frequency_pattern",
                    "question": "Which issue recurred?",
                    "answer": "heart failure",
                    "evidence": {"admissions": ["10"]},
                }
            ]
        }

        with self.assertRaises(QAValidationError):
            validate_cross_admission_qa(
                payload,
                subject_id="100",
                ordered_hadm_ids=["10", "12"],
                expected_count=1,
                expected_adversarial_count=0,
            )

    def test_validate_cross_admission_qa_normalizes_admission_order(self) -> None:
        payload = {
            "qas": [
                {
                    "qa_id": "x",
                    "scope": "cross_admission",
                    "question_type": "frequency_pattern",
                    "question": "Which issue recurred?",
                    "answer": "heart failure",
                    "evidence": {"admissions": ["12", "10"]},
                }
            ]
        }

        validated = validate_cross_admission_qa(
            payload,
            subject_id="100",
            ordered_hadm_ids=["10", "12"],
            expected_count=1,
            expected_adversarial_count=0,
        )
        self.assertEqual(validated["qas"][0]["evidence"]["admissions"], ["10", "12"])

    def test_validate_cross_admission_qa_normalizes_adversarial_alias_to_canonical_answer(self) -> None:
        payload = {
            "qas": [
                {
                    "qa_id": "x",
                    "scope": "cross_admission",
                    "question_type": "adversarial",
                    "question": "Which issue recurred?",
                    "answer": "not answerable from the provided context",
                    "evidence": {"admissions": ["12", "10"]},
                }
            ]
        }

        validated = validate_cross_admission_qa(
            payload,
            subject_id="100",
            ordered_hadm_ids=["10", "12"],
            expected_count=1,
            expected_adversarial_count=1,
            allowed_question_types=("adversarial",),
        )
        self.assertEqual(validated["qas"][0]["answer"], CANONICAL_ADVERSARIAL_ANSWER)
        self.assertEqual(validated["qas"][0]["evidence"]["admissions"], ["10", "12"])

    def test_validate_cross_admission_qa_coerces_mistyped_adversarial_item_from_canonical_answer(self) -> None:
        payload = {
            "qas": [
                {
                    "qa_id": "x",
                    "scope": "cross_admission",
                    "question_type": "frequency_pattern",
                    "question": "Which unsupported issue recurred?",
                    "answer": CANONICAL_ADVERSARIAL_ANSWER,
                    "evidence": {"admissions": ["12", "10"]},
                }
            ]
        }

        validated = validate_cross_admission_qa(
            payload,
            subject_id="100",
            ordered_hadm_ids=["10", "12"],
            expected_count=1,
            expected_adversarial_count=1,
            allowed_question_types=("adversarial",),
            coerce_adversarial_question_type_from_answer=True,
        )
        self.assertEqual(validated["qas"][0]["question_type"], "adversarial")
        self.assertEqual(validated["qas"][0]["answer"], CANONICAL_ADVERSARIAL_ANSWER)
        self.assertEqual(validated["qas"][0]["evidence"]["admissions"], ["10", "12"])

    def test_validate_cross_admission_qa_coerces_mistyped_adversarial_item_from_alias_answer(self) -> None:
        payload = {
            "qas": [
                {
                    "qa_id": "x",
                    "scope": "cross_admission",
                    "question_type": "longitudinal_progression",
                    "question": "Which unsupported issue recurred?",
                    "answer": "not answerable from the provided context",
                    "evidence": {"admissions": ["12", "10"]},
                }
            ]
        }

        validated = validate_cross_admission_qa(
            payload,
            subject_id="100",
            ordered_hadm_ids=["10", "12"],
            expected_count=1,
            expected_adversarial_count=1,
            allowed_question_types=("adversarial",),
            coerce_adversarial_question_type_from_answer=True,
        )
        self.assertEqual(validated["qas"][0]["question_type"], "adversarial")
        self.assertEqual(validated["qas"][0]["answer"], CANONICAL_ADVERSARIAL_ANSWER)

    def test_validate_cross_admission_qa_coercion_mode_rejects_mistyped_non_adversarial_answer(self) -> None:
        payload = {
            "qas": [
                {
                    "qa_id": "x",
                    "scope": "cross_admission",
                    "question_type": "frequency_pattern",
                    "question": "Which unsupported issue recurred?",
                    "answer": "heart failure",
                    "evidence": {"admissions": ["12", "10"]},
                }
            ]
        }

        with self.assertRaises(QAValidationError):
            validate_cross_admission_qa(
                payload,
                subject_id="100",
                ordered_hadm_ids=["10", "12"],
                expected_count=1,
                expected_adversarial_count=1,
                allowed_question_types=("adversarial",),
                coerce_adversarial_question_type_from_answer=True,
            )

    def test_validate_cross_admission_qa_regular_stream_does_not_coerce_mistyped_adversarial_answer(self) -> None:
        payload = {
            "qas": [
                {
                    "qa_id": "x",
                    "scope": "cross_admission",
                    "question_type": "frequency_pattern",
                    "question": "Which supported issue recurred?",
                    "answer": CANONICAL_ADVERSARIAL_ANSWER,
                    "evidence": {"admissions": ["12", "10"]},
                }
            ]
        }

        with self.assertRaises(QAValidationError):
            validate_cross_admission_qa(
                payload,
                subject_id="100",
                ordered_hadm_ids=["10", "12"],
                expected_count=1,
                expected_adversarial_count=0,
                allowed_question_types=("frequency_pattern",),
            )

    def test_validate_cross_admission_qa_normalizes_time_range_aliases(self) -> None:
        payload = {
            "qas": [
                {
                    "qa_id": "x",
                    "scope": "cross_admission",
                    "question_type": "frequency_pattern",
                    "question": "Which issue recurred over time?",
                    "answer": "heart failure",
                    "evidence": {
                        "admissions": [
                            "2020-03-01 08:00:00 to 2020-03-02 09:00:00",
                            "2020-01-01 08:00:00 to 2020-01-02 10:00:00",
                        ]
                    },
                }
            ]
        }

        validated = validate_cross_admission_qa(
            payload,
            subject_id="100",
            ordered_hadm_ids=["10", "12"],
            expected_count=1,
            expected_adversarial_count=0,
            admission_aliases={
                "2020-01-01 08:00:00 to 2020-01-02 10:00:00": "10",
                "2020-03-01 08:00:00 to 2020-03-02 09:00:00": "12",
            },
        )
        self.assertEqual(validated["qas"][0]["evidence"]["admissions"], ["10", "12"])

    def test_validate_cross_admission_qa_dedupes_duplicate_evidence_admissions(self) -> None:
        payload = {
            "qas": [
                {
                    "qa_id": "x",
                    "scope": "cross_admission",
                    "question_type": "adversarial",
                    "question": "Which issue recurred over time?",
                    "answer": CANONICAL_ADVERSARIAL_ANSWER,
                    "evidence": {"admissions": ["12", "10", "12"]},
                }
            ]
        }

        validated = validate_cross_admission_qa(
            payload,
            subject_id="100",
            ordered_hadm_ids=["10", "12"],
            expected_count=1,
            expected_adversarial_count=1,
        )
        self.assertEqual(validated["qas"][0]["evidence"]["admissions"], ["10", "12"])

    def test_validate_cross_admission_qa_rejects_wrong_adversarial_mix(self) -> None:
        payload = self._make_cross_qa_payload(["10", "12"], count=6, adversarial_count=1)

        with self.assertRaises(QAValidationError):
            validate_cross_admission_qa(
                payload,
                subject_id="100",
                ordered_hadm_ids=["10", "12"],
                expected_count=6,
                expected_adversarial_count=2,
            )

    def test_render_qa_prompts_use_expected_context_and_counts(self) -> None:
        conversation = {
            "subject_id": "100",
            "hadm_id": "10",
            "admission_start": "2020-01-01 08:00:00",
            "admission_end": "2020-01-02 10:00:00",
            "conversation_lines": [
                {"turn_number": 1, "time": "2020-01-01 08:05:00", "speaker": "Doctor", "text": "You came in with pneumonia."},
                {"turn_number": 2, "time": "2020-01-01 08:06:00", "speaker": "Patient", "text": "I felt short of breath."},
            ],
        }
        summary_contexts = [
            {
                "admission_id_for_evidence_only": "10",
                "admission_start": "2020-01-01 08:00:00",
                "admission_end": "2020-01-02 10:00:00",
                "summary_paragraph": "Pneumonia improved.",
                "problems": ["Pneumonia"],
            },
            {
                "admission_id_for_evidence_only": "12",
                "admission_start": "2020-03-01 08:00:00",
                "admission_end": "2020-03-02 09:00:00",
                "summary_paragraph": "Bronchoscopy improved symptoms.",
                "problems": ["Respiratory failure"],
            },
        ]
        rendered_single_regular = render_single_admission_regular_qa_prompt(
            conversation,
            question_count=2,
        )
        rendered_single_adversarial = render_single_admission_adversarial_qa_prompt(
            conversation,
            summary_contexts,
            question_count=1,
        )
        rendered_cross_regular = render_cross_admission_regular_qa_prompt(
            summary_contexts,
            question_count=4,
        )
        rendered_cross_adversarial = render_cross_admission_adversarial_qa_prompt(
            summary_contexts,
            question_count=2,
        )

        self.assertEqual(rendered_single_regular.question_count, 2)
        self.assertIn('"conversation_lines"', rendered_single_regular.context_json)
        self.assertIn("Generate exactly 2 hard answerable short-answer question-answer pairs", rendered_single_regular.user_message)
        self.assertIn("- medical_reasoning", rendered_single_regular.user_message)
        self.assertIn("- care_plan_rationale", rendered_single_regular.user_message)
        self.assertNotIn("- adversarial", rendered_single_regular.user_message)
        self.assertNotIn("question_class", rendered_single_regular.user_message)
        self.assertNotIn("multiple_choice", rendered_single_regular.user_message)
        self.assertNotIn('"options": [', rendered_single_regular.user_message)
        self.assertIn("never more than 10 words", rendered_single_regular.user_message)
        self.assertIn(
            "When possible, form the short answer using exact wording from the conversation rather than paraphrasing.",
            rendered_single_regular.user_message,
        )
        self.assertIn(
            "If exact wording is awkward, use only light normalization while keeping the wording as close as possible to the conversation.",
            rendered_single_regular.user_message,
        )
        self.assertIn("Do not use second-person wording like 'you' or 'your' in the question.", rendered_single_regular.user_message)
        self.assertIn(
            "Use third-person phrasing such as 'the patient', 'the patient's symptoms', or 'the doctor' when needed.",
            rendered_single_regular.user_message,
        )
        self.assertIn("Do not mention raw identifiers in the question.", rendered_single_regular.user_message)
        self.assertIn(
            "During the hospitalization from YYYY-MM-DD to YYYY-MM-DD, ...",
            rendered_single_regular.user_message,
        )
        self.assertIn(
            "During the hospitalization on YYYY-MM-DD, ...",
            rendered_single_regular.user_message,
        )

        self.assertEqual(rendered_single_adversarial.question_count, 1)
        self.assertIn('"target_admission_conversation"', rendered_single_adversarial.context_json)
        self.assertIn('"patient_admission_summaries"', rendered_single_adversarial.context_json)
        self.assertIn("Generate exactly 1 adversarial short-answer question-answer pair for this admission.", rendered_single_adversarial.user_message)
        self.assertIn("- Generate exactly 1 questions total.", rendered_single_adversarial.user_message)
        self.assertIn(CANONICAL_ADVERSARIAL_ANSWER, rendered_single_adversarial.user_message)
        self.assertIn("The requested fact must not appear anywhere else", rendered_single_adversarial.user_message)
        self.assertNotIn("question_class", rendered_single_adversarial.user_message)
        self.assertNotIn("LOCOMO", rendered_single_adversarial.system_message)
        self.assertNotIn("LOCOMO", rendered_single_adversarial.user_message)

        self.assertEqual(rendered_cross_regular.question_count, 4)
        self.assertIn('"summary_paragraph"', rendered_cross_regular.context_json)
        self.assertIn('"admission_id_for_evidence_only"', rendered_cross_regular.context_json)
        self.assertNotIn('"conversation_lines"', rendered_cross_regular.context_json)
        self.assertIn("Generate exactly 4 hard answerable cross-admission short-answer question-answer pairs", rendered_cross_regular.user_message)
        self.assertIn("- longitudinal_progression", rendered_cross_regular.user_message)
        self.assertIn("- cross_admission_comparison", rendered_cross_regular.user_message)
        self.assertIn("- frequency_pattern", rendered_cross_regular.user_message)
        self.assertNotIn("- adversarial", rendered_cross_regular.user_message)
        self.assertNotIn("question_class", rendered_cross_regular.user_message)
        self.assertNotIn("multiple_choice", rendered_cross_regular.user_message)
        self.assertIn("never more than 10 words", rendered_cross_regular.user_message)
        self.assertIn(
            "When possible, form the short answer using exact wording from the summaries rather than paraphrasing.",
            rendered_cross_regular.user_message,
        )
        self.assertIn(
            "If exact wording is awkward, use only light normalization while keeping the wording as close as possible to the summaries.",
            rendered_cross_regular.user_message,
        )
        self.assertIn("should not mention raw identifiers", rendered_cross_regular.user_message)
        self.assertIn(
            "Evidence must cite at least 2 unique admissions for every item.",
            rendered_cross_regular.user_message,
        )
        self.assertIn(
            "evidence.admissions must use only the exact admission_id_for_evidence_only values already present in the provided context.",
            rendered_cross_regular.user_message,
        )
        self.assertIn(
            "Do not invent, alter, extend, or reformat admission ids.",
            rendered_cross_regular.user_message,
        )
        self.assertNotIn(
            "During the hospitalization from YYYY-MM-DD to YYYY-MM-DD, ...",
            rendered_cross_regular.user_message,
        )

        self.assertEqual(rendered_cross_adversarial.question_count, 2)
        self.assertIn(
            "Generate exactly 2 adversarial cross-admission short-answer question-answer pairs from the chronological admission summaries below.",
            rendered_cross_adversarial.user_message,
        )
        self.assertIn("- Generate exactly 2 questions total.", rendered_cross_adversarial.user_message)
        self.assertIn(
            "Evidence must cite at least 2 unique admissions for every item.",
            rendered_cross_adversarial.user_message,
        )
        self.assertIn(
            "evidence.admissions must use only the exact admission_id_for_evidence_only values already present in the provided context.",
            rendered_cross_adversarial.user_message,
        )
        self.assertIn(
            "Do not invent, alter, extend, or reformat admission ids.",
            rendered_cross_adversarial.user_message,
        )
        self.assertIn(CANONICAL_ADVERSARIAL_ANSWER, rendered_cross_adversarial.user_message)
        self.assertIn("unsupported by the full provided context", rendered_cross_adversarial.user_message)
        self.assertNotIn("question_class", rendered_cross_adversarial.user_message)
        self.assertNotIn("LOCOMO", rendered_cross_adversarial.system_message)
        self.assertNotIn("LOCOMO", rendered_cross_adversarial.user_message)

    def test_generate_patient_sample_respects_filters_and_replaces_stale_output(self) -> None:
        config = self._config()
        pipeline = BenchmarkPipeline(config)
        self.addCleanup(pipeline.close)
        pipeline.llm_client = FakeLLMClient(
            [
                {
                    "parsed_output": {
                        "conversation_lines": [
                            {"turn_number": 1, "time": "2020-01-01 08:05:00", "speaker": "Doctor", "text": "You came in with pneumonia."},
                            {"turn_number": 2, "time": "2020-01-01 08:06:00", "speaker": "Patient", "text": "I felt short of breath."},
                        ],
                        "summary": {
                            "admission_start": "2020-01-01 08:00:00",
                            "admission_end": "2020-01-02 10:00:00",
                            "summary_paragraph": "Pneumonia improved during the admission.",
                            "problems": ["Pneumonia"],
                        },
                    },
                    "usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
                },
                {
                    "parsed_output": {
                        "conversation_lines": [
                            {"turn_number": 1, "time": "2020-03-01 08:10:00", "speaker": "Doctor", "text": "Your breathing improved after bronchoscopy."},
                            {"turn_number": 2, "time": "2020-03-01 08:12:00", "speaker": "Patient", "text": "I feel better now."},
                        ],
                        "summary": {
                            "admission_start": "2020-03-01 08:00:00",
                            "admission_end": "2020-03-02 09:00:00",
                            "summary_paragraph": "Respiratory symptoms improved after bronchoscopy.",
                            "problems": ["Respiratory failure"],
                        },
                    },
                    "usage": {"input_tokens": 120, "output_tokens": 60, "total_tokens": 180},
                },
            ]
        )
        stale_dir = self.output_dir / "100"
        stale_dir.mkdir(parents=True, exist_ok=True)
        (stale_dir / "stale.txt").write_text("old", encoding="utf-8")

        with patch("health_benchmark.scripts.pipeline._import_tiktoken", return_value=FakeTikToken()):
            summary = pipeline.generate_patient_sample(subject_id=100, max_admissions=2)

        self.assertEqual(summary["eligible_admissions"], 2)
        self.assertEqual(summary["processed_admissions"], 2)
        self.assertEqual(summary["processed_hadm_ids"], ["10", "12"])
        self.assertEqual(summary["llm_call_stats"]["total_input_tokens"], 220)
        self.assertEqual(summary["llm_call_stats"]["mean_output_tokens"], 55.0)
        self.assertEqual(summary["conversation_stats"]["total_turns"], 4)
        self.assertEqual(summary["conversation_stats"]["tokenizer"], "o200k_base")
        self.assertFalse((self.output_dir / "100" / "stale.txt").exists())

        combined = json.loads((self.output_dir / "100" / "combined_conversation.json").read_text(encoding="utf-8"))
        self.assertEqual(combined["processed_hadm_ids"], ["10", "12"])
        self.assertTrue((self.output_dir / "100" / "10" / "formed_packet.json").exists())
        self.assertTrue((self.output_dir / "100" / "10" / "prompt_record.json").exists())
        self.assertTrue((self.output_dir / "100" / "10" / "conversation.json").exists())
        self.assertTrue((self.output_dir / "100" / "10" / "summary.json").exists())
        prompt_record = json.loads((self.output_dir / "100" / "10" / "prompt_record.json").read_text(encoding="utf-8"))
        self.assertEqual(prompt_record["recommended_turn_range"], [40, 60])
        self.assertIn("Hospital course: pneumonia treated with antibiotics.", prompt_record["storyline_prompt_block"])
        self.assertIn("- Pneumonia", prompt_record["problem_prompt_block"])
        self.assertIn("Radiology:", prompt_record["supporting_context_block"])

    def test_generate_patient_sample_supports_two_subjects_sequentially_in_one_pipeline(self) -> None:
        config = self._config()
        pipeline = BenchmarkPipeline(config)
        self.addCleanup(pipeline.close)
        pipeline.llm_client = FakeLLMClient(
            [
                {
                    "parsed_output": {
                        "conversation_lines": [
                            {"turn_number": 1, "time": "2020-01-01 08:05:00", "speaker": "Doctor", "text": "You came in with pneumonia."},
                            {"turn_number": 2, "time": "2020-01-01 08:06:00", "speaker": "Patient", "text": "I felt short of breath."},
                        ],
                        "summary": {
                            "admission_start": "2020-01-01 08:00:00",
                            "admission_end": "2020-01-02 10:00:00",
                            "summary_paragraph": "Pneumonia improved during the admission.",
                            "problems": ["Pneumonia"],
                        },
                    },
                    "usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
                },
                {
                    "parsed_output": {
                        "conversation_lines": [
                            {"turn_number": 1, "time": "2020-03-01 08:10:00", "speaker": "Doctor", "text": "Your breathing improved after bronchoscopy."},
                            {"turn_number": 2, "time": "2020-03-01 08:12:00", "speaker": "Patient", "text": "I feel better now."},
                        ],
                        "summary": {
                            "admission_start": "2020-03-01 08:00:00",
                            "admission_end": "2020-03-02 09:00:00",
                            "summary_paragraph": "Respiratory symptoms improved after bronchoscopy.",
                            "problems": ["Respiratory failure"],
                        },
                    },
                    "usage": {"input_tokens": 120, "output_tokens": 60, "total_tokens": 180},
                },
                {
                    "parsed_output": {
                        "conversation_lines": [
                            {"turn_number": 1, "time": "2020-01-01 08:05:00", "speaker": "Doctor", "text": "The first stay focused on discharge planning."},
                            {"turn_number": 2, "time": "2020-01-01 08:06:00", "speaker": "Patient", "text": "I understand the plan."},
                        ],
                        "summary": {
                            "admission_start": "2020-01-01 08:00:00",
                            "admission_end": "2020-01-02 08:00:00",
                            "summary_paragraph": "The first admission ended with discharge planning.",
                            "problems": ["Follow-up planning"],
                        },
                    },
                    "usage": {"input_tokens": 80, "output_tokens": 40, "total_tokens": 120},
                },
                {
                    "parsed_output": {
                        "conversation_lines": [
                            {"turn_number": 1, "time": "2020-02-01 08:05:00", "speaker": "Doctor", "text": "The second stay reviewed recovery."},
                            {"turn_number": 2, "time": "2020-02-01 08:06:00", "speaker": "Patient", "text": "Recovery is going well."},
                        ],
                        "summary": {
                            "admission_start": "2020-02-01 08:00:00",
                            "admission_end": "2020-02-02 08:00:00",
                            "summary_paragraph": "The second admission reviewed interval recovery.",
                            "problems": ["Recovery"],
                        },
                    },
                    "usage": {"input_tokens": 90, "output_tokens": 45, "total_tokens": 135},
                },
            ]
        )

        with patch("health_benchmark.scripts.pipeline._import_tiktoken", return_value=FakeTikToken()):
            first_summary = pipeline.generate_patient_sample(subject_id=100, max_admissions=2)
            second_summary = pipeline.generate_patient_sample(subject_id=200, max_admissions=2)

        self.assertEqual(first_summary["eligible_admissions"], 2)
        self.assertEqual(second_summary["eligible_admissions"], 2)
        self.assertEqual(second_summary["processed_admissions"], 2)
        self.assertEqual(second_summary["processed_hadm_ids"], ["20", "21"])
        combined = json.loads((self.output_dir / "200" / "combined_conversation.json").read_text(encoding="utf-8"))
        self.assertEqual(combined["processed_hadm_ids"], ["20", "21"])
        self.assertTrue((self.output_dir / "200" / "20" / "conversation.json").exists())
        self.assertTrue((self.output_dir / "200" / "21" / "summary.json").exists())

    def test_build_default_config_supports_legacy_openai_block_and_vllm_defaults(self) -> None:
        project_dir = Path(__file__).resolve().parents[1] / "health_benchmark"

        with patch(
            "health_benchmark.scripts.config._load_yaml_config",
            return_value={"openai": {"model": "legacy-model", "max_output_tokens": 1234}},
        ):
            legacy_config = build_default_config(project_dir)

        self.assertEqual(legacy_config.llm.provider, "openai")
        self.assertEqual(legacy_config.llm.model, "legacy-model")
        self.assertEqual(legacy_config.llm.max_output_tokens, 1234)
        self.assertIsNone(legacy_config.llm.base_url)
        self.assertIs(legacy_config.openai, legacy_config.llm)

        with patch(
            "health_benchmark.scripts.config._load_yaml_config",
            return_value={"llm": {"provider": "vllm", "model": "Qwen/Qwen3.5-27B"}},
        ):
            vllm_config = build_default_config(project_dir)

        self.assertEqual(vllm_config.llm.provider, "vllm")
        self.assertEqual(vllm_config.llm.base_url, DEFAULT_VLLM_BASE_URL)

    def test_build_llm_client_selects_provider_implementations(self) -> None:
        config = self._config()
        config.llm.provider = "openai"
        self.assertIsInstance(build_llm_client(config), OpenAILLMClient)

        config.llm.provider = "vllm"
        config.llm.base_url = None
        self.assertIsInstance(build_llm_client(config), VLLMLLMClient)

    def test_openai_client_uses_native_structured_output_api(self) -> None:
        config = self._config()
        config.llm.provider = "openai"
        factory = FakeOpenAIClientFactory(parse_payload=self._make_single_qa_payload("10"))

        with patch.dict("os.environ", {config.llm.api_key_env: "test-key"}, clear=False), patch(
            "health_benchmark.scripts.llm_client._import_openai",
            return_value=factory,
        ):
            client = build_llm_client(config)
            result = client.generate_structured_response("system", "user", SingleAdmissionQAFile)

        self.assertEqual(result.usage["input_tokens"], 13)
        self.assertEqual(result.usage["output_tokens"], 5)
        self.assertEqual(len(factory.chat_requests), 0)
        request = factory.parse_requests[0]
        self.assertEqual(request["model"], config.llm.model)
        self.assertEqual(request["instructions"], "system")
        self.assertEqual(request["input"], "user")
        self.assertEqual(request["max_output_tokens"], config.llm.max_output_tokens)
        self.assertIs(request["text_format"], SingleAdmissionQAFile)

    def test_vllm_client_omits_response_format_keeps_top_k_and_disable_thinking(self) -> None:
        config = self._config()
        config.llm.provider = "vllm"
        config.llm.base_url = None
        config.llm.model = "Qwen/Qwen3.5-27B"
        config.vllm.top_k = 20
        config.vllm.enable_thinking = False
        factory = FakeOpenAIClientFactory(chat_content=json.dumps(self._make_single_qa_payload("10")))

        with patch("health_benchmark.scripts.llm_client._import_openai", return_value=factory):
            client = build_llm_client(config)
            result = client.generate_structured_response("system", "user", SingleAdmissionQAFile)

        self.assertEqual(result.usage["input_tokens"], 11)
        self.assertEqual(result.usage["output_tokens"], 7)
        self.assertEqual(factory.init_kwargs[0]["base_url"], DEFAULT_VLLM_BASE_URL)
        request = factory.chat_requests[0]
        self.assertEqual(request["model"], "Qwen/Qwen3.5-27B")
        self.assertNotIn("response_format", request)
        self.assertEqual(request["extra_body"]["top_k"], 20)
        self.assertFalse(request["extra_body"]["chat_template_kwargs"]["enable_thinking"])

    def test_vllm_client_accepts_markdown_fenced_json_content(self) -> None:
        config = self._config()
        config.llm.provider = "vllm"
        config.llm.base_url = None
        payload = self._make_single_qa_payload("10")
        factory = FakeOpenAIClientFactory(chat_content=f"```json\n{json.dumps(payload)}\n```")

        with patch("health_benchmark.scripts.llm_client._import_openai", return_value=factory):
            client = build_llm_client(config)
            result = client.generate_structured_response("system", "user", SingleAdmissionQAFile)

        self.assertEqual(result.parsed_output, payload)
        request = factory.chat_requests[0]
        self.assertNotIn("response_format", request)

    def test_vllm_client_accepts_prefaced_markdown_fenced_json_content(self) -> None:
        config = self._config()
        config.llm.provider = "vllm"
        config.llm.base_url = None
        payload = self._make_single_qa_payload("10")
        content = f"Based on the guidelines, here is the JSON:\n```json\n{json.dumps(payload)}\n```"
        factory = FakeOpenAIClientFactory(chat_content=content)

        with patch("health_benchmark.scripts.llm_client._import_openai", return_value=factory):
            client = build_llm_client(config)
            result = client.generate_structured_response("system", "user", SingleAdmissionQAFile)

        self.assertEqual(result.parsed_output, payload)
        request = factory.chat_requests[0]
        self.assertNotIn("response_format", request)

    def test_vllm_client_validates_schema_locally_without_response_format(self) -> None:
        config = self._config()
        config.llm.provider = "vllm"
        config.llm.base_url = None
        factory = FakeOpenAIClientFactory(chat_content=json.dumps({"unexpected": True}))

        with patch("health_benchmark.scripts.llm_client._import_openai", return_value=factory):
            client = build_llm_client(config)
            with self.assertRaisesRegex(RuntimeError, "vLLM response did not match expected schema"):
                client.generate_structured_response("system", "user", SingleAdmissionQAFile)

        request = factory.chat_requests[0]
        self.assertNotIn("response_format", request)

    def test_generate_patient_qa_requires_patient_directory(self) -> None:
        pipeline = BenchmarkPipeline(self._config())
        self.addCleanup(pipeline.close)

        with self.assertRaises(ValueError):
            pipeline.generate_patient_qa(subject_id=999)

    def test_generate_patient_qa_requires_all_conversation_and_summary_artifacts(self) -> None:
        patient_root = self.output_dir / "444"
        patient_root.mkdir(parents=True, exist_ok=True)
        admission_dir = patient_root / "10"
        admission_dir.mkdir(parents=True, exist_ok=True)
        self._write_json_file(
            admission_dir / "conversation.json",
            {
                "subject_id": "444",
                "hadm_id": "10",
                "admission_start": "2020-01-01 08:00:00",
                "admission_end": "2020-01-01 09:00:00",
                "conversation_lines": [
                    {"turn_number": 1, "time": "2020-01-01 08:00:00", "speaker": "Doctor", "text": "Hello."}
                ],
            },
        )
        pipeline = BenchmarkPipeline(self._config())
        self.addCleanup(pipeline.close)

        with self.assertRaises(ValueError):
            pipeline.generate_patient_qa(subject_id=444)

    def test_generate_patient_qa_requires_two_admissions_for_cross_generation(self) -> None:
        self._write_patient_artifacts(
            555,
            [
                {
                    "hadm_id": "10",
                    "admission_start": "2020-01-01 08:00:00",
                    "admission_end": "2020-01-01 10:00:00",
                    "summary_paragraph": "Single admission only.",
                    "problems": ["Pneumonia"],
                }
            ],
        )
        pipeline = BenchmarkPipeline(self._config())
        self.addCleanup(pipeline.close)

        with self.assertRaises(ValueError):
            pipeline.generate_patient_qa(subject_id=555)

    def test_generate_patient_qa_uses_default_cross_admission_count(self) -> None:
        self._write_patient_artifacts(
            556,
            [
                {
                    "hadm_id": "10",
                    "admission_start": "2020-01-01 08:00:00",
                    "admission_end": "2020-01-02 10:00:00",
                    "summary_paragraph": "Earlier admission summary.",
                    "problems": ["Pneumonia"],
                },
                {
                    "hadm_id": "12",
                    "admission_start": "2020-03-01 08:00:00",
                    "admission_end": "2020-03-02 09:00:00",
                    "summary_paragraph": "Later admission summary.",
                    "problems": ["Respiratory failure"],
                },
            ],
        )
        pipeline = BenchmarkPipeline(self._config())
        self.addCleanup(pipeline.close)
        pipeline.llm_client = FakeLLMClient(
            [
                {
                    "parsed_output": self._make_single_regular_payload("10"),
                    "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
                },
                {
                    "parsed_output": self._make_single_adversarial_payload("10"),
                    "usage": {"input_tokens": 11, "output_tokens": 21, "total_tokens": 32},
                },
                {
                    "parsed_output": self._make_single_regular_payload("12"),
                    "usage": {"input_tokens": 12, "output_tokens": 22, "total_tokens": 34},
                },
                {
                    "parsed_output": self._make_single_adversarial_payload("12"),
                    "usage": {"input_tokens": 13, "output_tokens": 23, "total_tokens": 36},
                },
                {
                    "parsed_output": self._make_cross_regular_payload(["10", "12"]),
                    "usage": {"input_tokens": 14, "output_tokens": 24, "total_tokens": 38},
                },
                {
                    "parsed_output": self._make_cross_adversarial_payload(["10", "12"]),
                    "usage": {"input_tokens": 15, "output_tokens": 25, "total_tokens": 40},
                },
            ]
        )

        summary = pipeline.generate_patient_qa(subject_id=556)

        self.assertEqual(summary["single_admission_qa_count"], 3)
        self.assertEqual(summary["cross_admission_qa_count"], 6)
        self.assertEqual(summary["cross_adversarial_count"], 2)
        self.assertEqual(summary["admission_count"], 2)
        self.assertEqual(summary["total_qas"], 12)
        cross_qas = json.loads((self.output_dir / "556" / "cross_admission_qa.json").read_text(encoding="utf-8"))
        self.assertEqual(len(cross_qas["qas"]), 6)

    def test_generate_patient_qa_coerces_cross_adversarial_question_types_from_abstention_answers(self) -> None:
        self._write_patient_artifacts(
            557,
            [
                {
                    "hadm_id": "10",
                    "admission_start": "2020-01-01 08:00:00",
                    "admission_end": "2020-01-02 10:00:00",
                    "summary_paragraph": "Earlier admission summary.",
                    "problems": ["Pneumonia"],
                },
                {
                    "hadm_id": "12",
                    "admission_start": "2020-03-01 08:00:00",
                    "admission_end": "2020-03-02 09:00:00",
                    "summary_paragraph": "Later admission summary.",
                    "problems": ["Respiratory failure"],
                },
            ],
        )
        cross_adversarial_payload = self._make_cross_adversarial_payload(["10", "12"])
        cross_adversarial_payload["qas"][0]["question_type"] = "frequency_pattern"
        cross_adversarial_payload["qas"][1]["question_type"] = "cross_admission_comparison"
        cross_adversarial_payload["qas"][1]["answer"] = "not answerable from the provided context"

        pipeline = BenchmarkPipeline(self._config())
        self.addCleanup(pipeline.close)
        pipeline.llm_client = FakeLLMClient(
            [
                {
                    "parsed_output": self._make_single_regular_payload("10"),
                    "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
                },
                {
                    "parsed_output": self._make_single_adversarial_payload("10"),
                    "usage": {"input_tokens": 11, "output_tokens": 21, "total_tokens": 32},
                },
                {
                    "parsed_output": self._make_single_regular_payload("12"),
                    "usage": {"input_tokens": 12, "output_tokens": 22, "total_tokens": 34},
                },
                {
                    "parsed_output": self._make_single_adversarial_payload("12"),
                    "usage": {"input_tokens": 13, "output_tokens": 23, "total_tokens": 36},
                },
                {
                    "parsed_output": self._make_cross_regular_payload(["10", "12"]),
                    "usage": {"input_tokens": 14, "output_tokens": 24, "total_tokens": 38},
                },
                {
                    "parsed_output": cross_adversarial_payload,
                    "usage": {"input_tokens": 15, "output_tokens": 25, "total_tokens": 40},
                },
            ]
        )

        summary = pipeline.generate_patient_qa(subject_id=557)

        self.assertEqual(summary["cross_admission_qa_count"], 6)
        self.assertEqual(summary["cross_adversarial_count"], 2)
        self.assertEqual(summary["total_qas"], 12)
        cross_qas = json.loads((self.output_dir / "557" / "cross_admission_qa.json").read_text(encoding="utf-8"))
        benchmark_qas = json.loads((self.output_dir / "557" / "benchmark_qa.json").read_text(encoding="utf-8"))
        self.assertEqual(len(cross_qas["qas"]), 6)
        self.assertEqual(len(benchmark_qas["qas"]), 12)
        self.assertEqual(
            [qa["question_type"] for qa in cross_qas["qas"][-2:]],
            ["adversarial", "adversarial"],
        )
        self.assertEqual(
            [qa["answer"] for qa in cross_qas["qas"][-2:]],
            [CANONICAL_ADVERSARIAL_ANSWER, CANONICAL_ADVERSARIAL_ANSWER],
        )
        self.assertEqual(
            sum(1 for qa in cross_qas["qas"] if qa["question_type"] == "adversarial"),
            2,
        )

    def test_generate_patient_qa_writes_and_replaces_only_qa_outputs(self) -> None:
        patient_root = self._write_patient_artifacts(
            777,
            [
                {
                    "hadm_id": "12",
                    "admission_start": "2020-03-01 08:00:00",
                    "admission_end": "2020-03-02 09:00:00",
                    "summary_paragraph": "Later admission summary.",
                    "problems": ["Respiratory failure"],
                },
                {
                    "hadm_id": "10",
                    "admission_start": "2020-01-01 08:00:00",
                    "admission_end": "2020-01-02 10:00:00",
                    "summary_paragraph": "Earlier admission summary.",
                    "problems": ["Pneumonia"],
                },
            ],
        )
        (patient_root / "10" / "qa.json").write_text('{"qas":[{"qa_id":"old"}]}\n', encoding="utf-8")
        (patient_root / "cross_admission_qa.json").write_text('{"qas":[{"qa_id":"old_cross"}]}\n', encoding="utf-8")
        (patient_root / "benchmark_qa.json").write_text('{"qas":[{"qa_id":"old_benchmark"}]}\n', encoding="utf-8")
        prompt_record_before = (patient_root / "10" / "prompt_record.json").read_text(encoding="utf-8")

        pipeline = BenchmarkPipeline(self._config())
        self.addCleanup(pipeline.close)
        pipeline.llm_client = FakeLLMClient(
            [
                {
                    "parsed_output": self._make_single_regular_payload("10"),
                    "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
                },
                {
                    "parsed_output": self._make_single_adversarial_payload("10"),
                    "usage": {"input_tokens": 11, "output_tokens": 21, "total_tokens": 32},
                },
                {
                    "parsed_output": self._make_single_regular_payload("12"),
                    "usage": {"input_tokens": 12, "output_tokens": 22, "total_tokens": 34},
                },
                {
                    "parsed_output": self._make_single_adversarial_payload("12"),
                    "usage": {"input_tokens": 13, "output_tokens": 23, "total_tokens": 36},
                },
                {
                    "parsed_output": self._make_cross_regular_payload(["10", "12"]),
                    "usage": {"input_tokens": 14, "output_tokens": 24, "total_tokens": 38},
                },
                {
                    "parsed_output": self._make_cross_adversarial_payload(["10", "12"]),
                    "usage": {"input_tokens": 15, "output_tokens": 25, "total_tokens": 40},
                },
            ]
        )

        summary = pipeline.generate_patient_qa(subject_id=777)

        self.assertEqual(summary["processed_admissions"], 2)
        self.assertEqual(summary["single_admission_qa_count"], 3)
        self.assertEqual(summary["cross_admission_qa_count"], 6)
        self.assertEqual(summary["cross_adversarial_count"], 2)
        self.assertEqual(summary["admission_count"], 2)
        self.assertEqual(summary["total_qas"], 12)
        admission_10_qas = json.loads((patient_root / "10" / "qa.json").read_text(encoding="utf-8"))
        admission_12_qas = json.loads((patient_root / "12" / "qa.json").read_text(encoding="utf-8"))
        cross_qas = json.loads((patient_root / "cross_admission_qa.json").read_text(encoding="utf-8"))
        benchmark_qas = json.loads((patient_root / "benchmark_qa.json").read_text(encoding="utf-8"))
        self.assertEqual(admission_10_qas["qas"][0]["qa_id"], "777_10_q01")
        self.assertEqual(admission_10_qas["qas"][0]["evidence"]["turn_ids"], [1, 2])
        self.assertEqual(len(admission_10_qas["qas"]), 3)
        self.assertEqual(
            sum(1 for qa in admission_10_qas["qas"] if qa["question_type"] == "adversarial"),
            1,
        )
        self.assertEqual(admission_10_qas["qas"][-1]["answer"], CANONICAL_ADVERSARIAL_ANSWER)
        self.assertTrue(
            admission_10_qas["qas"][0]["question"].startswith(
                "During the hospitalization from 2020-01-01 to 2020-01-02, "
            )
        )
        self.assertEqual(admission_12_qas["qas"][0]["qa_id"], "777_12_q01")
        self.assertTrue(
            admission_12_qas["qas"][0]["question"].startswith(
                "During the hospitalization from 2020-03-01 to 2020-03-02, "
            )
        )
        self.assertEqual(len(cross_qas["qas"]), 6)
        self.assertEqual(cross_qas["qas"][0]["qa_id"], "777_cross_q01")
        self.assertEqual(cross_qas["qas"][-1]["question_type"], "adversarial")
        self.assertEqual(len(benchmark_qas["qas"]), 12)
        expected_order = [
            "777_cross_q06",
            "777_cross_q03",
            "777_10_q02",
            "777_10_q01",
            "777_cross_q04",
            "777_10_q03",
            "777_cross_q01",
            "777_12_q02",
            "777_12_q03",
            "777_cross_q05",
            "777_cross_q02",
            "777_12_q01",
        ]
        self.assertEqual([qa["qa_id"] for qa in benchmark_qas["qas"]], expected_order)
        self.assertEqual(
            sum(1 for qa in cross_qas["qas"] if qa["question_type"] == "adversarial"),
            2,
        )
        self.assertEqual(
            pipeline.llm_client.structured_calls,
            [
                {"schema": "SingleAdmissionQAFile", "max_output_tokens": None},
                {"schema": "SingleAdmissionQAFile", "max_output_tokens": None},
                {"schema": "SingleAdmissionQAFile", "max_output_tokens": None},
                {"schema": "SingleAdmissionQAFile", "max_output_tokens": None},
                {"schema": "CrossAdmissionQAFile", "max_output_tokens": 30000},
                {"schema": "CrossAdmissionQAFile", "max_output_tokens": 30000},
            ],
        )
        self.assertEqual((patient_root / "10" / "prompt_record.json").read_text(encoding="utf-8"), prompt_record_before)
        self.assertFalse((self.output_dir / "_tmp" / "qa_777").exists())

    def test_generate_patient_qa_failure_preserves_existing_outputs(self) -> None:
        patient_root = self._write_patient_artifacts(
            888,
            [
                {
                    "hadm_id": "10",
                    "admission_start": "2020-01-01 08:00:00",
                    "admission_end": "2020-01-02 10:00:00",
                    "summary_paragraph": "Earlier admission summary.",
                    "problems": ["Pneumonia"],
                },
                {
                    "hadm_id": "12",
                    "admission_start": "2020-03-01 08:00:00",
                    "admission_end": "2020-03-02 09:00:00",
                    "summary_paragraph": "Later admission summary.",
                    "problems": ["Respiratory failure"],
                },
            ],
        )
        old_single = {"qas": [{"qa_id": "old_single"}]}
        old_cross = {"qas": [{"qa_id": "old_cross"}]}
        old_benchmark = {"qas": [{"qa_id": "old_benchmark"}]}
        self._write_json_file(patient_root / "10" / "qa.json", old_single)
        self._write_json_file(patient_root / "12" / "qa.json", old_single)
        self._write_json_file(patient_root / "cross_admission_qa.json", old_cross)
        self._write_json_file(patient_root / "benchmark_qa.json", old_benchmark)

        pipeline = BenchmarkPipeline(self._config())
        self.addCleanup(pipeline.close)
        pipeline.llm_client = FakeLLMClient(
            [
                {
                    "parsed_output": self._make_single_regular_payload("10"),
                    "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
                },
                {
                    "parsed_output": self._make_single_adversarial_payload("10"),
                    "usage": {"input_tokens": 11, "output_tokens": 21, "total_tokens": 32},
                },
                {
                    "parsed_output": self._make_single_regular_payload("12"),
                    "usage": {"input_tokens": 12, "output_tokens": 22, "total_tokens": 34},
                },
                {
                    "parsed_output": self._make_single_adversarial_payload("12"),
                    "usage": {"input_tokens": 13, "output_tokens": 23, "total_tokens": 36},
                },
                {
                    "parsed_output": {
                        "qas": [
                            {
                                "qa_id": "bad",
                                "scope": "cross_admission",
                                "question_type": "frequency_pattern",
                                "question": "Which issue recurred?",
                                "answer": "heart failure",
                                "evidence": {"admissions": ["10"]},
                            }
                        ]
                    },
                    "usage": {"input_tokens": 14, "output_tokens": 24, "total_tokens": 38},
                },
                {
                    "parsed_output": {
                        "qas": [
                            {
                                "qa_id": "bad_retry",
                                "scope": "cross_admission",
                                "question_type": "frequency_pattern",
                                "question": "Which issue recurred?",
                                "answer": "heart failure",
                                "evidence": {"admissions": ["10"]},
                            }
                        ]
                    },
                    "usage": {"input_tokens": 14, "output_tokens": 24, "total_tokens": 38},
                },
            ]
        )

        with self.assertRaises(QAValidationError):
            pipeline.generate_patient_qa(subject_id=888)

        self.assertEqual(json.loads((patient_root / "10" / "qa.json").read_text(encoding="utf-8")), old_single)
        self.assertEqual(json.loads((patient_root / "12" / "qa.json").read_text(encoding="utf-8")), old_single)
        self.assertEqual(json.loads((patient_root / "cross_admission_qa.json").read_text(encoding="utf-8")), old_cross)
        self.assertEqual(json.loads((patient_root / "benchmark_qa.json").read_text(encoding="utf-8")), old_benchmark)
        self.assertFalse((self.output_dir / "_tmp" / "qa_888").exists())

    def test_generate_patient_qa_retries_invalid_single_admission_output(self) -> None:
        admissions = [
            {
                "hadm_id": "10",
                "admission_start": "2020-01-01 08:00:00",
                "admission_end": "2020-01-02 10:00:00",
                "summary_paragraph": "Earlier admission summary.",
                "problems": ["Pneumonia"],
            },
            {
                "hadm_id": "12",
                "admission_start": "2020-03-01 08:00:00",
                "admission_end": "2020-03-02 09:00:00",
                "summary_paragraph": "Later admission summary.",
                "problems": ["Respiratory failure"],
            },
        ]
        self._write_patient_artifacts(889, admissions)

        invalid_single = self._make_single_regular_payload("10")
        invalid_single["qas"][0]["evidence"]["turn_ids"] = []  # type: ignore[index]

        config = self._config()
        config.llm.max_retries = 1
        pipeline = BenchmarkPipeline(config)
        self.addCleanup(pipeline.close)
        pipeline.llm_client = FakeLLMClient(
            [
                {
                    "parsed_output": invalid_single,
                    "usage": {"input_tokens": 9, "output_tokens": 19, "total_tokens": 28},
                },
                {
                    "parsed_output": self._make_single_regular_payload("10"),
                    "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
                },
                {
                    "parsed_output": self._make_single_regular_payload("10"),
                    "usage": {"input_tokens": 11, "output_tokens": 21, "total_tokens": 32},
                },
                {
                    "parsed_output": self._make_single_adversarial_payload("10"),
                    "usage": {"input_tokens": 12, "output_tokens": 22, "total_tokens": 34},
                },
                {
                    "parsed_output": self._make_single_regular_payload("12"),
                    "usage": {"input_tokens": 13, "output_tokens": 23, "total_tokens": 36},
                },
                {
                    "parsed_output": self._make_single_adversarial_payload("12"),
                    "usage": {"input_tokens": 14, "output_tokens": 24, "total_tokens": 38},
                },
                {
                    "parsed_output": self._make_cross_regular_payload(["10", "12"]),
                    "usage": {"input_tokens": 15, "output_tokens": 25, "total_tokens": 40},
                },
                {
                    "parsed_output": self._make_cross_adversarial_payload(["10", "12"]),
                    "usage": {"input_tokens": 16, "output_tokens": 26, "total_tokens": 42},
                },
            ]
        )

        summary = pipeline.generate_patient_qa(subject_id=889)

        self.assertEqual(summary["total_qas"], 12)
        self.assertEqual(len(pipeline.llm_client.structured_calls), 8)
        self.assertEqual(
            [call["schema"] for call in pipeline.llm_client.structured_calls],
            [
                "SingleAdmissionQAFile",
                "SingleAdmissionQAFile",
                "SingleAdmissionQAFile",
                "SingleAdmissionQAFile",
                "SingleAdmissionQAFile",
                "SingleAdmissionQAFile",
                "CrossAdmissionQAFile",
                "CrossAdmissionQAFile",
            ],
        )
        self.assertFalse((self.output_dir / "_tmp" / "qa_889").exists())

    def test_verify_patient_outputs_accepts_complete_patient_directory(self) -> None:
        admissions = [
            {
                "hadm_id": "10",
                "admission_start": "2020-01-01 08:00:00",
                "admission_end": "2020-01-02 10:00:00",
                "summary_paragraph": "Earlier admission summary.",
                "problems": ["Pneumonia"],
            },
            {
                "hadm_id": "12",
                "admission_start": "2020-03-01 08:00:00",
                "admission_end": "2020-03-02 09:00:00",
                "summary_paragraph": "Later admission summary.",
                "problems": ["Respiratory failure"],
            },
        ]
        patient_root = self._write_patient_artifacts(900, admissions)
        self._write_patient_qa_outputs(patient_root, admissions)

        summary = verify_patient_outputs(patient_root, expect_qa=True)

        self.assertEqual(summary["processed_admissions"], 2)
        self.assertEqual(summary["qa_counts"]["benchmark_total"], 12)

    def test_verify_patient_outputs_requires_qa_files_when_expected(self) -> None:
        admissions = [
            {
                "hadm_id": "10",
                "admission_start": "2020-01-01 08:00:00",
                "admission_end": "2020-01-02 10:00:00",
                "summary_paragraph": "Earlier admission summary.",
                "problems": ["Pneumonia"],
            },
            {
                "hadm_id": "12",
                "admission_start": "2020-03-01 08:00:00",
                "admission_end": "2020-03-02 09:00:00",
                "summary_paragraph": "Later admission summary.",
                "problems": ["Respiratory failure"],
            },
        ]
        patient_root = self._write_patient_artifacts(901, admissions)

        with self.assertRaises(ValueError):
            verify_patient_outputs(patient_root, expect_qa=True)

    def test_verify_patient_outputs_rejects_malformed_json(self) -> None:
        admissions = [
            {
                "hadm_id": "10",
                "admission_start": "2020-01-01 08:00:00",
                "admission_end": "2020-01-02 10:00:00",
                "summary_paragraph": "Earlier admission summary.",
                "problems": ["Pneumonia"],
            },
            {
                "hadm_id": "12",
                "admission_start": "2020-03-01 08:00:00",
                "admission_end": "2020-03-02 09:00:00",
                "summary_paragraph": "Later admission summary.",
                "problems": ["Respiratory failure"],
            },
        ]
        patient_root = self._write_patient_artifacts(902, admissions)
        self._write_patient_qa_outputs(patient_root, admissions)
        (patient_root / "benchmark_qa.json").write_text("{not valid json", encoding="utf-8")

        with self.assertRaises(ValueError):
            verify_patient_outputs(patient_root, expect_qa=True)

    def test_verify_patient_outputs_rejects_inconsistent_benchmark_counts(self) -> None:
        admissions = [
            {
                "hadm_id": "10",
                "admission_start": "2020-01-01 08:00:00",
                "admission_end": "2020-01-02 10:00:00",
                "summary_paragraph": "Earlier admission summary.",
                "problems": ["Pneumonia"],
            },
            {
                "hadm_id": "12",
                "admission_start": "2020-03-01 08:00:00",
                "admission_end": "2020-03-02 09:00:00",
                "summary_paragraph": "Later admission summary.",
                "problems": ["Respiratory failure"],
            },
        ]
        patient_root = self._write_patient_artifacts(903, admissions)
        self._write_patient_qa_outputs(patient_root, admissions)
        self._write_json_file(patient_root / "benchmark_qa.json", {"qas": [{"qa_id": "wrong"}]})

        with self.assertRaises(ValueError):
            verify_patient_outputs(patient_root, expect_qa=True)

    def test_generate_all_writes_batch_summary_on_success(self) -> None:
        config = self._config()
        pipeline = BenchmarkPipeline(config)
        self.addCleanup(pipeline.close)

        def fake_generate_patient_sample(*, subject_id, **kwargs):
            del kwargs
            return {"subject_id": str(subject_id), "processed_admissions": 2}

        def fake_generate_patient_qa(*, subject_id, **kwargs):
            del kwargs
            return {"subject_id": str(subject_id), "processed_admissions": 2, "total_qas": 12}

        with patch.object(pipeline, "generate_patient_sample", side_effect=fake_generate_patient_sample), patch.object(
            pipeline,
            "generate_patient_qa",
            side_effect=fake_generate_patient_qa,
        ), patch(
            "health_benchmark.scripts.pipeline.verify_patient_outputs",
            side_effect=lambda *_args, **_kwargs: {
                "processed_admissions": 2,
                "qa_counts": {"benchmark_total": 12},
            },
        ):
            summary = pipeline.generate_all(
                subject_ids=[100, 200],
                model_name="Qwen/Qwen3.5-27B",
            )

        self.assertEqual(summary["succeeded"], [100, 200])
        self.assertEqual(summary["failed"], [])
        batch_summary_path = Path(summary["batch_summary_path"])
        self.assertTrue(batch_summary_path.exists())
        saved_summary = json.loads(batch_summary_path.read_text(encoding="utf-8"))
        self.assertEqual(saved_summary["final_status"], "success")
        self.assertEqual(saved_summary["requested_subject_ids"], [100, 200])

    def test_generate_all_records_partial_failure_and_continues(self) -> None:
        config = self._config()
        pipeline = BenchmarkPipeline(config)
        self.addCleanup(pipeline.close)

        def fake_generate_patient_sample(*, subject_id, **kwargs):
            del kwargs
            if subject_id == 200:
                raise ValueError("bad patient")
            return {"subject_id": str(subject_id), "processed_admissions": 2}

        def fake_generate_patient_qa(*, subject_id, **kwargs):
            del kwargs
            return {"subject_id": str(subject_id), "processed_admissions": 2, "total_qas": 12}

        with patch.object(pipeline, "generate_patient_sample", side_effect=fake_generate_patient_sample), patch.object(
            pipeline,
            "generate_patient_qa",
            side_effect=fake_generate_patient_qa,
        ), patch(
            "health_benchmark.scripts.pipeline.verify_patient_outputs",
            side_effect=lambda *_args, **_kwargs: {"processed_admissions": 2},
        ):
            summary = pipeline.generate_all(subject_ids=[100, 200, 300], model_name="Qwen/Qwen3.5-27B")

        self.assertEqual(summary["succeeded"], [100, 300])
        self.assertEqual(summary["failed"], [200])
        self.assertEqual(summary["results"][1]["error"], "bad patient")

    def test_generate_all_stops_early_when_fail_fast_is_enabled(self) -> None:
        config = self._config()
        pipeline = BenchmarkPipeline(config)
        self.addCleanup(pipeline.close)
        sample_calls: list[int] = []

        def fake_generate_patient_sample(*, subject_id, **kwargs):
            del kwargs
            sample_calls.append(subject_id)
            if subject_id == 200:
                raise ValueError("bad patient")
            return {"subject_id": str(subject_id), "processed_admissions": 2}

        def fake_generate_patient_qa(*, subject_id, **kwargs):
            del kwargs
            return {"subject_id": str(subject_id), "processed_admissions": 2, "total_qas": 12}

        with patch.object(pipeline, "generate_patient_sample", side_effect=fake_generate_patient_sample), patch.object(
            pipeline,
            "generate_patient_qa",
            side_effect=fake_generate_patient_qa,
        ), patch(
            "health_benchmark.scripts.pipeline.verify_patient_outputs",
            side_effect=lambda *_args, **_kwargs: {"processed_admissions": 2},
        ):
            summary = pipeline.generate_all(
                subject_ids=[100, 200, 300],
                model_name="Qwen/Qwen3.5-27B",
                fail_fast=True,
            )

        self.assertEqual(sample_calls, [100, 200])
        self.assertEqual(summary["succeeded"], [100])
        self.assertEqual(summary["failed"], [200])
        self.assertEqual(len(summary["results"]), 2)

    def test_quest_batch_script_uses_submit_dir_or_project_root(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = (repo_root / "quest" / "run_generate_all_qwen.slurm").read_text(encoding="utf-8")

        self.assertIn("/projects/p33194/health-benchmark", script)
        self.assertIn("/projects/p33194/health-benchmark/data/mimic-iv", script)
        self.assertIn("/projects/p33194/health-benchmark/data/mimic-iv-notes", script)
        self.assertIn("/projects/p33194/health-benchmark/output/benchmark", script)
        self.assertIn("/projects/p33194/health-benchmark/runtime/hf_cache", script)
        self.assertIn("/hpc/software/mamba/24.3.0", script)
        self.assertIn("/projects/p33194/health-benchmark/runtime/envs/medbench", script)
        self.assertIn("/software/singularity/3.8.1/bin/singularity", script)
        self.assertIn("/projects/p33194/health-benchmark/runtime/containers/vllm-openai_latest.sif", script)
        self.assertIn("/gpfs/projects", script)
        self.assertIn("#SBATCH --output=quest/slurm_logs/%x-%j.out", script)
        self.assertIn("#SBATCH --gres=gpu:4", script)
        self.assertIn("#SBATCH --constraint=sxm", script)
        self.assertIn("#SBATCH --exclude=qgpu2014", script)
        self.assertIn('MODEL="${MODEL:-Qwen/Qwen3-235B-A22B-Instruct-2507-FP8}"', script)
        self.assertIn('MAX_MODEL_LEN="${MAX_MODEL_LEN:-49152}"', script)
        self.assertIn('VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-7200}"', script)
        self.assertIn('RETRY_LIMIT="${RETRY_LIMIT:-3}"', script)
        self.assertIn('READY_CHECK_ATTEMPTS="${READY_CHECK_ATTEMPTS:-$(( (VLLM_ENGINE_READY_TIMEOUT_S + READY_CHECK_SLEEP_SECONDS - 1) / READY_CHECK_SLEEP_SECONDS ))}"', script)
        self.assertIn('DEFAULT_EXCLUDED_NODES="qgpu2014"', script)
        self.assertIn("PROJECT_ROOT", script)
        self.assertIn("SLURM_SUBMIT_DIR", script)
        self.assertIn("MAMBA_ROOT", script)
        self.assertIn("ENV_PREFIX", script)
        self.assertIn("SINGULARITY_BIN", script)
        self.assertIn("VLLM_IMAGE", script)
        self.assertIn("MIMICIV_DIR", script)
        self.assertIn("MIMIC_IV_DIR", script)
        self.assertIn("MIMICIV_NOTE_DIR", script)
        self.assertIn("MIMIC_IV_NOTE_DIR", script)
        self.assertIn("OUTPUT_ROOT", script)
        self.assertIn("MEDBENCH_OUTPUT_ROOT", script)
        self.assertIn("HF_HOME", script)
        self.assertIn("HUGGINGFACE_HUB_CACHE", script)
        self.assertIn("resolve_optional_env", script)
        self.assertIn("activate_runtime_env", script)
        self.assertIn("normalize_quest_path", script)
        self.assertIn("validate_vllm_runtime", script)
        self.assertIn('mamba activate "$ENV_PREFIX"', script)
        self.assertIn('Using python: $python_path', script)
        self.assertIn('Using singularity: $SINGULARITY_BIN', script)
        self.assertIn('Using vLLM image: $VLLM_IMAGE', script)
        self.assertIn('Detected GPU count: $gpu_count', script)
        self.assertIn('Detected GPU model summary: $gpu_summary', script)
        self.assertIn('Visible GPU count ($gpu_count) is less than tensor parallel size ($TENSOR_PARALLEL_SIZE).', script)
        self.assertIn("SLURMD_NODENAME", script)
        self.assertIn("SLURM_JOB_NODELIST", script)
        self.assertIn("hostname -s", script)
        self.assertIn('nvidia-smi -L', script)
        self.assertIn('validate_non_negative_integer "RETRY_LIMIT" "$RETRY_LIMIT"', script)
        self.assertIn('echo "Using launcher default excluded nodes: $DEFAULT_EXCLUDED_NODES"', script)
        self.assertIn('echo "Using vLLM ready timeout: $VLLM_ENGINE_READY_TIMEOUT_S"', script)
        self.assertIn('echo "Using generation retry limit: $RETRY_LIMIT"', script)
        self.assertIn('VLLM_LOG="${VLLM_LOG:-$OUTPUT_ROOT/logs/vllm/vllm_${SLURM_JOB_ID:-manual}.log}"', script)
        self.assertIn('mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$(dirname "$VLLM_LOG")"', script)
        self.assertIn('"$SINGULARITY_BIN" exec --nv -B /projects:/projects "$VLLM_IMAGE"', script)
        self.assertIn('vLLM exited before becoming ready. See $VLLM_LOG', script)
        self.assertIn('kill -0 "$VLLM_PID"', script)
        self.assertIn("resolve_required_env", script)
        self.assertIn('python "$REPO_ROOT/main.py" generate-all', script)
        self.assertIn('--retry-limit "$RETRY_LIMIT"', script)
        self.assertIn("Resolved repo root: $REPO_ROOT", script)
        self.assertNotIn("gpu:a100", script)
        self.assertNotIn('REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"', script)
        self.assertNotIn('export HF_HOME="${HF_HOME:-$REPO_ROOT/.cache/hf}"', script)
        self.assertNotIn('Using vllm: $vllm_path', script)

    def test_quest_batch_script_rejects_invalid_retry_limit(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script_path = repo_root / "quest" / "run_generate_all_qwen.slurm"

        completed = subprocess.run(
            ["/bin/bash", str(script_path), "11826927"],
            check=False,
            capture_output=True,
            text=True,
            env={"RETRY_LIMIT": "-1"},
        )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("RETRY_LIMIT must be a non-negative integer, got: -1", completed.stderr)

    def test_quest_generation_submission_helper_emits_smoke_test_plus_low_batches(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        helper = repo_root / "quest" / "emit_generation_submission_commands.py"

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "top_100_eligible_patients.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["subject_id", "eligible_admissions", "quest_generation_complete"],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {"subject_id": 999, "eligible_admissions": 1, "quest_generation_complete": "yes"},
                        {"subject_id": 101, "eligible_admissions": 9, "quest_generation_complete": "no"},
                        {"subject_id": 102, "eligible_admissions": 5, "quest_generation_complete": "no"},
                        {"subject_id": 103, "eligible_admissions": 5, "quest_generation_complete": "no"},
                        {"subject_id": 104, "eligible_admissions": 4, "quest_generation_complete": "yes"},
                        {"subject_id": 105, "eligible_admissions": 4, "quest_generation_complete": "no"},
                        {"subject_id": 106, "eligible_admissions": 7, "quest_generation_complete": "no"},
                        {"subject_id": 107, "eligible_admissions": 10, "quest_generation_complete": "no"},
                    ]
                )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(helper),
                    "--csv",
                    str(csv_path),
                    "--account",
                    "demo",
                    "--launcher",
                    "/tmp/run_generate_all_qwen.slurm",
                    "--smoke-test-subject",
                    "999",
                    "--lowest-count",
                    "6",
                    "--batch-size",
                    "2",
                ],
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("SCRIPT=/tmp/run_generate_all_qwen.slurm", completed.stdout)
        self.assertIn("ACCOUNT=demo", completed.stdout)
        self.assertIn("sbatch --account=$ACCOUNT $SCRIPT 999", completed.stdout)
        self.assertIn("# Job 2\n# counts: 4 4\nsbatch --account=$ACCOUNT $SCRIPT 104 105", completed.stdout)
        self.assertIn("# Job 3\n# counts: 5 5\nsbatch --account=$ACCOUNT $SCRIPT 102 103", completed.stdout)
        self.assertIn("# Job 4\n# counts: 7 9\nsbatch --account=$ACCOUNT $SCRIPT 106 101", completed.stdout)
        self.assertNotIn(" 999 ", completed.stdout)

    def test_quest_shortlist_status_updater_marks_complete_generation_outputs(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        updater = repo_root / "quest" / "update_shortlist_generation_status.py"

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            csv_path = temp_root / "top_100_eligible_patients.csv"
            quest_output_root = temp_root / "benchmark"
            quest_output_root.mkdir()

            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["subject_id", "eligible_admissions", "notes"])
                writer.writeheader()
                writer.writerows(
                    [
                        {"subject_id": 101, "eligible_admissions": 9, "notes": "first"},
                        {"subject_id": 102, "eligible_admissions": 5, "notes": "second"},
                        {"subject_id": 103, "eligible_admissions": 4, "notes": "third"},
                    ]
                )

            complete_root = quest_output_root / "101"
            complete_root.mkdir()
            for filename in (
                "combined_conversation.json",
                "patient_summary.json",
                "cross_admission_qa.json",
                "benchmark_qa.json",
            ):
                (complete_root / filename).write_text("{}", encoding="utf-8")

            partial_root = quest_output_root / "102"
            partial_root.mkdir()
            (partial_root / "combined_conversation.json").write_text("{}", encoding="utf-8")
            (partial_root / "patient_summary.json").write_text("{}", encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    str(updater),
                    "--csv",
                    str(csv_path),
                    "--quest-output-root",
                    str(quest_output_root),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            with csv_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                fieldnames = reader.fieldnames

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(fieldnames, ["subject_id", "eligible_admissions", "notes", "quest_generation_complete"])
        self.assertEqual([row["subject_id"] for row in rows], ["101", "102", "103"])
        self.assertEqual([row["quest_generation_complete"] for row in rows], ["yes", "no", "no"])
        self.assertEqual([row["notes"] for row in rows], ["first", "second", "third"])

    def test_quest_interactive_script_validates_repo_root_and_uses_absolute_main(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = (repo_root / "quest" / "debug_qwen_interactive.sh").read_text(encoding="utf-8")

        self.assertIn("/projects/p33194/health-benchmark", script)
        self.assertIn("/projects/p33194/health-benchmark/data/mimic-iv", script)
        self.assertIn("/projects/p33194/health-benchmark/data/mimic-iv-notes", script)
        self.assertIn("/projects/p33194/health-benchmark/output/benchmark", script)
        self.assertIn("/projects/p33194/health-benchmark/runtime/hf_cache", script)
        self.assertIn("/hpc/software/mamba/24.3.0", script)
        self.assertIn("/projects/p33194/health-benchmark/runtime/envs/medbench", script)
        self.assertIn("/software/singularity/3.8.1/bin/singularity", script)
        self.assertIn("/projects/p33194/health-benchmark/runtime/containers/vllm-openai_latest.sif", script)
        self.assertIn("/gpfs/projects", script)
        self.assertIn('MODEL="${MODEL:-Qwen/Qwen3-235B-A22B-Instruct-2507-FP8}"', script)
        self.assertIn('MAX_MODEL_LEN="${MAX_MODEL_LEN:-49152}"', script)
        self.assertIn('VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-3600}"', script)
        self.assertIn('READY_CHECK_ATTEMPTS="${READY_CHECK_ATTEMPTS:-$(( (VLLM_ENGINE_READY_TIMEOUT_S + READY_CHECK_SLEEP_SECONDS - 1) / READY_CHECK_SLEEP_SECONDS ))}"', script)
        self.assertIn("PROJECT_ROOT", script)
        self.assertIn("MAMBA_ROOT", script)
        self.assertIn("ENV_PREFIX", script)
        self.assertIn("SINGULARITY_BIN", script)
        self.assertIn("VLLM_IMAGE", script)
        self.assertIn("MIMICIV_DIR", script)
        self.assertIn("MIMIC_IV_DIR", script)
        self.assertIn("MIMICIV_NOTE_DIR", script)
        self.assertIn("MIMIC_IV_NOTE_DIR", script)
        self.assertIn("OUTPUT_ROOT", script)
        self.assertIn("MEDBENCH_OUTPUT_ROOT", script)
        self.assertIn("HF_HOME", script)
        self.assertIn("HUGGINGFACE_HUB_CACHE", script)
        self.assertIn("resolve_optional_env", script)
        self.assertIn("activate_runtime_env", script)
        self.assertIn("normalize_quest_path", script)
        self.assertIn("validate_vllm_runtime", script)
        self.assertIn('mamba activate "$ENV_PREFIX"', script)
        self.assertIn('Using python: $python_path', script)
        self.assertIn('Using singularity: $SINGULARITY_BIN', script)
        self.assertIn('Using vLLM image: $VLLM_IMAGE', script)
        self.assertIn('Detected GPU count: $gpu_count', script)
        self.assertIn('Detected GPU model summary: $gpu_summary', script)
        self.assertIn('Visible GPU count ($gpu_count) is less than tensor parallel size ($TENSOR_PARALLEL_SIZE).', script)
        self.assertIn("SLURMD_NODENAME", script)
        self.assertIn("SLURM_JOB_NODELIST", script)
        self.assertIn("hostname -s", script)
        self.assertIn('nvidia-smi -L', script)
        self.assertIn('"$SINGULARITY_BIN" exec --nv -B /projects:/projects "$VLLM_IMAGE"', script)
        self.assertIn('vLLM exited before becoming ready. See $VLLM_LOG', script)
        self.assertIn('kill -0 "$VLLM_PID"', script)
        self.assertIn("resolve_required_env", script)
        self.assertIn('python "$REPO_ROOT/main.py" generate-all', script)
        self.assertIn("Resolved repo root: $REPO_ROOT", script)
        self.assertIn("Expected main.py and health_benchmark/", script)
        self.assertNotIn('export HF_HOME="${HF_HOME:-$REPO_ROOT/.cache/hf}"', script)
        self.assertNotIn('Using vllm: $vllm_path', script)

    def test_quest_evaluation_scripts_use_trio_profile_and_manifest_flow(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        canonical_slurm_script = (repo_root / "quest" / "evaluate_models.slurm").read_text(encoding="utf-8")
        run_script = (repo_root / "quest" / "run_multi_patient_eval_job.sh").read_text(encoding="utf-8")
        launch_script = (repo_root / "quest" / "launch_vllm_server.sh").read_text(encoding="utf-8")
        stop_script = (repo_root / "quest" / "stop_vllm_server.sh").read_text(encoding="utf-8")
        wait_script = (repo_root / "quest" / "wait_for_server.py").read_text(encoding="utf-8")
        ignored = subprocess.run(
            ["git", "check-ignore", "-q", "quest/evaluate_models.slurm"],
            cwd=repo_root,
            check=False,
        )

        self.assertEqual(ignored.returncode, 0)
        self.assertFalse((repo_root / "quest" / "qwen_open_eval_multi_patient.slurm").exists())
        self.assertFalse((repo_root / "quest" / "qwen_open_eval_small_models.slurm").exists())
        self.assertFalse((repo_root / "quest" / "qwen_open_eval_small_models_1gpu.slurm").exists())
        self.assertFalse((repo_root / "quest" / "qwen_open_eval_27b_2gpu.slurm").exists())
        self.assertIn("#SBATCH --job-name=medbench-eval", canonical_slurm_script)
        self.assertIn("#SBATCH --time=06:00:00", canonical_slurm_script)
        self.assertIn('EVAL_MODEL_PRESET="${EVAL_MODEL_PRESET:-qwen3.5_trio}"', canonical_slurm_script)
        self.assertIn('EVAL_VARIANT="${EVAL_VARIANT:-normal}"', canonical_slurm_script)
        self.assertIn('EVAL_STAGE="${EVAL_STAGE:-full}"', canonical_slurm_script)
        self.assertIn('EVAL_RETRY_LIMIT="${EVAL_RETRY_LIMIT:-3}"', canonical_slurm_script)
        self.assertIn('EVAL_TIMEOUT_SECONDS="${EVAL_TIMEOUT_SECONDS:-600}"', canonical_slurm_script)
        self.assertIn('SERVER_READY_TIMEOUT_S="${SERVER_READY_TIMEOUT_S:-1800}"', canonical_slurm_script)
        self.assertIn('JUDGE_TENSOR_PARALLEL_SIZE="${JUDGE_TENSOR_PARALLEL_SIZE:-}"', canonical_slurm_script)
        self.assertIn('JUDGE_MAX_MODEL_LEN="${JUDGE_MAX_MODEL_LEN:-}"', canonical_slurm_script)
        self.assertIn('JUDGE_GPU_DEVICE_IDS="${JUDGE_GPU_DEVICE_IDS:-}"', canonical_slurm_script)
        self.assertIn("/projects/p33194/health-benchmark/output/benchmark", canonical_slurm_script)
        self.assertIn("/projects/p33194/health-benchmark/output/evaluation", canonical_slurm_script)
        self.assertIn('EVALUATION_ROOT="$(resolve_optional_env EVALUATION_ROOT "$DEFAULT_EVALUATION_ROOT")"', canonical_slurm_script)
        self.assertIn('QUEST_JOB_OUTPUT_ROOT="$EVALUATION_ROOT/quest_job_outputs/${SLURM_JOB_ID:-manual}"', canonical_slurm_script)
        self.assertIn('REQUIRED_TOTAL_GPU_COUNT="${REQUIRED_TOTAL_GPU_COUNT:-2}"', canonical_slurm_script)
        self.assertIn('MODEL_RECORD_ARGS=()', canonical_slurm_script)
        self.assertIn('MODEL_ARGS=()', canonical_slurm_script)
        self.assertIn('PATIENT_ARGS=()', canonical_slurm_script)
        self.assertIn('MODEL_PRESET_ARG_SET=0', canonical_slurm_script)
        self.assertIn("--models MODEL...", canonical_slurm_script)
        self.assertIn("--model-preset PRESET", canonical_slurm_script)
        self.assertIn("medgemma_pair", canonical_slurm_script)
        self.assertIn("medical_next_small", canonical_slurm_script)
        self.assertIn("medical_next_32b", canonical_slurm_script)
        self.assertIn("--model-record RECORD", canonical_slurm_script)
        self.assertIn("--patients SUBJECT_ID...", canonical_slurm_script)
        self.assertIn("--patient-manifest PATH", canonical_slurm_script)
        self.assertIn("--memory, --mem0", canonical_slurm_script)
        self.assertIn("--stage full|answers|judge", canonical_slurm_script)
        self.assertIn("--judge-tensor-parallel-size N", canonical_slurm_script)
        self.assertIn("--judge-max-model-len N", canonical_slurm_script)
        self.assertIn("--judge-gpu-device-ids IDS", canonical_slurm_script)
        self.assertIn("--retry-limit N", canonical_slurm_script)
        self.assertIn("--timeout-seconds N", canonical_slurm_script)
        self.assertIn("--server-ready-timeout-seconds N", canonical_slurm_script)
        self.assertIn("--mem0-answer-retrieval-top-k N", canonical_slurm_script)
        self.assertIn("--mem0-embedding-model MODEL", canonical_slurm_script)
        self.assertIn("--mem0-model-max-len N", canonical_slurm_script)
        self.assertIn('validate_non_negative_integer "EVAL_RETRY_LIMIT" "$EVAL_RETRY_LIMIT"', canonical_slurm_script)
        self.assertIn('validate_positive_integer "EVAL_TIMEOUT_SECONDS" "$EVAL_TIMEOUT_SECONDS"', canonical_slurm_script)
        self.assertIn('validate_positive_integer "SERVER_READY_TIMEOUT_S" "$SERVER_READY_TIMEOUT_S"', canonical_slurm_script)
        self.assertIn('EVAL_STAGE must be one of: full, answers, judge.', canonical_slurm_script)
        self.assertIn('validate_positive_integer "JUDGE_TENSOR_PARALLEL_SIZE" "$JUDGE_TENSOR_PARALLEL_SIZE"', canonical_slurm_script)
        self.assertIn('validate_positive_integer "JUDGE_MAX_MODEL_LEN" "$JUDGE_MAX_MODEL_LEN"', canonical_slurm_script)
        self.assertIn('if [[ "$EVAL_STAGE" == "judge" && "$REQUIRED_TOTAL_GPU_COUNT_ENV_SET" -eq 0 ]]; then', canonical_slurm_script)
        self.assertIn('EVAL_VARIANT must be one of: normal, memory, mem0, rag.', canonical_slurm_script)
        self.assertIn('validate_positive_integer "MEM0_ANSWER_RETRIEVAL_TOP_K" "$MEM0_ANSWER_RETRIEVAL_TOP_K"', canonical_slurm_script)
        self.assertIn('validate_positive_integer "MEM0_EMBEDDING_BATCH_SIZE" "$MEM0_EMBEDDING_BATCH_SIZE"', canonical_slurm_script)
        self.assertIn('validate_positive_integer "MEM0_MODEL_MAX_LEN" "$MEM0_MODEL_MAX_LEN"', canonical_slurm_script)
        self.assertIn('EVAL_MODELS="$(printf', canonical_slurm_script)
        self.assertIn('EVAL_MODEL_RECORDS="$(printf', canonical_slurm_script)
        self.assertIn('Use --model-preset by itself', canonical_slurm_script)
        self.assertIn('Using evaluation variant: $EVAL_VARIANT', canonical_slurm_script)
        self.assertIn('Using evaluation stage: $EVAL_STAGE', canonical_slurm_script)
        self.assertIn('Using judge tensor parallel size override: ${JUDGE_TENSOR_PARALLEL_SIZE:-auto}', canonical_slurm_script)
        self.assertIn('Using Gemma vLLM dtype: ${GEMMA_VLLM_DTYPE:-auto}', canonical_slurm_script)
        self.assertIn('EVAL_RETRY_LIMIT EVAL_TIMEOUT_SECONDS', canonical_slurm_script)
        self.assertIn('JUDGE_TENSOR_PARALLEL_SIZE JUDGE_MAX_MODEL_LEN JUDGE_GPU_DEVICE_IDS', canonical_slurm_script)
        self.assertIn('VLLM_DTYPE VLLM_CHAT_TEMPLATE GEMMA_VLLM_DTYPE GEMMA_VLLM_CHAT_TEMPLATE', canonical_slurm_script)
        self.assertIn('MEM0_CHUNK_TOKEN_CAP MEM0_PREVIOUS_CHUNK_SUMMARIES', canonical_slurm_script)
        self.assertIn('MEM0_EMBEDDING_MODEL MEM0_EMBEDDING_DEVICE', canonical_slurm_script)
        self.assertIn('bash "$REPO_ROOT/quest/run_multi_patient_eval_job.sh" "${RUN_PATIENT_ARGS[@]}"', canonical_slurm_script)

        self.assertIn('EVAL_MODEL_PRESET="${EVAL_MODEL_PRESET:-qwen3.5_trio}"', run_script)
        self.assertIn('REASONING_PARSER="${REASONING_PARSER-qwen3}"', run_script)
        self.assertNotIn('REASONING_PARSER="${REASONING_PARSER:-qwen3}"', run_script)
        self.assertIn('EVAL_VARIANT="${EVAL_VARIANT:-normal}"', run_script)
        self.assertIn('EVAL_STAGE="${EVAL_STAGE:-full}"', run_script)
        self.assertIn('EVAL_RETRY_LIMIT="${EVAL_RETRY_LIMIT:-3}"', run_script)
        self.assertIn('EVAL_TIMEOUT_SECONDS="${EVAL_TIMEOUT_SECONDS:-600}"', run_script)
        self.assertIn('SERVER_READY_TIMEOUT_S="${SERVER_READY_TIMEOUT_S:-1800}"', run_script)
        self.assertIn('EVAL_MODEL_RECORDS', run_script)
        self.assertIn('EVAL_MODELS', run_script)
        self.assertIn('model_record_for_name()', run_script)
        self.assertIn("trio|qwen3.5_trio)", run_script)
        self.assertIn("gemma3_trio|gemma-3-trio|gemma_trio)", run_script)
        self.assertIn("medgemma_pair|medgemma-pair|medgemma)", run_script)
        self.assertIn("medical_next_small|medical-next-small|medical_small)", run_script)
        self.assertIn("medical_next_32b|medical-next-32b|medical_32b)", run_script)
        self.assertIn("medical_next_all|medical-next-all|medical_all)", run_script)
        self.assertIn("small|qwen3.5_small|small_1gpu|qwen3.5_small_1gpu)", run_script)
        self.assertIn("27b_2gpu|qwen3.5_27b_2gpu)", run_script)
        self.assertNotIn("retired for judged evaluation", run_script)
        self.assertIn('model_record_for_name "Qwen/Qwen3.5-4B"', run_script)
        self.assertIn('model_record_for_name "Qwen/Qwen3.5-9B"', run_script)
        self.assertIn('model_record_for_name "Qwen/Qwen3.5-27B"', run_script)
        self.assertIn('Qwen/Qwen3.5-27B|qwen3.5-27b|2|131072|', run_script)
        self.assertIn('google/gemma-3-4b-it|gemma-3-4b-it', run_script)
        self.assertIn('google/gemma-3-12b-it|gemma-3-12b-it', run_script)
        self.assertIn('google/gemma-3-27b-it|gemma-3-27b-it|2|131072|', run_script)
        self.assertIn('google/medgemma-4b-it|medgemma-4b-it', run_script)
        self.assertIn('google/medgemma-27b-it|medgemma-27b-it|2|131072|', run_script)
        self.assertIn('MBZUAI/MedMO-4B-Next" "medmo-4b-next"', run_script)
        self.assertIn('MBZUAI/MedMO-8B-Next" "medmo-8b-next"', run_script)
        self.assertIn('microsoft/MediPhi-Instruct" "mediphi-instruct"', run_script)
        self.assertIn('lingshu-medical-mllm/Lingshu-32B|lingshu-32b|2|128000|', run_script)
        self.assertNotIn('baichuan-inc/Baichuan-M2-32B|baichuan-m2-32b|2|131072|', run_script)
        self.assertIn('google/medgemma-*|medgemma-*', run_script)
        self.assertIn('is_gemma3_model()', run_script)
        self.assertIn('is_medical_next_parser_free_model()', run_script)
        self.assertNotIn('is_baichuan_m2_model()', run_script)
        self.assertIn('is_bfloat16_answer_model()', run_script)
        self.assertIn('printf \'%s\\n\' "$REASONING_PARSER"', run_script)
        self.assertIn('printf \'%s\\n\' "bfloat16"', run_script)
        self.assertIn('snapshot_has_tokenizer_files()', run_script)
        self.assertIn('snapshot_has_weight_files()', run_script)
        self.assertIn('snapshot_is_complete_for_vllm()', run_script)
        self.assertIn('snapshot_is_complete_for_vllm "$snapshot"', run_script)
        self.assertIn('server_reasoning_parser_for_model()', run_script)
        self.assertIn('GEMMA_VLLM_DTYPE="${GEMMA_VLLM_DTYPE:-bfloat16}"', run_script)
        self.assertIn('reasoning parser: ${server_reasoning_parser:-none}', run_script)
        self.assertIn('MEM0_EMBEDDING_MODEL="${MEM0_EMBEDDING_MODEL:-Qwen/Qwen3-Embedding-8B}"', run_script)
        self.assertIn('MEM0_EMBEDDING_GPU_DEVICE_IDS="${MEM0_EMBEDDING_GPU_DEVICE_IDS:-1}"', run_script)
        self.assertIn('MEM0_MODEL_MAX_LEN="${MEM0_MODEL_MAX_LEN:-32768}"', run_script)
        self.assertIn('MEM0_MODEL_TENSOR_PARALLEL_SIZE="${MEM0_MODEL_TENSOR_PARALLEL_SIZE:-1}"', run_script)
        self.assertIn('MEM0_MODEL_GPU_DEVICE_IDS="${MEM0_MODEL_GPU_DEVICE_IDS:-0}"', run_script)
        self.assertIn('mapfile -t MODELS < <(resolve_model_records "$EVAL_MODEL_PRESET")', run_script)
        self.assertIn('write_patient_manifest_snapshot()', run_script)
        self.assertIn('csv.DictReader', run_script)
        self.assertIn('"subject_id"', run_script)
        self.assertIn('write_status_row()', run_script)
        self.assertIn('DEFAULT_ANSWER_VLLM_PORT="$((20000 + (SLURM_JOB_ID % 10000) * 4))"', run_script)
        self.assertIn('ANSWER_VLLM_PORT="${ANSWER_VLLM_PORT:-${VLLM_PORT:-$DEFAULT_ANSWER_VLLM_PORT}}"', run_script)
        self.assertIn('DEFAULT_JUDGE_VLLM_PORT="$((ANSWER_VLLM_PORT + 2))"', run_script)
        self.assertIn('JUDGE_VLLM_PORT="${JUDGE_VLLM_PORT:-$DEFAULT_JUDGE_VLLM_PORT}"', run_script)
        self.assertIn('DEFAULT_ANSWER_GPU_DEVICE_IDS="${DEFAULT_ANSWER_GPU_DEVICE_IDS:-0}"', run_script)
        self.assertIn('validate_tcp_port "ANSWER_VLLM_PORT" "$ANSWER_VLLM_PORT"', run_script)
        self.assertIn('validate_tcp_port "JUDGE_VLLM_PORT" "$JUDGE_VLLM_PORT"', run_script)
        self.assertIn('Using answer vLLM port: $ANSWER_VLLM_PORT', run_script)
        self.assertIn('Using judge vLLM port: $JUDGE_VLLM_PORT', run_script)
        self.assertIn('VLLM_DTYPE_CLI=()', launch_script)
        self.assertIn('--dtype "$VLLM_DTYPE"', launch_script)
        self.assertIn('VLLM_CHAT_TEMPLATE_CLI=()', launch_script)
        self.assertIn('--chat-template "$VLLM_CHAT_TEMPLATE"', launch_script)
        self.assertIn('DEFAULT_JUDGE_TENSOR_PARALLEL_SIZE="1"', run_script)
        self.assertIn('DEFAULT_JUDGE_MAX_MODEL_LEN="32768"', run_script)
        self.assertIn('DEFAULT_JUDGE_GPU_DEVICE_IDS="0"', run_script)
        self.assertIn('JUDGE_TENSOR_PARALLEL_SIZE="${JUDGE_TENSOR_PARALLEL_SIZE:-$DEFAULT_JUDGE_TENSOR_PARALLEL_SIZE}"', run_script)
        self.assertIn('JUDGE_MAX_MODEL_LEN="${JUDGE_MAX_MODEL_LEN:-$DEFAULT_JUDGE_MAX_MODEL_LEN}"', run_script)
        self.assertIn('JUDGE_GPU_DEVICE_IDS="${JUDGE_GPU_DEVICE_IDS:-$DEFAULT_JUDGE_GPU_DEVICE_IDS}"', run_script)
        self.assertIn('validate_positive_integer "JUDGE_TENSOR_PARALLEL_SIZE" "$JUDGE_TENSOR_PARALLEL_SIZE"', run_script)
        self.assertIn('validate_positive_integer "JUDGE_MAX_MODEL_LEN" "$JUDGE_MAX_MODEL_LEN"', run_script)
        self.assertIn('EVAL_COMMAND="evaluate-memory"', run_script)
        self.assertIn('MEMORY_EVAL_ARGS+=(--mem0-answer-retrieval-top-k "$MEM0_ANSWER_RETRIEVAL_TOP_K")', run_script)
        self.assertIn('MEMORY_EVAL_ARGS+=(--mem0-embedding-model "$MEM0_EMBEDDING_MODEL")', run_script)
        self.assertIn('MEMORY_EVAL_ARGS+=(--mem0-model-max-len "$MEM0_MODEL_MAX_LEN")', run_script)
        self.assertIn('Using evaluation command: $EVAL_COMMAND', run_script)
        self.assertIn('Using evaluation stage: $EVAL_STAGE', run_script)
        self.assertIn('Using memory embedding GPU device ids: $MEM0_EMBEDDING_GPU_DEVICE_IDS', run_script)
        self.assertIn('Dense memory retrieval requires torch and transformers', run_script)
        self.assertIn('CUDA_VISIBLE_DEVICES_OVERRIDE', run_script)
        self.assertIn('start_named_server', run_script)
        self.assertIn('stop_named_server', run_script)
        self.assertIn('run_answers_stage()', run_script)
        self.assertIn('run_judge_stage()', run_script)
        self.assertIn('--stage answers', run_script)
        self.assertIn('--stage judge', run_script)
        self.assertIn('python "$REPO_ROOT/main.py" "$EVAL_COMMAND"', run_script)
        self.assertIn('--base-url "http://127.0.0.1:${answer_port}/v1"', run_script)
        self.assertIn('--judge-base-url "http://127.0.0.1:${judge_port}/v1"', run_script)
        self.assertIn('--benchmark-root "$OUTPUT_ROOT"', run_script)
        self.assertIn('--evaluation-root "$EVALUATION_ROOT"', run_script)
        self.assertIn('--expected-model "$model"', run_script)
        self.assertIn('--pid-file "$vllm_pid_file"', run_script)
        self.assertIn('--patient-manifest "$PATIENT_MANIFEST"', run_script)
        self.assertIn('--models "$model"', run_script)
        self.assertIn('--models "$@"', run_script)
        self.assertIn('--retry-limit "$EVAL_RETRY_LIMIT"', run_script)
        self.assertIn('--timeout-seconds "$EVAL_TIMEOUT_SECONDS"', run_script)
        self.assertIn('"${MEMORY_EVAL_ARGS[@]}"', run_script)
        self.assertIn('judge_mem0.log', run_script)
        self.assertIn('quest_job_outputs', run_script)
        self.assertIn('patient_manifest_snapshot.txt', run_script)
        self.assertIn('successful_models+=("$MODEL")', run_script)
        self.assertIn('if [[ "$EVAL_STAGE" == "judge" ]]; then', run_script)
        self.assertIn('Skipping answer stage; judging existing artifacts for requested models.', run_script)
        self.assertIn('write_status_row "$MODEL_SLUG" "$MODEL" "existing" "pending" "judge_pending"', run_script)
        self.assertIn('if [[ "$EVAL_STAGE" == "full" || "$EVAL_STAGE" == "judge" ]]; then', run_script)
        self.assertIn('Skipping judge stage because EVAL_STAGE=$EVAL_STAGE.', run_script)
        self.assertIn('"eval_stage": eval_stage', run_script)
        self.assertIn('successful_statuses.add("answers_completed")', run_script)
        self.assertIn('run_judge_stage "$JUDGE_VLLM_PORT" "${successful_models[@]}"', run_script)
        self.assertIn('start_named_server "$MODEL" "$MODEL_SLUG" "$TENSOR_PARALLEL_SIZE" "$MAX_MODEL_LEN" "$ANSWER_VLLM_PORT" "$GPU_DEVICE_IDS"', run_script)
        self.assertIn('start_named_server "$JUDGE_MODEL" "$JUDGE_MODEL_SLUG" "$JUDGE_TENSOR_PARALLEL_SIZE" "$JUDGE_MAX_MODEL_LEN" "$JUDGE_VLLM_PORT" "$JUDGE_GPU_DEVICE_IDS"', run_script)
        self.assertNotIn('--base-url "http://127.0.0.1:${VLLM_PORT}/v1"', run_script)
        self.assertIn('MODEL="$model" \\', run_script)
        self.assertIn('MODEL_SLUG="$server_slug" \\', run_script)
        self.assertNotIn('export MODEL="$model"', run_script)
        self.assertNotIn('export MODEL_SLUG="$server_slug"', run_script)

        self.assertIn("--distributed-executor-backend ray", launch_script)
        self.assertIn('REASONING_PARSER="${REASONING_PARSER-qwen3}"', launch_script)
        self.assertNotIn('REASONING_PARSER="${REASONING_PARSER:-qwen3}"', launch_script)
        self.assertIn('ray start --head', launch_script)
        self.assertIn('ray start --address', launch_script)
        self.assertIn('launch_local_server()', launch_script)
        self.assertIn('exec "$SINGULARITY_BIN" exec --nv -B /projects:/projects "$VLLM_IMAGE" \\', launch_script)
        self.assertIn('( launch_local_server ) >"$VLLM_LOG" 2>&1 &', launch_script)
        self.assertIn('CUDA_VISIBLE_DEVICES_OVERRIDE="${CUDA_VISIBLE_DEVICES_OVERRIDE:-}"', launch_script)
        self.assertIn('CUDA_VISIBLE_DEVICES_OVERRIDE is not supported for distributed ray launches.', launch_script)
        self.assertIn('RAY_CLUSTER_PID_FILE', launch_script)
        self.assertIn('VLLM_PID_FILE', stop_script)
        self.assertIn('RAY_CLUSTER_PID_FILE', stop_script)
        self.assertIn('kill -9 "$pid"', stop_script)
        self.assertIn('for _ in $(seq 1 20); do', stop_script)
        self.assertIn('"/models"', wait_script)
        self.assertIn('"/health"', wait_script)
        self.assertIn("--expected-model", wait_script)
        self.assertIn("--pid-file", wait_script)
        self.assertIn("_process_exited_or_zombie", wait_script)
        self.assertIn('fields[2] == "Z"', wait_script)
        self.assertIn("http.client.BadStatusLine", wait_script)

    def test_wait_for_server_exits_when_pid_is_dead(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        wait_script = repo_root / "quest" / "wait_for_server.py"
        with tempfile.TemporaryDirectory() as temp_dir:
            pid_file = Path(temp_dir) / "vllm.pid"
            proc = subprocess.Popen(["/bin/true"])
            try:
                time.sleep(0.1)
                pid_file.write_text(f"{proc.pid}\n", encoding="utf-8")
                started = time.monotonic()
                result = subprocess.run(
                    [
                        sys.executable,
                        str(wait_script),
                        "--base-url",
                        "http://127.0.0.1:9/v1",
                        "--pid-file",
                        str(pid_file),
                        "--timeout-seconds",
                        "30",
                        "--sleep-seconds",
                        "1",
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
            finally:
                proc.wait(timeout=5)

        self.assertEqual(result.returncode, 1)
        self.assertLess(time.monotonic() - started, 5)

    def test_quest_evaluation_script_rejects_invalid_retry_and_timeout_args(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script_path = repo_root / "quest" / "evaluate_models.slurm"

        invalid_retry = subprocess.run(
            ["/bin/bash", str(script_path), "--retry-limit", "-1", "--patients", "11826927"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(invalid_retry.returncode, 2)
        self.assertIn("EVAL_RETRY_LIMIT must be a non-negative integer, got: -1", invalid_retry.stderr)

        invalid_timeout = subprocess.run(
            ["/bin/bash", str(script_path), "--timeout-seconds", "0", "--patients", "11826927"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(invalid_timeout.returncode, 2)
        self.assertIn("EVAL_TIMEOUT_SECONDS must be a positive integer, got: 0", invalid_timeout.stderr)

        invalid_stage = subprocess.run(
            ["/bin/bash", str(script_path), "--stage", "score", "--patients", "11826927"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(invalid_stage.returncode, 2)
        self.assertIn("EVAL_STAGE must be one of: full, answers, judge. Got: score", invalid_stage.stderr)

        invalid_judge_tp = subprocess.run(
            [
                "/bin/bash",
                str(script_path),
                "--stage",
                "judge",
                "--judge-tensor-parallel-size",
                "0",
                "--patients",
                "11826927",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(invalid_judge_tp.returncode, 2)
        self.assertIn(
            "JUDGE_TENSOR_PARALLEL_SIZE must be a positive integer, got: 0",
            invalid_judge_tp.stderr,
        )

        invalid_memory_arg = subprocess.run(
            [
                "/bin/bash",
                str(script_path),
                "--memory",
                "--mem0-answer-retrieval-top-k",
                "0",
                "--patients",
                "11826927",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(invalid_memory_arg.returncode, 2)
        self.assertIn(
            "MEM0_ANSWER_RETRIEVAL_TOP_K must be a positive integer, got: 0",
            invalid_memory_arg.stderr,
        )

    def test_quest_evaluation_shell_scripts_have_valid_syntax(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        for relative_path in (
            "quest/evaluate_models.slurm",
            "quest/run_multi_patient_eval_job.sh",
            "quest/launch_vllm_server.sh",
            "quest/stop_vllm_server.sh",
        ):
            completed = subprocess.run(
                ["/bin/bash", "-n", str(repo_root / relative_path)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, f"{relative_path}: {completed.stderr}")

    def test_main_generate_qa_uses_fixed_harder_policy(self) -> None:
        fake_pipeline = Mock()
        fake_pipeline.generate_patient_qa.return_value = {
            "subject_id": "777",
            "processed_admissions": 2,
            "total_qas": 12,
            "patient_dir": "/tmp/777",
        }

        with patch.object(main_cli, "build_default_config", return_value=self._config()), patch.object(
            main_cli,
            "BenchmarkPipeline",
            return_value=fake_pipeline,
        ):
            exit_code = main_cli.main(
                [
                    "generate-qa",
                    "--subject-id",
                    "777",
                ]
            )

        self.assertEqual(exit_code, 0)
        fake_pipeline.generate_patient_qa.assert_called_once()
        self.assertNotIn("cross_admission_qa_count", fake_pipeline.generate_patient_qa.call_args.kwargs)
        self.assertNotIn("single_admission_qa_count", fake_pipeline.generate_patient_qa.call_args.kwargs)
        fake_pipeline.close.assert_called_once()

    def test_main_generate_patient_accepts_provider_flags_and_base_url_alias(self) -> None:
        fake_pipeline = Mock()
        fake_pipeline.generate_patient_sample.return_value = {
            "subject_id": "100",
            "processed_admissions": 2,
        }
        captured: dict[str, object] = {}

        def pipeline_factory(config):
            captured["config"] = config
            return fake_pipeline

        with patch.object(main_cli, "build_default_config", return_value=self._config()), patch.object(
            main_cli,
            "BenchmarkPipeline",
            side_effect=pipeline_factory,
        ):
            exit_code = main_cli.main(
                [
                    "generate-patient",
                    "--subject-id",
                    "100",
                    "--provider",
                    "vllm",
                    "--openai-base-url",
                    "http://127.0.0.1:9000/v1",
                ]
            )

        self.assertEqual(exit_code, 0)
        config = captured["config"]
        self.assertEqual(config.llm.provider, "vllm")  # type: ignore[attr-defined]
        self.assertEqual(config.llm.base_url, "http://127.0.0.1:9000/v1")  # type: ignore[attr-defined]
        fake_pipeline.close.assert_called_once()

    def test_main_generate_all_returns_nonzero_when_batch_has_failures(self) -> None:
        fake_pipeline = Mock()
        fake_pipeline.generate_all.return_value = {
            "requested_subject_ids": [100, 200],
            "succeeded": [100],
            "failed": [200],
            "batch_summary_path": "/tmp/batch_summary.json",
        }

        with patch.object(main_cli, "build_default_config", return_value=self._config()), patch.object(
            main_cli,
            "BenchmarkPipeline",
            return_value=fake_pipeline,
        ):
            exit_code = main_cli.main(
                [
                    "generate-all",
                    "--subject-ids",
                    "100",
                    "200",
                ]
            )

        self.assertEqual(exit_code, 1)
        fake_pipeline.generate_all.assert_called_once_with(
            subject_ids=[100, 200],
            model_name=None,
            max_admissions=None,
            fail_fast=False,
        )
        fake_pipeline.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
