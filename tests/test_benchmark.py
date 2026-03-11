from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from health_benchmark.scripts import BenchmarkPipeline, build_default_config
from health_benchmark.scripts.llm_client import LLMCallResult
from health_benchmark.scripts.prompting import render_prompt
from health_benchmark.scripts.qa_prompting import (
    render_cross_admission_qa_prompt,
    render_single_admission_qa_prompt,
)
from health_benchmark.scripts.qa_validation import (
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

    def generate_structured_response(self, _system_message: str, _user_message: str, _response_schema) -> LLMCallResult:
        return self._pop_result()


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

    def _make_single_qa_payload(self, hadm_id: str, count: int = 12) -> dict[str, object]:
        qas: list[dict[str, object]] = []
        for index in range(1, count + 1):
            qas.append(
                {
                    "qa_id": f"raw_{index}",
                    "scope": "single_admission",
                    "question_type": "medical_reasoning",
                    "question": f"Why did the team make decision {index}?",
                    "answer": f"reason {index}",
                    "evidence": {
                        "admissions": [hadm_id],
                        "turn_ids": [2, 1, 2],
                    },
                }
            )
        return {"qas": qas}

    def _make_cross_qa_payload(self, admissions: list[str], count: int = 50) -> dict[str, object]:
        evidence_admissions = admissions[: min(3, len(admissions))]
        qas: list[dict[str, object]] = []
        for index in range(1, count + 1):
            qas.append(
                {
                    "qa_id": f"cross_{index}",
                    "scope": "cross_admission",
                    "question_type": "recurrence_pattern",
                    "question": f"Which pattern recurred over admissions {index}?",
                    "answer": f"pattern {index}",
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
        staged_root = repo_root / "output" / "_tmp" / "11826927"
        if not staged_root.exists():
            staged_root = repo_root / "output" / "11826927"
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
            )

    def test_validate_single_admission_qa_adds_multi_day_question_prefix(self) -> None:
        validated = validate_single_admission_qa(
            self._make_single_qa_payload("10", count=1),
            subject_id="100",
            hadm_id="10",
            admission_start="2020-01-01 08:00:00",
            admission_end="2020-01-02 10:00:00",
            valid_turn_ids={1, 2},
            expected_count=1,
        )

        self.assertEqual(
            validated["qas"][0]["question"],
            "During the hospitalization from 2020-01-01 to 2020-01-02, Why did the team make decision 1?",
        )

    def test_validate_single_admission_qa_uses_same_day_question_prefix(self) -> None:
        payload = self._make_single_qa_payload("10", count=1)
        payload["qas"][0]["question"] = "Why was the patient monitored closely?"

        validated = validate_single_admission_qa(
            payload,
            subject_id="100",
            hadm_id="10",
            admission_start="2020-01-01 08:00:00",
            admission_end="2020-01-01 18:00:00",
            valid_turn_ids={1, 2},
            expected_count=1,
        )

        self.assertEqual(
            validated["qas"][0]["question"],
            "During the hospitalization on 2020-01-01, Why was the patient monitored closely?",
        )

    def test_validate_single_admission_qa_normalizes_existing_leading_anchor(self) -> None:
        payload = self._make_single_qa_payload("10", count=1)
        payload["qas"][0]["question"] = "During this hospitalization, why was oxygen continued?"

        validated = validate_single_admission_qa(
            payload,
            subject_id="100",
            hadm_id="10",
            admission_start="2020-01-01 08:00:00",
            admission_end="2020-01-02 10:00:00",
            valid_turn_ids={1, 2},
            expected_count=1,
        )

        self.assertEqual(
            validated["qas"][0]["question"],
            "During the hospitalization from 2020-01-01 to 2020-01-02, why was oxygen continued?",
        )

    def test_validate_single_admission_qa_preserves_canonical_prefix(self) -> None:
        payload = self._make_single_qa_payload("10", count=1)
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
                    "question_type": "recurrence_pattern",
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
            )

    def test_validate_cross_admission_qa_normalizes_admission_order(self) -> None:
        payload = {
            "qas": [
                {
                    "qa_id": "x",
                    "scope": "cross_admission",
                    "question_type": "recurrence_pattern",
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
        )
        self.assertEqual(validated["qas"][0]["evidence"]["admissions"], ["10", "12"])

    def test_validate_cross_admission_qa_normalizes_time_range_aliases(self) -> None:
        payload = {
            "qas": [
                {
                    "qa_id": "x",
                    "scope": "cross_admission",
                    "question_type": "recurrence_pattern",
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
            admission_aliases={
                "2020-01-01 08:00:00 to 2020-01-02 10:00:00": "10",
                "2020-03-01 08:00:00 to 2020-03-02 09:00:00": "12",
            },
        )
        self.assertEqual(validated["qas"][0]["evidence"]["admissions"], ["10", "12"])

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
        rendered_single = render_single_admission_qa_prompt(conversation, question_count=12)
        rendered_cross = render_cross_admission_qa_prompt(
            [
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
            ],
            question_count=50,
        )

        self.assertEqual(rendered_single.question_count, 12)
        self.assertIn('"conversation_lines"', rendered_single.context_json)
        self.assertIn("Generate 12 hard question-answer pairs", rendered_single.user_message)
        self.assertIn("Questions are written for benchmark users, not for the patient.", rendered_single.user_message)
        self.assertIn("Do not use second-person wording like 'you' or 'your' in the question.", rendered_single.user_message)
        self.assertIn(
            "Use third-person phrasing such as 'the patient', 'the patient's symptoms', or 'the doctor' when needed.",
            rendered_single.user_message,
        )
        self.assertIn("Do not mention raw identifiers in the question.", rendered_single.user_message)
        self.assertIn(
            "During the hospitalization from YYYY-MM-DD to YYYY-MM-DD, ...",
            rendered_single.user_message,
        )
        self.assertIn(
            "During the hospitalization on YYYY-MM-DD, ...",
            rendered_single.user_message,
        )
        self.assertEqual(rendered_cross.question_count, 50)
        self.assertIn('"summary_paragraph"', rendered_cross.context_json)
        self.assertIn('"admission_id_for_evidence_only"', rendered_cross.context_json)
        self.assertNotIn('"conversation_lines"', rendered_cross.context_json)
        self.assertIn("Generate 50 hard cross-admission question-answer pairs", rendered_cross.user_message)
        self.assertIn("should not mention raw identifiers", rendered_cross.user_message)
        self.assertNotIn(
            "During the hospitalization from YYYY-MM-DD to YYYY-MM-DD, ...",
            rendered_cross.user_message,
        )

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
                    "parsed_output": self._make_single_qa_payload("10", count=2),
                    "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
                },
                {
                    "parsed_output": self._make_single_qa_payload("12", count=2),
                    "usage": {"input_tokens": 11, "output_tokens": 21, "total_tokens": 32},
                },
                {
                    "parsed_output": self._make_cross_qa_payload(["10", "12"], count=50),
                    "usage": {"input_tokens": 12, "output_tokens": 22, "total_tokens": 34},
                },
            ]
        )

        summary = pipeline.generate_patient_qa(subject_id=777, single_admission_qa_count=2)

        self.assertEqual(summary["processed_admissions"], 2)
        self.assertEqual(summary["single_admission_qa_count"], 2)
        self.assertEqual(summary["cross_admission_qa_count"], 50)
        self.assertEqual(summary["total_qas"], 54)
        admission_10_qas = json.loads((patient_root / "10" / "qa.json").read_text(encoding="utf-8"))
        admission_12_qas = json.loads((patient_root / "12" / "qa.json").read_text(encoding="utf-8"))
        cross_qas = json.loads((patient_root / "cross_admission_qa.json").read_text(encoding="utf-8"))
        benchmark_qas = json.loads((patient_root / "benchmark_qa.json").read_text(encoding="utf-8"))
        self.assertEqual(admission_10_qas["qas"][0]["qa_id"], "777_10_q01")
        self.assertEqual(admission_10_qas["qas"][0]["evidence"]["turn_ids"], [1, 2])
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
        self.assertEqual(len(cross_qas["qas"]), 50)
        self.assertEqual(cross_qas["qas"][0]["qa_id"], "777_cross_q01")
        self.assertEqual(cross_qas["qas"][0]["question"], "Which pattern recurred over admissions 1?")
        self.assertEqual(len(benchmark_qas["qas"]), 54)
        self.assertEqual(benchmark_qas["qas"][0]["qa_id"], "777_10_q01")
        self.assertEqual(benchmark_qas["qas"][2]["qa_id"], "777_12_q01")
        self.assertEqual(benchmark_qas["qas"][-1]["qa_id"], "777_cross_q50")
        self.assertTrue(
            benchmark_qas["qas"][0]["question"].startswith(
                "During the hospitalization from 2020-01-01 to 2020-01-02, "
            )
        )
        self.assertEqual(benchmark_qas["qas"][-1]["question"], "Which pattern recurred over admissions 50?")
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
                    "parsed_output": self._make_single_qa_payload("10", count=2),
                    "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
                },
                {
                    "parsed_output": self._make_single_qa_payload("12", count=2),
                    "usage": {"input_tokens": 11, "output_tokens": 21, "total_tokens": 32},
                },
                {
                    "parsed_output": {
                        "qas": [
                            {
                                "qa_id": "bad",
                                "scope": "cross_admission",
                                "question_type": "recurrence_pattern",
                                "question": "Which issue recurred?",
                                "answer": "heart failure",
                                "evidence": {"admissions": ["10"]},
                            }
                        ]
                    },
                    "usage": {"input_tokens": 12, "output_tokens": 22, "total_tokens": 34},
                },
            ]
        )

        with self.assertRaises(QAValidationError):
            pipeline.generate_patient_qa(subject_id=888, single_admission_qa_count=2)

        self.assertEqual(json.loads((patient_root / "10" / "qa.json").read_text(encoding="utf-8")), old_single)
        self.assertEqual(json.loads((patient_root / "12" / "qa.json").read_text(encoding="utf-8")), old_single)
        self.assertEqual(json.loads((patient_root / "cross_admission_qa.json").read_text(encoding="utf-8")), old_cross)
        self.assertEqual(json.loads((patient_root / "benchmark_qa.json").read_text(encoding="utf-8")), old_benchmark)
        self.assertFalse((self.output_dir / "_tmp" / "qa_888").exists())


if __name__ == "__main__":
    unittest.main()
