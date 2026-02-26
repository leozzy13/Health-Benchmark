from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class DatasetVersions:
    mimiciv: str = "3.1"
    mimiciv_note: str = "2.2"


@dataclass(slots=True)
class NullHandlingPolicy:
    labevents_hadm_id_null: str = "PROXIMAL_TIME_JOIN"
    microbiologyevents_hadm_id_null: str = "PROXIMAL_TIME_JOIN"
    emar_hadm_id_null: str = "LINK_VIA_PHARMACY_OR_TIME_WINDOW"
    radiology_hadm_id_null: str = "LOG_AND_EXCLUDE_BY_DEFAULT"


@dataclass(slots=True)
class TruncationConfig:
    ruleset_id: str = "trunc.v1"
    # `None` means uncapped.
    per_section_row_caps: dict[str, int | None] = field(
        default_factory=lambda: {
            "transfers": None,
            "services": None,
            "discharge": None,
            "discharge_detail": None,
            "radiology": None,
            "radiology_detail": None,
            "labs": 800,
            "microbiology": 200,
            "poe": 400,
            "poe_detail": 800,
            "prescriptions": 400,
            "pharmacy": 400,
            "emar": 600,
            "emar_detail": 1200,
            "diagnoses_icd": 80,
            "procedures_icd": 80,
            "drgcodes": 20,
            "icustays": 20,
        }
    )


@dataclass(slots=True)
class ExtractionConfig:
    proximal_lab_capture: bool = True
    proximal_micro_capture: bool = True
    proximal_padding_hours: int = 0
    include_icu_stays: bool = True
    include_omr_baseline: bool = False
    require_discharge_note: bool = True
    only_admissions_with_discharge: bool = True
    include_unlinked_radiology_in_sidecar: bool = True
    emar_time_window_fallback: bool = True
    stable_order_admissions_by: str = "admittime ASC, hadm_id ASC"


@dataclass(slots=True)
class PromptConfig:
    template_version: str = "prompt.v1.0"
    delimiters: dict[str, str] = field(
        default_factory=lambda: {
            "metadata": "<<BENCHMARK_METADATA>>",
            "metadata_end": "<<END_BENCHMARK_METADATA>>",
            "prev_summary": "<<PREVIOUS_ADMISSION_SUMMARY>>",
            "prev_summary_end": "<<END_PREVIOUS_ADMISSION_SUMMARY>>",
            "ehr_json": "<<EHR_PACKET_JSON>>",
            "ehr_json_end": "<<END_EHR_PACKET_JSON>>",
            "task": "<<TASK>>",
            "task_end": "<<END_TASK>>",
        }
    )


@dataclass(slots=True)
class ModelConfig:
    provider: str = "openai"
    name: str = "gpt-4.1-mini"
    temperature: float = 0.0
    max_output_tokens: int = 12000
    reasoning_effort: str | None = None
    seed: int | None = None
    retry_limit: int = 2
    timeout_seconds: int = 180
    api_key_env: str = "OPENAI_API_KEY"


@dataclass(slots=True)
class PathsConfig:
    project_dir: Path
    mimiciv_dir: Path
    mimiciv_note_dir: Path
    output_root: Path

    def as_serializable(self) -> dict[str, str]:
        return {
            "project_dir": str(self.project_dir),
            "mimiciv_dir": str(self.mimiciv_dir),
            "mimiciv_note_dir": str(self.mimiciv_note_dir),
            "output_root": str(self.output_root),
        }


@dataclass(slots=True)
class BenchmarkConfig:
    benchmark_name: str = "mimic_longctx"
    benchmark_version: str = "0.1.0"
    packet_schema_version: str = "0.1.0"
    manifest_schema_version: str = "0.1.0"
    dataset_versions: DatasetVersions = field(default_factory=DatasetVersions)
    null_handling: NullHandlingPolicy = field(default_factory=NullHandlingPolicy)
    truncation: TruncationConfig = field(default_factory=TruncationConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    paths: PathsConfig | None = None
    strict_validation: bool = True

    def to_log_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.paths is not None:
            data["paths"] = self.paths.as_serializable()
        return data


def build_default_config(project_dir: Path) -> BenchmarkConfig:
    project_dir = project_dir.resolve()
    repo_root = project_dir.parent
    data_dir = repo_root / "data"
    cfg = BenchmarkConfig()
    cfg.paths = PathsConfig(
        project_dir=project_dir,
        mimiciv_dir=(data_dir / "mimic-iv").resolve(),
        mimiciv_note_dir=(data_dir / "mimic-iv-notes").resolve(),
        output_root=(project_dir / "output").resolve(),
    )
    return cfg
