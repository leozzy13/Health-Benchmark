from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_VLLM_BASE_URL = "http://127.0.0.1:8000/v1"


def _import_yaml():
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("PyYAML is required. Install dependencies in requirements.txt") from exc
    return yaml


@dataclass
class DatasetConfig:
    mimiciv_hosp_path: Path
    mimiciv_note_path: Path


@dataclass
class OutputConfig:
    root: Path


@dataclass
class SelectionConfig:
    subject_id: int | None
    max_admissions: int | None


@dataclass
class LLMConfig:
    provider: str
    model: str
    temperature: float
    max_output_tokens: int
    timeout_seconds: int
    max_retries: int
    api_key_env: str
    base_url: str | None
    tokenizer_name: str | None


OpenAIConfig = LLMConfig


@dataclass
class VLLMConfig:
    tensor_parallel_size: int
    max_model_len: int
    reasoning_parser: str | None
    language_model_only: bool
    enable_thinking: bool
    top_k: int | None


@dataclass
class RuntimeConfig:
    use_duckdb: bool
    save_raw_response: bool
    replace_existing_patient_output: bool
    staging_dir_name: str = "_tmp"


@dataclass
class PathsConfig:
    repo_root: Path
    project_dir: Path
    config_path: Path


@dataclass
class BenchmarkConfig:
    benchmark_name: str
    benchmark_version: str
    packet_schema_version: str
    dataset_versions: dict[str, str]
    dataset: DatasetConfig
    output: OutputConfig
    selection: SelectionConfig
    llm: LLMConfig
    vllm: VLLMConfig
    runtime: RuntimeConfig
    paths: PathsConfig

    @property
    def openai(self) -> LLMConfig:
        return self.llm


def build_default_config(project_dir: Path) -> BenchmarkConfig:
    project_dir = project_dir.resolve()
    repo_root = project_dir.parent
    config_path = repo_root / "config.yaml"
    raw = _load_yaml_config(config_path)

    data_dir = repo_root / "data"
    default_hosp = data_dir / "mimic-iv" / "hosp"
    default_note = data_dir / "mimic-iv-notes"
    default_output = repo_root / "output" / "benchmark"

    dataset_cfg = raw.get("dataset", {})
    output_cfg = raw.get("output", {})
    selection_cfg = raw.get("selection", {})
    llm_cfg = raw.get("llm", {})
    openai_cfg = raw.get("openai", {})
    vllm_cfg = raw.get("vllm", {})
    runtime_cfg = raw.get("runtime", {})

    if not isinstance(dataset_cfg, dict):
        raise RuntimeError("Config section 'dataset' must be a YAML object.")
    if not isinstance(output_cfg, dict):
        raise RuntimeError("Config section 'output' must be a YAML object.")
    if not isinstance(selection_cfg, dict):
        raise RuntimeError("Config section 'selection' must be a YAML object.")
    if not isinstance(llm_cfg, dict):
        raise RuntimeError("Config section 'llm' must be a YAML object.")
    if not isinstance(openai_cfg, dict):
        raise RuntimeError("Config section 'openai' must be a YAML object.")
    if not isinstance(vllm_cfg, dict):
        raise RuntimeError("Config section 'vllm' must be a YAML object.")
    if not isinstance(runtime_cfg, dict):
        raise RuntimeError("Config section 'runtime' must be a YAML object.")

    hosp_path = _resolve_path(
        os.getenv("MIMIC_IV_HOSP_PATH")
        or _normalize_mimiciv_root_env(os.getenv("MIMIC_IV_DIR"))
        or dataset_cfg.get("mimiciv_hosp_path")
        or str(default_hosp),
        repo_root,
    )
    note_path = _resolve_path(
        os.getenv("MIMIC_IV_NOTE_PATH")
        or os.getenv("MIMIC_IV_NOTE_DIR")
        or dataset_cfg.get("mimiciv_note_path")
        or str(default_note),
        repo_root,
    )
    output_root = _resolve_path(
        os.getenv("MEDBENCH_OUTPUT_ROOT") or output_cfg.get("root") or str(default_output),
        repo_root,
    )
    llm_provider = str(llm_cfg.get("provider", "openai") or "openai").strip().lower()
    llm_base_url = resolve_llm_base_url(
        llm_provider,
        _coerce_optional_str(llm_cfg.get("base_url", openai_cfg.get("base_url"))),
    )

    return BenchmarkConfig(
        benchmark_name=str(raw.get("benchmark_name", "mimic_longctx")),
        benchmark_version=str(raw.get("benchmark_version", "0.2.0")),
        packet_schema_version=str(raw.get("packet_schema_version", "0.2.0")),
        dataset_versions={
            "mimiciv": str(raw.get("dataset_versions", {}).get("mimiciv", "3.1")),
            "mimiciv_note": str(raw.get("dataset_versions", {}).get("mimiciv_note", "2.2")),
        },
        dataset=DatasetConfig(
            mimiciv_hosp_path=hosp_path,
            mimiciv_note_path=note_path,
        ),
        output=OutputConfig(root=output_root),
        selection=SelectionConfig(
            subject_id=_coerce_optional_int(selection_cfg.get("subject_id")),
            max_admissions=_coerce_optional_int(selection_cfg.get("max_admissions")),
        ),
        llm=LLMConfig(
            provider=llm_provider,
            model=str(llm_cfg.get("model", openai_cfg.get("model", "gpt-5.2"))),
            temperature=float(llm_cfg.get("temperature", openai_cfg.get("temperature", 0.2))),
            max_output_tokens=int(llm_cfg.get("max_output_tokens", openai_cfg.get("max_output_tokens", 8000))),
            timeout_seconds=int(llm_cfg.get("timeout_seconds", openai_cfg.get("timeout_seconds", 300))),
            max_retries=int(llm_cfg.get("max_retries", openai_cfg.get("max_retries", 1))),
            api_key_env=str(llm_cfg.get("api_key_env", openai_cfg.get("api_key_env", "OPENAI_API_KEY"))),
            base_url=llm_base_url,
            tokenizer_name=_coerce_optional_str(llm_cfg.get("tokenizer_name")),
        ),
        vllm=VLLMConfig(
            tensor_parallel_size=int(vllm_cfg.get("tensor_parallel_size", 4)),
            max_model_len=int(vllm_cfg.get("max_model_len", 65536)),
            reasoning_parser=_coerce_optional_str(vllm_cfg.get("reasoning_parser", "qwen3")),
            language_model_only=bool(vllm_cfg.get("language_model_only", True)),
            enable_thinking=bool(vllm_cfg.get("enable_thinking", False)),
            top_k=_coerce_optional_int(vllm_cfg.get("top_k", 20)),
        ),
        runtime=RuntimeConfig(
            use_duckdb=bool(runtime_cfg.get("use_duckdb", True)),
            save_raw_response=bool(runtime_cfg.get("save_raw_response", True)),
            replace_existing_patient_output=bool(runtime_cfg.get("replace_existing_patient_output", True)),
        ),
        paths=PathsConfig(
            repo_root=repo_root,
            project_dir=project_dir,
            config_path=config_path,
        ),
    )


def _load_yaml_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        return {}
    yaml = _import_yaml()
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise RuntimeError(f"Config file must contain a YAML object: {config_path}")
    return data


def _resolve_path(value: str | os.PathLike[str], repo_root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    return path.resolve()


def resolve_llm_base_url(provider: str, base_url: str | None) -> str | None:
    normalized_provider = str(provider or "openai").strip().lower()
    if normalized_provider == "vllm":
        return base_url or DEFAULT_VLLM_BASE_URL
    return base_url


def _normalize_mimiciv_root_env(value: str | None) -> str | None:
    if not value:
        return None
    base = Path(value).expanduser()
    hosp_dir = base / "hosp"
    if hosp_dir.exists():
        return str(hosp_dir)
    return str(base)


def _coerce_optional_int(value: Any) -> int | None:
    if value in (None, "", "null"):
        return None
    return int(value)


def _coerce_optional_str(value: Any) -> str | None:
    if value in (None, "", "null"):
        return None
    return str(value)
