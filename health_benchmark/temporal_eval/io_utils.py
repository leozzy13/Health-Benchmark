from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any, Iterable

from ..scripts.utils import ensure_dir, write_json, write_text
from .types import ModelArtifactPaths


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def replace_dir(path: Path, *, replace_existing: bool) -> None:
    if path.exists():
        if not replace_existing:
            raise RuntimeError(f"Evaluation output already exists: {path}")
        shutil.rmtree(path)
    ensure_dir(path.parent)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not records:
        path.write_text("", encoding="utf-8")
        return
    rendered = "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n"
    write_text(path, rendered)


def write_csv(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def write_model_outputs(
    paths: ModelArtifactPaths,
    *,
    replace_existing: bool,
    run_config: dict[str, Any],
    question_batches: dict[str, Any],
    raw_predictions: list[dict[str, Any]],
    scored_predictions: list[dict[str, Any]],
    summary_payload: dict[str, Any],
    error_records: list[dict[str, Any]],
) -> None:
    staging_dir = paths.model_dir.parent / f".tmp_{paths.model_dir.name}"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    ensure_dir(staging_dir)
    staged = ModelArtifactPaths(
        model_dir=staging_dir,
        run_config_json=staging_dir / "run_config.json",
        question_batches_json=staging_dir / "question_batches.json",
        raw_predictions_jsonl=staging_dir / "raw_predictions.jsonl",
        scored_predictions_jsonl=staging_dir / "scored_predictions.jsonl",
        summary_json=staging_dir / "summary.json",
        errors_jsonl=staging_dir / "errors.jsonl",
    )
    write_json(staged.run_config_json, run_config)
    write_json(staged.question_batches_json, question_batches)
    write_jsonl(staged.raw_predictions_jsonl, raw_predictions)
    write_jsonl(staged.scored_predictions_jsonl, scored_predictions)
    write_json(staged.summary_json, summary_payload)
    write_jsonl(staged.errors_jsonl, error_records)
    replace_dir(paths.model_dir, replace_existing=replace_existing)
    staged.model_dir.rename(paths.model_dir)


def list_existing_model_summary_paths(evaluation_root: Path) -> list[Path]:
    summary_paths: list[Path] = []
    if not evaluation_root.exists():
        return summary_paths
    for child in sorted(evaluation_root.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith(".") or child.name == "comparison":
            continue
        summary_path = child / "summary.json"
        if summary_path.exists():
            summary_paths.append(summary_path)
    return summary_paths
