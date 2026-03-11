from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _import_duckdb():
    try:
        import duckdb  # type: ignore
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("duckdb is required. Install dependencies in requirements.txt") from exc
    return duckdb


def _sql_quote_path(path: Path) -> str:
    return str(path).replace("'", "''")


def _find_table_file(directory: Path, stem: str) -> Path:
    candidates = [directory / f"{stem}.csv", directory / f"{stem}.csv.gz"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing required CSV: {candidates[0]}")


@dataclass
class TablePaths:
    hosp_dir: Path
    note_dir: Path


class MimicDuckDBStore:
    """Thin wrapper around DuckDB views over the required MIMIC tables."""

    def __init__(self, hosp_dir: Path, note_dir: Path) -> None:
        self.table_paths = TablePaths(hosp_dir=hosp_dir, note_dir=note_dir)
        self._conn = None
        self._cached_subject_id: int | None = None
        self._diagnosis_titles_cache: set[str] | None = None
        self._procedure_titles_cache: set[str] | None = None
        self._microbiology_terms_cache: set[str] | None = None

    @property
    def conn(self):
        if self._conn is None:
            duckdb = _import_duckdb()
            self._conn = duckdb.connect(database=":memory:")
            self._conn.execute("PRAGMA threads=4;")
            self._create_views()
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        self._cached_subject_id = None

    def _create_views(self) -> None:
        views = {
            "hosp_admissions": _find_table_file(self.table_paths.hosp_dir, "admissions"),
            "hosp_diagnoses_icd": _find_table_file(self.table_paths.hosp_dir, "diagnoses_icd"),
            "hosp_d_icd_diagnoses": _find_table_file(self.table_paths.hosp_dir, "d_icd_diagnoses"),
            "hosp_procedures_icd": _find_table_file(self.table_paths.hosp_dir, "procedures_icd"),
            "hosp_d_icd_procedures": _find_table_file(self.table_paths.hosp_dir, "d_icd_procedures"),
            "hosp_microbiologyevents": _find_table_file(self.table_paths.hosp_dir, "microbiologyevents"),
            "note_discharge": _find_table_file(self.table_paths.note_dir, "discharge"),
            "note_radiology": _find_table_file(self.table_paths.note_dir, "radiology"),
        }
        view_options = {
            "note_discharge": ", all_varchar=true",
            "note_radiology": ", all_varchar=true",
        }

        for view_name, csv_path in views.items():
            self.conn.execute(
                "CREATE OR REPLACE VIEW "
                f"{view_name} AS SELECT * FROM read_csv_auto("
                f"'{_sql_quote_path(csv_path)}', header=true, strict_mode=false, null_padding=true, parallel=false"
                f"{view_options.get(view_name, '')});"
            )

    def prepare_subject_cache(self, subject_id: int) -> None:
        if self._cached_subject_id == int(subject_id):
            return
        sid = int(subject_id)
        conn = self.conn
        subject_scoped_tables = [
            "hosp_admissions",
            "hosp_diagnoses_icd",
            "hosp_procedures_icd",
            "hosp_microbiologyevents",
            "note_discharge",
            "note_radiology",
        ]
        for table_name in subject_scoped_tables:
            where_clause = "subject_id = ?"
            if table_name.startswith("note_"):
                where_clause = "TRY_CAST(subject_id AS BIGINT) = ?"
            conn.execute(
                f"CREATE OR REPLACE TEMP TABLE {table_name} AS "
                f"SELECT * FROM main.{table_name} WHERE {where_clause}",
                [sid],
            )
        self._cached_subject_id = sid

    def fetch_rows(self, sql: str, params: list[Any] | tuple[Any, ...] | None = None) -> list[dict[str, Any]]:
        cur = self.conn.execute(sql, params or [])
        cols = [desc[0] for desc in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def fetch_one(self, sql: str, params: list[Any] | tuple[Any, ...] | None = None) -> dict[str, Any] | None:
        rows = self.fetch_rows(sql, params)
        if not rows:
            return None
        return rows[0]

    def fetch_all_diagnosis_titles(self) -> set[str]:
        if self._diagnosis_titles_cache is None:
            rows = self.fetch_rows(
                """
                SELECT DISTINCT lower(trim(long_title)) AS title
                FROM main.hosp_d_icd_diagnoses
                WHERE long_title IS NOT NULL AND trim(long_title) <> ''
                """
            )
            self._diagnosis_titles_cache = {str(row["title"]) for row in rows}
        return self._diagnosis_titles_cache

    def fetch_all_procedure_titles(self) -> set[str]:
        if self._procedure_titles_cache is None:
            rows = self.fetch_rows(
                """
                SELECT DISTINCT lower(trim(long_title)) AS title
                FROM main.hosp_d_icd_procedures
                WHERE long_title IS NOT NULL AND trim(long_title) <> ''
                """
            )
            self._procedure_titles_cache = {str(row["title"]) for row in rows}
        return self._procedure_titles_cache

    def fetch_all_microbiology_terms(self) -> set[str]:
        if self._microbiology_terms_cache is None:
            rows = self.fetch_rows(
                """
                SELECT DISTINCT lower(trim(term)) AS term
                FROM (
                  SELECT org_name AS term FROM main.hosp_microbiologyevents
                  UNION ALL
                  SELECT spec_type_desc AS term FROM main.hosp_microbiologyevents
                ) terms
                WHERE term IS NOT NULL AND trim(term) <> ''
                """
            )
            self._microbiology_terms_cache = {str(row["term"]) for row in rows}
        return self._microbiology_terms_cache
