from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _import_duckdb():
    try:
        import duckdb  # type: ignore
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "duckdb is required. Install dependencies in medical_benchmark/requirements.txt"
        ) from exc
    return duckdb


def _sql_quote_path(path: Path) -> str:
    return str(path).replace("'", "''")


@dataclass(slots=True)
class TablePaths:
    hosp_dir: Path
    icu_dir: Path
    note_dir: Path


class MimicDuckDBStore:
    """Thin wrapper around DuckDB views over CSV files."""

    def __init__(self, hosp_dir: Path, icu_dir: Path, note_dir: Path) -> None:
        self.table_paths = TablePaths(hosp_dir=hosp_dir, icu_dir=icu_dir, note_dir=note_dir)
        self._conn = None
        self._cached_subject_id: int | None = None

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
            # hosp
            "hosp_admissions": self.table_paths.hosp_dir / "admissions.csv",
            "hosp_patients": self.table_paths.hosp_dir / "patients.csv",
            "hosp_transfers": self.table_paths.hosp_dir / "transfers.csv",
            "hosp_services": self.table_paths.hosp_dir / "services.csv",
            "hosp_labevents": self.table_paths.hosp_dir / "labevents.csv",
            "hosp_d_labitems": self.table_paths.hosp_dir / "d_labitems.csv",
            "hosp_microbiologyevents": self.table_paths.hosp_dir / "microbiologyevents.csv",
            "hosp_poe": self.table_paths.hosp_dir / "poe.csv",
            "hosp_poe_detail": self.table_paths.hosp_dir / "poe_detail.csv",
            "hosp_prescriptions": self.table_paths.hosp_dir / "prescriptions.csv",
            "hosp_pharmacy": self.table_paths.hosp_dir / "pharmacy.csv",
            "hosp_emar": self.table_paths.hosp_dir / "emar.csv",
            "hosp_emar_detail": self.table_paths.hosp_dir / "emar_detail.csv",
            "hosp_diagnoses_icd": self.table_paths.hosp_dir / "diagnoses_icd.csv",
            "hosp_d_icd_diagnoses": self.table_paths.hosp_dir / "d_icd_diagnoses.csv",
            "hosp_procedures_icd": self.table_paths.hosp_dir / "procedures_icd.csv",
            "hosp_d_icd_procedures": self.table_paths.hosp_dir / "d_icd_procedures.csv",
            "hosp_drgcodes": self.table_paths.hosp_dir / "drgcodes.csv",
            "hosp_omr": self.table_paths.hosp_dir / "omr.csv",
            # optional provider table exists in provided data
            "hosp_provider": self.table_paths.hosp_dir / "provider.csv",
            # icu
            "icu_icustays": self.table_paths.icu_dir / "icustays.csv",
            # notes
            "note_discharge": self.table_paths.note_dir / "discharge.csv",
            "note_discharge_detail": self.table_paths.note_dir / "discharge_detail.csv",
            "note_radiology": self.table_paths.note_dir / "radiology.csv",
            "note_radiology_detail": self.table_paths.note_dir / "radiology_detail.csv",
        }
        optional_views = {"hosp_provider"}
        # Some MIMIC CSVs contain mixed text/numeric values in the same column.
        # DuckDB's sampled auto-detection can infer a numeric type and later fail.
        view_read_options = {
            # pharmacy.csv has multiple mixed-type columns (numeric/text variants).
            # Use all-varchar for this table to avoid late conversion failures.
            "hosp_pharmacy": ", all_varchar=true, strict_mode=false, null_padding=true",
            # emar_detail.csv also has mixed numeric/text rate fields (e.g. "100-500").
            # Read as varchar to avoid conversion crashes during extraction.
            "hosp_emar_detail": ", all_varchar=true, strict_mode=false, null_padding=true",
        }

        for view_name, csv_path in views.items():
            if not csv_path.exists():
                if view_name in optional_views:
                    continue
                raise FileNotFoundError(f"Missing required CSV: {csv_path}")
            extra_opts = view_read_options.get(view_name, "")
            sql = (
                f"CREATE OR REPLACE VIEW {view_name} AS "
                f"SELECT * FROM read_csv_auto('{_sql_quote_path(csv_path)}', header=true{extra_opts});"
            )
            self._conn.execute(sql)

    def prepare_subject_cache(self, subject_id: int) -> None:
        """Materialize subject-scoped temp tables to avoid repeated full CSV scans."""
        if self._cached_subject_id == int(subject_id):
            return
        conn = self.conn
        sid = int(subject_id)
        subject_scoped_tables = [
            "hosp_admissions",
            "hosp_patients",
            "hosp_transfers",
            "hosp_services",
            "hosp_labevents",
            "hosp_microbiologyevents",
            "hosp_poe",
            "hosp_poe_detail",
            "hosp_prescriptions",
            "hosp_pharmacy",
            "hosp_emar",
            "hosp_emar_detail",
            "hosp_diagnoses_icd",
            "hosp_procedures_icd",
            "hosp_drgcodes",
            "hosp_omr",
            "icu_icustays",
            "note_discharge",
            "note_discharge_detail",
            "note_radiology",
            "note_radiology_detail",
        ]
        for table_name in subject_scoped_tables:
            conn.execute(
                f"CREATE OR REPLACE TEMP TABLE {table_name} AS "
                f"SELECT * FROM main.{table_name} WHERE subject_id = ?",
                [sid],
            )
        self._cached_subject_id = sid

    def fetch_rows(self, sql: str, params: list[Any] | tuple[Any, ...] | None = None) -> list[dict[str, Any]]:
        cur = self.conn.execute(sql, params or [])
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def fetch_one(self, sql: str, params: list[Any] | tuple[Any, ...] | None = None) -> dict[str, Any] | None:
        rows = self.fetch_rows(sql, params=params)
        if not rows:
            return None
        return rows[0]
