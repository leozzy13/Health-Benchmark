from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from .config import BenchmarkConfig
from .duckdb_store import MimicDuckDBStore
from .utils import dedupe_rows, format_dt, normalize_date_to_noon, normalize_row, parse_dt


class AdmissionSkipError(ValueError):
    pass


@dataclass
class PacketExtractionResult:
    packet: dict[str, Any]


class PacketExtractor:
    def __init__(self, store: MimicDuckDBStore, config: BenchmarkConfig) -> None:
        self.store = store
        self.config = config

    def list_admissions_for_subject(self, subject_id: int) -> list[dict[str, Any]]:
        rows = self.store.fetch_rows(
            """
            SELECT
              a.subject_id,
              a.hadm_id,
              a.admittime,
              a.dischtime,
              COUNT(d.note_id) AS discharge_note_count
            FROM hosp_admissions a
            LEFT JOIN note_discharge d
              ON a.subject_id = TRY_CAST(d.subject_id AS BIGINT)
             AND a.hadm_id = TRY_CAST(d.hadm_id AS BIGINT)
            WHERE a.subject_id = ?
            GROUP BY a.subject_id, a.hadm_id, a.admittime, a.dischtime
            ORDER BY a.admittime ASC NULLS LAST, a.hadm_id ASC
            """,
            [int(subject_id)],
        )
        normalized = [normalize_row(row) for row in rows]
        return [row for row in normalized if int(row.get("discharge_note_count") or 0) > 0]

    def extract_admission_packet(
        self,
        subject_id: int,
        hadm_id: int,
        previous_admission_summary: dict[str, Any] | None,
    ) -> PacketExtractionResult:
        admission = self._fetch_admission(subject_id, hadm_id)
        if admission is None:
            raise AdmissionSkipError(f"Admission not found for subject_id={subject_id}, hadm_id={hadm_id}")

        admission = normalize_row(admission)
        admission_start_dt = parse_dt(admission.get("admittime"))
        if admission_start_dt is None:
            raise AdmissionSkipError(f"Admission {hadm_id} missing admittime")

        discharge_notes = self._fetch_discharge_notes(subject_id, hadm_id)
        if not discharge_notes:
            raise AdmissionSkipError(f"Admission {hadm_id} has no discharge note")

        admission_end_dt = self._resolve_admission_end(admission, discharge_notes)
        if admission_end_dt is None:
            raise AdmissionSkipError(f"Admission {hadm_id} lacks an exact end time")

        radiology_notes = self._fetch_radiology_notes(subject_id, hadm_id)
        diagnoses = self._fetch_diagnoses(subject_id, hadm_id, admission_start_dt, admission_end_dt)
        procedures = self._fetch_procedures(subject_id, hadm_id)
        microbiology = self._fetch_microbiology(subject_id, hadm_id)

        packet = {
            "subject_id": str(subject_id),
            "hadm_id": str(hadm_id),
            "admission_start": format_dt(admission_start_dt),
            "admission_end": format_dt(admission_end_dt),
            "previous_admission_summary": previous_admission_summary,
            "discharge_notes": discharge_notes,
            "radiology_notes": radiology_notes,
            "diagnoses": diagnoses,
            "procedures": procedures,
            "microbiology": microbiology,
            "timing_rules": {
                "discharge_backbone": True,
                "ignore_unlinked_radiology": True,
                "diagnoses_policy": "anchor_to_discharge_phase",
                "procedure_date_only_policy": "chartdate_plus_12_00_00",
                "microbiology_date_only_policy": "chartdate_plus_12_00_00",
                "admission_end_fallback": "latest_discharge_charttime_or_storetime",
            },
        }
        return PacketExtractionResult(packet=packet)

    def _fetch_admission(self, subject_id: int, hadm_id: int) -> dict[str, Any] | None:
        return self.store.fetch_one(
            """
            SELECT subject_id, hadm_id, admittime, dischtime
            FROM hosp_admissions
            WHERE subject_id = ? AND hadm_id = ?
            """,
            [int(subject_id), int(hadm_id)],
        )

    def _fetch_discharge_notes(self, subject_id: int, hadm_id: int) -> list[dict[str, Any]]:
        rows = self.store.fetch_rows(
            """
            SELECT note_id, note_seq, charttime, storetime, text
            FROM note_discharge
            WHERE TRY_CAST(subject_id AS BIGINT) = ? AND TRY_CAST(hadm_id AS BIGINT) = ?
            ORDER BY
              TRY_CAST(note_seq AS BIGINT) ASC NULLS LAST,
              charttime ASC NULLS LAST,
              storetime ASC NULLS LAST,
              note_id ASC
            """,
            [int(subject_id), int(hadm_id)],
        )
        normalized: list[dict[str, Any]] = []
        for row in rows:
            current = normalize_row(row)
            normalized.append(
                {
                    "note_id": str(current["note_id"]),
                    "note_seq": self._coerce_optional_int(current.get("note_seq")),
                    "charttime": current.get("charttime"),
                    "storetime": current.get("storetime"),
                    "text": current.get("text") or "",
                }
            )
        return dedupe_rows(normalized)

    def _fetch_radiology_notes(self, subject_id: int, hadm_id: int) -> list[dict[str, Any]]:
        rows = self.store.fetch_rows(
            """
            SELECT note_id, charttime, storetime, text
            FROM note_radiology
            WHERE TRY_CAST(subject_id AS BIGINT) = ? AND TRY_CAST(hadm_id AS BIGINT) = ?
            ORDER BY charttime ASC NULLS LAST, storetime ASC NULLS LAST, note_id ASC
            """,
            [int(subject_id), int(hadm_id)],
        )
        normalized: list[dict[str, Any]] = []
        for row in rows:
            current = normalize_row(row)
            normalized.append(
                {
                    "note_id": str(current["note_id"]),
                    "charttime": current.get("charttime"),
                    "storetime": current.get("storetime"),
                    "text": current.get("text") or "",
                }
            )
        return dedupe_rows(normalized)

    def _fetch_diagnoses(
        self,
        subject_id: int,
        hadm_id: int,
        admission_start_dt: datetime,
        admission_end_dt: datetime,
    ) -> list[dict[str, Any]]:
        rows = self.store.fetch_rows(
            """
            SELECT
              dx.seq_num,
              dx.icd_code,
              dx.icd_version,
              dd.long_title
            FROM hosp_diagnoses_icd dx
            LEFT JOIN main.hosp_d_icd_diagnoses dd
              ON dx.icd_code = dd.icd_code AND dx.icd_version = dd.icd_version
            WHERE dx.subject_id = ? AND dx.hadm_id = ?
            ORDER BY TRY_CAST(dx.seq_num AS BIGINT) ASC NULLS LAST, dx.icd_code ASC
            """,
            [int(subject_id), int(hadm_id)],
        )
        normalized_rows = [normalize_row(row) for row in rows]
        normalized_times = self._discharge_phase_timestamps(
            admission_start_dt=admission_start_dt,
            admission_end_dt=admission_end_dt,
            count=len(normalized_rows),
        )
        diagnoses: list[dict[str, Any]] = []
        for index, row in enumerate(normalized_rows):
            diagnoses.append(
                {
                    "seq_num": self._coerce_optional_int(row.get("seq_num")),
                    "icd_code": str(row.get("icd_code") or ""),
                    "icd_version": self._coerce_optional_int(row.get("icd_version")),
                    "long_title": str(row.get("long_title") or ""),
                    "normalized_time": format_dt(normalized_times[index]) if index < len(normalized_times) else None,
                    "time_provenance": "discharge_phase_anchor",
                }
            )
        return dedupe_rows(diagnoses)

    def _fetch_procedures(self, subject_id: int, hadm_id: int) -> list[dict[str, Any]]:
        rows = self.store.fetch_rows(
            """
            SELECT
              px.seq_num,
              px.chartdate,
              px.icd_code,
              px.icd_version,
              dp.long_title
            FROM hosp_procedures_icd px
            LEFT JOIN main.hosp_d_icd_procedures dp
              ON px.icd_code = dp.icd_code AND px.icd_version = dp.icd_version
            WHERE px.subject_id = ? AND px.hadm_id = ?
            ORDER BY px.chartdate ASC NULLS LAST, TRY_CAST(px.seq_num AS BIGINT) ASC NULLS LAST, px.icd_code ASC
            """,
            [int(subject_id), int(hadm_id)],
        )
        procedures: list[dict[str, Any]] = []
        for row in rows:
            current = normalize_row(row)
            normalized_time = normalize_date_to_noon(current.get("chartdate"))
            procedures.append(
                {
                    "seq_num": self._coerce_optional_int(current.get("seq_num")),
                    "chartdate": current.get("chartdate"),
                    "icd_code": str(current.get("icd_code") or ""),
                    "icd_version": self._coerce_optional_int(current.get("icd_version")),
                    "long_title": str(current.get("long_title") or ""),
                    "normalized_time": normalized_time,
                    "time_provenance": "date_only_normalized" if normalized_time else None,
                }
            )
        return dedupe_rows(procedures)

    def _fetch_microbiology(self, subject_id: int, hadm_id: int) -> list[dict[str, Any]]:
        rows = self.store.fetch_rows(
            """
            SELECT
              chartdate,
              charttime,
              spec_type_desc,
              test_name,
              org_name,
              ab_name,
              interpretation,
              comments
            FROM hosp_microbiologyevents
            WHERE subject_id = ? AND hadm_id = ?
            ORDER BY
              COALESCE(charttime, CAST(chartdate AS TIMESTAMP)) ASC NULLS LAST,
              charttime ASC NULLS LAST
            """,
            [int(subject_id), int(hadm_id)],
        )
        microbiology: list[dict[str, Any]] = []
        for row in rows:
            current = normalize_row(row)
            normalized_time = None
            provenance = None
            if current.get("charttime") in (None, "") and current.get("chartdate") not in (None, ""):
                normalized_time = normalize_date_to_noon(current.get("chartdate"))
                provenance = "date_only_normalized"
            microbiology.append(
                {
                    "chartdate": current.get("chartdate"),
                    "charttime": current.get("charttime"),
                    "normalized_time": normalized_time,
                    "time_provenance": provenance,
                    "spec_type_desc": str(current.get("spec_type_desc") or ""),
                    "test_name": str(current.get("test_name") or ""),
                    "org_name": str(current.get("org_name") or ""),
                    "ab_name": str(current.get("ab_name") or ""),
                    "interpretation": str(current.get("interpretation") or ""),
                    "comments": str(current.get("comments") or ""),
                }
            )
        return dedupe_rows(microbiology)

    def _resolve_admission_end(
        self,
        admission: dict[str, Any],
        discharge_notes: list[dict[str, Any]],
    ) -> datetime | None:
        dischtime = parse_dt(admission.get("dischtime"))
        if dischtime is not None:
            return dischtime

        discharge_charttimes = [
            parse_dt(note.get("charttime")) for note in discharge_notes if parse_dt(note.get("charttime")) is not None
        ]
        if discharge_charttimes:
            return max(discharge_charttimes)

        discharge_storetimes = [
            parse_dt(note.get("storetime")) for note in discharge_notes if parse_dt(note.get("storetime")) is not None
        ]
        if discharge_storetimes:
            return max(discharge_storetimes)
        return None

    @staticmethod
    def _discharge_phase_timestamps(
        *,
        admission_start_dt: datetime,
        admission_end_dt: datetime,
        count: int,
    ) -> list[datetime]:
        if count <= 0:
            return []

        stay_seconds = max(0, int((admission_end_dt - admission_start_dt).total_seconds()))
        window_seconds = min(3600, stay_seconds) if stay_seconds > 0 else 0
        window_start = admission_end_dt - timedelta(seconds=window_seconds)

        if count == 1 or window_seconds <= 0:
            return [admission_end_dt for _ in range(count)]

        step = window_seconds / (count - 1)
        timestamps: list[datetime] = []
        for index in range(count):
            anchored = window_start + timedelta(seconds=round(step * index))
            if anchored < admission_start_dt:
                anchored = admission_start_dt
            if anchored > admission_end_dt:
                anchored = admission_end_dt
            timestamps.append(anchored)
        return timestamps

    @staticmethod
    def _coerce_optional_int(value: Any) -> int | None:
        if value in (None, ""):
            return None
        return int(value)
