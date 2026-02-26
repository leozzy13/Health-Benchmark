from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import BenchmarkConfig
from .duckdb_store import MimicDuckDBStore
from .utils import (
    canonical_json_dumps,
    clip_rows,
    dt_to_iso,
    flatten_eids,
    minutes_since,
    normalize_row,
    parse_dt,
    sha256_hex,
    utc_now_iso,
)


@dataclass(slots=True)
class PacketExtractionResult:
    packet: dict[str, Any]
    input_data_manifest: dict[str, Any]
    unlinked_radiology_notes: list[dict[str, Any]]


class PacketExtractor:
    def __init__(self, store: MimicDuckDBStore, config: BenchmarkConfig) -> None:
        self.store = store
        self.config = config

    def list_admissions_for_subject(
        self,
        subject_id: int,
        *,
        only_with_discharge: bool | None = None,
        max_admissions: int | None = None,
    ) -> list[dict[str, Any]]:
        if only_with_discharge is None:
            only_with_discharge = self.config.extraction.only_admissions_with_discharge

        base_sql = """
            SELECT
              a.subject_id,
              a.hadm_id,
              a.admittime,
              a.dischtime,
              COUNT(nd.note_id) AS discharge_note_count
            FROM hosp_admissions a
            LEFT JOIN note_discharge nd
              ON a.subject_id = nd.subject_id AND a.hadm_id = nd.hadm_id
            WHERE a.subject_id = ?
            GROUP BY a.subject_id, a.hadm_id, a.admittime, a.dischtime
        """
        rows = self.store.fetch_rows(base_sql, [subject_id])
        rows = [normalize_row(r) for r in rows]
        if only_with_discharge:
            rows = [r for r in rows if (r.get("discharge_note_count") or 0) > 0]
        rows.sort(key=lambda r: (r.get("admittime") or "9999", int(r["hadm_id"])))
        if max_admissions is not None:
            rows = rows[:max_admissions]
        return rows

    def extract_admission_packet(self, subject_id: int, hadm_id: int) -> PacketExtractionResult:
        adm_patient = self._fetch_admission_and_patient(subject_id, hadm_id)
        if adm_patient is None:
            raise ValueError(f"Admission not found for subject_id={subject_id}, hadm_id={hadm_id}")

        adm_patient = normalize_row(adm_patient)
        admit_dt = parse_dt(adm_patient.get("admittime"))
        if admit_dt is None:
            raise ValueError(f"Admission {hadm_id} missing admittime")

        section_stats: dict[str, dict[str, Any]] = {}
        truncation_applied = False

        transfers = self._query_rows(
            """
            SELECT subject_id, hadm_id, transfer_id, eventtype, careunit, intime, outtime
            FROM hosp_transfers
            WHERE subject_id = ? AND hadm_id = ?
            ORDER BY intime ASC NULLS LAST, outtime ASC NULLS LAST, transfer_id ASC
            """,
            [subject_id, hadm_id],
            "transfers",
            section_stats,
        )
        services = self._query_rows(
            """
            SELECT subject_id, hadm_id, transfertime, prev_service, curr_service
            FROM hosp_services
            WHERE subject_id = ? AND hadm_id = ?
            ORDER BY transfertime ASC NULLS LAST, curr_service ASC
            """,
            [subject_id, hadm_id],
            "services",
            section_stats,
        )

        labs_strict = self._query_rows(
            """
            SELECT
              le.labevent_id, le.subject_id, le.hadm_id, le.specimen_id, le.itemid,
              le.order_provider_id, le.charttime, le.storetime, le.value, le.valuenum,
              le.valueuom, le.ref_range_lower, le.ref_range_upper, le.flag, le.priority,
              le.comments,
              dli.label, dli.fluid, dli.category
            FROM hosp_labevents le
            LEFT JOIN hosp_d_labitems dli USING (itemid)
            WHERE le.subject_id = ? AND le.hadm_id = ?
            ORDER BY le.charttime ASC NULLS LAST, le.storetime ASC NULLS LAST, le.labevent_id ASC
            """,
            [subject_id, hadm_id],
            "labs_strict",
            section_stats,
        )
        labs = labs_strict
        lab_linkage_counts = {"strict_hadm": len(labs_strict), "proximal_null_hadm": 0}
        if self.config.extraction.proximal_lab_capture:
            pad_hours = int(self.config.extraction.proximal_padding_hours)
            labs_prox = self._query_rows(
                f"""
                SELECT
                  le.labevent_id, le.subject_id, le.hadm_id, le.specimen_id, le.itemid,
                  le.order_provider_id, le.charttime, le.storetime, le.value, le.valuenum,
                  le.valueuom, le.ref_range_lower, le.ref_range_upper, le.flag, le.priority,
                  le.comments,
                  dli.label, dli.fluid, dli.category
                FROM hosp_labevents le
                LEFT JOIN hosp_d_labitems dli USING (itemid)
                JOIN hosp_admissions a
                  ON a.subject_id = le.subject_id
                WHERE le.subject_id = ?
                  AND le.hadm_id IS NULL
                  AND a.hadm_id = ?
                  AND le.charttime BETWEEN
                        (a.admittime - INTERVAL '{pad_hours} hour')
                        AND (COALESCE(a.dischtime, a.admittime) + INTERVAL '{pad_hours} hour')
                ORDER BY le.charttime ASC NULLS LAST, le.storetime ASC NULLS LAST, le.labevent_id ASC
                """,
                [subject_id, hadm_id],
                "labs_proximal",
                section_stats,
            )
            labs = self._dedupe_by_key(labs + labs_prox, "labevent_id")
            lab_linkage_counts["proximal_null_hadm"] = len(labs_prox)

        micro_strict = self._query_rows(
            """
            SELECT
              microevent_id, subject_id, hadm_id, micro_specimen_id, order_provider_id,
              chartdate, charttime, storedate, storetime, spec_itemid, spec_type_desc,
              test_seq, test_itemid, test_name, org_itemid, org_name, isolate_num,
              quantity, ab_itemid, ab_name, dilution_text, dilution_comparison,
              dilution_value, interpretation, comments
            FROM hosp_microbiologyevents
            WHERE subject_id = ? AND hadm_id = ?
            ORDER BY charttime ASC NULLS LAST, chartdate ASC NULLS LAST, storetime ASC NULLS LAST, microevent_id ASC
            """,
            [subject_id, hadm_id],
            "micro_strict",
            section_stats,
        )
        microbiology = micro_strict
        micro_linkage_counts = {"strict_hadm": len(micro_strict), "proximal_null_hadm": 0}
        if self.config.extraction.proximal_micro_capture:
            pad_hours = int(self.config.extraction.proximal_padding_hours)
            micro_prox = self._query_rows(
                f"""
                SELECT
                  m.microevent_id, m.subject_id, m.hadm_id, m.micro_specimen_id, m.order_provider_id,
                  m.chartdate, m.charttime, m.storedate, m.storetime, m.spec_itemid, m.spec_type_desc,
                  m.test_seq, m.test_itemid, m.test_name, m.org_itemid, m.org_name, m.isolate_num,
                  m.quantity, m.ab_itemid, m.ab_name, m.dilution_text, m.dilution_comparison,
                  m.dilution_value, m.interpretation, m.comments
                FROM hosp_microbiologyevents m
                JOIN hosp_admissions a
                  ON a.subject_id = m.subject_id
                WHERE m.subject_id = ?
                  AND m.hadm_id IS NULL
                  AND a.hadm_id = ?
                  AND COALESCE(m.charttime, CAST(m.chartdate AS TIMESTAMP)) BETWEEN
                        (a.admittime - INTERVAL '{pad_hours} hour')
                        AND (COALESCE(a.dischtime, a.admittime) + INTERVAL '{pad_hours} hour')
                ORDER BY m.charttime ASC NULLS LAST, m.chartdate ASC NULLS LAST, m.storetime ASC NULLS LAST, m.microevent_id ASC
                """,
                [subject_id, hadm_id],
                "micro_proximal",
                section_stats,
            )
            microbiology = self._dedupe_by_key(microbiology + micro_prox, "microevent_id")
            micro_linkage_counts["proximal_null_hadm"] = len(micro_prox)

        poe = self._query_rows(
            """
            SELECT
              poe_id, poe_seq, subject_id, hadm_id, ordertime, order_type, order_subtype,
              transaction_type, discontinue_of_poe_id, discontinued_by_poe_id,
              order_provider_id, order_status
            FROM hosp_poe
            WHERE subject_id = ? AND hadm_id = ?
            ORDER BY ordertime ASC NULLS LAST, poe_seq ASC
            """,
            [subject_id, hadm_id],
            "poe",
            section_stats,
        )
        poe_detail = self._query_rows(
            """
            SELECT poe_id, poe_seq, subject_id, field_name, field_value
            FROM hosp_poe_detail
            WHERE subject_id = ?
              AND poe_id IN (
                SELECT poe_id FROM hosp_poe WHERE subject_id = ? AND hadm_id = ?
              )
            ORDER BY poe_seq ASC NULLS LAST, field_name ASC
            """,
            [subject_id, subject_id, hadm_id],
            "poe_detail",
            section_stats,
        )
        prescriptions = self._query_rows(
            """
            SELECT
              subject_id, hadm_id, pharmacy_id, poe_id, poe_seq, order_provider_id,
              starttime, stoptime, drug_type, drug, formulary_drug_cd, gsn, ndc,
              prod_strength, form_rx, dose_val_rx, dose_unit_rx, form_val_disp,
              form_unit_disp, doses_per_24_hrs, route
            FROM hosp_prescriptions
            WHERE subject_id = ? AND hadm_id = ?
            ORDER BY starttime ASC NULLS LAST, stoptime ASC NULLS LAST, pharmacy_id ASC NULLS LAST
            """,
            [subject_id, hadm_id],
            "prescriptions",
            section_stats,
        )
        pharmacy = self._query_rows(
            """
            SELECT
              subject_id, hadm_id, pharmacy_id, poe_id, starttime, stoptime, medication, proc_type,
              status, entertime, verifiedtime, route, frequency, disp_sched, infusion_type,
              sliding_scale, lockout_interval, basal_rate, one_hr_max, doses_per_24_hrs,
              duration, duration_interval, expiration_value, expiration_unit, expirationdate,
              dispensation, fill_quantity
            FROM hosp_pharmacy
            WHERE subject_id = ? AND hadm_id = ?
            ORDER BY entertime ASC NULLS LAST, pharmacy_id ASC NULLS LAST
            """,
            [subject_id, hadm_id],
            "pharmacy",
            section_stats,
        )

        emar_candidates = self._query_rows(
            """
            SELECT
              subject_id, hadm_id, emar_id, emar_seq, poe_id, pharmacy_id, enter_provider_id,
              charttime, medication, event_txt, scheduletime, storetime
            FROM hosp_emar
            WHERE subject_id = ? AND (hadm_id = ? OR hadm_id IS NULL)
            ORDER BY charttime ASC NULLS LAST, emar_seq ASC NULLS LAST
            """,
            [subject_id, hadm_id],
            "emar_candidates",
            section_stats,
        )
        emar, emar_linkage_counts = self._filter_emar_candidates(
            emar_candidates, hadm_id=hadm_id, admit_dt=admit_dt, disch_dt=parse_dt(adm_patient.get("dischtime")),
            poe=poe, pharmacy=pharmacy
        )

        emar_detail_candidates = self._query_rows(
            """
            SELECT
              subject_id, emar_id, emar_seq, parent_field_ordinal, administration_type,
              pharmacy_id, reason_for_no_barcode, complete_dose_not_given, dose_due, dose_due_unit,
              dose_given, dose_given_unit, product_code, product_description, prior_infusion_rate,
              infusion_rate, infusion_rate_unit, route
            FROM hosp_emar_detail
            WHERE subject_id = ?
              AND emar_id IN (
                SELECT emar_id FROM hosp_emar WHERE subject_id = ? AND (hadm_id = ? OR hadm_id IS NULL)
              )
            ORDER BY emar_seq ASC NULLS LAST, parent_field_ordinal ASC NULLS LAST
            """,
            [subject_id, subject_id, hadm_id],
            "emar_detail_candidates",
            section_stats,
        )
        selected_emar_ids = {str(r.get("emar_id")) for r in emar}
        emar_detail = [r for r in emar_detail_candidates if str(r.get("emar_id")) in selected_emar_ids]

        diagnoses_icd = self._query_rows(
            """
            SELECT
              di.subject_id, di.hadm_id, di.seq_num, di.icd_code, di.icd_version, dd.long_title
            FROM hosp_diagnoses_icd di
            LEFT JOIN hosp_d_icd_diagnoses dd
              ON di.icd_code = dd.icd_code AND di.icd_version = dd.icd_version
            WHERE di.subject_id = ? AND di.hadm_id = ?
            ORDER BY di.seq_num ASC NULLS LAST, di.icd_code ASC
            """,
            [subject_id, hadm_id],
            "diagnoses_icd",
            section_stats,
        )
        procedures_icd = self._query_rows(
            """
            SELECT
              pi.subject_id, pi.hadm_id, pi.seq_num, pi.chartdate, pi.icd_code, pi.icd_version, dp.long_title
            FROM hosp_procedures_icd pi
            LEFT JOIN hosp_d_icd_procedures dp
              ON pi.icd_code = dp.icd_code AND pi.icd_version = dp.icd_version
            WHERE pi.subject_id = ? AND pi.hadm_id = ?
            ORDER BY pi.chartdate ASC NULLS LAST, pi.seq_num ASC NULLS LAST, pi.icd_code ASC
            """,
            [subject_id, hadm_id],
            "procedures_icd",
            section_stats,
        )
        drgcodes = self._query_rows(
            """
            SELECT
              subject_id, hadm_id, drg_type, drg_code, description, drg_severity, drg_mortality
            FROM hosp_drgcodes
            WHERE subject_id = ? AND hadm_id = ?
            ORDER BY drg_type ASC NULLS LAST, drg_code ASC NULLS LAST
            """,
            [subject_id, hadm_id],
            "drgcodes",
            section_stats,
        )

        discharge_notes = self._query_rows(
            """
            SELECT
              note_id, subject_id, hadm_id, note_type, note_seq, charttime, storetime, text
            FROM note_discharge
            WHERE subject_id = ? AND hadm_id = ?
            ORDER BY note_type ASC NULLS LAST, note_seq ASC NULLS LAST
            """,
            [subject_id, hadm_id],
            "discharge",
            section_stats,
        )
        if self.config.extraction.require_discharge_note and not discharge_notes:
            raise ValueError(
                f"Admission {hadm_id} has no discharge note; cannot build prompt under current policy."
            )

        discharge_detail = self._query_rows(
            """
            SELECT note_id, subject_id, field_name, field_value, field_ordinal
            FROM note_discharge_detail
            WHERE subject_id = ?
              AND note_id IN (
                SELECT note_id FROM note_discharge WHERE subject_id = ? AND hadm_id = ?
              )
            ORDER BY note_id ASC, field_ordinal ASC NULLS LAST
            """,
            [subject_id, subject_id, hadm_id],
            "discharge_detail",
            section_stats,
        )
        radiology = self._query_rows(
            """
            SELECT
              note_id, subject_id, hadm_id, note_type, note_seq, charttime, storetime, text
            FROM note_radiology
            WHERE subject_id = ? AND hadm_id = ?
            ORDER BY charttime ASC NULLS LAST, note_seq ASC NULLS LAST
            """,
            [subject_id, hadm_id],
            "radiology",
            section_stats,
        )
        radiology_detail = self._query_rows(
            """
            SELECT note_id, subject_id, field_name, field_value, field_ordinal
            FROM note_radiology_detail
            WHERE subject_id = ?
              AND note_id IN (
                SELECT note_id FROM note_radiology WHERE subject_id = ? AND hadm_id = ?
              )
            ORDER BY note_id ASC, field_ordinal ASC NULLS LAST
            """,
            [subject_id, subject_id, hadm_id],
            "radiology_detail",
            section_stats,
        )

        unlinked_radiology_notes: list[dict[str, Any]] = []
        if self.config.extraction.include_unlinked_radiology_in_sidecar:
            unlinked_radiology_notes = self._query_rows(
                """
                SELECT
                  note_id, subject_id, hadm_id, note_type, note_seq, charttime, storetime, text
                FROM note_radiology
                WHERE subject_id = ? AND hadm_id IS NULL
                ORDER BY charttime ASC NULLS LAST, note_seq ASC NULLS LAST
                """,
                [subject_id],
                "radiology_unlinked",
                section_stats,
            )

        icustays = []
        if self.config.extraction.include_icu_stays:
            icustays = self._query_rows(
                """
                SELECT
                  subject_id, hadm_id, stay_id, first_careunit, last_careunit, intime, outtime, los
                FROM icu_icustays
                WHERE subject_id = ? AND hadm_id = ?
                ORDER BY intime ASC NULLS LAST, stay_id ASC
                """,
                [subject_id, hadm_id],
                "icustays",
                section_stats,
            )

        # Normalize all rows before packet shaping.
        transfers = [normalize_row(r) for r in transfers]
        services = [normalize_row(r) for r in services]
        labs = [normalize_row(r) for r in labs]
        microbiology = [normalize_row(r) for r in microbiology]
        poe = [normalize_row(r) for r in poe]
        poe_detail = [normalize_row(r) for r in poe_detail]
        prescriptions = [normalize_row(r) for r in prescriptions]
        pharmacy = [normalize_row(r) for r in pharmacy]
        emar = [normalize_row(r) for r in emar]
        emar_detail = [normalize_row(r) for r in emar_detail]
        diagnoses_icd = [normalize_row(r) for r in diagnoses_icd]
        procedures_icd = [normalize_row(r) for r in procedures_icd]
        drgcodes = [normalize_row(r) for r in drgcodes]
        discharge_notes = [normalize_row(r) for r in discharge_notes]
        discharge_detail = [normalize_row(r) for r in discharge_detail]
        radiology = [normalize_row(r) for r in radiology]
        radiology_detail = [normalize_row(r) for r in radiology_detail]
        unlinked_radiology_notes = [normalize_row(r) for r in unlinked_radiology_notes]
        icustays = [normalize_row(r) for r in icustays]

        # Deterministic truncation (row caps).
        section_lists: dict[str, list[dict[str, Any]]] = {
            "transfers": transfers,
            "services": services,
            "discharge": discharge_notes,
            "discharge_detail": discharge_detail,
            "radiology": radiology,
            "radiology_detail": radiology_detail,
            "labs": labs,
            "microbiology": microbiology,
            "poe": poe,
            "poe_detail": poe_detail,
            "prescriptions": prescriptions,
            "pharmacy": pharmacy,
            "emar": emar,
            "emar_detail": emar_detail,
            "diagnoses_icd": diagnoses_icd,
            "procedures_icd": procedures_icd,
            "drgcodes": drgcodes,
            "icustays": icustays,
        }
        truncation_stats: dict[str, dict[str, Any]] = {}
        for section_name, rows in section_lists.items():
            cap = self.config.truncation.per_section_row_caps.get(section_name)
            seen = len(rows)
            clipped, was_truncated = clip_rows(rows, cap)
            section_lists[section_name] = clipped
            truncation_stats[section_name] = {
                "seen": seen,
                "retained": len(clipped),
                "cap": cap,
                "truncated": was_truncated,
            }
            truncation_applied = truncation_applied or was_truncated

        # Assign EIDs and relative times after truncation to keep numbering aligned with prompt packet.
        transfers = self._attach_eids_and_times(
            section_lists["transfers"],
            prefix="XFER",
            admit_dt=admit_dt,
            time_fields={"t_rel_min": ("intime",)},
            drop_fields={"subject_id", "hadm_id"},
        )
        services = self._attach_eids_and_times(
            section_lists["services"],
            prefix="SVC",
            admit_dt=admit_dt,
            time_fields={"t_rel_min": ("transfertime",)},
            drop_fields={"subject_id", "hadm_id"},
        )
        discharge_notes = self._attach_eids_and_times(
            section_lists["discharge"],
            prefix="DS",
            admit_dt=admit_dt,
            time_fields={"t_rel_min": ("charttime",)},
            drop_fields={"subject_id", "hadm_id"},
        )
        discharge_detail = self._attach_eids_and_times(
            section_lists["discharge_detail"],
            prefix="DSD",
            admit_dt=admit_dt,
            time_fields={},
            drop_fields={"subject_id"},
        )
        radiology = self._attach_eids_and_times(
            section_lists["radiology"],
            prefix="RAD",
            admit_dt=admit_dt,
            time_fields={"t_rel_min": ("charttime",)},
            drop_fields={"subject_id", "hadm_id"},
        )
        radiology_detail = self._attach_eids_and_times(
            section_lists["radiology_detail"],
            prefix="RADD",
            admit_dt=admit_dt,
            time_fields={},
            drop_fields={"subject_id"},
        )
        labs = self._attach_eids_and_times(
            section_lists["labs"],
            prefix="LAB",
            admit_dt=admit_dt,
            time_fields={"t_rel_min": ("charttime", "storetime")},
            drop_fields={"subject_id", "hadm_id"},
        )
        microbiology = self._attach_eids_and_times(
            section_lists["microbiology"],
            prefix="MICRO",
            admit_dt=admit_dt,
            time_fields={"t_rel_min": ("charttime", "chartdate", "storetime", "storedate")},
            drop_fields={"subject_id", "hadm_id"},
        )
        poe = self._attach_eids_and_times(
            section_lists["poe"],
            prefix="POE",
            admit_dt=admit_dt,
            time_fields={"t_rel_min": ("ordertime",)},
            drop_fields={"subject_id", "hadm_id"},
        )
        poe_detail = self._attach_eids_and_times(
            section_lists["poe_detail"],
            prefix="POED",
            admit_dt=admit_dt,
            time_fields={},
            drop_fields={"subject_id"},
        )
        prescriptions = self._attach_eids_and_times(
            section_lists["prescriptions"],
            prefix="RX",
            admit_dt=admit_dt,
            time_fields={"t_rel_min": ("starttime", "stoptime")},
            drop_fields={"subject_id", "hadm_id"},
        )
        pharmacy = self._attach_eids_and_times(
            section_lists["pharmacy"],
            prefix="PHARM",
            admit_dt=admit_dt,
            time_fields={"t_rel_min": ("entertime", "starttime")},
            drop_fields={"subject_id", "hadm_id"},
        )
        emar = self._attach_eids_and_times(
            section_lists["emar"],
            prefix="EMAR",
            admit_dt=admit_dt,
            time_fields={"t_rel_min": ("charttime", "scheduletime", "storetime")},
            drop_fields={"subject_id", "hadm_id"},
        )
        emar_detail = self._attach_eids_and_times(
            section_lists["emar_detail"],
            prefix="EMD",
            admit_dt=admit_dt,
            time_fields={},
            drop_fields={"subject_id"},
        )
        diagnoses_icd = self._attach_eids_and_times(
            section_lists["diagnoses_icd"],
            prefix="DX",
            admit_dt=admit_dt,
            time_fields={},
            drop_fields={"subject_id", "hadm_id"},
        )
        procedures_icd = self._attach_eids_and_times(
            section_lists["procedures_icd"],
            prefix="PX",
            admit_dt=admit_dt,
            time_fields={"t_rel_min": ("chartdate",)},
            drop_fields={"subject_id", "hadm_id"},
        )
        drgcodes = self._attach_eids_and_times(
            section_lists["drgcodes"],
            prefix="DRG",
            admit_dt=admit_dt,
            time_fields={},
            drop_fields={"subject_id", "hadm_id"},
        )
        icustays = self._attach_eids_and_times(
            section_lists["icustays"],
            prefix="ICU",
            admit_dt=admit_dt,
            time_fields={"t_in_min": ("intime",), "t_out_min": ("outtime",)},
            drop_fields={"subject_id", "hadm_id"},
        )

        packet = {
            "packet_schema_version": self.config.packet_schema_version,
            "benchmark_version": self.config.benchmark_version,
            "dataset_versions": {
                "mimiciv": self.config.dataset_versions.mimiciv,
                "mimiciv_note": self.config.dataset_versions.mimiciv_note,
            },
            "ids": {"subject_id": int(subject_id), "hadm_id": int(hadm_id)},
            "time_basis": {
                "admittime": dt_to_iso(adm_patient.get("admittime"), "admittime"),
                "dischtime": dt_to_iso(adm_patient.get("dischtime"), "dischtime"),
                "timezone": "DEIDENTIFIED/UNKNOWN",
                "relative_time_unit": "minutes_since_admit",
            },
            "patient": {
                "eid": "PT#000001",
                "gender": adm_patient.get("gender"),
                "anchor_age": adm_patient.get("anchor_age"),
                "anchor_year": adm_patient.get("anchor_year"),
                "anchor_year_group": adm_patient.get("anchor_year_group"),
                "dod": dt_to_iso(adm_patient.get("dod"), "dod"),
            },
            "admission": {
                "eid": "ADM#000001",
                "deathtime": dt_to_iso(adm_patient.get("deathtime"), "deathtime"),
                "admission_type": adm_patient.get("admission_type"),
                "admit_provider_id": adm_patient.get("admit_provider_id"),
                "admission_location": adm_patient.get("admission_location"),
                "discharge_location": adm_patient.get("discharge_location"),
                "insurance": adm_patient.get("insurance"),
                "language": adm_patient.get("language"),
                "marital_status": adm_patient.get("marital_status"),
                "race": adm_patient.get("race"),
                "edregtime": dt_to_iso(adm_patient.get("edregtime"), "edregtime"),
                "edouttime": dt_to_iso(adm_patient.get("edouttime"), "edouttime"),
                "hospital_expire_flag": adm_patient.get("hospital_expire_flag"),
            },
            "location_timeline": {
                "transfers": transfers,
                "services": services,
            },
            "notes": {
                "discharge": discharge_notes,
                "discharge_detail": discharge_detail,
                "radiology": radiology,
                "radiology_detail": radiology_detail,
            },
            "labs": labs,
            "microbiology": microbiology,
            "orders": {
                "poe": poe,
                "poe_detail": poe_detail,
                "prescriptions": prescriptions,
                "pharmacy": pharmacy,
                "emar": emar,
                "emar_detail": emar_detail,
            },
            "billing": {
                "diagnoses_icd": diagnoses_icd,
                "procedures_icd": procedures_icd,
                "drgcodes": drgcodes,
            },
            "icu": {
                "has_icu_stay": bool(icustays),
                "icustays": icustays,
            },
            "packet_stats": {
                "row_counts": {
                    "transfers": len(transfers),
                    "services": len(services),
                    "labevents": len(labs),
                    "microbiologyevents": len(microbiology),
                    "radiology": len(radiology),
                    "discharge": len(discharge_notes),
                },
                "sha256_canonical_packet": None,
            },
        }

        hash_basis = canonical_json_dumps(packet)
        packet_hash = sha256_hex(hash_basis)
        packet["packet_stats"]["sha256_canonical_packet"] = packet_hash

        input_data_manifest = {
            "schema_version": self.config.manifest_schema_version,
            "benchmark_version": self.config.benchmark_version,
            "dataset_versions": {
                "mimiciv": self.config.dataset_versions.mimiciv,
                "mimiciv_note": self.config.dataset_versions.mimiciv_note,
            },
            "ids": {"subject_id": int(subject_id), "hadm_id": int(hadm_id)},
            "extraction_timestamp_utc": utc_now_iso(),
            "tables_used": self._build_tables_used(section_stats, truncation_stats),
            "null_handling_policies": {
                "labevents_hadm_id_null": self.config.null_handling.labevents_hadm_id_null,
                "microbiologyevents_hadm_id_null": self.config.null_handling.microbiologyevents_hadm_id_null,
                "emar_hadm_id_null": self.config.null_handling.emar_hadm_id_null,
                "radiology_hadm_id_null": self.config.null_handling.radiology_hadm_id_null,
            },
            "policy_capture_counts": {
                "labs": lab_linkage_counts,
                "microbiology": micro_linkage_counts,
                "emar": emar_linkage_counts,
                "radiology": {
                    "linked_hadm": len(radiology),
                    "unlinked_hadm_excluded": len(unlinked_radiology_notes),
                },
            },
            "truncation": {
                "applied": truncation_applied,
                "ruleset_id": self.config.truncation.ruleset_id,
                "per_section_limits": self.config.truncation.per_section_row_caps,
                "per_section_counts": truncation_stats,
            },
            "hashes": {"packet_sha256": packet_hash},
            "packet_eid_count": len(flatten_eids(packet)),
        }

        return PacketExtractionResult(
            packet=packet,
            input_data_manifest=input_data_manifest,
            unlinked_radiology_notes=unlinked_radiology_notes,
        )

    def _fetch_admission_and_patient(self, subject_id: int, hadm_id: int) -> dict[str, Any] | None:
        return self.store.fetch_one(
            """
            SELECT
              a.subject_id, a.hadm_id,
              a.admittime, a.dischtime, a.deathtime,
              a.admission_type, a.admit_provider_id,
              a.admission_location, a.discharge_location,
              a.insurance, a.language, a.marital_status, a.race,
              a.edregtime, a.edouttime,
              a.hospital_expire_flag,
              p.gender, p.anchor_age, p.anchor_year, p.anchor_year_group, p.dod
            FROM hosp_admissions a
            JOIN hosp_patients p USING (subject_id)
            WHERE a.subject_id = ? AND a.hadm_id = ?
            """,
            [subject_id, hadm_id],
        )

    def _query_rows(
        self,
        sql: str,
        params: list[Any],
        section_key: str,
        section_stats: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        rows = self.store.fetch_rows(sql, params)
        section_stats[section_key] = {"query_row_count": len(rows)}
        return rows

    def _build_tables_used(
        self,
        section_stats: dict[str, dict[str, Any]],
        truncation_stats: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        table_map = {
            "transfers": "hosp.transfers",
            "services": "hosp.services",
            "labs_strict": "hosp.labevents",
            "labs_proximal": "hosp.labevents",
            "micro_strict": "hosp.microbiologyevents",
            "micro_proximal": "hosp.microbiologyevents",
            "poe": "hosp.poe",
            "poe_detail": "hosp.poe_detail",
            "prescriptions": "hosp.prescriptions",
            "pharmacy": "hosp.pharmacy",
            "emar_candidates": "hosp.emar",
            "emar_detail_candidates": "hosp.emar_detail",
            "diagnoses_icd": "hosp.diagnoses_icd+d_icd_diagnoses",
            "procedures_icd": "hosp.procedures_icd+d_icd_procedures",
            "drgcodes": "hosp.drgcodes",
            "discharge": "note.discharge",
            "discharge_detail": "note.discharge_detail",
            "radiology": "note.radiology",
            "radiology_detail": "note.radiology_detail",
            "radiology_unlinked": "note.radiology",
            "icustays": "icu.icustays",
        }

        rows: list[dict[str, Any]] = []
        for key in sorted(section_stats.keys()):
            row = {
                "table": table_map.get(key, key),
                "section_key": key,
                "row_count": section_stats[key]["query_row_count"],
            }
            if key in truncation_stats:
                row["row_count_retained"] = truncation_stats[key]["retained"]
                row["truncated"] = truncation_stats[key]["truncated"]
            rows.append(row)
        # add single-row admission/patient join reference explicitly
        rows.insert(0, {"table": "hosp.admissions+hosp.patients", "section_key": "admission_patient", "row_count": 1})
        return rows

    def _filter_emar_candidates(
        self,
        emar_candidates: list[dict[str, Any]],
        *,
        hadm_id: int,
        admit_dt,
        disch_dt,
        poe: list[dict[str, Any]],
        pharmacy: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        target_poe_ids = {str(r.get("poe_id")) for r in poe if r.get("poe_id") not in (None, "")}
        target_pharmacy_ids = {
            str(r.get("pharmacy_id")) for r in pharmacy if r.get("pharmacy_id") not in (None, "")
        }
        counts = {
            "direct_hadm": 0,
            "linked_via_pharmacy_id": 0,
            "linked_via_poe_id": 0,
            "linked_via_time_window": 0,
            "excluded_null_hadm": 0,
        }
        selected: list[dict[str, Any]] = []
        for row in emar_candidates:
            row_hadm = row.get("hadm_id")
            if row_hadm == hadm_id:
                counts["direct_hadm"] += 1
                selected.append(row)
                continue

            if row_hadm not in (None, ""):
                counts["excluded_null_hadm"] += 1
                continue

            pharmacy_id = row.get("pharmacy_id")
            poe_id = row.get("poe_id")
            if pharmacy_id not in (None, "") and str(pharmacy_id) in target_pharmacy_ids:
                counts["linked_via_pharmacy_id"] += 1
                selected.append(row)
                continue
            if poe_id not in (None, "") and str(poe_id) in target_poe_ids:
                counts["linked_via_poe_id"] += 1
                selected.append(row)
                continue
            if self.config.extraction.emar_time_window_fallback:
                chart_dt = parse_dt(row.get("charttime"))
                if chart_dt is not None and admit_dt is not None and disch_dt is not None:
                    if admit_dt <= chart_dt <= disch_dt:
                        counts["linked_via_time_window"] += 1
                        selected.append(row)
                        continue
            counts["excluded_null_hadm"] += 1

        # Keep deterministic order.
        selected.sort(
            key=lambda r: (
                dt_to_iso(r.get("charttime"), "charttime") or "9999-99-99T99:99:99",
                r.get("emar_seq") if r.get("emar_seq") is not None else 10**18,
                str(r.get("emar_id") or ""),
            )
        )
        selected = self._dedupe_by_key(selected, "emar_id")
        return selected, counts

    @staticmethod
    def _dedupe_by_key(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
        seen: set[Any] = set()
        out: list[dict[str, Any]] = []
        for row in rows:
            value = row.get(key)
            if value in seen:
                continue
            seen.add(value)
            out.append(row)
        return out

    def _attach_eids_and_times(
        self,
        rows: list[dict[str, Any]],
        *,
        prefix: str,
        admit_dt,
        time_fields: dict[str, tuple[str, ...]],
        drop_fields: set[str],
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for idx, row in enumerate(rows, start=1):
            new_row = {k: row.get(k) for k in row.keys() if k not in drop_fields}
            new_row["eid"] = f"{prefix}#{idx:06d}"
            for out_field, candidates in time_fields.items():
                rel_val = None
                for source_field in candidates:
                    rel_val = minutes_since(admit_dt, row.get(source_field))
                    if rel_val is not None:
                        break
                new_row[out_field] = rel_val
            out.append(new_row)
        return out
