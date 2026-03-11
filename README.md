# A Long Context Medical Benchmark

This project generates a simplified, generation-only long-context benchmark from MIMIC-IV v3.1 and MIMIC-IV-Note v2.2.

- One patient = one benchmark sample
- One admission = one session
- Only two speakers: `Doctor` and `Patient`
- Conversations are grounded only in:
  - discharge notes
  - radiology notes
  - diagnoses ICD
  - procedures ICD
  - microbiology

Admissions without discharge notes are skipped. Re-running the same patient replaces that patient's output directory instead of appending to it.

## Install

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset Layout

Expected defaults:

```text
data/
  mimic-iv/
    hosp/
  mimic-iv-notes/
```



## Commands

Use the existing
`output/top_100_eligible_patients.csv` file as the patient shortlist for
generation runs.


```bash
export OPENAI_API_KEY="YOUR_KEY"
python main.py generate-patient --subject-id 10000032
```


Override dataset, model, or output paths:

```bash
python main.py generate-patient \
  --mimiciv-dir /path/to/mimic-iv \
  --mimiciv-note-dir /path/to/mimic-iv-notes \
  --output-root /path/to/output \
  --subject-id 10000032 \
  --model gpt-5.2
```

Generate QA for one patient after the conversation benchmark exists:

```bash
python main.py generate-qa \
  --subject-id 10000032 \
  --model gpt-5.2
```


`generate-qa` uses one model call per admission plus one cross-admission call.
It requires that every admission folder already contains both `conversation.json`
and `summary.json`.

## Output Layout

```text
output/
  top_100_eligible_patients.csv
  <subject_id>/
    combined_conversation.json
    patient_summary.json
    cross_admission_qa.json
    benchmark_qa.json
    <hadm_id>/
      formed_packet.json
      prompt_record.json
      conversation.json
      summary.json
      qa.json
```

Per-patient generation writes to `output/_tmp/<subject_id>/` first and only replaces `output/<subject_id>/` after a successful run.
QA generation writes to `output/_tmp/qa_<subject_id>/` first and only replaces QA artifacts on success.

## Notes
- Admission-level QA writes `12` single-admission questions per admission by default
- Patient-level QA also writes `50` cross-admission questions and merges everything into `benchmark_qa.json`
