# MedLoCoMo: A Long-Context Multi-Session Medical Dialogue Benchmark for Large Language Models

![MedLoCoMo pipeline overview](pipeline.png)

[Open the pipeline PDF](pipeline.pdf)

MedLoCoMo generates long-context, multi-session medical dialogue benchmark samples from MIMIC-IV v3.1 and MIMIC-IV-Note v2.2.

- One patient = one MedLoCoMo sample
- One admission = one session
- Only two speakers: `Doctor` and `Patient`
- Dialogues are grounded in:
  - discharge notes
  - radiology notes
  - diagnoses ICD
  - procedures ICD
  - microbiology

Admissions without discharge notes are skipped. Re-running the same patient replaces that patient's output directory instead of appending to it.

The core benchmark artifacts for each patient are `combined_conversation.json`, the full chronological multi-session dialogue, and `benchmark_qa.json`, the final QA set paired with that dialogue.

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

Use the existing `output/top_100_eligible_patients.csv` file as the patient shortlist for generation runs.

Generate the MedLoCoMo dialogue artifacts for one patient:

```bash
export OPENAI_API_KEY="YOUR_KEY"
python main.py generate-patient --subject-id 10000032
```

Override dataset, model, or output paths:

```bash
python main.py generate-patient \
  --mimiciv-dir /path/to/mimic-iv \
  --mimiciv-note-dir /path/to/mimic-iv-notes \
  --output-root /path/to/output/benchmark \
  --subject-id 10000032 \
  --model gpt-5.2
```

Generate QA for one patient after the dialogue artifacts exist:

```bash
python main.py generate-qa \
  --subject-id 10000032 \
  --model gpt-5.2
```

Generate dialogue and QA artifacts together:

```bash
python main.py generate-all \
  --subject-id 10000032 \
  --model gpt-5.2
```

`generate-qa` requires that every admission folder already contains both `conversation.json` and `summary.json`.

## MedLoCoMo Output Layout

```text
output/
  top_100_eligible_patients.csv
  benchmark/
    <subject_id>/
      combined_conversation.json  # core chronological multi-session dialogue
      patient_summary.json
      benchmark_qa.json           # core benchmark QA set
      cross_admission_qa.json
      <hadm_id>/
        formed_packet.json
        prompt_record.json
        conversation.json
        summary.json
        qa.json
```

Per-patient generation writes to `output/benchmark/_tmp/<subject_id>/` first and only replaces `output/benchmark/<subject_id>/` after a successful run.

QA generation writes to `output/benchmark/_tmp/qa_<subject_id>/` first and only replaces QA artifacts on success.
