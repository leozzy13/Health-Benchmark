# Healthcare Benchmark Generation (MIMIC-IV + MIMIC-IV-Note)

This project generates benchmark samples for a **single patient** by:

1. extracting a deterministic per-admission EHR packet from MIMIC-IV + MIMIC-IV-Note,
2. rendering a deterministic prompt (with mandatory discharge notes),
3. calling an LLM model to synthesize a chronological doctor-patient inpatient conversation,
4. writing reproducible artifacts (`packet.json`, prompts, logs, `conversation.jsonl`, `summary.json`).

Scope in this implementation:
- `Benchmark generation`: implemented


## Folder Structure

```text
medical_benchmark/
├── main.py
├── requirements.txt
├── README.md
└── med_benchmark/
    ├── config.py
    ├── duckdb_store.py
    ├── extractor.py
    ├── llm_client.py
    ├── pipeline.py
    ├── prompting.py
    ├── utils.py
    ├── validation.py
    └── writers.py
```

## Prerequisites

- Local datasets already present (expected by default):
  - `data/mimic-iv/`
  - `data/mimic-iv-notes/`
- OpenAI API key

## Install

From the repository root:

```bash
python3 -m venv medical_benchmark/.venv
source medical_benchmark/.venv/bin/activate
pip install -r medical_benchmark/requirements.txt
```

## Main Script Usage

### 1) Build top cohort CSV 

Writes the deterministic top-N patient cohort by admission count.

```bash
python medical_benchmark/main.py build-cohort --limit 1000
```

Output (default):
- `medical_benchmark/output/top1000_by_admission_count.csv`

### 2) Full generation for one patient 

```bash
export OPENAI_API_KEY="YOUR_KEY"

python medical_benchmark/main.py generate-patient \
  --subject-id 10000032 \
  --model gpt-4.1-mini
```

### 3) Generate only one admission for a patient

```bash
python medical_benchmark/main.py generate-patient \
  --subject-id 10000032 \
  --hadm-id 22595853 \
  --model gpt-4.1-mini
```

## Important CLI Options

- `--max-admissions N`: process only first `N` admissions after sorting by `admittime, hadm_id`.
- `--include-admissions-without-discharge`: disabled by default (the prompt policy requires discharge notes).
- `--retry-limit N`: retries for transport/schema-repair loop.
- `--max-output-tokens N`: override model output token cap.
- `--row-cap-labs N`, `--row-cap-radiology N`, `--row-cap-emar N`: deterministic row caps for prompt size control.
- Re-running generation for the same patient replaces that patient's existing output folder.

## Output Artifacts (per admission)

Generated under:

```text
medical_benchmark/output/
├── top1000_by_admission_count.csv
└── <SUBJECT_ID>/
    ├── conversation_details.jsonl    # all turns with full metadata (adds subject_id, hadm_id)
    ├── conversation_only.json        # grouped by admission; each turn keeps only speaker + text
    ├── patient_manifest.json
    └── admissions/<HADM_ID>/
        ├── packet.json
        ├── input_data_manifest.json
        ├── prompt_record.json
        ├── model_call_record.json
        ├── raw_model_output.json
        ├── conversation.jsonl
        ├── summary.json
        └── unlinked_notes.json        # optional (if unlinked radiology notes exist)
```
