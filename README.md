# Healthcare Benchmark Generation

This project generates benchmark samples for a single patient from MIMIC-IV and MIMIC-IV-Note by:

1. extracting a deterministic per-admission EHR packet,
2. rendering a deterministic prompt with mandatory discharge-note grounding,
3. calling an OpenAI model to synthesize a chronological inpatient conversation,
4. writing reproducible artifacts for each admission.

## Prerequisites

- Local datasets available under:
  - `data/mimic-iv/`
  - `data/mimic-iv-notes/`
- An OpenAI API key in `OPENAI_API_KEY` or another env var passed with `--api-key-env`

The repository includes only the `data/` directory structure. CSV data files are intentionally not committed.

## Install

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## CLI Usage

Build the top cohort CSV:

```bash
python main.py build-cohort --limit 1000
```

Generate all admissions for one patient:

```bash
export OPENAI_API_KEY="YOUR_KEY"

python main.py generate-patient \
  --subject-id 10000032 \
  --model gpt-4.1-mini
```

Generate only one admission:

```bash
python main.py generate-patient \
  --subject-id 10000032 \
  --hadm-id 22595853 \
  --model gpt-4.1-mini
```

Use custom dataset or output paths:

```bash
python main.py generate-patient \
  --mimiciv-dir /path/to/mimic-iv \
  --mimiciv-note-dir /path/to/mimic-iv-notes \
  --output-root /path/to/output \
  --subject-id 10000032 \
  --model gpt-4.1-mini
```

## Important Options

- `--mimiciv-dir PATH`: override the MIMIC-IV directory
- `--mimiciv-note-dir PATH`: override the MIMIC-IV-Note directory
- `--output-root PATH`: write outputs to a custom location
- `--max-admissions N`: cap the number of admissions processed
- `--include-admissions-without-discharge`: include admissions without discharge notes
- `--retry-limit N`: retry count for model/schema-repair attempts
- `--max-output-tokens N`: override the OpenAI output token cap
- `--api-key-env NAME`: env var name containing the API key
- `--openai-base-url URL`: optional OpenAI-compatible base URL override
- `--row-cap-labs N`, `--row-cap-radiology N`, `--row-cap-emar N`: deterministic prompt-size controls

## Output Layout

Generated under `output/` by default:

```text
output/
├── top1000_by_admission_count.csv
└── <SUBJECT_ID>/
    ├── conversation_details.jsonl
    ├── conversation_only.json
    ├── patient_manifest.json
    └── admissions/<HADM_ID>/
        ├── packet.json
        ├── input_data_manifest.json
        ├── prompt_record.json
        ├── model_call_record.json
        ├── raw_model_output.json
        ├── conversation.jsonl
        ├── summary.json
        └── unlinked_notes.json
```

Re-running generation for the same patient replaces that patient's existing output folder.
