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
  --output-root /path/to/output/benchmark \
  --subject-id 10000032 \
  --model gpt-5.2
```

Generate QA for one patient after the conversation benchmark exists:

```bash
python main.py generate-qa \
  --subject-id 10000032 \
  --model gpt-5.2
```


`generate-qa` uses two model calls per admission plus two cross-admission calls.
It requires that every admission folder already contains both `conversation.json`
and `summary.json`.

Evaluate one or more existing patient outputs with open-answer scoring:

```bash
python main.py evaluate --subject-id 10000032
python main.py evaluate --subject-ids 10000032 10000048 --models Qwen/Qwen3.5-4B
python main.py evaluate --patient-manifest /path/to/patients.txt --provider vllm --base-url http://127.0.0.1:8000/v1
python main.py evaluate-memory --subject-id 10000032 --models Qwen/Qwen3.5-4B
python main.py refresh-summary-tokens --patient-manifest health_benchmark/evaluation/cohorts/top10_patients.txt --tokenizer-model Qwen/Qwen3.5-4B
```

## Output Layout

```text
output/
  top_100_eligible_patients.csv
  benchmark/
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
  evaluation/
    <subject_id>/
      config.json
      context_stats.json
      benchmark_snapshot.json
      comparison/
        leaderboard.csv
        leaderboard.json
        breakdowns.csv
        breakdowns.json
        summary.md
      qwen3.5-4b/
      qwen3.5-4b-mem0/
        memory_store.json
        memory_events.jsonl
      qwen3.5-9b/
      qwen3.5-27b/
```

Per-patient generation writes to `output/benchmark/_tmp/<subject_id>/` first and only replaces `output/benchmark/<subject_id>/` after a successful run.
QA generation writes to `output/benchmark/_tmp/qa_<subject_id>/` first and only replaces QA artifacts on success.

## Notes
- Admission-level QA now writes exactly `3` questions per admission
- Each admission-level QA set contains exactly `2` regular short-answer questions and `1` adversarial short-answer question
- All QA items use the same open-answer schema: `qa_id`, `scope`, `question_type`, `question`, `answer`, and `evidence`
- The canonical adversarial answer stored in benchmark outputs is `the question is not answerable`
- Patient-level cross-admission QA count is derived from admission count as `3 * admission_count`
- Exactly `1/3` of cross-admission QA items are adversarial short-answer questions
- `qa.json` and `cross_admission_qa.json` keep grouped regular-then-adversarial ordering; `benchmark_qa.json` is the final deterministic shuffle
- Evaluation reads `combined_conversation.json` plus `benchmark_qa.json`, uses fixed 10-question batches, and writes model-separated outputs under `output/evaluation/<subject_id>/`
- Evaluation caps answer-model context at 128k tokens (`131072`), reserves `4096` output tokens plus an `8192` token safety margin, counts prompts with the Hugging Face tokenizer chat template, and truncates conversation context recent-first when the full prompt would exceed budget
- Evaluation prompts render the patient conversation as flat chronological `time | speaker | text` lines and hide internal `hadm_id`, admission boundaries, turn numbers, turn IDs, and global turn indices from LLM-visible context while retaining those IDs in JSON metadata for traceability
- `evaluate-memory` runs a separate Mem0-style memory evaluation and writes sibling model variants such as `qwen3.5-4b-mem0`; construction uses admission chunks, batched ADD/UPDATE/DELETE/NOOP memory updates, local sparse retrieval, and one-question answer prompts
- Evaluation scores answerable questions with normalized token overlap precision/recall/F1 plus binary `0/1` LLM judge accuracy; adversarial questions use exact abstention accuracy against `the question is not answerable` and are not LLM-judged
