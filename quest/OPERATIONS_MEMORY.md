# Quest Operations Memory

Read this first: this is the short persistent Quest playbook distilled from live runs, while `quest/README.md` remains the fuller runbook.

## Purpose

Use this document for the operational facts that are easy to forget between sessions:

- the canonical Quest checkout and storage paths
- the baked-in generation defaults that are currently safe
- which launcher to use for each evaluation mode
- how to monitor jobs quickly
- how to recognize common failure patterns
- how to rerun only the right patients after a partial success

## Canonical Quest Defaults

Canonical Quest repo and paths:

```bash
PROJECT_ROOT=/projects/p33194/health-benchmark
MIMICIV_DIR=/projects/p33194/health-benchmark/data/mimic-iv
MIMICIV_NOTE_DIR=/projects/p33194/health-benchmark/data/mimic-iv-notes
OUTPUT_ROOT=/projects/p33194/medbench-output
HF_HOME=/projects/p33194/hf_cache
ENV_PREFIX=/projects/p33194/envs/medbench-qwen
```

Canonical generation launcher:

```bash
/projects/p33194/health-benchmark/quest/run_generate_all_qwen.slurm
```

Current safe generation defaults baked into that launcher:

- `#SBATCH --exclude=qgpu2014`
- `VLLM_ENGINE_READY_TIMEOUT_S=7200`
- `RETRY_LIMIT=3`
- `MODEL=Qwen/Qwen3-235B-A22B-Instruct-2507-FP8`
- `TENSOR_PARALLEL_SIZE=4`
- `MAX_MODEL_LEN=49152`
- `REASONING_PARSER=qwen3`

Interpretation:

- normal generation submissions should not need extra timeout, retry, or node-exclude flags
- override only when intentionally changing behavior

## Generation Commands

Standard single or multi-patient generation:

```bash
sbatch --account=p33194 /projects/p33194/health-benchmark/quest/run_generate_all_qwen.slurm 11826927
sbatch --account=p33194 /projects/p33194/health-benchmark/quest/run_generate_all_qwen.slurm 11826927 12345678
```

Interactive smoke test after getting an interactive GPU shell:

```bash
srun --account=p33194 --partition=gengpu --gres=gpu:4 --constraint=sxm --mem=256G --time=24:00:00 --pty bash
quest/debug_qwen_interactive.sh 11826927
```

If the shortlist helper is needed:

```bash
python /projects/p33194/health-benchmark/quest/emit_generation_submission_commands.py
```

## Evaluation Commands

Small-model judged evaluation (`4B`, `9B`) on `2` GPUs:

```bash
sbatch --account=p33194 /projects/p33194/health-benchmark/quest/qwen_open_eval_small_models.slurm 11826927 17207245 17072793
```

Small-model judged evaluation (`4B`, `9B`) on `1` GPU when queue time matters more:

```bash
sbatch --account=p33194 /projects/p33194/health-benchmark/quest/qwen_open_eval_small_models_1gpu.slurm 11826927 17207245 17072793
```

`27B` only on `2` GPUs, reusing existing patient outputs:

```bash
sbatch --account=p33194 /projects/p33194/health-benchmark/quest/qwen_open_eval_27b_2gpu.slurm 11826927 17207245 17072793
```

Launcher choice:

- use `qwen_open_eval_small_models_1gpu.slurm` when the `2`-GPU small-model job is stuck in queue and you want the lower-wait-time option
- use `qwen_open_eval_small_models.slurm` when you want the safer existing `2`-GPU judged path for `4B` and `9B`
- use `qwen_open_eval_27b_2gpu.slurm` only for `Qwen/Qwen3.5-27B` answer evaluation
- do not use `qwen_open_eval_multi_patient.slurm`; it is retired and redirects to the split launchers

Evaluation output locations:

- patient-level outputs: `/projects/p33194/medbench-output/<subject_id>/evaluation/`
- job-level summaries: `/projects/p33194/medbench-output/quest_job_outputs/<job_id>/`

## Queue Pressure

If GPU jobs are pending with reason `(Priority)`, the most likely cause is Quest fairshare, not a broken launcher.

Official Quest scheduling facts that matter here:

- Slurm priority on Quest depends on requested resources plus the fairshare score for your Slurm account
- recent resource use by you or other users on the same General Access account lowers the priority of current jobs
- job age helps over time, but fairshare recovery is gradual rather than immediate
- accurate wall-time, core, and memory requests help Slurm backfill jobs sooner

Quest GPU pool facts that matter here:

- the Quest specs page lists `58` General Access GPU nodes total
- with `--constraint=sxm`, these workflows effectively compete for the `24` H100 SXM nodes plus the `18` 80 GB A100 SXM nodes, or about `42` eligible nodes
- Quest documents an `8`-GPU running-user cap on General Access GPU partitions; if you hit it, pending jobs show `(QOSMaxGRESPerUser)` rather than `(Priority)` or `(Resources)`

Workload difficulty by placement:

- `4`-GPU generation jobs are hardest to place
- `2`-GPU `27B` answer evaluation is next
- `1`-GPU small-model judged evaluation is easiest

Queue triage:

- `(Priority)`: fairshare / account usage is likely the limiter
- `(Resources)`: no eligible SXM GPUs are free right now
- `(QOSMaxGRESPerUser)`: too many GPUs are already running under your user on General Access

Current operational guidance:

- keep `--constraint=sxm` for these generation and judged-eval flows because they need the 80 GB GPU tier
- do not add a more specific GPU type request such as `--gres=gpu:h100:X` unless a real compatibility issue requires it
- when the queue is busy, submit in this order: `1`-GPU small eval, then `2`-GPU `27B` eval, then `4`-GPU generation
- after one successful run per job family, shrink wall times to about `1.5x` the observed elapsed time instead of leaving them at `24:00:00`

Useful commands:

```bash
squeue -u kqm5166
squeue -j <JOB_ID> --start
sacct -j <JOB_ID> --format=JobID,State,ExitCode,Elapsed
```

## Monitoring And Debugging

Queue and accounting:

```bash
squeue -u kqm5166
sacct -j <JOB_ID> --format=JobID,State,ExitCode,Elapsed
```

Find the batch log:

```bash
ls -l /projects/p33194/health-benchmark/quest/slurm_logs/*-<JOB_ID>.out
```

Check the tail of the main job log:

```bash
tail -n 80 /projects/p33194/health-benchmark/quest/slurm_logs/*-<JOB_ID>.out
```

Check the vLLM log:

```bash
tail -n 120 /projects/p33194/medbench-output/logs/vllm/vllm_<JOB_ID>.log
```

Find the batch summary from a finished job:

```bash
grep -o '/gpfs/projects[^"]*batch_summary.json' /projects/p33194/health-benchmark/quest/slurm_logs/*-<JOB_ID>.out | tail -n 1
cat /gpfs/projects/p33194/medbench-output/runs/<run_id>/batch_summary.json
```

Quick patient-output check:

```bash
for SID in 11826927 17207245 17072793; do
  ROOT="/projects/p33194/medbench-output/$SID"
  echo "===== SUBJECT $SID ====="
  test -f "$ROOT/combined_conversation.json" && echo "conversation: yes" || echo "conversation: no"
  test -f "$ROOT/patient_summary.json" && echo "patient_summary: yes" || echo "patient_summary: no"
  test -f "$ROOT/cross_admission_qa.json" && echo "cross_admission_qa: yes" || echo "cross_admission_qa: no"
  test -f "$ROOT/benchmark_qa.json" && echo "benchmark_qa: yes" || echo "benchmark_qa: no"
  echo
done
```

## Known Failure Patterns

### Wrong Quest checkout or stale files

Signs:

- expected script is missing
- launcher behavior on Quest does not match local edits
- tests fail because Quest is still asserting old defaults

Response:

- use the lowercase canonical checkout: `/projects/p33194/health-benchmark`
- sync the exact file or subtree again
- verify with `grep` before submitting

### Broken shell or PATH on Quest

Signs:

- `sbatch`, `scancel`, `dirname`, or even `bash` appear missing
- logs show `/usr/bin/env: ‘bash’: Not a directory`

Response:

```bash
exec /bin/bash -l
SQUEUE_BIN=$(command -v squeue)
SLURM_BIN_DIR=${SQUEUE_BIN%/*}
export PATH="$SLURM_BIN_DIR:/usr/bin:/bin:/usr/sbin:/sbin"
hash -r
```

### vLLM startup timeout or engine-core initialization failure

Signs:

- `vLLM server did not become ready`
- `vLLM exited before becoming ready`
- repeated shared-memory broadcast waits
- engine-core startup timeout

Response:

- first inspect `/projects/p33194/medbench-output/logs/vllm/vllm_<JOB_ID>.log`
- prefer the baked-in `7200` timeout
- keep the default `qgpu2014` exclusion
- if startup is still flaky, rerun the failed subject(s) rather than a large batch

### Hugging Face transient 503 failures

Signs:

- `HTTP Error 503`
- `Error retrieving file list`
- model startup dies before requests begin

Response:

- treat as transient infrastructure failure
- rerun the job later
- do not interpret partial output as a model-quality signal

### Generation validation failures

Common messages seen:

- `conversation line <n> has empty text`
- `conversation timestamps must be monotonically increasing`
- `conversation timestamp is outside the admission window`
- `summary.admission_end must match the packet admission_end exactly`

Response:

- keep any successful patient outputs
- inspect `batch_summary.json` for subject-level outcomes
- rerun only failed subjects
- if failures cluster, reduce batch size

### QA validation failures

Common message seen:

- `qas[...].answer must contain at most 10 words for non-adversarial questions`

Response:

- treat this as a patient-level content failure, not a full-batch infrastructure failure
- keep successful patient outputs from the same batch
- rerun only the failed subject(s)

## Sync Rules

Patient outputs and Quest job outputs are different sync targets.

Patient artifacts:

- Quest source: `/projects/p33194/medbench-output/<subject_id>/`
- local destination: `output/<subject_id>/`

Job artifacts:

- Quest source: `/projects/p33194/medbench-output/quest_job_outputs/<job_id>/`
- keep remote unless intentionally archived outside local `output/`

Correct patient-only sync:

```bash
rsync -av --progress \
  kqm5166@login.quest.northwestern.edu:/projects/p33194/medbench-output/11826927/ \
  /Users/zhangzeyu/Downloads/Medical-Benchmark/output/11826927/
```

Multiple patients:

```bash
for SID in 11826927 17207245 17072793; do
  rsync -av --progress \
    "kqm5166@login.quest.northwestern.edu:/projects/p33194/medbench-output/$SID/" \
    "/Users/zhangzeyu/Downloads/Medical-Benchmark/output/$SID/"
done
```

Never broad-sync `/projects/p33194/medbench-output/` into local `output/`, because that can pull in `quest_job_outputs/`.

## Rerun Heuristics

Operational rules we learned:

- keep successful patient outputs even when the overall batch exits `1`
- use `batch_summary.json` as the source of truth for patient-level success vs failure
- rerun only failed subjects after a mixed batch
- if a new failure mode appears, shrink the next wave instead of retrying a large batch unchanged
- prefer singleton reruns for stubborn patients
- use smaller batches while isolating content-validation failures
- once a pattern looks stable again, scale back up gradually

Partial-success next step:

1. inspect `batch_summary.json`
2. list succeeded and failed subject ids
3. verify the four required generation artifacts for each succeeded subject
4. keep successful patient folders
5. rerun only the failed subject(s), preferably in smaller batches
