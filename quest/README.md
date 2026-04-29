# Quest Qwen Runbook

These scripts launch the benchmark workflows against a node-local `vllm` server.

## Quest Defaults And Overrides

These launchers now work out of the box for the canonical lowercase Quest checkout under account `p33194`. If you do not export any path variables, they fall back to:

```bash
PROJECT_ROOT=/projects/p33194/health-benchmark
MIMICIV_DIR=/projects/p33194/health-benchmark/data/mimic-iv
MIMICIV_NOTE_DIR=/projects/p33194/health-benchmark/data/mimic-iv-notes
OUTPUT_ROOT=/projects/p33194/medbench-output
HF_HOME=/projects/p33194/hf_cache
MAMBA_ROOT=/hpc/software/mamba/24.3.0
ENV_PREFIX=/projects/p33194/envs/medbench-qwen
SINGULARITY_BIN=/software/singularity/3.8.1/bin/singularity
VLLM_IMAGE=/projects/p33194/containers/vllm-openai_latest.sif
```

The lowercase repo at `/projects/p33194/health-benchmark` is the canonical Quest checkout going forward.

You can still override any of those paths per run. Preferred Quest-style overrides:

```bash
export PROJECT_ROOT=/projects/<account>/health-benchmark
export MIMICIV_DIR=/projects/<account>/mimic-iv
export MIMICIV_NOTE_DIR=/projects/<account>/mimic-iv-note
export OUTPUT_ROOT=/projects/<account>/medbench-output
export HF_HOME=/projects/<account>/hf_cache
export MAMBA_ROOT=/hpc/software/mamba/24.3.0
export ENV_PREFIX=/projects/<account>/envs/medbench-qwen
export SINGULARITY_BIN=/software/singularity/3.8.1/bin/singularity
export VLLM_IMAGE=/projects/<account>/containers/vllm-openai_latest.sif
```

Accepted aliases:

```bash
export MIMIC_IV_DIR=/projects/<account>/mimic-iv
export MIMIC_IV_NOTE_DIR=/projects/<account>/mimic-iv-note
export MEDBENCH_OUTPUT_ROOT=/projects/<account>/medbench-output
```

If both styles are set, the launchers use `MIMICIV_DIR`, `MIMICIV_NOTE_DIR`, and `OUTPUT_ROOT`.
They also auto-activate the resolved Quest env before they run `vllm` or `python`.
They default the Hugging Face cache to `/projects/p33194/hf_cache` so the cache stays writable and stable across Quest jobs.
They launch `vllm` directly from the pinned Singularity image instead of depending on an external env wrapper script.

Optional overrides:

```bash
export MODEL=Qwen/Qwen3-235B-A22B-Instruct-2507-FP8
export VLLM_PORT=8000
export TENSOR_PARALLEL_SIZE=4
export MAX_MODEL_LEN=49152
export REASONING_PARSER=qwen3
export VLLM_ENGINE_READY_TIMEOUT_S=7200
export RETRY_LIMIT=3
```

## Interactive Smoke Test

Run this after you already have an interactive GPU allocation:

```bash
quest/debug_qwen_interactive.sh 11826927
```

For an interactive allocation that matches the batch scripts closely, request the same 80 GB SXM tier first:

```bash
srun --account=p33194 --partition=gengpu --gres=gpu:4 --constraint=sxm --mem=256G --time=24:00:00 --pty bash
```

## Batch Run With Slurm

Submit one Quest job and process several patients sequentially. With the baked-in defaults, the common case is now just:

```bash
sbatch --account=p33194 /projects/p33194/health-benchmark/quest/run_generate_all_qwen.slurm 11826927 12345678 13579246
```

The canonical generation launcher now bakes in the current safe defaults for routine runs:

- excludes `qgpu2014` by default,
- defaults `VLLM_ENGINE_READY_TIMEOUT_S=7200`,
- defaults `RETRY_LIMIT=3` and forwards it to `main.py generate-all`.
- writes Slurm batch logs to `quest/slurm_logs/%x-%j.out`,
- writes generation vLLM logs to `medbench-output/logs/vllm/vllm_<JOB_ID>.log`.

You only need extra `sbatch` flags or env overrides when you intentionally want a different node policy or retry/timeout setting.

You can still submit from inside the repo if you prefer:

```bash
cd /projects/<account>/health-benchmark
sbatch --account=<account> quest/run_generate_all_qwen.slurm 11826927 12345678 13579246
```

If you need to submit from outside the repo checkout, set `PROJECT_ROOT` explicitly:

```bash
PROJECT_ROOT=/projects/<account>/health-benchmark \
  sbatch --account=<account> /projects/<account>/health-benchmark/quest/run_generate_all_qwen.slurm 11826927 12345678 13579246
```

Recommended self-contained submission pattern when you want every path to be explicit:

```bash
PROJECT_ROOT=/projects/<account>/health-benchmark \
MIMICIV_DIR=/projects/<account>/health-benchmark/data/mimic-iv \
MIMICIV_NOTE_DIR=/projects/<account>/health-benchmark/data/mimic-iv-notes \
OUTPUT_ROOT=/projects/<account>/medbench-output \
HF_HOME=/projects/<account>/hf_cache \
sbatch --account=<account> /projects/<account>/health-benchmark/quest/run_generate_all_qwen.slurm 11826927
```

This avoids depending on whichever directory or exported variables happen to be active in the current shell.

If you want the repo to generate the smoke-test plus low-eligibility submission commands for you, use:

```bash
python /projects/p33194/health-benchmark/quest/emit_generation_submission_commands.py
```

That helper emits:

- Job 1: `11826927` as a single-patient smoke test
- Jobs 2-11: the `50` patients with the lowest `eligible_admissions` from `output/top_100_eligible_patients.csv`
- stable tie-breaking by CSV order, with `5` patients per job

The `quest_generation_complete` column in `output/top_100_eligible_patients.csv` is not updated automatically by generation jobs. Refresh it manually from the current Quest patient outputs when you want the shortlist status to reflect the latest runs:

```bash
cd /projects/p33194/health-benchmark
python3 quest/update_shortlist_generation_status.py \
  --csv output/top_100_eligible_patients.csv \
  --quest-output-root /projects/p33194/medbench-output
```

If you are unsure which Quest repo copy you are using, verify the canonical lowercase checkout first:

```bash
grep -nE 'resolve_required_env|PROJECT_ROOT|SLURM_SUBMIT_DIR|python "\\$REPO_ROOT/main.py"' \
  /projects/<account>/health-benchmark/quest/run_generate_all_qwen.slurm
grep -n '\\${MIMICIV_DIR:?|\\${MIMICIV_NOTE_DIR:?|\\${OUTPUT_ROOT:?' \
  /projects/<account>/health-benchmark/quest/run_generate_all_qwen.slurm
```

If an older uppercase checkout exists, inspect it separately and avoid submitting from it unless you intentionally keep it updated:

```bash
test -f /projects/<account>/Health-Benchmark/quest/run_generate_all_qwen.slurm && \
  grep -nE 'resolve_required_env|PROJECT_ROOT|SLURM_SUBMIT_DIR|python "\\$REPO_ROOT/main.py"' \
  /projects/<account>/Health-Benchmark/quest/run_generate_all_qwen.slurm
```

The Slurm script:

- auto-activates the resolved Quest mamba env and prints the resulting `python`, Singularity binary, and vLLM image paths,
- keeps launcher-resolved paths under `/projects/...` and uses `/projects/p33194/hf_cache` as the default Hugging Face cache for container compatibility,
- verifies GPU visibility with `nvidia-smi -L`, logs the node plus GPU count and model summary, and fails fast if the visible GPU count is lower than `TENSOR_PARALLEL_SIZE`,
- starts one local `vllm serve` process from the pinned Singularity image with `--nv` and `-B /projects:/projects`,
- resolves the repo root from `PROJECT_ROOT`, `SLURM_SUBMIT_DIR`, or the baked-in default and fails fast if it cannot validate the checkout,
- waits for the OpenAI-compatible endpoint to become ready and fails immediately if the background `vllm` process exits first,
- defaults to `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8` with the FP8 Quest profile (`TENSOR_PARALLEL_SIZE=4`, `MAX_MODEL_LEN=49152`, `REASONING_PARSER=qwen3`, `VLLM_ENGINE_READY_TIMEOUT_S=7200`, `RETRY_LIMIT=3`),
- excludes `qgpu2014` by default for the canonical generation path,
- runs `python main.py generate-all --provider vllm ...`,
- prints the final `batch_summary.json` path before exiting.

## Notes

- The default profile is 1 node, 4 GPUs, `gengpu`, `sxm`, 256 GB RAM, and 24 hours.
- On Quest, `sxm` is the documented way to request the 80 GB general-access GPU tier, which may allocate either A100 80 GB or H100 80 GB GPUs.
- The standard Quest Qwen path now defaults to `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8`; keep using `MODEL=...` if you want to override it for a one-off run.
- `MAX_MODEL_LEN` now defaults to `49152` to leave more headroom for large patient prompts on the FP8 235B path without jumping all the way to `65536`.
- The canonical batch generation launcher now skips `qgpu2014` by default. If you ever need a different node policy, override it explicitly with `sbatch` flags.
- The first cold start can spend well over 30 minutes downloading weights and compiling graphs for the FP8 235B profile, so the canonical batch generation launcher now defaults `VLLM_ENGINE_READY_TIMEOUT_S=7200`.
- The canonical batch generation launcher also defaults `RETRY_LIMIT=3` so transient validation misses get a few more repair attempts before the patient is marked failed.
- The scripts auto-activate `/projects/p33194/envs/medbench-qwen` through `/hpc/software/mamba/24.3.0` by default, so you do not need to pre-activate that env in the submitting shell.
- The scripts do not rely on `/projects/p33194/envs/medbench-qwen/bin/vllm`; they launch the pinned container directly.
- `/projects/...` paths are intentional here because the Singularity-backed `vllm` setup can treat `/gpfs/projects/...` as read-only even when the logical `/projects/...` path is writable.
- `generate-all` exits `0` only when every patient succeeds. If any patient fails, the command exits `1` and the batch summary records the failure details.

## Syncing Results Back To Local

Treat patient outputs and Quest job outputs as separate sync targets.

- Patient artifacts live under `/projects/<account>/medbench-output/<subject_id>/` and should sync into your local `output/<subject_id>/`
- Quest job artifacts live under `/projects/<account>/medbench-output/quest_job_outputs/<job_id>/` and should stay remote unless you intentionally archive them somewhere outside local `output/`

Sync a single patient folder into your local repo:

```bash
rsync -av --progress \
  kqm5166@login.quest.northwestern.edu:/projects/p33194/medbench-output/11826927/ \
  /Users/zhangzeyu/Downloads/Medical-Benchmark/output/11826927/
```

Sync several patient folders without touching `quest_job_outputs/`:

```bash
for SID in 11826927 17207245 17072793; do
  rsync -av --progress \
    "kqm5166@login.quest.northwestern.edu:/projects/p33194/medbench-output/$SID/" \
    "/Users/zhangzeyu/Downloads/Medical-Benchmark/output/$SID/"
done
```

Do not use `/projects/<account>/medbench-output/` as a broad `rsync` source into local `output/`, because that can pull `quest_job_outputs/` into the patient output tree.

If you want Quest-side job logs locally, sync them to a separate folder such as `quest_job_outputs_local/` rather than mixing them into local `output/`.

## Multi-Patient Evaluation

Use the evaluation Slurm launcher when you want to score existing patient outputs with the fixed Qwen3.5 trio (`4B`, `9B`, `27B`) inside one Quest job.

Submit explicit patient ids:

Judged temporal evaluation now uses split Quest launchers instead of the old all-in-one trio path.

For the `Qwen/Qwen3.5-4B` and `Qwen/Qwen3.5-9B` answer models, use the small-model judged launcher:

```bash
sbatch --account=p33194 /projects/p33194/health-benchmark/quest/qwen_open_eval_small_models.slurm 11826927 17207245 17072793
```

The small-model evaluation job:

- requests `1` SXM node with `2` GPUs
- launches only `Qwen/Qwen3.5-4B` and `Qwen/Qwen3.5-9B` as answer models
- runs each small model serially: answer generation first on one GPU, then LLM judging with `Qwen/Qwen3.5-27B` on both allocated GPUs
- uses internal `main.py evaluate --stage answers` then `--stage judge` so the final outputs still include `llm_score`
- writes the same patient-level evaluation outputs under `output/<subject_id>/evaluation/`
- writes the same Quest job summary under `medbench-output/quest_job_outputs/<job_id>/`

If queue pressure is high and you want the lower-wait-time option, use the dedicated `1`-GPU small-model launcher:

```bash
sbatch --account=p33194 /projects/p33194/health-benchmark/quest/qwen_open_eval_small_models_1gpu.slurm 11826927 17207245 17072793
```

The `1`-GPU small-model evaluation job:

- requests `1` SXM node with `1` GPU
- still evaluates only `Qwen/Qwen3.5-4B` and `Qwen/Qwen3.5-9B`
- still runs each small model serially with `answers` first and `judge` second
- uses `Qwen/Qwen3.5-27B` as the judge on the same single GPU with `tensor_parallel_size=1`
- uses a smaller judge-only `max_model_len=32768` because judge prompts are short
- writes the same patient-level outputs and Quest job summaries as the `2`-GPU small-model launcher

For `Qwen/Qwen3.5-27B` judged evaluation, use the dedicated 2-GPU launcher:

```bash
sbatch --account=p33194 /projects/p33194/health-benchmark/quest/qwen_open_eval_27b_2gpu.slurm 11826927 17207245 17072793
```

The 27B-only evaluation job:

- requests `1` SXM node with `2` GPUs
- launches only `Qwen/Qwen3.5-27B`
- uses `tensor_parallel_size=2` with `max_model_len=131072`
- reuses the same `27B` server sequentially for answer generation and LLM judging
- writes `output/<subject_id>/evaluation/qwen3.5-27b/`
- rebuilds the same `comparison/leaderboard.*` files so existing `qwen3.5-9b` results remain alongside the new `qwen3.5-27b` row

The old `quest/qwen_open_eval_multi_patient.slurm` launcher is intentionally retired and now exits with a redirect message. Use the split small-model and 27B launchers above instead.

## Queueing On Quest

If your Quest GPU jobs sit in `PD` with reason `(Priority)`, that usually points to scheduler fairshare rather than a broken launcher.

The official Quest Slurm guide says job start order depends on requested resources and your fairshare score, which reflects your account priority and recent Quest usage. It also says accurate resource requests, especially wall time, help Slurm backfill jobs into open windows.

For this repo, the practical queue pressures are:

- all current GPU jobs run on `gengpu`, which is a high-demand shared partition
- `--constraint=sxm` is currently justified because these workflows need the 80 GB GPU tier, but it narrows placement to the SXM subset of Quest General Access GPUs
- the Quest specs list `58` General Access GPU nodes total, but only the `24` H100 SXM nodes and `18` 80 GB A100 SXM nodes match the current `sxm` requirement, so the effective pool is about `42` nodes
- `4`-GPU generation jobs are the hardest to place because they need a full SXM node with all four GPUs available at once
- `2`-GPU `27B` answer evaluation is easier than generation but still needs a partially free SXM node
- the `1`-GPU small-model judged path is the most queue-friendly option, but it still needs `sxm` because the `27B` judge wants an 80 GB card
- the GPU docs warn that asking for a more specific GPU type, such as `--gres=gpu:h100:X`, usually increases wait time unless the workload truly requires it

Pending reason quick guide:

- `(Priority)`: fairshare / scheduler ordering is the likely bottleneck
- `(Resources)`: not enough eligible SXM GPUs are free right now
- `(QOSMaxGRESPerUser)`: you are above Quest's documented `8`-GPU running-user cap on General Access GPU partitions

Recommended submission order when the queue is congested:

- `1`-GPU small-model judged eval first
- `2`-GPU `27B` eval second
- `4`-GPU generation last

Recommended operational habit:

- do not leave every GPU job at `24:00:00` after you have real runtime data
- collect elapsed times with `sacct`
- set future wall times to roughly `1.5x` the observed elapsed time for that job family
- tighten the small-model eval wall time first, since it is the easiest job to backfill

Useful commands:

```bash
squeue -u kqm5166
squeue -j <JOB_ID> --start
sacct -j <JOB_ID> --format=JobID,State,ExitCode,Elapsed
```

Official references:

- `https://rcdsdocs.it.northwestern.edu/systems/quest/user-guide/slurm/slurm.html`
- `https://rcdsdocs.it.northwestern.edu/systems/quest/user-guide/gpu/gpu.html`
- `https://rcdsdocs.it.northwestern.edu/systems/quest/specs/quest-specs.html`

Helper scripts used by the Slurm launcher:

```text
quest/qwen_open_eval_multi_patient.slurm
quest/qwen_open_eval_small_models.slurm
quest/qwen_open_eval_small_models_1gpu.slurm
quest/qwen_open_eval_27b_2gpu.slurm
quest/run_multi_patient_eval_job.sh
quest/launch_vllm_server.sh
quest/stop_vllm_server.sh
quest/wait_for_server.py
```
