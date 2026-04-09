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
export VLLM_ENGINE_READY_TIMEOUT_S=3600
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
- defaults to `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8` with the FP8 Quest profile (`TENSOR_PARALLEL_SIZE=4`, `MAX_MODEL_LEN=49152`, `REASONING_PARSER=qwen3`, `VLLM_ENGINE_READY_TIMEOUT_S=3600`),
- runs `python main.py generate-all --provider vllm ...`,
- prints the final `batch_summary.json` path before exiting.

## Notes

- The default profile is 1 node, 4 GPUs, `gengpu`, `sxm`, 256 GB RAM, and 24 hours.
- On Quest, `sxm` is the documented way to request the 80 GB general-access GPU tier, which may allocate either A100 80 GB or H100 80 GB GPUs.
- The standard Quest Qwen path now defaults to `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8`; keep using `MODEL=...` if you want to override it for a one-off run.
- `MAX_MODEL_LEN` now defaults to `49152` to leave more headroom for large patient prompts on the FP8 235B path without jumping all the way to `65536`.
- The first cold start can spend well over 30 minutes downloading weights and compiling graphs for the FP8 235B profile, so the scripts default `VLLM_ENGINE_READY_TIMEOUT_S=3600`.
- The scripts auto-activate `/projects/p33194/envs/medbench-qwen` through `/hpc/software/mamba/24.3.0` by default, so you do not need to pre-activate that env in the submitting shell.
- The scripts do not rely on `/projects/p33194/envs/medbench-qwen/bin/vllm`; they launch the pinned container directly.
- `/projects/...` paths are intentional here because the Singularity-backed `vllm` setup can treat `/gpfs/projects/...` as read-only even when the logical `/projects/...` path is writable.
- `generate-all` exits `0` only when every patient succeeds. If any patient fails, the command exits `1` and the batch summary records the failure details.

## Multi-Patient Evaluation

Use the evaluation Slurm launcher when you want to score existing patient outputs with the fixed Qwen3.5 trio (`4B`, `9B`, `27B`) inside one Quest job.

Submit explicit patient ids:

```bash
sbatch --account=p33194 /projects/p33194/health-benchmark/quest/qwen_open_eval_multi_patient.slurm 11826927 17207245 17072793
```

Or submit a manifest file:

```bash
sbatch --account=p33194 /projects/p33194/health-benchmark/quest/qwen_open_eval_multi_patient.slurm \
  --patient-manifest /projects/p33194/health-benchmark/patients.txt
```

The evaluation job:

- requests `2` SXM nodes with `4` GPUs per node so the `Qwen/Qwen3.5-27B` phase can run with `tensor_parallel_size=8`
- launches `Qwen/Qwen3.5-4B`, `Qwen/Qwen3.5-9B`, and `Qwen/Qwen3.5-27B` sequentially
- reuses the same patient manifest for every model phase
- writes patient-level evaluation outputs under `output/<subject_id>/evaluation/`
- writes a Quest job summary under `medbench-output/quest_job_outputs/<job_id>/`

For a lighter first-pass evaluation with only `Qwen/Qwen3.5-4B` and `Qwen/Qwen3.5-9B`, use the small-model launcher:

```bash
sbatch --account=p33194 /projects/p33194/health-benchmark/quest/qwen_open_eval_small_models.slurm 11826927 17207245 17072793
```

The small-model evaluation job:

- requests `1` SXM node with `1` GPU
- launches only `Qwen/Qwen3.5-4B` and `Qwen/Qwen3.5-9B`
- writes the same patient-level evaluation outputs under `output/<subject_id>/evaluation/`
- writes the same Quest job summary under `medbench-output/quest_job_outputs/<job_id>/`

To add `Qwen/Qwen3.5-27B` on top of existing patient evaluation folders without rerunning `Qwen/Qwen3.5-9B`, use the dedicated 2-GPU launcher:

```bash
sbatch --account=p33194 /projects/p33194/health-benchmark/quest/qwen_open_eval_27b_2gpu.slurm 11826927 17207245 17072793
```

The 27B-only evaluation job:

- requests `1` SXM node with `2` GPUs
- launches only `Qwen/Qwen3.5-27B`
- uses `tensor_parallel_size=2` with `max_model_len=131072`
- writes `output/<subject_id>/evaluation/qwen3.5-27b/`
- rebuilds the same `comparison/leaderboard.*` files so existing `qwen3.5-9b` results remain alongside the new `qwen3.5-27b` row

Helper scripts used by the Slurm launcher:

```text
quest/qwen_open_eval_multi_patient.slurm
quest/qwen_open_eval_small_models.slurm
quest/qwen_open_eval_27b_2gpu.slurm
quest/run_multi_patient_eval_job.sh
quest/launch_vllm_server.sh
quest/stop_vllm_server.sh
quest/wait_for_server.py
```
