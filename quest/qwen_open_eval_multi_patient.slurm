#!/usr/bin/env bash
#SBATCH --job-name=medbench-qwen-eval
#SBATCH --partition=gengpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --constraint=sxm
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --output=quest/slurm_logs/%x-%j.out

set -euo pipefail

echo "qwen_open_eval_multi_patient.slurm is retired for judged evaluation." >&2
echo "Use quest/qwen_open_eval_small_models.slurm for 4B/9B and quest/qwen_open_eval_27b_2gpu.slurm for 27B." >&2
exit 2
