#!/usr/bin/env bash
#SBATCH --job-name=uwm-health1
#SBATCH --partition=tier1_charity_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/prj/inf_elandi/k2480198/uwm/runs/health_stage1/slurm-%j.out
#
# Health Stage-1 intrinsic-prediction eval (next-day wellness state, base/+profile/+memory).
# Generation-only, frozen Qwen3-4B via vLLM — no model save. ~314 test recs x 4 LLM arms.
# Submit:  sbatch scripts/health/run_health_stage1.slurm
# (Adjust --partition if tier1_charity_gpu is unavailable: interruptible_gpu / gpu.)
set -euo pipefail
cd "$(dirname "$0")/../.."

# Explicit env (env.sh propagation was flaky in backgrounded jobs — plan 2026-06-17).
export UWM_SCRATCH=/scratch/prj/inf_elandi/k2480198/uwm
export UWM_DATA=$UWM_SCRATCH/data
export HF_HOME=$UWM_SCRATCH/hf_cache
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
mkdir -p "$UWM_SCRATCH/runs/health_stage1"
PY=/scratch/users/k2480198/conda/.conda/envs/vllm_env/bin/python

echo "[health1] node=$(hostname) gpu=${SLURM_STEP_GPUS:-${CUDA_VISIBLE_DEVICES:-?}} $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# All conditions (persistence + pop-mean are no-LLM; the rest are frozen-LLM arms).
"$PY" scripts/health/eval_health_stage1.py "$@"

echo "[health1] done $(date)"
