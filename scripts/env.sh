#!/usr/bin/env bash
# user_world_model — shared environment. Source at the top of every run:
#   source scripts/env.sh
# See CONVENTIONS.md §0 (env), §2 (storage), §4 (concurrency).

# --- conda env (KCL CREATE: vllm_env lives on scratch) ---
if [ "${CONDA_DEFAULT_ENV:-}" != "vllm_env" ]; then
    # shellcheck disable=SC1091
    source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null || true
    conda activate vllm_env 2>/dev/null || echo "[env.sh] WARN: could not activate vllm_env" >&2
fi

# --- large-artifact root (cllm AND inf_embrace_llm are FULL -> inf_elandi) ---
# 2026-06-17: inf_embrace_llm project quota filled (shared); migrated to inf_elandi.
export UWM_SCRATCH="${UWM_SCRATCH:-/scratch/prj/inf_elandi/k2480198/uwm}"
export HF_HOME="${HF_HOME:-$UWM_SCRATCH/hf_cache}"

# --- repo root (works regardless of which path alias the node cd'd through) ---
UWM_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export UWM_REPO

# --- canonical scratch layout (CONVENTIONS.md §2) ---
export UWM_DATA="$UWM_SCRATCH/data"
export UWM_MODELS="$UWM_SCRATCH/models"
export UWM_RUNS="$UWM_SCRATCH/runs"
export UWM_SHARED="$UWM_SCRATCH/shared"
mkdir -p "$UWM_DATA" "$UWM_MODELS" "$UWM_RUNS" "$UWM_SHARED" "$HF_HOME" \
         "$UWM_REPO/.uwm/locks" "$UWM_REPO/.uwm/heartbeat" 2>/dev/null || true

# --- secrets: expected in the environment (you export them); code reads os.environ ---
[ -z "${OPENAI_API_KEY:-}" ] && echo "[env.sh] WARN: OPENAI_API_KEY unset (needed for gpt41 baselines)" >&2
[ -z "${HF_TOKEN:-}" ]       && echo "[env.sh] note: HF_TOKEN unset (only needed for gated HF models)" >&2

echo "[env.sh] UWM_SCRATCH=$UWM_SCRATCH | HF_HOME=$HF_HOME | conda=${CONDA_DEFAULT_ENV:-none}"
