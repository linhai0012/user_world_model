#!/usr/bin/env bash
#SBATCH --job-name=uwm-bl
#SBATCH --partition=tier1_charity_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#
# Generic single-GPU baseline launcher. See CONVENTIONS.md §3-§5.
# Submit:  sbatch scripts/general/run_baseline.slurm --run-id bl__oracle__qwen3-4b__pm128k
# (Adjust --partition / --time to your allocation.)
set -euo pipefail
cd "$(dirname "$0")/../.."
# shellcheck disable=SC1091
source scripts/env.sh

RUN_ID=""
EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    *)        EXTRA+=("$1"); shift ;;
  esac
done
[ -z "$RUN_ID" ] && { echo "ERROR: --run-id required" >&2; exit 2; }

# claim + register (CONVENTIONS.md §4)
scripts/claim_run.sh "$RUN_ID" || { echo "ERROR: run already claimed" >&2; exit 1; }
RUN_OUT="$UWM_RUNS/$RUN_ID"; mkdir -p "$RUN_OUT"
echo "[run] $RUN_ID  ->  $RUN_OUT  (GPU: ${SLURM_STEP_GPUS:-?})"

# TODO(next phase): dispatch to domains/general/baselines/<method>/run.py by RUN_ID, reading
#   experiments/configs/$RUN_ID.yaml and writing:
#     - large artifacts -> $RUN_OUT/
#     - headline JSON    -> experiments/results/general/$RUN_ID__acc.json
#     - status update    -> experiments/runs/$RUN_ID.yaml  (status: done, acc: ...)
# python -m domains.general.baselines.dispatch --run-id "$RUN_ID" "${EXTRA[@]:-}"

echo "[run] template only — baseline dispatch is implemented in the next phase."
