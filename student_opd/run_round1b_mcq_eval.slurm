#!/bin/bash
#SBATCH --job-name=r1b_eval
#SBATCH --output=logs/r1b_eval_%j.out
#SBATCH --error=logs/r1b_eval_%j.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=16:00:00

# Phase 2b R1b — MCQ-PPL eval via sbatch.
#
# Wraps run_round1b_mcq_eval.sh with Slurm directives + miniforge conda
# init. Default evaluates all 20 personas, all available ckpts. Cached
# JSONs are skipped, so re-submitting after wall-time picks up where it
# stopped.
#
# Submit (full eval, ~35h, will need 2-3 sittings):
#   sbatch dynamic_usersim/student_opd/run_round1b_mcq_eval.slurm
#
# Submit (subset of personas via env):
#   sbatch --export=ALL,PERSONAS="0 1 2 3 4" \
#       dynamic_usersim/student_opd/run_round1b_mcq_eval.slurm
#
# Submit (different MCQ version):
#   sbatch --export=ALL,VERSION=1M \
#       dynamic_usersim/student_opd/run_round1b_mcq_eval.slurm

set -euxo pipefail

module load brics/nccl
source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate grpo

REPO_DIR="/home/u6aa/linhai.u6aa/P-OPSD"
cd "$REPO_DIR"
mkdir -p logs

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_HOME="$SCRATCHDIR/hf_cache"
mkdir -p "$HF_HOME"

# Default: all 20 personas; can be overridden via --export=ALL,PERSONAS=...
export PERSONAS="${PERSONAS:-0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19}"
export VERSION="${VERSION:-128k}"

echo "=== r1b_eval sbatch ==="
echo "Job:      $SLURM_JOB_ID"
echo "Node:     $SLURM_NODELIST"
echo "Time:     $(date)"
echo "Personas: $PERSONAS"
echo "Version:  $VERSION"

bash dynamic_usersim/student_opd/run_round1b_mcq_eval.sh

echo ""
echo "=== sbatch done at $(date) ==="
