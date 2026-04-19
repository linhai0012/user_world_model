#!/bin/bash
#SBATCH --job-name=r1b_extend
#SBATCH --output=logs/r1b_extend_%j.out
#SBATCH --error=logs/r1b_extend_%j.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=16:00:00

# Phase 2b R1b — extend to all 20 personas via sbatch.
#
# Wraps run_round1b_extend.sh with Slurm directives + 16h wall time
# (enough for ~3-4 remaining batches at ~4h each). Idempotent:
# re-submitting after wall-time / failure resumes from where it stopped.
#
# Submit:  sbatch dynamic_usersim/student_opd/run_round1b_extend.slurm
# Monitor: squeue -u $USER; tail -f logs/r1b_extend_*.out

set -euxo pipefail

module load brics/nccl

# sbatch doesn't source ~/.bashrc → conda not on PATH. Explicitly source
# miniforge init script before activate.
source $HOME/miniforge3/etc/profile.d/conda.sh
conda activate grpo

REPO_DIR="/home/u6aa/linhai.u6aa/P-OPSD"
cd "$REPO_DIR"
mkdir -p logs

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_HOME="$SCRATCHDIR/hf_cache"
mkdir -p "$HF_HOME"

echo "=== r1b_extend sbatch ==="
echo "Job:  $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Time: $(date)"

bash dynamic_usersim/student_opd/run_round1b_extend.sh

echo ""
echo "=== sbatch done at $(date) ==="
