#!/bin/bash
#SBATCH --job-name=opd_dual_r1
#SBATCH --output=logs/opd_dual_r1_%j.out
#SBATCH --error=logs/opd_dual_r1_%j.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=08:00:00

# Phase 2b Round 1 — dual-LoRA OPD against R3 teacher.
#
# 4 personas (0, 4, 12, 14), one per GPU, run in parallel as background
# subprocesses. Each persona is a single-GPU job (no DDP); the trainer
# loads its own teacher + base.
#
# Layout assumes data files have already been re-built with
# --n-context-turns 2:
#   python dynamic_usersim/student_opd/build_opd_data.py \
#       --personas 0 4 12 14 --n-context-turns 2

set -euxo pipefail

module load brics/nccl
eval "$(conda shell.bash hook)"
conda activate grpo

REPO_DIR="/home/u6aa/linhai.u6aa/P-OPSD"
cd "$REPO_DIR"
mkdir -p logs

R3_TEACHER="${R3:-$SCRATCHDIR/P-OPSD/teacher_sft_128k_k3/final}"
BASE_MODEL="Qwen/Qwen3-4B-Instruct-2507"
PERSONAS=(0 4 12 14)

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_HOME="$SCRATCHDIR/hf_cache"
mkdir -p "$HF_HOME"

if [ ! -f "$R3_TEACHER/config.json" ]; then
    echo "ERROR: R3 teacher not found at $R3_TEACHER"
    exit 1
fi

echo "=== Phase 2b Round 1 dual-LoRA OPD ==="
echo "Job: $SLURM_JOB_ID | Node: $SLURM_NODELIST"
echo "R3 teacher: $R3_TEACHER"
echo "Personas:   ${PERSONAS[*]}"

PIDS=()
for i in "${!PERSONAS[@]}"; do
    pid="${PERSONAS[$i]}"
    gpu=$i
    data_path="$REPO_DIR/dynamic_usersim/outputs/opd_128k_pid${pid}_k3.jsonl"
    out_dir="$REPO_DIR/dynamic_usersim/outputs/lora_pid${pid}_dual_s32f16_ep1_r3teacher"
    log_file="logs/opd_dual_r1_pid${pid}_${SLURM_JOB_ID}.log"

    if [ ! -f "$data_path" ]; then
        echo "WARN pid=$pid: data missing at $data_path; skipping"
        continue
    fi

    echo "  launching pid=$pid on gpu=$gpu -> $log_file"
    python dynamic_usersim/student_opd/train_opd_dual.py \
        --persona-id "$pid" \
        --teacher-path "$R3_TEACHER" \
        --student-base "$BASE_MODEL" \
        --data-path "$data_path" \
        --output-dir "$out_dir" \
        --gpu "$gpu" \
        --epochs 1 \
        --save-every 200 \
        --save-total-limit 2 \
        --resume \
        > "$log_file" 2>&1 &
    PIDS+=($!)
done

# Wait for all background jobs
fail=0
for p in "${PIDS[@]}"; do
    if ! wait "$p"; then
        echo "subprocess $p failed"
        fail=1
    fi
done

if [ $fail -ne 0 ]; then
    echo "=== one or more personas failed ==="
    exit 1
fi

echo "=== all personas done ==="
