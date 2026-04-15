#!/bin/bash
#SBATCH --job-name=teacher_sft
#SBATCH --output=logs/teacher_sft_%j.out
#SBATCH --error=logs/teacher_sft_%j.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=12:00:00

# Teacher SFT for dynamic UserSim: Qwen3-4B full-parameter fine-tune on
# PersonaMem-v1 progressive-context samples. 4x H100 96GB, FSDP full-shard
# + gradient checkpointing + flash-attn 2. Plan: dynamic_usersim_complete_plan.md §3.

set -euxo pipefail

module load brics/nccl
eval "$(conda shell.bash hook)"
conda activate grpo  # reuse existing env; ensure transformers>=4.51 for Qwen3

REPO_DIR="/home/u6aa/linhai.u6aa/P-OPSD"
cd "$REPO_DIR"
mkdir -p logs

# Data path: built by dynamic_usersim/data_prep/build_teacher_sft_data.py
TRAIN_JSONL="$REPO_DIR/dynamic_usersim/outputs/teacher_sft_128k.jsonl"
if [ ! -f "$TRAIN_JSONL" ]; then
    echo "train jsonl missing at $TRAIN_JSONL — run build_teacher_sft_data.py first"
    exit 1
fi

# Checkpoint / output to SCRATCHDIR (HOME is only 100GB)
OUTPUT_DIR="$SCRATCHDIR/P-OPSD/teacher_sft_128k"
mkdir -p "$OUTPUT_DIR"

NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")

# Memory / NCCL env
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_NET=Socket            # single-node: NVLink via socket net
export TOKENIZERS_PARALLELISM=false
export HF_HOME="$SCRATCHDIR/hf_cache"
mkdir -p "$HF_HOME"

echo "=== Teacher SFT ==="
echo "GPUs: $NUM_GPUS"
echo "Train: $TRAIN_JSONL"
echo "Output: $OUTPUT_DIR"
echo "Job: $SLURM_JOB_ID | Node: $SLURM_NODELIST"

torchrun --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS" \
    dynamic_usersim/teacher_sft/train.py \
    --train-jsonl "$TRAIN_JSONL" \
    --output-dir "$OUTPUT_DIR" \
    --epochs 2 \
    --lr 1e-5 \
    --per-device-batch 1 \
    --grad-accum 4 \
    --resume

echo "=== done ==="
