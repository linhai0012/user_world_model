"""Teacher SFT hyperparameters for Phase 1.

Plan reference: dynamic_usersim_complete_plan.md §3.4
Start with 128k version; switch to 1M once FSDP+grad-checkpoint stable.
"""

from __future__ import annotations

from pathlib import Path

# ---- Model ----
MODEL_NAME = "Qwen/Qwen3-4B"
QWEN_LAYER_CLS = "Qwen3DecoderLayer"  # for FSDP transformer auto-wrap policy

# ---- Data ----
DATA_VERSION = "128k"  # "32k" / "128k" / "1M"
# Max tokens per sample. 128k data hits ~131k worst-case; even with FSDP
# + flash-attn + liger fused CE, full 131k OOMs the logits-grad path on
# H100 96GB. 65536 fits comfortably; target is always preserved, prefix
# sessions dropped from oldest end. Expect ~10% samples to lose some prefix.
MAX_SEQ_LEN = 65536

# ---- Optim (plan §3.4) ----
LEARNING_RATE = 1e-5
WARMUP_RATIO = 0.10
NUM_EPOCHS = 1  # plan says 1-2; start with 1
WEIGHT_DECAY = 0.0
MAX_GRAD_NORM = 1.0
OPTIMIZER = "adamw_torch"  # FSDP-friendly
LR_SCHEDULER = "cosine"

# ---- Batch sizing (per plan §3.4 compute notes) ----
# At 131k seq len on 96GB H100 with FSDP + activation checkpointing, per-device
# batch must be 1. Use grad accumulation to hit a meaningful effective batch.
PER_DEVICE_BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 4  # effective batch = 1 * 4gpu * 4accum = 16

# ---- Precision / attn ----
BF16 = True
ATTN_IMPL = "flash_attention_2"  # required to fit 131k context
GRADIENT_CHECKPOINTING = True

# ---- Paths ----
# Repo-relative paths; runtime will resolve from env / CLI
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_JSONL = ROOT / "dynamic_usersim" / "outputs" / f"teacher_sft_{DATA_VERSION}.jsonl"

# ---- Logging / checkpoint ----
LOGGING_STEPS = 5
SAVE_STRATEGY = "epoch"   # save at end of each epoch
SAVE_TOTAL_LIMIT = 2
DATALOADER_NUM_WORKERS = 2
SEED = 42
