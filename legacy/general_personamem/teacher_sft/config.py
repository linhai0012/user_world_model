"""Teacher SFT hyperparameters for Phase 1.

Plan reference: dynamic_usersim_complete_plan.md §3.4
Start with 128k version; switch to 1M once FSDP+grad-checkpoint stable.
"""

from __future__ import annotations

from pathlib import Path

# ---- Model ----
# Qwen3-4B-Instruct-2507 vs "Qwen/Qwen3-4B":
#   - No thinking mode (hybrid model's thinking tendency not suited for
#     spontaneous user simulation — plan §1 wants natural user reactions,
#     not analytical ones).
#   - Native 262k max_position_embeddings (vs 40960) -> no RoPE extrapolation
#     for any K we might pick. Eval 2's tiny context_benefit may have been
#     confounded by extrapolation on Qwen3-4B; this fixes that variable.
#   - Chat template has no <think></think> wrapper around assistant history,
#     matching our _encode_message format exactly (no workarounds needed).
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
QWEN_LAYER_CLS = "Qwen3DecoderLayer"  # same architecture as Qwen3-4B

# ---- Data ----
DATA_VERSION = "128k"  # "32k" / "128k" / "1M"
# Progressive-context window cap (session level). Plan §3.3 originally had
# K=20 (~full 128k). Eval 2 on the K=20 run showed prior sessions only give
# +0.02 nats benefit on top of current session — most of SFT's gain comes
# from user-style learning and long-ctx robustness, not history-mining.
# K=3 keeps 99% of samples within Qwen3-4B's native max_position_embeddings
# (40960), eliminating RoPE extrapolation entirely and matching the
# model's trained distribution cleanly.
CONTEXT_WINDOW_K = 3
_K_TAG = f"k{CONTEXT_WINDOW_K}"

# Max tokens per sample. Qwen3-4B has max_position_embeddings=40960 and
# no default rope_scaling (confirmed from HF config). Beyond this, RoPE
# extrapolates onto untrained frequencies. Cap at exactly 40960 so every
# sample runs on in-distribution positions. K=3 data: 99% of samples fit;
# the remaining ~1% get prefix-truncated (oldest prior sessions dropped).
MAX_SEQ_LEN = 40960

# ---- Optim (plan §3.4) ----
LEARNING_RATE = 1e-5
WARMUP_RATIO = 0.10
NUM_EPOCHS = 2  # plan §3.4 says 1-2; interactive run chose 2
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
DEFAULT_TRAIN_JSONL = (ROOT / "dynamic_usersim" / "outputs"
                       / f"teacher_sft_{DATA_VERSION}_{_K_TAG}.jsonl")

# ---- Logging / checkpoint ----
LOGGING_STEPS = 5
# Save every N optimizer steps. 74 steps/epoch at 1172 samples / 16 effective
# batch; save_steps=25 gives ~3 checkpoints per epoch for crash recovery.
SAVE_STRATEGY = "steps"
SAVE_STEPS = 25
SAVE_TOTAL_LIMIT = 2
DATALOADER_NUM_WORKERS = 2
SEED = 42
