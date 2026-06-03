"""
Training configuration for the Digital Twin omnimodal LLM.
"""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainConfig:
    # ── Model ────────────────────────────────────────────────────────────
    model_name: str = "Qwen/Qwen3-4B"
    trust_remote_code: bool = True
    torch_dtype: str = "bfloat16"           # bf16 on H100
    attn_implementation: str = "sdpa"

    # ── Data ─────────────────────────────────────────────────────────────
    data_dir: str = "./output"
    train_file: str = "train.jsonl"
    val_file: str = "val.jsonl"
    test_file: str = "test.jsonl"
    special_tokens_file: str = "special_tokens.json"
    max_seq_len: int = 768                  # covers text + HR comfortably

    # ── Vocab extension ──────────────────────────────────────────────────
    pad_vocab_to_multiple_of: int = 128     # tensor-core friendly
    hr_embed_init: str = "linear"           # "linear" | "random" | "mean"

    # ── Loss ─────────────────────────────────────────────────────────────
    loss_alpha: float = 0.5                 # weight for text loss
    # L = alpha * L_text + (1 - alpha) * L_hr
    # set to -1.0 to disable weighting (uniform CE over all output tokens)

    # ── Training hyperparameters ─────────────────────────────────────────
    num_epochs: int = 8
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4    # effective batch = 2*4*4 = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0

    # ── Logging & saving ─────────────────────────────────────────────────
    output_dir: str = "./checkpoints"
    logging_steps: int = 10
    eval_steps: int = 50                    # evaluate every N steps
    save_steps: int = 100
    save_total_limit: int = 3
    report_to: str = "tensorboard"          # "wandb" | "tensorboard" | "none"
    run_name: str = "digital_twin_v1"

    # ── Hardware ─────────────────────────────────────────────────────────
    bf16: bool = True
    gradient_checkpointing: bool = True     # saves memory on 4B model
    dataloader_num_workers: int = 4
    seed: int = 42

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)

    @property
    def effective_batch_size(self) -> int:
        # Assuming 4 GPUs
        return self.per_device_train_batch_size * self.gradient_accumulation_steps * 4
