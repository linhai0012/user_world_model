"""Teacher SFT training entrypoint (Qwen3-4B, full-parameter, FSDP).

Launch via torchrun --nproc-per-node=4 (see run_isambard.slurm).

Single-process / CPU smoke test:
    python dynamic_usersim/teacher_sft/train.py --smoke-test
  -> builds dataset, checks one batch, no forward pass on model.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Local imports
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from config import (
    ATTN_IMPL, BF16, CONTEXT_WINDOW_K, DATA_VERSION, DEFAULT_TRAIN_JSONL,
    GRAD_ACCUM_STEPS, GRADIENT_CHECKPOINTING, LEARNING_RATE, LOGGING_STEPS,
    LR_SCHEDULER, MAX_GRAD_NORM, MAX_SEQ_LEN, MODEL_NAME, NUM_EPOCHS,
    OPTIMIZER, PER_DEVICE_BATCH_SIZE, QWEN_LAYER_CLS, SAVE_STEPS,
    SAVE_STRATEGY, SAVE_TOTAL_LIMIT, SEED, WARMUP_RATIO, WEIGHT_DECAY,
    DATALOADER_NUM_WORKERS,
)
from dataset import PadToBatchMaxCollator, TeacherSFTDataset

_DP = _HERE.parent / "data_prep"
if str(_DP) not in sys.path:
    sys.path.insert(0, str(_DP))
from tokenize_teacher_sft import SFTTokenizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-jsonl", type=Path, default=DEFAULT_TRAIN_JSONL)
    ap.add_argument("--output-dir", type=Path, required=False,
                    help="checkpoint dir; defaults to "
                         "$SCRATCHDIR/P-OPSD/teacher_sft_<ver> or ./outputs/...")
    ap.add_argument("--model-name", default=MODEL_NAME)
    ap.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    ap.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    ap.add_argument("--lr", type=float, default=LEARNING_RATE)
    ap.add_argument("--per-device-batch", type=int, default=PER_DEVICE_BATCH_SIZE)
    ap.add_argument("--grad-accum", type=int, default=GRAD_ACCUM_STEPS)
    ap.add_argument("--smoke-test", action="store_true",
                    help="Build dataset + one batch, skip model load.")
    ap.add_argument("--no-fsdp", action="store_true",
                    help="Skip FSDP (for single-GPU debugging).")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from latest checkpoint in output_dir.")
    return ap.parse_args()


def resolve_output_dir(cli_dir: Path | None) -> Path:
    if cli_dir is not None:
        return cli_dir
    name = f"teacher_sft_{DATA_VERSION}_k{CONTEXT_WINDOW_K}"
    scratch = os.environ.get("SCRATCHDIR")
    if scratch:
        return Path(scratch) / "P-OPSD" / name
    return _HERE.parent / "outputs" / name


def build_training_args(args: argparse.Namespace, output_dir: Path) -> TrainingArguments:
    fsdp_str = "" if args.no_fsdp else "full_shard auto_wrap"
    fsdp_cfg: dict = {} if args.no_fsdp else {
        "transformer_layer_cls_to_wrap": [QWEN_LAYER_CLS],
        # When FSDP is on, activation checkpointing must live in fsdp_config
        # (not TrainingArguments.gradient_checkpointing) — the two are mutually
        # exclusive per transformers>=4.40. See HF issue #30404.
        "activation_checkpointing": GRADIENT_CHECKPOINTING,
        "use_orig_params": True,
        "backward_prefetch": "backward_pre",
        "forward_prefetch": False,
        "sync_module_states": True,
    }

    # Use TrainingArguments-level checkpointing only in the single-GPU
    # fallback path (--no-fsdp).
    ta_grad_ckpt = GRADIENT_CHECKPOINTING and args.no_fsdp
    ta_grad_ckpt_kwargs = {"use_reentrant": False} if ta_grad_ckpt else None

    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type=LR_SCHEDULER,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=MAX_GRAD_NORM,
        optim=OPTIMIZER,
        bf16=BF16,
        gradient_checkpointing=ta_grad_ckpt,
        gradient_checkpointing_kwargs=ta_grad_ckpt_kwargs,
        logging_steps=LOGGING_STEPS,
        save_strategy=SAVE_STRATEGY,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        dataloader_num_workers=DATALOADER_NUM_WORKERS,
        remove_unused_columns=False,
        report_to=["none"],  # plain console logs; wandb opt-in later
        seed=SEED,
        fsdp=fsdp_str,
        fsdp_config=fsdp_cfg,
        # Let model's own eos token id be used for pad; we set it below too.
    )


def smoke_test(args: argparse.Namespace) -> None:
    print(f"[smoke] loading tokenizer (tokenizer-only, no model weights)...")
    tok = SFTTokenizer(model_name=args.model_name, max_len=args.max_seq_len)
    print(f"[smoke] building dataset from {args.train_jsonl}")
    ds = TeacherSFTDataset(args.train_jsonl, tok)
    print(f"[smoke] dataset size: {len(ds)}")

    sample = ds[0]
    print(f"[smoke] sample 0 keys: {list(sample.keys())}")
    print(f"[smoke] input_ids shape: {sample['input_ids'].shape}  "
          f"labels shape: {sample['labels'].shape}")
    n_loss = (sample["labels"] != -100).sum().item()
    print(f"[smoke] loss tokens in sample 0: {n_loss}")

    pad_id = tok.tok.pad_token_id or tok.tok.eos_token_id
    collator = PadToBatchMaxCollator(pad_token_id=pad_id)
    batch = collator([ds[0], ds[1]])
    print(f"[smoke] batched input_ids shape: {batch['input_ids'].shape}")
    print(f"[smoke] batched labels dtype: {batch['labels'].dtype}")
    print(f"[smoke] batched attention_mask sum per row: "
          f"{batch['attention_mask'].sum(dim=1).tolist()}")
    print("[smoke] OK")


def main() -> None:
    args = parse_args()
    set_seed(SEED)

    if args.smoke_test:
        smoke_test(args)
        return

    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train] output_dir = {output_dir}")
    print(f"[train] train_jsonl = {args.train_jsonl}")
    print(f"[train] model = {args.model_name}")

    # ---- Tokenizer / data ----
    tok = SFTTokenizer(model_name=args.model_name, max_len=args.max_seq_len)
    if tok.tok.pad_token_id is None:
        tok.tok.pad_token = tok.tok.eos_token
    pad_id = tok.tok.pad_token_id

    train_ds = TeacherSFTDataset(args.train_jsonl, tok)
    print(f"[train] train samples: {len(train_ds)}")

    # ---- Liger Kernel patch ----
    # Fused RMSNorm + RoPE + SwiGLU + (crucially) FusedLinearCrossEntropy
    # that never materializes the full [batch, seq, vocab] logits tensor.
    # Must be applied BEFORE loading the model. Mathematically equivalent
    # to standard ops (up to fp tolerance); typically 20-30% faster and
    # saves 30-50 GB at long context.
    try:
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3
        apply_liger_kernel_to_qwen3(
            rope=True, swiglu=True, cross_entropy=False,
            fused_linear_cross_entropy=True, rms_norm=True,
        )
        print("[train] liger kernel applied to Qwen3")
    except ImportError:
        print("[train] liger_kernel not installed — falling back to stock kernels "
              "(may OOM at long context)")

    # ---- Model ----
    # Flash-attn 2 for efficient attention; bf16 master compute.
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation=ATTN_IMPL,
        trust_remote_code=True,
    )
    # Required when using grad checkpointing with Trainer
    model.config.use_cache = False

    # ---- Trainer ----
    training_args = build_training_args(args, output_dir)
    collator = PadToBatchMaxCollator(pad_token_id=pad_id)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collator,
        tokenizer=tok.tok,
    )

    # resume_from_checkpoint=True auto-finds the latest checkpoint-N dir in
    # output_dir. Scheduler, optimizer state, RNG, and trainer_state resume;
    # FSDP full_state_dict checkpoint is sharded back across ranks.
    trainer.train(resume_from_checkpoint=True if args.resume else None)
    trainer.save_model(str(output_dir / "final"))
    tok.tok.save_pretrained(str(output_dir / "final"))


if __name__ == "__main__":
    main()
