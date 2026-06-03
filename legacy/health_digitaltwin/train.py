#!/usr/bin/env python3
"""
train.py — Train the Digital Twin omnimodal LLM.

Usage:
    # Single-node multi-GPU with Accelerate (recommended for 4×H100)
    accelerate launch --num_processes 4 train.py

    # With custom config overrides
    accelerate launch --num_processes 4 train.py \
        --loss_alpha 0.5 \
        --num_epochs 10 \
        --learning_rate 2e-5

    # Quick debug on 1 GPU
    python train.py --debug

Prerequisites:
    pip install torch transformers accelerate datasets tensorboard
    pip install flash-attn --no-build-isolation
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
import numpy as np
from transformers import TrainingArguments, EvalPrediction

from train_config import TrainConfig
from model_setup import setup_tokenizer, setup_model
from dataset import DigitalTwinDataset, DigitalTwinDataCollator
from trainer import WeightedLossTrainer
from metrics import parse_generated_output

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  Eval metric callback (computed during training eval steps)
# ═══════════════════════════════════════════════════════════════════════════════

def build_compute_metrics(tokenizer):
    """Build a compute_metrics function compatible with HF Trainer."""

    def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
        """
        Compute eval metrics from model predictions.
        Predictions are already argmax'd by preprocess_logits_for_eval.
        """
        preds = eval_pred.predictions     # (N, seq_len) — already argmax
        labels = eval_pred.label_ids      # (N, seq_len)

        if isinstance(preds, tuple):
            preds = preds[0]

        # Token-level accuracy on non-masked positions
        valid = labels != -100
        correct = (preds == labels) & valid
        accuracy = correct.sum() / valid.sum() if valid.sum() > 0 else 0

        # Separate accuracy for HR tokens vs text tokens
        hr_token_ids = set()
        for tid in range(len(tokenizer)):
            tok = tokenizer.convert_ids_to_tokens(tid)
            if tok and tok.startswith("<hr_"):
                hr_token_ids.add(tid)

        is_hr = np.zeros_like(labels, dtype=bool)
        for tid in hr_token_ids:
            is_hr |= (labels == tid)

        hr_valid = valid & is_hr
        text_valid = valid & ~is_hr

        hr_acc = (correct & is_hr).sum() / hr_valid.sum() if hr_valid.sum() > 0 else 0
        text_acc = (correct & ~is_hr & valid).sum() / text_valid.sum() if text_valid.sum() > 0 else 0

        return {
            "eval_accuracy": float(accuracy),
            "eval_text_accuracy": float(text_acc),
            "eval_hr_accuracy": float(hr_acc),
        }

    return compute_metrics


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train Digital Twin LLM")
    parser.add_argument("--loss_alpha", type=float, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--hr_embed_init", type=str, default=None,
                        choices=["linear", "mean", "random"])
    parser.add_argument("--debug", action="store_true",
                        help="Quick debug run on 1 GPU with small data")
    args = parser.parse_args()

    # ── Build config ─────────────────────────────────────────────────────
    cfg = TrainConfig()
    if args.loss_alpha is not None:
        cfg.loss_alpha = args.loss_alpha
    if args.num_epochs is not None:
        cfg.num_epochs = args.num_epochs
    if args.learning_rate is not None:
        cfg.learning_rate = args.learning_rate
    if args.hr_embed_init is not None:
        cfg.hr_embed_init = args.hr_embed_init

    if args.debug:
        cfg.num_epochs = 2
        cfg.per_device_train_batch_size = 1
        cfg.gradient_accumulation_steps = 1
        cfg.logging_steps = 2
        cfg.eval_steps = 10
        cfg.save_steps = 999999
        cfg.gradient_checkpointing = False
        cfg.attn_implementation = "eager"
        log.info("DEBUG MODE: reduced config")

    log.info(f"Config: alpha={cfg.loss_alpha}, epochs={cfg.num_epochs}, "
             f"lr={cfg.learning_rate}, batch={cfg.effective_batch_size}, "
             f"hr_init={cfg.hr_embed_init}")

    # ── Setup tokenizer & model ──────────────────────────────────────────
    log.info("Setting up tokenizer ...")
    tokenizer = setup_tokenizer(cfg)

    log.info("Setting up model ...")
    model = setup_model(cfg, tokenizer)

    # ── Load datasets ────────────────────────────────────────────────────
    log.info("Loading datasets ...")
    train_ds = DigitalTwinDataset(
        cfg.data_path / cfg.train_file, tokenizer, cfg.max_seq_len
    )
    val_ds = DigitalTwinDataset(
        cfg.data_path / cfg.val_file, tokenizer, cfg.max_seq_len
    )

    if args.debug:
        # Subsample for debugging
        train_ds.samples = train_ds.samples[:32]
        val_ds.samples = val_ds.samples[:16]
        log.info(f"DEBUG: using {len(train_ds)} train, {len(val_ds)} val samples")

    log.info(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

    # ── Training arguments ───────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        run_name=cfg.run_name,

        # Epochs & batching
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,

        # Optimizer
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        max_grad_norm=cfg.max_grad_norm,
        optim="adamw_torch_fused",

        # Precision
        bf16=cfg.bf16,

        # Evaluation
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        prediction_loss_only=True,          # only compute eval_loss, skip logits accumulation (OOM fix)

        # Logging
        logging_steps=cfg.logging_steps,
        logging_first_step=True,
        report_to=cfg.report_to,

        # Saving
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # Data
        dataloader_num_workers=cfg.dataloader_num_workers,
        remove_unused_columns=False,    # keep is_hr_token column

        # Misc
        seed=cfg.seed,
        label_names=["labels"],
    )

    # ── Initialize Trainer ───────────────────────────────────────────────
    trainer = WeightedLossTrainer(
        loss_alpha=cfg.loss_alpha,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DigitalTwinDataCollator(),
        compute_metrics=build_compute_metrics(tokenizer),
        tokenizer=tokenizer,
    )

    # ── Train ────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("  STARTING TRAINING")
    log.info(f"  Model:            {cfg.model_name}")
    log.info(f"  Loss alpha:       {cfg.loss_alpha}")
    log.info(f"  Effective batch:  {cfg.effective_batch_size}")
    log.info(f"  Epochs:           {cfg.num_epochs}")
    log.info(f"  LR:               {cfg.learning_rate}")
    log.info(f"  HR embed init:    {cfg.hr_embed_init}")
    log.info("=" * 60)

    train_result = trainer.train()

    # ── Save final model ─────────────────────────────────────────────────
    final_dir = Path(cfg.output_dir) / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    log.info(f"Final model saved → {final_dir}")

    # Save training metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_ds)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # ── Final evaluation ─────────────────────────────────────────────────
    log.info("Running final evaluation ...")
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    log.info("=" * 60)
    log.info("  TRAINING COMPLETE")
    for key in ["eval_loss", "eval_text_accuracy", "eval_hr_accuracy"]:
        val = eval_metrics.get(key)
        log.info(f"  {key:30s} {val:.4f}" if isinstance(val, (int, float)) else f"  {key:30s} {val}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
