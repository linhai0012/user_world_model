"""Train a per-user LoRA (the `+per-user weights` arm) and merge it for vLLM serving.

  source scripts/env.sh
  python scripts/general/train_peruser.py --bench pm128k --persona-id 4 --epochs 3

Direct CE-on-ground-truth SFT (no teacher) over one persona's own user-turns; saves a merged
model to $UWM_MODELS/peruser/<bench>_pid<P>/ (merged so vLLM serves it normally — B200 punica
JIT is flaky with live LoRA adapters). Eval with scripts/general/eval_peruser.py.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch
from common.sft import tokenize_sample
from domains.general.peruser_data import build_user_samples

BASE = "Qwen/Qwen3-4B-Instruct-2507"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="pm128k")
    ap.add_argument("--persona-id", required=True)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--max-samples", type=int, default=None)
    args = ap.parse_args()

    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainingArguments)
    from peft import LoraConfig, get_peft_model

    tok = AutoTokenizer.from_pretrained(BASE)
    raw = build_user_samples(args.bench, args.persona_id, args.max_samples)
    ds = [s for s in (tokenize_sample(tok, x, args.max_len) for x in raw) if s]
    print(f"[train] persona={args.persona_id} bench={args.bench} samples={len(ds)} "
          f"(raw {len(raw)})", flush=True)
    if len(ds) < 20:
        raise SystemExit(f"too few samples ({len(ds)}) — pick a persona with more turns")

    def collate(batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        pad = tok.pad_token_id or tok.eos_token_id
        ids, lab, att = [], [], []
        for b in batch:
            n = maxlen - len(b["input_ids"])
            ids.append(b["input_ids"] + [pad] * n)
            lab.append(b["labels"] + [-100] * n)
            att.append([1] * len(b["input_ids"]) + [0] * n)
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(lab),
                "attention_mask": torch.tensor(att)}

    # NB: the standalone flash-attn 2.8.3 wheel has no B200 (sm_100) kernel for the HF training
    # path ("no kernel image"); SDPA works on Blackwell. (vLLM serving uses its own kernels.)
    model = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.use_cache = False
    lora = LoraConfig(
        r=args.lora_rank, lora_alpha=2 * args.lora_rank, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    out = Path(os.environ["UWM_MODELS"]) / "peruser" / f"{args.bench}_pid{args.persona_id}"
    targs = TrainingArguments(
        output_dir=str(out / "_ckpt"), per_device_train_batch_size=2,
        gradient_accumulation_steps=4, num_train_epochs=args.epochs,
        learning_rate=args.lr, bf16=True, logging_steps=10, save_strategy="no",
        report_to=[], warmup_ratio=0.03, lr_scheduler_type="cosine")
    Trainer(model=model, args=targs, train_dataset=ds, data_collator=collate).train()

    print("[train] merging LoRA into base for serving…", flush=True)
    merged = model.merge_and_unload()
    merged.save_pretrained(str(out))
    tok.save_pretrained(str(out))
    print(f"[done] merged per-user model → {out}", flush=True)


if __name__ == "__main__":
    main()
