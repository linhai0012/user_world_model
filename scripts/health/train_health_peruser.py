"""Train a per-user LoRA for the HEALTH domain (the `+per-user weights` arm) — the simplest
per-user world model: direct CE-SFT on ONE participant's own (activity -> next-day state) records.

  source scripts/env.sh   # or set UWM_DATA / UWM_MODELS / HF_HOME explicitly
  python scripts/health/train_health_peruser.py --pid p14 --epochs 3

Saves the LoRA **adapter only** (~tens of MB, CONVENTIONS §2 — never an 8GB merged model) to
$UWM_MODELS/peruser/health_<pid>/. Eval with scripts/health/eval_health_peruser.py (loads base+adapter).

Notes: the next-state target is the REAL PMData self-report (no teacher). Context is `base`
(activity only) so the user is encoded in weights, not the prompt — the clean per-user-weights
vs frozen-base comparison. HF training uses attn_implementation="sdpa" (the standalone flash-attn
wheel lacks a Blackwell sm_100 kernel for the HF path; sdpa works on every GPU; vLLM serving is
unaffected).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch
from domains.health.peruser_data import (build_health_user_samples, cond_tag,
                                        participant_record_counts, tokenize_sample)

BASE = "Qwen/Qwen3-4B-Instruct-2507"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", default=None, help="one participant -> per-user LoRA")
    ap.add_argument("--pids", default=None,
                    help="comma list of pids -> ONE pooled SHARED LoRA (the null control)")
    ap.add_argument("--cond", default="base")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--min-samples", type=int, default=30)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainingArguments)
    from peft import LoraConfig, get_peft_model

    if not (args.pid or args.pids):
        raise SystemExit("pass --pid <p> (per-user) or --pids p1,p2,... (shared null)")
    if args.pids:
        pid_list = [p.strip() for p in args.pids.split(",")]
        label = "shared"
    else:
        pid_list = [args.pid]
        label = args.pid

    tok = AutoTokenizer.from_pretrained(BASE)
    raw = []
    for p in pid_list:
        raw += build_health_user_samples(p, "train", args.cond)
    ds = [s for s in (tokenize_sample(tok, x, args.max_len) for x in raw) if s]
    counts = participant_record_counts("train")
    print(f"[train] label={label} pids={pid_list} cond={args.cond} samples={len(ds)} "
          f"(raw {len(raw)})", flush=True)
    if len(ds) < args.min_samples:
        raise SystemExit(f"too few samples ({len(ds)} < {args.min_samples}) — "
                         f"pick data-rich participant(s) (counts: {counts})")

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

    out = (Path(args.output_dir or (Path(os.environ["UWM_MODELS"]) / "peruser"))
           / f"health_{label}_{cond_tag(args.cond)}")
    targs = TrainingArguments(
        output_dir=str(out / "_ckpt"), per_device_train_batch_size=8,
        gradient_accumulation_steps=2, num_train_epochs=args.epochs,
        learning_rate=args.lr, bf16=True, logging_steps=5, save_strategy="no",
        report_to=[], warmup_ratio=0.03, lr_scheduler_type="cosine")
    Trainer(model=model, args=targs, train_dataset=ds, data_collator=collate).train()

    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))        # adapter only (~tens of MB)
    (out / "run_meta.json").write_text(json.dumps({
        "domain": "health", "label": label, "pids": pid_list, "cond": args.cond, "base": BASE,
        "n_samples": len(ds), "epochs": args.epochs, "lr": args.lr,
        "lora_rank": args.lora_rank, "max_len": args.max_len, "target": "next_state_json",
    }, indent=2))
    print(f"[done] LoRA adapter ({label}) -> {out}", flush=True)


if __name__ == "__main__":
    main()
