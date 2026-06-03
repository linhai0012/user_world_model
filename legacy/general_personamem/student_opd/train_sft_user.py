"""Full-parameter user-memorization SFT (n=0 context).

Goal: measure the CEILING of 'how much user-specific knowledge can a 4B
Instruct-2507 model absorb into its parameters' — no LoRA, no OPD/OPSD
distillation, just direct SFT on (persona + chatbot_prev) -> user_response.

If this baseline's verbal judge / MCQ-PPL **beats** R1b dual-LoRA and
OPSD dual-LoRA by a clear margin, it tells us:
  - Memorization IS possible with enough capacity + right format
  - R1b/OPSD's LoRA + KL recipe leaves headroom on the table
If this matches R1b, memorization is capacity-bound and R1b already
captures the recoverable signal. If it's lower, SFT has catastrophic
forgetting and distillation is needed.

Input (n=0, matches Phase 2 canonical demo-only inference exactly):
    <|im_start|>system
    {persona_card}<|im_end|>
    <|im_start|>assistant
    {chatbot_prev}<|im_end|>
    <|im_start|>user
    {user_response}<|im_end|>
Loss mask: user_response content + its <|im_end|>\\n ONLY (R3-style).

Target env: B200 (sm_100) with flash-attn ≥ 2.7 for hardware kernel support.
If flash-attn is older, pass --attn-impl sdpa (~10-20% slower, same result).

Usage (single B200, ~900 samples × 2 epochs ≈ 15-30 min):
    python dynamic_usersim/student_opd/train_sft_user.py \\
        --persona-id 4 \\
        --data-path $HOME/P-OPSD/dynamic_usersim/outputs/opd_128k_pid4_k3.jsonl \\
        --output-dir $SCRATCHDIR/P-OPSD/sft_user/pid4 \\
        --attn-impl flash_attention_2

    # dry-run first to verify tokenization + loss-mask positions
    python dynamic_usersim/student_opd/train_sft_user.py ... --dry-run
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
class UserSFTDataset(Dataset):
    """n=0: (system=persona) + (assistant=chatbot_prev) + (user=user_response).

    Returns dict with input_ids and labels. labels == -100 everywhere
    except the user_response content + its trailing <|im_end|>\\n.
    """

    def __init__(self, samples: list[dict], tokenizer, max_len: int = 4096):
        self.samples = samples
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        tok = self.tok

        sys_ids = tok.encode(
            f"<|im_start|>system\n{s['demographics']}<|im_end|>\n",
            add_special_tokens=False,
        )
        asst_ids = tok.encode(
            f"<|im_start|>assistant\n{s['chatbot_prev']}<|im_end|>\n",
            add_special_tokens=False,
        )
        user_prefix = tok.encode(
            "<|im_start|>user\n", add_special_tokens=False
        )
        user_body = tok.encode(s["user_response"], add_special_tokens=False)
        user_end = tok.encode("<|im_end|>\n", add_special_tokens=False)

        input_ids = sys_ids + asst_ids + user_prefix + user_body + user_end
        labels = (
            [-100] * (len(sys_ids) + len(asst_ids) + len(user_prefix))
            + user_body + user_end
        )

        # If over max_len, drop OLDEST system/assistant tokens (preserve target)
        if len(input_ids) > self.max_len:
            target_len = len(user_prefix) + len(user_body) + len(user_end)
            budget = self.max_len - target_len
            if budget < 0:
                # target alone exceeds cap — truncate target right-side too
                input_ids = (user_prefix + user_body + user_end)[: self.max_len]
                labels = ([-100] * len(user_prefix) + user_body + user_end)[: self.max_len]
            else:
                overflow = (len(sys_ids) + len(asst_ids)) - budget
                # drop from start (oldest): keep tail of sys+asst
                sa_ids = (sys_ids + asst_ids)[overflow:]
                input_ids = sa_ids + user_prefix + user_body + user_end
                labels = [-100] * (len(sa_ids) + len(user_prefix)) + user_body + user_end

        return {"input_ids": input_ids, "labels": labels}


def collate(batch: list[dict], pad_token_id: int) -> dict:
    max_len = max(len(b["input_ids"]) for b in batch)
    bsz = len(batch)
    input_ids = torch.full((bsz, max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((bsz, max_len), -100, dtype=torch.long)
    attn = torch.zeros((bsz, max_len), dtype=torch.long)
    for i, b in enumerate(batch):
        L = len(b["input_ids"])
        input_ids[i, :L] = torch.tensor(b["input_ids"])
        labels[i, :L] = torch.tensor(b["labels"])
        attn[i, :L] = 1
    return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--persona-id", required=True)
    ap.add_argument("--data-path", type=Path, required=True,
                    help="opd_128k_pid{N}_k3.jsonl (pre-filtered per persona)")
    ap.add_argument("--base-model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--output-dir", type=Path, required=True)
    # ---- optim ----
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--per-device-batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--adam-beta2", type=float, default=0.95)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--max-len", type=int, default=4096)
    # ---- runtime ----
    ap.add_argument("--attn-impl",
                    choices=["flash_attention_2", "sdpa", "eager"],
                    default="flash_attention_2",
                    help="B200 needs flash-attn >= 2.7; else use sdpa")
    ap.add_argument("--grad-checkpointing", action="store_true",
                    help="enable activation checkpointing (saves GPU mem, "
                         "~15%% slower; probably unneeded on B200 192GB)")
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--save-total-limit", type=int, default=2)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true",
                    help="print first sample's token layout + loss mask, exit")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    random.seed(args.seed)

    # --- load samples ---
    with open(args.data_path, encoding="utf-8") as f:
        all_samples = [json.loads(line) for line in f]
    samples = [s for s in all_samples
               if str(s.get("persona_id", "")) == str(args.persona_id)]
    print(f"[sft] {len(samples)}/{len(all_samples)} samples for pid={args.persona_id}")
    if not samples:
        raise SystemExit("no samples match --persona-id")

    # --- tokenizer ---
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # --- dataset ---
    dataset = UserSFTDataset(samples, tok, max_len=args.max_len)

    if args.dry_run:
        ex = dataset[0]
        print("\n--- dry-run sample[0] ---")
        print(f"  total len: {len(ex['input_ids'])}")
        n_loss = sum(1 for x in ex["labels"] if x != -100)
        print(f"  loss tokens (user content + <|im_end|>\\n): {n_loss}")
        # Decode the loss-covered region to confirm
        covered_ids = [tid for tid, lbl in zip(ex["input_ids"], ex["labels"])
                        if lbl != -100]
        covered_text = tok.decode(covered_ids, skip_special_tokens=False)
        print(f"  loss-covered decoded: {covered_text[:300]!r}")
        # Persona snippet
        non_loss_ids = [tid for tid, lbl in zip(ex["input_ids"], ex["labels"])
                        if lbl == -100]
        print(f"  non-loss region (first 200 chars): "
              f"{tok.decode(non_loss_ids[:80], skip_special_tokens=False)[:200]!r}")
        return

    # --- model ---
    print(f"[sft] loading {args.base_model} (attn={args.attn_impl})")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_impl,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # --- training args ---
    targs = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        adam_beta2=args.adam_beta2,
        max_grad_norm=args.max_grad_norm,
        lr_scheduler_type="cosine",
        bf16=True,
        gradient_checkpointing=args.grad_checkpointing,
        logging_steps=args.log_every,
        save_steps=args.save_every,
        save_total_limit=args.save_total_limit,
        save_strategy="steps",
        seed=args.seed,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=args.num_workers,
        optim="adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=dataset,
        data_collator=lambda b: collate(b, tok.pad_token_id),
    )

    print(f"[sft] starting training — {len(dataset)} samples × {args.epochs} epochs "
          f"= {len(dataset)*args.epochs} total, "
          f"effective batch = {args.per_device_batch_size * args.grad_accum}, "
          f"expected steps = {(len(dataset) * args.epochs) // (args.per_device_batch_size * args.grad_accum)}")
    trainer.train()

    final_dir = args.output_dir / "final"
    trainer.save_model(str(final_dir))
    tok.save_pretrained(str(final_dir))
    print(f"[sft] saved final to {final_dir}")


if __name__ == "__main__":
    main()
