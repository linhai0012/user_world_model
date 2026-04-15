"""Phase-2 OPD training: single-LoRA per persona.

Standard on-policy distillation (matches opd/s01_opd_train.py):
  for each sample:
    1. Student rollout (base + LoRA, no_grad, sampling)
    2. Teacher logprobs on rollout (frozen teacher sees full K=3 history)
    3. Student logprobs on rollout (grad, student sees demographics + chatbot_prev)
    4. KL(student || teacher) = sum_v P_s(v) * (log P_s(v) - log P_t(v))
       per-token, mean over rollout tokens
    5. Backward, clip, step (LoRA params only)

One invocation = one persona. Slurm orchestrates across {0, 12, 14}.

Student input (no history):
    <|im_start|>system\n{demographics}<|im_end|>\n
    <|im_start|>assistant\n{chatbot_prev}<|im_end|>\n
    <|im_start|>user\n                                ← rollout starts here

Teacher input (K=3 prior sessions + current session up to chatbot_prev):
    concat of history_messages via ChatML
    <|im_start|>user\n                                ← same generation prefix

Usage (Isambard):
    python train_opd.py --persona-id 14 \
        --teacher-path /path/to/teacher_sft_ckpt-50 \
        --data-path dynamic_usersim/outputs/opd_128k_pid14_k3.jsonl \
        --output-dir dynamic_usersim/outputs/lora_pid14_r32_ep1
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------- ChatML prefix construction ----------

def build_chatml_prefix(
    messages: list[dict],
    tokenizer,
    trailing_role: str = "user",
) -> list[int]:
    """Encode messages as ChatML, then append '<|im_start|>{trailing_role}\\n'.

    The returned ids are the prefix for rollout generation — they end at the
    role header of the turn to be generated (no content, no end marker).
    """
    ids: list[int] = []
    for m in messages:
        s = f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        ids.extend(tokenizer.encode(s, add_special_tokens=False))
    ids.extend(
        tokenizer.encode(
            f"<|im_start|>{trailing_role}\n", add_special_tokens=False
        )
    )
    return ids


def build_student_prefix(sample: dict, tokenizer) -> list[int]:
    """Student view: demographics (system) + chatbot_prev (assistant)."""
    msgs = [
        {"role": "system", "content": sample["demographics"]},
        {"role": "assistant", "content": sample["chatbot_prev"]},
    ]
    return build_chatml_prefix(msgs, tokenizer, trailing_role="user")


def build_teacher_prefix(sample: dict, tokenizer) -> list[int]:
    """Teacher view: full K=3 history (ends at chatbot_prev)."""
    return build_chatml_prefix(
        sample["history_messages"], tokenizer, trailing_role="user"
    )


def truncate_teacher_prefix(
    prefix_ids: list[int], max_tokens: int
) -> tuple[list[int], bool]:
    """Right-truncation keeping the tail intact (preserves chatbot_prev)."""
    if len(prefix_ids) <= max_tokens:
        return prefix_ids, False
    return prefix_ids[-max_tokens:], True


# ---------- Core OPD ops ----------

@torch.no_grad()
def student_rollout(
    model,
    tokenizer,
    prefix_ids: list[int],
    device,
    max_new_tokens: int,
    temperature: float,
    eos_token_id: int,
    pad_token_id: int,
) -> list[int]:
    """Sample continuation from the student. Returns response token ids
    (stop at <|im_end|>, not including padding)."""
    input_ids = torch.tensor([prefix_ids], device=device)
    out = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0),
        temperature=max(temperature, 1e-5),
        top_p=1.0,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )
    return out[0, input_ids.shape[1]:].tolist()


def compute_response_logprobs(
    model,
    prefix_ids: list[int],
    response_ids: list[int],
    device,
) -> torch.Tensor:
    """Per-token log-softmax over response positions. Shape [R, V]."""
    full = torch.tensor([prefix_ids + response_ids], device=device)
    logits = model(input_ids=full).logits[0]
    # Next-token prediction: position (len(prefix)-1) predicts response[0].
    start = len(prefix_ids) - 1
    end = start + len(response_ids)
    return F.log_softmax(logits[start:end].float(), dim=-1)


# ---------- Main ----------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--persona-id", required=True, type=str)
    ap.add_argument("--data-path", type=Path, default=None,
                    help="JSONL from build_opd_data.py; auto-resolved from persona-id")
    ap.add_argument("--teacher-path", type=Path, required=True,
                    help="R1 ckpt-50 (full-parameter SFT)")
    ap.add_argument("--student-base", type=str, default="Qwen/Qwen3-4B")
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    ap.add_argument("--lora-targets", nargs="+",
                    default=["q_proj", "k_proj", "v_proj", "o_proj",
                             "gate_proj", "up_proj"])
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--rollout-max-tokens", type=int, default=160)
    ap.add_argument("--rollout-temperature", type=float, default=1.0)
    ap.add_argument("--max-teacher-tokens", type=int, default=32768,
                    help="truncate teacher prefix from the left if longer")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--max-samples", type=int, default=-1,
                    help="cap for quick sanity runs; -1 = all")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true",
                    help="CPU forward-only shape check on 1 sample; no training")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # --- Paths ---
    repo_root = Path(__file__).resolve().parents[2]
    if args.data_path is None:
        args.data_path = (
            repo_root / "dynamic_usersim" / "outputs"
            / f"opd_128k_pid{args.persona_id}_k3.jsonl"
        )
    if args.output_dir is None:
        args.output_dir = (
            repo_root / "dynamic_usersim" / "outputs"
            / f"lora_pid{args.persona_id}_r{args.lora_rank}_ep{args.epochs}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- Device ---
    if args.dry_run:
        device = torch.device("cpu")
        dtype = torch.float32
    else:
        device = torch.device(f"cuda:{args.gpu}")
        dtype = torch.bfloat16

    # --- Tokenizer (shared) ---
    tokenizer = AutoTokenizer.from_pretrained(
        args.student_base, trust_remote_code=True
    )
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    pad_id = tokenizer.pad_token_id or im_end_id
    assert isinstance(im_end_id, int) and im_end_id > 0, (
        f"could not resolve <|im_end|>, got {im_end_id}"
    )

    # --- Data ---
    with args.data_path.open("r", encoding="utf-8") as fh:
        samples = [json.loads(line) for line in fh]
    if args.max_samples > 0:
        samples = samples[: args.max_samples]
    print(f"[pid={args.persona_id}] loaded {len(samples)} samples from {args.data_path}")

    if args.dry_run:
        samples = samples[:1]
        print("DRY RUN: 1 sample, CPU, no training")

    # --- Teacher (frozen) ---
    print(f"loading teacher: {args.teacher_path}")
    t0 = time.time()
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_path, torch_dtype=dtype, trust_remote_code=True
    ).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"  teacher loaded in {time.time()-t0:.1f}s")

    # --- Student base + fresh LoRA ---
    print(f"loading student base: {args.student_base}")
    t0 = time.time()
    student_base = AutoModelForCausalLM.from_pretrained(
        args.student_base, torch_dtype=dtype, trust_remote_code=True
    ).to(device)
    print(f"  student base loaded in {time.time()-t0:.1f}s")

    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_targets,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    student = get_peft_model(student_base, lora_cfg)
    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    total = sum(p.numel() for p in student.parameters())
    print(f"  LoRA trainable: {trainable:,} / {total:,} "
          f"({100*trainable/total:.3f}%)")

    if not args.dry_run:
        alloc = torch.cuda.memory_allocated(device) / 1e9
        print(f"  GPU memory after both models: {alloc:.1f} GB")

    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=args.lr,
    )

    # --- Training loop ---
    training_log = []
    step_losses = []
    truncation_count = 0
    skipped_empty = 0

    for epoch in range(args.epochs):
        idx = list(range(len(samples)))
        random.shuffle(idx)

        epoch_kl = 0.0
        epoch_tokens = 0
        t_epoch = time.time()

        for step, i in enumerate(idx):
            sample = samples[i]

            # Build prefixes
            s_prefix = build_student_prefix(sample, tokenizer)
            t_prefix = build_teacher_prefix(sample, tokenizer)
            t_prefix, truncated = truncate_teacher_prefix(
                t_prefix, args.max_teacher_tokens
            )
            truncation_count += int(truncated)

            # 1. Student rollout
            student.eval()
            response_ids = student_rollout(
                student, tokenizer, s_prefix, device,
                max_new_tokens=args.rollout_max_tokens,
                temperature=args.rollout_temperature,
                eos_token_id=im_end_id,
                pad_token_id=pad_id,
            )
            # Strip trailing pad/eos duplicates; keep the first <|im_end|> if present.
            if im_end_id in response_ids:
                cut = response_ids.index(im_end_id) + 1
                response_ids = response_ids[:cut]
            if len(response_ids) == 0:
                skipped_empty += 1
                continue

            if args.dry_run:
                print(f"DRY: s_prefix={len(s_prefix)} t_prefix={len(t_prefix)} "
                      f"response={len(response_ids)}")
                print("  rollout text:",
                      tokenizer.decode(response_ids, skip_special_tokens=False)[:200])

            # 2. Teacher logprobs (no grad)
            with torch.no_grad():
                t_lp = compute_response_logprobs(
                    teacher, t_prefix, response_ids, device
                )

            # 3. Student logprobs (with grad)
            student.train()
            s_lp = compute_response_logprobs(
                student, s_prefix, response_ids, device
            )

            # 4. KL(student || teacher) per-token, mean
            min_len = min(s_lp.shape[0], t_lp.shape[0])
            s_lp = s_lp[:min_len]
            t_lp = t_lp[:min_len]
            kl_per_token = (s_lp.exp() * (s_lp - t_lp)).sum(dim=-1)
            loss = kl_per_token.mean()

            if args.dry_run:
                print(f"DRY: loss={loss.item():.4f}, min_len={min_len}")
                break

            # 5. Step
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

            epoch_kl += loss.item() * min_len
            epoch_tokens += min_len
            step_losses.append({
                "epoch": epoch, "step": step, "loss": loss.item(),
                "min_len": min_len, "sample_idx": i,
            })

            if (step + 1) % args.log_every == 0:
                recent = [x["loss"] for x in step_losses[-args.log_every:]]
                print(f"  e{epoch} step {step+1}/{len(idx)}: "
                      f"loss(recent)={sum(recent)/len(recent):.4f} "
                      f"min_len={min_len}")

        if args.dry_run:
            break

        avg_kl = epoch_kl / max(epoch_tokens, 1)
        training_log.append({
            "epoch": epoch,
            "avg_kl": avg_kl,
            "n_samples": len(idx) - skipped_empty,
            "n_tokens": epoch_tokens,
            "elapsed_s": time.time() - t_epoch,
        })
        print(f"[epoch {epoch}] avg_kl={avg_kl:.4f} "
              f"samples={len(idx)-skipped_empty} "
              f"tokens={epoch_tokens} "
              f"time={training_log[-1]['elapsed_s']:.0f}s")

    if args.dry_run:
        print("DRY RUN complete (no LoRA saved).")
        return

    # --- Save LoRA + log ---
    student.save_pretrained(args.output_dir)
    log_path = args.output_dir / "training_log.json"
    with log_path.open("w", encoding="utf-8") as fh:
        json.dump({
            "persona_id": args.persona_id,
            "n_samples": len(samples),
            "n_epochs": args.epochs,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_targets": args.lora_targets,
            "lr": args.lr,
            "rollout_max_tokens": args.rollout_max_tokens,
            "rollout_temperature": args.rollout_temperature,
            "max_teacher_tokens": args.max_teacher_tokens,
            "truncation_count": truncation_count,
            "skipped_empty_rollouts": skipped_empty,
            "training_log": training_log,
            "per_step_losses": step_losses,
            "args": {k: str(v) for k, v in vars(args).items()},
        }, fh, indent=2)
    print(f"LoRA saved to {args.output_dir}")
    print(f"log saved to  {log_path}")


if __name__ == "__main__":
    main()
