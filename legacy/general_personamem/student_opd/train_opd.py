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


# ---------- Checkpoint pruning ----------

def find_latest_ckpt(output_dir: Path) -> Path | None:
    """Return the highest-step ckpt-step-N/ dir in output_dir, or None."""
    if not output_dir.exists():
        return None
    ckpts = sorted(
        output_dir.glob("ckpt-step-*"),
        key=lambda p: int(p.name.rsplit("-", 1)[-1]),
    )
    return ckpts[-1] if ckpts else None


def prune_checkpoints(output_dir: Path, keep: int) -> None:
    """Keep only the `keep` most recent ckpt-step-* dirs; delete older ones.

    Sort key is the integer step count; pruning is by mtime-independent
    order so it's robust to disk clock skew.
    """
    import shutil
    ckpts = sorted(
        output_dir.glob("ckpt-step-*"),
        key=lambda p: int(p.name.rsplit("-", 1)[-1]),
    )
    while len(ckpts) > keep:
        old = ckpts.pop(0)
        shutil.rmtree(old)
        print(f"  pruned old checkpoint: {old.name}", flush=True)


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
    ap.add_argument("--rollout-max-tokens", type=int, default=256,
                    help="User response p50 is ~190 tokens in PersonaMem; "
                         "160 truncated below median. 256 covers ~p75.")
    ap.add_argument("--rollout-temperature", type=float, default=1.0)
    ap.add_argument("--max-teacher-tokens", type=int, default=32768,
                    help="truncate teacher prefix from the left if longer")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-every", type=int, default=1,
                    help="Print a per-step line every N steps (default 1).")
    ap.add_argument("--smooth-window", type=int, default=10,
                    help="Rolling window (steps) for avg loss in log line.")
    ap.add_argument("--progress-every", type=int, default=50,
                    help="Every N steps, print a banner + rollout snippet "
                         "for eyeballing quality.")
    ap.add_argument("--max-samples", type=int, default=-1,
                    help="cap for quick sanity runs; -1 = all")
    ap.add_argument("--save-every", type=int, default=200,
                    help="Save a LoRA checkpoint every N steps. 0 disables "
                         "intermediate saves (only final is written).")
    ap.add_argument("--save-total-limit", type=int, default=2,
                    help="Keep only the N most recent intermediate "
                         "checkpoints (final is always kept separately).")
    ap.add_argument("--resume", action="store_true",
                    help="If set, auto-resume from the latest ckpt-step-N/ "
                         "in output_dir. Loads LoRA weights and optimizer "
                         "state if available, skips already-trained samples.")
    ap.add_argument("--resume-from-ckpt", type=Path, default=None,
                    help="Resume from a specific checkpoint directory "
                         "(overrides --resume auto-detection).")
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

    # --- Decide resume ---
    resume_ckpt: Path | None = None
    if args.resume_from_ckpt is not None:
        resume_ckpt = args.resume_from_ckpt
        if not (resume_ckpt / "adapter_config.json").exists():
            raise FileNotFoundError(
                f"--resume-from-ckpt pointed at {resume_ckpt} but no "
                f"adapter_config.json there"
            )
    elif args.resume:
        resume_ckpt = find_latest_ckpt(args.output_dir)
        if resume_ckpt is None:
            print(f"  --resume set but no ckpt-step-*/ in {args.output_dir}; "
                  f"starting fresh.")

    # --- Student: fresh LoRA OR resume ---
    if resume_ckpt is not None:
        from peft import PeftModel
        print(f"resuming LoRA from: {resume_ckpt}")
        student = PeftModel.from_pretrained(
            student_base, str(resume_ckpt), is_trainable=True
        )
    else:
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

    # --- Optimizer resume (optional — ckpts may predate this feature) ---
    start_step = 0
    if resume_ckpt is not None:
        start_step = int(resume_ckpt.name.rsplit("-", 1)[-1])
        opt_path = resume_ckpt / "optimizer.pt"
        if opt_path.exists():
            optimizer.load_state_dict(torch.load(opt_path, map_location=device))
            print(f"  optimizer state restored from {opt_path}")
        else:
            print(f"  no optimizer.pt in {resume_ckpt}; AdamW starts fresh "
                  f"(training will continue but momentum/variance reset)")
        print(f"  resuming at step {start_step}")

    # --- Training loop ---
    training_log = []
    step_losses = []
    truncation_count = 0
    skipped_empty = 0

    for epoch in range(args.epochs):
        # Deterministic shuffle: independent RNG so resume lands on the same
        # idx order as a fresh run would have gotten.
        idx = list(range(len(samples)))
        shuffle_rng = random.Random(args.seed + epoch)
        shuffle_rng.shuffle(idx)

        # If resuming inside this epoch, skip the first start_step samples.
        # (Multi-epoch resume is not handled — we assume start_step < len(idx);
        # the common case is single-epoch training.)
        total_steps = len(idx)
        if start_step > 0 and epoch == 0:
            if start_step >= total_steps:
                print(f"  start_step {start_step} >= samples in epoch "
                      f"({total_steps}); skipping to next epoch")
                continue
            skipped_for_resume = start_step
            idx = idx[start_step:]
            print(f"  [resume] skipping first {skipped_for_resume} samples "
                  f"of epoch {epoch}; {len(idx)} remain")
        else:
            skipped_for_resume = 0

        epoch_kl = 0.0
        epoch_tokens = 0
        t_epoch = time.time()
        t_step_last = time.time()

        for local_step, i in enumerate(idx):
            step = skipped_for_resume + local_step  # global step within epoch
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

            # Per-step line (default every step)
            if (step + 1) % args.log_every == 0:
                window = step_losses[-args.smooth_window:]
                losses_w = [x["loss"] for x in window]
                lens_w = [x["min_len"] for x in window]
                step_dt = time.time() - t_step_last
                t_step_last = time.time()
                print(f"  e{epoch} step {step+1}/{total_steps}: "
                      f"loss={loss.item():.4f} "
                      f"avg{len(window)}={sum(losses_w)/len(losses_w):.4f} "
                      f"resp_len={min_len}(avg{len(window)}={sum(lens_w)//len(lens_w)}) "
                      f"dt={step_dt:.1f}s",
                      flush=True)

            # Periodic intermediate LoRA + optimizer save + prune
            if args.save_every > 0 and (step + 1) % args.save_every == 0:
                ckpt_dir = args.output_dir / f"ckpt-step-{step+1}"
                student.save_pretrained(ckpt_dir)
                torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
                print(f"  [saved] {ckpt_dir} (+ optimizer.pt)", flush=True)
                prune_checkpoints(args.output_dir, args.save_total_limit)

            # Periodic progress banner with rollout snippet
            if args.progress_every > 0 and (step + 1) % args.progress_every == 0:
                snippet = tokenizer.decode(
                    response_ids, skip_special_tokens=False
                )[:160].replace("\n", " ")
                window = step_losses[-args.progress_every:]
                losses_w = [x["loss"] for x in window]
                lens_w = [x["min_len"] for x in window]
                elapsed = time.time() - t_epoch
                # ETA based on local_step progress (elapsed covers only post-resume work)
                steps_done_this_run = local_step + 1
                steps_remain_this_run = len(idx) - steps_done_this_run
                eta = (elapsed / steps_done_this_run) * steps_remain_this_run
                print(f"  --- [progress pid={args.persona_id} "
                      f"e{epoch} step {step+1}/{total_steps}] "
                      f"loss window min/mean/max = "
                      f"{min(losses_w):.3f}/{sum(losses_w)/len(losses_w):.3f}/"
                      f"{max(losses_w):.3f}  "
                      f"resp_len mean={sum(lens_w)//len(lens_w)}  "
                      f"elapsed={elapsed:.0f}s  eta={eta:.0f}s ---",
                      flush=True)
                print(f"      rollout: {snippet!r}", flush=True)

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
