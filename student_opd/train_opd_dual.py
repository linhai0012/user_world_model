"""Phase-2b Round 1 OPD training: dual-LoRA per persona.

Differences vs train_opd.py (Phase 2 single-LoRA):

  - Dual LoRA structure (Phase 2b plan §3):
      LoRA_slow: target_modules = MLP {gate_proj, up_proj}
                 rank=32 alpha=64  lr=1e-5  (~52M params)
      LoRA_fast: target_modules = Attention {q,k,v,o}
                 rank=16 alpha=32  lr=2e-4  (~16.5M params)
      Both adapters active simultaneously (PEFT additive).
      Two optimizers, both .step() per training step.

  - Student input includes the last N intra-session exchanges
    (Phase 2b plan §4); falls back to legacy demo-only if a sample's
    `student_recent_messages` is empty.

  - KL loss: reverse KL = KL(student || teacher) — same formula as Phase 2
    and the Thinking Machines on-policy distillation recipe (mode-seeking,
    "low KL ↔ high prob of desirable behaviour from teacher's POV"; see
    https://thinkingmachines.ai/blog/on-policy-distillation/ ). Note: the
    classical Hinton-2015 KD literature confusingly uses "forward KL" for
    the same direction; we use the modern RLHF/TM convention. To enable
    gated KL (Round 2), pass --gated-kl.
  - Teacher OPD context: K=3 (unchanged; K=10 is Round 2b).

Layout on disk (per persona output dir):
    lora_pid{pid}_dual_s{Rs}f{Rf}_ep1_r3teacher/
        slow/                                    <- final dual-adapter
        fast/                                    <- final dual-adapter
        ckpt-step-200/{slow,fast}/               <- intermediate
        ckpt-step-200/opt_slow.pt
        ckpt-step-200/opt_fast.pt
        training_log.json

Usage (Isambard, single GPU per persona):
    python dynamic_usersim/student_opd/train_opd_dual.py \\
        --persona-id 14 \\
        --teacher-path $SCRATCHDIR/P-OPSD/teacher_sft_128k_k3/final \\
        --data-path dynamic_usersim/outputs/opd_128k_pid14_k3.jsonl \\
        --output-dir dynamic_usersim/outputs/lora_pid14_dual_s32f16_ep1_r3teacher

Sanity check before launching all 4:
    python ... --persona-id 14 ... --dry-run
        Loads on CPU, runs 1 forward, prints both adapter grad norms,
        prints student/teacher prefix lengths. Verifies both adapters
        actually receive gradients (Phase 2b §9 Step 2 sanity check).
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------- ChatML prefix construction ----------

def build_chatml_prefix(
    messages: list[dict],
    tokenizer,
    trailing_role: str = "user",
) -> list[int]:
    """Encode messages as ChatML, then append '<|im_start|>{trailing_role}\\n'."""
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


def build_student_prefix(sample: dict, tokenizer,
                          ctx_mode: str = "recent") -> list[int]:
    """Student view.

    ctx_mode = 'recent' (Phase 2b Round 1 default):
        demographics + last N exchanges of current session (ending at
        chatbot_prev). Falls back to demo-only if the sample has no
        `student_recent_messages`.

    ctx_mode = 'demo' (Phase 2 / Round 1b ablation):
        demographics + chatbot_prev only. Ignores student_recent_messages
        even if the data file has it. Use this to isolate the dual-LoRA
        structure effect from the 2-turn-context effect.
    """
    msgs: list[dict] = [{"role": "system", "content": sample["demographics"]}]
    if ctx_mode == "demo":
        msgs.append({"role": "assistant", "content": sample["chatbot_prev"]})
    elif ctx_mode == "recent":
        recent = sample.get("student_recent_messages") or []
        if recent:
            msgs.extend(recent)
        else:
            msgs.append({"role": "assistant", "content": sample["chatbot_prev"]})
    else:
        raise ValueError(f"unknown ctx_mode: {ctx_mode}")
    return build_chatml_prefix(msgs, tokenizer, trailing_role="user")


def build_teacher_prefix(sample: dict, tokenizer) -> list[int]:
    return build_chatml_prefix(
        sample["history_messages"], tokenizer, trailing_role="user"
    )


def truncate_teacher_prefix(
    prefix_ids: list[int], max_tokens: int
) -> tuple[list[int], bool]:
    if len(prefix_ids) <= max_tokens:
        return prefix_ids, False
    return prefix_ids[-max_tokens:], True


# ---------- Core OPD ops ----------

@torch.no_grad()
def student_rollout(
    model, tokenizer, prefix_ids, device,
    max_new_tokens, temperature, eos_token_id, pad_token_id,
):
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


def compute_response_logprobs(model, prefix_ids, response_ids, device):
    full = torch.tensor([prefix_ids + response_ids], device=device)
    logits = model(input_ids=full).logits[0]
    start = len(prefix_ids) - 1
    end = start + len(response_ids)
    return F.log_softmax(logits[start:end].float(), dim=-1)


# ---------- Dual-LoRA setup / save / load ----------

def activate_adapters(peft_model, names: list[str]) -> None:
    """Activate multiple LoRA adapters simultaneously.

    PEFT gotcha: `PeftModel.set_adapter(name)` only accepts a string and
    rejects lists with 'unhashable type: list' (it does `name not in
    self.peft_config` where peft_config is a dict). To activate several
    adapters at once we call the tuner layer below it (`LoraModel`), whose
    `set_adapter` accepts `list[str]`, propagates to every LoraLayer's
    `active_adapters`, and correctly toggles `requires_grad` so every
    listed adapter becomes trainable.
    """
    peft_model.base_model.set_adapter(names)


def setup_dual_lora(
    student_base,
    slow_rank: int, slow_alpha: int, slow_dropout: float,
    fast_rank: int, fast_alpha: int, fast_dropout: float,
    slow_targets: list[str], fast_targets: list[str],
):
    """Attach two non-overlapping LoRA adapters, both active."""
    cfg_slow = LoraConfig(
        r=slow_rank, lora_alpha=slow_alpha, lora_dropout=slow_dropout,
        target_modules=slow_targets, bias="none", task_type="CAUSAL_LM",
    )
    cfg_fast = LoraConfig(
        r=fast_rank, lora_alpha=fast_alpha, lora_dropout=fast_dropout,
        target_modules=fast_targets, bias="none", task_type="CAUSAL_LM",
    )
    student = get_peft_model(student_base, cfg_slow, adapter_name="slow")
    student.add_adapter("fast", cfg_fast)
    # Activate both — their LoRA increments add at every targeted module.
    activate_adapters(student, ["slow", "fast"])
    return student


def split_lora_params(student) -> tuple[list, list]:
    """Return (slow_params, fast_params) lists of trainable tensors.

    PEFT names adapter weights as `...lora_A.{adapter_name}.weight` etc.,
    so `f".{name}." in pname` is unambiguous against module/parameter names.
    """
    slow_params, fast_params = [], []
    for n, p in student.named_parameters():
        if not p.requires_grad:
            continue
        if ".slow." in n:
            slow_params.append(p)
        elif ".fast." in n:
            fast_params.append(p)
        # silently skip anything else; nothing else should be trainable
    return slow_params, fast_params


def save_dual_ckpt(student, opt_slow, opt_fast, ckpt_dir: Path) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    student.save_pretrained(
        ckpt_dir, selected_adapters=["slow", "fast"],
    )
    torch.save(opt_slow.state_dict(), ckpt_dir / "opt_slow.pt")
    torch.save(opt_fast.state_dict(), ckpt_dir / "opt_fast.pt")


def load_dual_lora_for_training(student_base, ckpt_dir: Path):
    """Resume both adapters into a fresh-base model as trainable."""
    slow_dir = ckpt_dir / "slow"
    fast_dir = ckpt_dir / "fast"
    if not (slow_dir / "adapter_config.json").exists():
        raise FileNotFoundError(f"missing slow adapter at {slow_dir}")
    if not (fast_dir / "adapter_config.json").exists():
        raise FileNotFoundError(f"missing fast adapter at {fast_dir}")
    student = PeftModel.from_pretrained(
        student_base, str(slow_dir), adapter_name="slow", is_trainable=True,
    )
    student.load_adapter(str(fast_dir), adapter_name="fast", is_trainable=True)
    activate_adapters(student, ["slow", "fast"])
    return student


# ---------- Checkpoint pruning ----------

def find_latest_ckpt(output_dir: Path) -> Path | None:
    if not output_dir.exists():
        return None
    ckpts = sorted(
        output_dir.glob("ckpt-step-*"),
        key=lambda p: int(p.name.rsplit("-", 1)[-1]),
    )
    return ckpts[-1] if ckpts else None


def prune_checkpoints(output_dir: Path, keep: int) -> None:
    """Keep only the `keep` most recent ckpt-step-*/ dirs.

    keep <= 0 disables pruning — every intermediate ckpt is preserved.
    Use this when you want the full learning curve (e.g. for best-ckpt
    oracle selection across 200/400/600/800/... evaluations).
    """
    if keep <= 0:
        return
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
    ap.add_argument("--data-path", type=Path, default=None)
    ap.add_argument("--teacher-path", type=Path, required=True)
    ap.add_argument("--student-base", type=str,
                    default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--output-dir", type=Path, default=None)

    # Dual LoRA
    ap.add_argument("--slow-rank", type=int, default=32)
    ap.add_argument("--slow-alpha", type=int, default=64)
    ap.add_argument("--slow-dropout", type=float, default=0.0)
    ap.add_argument("--slow-targets", nargs="+",
                    default=["gate_proj", "up_proj"],
                    help="MLP modules — Phase 2b §3.2")
    ap.add_argument("--slow-lr", type=float, default=1e-5,
                    help="20x lower than fast — slow facts adapter")
    ap.add_argument("--fast-rank", type=int, default=16)
    ap.add_argument("--fast-alpha", type=int, default=32)
    ap.add_argument("--fast-dropout", type=float, default=0.0)
    ap.add_argument("--fast-targets", nargs="+",
                    default=["q_proj", "k_proj", "v_proj", "o_proj"],
                    help="Attention modules — Phase 2b §3.2")
    ap.add_argument("--fast-lr", type=float, default=2e-4,
                    help="Same as Phase 2 single-LoRA — routing adapter")

    # Student view at training time
    ap.add_argument("--student-ctx", choices=["recent", "demo"],
                    default="recent",
                    help="recent = demographics + 2-turn intra-session context "
                         "(Phase 2b Round 1 default). "
                         "demo = demographics + chatbot_prev only (Phase 2 / "
                         "Round 1b ablation — isolates dual structure effect).")

    # Rollout / KL
    ap.add_argument("--rollout-max-tokens", type=int, default=256)
    ap.add_argument("--rollout-temperature", type=float, default=1.0)
    ap.add_argument("--max-teacher-tokens", type=int, default=32768)
    ap.add_argument("--grad-clip", type=float, default=1.0)

    # Gated KL (Phase 2b Round 2 family). Default off = R1 / R1b behaviour.
    ap.add_argument("--gated-kl", action="store_true",
                    help="Enable per-token gating (Phase 2b §5). The gate "
                         "decides which rollout tokens contribute to the "
                         "reverse-KL loss. Mode set by --gate-mode.")
    ap.add_argument("--gate-mode", choices=["entropy", "joint"],
                    default="entropy",
                    help="entropy (R2a default): gate=H_teacher+margin<H_student. "
                         "joint (R2c): gate=(H_teacher<tau) AND "
                         "(argmax_teacher!=argmax_student). Joint gate "
                         "addresses two R2a failure modes — Instruct base "
                         "overconfidence drowning out the entropy signal, "
                         "and gating closing on tokens where student already "
                         "agrees with teacher anyway.")
    ap.add_argument("--kl-gate-margin", type=float, default=0.0,
                    help="ENTROPY mode only: margin for H_t + margin < H_s. "
                         "0.0 = strict comparison; >0 = require teacher "
                         "markedly more confident; <0 = relax the gate.")
    ap.add_argument("--gate-tau", type=float, default=1.0,
                    help="JOINT mode only: teacher confidence threshold. "
                         "Default 1.0 nat (~top-3 dominance). Lower = "
                         "stricter (only train on highly-confident teacher "
                         "predictions). Higher = include less confident "
                         "teacher predictions.")

    # Loop
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--smooth-window", type=int, default=10)
    ap.add_argument("--progress-every", type=int, default=50)
    ap.add_argument("--max-samples", type=int, default=-1)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--save-total-limit", type=int, default=2)

    # Resume
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--resume-from-ckpt", type=Path, default=None)

    # Misc
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true",
                    help="CPU forward+backward on 1 sample; reports both "
                         "adapter grad norms; no checkpoint written.")
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
            / f"lora_pid{args.persona_id}_dual_s{args.slow_rank}"
              f"f{args.fast_rank}_ep{args.epochs}_r3teacher"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- Device / dtype ---
    if args.dry_run:
        device = torch.device("cpu")
        dtype = torch.float32
    else:
        device = torch.device(f"cuda:{args.gpu}")
        dtype = torch.bfloat16

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(
        args.student_base, trust_remote_code=True,
    )
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    pad_id = tokenizer.pad_token_id or im_end_id
    assert isinstance(im_end_id, int) and im_end_id > 0

    # --- Data ---
    with args.data_path.open("r", encoding="utf-8") as fh:
        samples = [json.loads(line) for line in fh]
    if args.max_samples > 0:
        samples = samples[: args.max_samples]
    if args.dry_run:
        samples = samples[:1]

    n_with_recent = sum(
        1 for s in samples if s.get("student_recent_messages")
    )
    print(f"[pid={args.persona_id}] loaded {len(samples)} samples "
          f"from {args.data_path}")
    print(f"  samples with student_recent_messages: "
          f"{n_with_recent}/{len(samples)}")
    print(f"  student_ctx mode: {args.student_ctx}")
    if args.student_ctx == "recent" and n_with_recent == 0:
        print(f"  WARNING: --student-ctx recent but no sample has "
              f"student_recent_messages — will silently fall back to "
              f"demo-only for every sample. Re-run build_opd_data.py "
              f"with --n-context-turns 2 first?")

    # --- Teacher (frozen) ---
    print(f"loading teacher: {args.teacher_path}")
    t0 = time.time()
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_path, torch_dtype=dtype, trust_remote_code=True,
    ).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"  teacher loaded in {time.time()-t0:.1f}s")

    # --- Student base ---
    print(f"loading student base: {args.student_base}")
    t0 = time.time()
    student_base = AutoModelForCausalLM.from_pretrained(
        args.student_base, torch_dtype=dtype, trust_remote_code=True,
    ).to(device)
    print(f"  student base loaded in {time.time()-t0:.1f}s")

    # --- Decide resume ---
    resume_ckpt: Path | None = None
    if args.resume_from_ckpt is not None:
        resume_ckpt = args.resume_from_ckpt
    elif args.resume:
        resume_ckpt = find_latest_ckpt(args.output_dir)
        if resume_ckpt is None:
            print(f"  --resume set but no ckpt-step-*/ in {args.output_dir}; "
                  f"starting fresh.")

    # --- Dual LoRA: fresh OR resume ---
    if resume_ckpt is not None:
        print(f"resuming dual LoRA from: {resume_ckpt}")
        student = load_dual_lora_for_training(student_base, resume_ckpt)
    else:
        student = setup_dual_lora(
            student_base,
            slow_rank=args.slow_rank, slow_alpha=args.slow_alpha,
            slow_dropout=args.slow_dropout,
            fast_rank=args.fast_rank, fast_alpha=args.fast_alpha,
            fast_dropout=args.fast_dropout,
            slow_targets=args.slow_targets,
            fast_targets=args.fast_targets,
        )

    slow_params, fast_params = split_lora_params(student)
    n_slow = sum(p.numel() for p in slow_params)
    n_fast = sum(p.numel() for p in fast_params)
    n_total = sum(p.numel() for p in student.parameters())
    print(f"  LoRA_slow: {n_slow:,} params (targets={args.slow_targets})")
    print(f"  LoRA_fast: {n_fast:,} params (targets={args.fast_targets})")
    print(f"  trainable total: {n_slow + n_fast:,} / {n_total:,} "
          f"({100*(n_slow+n_fast)/n_total:.3f}%)")
    if n_slow == 0 or n_fast == 0:
        raise RuntimeError(
            f"one of the adapters has zero trainable params — "
            f"check target_modules and adapter naming "
            f"(slow={n_slow}, fast={n_fast})"
        )

    if not args.dry_run:
        alloc = torch.cuda.memory_allocated(device) / 1e9
        print(f"  GPU memory after both models: {alloc:.1f} GB")

    # --- Two optimizers ---
    opt_slow = torch.optim.AdamW(slow_params, lr=args.slow_lr)
    opt_fast = torch.optim.AdamW(fast_params, lr=args.fast_lr)

    # --- Optimizer resume ---
    start_step = 0
    if resume_ckpt is not None:
        start_step = int(resume_ckpt.name.rsplit("-", 1)[-1])
        for opt, fname, label in [
            (opt_slow, "opt_slow.pt", "slow"),
            (opt_fast, "opt_fast.pt", "fast"),
        ]:
            opath = resume_ckpt / fname
            if opath.exists():
                opt.load_state_dict(torch.load(opath, map_location=device))
                print(f"  optimizer[{label}] state restored from {opath}")
            else:
                print(f"  no {fname} in {resume_ckpt}; AdamW[{label}] fresh")
        print(f"  resuming at step {start_step}")

    # --- Training loop ---
    training_log: list[dict] = []
    step_losses: list[dict] = []
    truncation_count = 0
    skipped_empty = 0
    n_fully_gated_skipped = 0  # gated KL only: rollouts where every token gated out

    for epoch in range(args.epochs):
        idx = list(range(len(samples)))
        shuffle_rng = random.Random(args.seed + epoch)
        shuffle_rng.shuffle(idx)

        total_steps = len(idx)
        if start_step > 0 and epoch == 0:
            if start_step >= total_steps:
                print(f"  start_step {start_step} >= samples in epoch "
                      f"({total_steps}); skipping")
                continue
            skipped_for_resume = start_step
            idx = idx[start_step:]
            print(f"  [resume] skipping first {skipped_for_resume} samples; "
                  f"{len(idx)} remain")
        else:
            skipped_for_resume = 0

        epoch_kl = 0.0
        epoch_tokens = 0
        t_epoch = time.time()
        t_step_last = time.time()

        for local_step, i in enumerate(idx):
            step = skipped_for_resume + local_step
            sample = samples[i]

            s_prefix = build_student_prefix(sample, tokenizer, args.student_ctx)
            t_prefix = build_teacher_prefix(sample, tokenizer)
            t_prefix, truncated = truncate_teacher_prefix(
                t_prefix, args.max_teacher_tokens,
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
                      tokenizer.decode(response_ids,
                                       skip_special_tokens=False)[:200])

            # 2. Teacher logprobs
            with torch.no_grad():
                t_lp = compute_response_logprobs(
                    teacher, t_prefix, response_ids, device,
                )

            # 3. Student logprobs (dual adapters active)
            student.train()
            s_lp = compute_response_logprobs(
                student, s_prefix, response_ids, device,
            )

            # 4. Reverse KL(student || teacher) per-token. TM convention.
            #    Optionally gated by per-token entropy comparison (Round 2).
            min_len = min(s_lp.shape[0], t_lp.shape[0])
            s_lp = s_lp[:min_len]
            t_lp = t_lp[:min_len]
            kl_per_token = (s_lp.exp() * (s_lp - t_lp)).sum(dim=-1)  # [T]

            if args.gated_kl:
                # Per-token gate. Two modes:
                #   entropy: H_t + margin < H_s — protect student on tokens
                #            where teacher would be uncertain. R2a default.
                #   joint:   (H_t < tau) AND (argmax_t != argmax_s) — train
                #            only when teacher is genuinely confident AND
                #            student doesn't already agree. R2c default.
                #            Addresses R2a failure: entropy alone gets
                #            drowned out by Instruct-base overconfidence
                #            and wastes effort on tokens where s & t agree.
                H_t = -(t_lp.exp() * t_lp).sum(dim=-1)               # [T]
                if args.gate_mode == "entropy":
                    H_s = -(s_lp.exp() * s_lp).sum(dim=-1)           # [T]
                    gate = (H_t + args.kl_gate_margin < H_s).float() # [T]
                elif args.gate_mode == "joint":
                    top_t = t_lp.argmax(dim=-1)                      # [T]
                    top_s = s_lp.argmax(dim=-1)                      # [T]
                    disagree = (top_t != top_s)
                    confident = (H_t < args.gate_tau)
                    gate = (disagree & confident).float()            # [T]
                else:
                    raise ValueError(f"unknown gate_mode: {args.gate_mode}")
                gate_sum_val = gate.sum().item()

                if gate_sum_val == 0.0:
                    # Whole rollout gated out — student knows everything
                    # teacher could teach on this sample. Log + skip.
                    n_fully_gated_skipped += 1
                    step_losses.append({
                        "epoch": epoch, "step": step, "loss": 0.0,
                        "min_len": min_len, "sample_idx": i,
                        "slow_gnorm": 0.0, "fast_gnorm": 0.0,
                        "gate_ratio": 0.0, "fully_gated": True,
                        "context_id": sample["context_id"],
                        "session_idx": sample["session_idx"],
                    })
                    if args.dry_run:
                        print(f"DRY: gate_ratio=0.0 (all tokens gated) — "
                              f"would skip backward in real training")
                        break
                    continue  # next sample, no backward

                loss = (gate * kl_per_token).sum() / gate_sum_val
                gate_ratio = gate_sum_val / gate.numel()
            else:
                loss = kl_per_token.mean()
                gate_ratio = 1.0  # convention: ungated = "all tokens used"

            # 5. Backward — single backward populates grads on BOTH adapters
            loss.backward()

            # Sanity: per-adapter pre-clip grad norms
            slow_gnorm = torch.nn.utils.clip_grad_norm_(
                slow_params, args.grad_clip,
            )
            fast_gnorm = torch.nn.utils.clip_grad_norm_(
                fast_params, args.grad_clip,
            )

            if args.dry_run:
                # Wiring check: did autograd actually reach both adapter
                # parameter sets? `p.grad is not None` proves the param was
                # part of the autograd graph. Magnitude is a separate test:
                # when student_base == teacher_path AND LoRA is fresh
                # (B=0), forward pass through LoRA contributes nothing, so
                # student==teacher pointwise → KL is exactly at its minimum
                # → gradient is mathematically zero. That's expected for
                # a same-model dry-run; with the real R3 teacher loss>0.
                slow_with_grad = sum(1 for p in slow_params if p.grad is not None)
                fast_with_grad = sum(1 for p in fast_params if p.grad is not None)
                slow_gn = float(slow_gnorm)
                fast_gn = float(fast_gnorm)
                gate_desc = (f" (gated, mode={args.gate_mode})"
                             if args.gated_kl else " (ungated)")
                print(f"DRY: loss={loss.item():.6f} min_len={min_len} "
                      f"gate_ratio={gate_ratio:.3f}{gate_desc}")
                print(f"     slow params reached by autograd: "
                      f"{slow_with_grad}/{len(slow_params)}  gnorm={slow_gn:.6e}")
                print(f"     fast params reached by autograd: "
                      f"{fast_with_grad}/{len(fast_params)}  gnorm={fast_gn:.6e}")

                wired = (slow_with_grad == len(slow_params)
                         and fast_with_grad == len(fast_params))
                if not wired:
                    print("  ERROR: some adapter params not in autograd graph "
                          "— check setup")
                elif slow_gn < 1e-6 and fast_gn < 1e-6 and loss.item() < 1e-6:
                    teacher_is_base = (str(args.teacher_path).rstrip("/") ==
                                       args.student_base)
                    note = (" (teacher_path == student_base — student equals "
                            "teacher pointwise so KL is exactly 0; with the "
                            "real R3 SFT teacher loss>0 will produce nonzero "
                            "gradients on both adapters)" if teacher_is_base
                            else " (loss is at its minimum on this sample)")
                    print(f"  OK wiring: autograd reached every adapter "
                          f"param. Gradients are zero because loss=0{note}.")
                else:
                    print("  OK: both adapters received non-zero gradients")
                break

            opt_slow.step()
            opt_fast.step()
            opt_slow.zero_grad()
            opt_fast.zero_grad()

            epoch_kl += loss.item() * min_len
            epoch_tokens += min_len
            step_losses.append({
                "epoch": epoch, "step": step, "loss": loss.item(),
                "min_len": min_len, "sample_idx": i,
                "slow_gnorm": float(slow_gnorm),
                "fast_gnorm": float(fast_gnorm),
                "gate_ratio": gate_ratio,
                "fully_gated": False,
                "context_id": sample["context_id"],
                "session_idx": sample["session_idx"],
            })

            if (step + 1) % args.log_every == 0:
                window = step_losses[-args.smooth_window:]
                losses_w = [x["loss"] for x in window]
                lens_w = [x["min_len"] for x in window]
                step_dt = time.time() - t_step_last
                t_step_last = time.time()
                gate_str = ""
                if args.gated_kl:
                    gates_w = [x.get("gate_ratio", 1.0) for x in window]
                    gate_str = (f"gate={gate_ratio:.2f}"
                                f"(avg{len(window)}={sum(gates_w)/len(gates_w):.2f}) ")
                print(f"  e{epoch} step {step+1}/{total_steps}: "
                      f"loss={loss.item():.4f} "
                      f"avg{len(window)}={sum(losses_w)/len(losses_w):.4f} "
                      f"slow_gn={float(slow_gnorm):.4f} "
                      f"fast_gn={float(fast_gnorm):.4f} "
                      f"{gate_str}"
                      f"resp_len={min_len}"
                      f"(avg{len(window)}={sum(lens_w)//len(lens_w)}) "
                      f"dt={step_dt:.1f}s",
                      flush=True)

            if args.save_every > 0 and (step + 1) % args.save_every == 0:
                ckpt_dir = args.output_dir / f"ckpt-step-{step+1}"
                save_dual_ckpt(student, opt_slow, opt_fast, ckpt_dir)
                print(f"  [saved dual ckpt] {ckpt_dir} "
                      f"(+ opt_slow.pt + opt_fast.pt)", flush=True)
                prune_checkpoints(args.output_dir, args.save_total_limit)

            if args.progress_every > 0 and (step + 1) % args.progress_every == 0:
                snippet = tokenizer.decode(
                    response_ids, skip_special_tokens=False,
                )[:160].replace("\n", " ")
                window = step_losses[-args.progress_every:]
                losses_w = [x["loss"] for x in window]
                lens_w = [x["min_len"] for x in window]
                slow_gn_w = [x["slow_gnorm"] for x in window]
                fast_gn_w = [x["fast_gnorm"] for x in window]
                elapsed = time.time() - t_epoch
                steps_done_this_run = local_step + 1
                steps_remain_this_run = len(idx) - steps_done_this_run
                eta = (elapsed / steps_done_this_run) * steps_remain_this_run
                print(f"  --- [progress pid={args.persona_id} "
                      f"e{epoch} step {step+1}/{total_steps}] "
                      f"loss min/mean/max = "
                      f"{min(losses_w):.3f}/{sum(losses_w)/len(losses_w):.3f}/"
                      f"{max(losses_w):.3f}  "
                      f"slow_gn mean={sum(slow_gn_w)/len(slow_gn_w):.3f}  "
                      f"fast_gn mean={sum(fast_gn_w)/len(fast_gn_w):.3f}  "
                      f"resp_len mean={sum(lens_w)//len(lens_w)}  "
                      f"elapsed={elapsed:.0f}s eta={eta:.0f}s ---",
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

    # --- Final save (root dir, both adapters as subdirs) ---
    student.save_pretrained(args.output_dir, selected_adapters=["slow", "fast"])
    log_path = args.output_dir / "training_log.json"
    with log_path.open("w", encoding="utf-8") as fh:
        json.dump({
            "persona_id": args.persona_id,
            "n_samples": len(samples),
            "n_epochs": args.epochs,
            "slow_rank": args.slow_rank, "slow_alpha": args.slow_alpha,
            "slow_targets": args.slow_targets, "slow_lr": args.slow_lr,
            "fast_rank": args.fast_rank, "fast_alpha": args.fast_alpha,
            "fast_targets": args.fast_targets, "fast_lr": args.fast_lr,
            "student_ctx": args.student_ctx,
            "rollout_max_tokens": args.rollout_max_tokens,
            "rollout_temperature": args.rollout_temperature,
            "max_teacher_tokens": args.max_teacher_tokens,
            "gated_kl": args.gated_kl,
            "gate_mode": args.gate_mode,
            "kl_gate_margin": args.kl_gate_margin,
            "gate_tau": args.gate_tau,
            "n_fully_gated_skipped": n_fully_gated_skipped,
            "truncation_count": truncation_count,
            "skipped_empty_rollouts": skipped_empty,
            "training_log": training_log,
            "per_step_losses": step_losses,
            "args": {k: str(v) for k, v in vars(args).items()},
        }, fh, indent=2)
    print(f"Dual LoRA saved to {args.output_dir}/{{slow,fast}}/")
    print(f"log saved to  {log_path}")


if __name__ == "__main__":
    main()
