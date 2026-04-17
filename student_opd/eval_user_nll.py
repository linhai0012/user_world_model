"""Phase-2 eval (1a): logit-level user-fit NLL.

Five conditions over the same N samples from opd_128k_pid{N}_k3.jsonl:

  - base_demo    : Instruct-2507, demographics + chatbot_prev
  - base_full    : Instruct-2507, full K=3 history (ends at chatbot_prev)
  - teacher_demo : R3 final, demographics + chatbot_prev
  - teacher_full : R3 final, full K=3 history
  - student_demo : base + per-user LoRA (merged), demographics + chatbot_prev

Every prompt ends at '<|im_start|>user\\n'; we score the GT user_response
tokens + closing '<|im_end|>\\n'. Lower mean NLL = better logit-level user
fit. All five conditions see the immediately-preceding assistant turn so
the model has a real query to react to (a pure demographics-only prompt
has no anchor for "which user turn am I simulating").

Three model loads total (base, teacher, base+LoRA) each covering 1-2
conditions. Single GPU is fine (~3-5 min at n=100).

Phase-2 thesis (what success looks like):
    student_demo  ≈ teacher_full   — LoRA made up for the missing context
    student_demo  ≪ base_demo      — LoRA adds value beyond demographics
    teacher_full  <  teacher_demo  — teacher uses context (sanity)
    base_full     <  base_demo     — context helps a vanilla base too

Usage:
    python dynamic_usersim/student_opd/eval_user_nll.py \\
        --persona-id 14 --teacher-path $R3 \\
        --lora-path dynamic_usersim/outputs/lora_pid14_r32_ep1_r3teacher \\
        --num-samples 100 \\
        --out-json dynamic_usersim/outputs/user_nll_pid14.json
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

_HERE = Path(__file__).resolve().parent
_TSFT = _HERE.parent / "teacher_sft"
for p in (_HERE, _TSFT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from config import MAX_SEQ_LEN, MODEL_NAME  # noqa: E402


# Condition descriptors: (name, model_key, context_mode)
# context_mode values understood by build_prompt_ids: demo-only, full, recent.
# Phase 2 baseline conditions still pass demo-only / full.
# Phase 2b adds `recent` (demographics + N intra-session exchanges) for the
# student view.
CONDITIONS_PHASE2 = [
    ("base_demo",    "base",    "demo-only"),
    ("base_full",    "base",    "full"),
    ("teacher_demo", "teacher", "demo-only"),
    ("teacher_full", "teacher", "full"),
    ("student_demo", "student", "demo-only"),
]

CONDITIONS_PHASE2B = [
    ("base_demo",      "base",    "demo-only"),
    ("base_recent",    "base",    "recent"),
    ("base_full",      "base",    "full"),
    ("teacher_demo",   "teacher", "demo-only"),
    ("teacher_full",   "teacher", "full"),
    ("student_recent", "student", "recent"),
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--persona-id", type=str, required=True)
    ap.add_argument("--base-model", type=str, default=MODEL_NAME)
    ap.add_argument("--teacher-path", type=Path, required=True)
    ap.add_argument("--lora-path", type=Path, required=True,
                    help="Single adapter dir (lora-mode=single) or parent dir "
                         "containing slow/+fast/ (lora-mode=dual).")
    ap.add_argument("--lora-mode", choices=["single", "dual"], default="single",
                    help="single = Phase 2; dual = Phase 2b Round 1.")
    ap.add_argument("--phase", choices=["phase2", "phase2b"], default="phase2",
                    help="phase2 = legacy 5 conditions; phase2b = adds "
                         "base_recent + student_recent (Phase 2b plan §7.4).")
    ap.add_argument("--data-path", type=Path, default=None,
                    help="Defaults to opd_128k_pid{N}_k3.jsonl")
    ap.add_argument("--num-samples", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    ap.add_argument("--lm-head-chunk", type=int, default=2048)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out-json", type=Path, required=True)
    return ap.parse_args()


# ---------- Prompt construction ----------

def build_prompt_ids(sample: dict, tokenizer, context_mode: str) -> list[int]:
    """Build ChatML prompt ending at '<|im_start|>user\\n'.

    context_mode:
      demo-only : [system: demographics, assistant: chatbot_prev]  (Phase 2)
      recent    : [system: demographics] + student_recent_messages (Phase 2b)
      full      : sample['history_messages']                        (teacher view)
    """
    if context_mode == "demo-only":
        msgs = [
            {"role": "system", "content": sample["demographics"]},
            {"role": "assistant", "content": sample["chatbot_prev"]},
        ]
    elif context_mode == "recent":
        recent = sample.get("student_recent_messages") or []
        if not recent:
            # Fall back to demo-only if data was built without N>0
            msgs = [
                {"role": "system", "content": sample["demographics"]},
                {"role": "assistant", "content": sample["chatbot_prev"]},
            ]
        else:
            msgs = [{"role": "system", "content": sample["demographics"]}]
            msgs.extend(recent)
    elif context_mode == "full":
        msgs = sample["history_messages"]
    else:
        raise ValueError(f"unknown context_mode: {context_mode}")

    ids: list[int] = []
    for m in msgs:
        s = f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        ids.extend(tokenizer.encode(s, add_special_tokens=False))
    ids.extend(
        tokenizer.encode("<|im_start|>user\n", add_special_tokens=False)
    )
    return ids


def score_gt_nll(model, tokenizer, prompt_ids: list[int], gt_text: str,
                 max_seq_len: int, lm_head_chunk: int) -> tuple[float, int, int]:
    """Score NLL of gt_text (+ '<|im_end|>\\n') under `model`.

    Returns (mean_nll, n_target_tokens, total_seq_len). Left-truncates the
    prompt to keep the target (and thus the predicted positions) intact.
    """
    content_ids = tokenizer.encode(gt_text, add_special_tokens=False)
    end_ids = tokenizer.encode("<|im_end|>\n", add_special_tokens=False)
    target_ids = content_ids + end_ids
    n_target = len(target_ids)

    total = len(prompt_ids) + n_target
    if total > max_seq_len:
        budget = max_seq_len - n_target
        prompt_ids = prompt_ids[-budget:]
        total = len(prompt_ids) + n_target

    full_ids = prompt_ids + target_ids
    device = torch.cuda.current_device()
    ids_t = torch.tensor([full_ids], dtype=torch.long, device=device)

    # Position p predicts token at p+1. Target starts at (total - n_target).
    pred_start = total - n_target - 1
    pred_end = total - 1

    with torch.no_grad():
        hidden = model.model(input_ids=ids_t,
                             use_cache=False).last_hidden_state[0]
        h_pred = hidden[pred_start:pred_end]          # [n_target, H]
        labels = torch.tensor(target_ids, device=device)

        total_nll = 0.0
        for i in range(0, n_target, lm_head_chunk):
            h = h_pred[i:i + lm_head_chunk]
            t = labels[i:i + lm_head_chunk]
            logits = model.lm_head(h).float()         # [c, V]
            total_nll += F.cross_entropy(logits, t, reduction="sum").item()

    return total_nll / max(n_target, 1), n_target, total


# ---------- Model loading ----------

def apply_liger() -> None:
    try:
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3
        apply_liger_kernel_to_qwen3(
            rope=True, swiglu=True, cross_entropy=False,
            fused_linear_cross_entropy=True, rms_norm=True,
        )
        print("[eval] liger applied")
    except ImportError:
        print("[eval] liger not available; continuing")


def load_hf_model(path: str | Path) -> torch.nn.Module:
    m = AutoModelForCausalLM.from_pretrained(
        str(path), torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", trust_remote_code=True,
    )
    m.config.use_cache = False
    m.eval()
    m.to(torch.cuda.current_device())
    return m


def load_student_merged(base_path: str, lora_path: Path,
                        lora_mode: str = "single") -> torch.nn.Module:
    """Load base + LoRA(s) and merge into base. Returns a plain HF model.

    lora_mode='single': lora_path is a single adapter dir.
    lora_mode='dual':   lora_path is a parent dir with slow/+fast/ subdirs.
    """
    base = load_hf_model(base_path)
    from peft import PeftModel
    if lora_mode == "single":
        m = PeftModel.from_pretrained(base, str(lora_path))
    elif lora_mode == "dual":
        slow_dir = Path(lora_path) / "slow"
        fast_dir = Path(lora_path) / "fast"
        if not (slow_dir / "adapter_config.json").exists():
            raise FileNotFoundError(f"missing slow adapter at {slow_dir}")
        if not (fast_dir / "adapter_config.json").exists():
            raise FileNotFoundError(f"missing fast adapter at {fast_dir}")
        m = PeftModel.from_pretrained(
            base, str(slow_dir), adapter_name="slow",
        )
        m.load_adapter(str(fast_dir), adapter_name="fast")
        m.set_adapter(["slow", "fast"])
    else:
        raise ValueError(f"unknown lora_mode: {lora_mode}")
    m = m.merge_and_unload()
    m.config.use_cache = False
    m.eval()
    m.to(torch.cuda.current_device())
    return m


# ---------- Main ----------

def main() -> None:
    args = parse_args()
    assert torch.cuda.is_available()
    torch.cuda.set_device(args.gpu)

    # Resolve data path
    if args.data_path is None:
        repo_root = Path(__file__).resolve().parents[2]
        args.data_path = (
            repo_root / "dynamic_usersim" / "outputs"
            / f"opd_128k_pid{args.persona_id}_k3.jsonl"
        )

    # Load + sample
    all_samples = [
        json.loads(L)
        for L in args.data_path.open("r", encoding="utf-8")
    ]
    rng = random.Random(args.seed)
    if 0 < args.num_samples < len(all_samples):
        samples = rng.sample(all_samples, args.num_samples)
    else:
        samples = all_samples
    print(f"[eval] pid={args.persona_id}: {len(samples)} samples "
          f"of {len(all_samples)} total (seed={args.seed})")

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True
    )
    apply_liger()

    conditions = (CONDITIONS_PHASE2B if args.phase == "phase2b"
                  else CONDITIONS_PHASE2)
    needed_modes = sorted({c[2] for c in conditions})

    # Pre-tokenize prompts per context mode (avoid redoing work per condition)
    prompts_by_mode: dict[str, list[list[int]]] = {}
    for mode in needed_modes:
        prompts_by_mode[mode] = [
            build_prompt_ids(s, tokenizer, mode) for s in samples
        ]
    # Length report
    for mode, ids_list in prompts_by_mode.items():
        lens = sorted(len(x) for x in ids_list)
        n = len(lens)
        print(f"  prompt[{mode}] len: min={lens[0]} p50={lens[n//2]} "
              f"p90={lens[min(n-1, 9*n//10)]} max={lens[-1]}")

    # Group conditions by model to minimize loads
    by_model: dict[str, list[tuple]] = defaultdict(list)
    for cond in conditions:
        by_model[cond[1]].append(cond)

    results: dict[str, list[dict]] = defaultdict(list)
    for model_key, conds in by_model.items():
        t0 = time.time()
        print(f"\n[eval] loading model: {model_key}")
        if model_key == "base":
            model = load_hf_model(args.base_model)
        elif model_key == "teacher":
            model = load_hf_model(args.teacher_path)
        elif model_key == "student":
            model = load_student_merged(
                args.base_model, args.lora_path, args.lora_mode,
            )
        else:
            raise ValueError(model_key)
        print(f"  loaded in {time.time()-t0:.1f}s  "
              f"({torch.cuda.memory_allocated()/1e9:.1f} GB)")

        for cond_name, _, mode in conds:
            print(f"\n[cond] {cond_name} ({mode})")
            t_cond = time.time()
            for i, sample in enumerate(samples):
                prompt_ids = prompts_by_mode[mode][i]
                mean_nll, n_target, total = score_gt_nll(
                    model, tokenizer, prompt_ids, sample["user_response"],
                    args.max_seq_len, args.lm_head_chunk,
                )
                results[cond_name].append({
                    "sample_idx": i,
                    "persona_id": sample["persona_id"],
                    "context_id": sample["context_id"][:8],
                    "session_idx": sample["session_idx"],
                    "user_turn_idx": sample["user_turn_idx"],
                    "mean_nll": mean_nll,
                    "n_target_tokens": n_target,
                    "total_seq_len": total,
                })
                if (i + 1) % 20 == 0:
                    last20 = results[cond_name][-20:]
                    avg = sum(r["mean_nll"] for r in last20) / len(last20)
                    print(f"  [{cond_name}] {i+1}/{len(samples)}  "
                          f"avg20={avg:.4f}", flush=True)
            n = len(results[cond_name])
            avg_nll = sum(r["mean_nll"] for r in results[cond_name]) / n
            print(f"  [{cond_name}] done  n={n}  mean_nll={avg_nll:.4f}  "
                  f"({time.time()-t_cond:.1f}s)")

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # ---- Summary ----
    print("\n" + "=" * 72)
    print(f"SUMMARY — eval_user_nll  pid={args.persona_id}  "
          f"phase={args.phase}  n={len(samples)}")
    print("=" * 72)
    print(f"{'condition':<16} {'mean_nll':>9}  {'vs base_demo':>13}  "
          f"{'vs teacher_full':>17}")
    summary = {
        cond: sum(r["mean_nll"] for r in results[cond]) / len(samples)
        for cond, _, _ in conditions
    }
    anchor_base = summary["base_demo"]
    anchor_tf = summary["teacher_full"]
    for cond_name, _, _ in conditions:
        m = summary[cond_name]
        print(f"{cond_name:<16} {m:>9.4f}  {m - anchor_base:>+13.4f}  "
              f"{m - anchor_tf:>+17.4f}")

    # Phase-aware diagnostics — pick the student condition that exists.
    student_key = ("student_recent" if args.phase == "phase2b"
                   else "student_demo")
    print(f"\n--- diagnostics (student = {student_key}) ---")
    student_nll = summary[student_key]
    gap_vs_teacher_full = student_nll - summary["teacher_full"]
    gap_vs_teacher_demo = student_nll - summary["teacher_demo"]
    gap_vs_base_demo = student_nll - summary["base_demo"]
    base_ctx_gain = summary["base_demo"] - summary["base_full"]
    teacher_ctx_gain = summary["teacher_demo"] - summary["teacher_full"]
    print(f"{student_key} − teacher_full = {gap_vs_teacher_full:+.4f}  "
          f"(thesis: LoRA at low-ctx matches teacher at full ctx)")
    print(f"{student_key} − teacher_demo = {gap_vs_teacher_demo:+.4f}  "
          f"(thesis: LoRA beats teacher when teacher has no ctx)")
    print(f"{student_key} − base_demo    = {gap_vs_base_demo:+.4f}  "
          f"(want strongly negative: LoRA adds beyond demographics)")
    if args.phase == "phase2b":
        # Isolate the contribution of the 2-turn context vs LoRA.
        print(f"base_recent    − base_demo    = "
              f"{summary['base_recent'] - anchor_base:+.4f}  "
              f"(2-turn context alone, no LoRA)")
        print(f"{student_key} − base_recent  = "
              f"{student_nll - summary['base_recent']:+.4f}  "
              f"(dual-LoRA gain on top of 2-turn context — Phase 2b §4 ablation)")
    print(f"base  context gain (demo−full)    = {base_ctx_gain:+.4f}")
    print(f"teacher context gain (demo−full)  = {teacher_ctx_gain:+.4f}")

    # Paired per-sample win rates
    n = len(samples)
    def wins(a: str, b: str) -> int:
        return sum(
            1 for i in range(n)
            if results[a][i]["mean_nll"] < results[b][i]["mean_nll"]
        )
    print(f"\n--- paired per-sample wins (lower NLL) ---")
    print(f"{student_key:<16} < base_demo     : "
          f"{wins(student_key,'base_demo')}/{n}")
    print(f"{student_key:<16} < teacher_demo  : "
          f"{wins(student_key,'teacher_demo')}/{n}")
    print(f"{student_key:<16} < teacher_full  : "
          f"{wins(student_key,'teacher_full')}/{n}")
    print(f"base_full        < base_demo     : {wins('base_full','base_demo')}/{n}")
    print(f"teacher_full     < teacher_demo  : "
          f"{wins('teacher_full','teacher_demo')}/{n}")
    if args.phase == "phase2b":
        print(f"{student_key:<16} < base_recent   : "
              f"{wins(student_key,'base_recent')}/{n}")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps({
        "args": {k: str(v) for k, v in vars(args).items()},
        "n_samples": len(samples),
        "per_condition": {k: v for k, v in results.items()},
        "summary": summary,
    }, indent=2, ensure_ascii=False))
    print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
