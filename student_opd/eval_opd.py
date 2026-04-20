"""Phase-2 MCQ-PPL eval for student UserSim (base +/- per-user LoRA).

Reuses the §9.5 scoring protocol from teacher_sft/eval_mcq_ppl.py — score
each of the 4 choices as an assistant continuation under the model, pick
argmin(mean NLL). Then filters to one persona's MCQs (since the LoRA is
per-user).

Two compare runs per persona:
  (1) --base-model Qwen/Qwen3-4B  (base, no LoRA)   -> floor
  (2) --base-model Qwen/Qwen3-4B  --lora-path ...   -> UserSim with LoRA

Context mode — Phase 2's canonical promise is "0 tokens of inference
context" (plan §11.1), so the default is --context-mode demo-only:
the model sees only the persona card + the MCQ's user question, then scores
each choice. --context-mode full matches eval_mcq_ppl.py protocol (full
conversation history up to end_index) for apples-to-apples comparison with
§4.4.3.

Launch (single-node, 4-GPU DP):
    torchrun --standalone --nnodes=1 --nproc_per_node=4 \
        dynamic_usersim/student_opd/eval_opd.py \
        --persona-id 14 \
        --lora-path dynamic_usersim/outputs/lora_pid14_r32_ep1 \
        --context-mode demo-only \
        --out-json dynamic_usersim/outputs/mcqppl_pid14_lora.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM

# Reuse helpers from teacher_sft + data_prep.
_HERE = Path(__file__).resolve().parent
_TSFT = _HERE.parent / "teacher_sft"
_DP = _HERE.parent / "data_prep"
for p in (_HERE, _TSFT, _DP):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from config import MAX_SEQ_LEN, MODEL_NAME  # noqa: E402
from load_personamem import load_contexts, load_questions, strip_role_prefix  # noqa: E402
from tokenize_teacher_sft import SFTTokenizer  # noqa: E402
from eval_mcq import parse_choices, build_context_messages  # noqa: E402
from eval_mcq_ppl import (  # noqa: E402
    apply_liger, init_dist, score_choice_ppl,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", type=str, default=MODEL_NAME,
                    help="HF base model (default config.MODEL_NAME)")
    ap.add_argument("--lora-path", type=Path, default=None,
                    help="PEFT LoRA adapter dir; omit for base-only eval. "
                         "For Phase 2b dual LoRA pass the parent dir "
                         "containing slow/ and fast/ subdirs and set "
                         "--lora-mode dual.")
    ap.add_argument("--lora-mode", choices=["single", "dual"], default="single",
                    help="single = legacy Phase 2 (one adapter at lora-path). "
                         "dual = Phase 2b Round 1 (slow/ + fast/ subdirs).")
    ap.add_argument("--persona-id", type=str, required=True)
    ap.add_argument("--mcq-version", choices=["32k", "128k", "1M"],
                    default="128k")
    ap.add_argument("--context-mode",
                    choices=["demo-only", "last-n", "full", "recent-turns"],
                    default="demo-only",
                    help="demo-only: just the persona card + MCQ question "
                         "(Phase 2 canonical 0-token-inference). "
                         "last-n: keep the last N sessions. "
                         "recent-turns: persona card + last N user/assistant "
                         "exchanges within the current session (Phase 2b). "
                         "full: entire history up to end_index.")
    ap.add_argument("--last-n-sessions", type=int, default=3,
                    help="only used when --context-mode=last-n")
    ap.add_argument("--recent-turns", type=int, default=2,
                    help="only used when --context-mode=recent-turns; "
                         "number of user/assistant exchanges (=2*N msgs)")
    ap.add_argument("--num-mcqs", type=int, default=-1,
                    help="-1 = all MCQs for this persona")
    ap.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    ap.add_argument("--lm-head-chunk", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-json", type=Path, required=True)
    return ap.parse_args()


def load_model_with_lora(base_path: str | Path,
                         lora_path: Path | None,
                         lora_mode: str = "single",
                         attn_impl: str = "flash_attention_2") -> torch.nn.Module:
    """Load base; optionally wrap with PEFT and merge_and_unload.

    lora_mode='single': lora_path is a single adapter dir (Phase 2 layout).
    lora_mode='dual':   lora_path is a parent dir holding slow/ + fast/
                        subdirs (Phase 2b Round 1 layout). Both adapters
                        are loaded, activated together, then merged into
                        base weights so the downstream PPL scorer sees a
                        plain HF model.

    attn_impl: passed to from_pretrained. Use "sdpa" on Blackwell (B200)
               if flash-attn 2 version predates sm_100 support.
    """
    model = AutoModelForCausalLM.from_pretrained(
        str(base_path),
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        trust_remote_code=True,
    )
    # Move base to GPU BEFORE LoRA wrapping so merge_and_unload happens on
    # GPU (not CPU). Merging on CPU spikes host RAM to ~2x base weights
    # which gets SIGKILL'd under tight cgroup limits on some clusters
    # (e.g. KCL CREATE). GPU side has the headroom.
    model.to(torch.cuda.current_device())
    if lora_path is not None:
        from peft import PeftModel
        is_rank0 = int(os.environ.get("RANK", 0)) == 0
        if lora_mode == "single":
            if is_rank0:
                print(f"[eval] wrapping with single LoRA: {lora_path}")
            model = PeftModel.from_pretrained(model, str(lora_path))
            model = model.merge_and_unload()
        elif lora_mode == "dual":
            slow_dir = Path(lora_path) / "slow"
            fast_dir = Path(lora_path) / "fast"
            if not (slow_dir / "adapter_config.json").exists():
                raise FileNotFoundError(f"missing slow adapter at {slow_dir}")
            if not (fast_dir / "adapter_config.json").exists():
                raise FileNotFoundError(f"missing fast adapter at {fast_dir}")
            if is_rank0:
                print(f"[eval] wrapping with dual LoRA: "
                      f"slow={slow_dir} fast={fast_dir}")
            model = PeftModel.from_pretrained(
                model, str(slow_dir), adapter_name="slow",
            )
            model.load_adapter(str(fast_dir), adapter_name="fast")
            # PeftModel.set_adapter rejects lists; go through tuner layer.
            model.base_model.set_adapter(["slow", "fast"])
            model = model.merge_and_unload()
        else:
            raise ValueError(f"unknown lora_mode: {lora_mode}")
    model.config.use_cache = False
    model.eval()
    return model


def get_demographics(raw_ctx: list[dict]) -> str:
    """First system message content of a shared_context = persona card."""
    for m in raw_ctx:
        if m["role"] == "system":
            return strip_role_prefix(m["content"], "system")
    raise ValueError("shared_context has no system message")


def build_eval_context(raw_ctx: list[dict], end_index: int,
                       mode: str, last_n: int,
                       recent_turns: int = 2) -> list[dict]:
    """Build context messages per chosen mode.

    recent-turns: persona card + the last 2*recent_turns user/assistant
    messages from raw_ctx[:end_index] (intra-session anchor for Phase 2b
    student input). System messages from prior session restatements are
    dropped — the persona card is already included once at the top.
    """
    if mode == "demo-only":
        return [{"role": "system", "content": get_demographics(raw_ctx)}]
    if mode == "last-n":
        return build_context_messages(raw_ctx, end_index, last_n)
    if mode == "full":
        return build_context_messages(raw_ctx, end_index, None)
    if mode == "recent-turns":
        demo_msg = {"role": "system", "content": get_demographics(raw_ctx)}
        sliced = raw_ctx[:end_index]
        cleaned = [
            {"role": m["role"],
             "content": strip_role_prefix(m["content"], m["role"])}
            for m in sliced if m["role"] != "system"
        ]
        n_msgs = recent_turns * 2
        recent = cleaned[-n_msgs:] if len(cleaned) > n_msgs else cleaned
        return [demo_msg] + recent
    raise ValueError(f"unknown context mode: {mode}")


def main() -> None:
    args = parse_args()
    assert torch.cuda.is_available()
    rank, world_size, is_dist = init_dist()

    # --- Data ---
    if rank == 0:
        print(f"[eval] loading MCQs + contexts ({args.mcq_version})")
    mcqs = load_questions(args.mcq_version)
    ctxs = load_contexts(args.mcq_version)
    mcqs = [q for q in mcqs
            if q["shared_context_id"] in ctxs
            and q["persona_id"] == args.persona_id]
    rng = random.Random(args.seed)
    rng.shuffle(mcqs)
    if args.num_mcqs > 0:
        mcqs = mcqs[: args.num_mcqs]
    if rank == 0:
        print(f"[eval] persona={args.persona_id} "
              f"evaluating {len(mcqs)} MCQs "
              f"(context={args.context_mode}, world_size={world_size})")

    # --- Tokenizer + model ---
    sft_tok = SFTTokenizer(model_name=args.base_model, max_len=args.max_seq_len)
    apply_liger()
    if rank == 0:
        print(f"[eval] loading base: {args.base_model}"
              f"{' + LoRA' if args.lora_path else ' (no LoRA)'}")
    model = load_model_with_lora(
        args.base_model, args.lora_path, args.lora_mode,
    )

    # --- Score MCQs (DP split) ---
    my_mcqs = mcqs[rank::world_size]
    local_results: list[dict] = []
    t0 = time.time()
    for i, q in enumerate(my_mcqs):
        t_mcq = time.time()
        ctx_msgs = build_eval_context(
            ctxs[q["shared_context_id"]],
            int(q["end_index_in_shared_context"]),
            args.context_mode, args.last_n_sessions, args.recent_turns,
        )
        question = q["user_question_or_message"]
        choices = parse_choices(q["all_options"])

        scores: list[dict] = []
        for label, choice_text in choices:
            mean_nll, n_target = score_choice_ppl(
                model, sft_tok, ctx_msgs, question, choice_text,
                args.max_seq_len, args.lm_head_chunk,
            )
            scores.append({
                "label": label,
                "choice": choice_text,
                "mean_nll": mean_nll,
                "ppl": float(torch.exp(torch.tensor(mean_nll)).item()),
                "n_target_tokens": n_target,
            })
        best = min(scores, key=lambda x: x["mean_nll"])
        pred_letter = best["label"]
        correct = q["correct_answer"].strip("()").lower()
        is_correct = (pred_letter == correct)

        local_results.append({
            "global_idx": mcqs.index(q),
            "persona_id": q["persona_id"],
            "question_id": q["question_id"],
            "qtype_canonical": q["qtype_canonical"],
            "topic": q["topic"],
            "correct_answer": correct,
            "pred_answer": pred_letter,
            "is_correct": is_correct,
            "scores": scores,
            "wall_s": time.time() - t_mcq,
        })
        score_map = {s["label"]: s["mean_nll"] for s in scores}
        gt_nll = score_map.get(correct, float("nan"))
        gap = best["mean_nll"] - gt_nll
        print(f"  [rank{rank}] {i+1}/{len(my_mcqs)} "
              f"qtype={q['qtype_canonical']} "
              f"pred={pred_letter} gt={correct} "
              f"{'OK' if is_correct else 'MISS'} "
              f"gt_nll={gt_nll:.3f} best_nll={best['mean_nll']:.3f} "
              f"gap={gap:+.3f} ({time.time()-t_mcq:.1f}s)", flush=True)

    # --- Gather ---
    if is_dist:
        gathered = [None] * world_size if rank == 0 else None
        dist.gather_object(local_results, gathered, dst=0)
        if rank == 0:
            all_results = [x for lst in gathered for x in lst]
            all_results.sort(key=lambda d: d["global_idx"])
        else:
            all_results = local_results
    else:
        all_results = local_results

    del model
    gc.collect()
    torch.cuda.empty_cache()

    if rank != 0:
        if is_dist:
            dist.barrier()
            dist.destroy_process_group()
        return

    # --- Summary ---
    n = len(all_results)
    n_correct = sum(1 for r in all_results if r["is_correct"])
    acc = n_correct / max(n, 1)
    print("\n" + "=" * 60)
    print(f"SUMMARY  (pid={args.persona_id}, context={args.context_mode})")
    print("=" * 60)
    print(f"Base:        {args.base_model}")
    print(f"LoRA:        {args.lora_path if args.lora_path else '(none)'}")
    print(f"MCQs:        {n}  (correct {n_correct}, acc {acc:.3f}, "
          f"random baseline 0.25)")
    print(f"Wall clock:  {time.time()-t0:.1f}s")

    by_type_correct: dict[str, int] = defaultdict(int)
    by_type_total: dict[str, int] = defaultdict(int)
    for r in all_results:
        by_type_total[r["qtype_canonical"]] += 1
        if r["is_correct"]:
            by_type_correct[r["qtype_canonical"]] += 1
    print("\n--- by question type ---")
    for qt in sorted(by_type_total):
        t = by_type_total[qt]
        c = by_type_correct[qt]
        print(f"  {qt:22s} {c:3d}/{t:3d}  {c/t:.3f}")

    # NLL-gap diagnostic: on wrong MCQs, how much more confident was the
    # model about pred vs correct? Smaller = near miss; larger = far miss.
    print("\n--- NLL margin when wrong (NLL_correct - NLL_pred) ---")
    wrong = [r for r in all_results if not r["is_correct"]]
    if wrong:
        margins = []
        for r in wrong:
            sm = {s["label"]: s["mean_nll"] for s in r["scores"]}
            margins.append(sm[r["correct_answer"]] - sm[r["pred_answer"]])
        margins.sort()
        ng = len(margins)
        print(f"  n_wrong={ng}  p25={margins[ng//4]:.3f}  "
              f"p50={margins[ng//2]:.3f}  p75={margins[3*ng//4]:.3f}  "
              f"max={margins[-1]:.3f}")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps({
        "args": {k: str(v) for k, v in vars(args).items()},
        "accuracy": acc,
        "n_total": n,
        "n_correct": n_correct,
        "results": all_results,
    }, indent=2, ensure_ascii=False))
    print(f"\nwrote {args.out_json}")

    if is_dist:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
