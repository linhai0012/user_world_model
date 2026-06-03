"""MCQ evaluation via choice perplexity (plan §9.5, no agent needed).

For each MCQ: score the FOUR choices by their conditional per-token NLL
under the UserSim model, given the full conversation context plus the
MCQ's user_question_or_message. Predicted choice = argmin(mean NLL).

Why this is better than the generate+agent pipeline (eval_mcq.py):
- Deterministic (no sampling noise); same model + MCQ always gives same
  score.
- Bypasses the "SFT monologues 'I also...' regardless of choice" failure
  mode observed in eval_mcq v1/v2: we never generate user reactions.
- No OpenAI cost / rate limit / agent intermediary.
- Much faster (one short forward per choice; no 200-token autoregression).

The scored quantity is:
    P(choice_text | full_context + user_question_merged, "<|im_start|>assistant\\n")
i.e. how likely the UserSim thinks this particular assistant response
would be — teacher SFT that has absorbed the persona should prefer
responses aligned with the user's real preferences.

4-GPU DP via torchrun; same pattern as other eval scripts.

Launch:
    torchrun --standalone --nnodes=1 --nproc_per_node=4 \
        dynamic_usersim/teacher_sft/eval_mcq_ppl.py \
        --usersim-model $SCRATCHDIR/P-OPSD/teacher_sft_128k/checkpoint-50 \
        --mcq-version 32k --num-mcqs 50 \
        --out-json dynamic_usersim/outputs/mcqppl_sft_ckpt50.json
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
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

_HERE = Path(__file__).resolve().parent
_DP = _HERE.parent / "data_prep"
for p in (_HERE, _DP):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from config import MAX_SEQ_LEN, MODEL_NAME  # noqa: E402
from load_personamem import (  # noqa: E402
    load_contexts, load_questions, strip_role_prefix,
)
from tokenize_teacher_sft import SFTTokenizer  # noqa: E402
from eval_mcq import parse_choices, build_context_messages  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--usersim-model", type=str, required=True,
                    help="HF name (e.g. Qwen/Qwen3-4B) or local checkpoint dir")
    ap.add_argument("--mcq-version", choices=["32k", "128k", "1M"], default="32k")
    ap.add_argument("--num-mcqs", type=int, default=50)
    ap.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    ap.add_argument("--last-n-sessions", type=int, default=None,
                    help="Cap context to most recent N sessions (match SFT K).")
    ap.add_argument("--lm-head-chunk", type=int, default=4096,
                    help="Positions through lm_head per chunk (memory cap).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-json", type=Path, required=True)
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Boilerplate: liger / dist / model load (mirrors eval_mcq.py)
# ---------------------------------------------------------------------------
def apply_liger() -> None:
    try:
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3
        apply_liger_kernel_to_qwen3(
            rope=True, swiglu=True, cross_entropy=False,
            fused_linear_cross_entropy=True, rms_norm=True,
        )
        if int(os.environ.get("RANK", 0)) == 0:
            print("[eval] liger applied")
    except ImportError:
        if int(os.environ.get("RANK", 0)) == 0:
            print("[eval] liger not available; continuing")


def init_dist() -> tuple[int, int, bool]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        ws = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        return rank, ws, True
    torch.cuda.set_device(0)
    return 0, 1, False


def load_model(path: str | Path) -> torch.nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        str(path),
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.eval()
    model.to(torch.cuda.current_device())
    return model


# ---------------------------------------------------------------------------
# Core scoring
# ---------------------------------------------------------------------------
def _encode_msgs_plain(sft_tok: SFTTokenizer, msgs: list[dict]) -> list[int]:
    """Match training format exactly (no chat-template think-mode markers)."""
    ids: list[int] = []
    for m in msgs:
        chunk = sft_tok._encode_message(m["role"], m["content"], False)
        ids.extend(chunk.input_ids)
    return ids


def score_choice_ppl(model: torch.nn.Module, sft_tok: SFTTokenizer,
                     context_msgs: list[dict], question: str, choice_text: str,
                     max_len: int, lm_head_chunk: int) -> tuple[float, int]:
    """Return (mean_nll, n_target_tokens) for one (context, choice).

    Scored tokens: the choice text PLUS its closing <|im_end|>\\n.
    (Including the end marker rewards natural-length choices that
    terminate cleanly, matching training.)

    Context truncation: if total exceeds max_len, drop OLDEST context
    messages entirely until it fits. The choice is always preserved.
    """
    msgs = [dict(m) for m in context_msgs]
    if msgs and msgs[-1]["role"] == "user":
        msgs[-1]["content"] = msgs[-1]["content"].rstrip() + "\n\n" + question
    else:
        msgs.append({"role": "user", "content": question})

    asst_prefix_ids = sft_tok.tok.encode("<|im_start|>assistant\n",
                                          add_special_tokens=False)
    choice_content_ids = sft_tok.tok.encode(choice_text,
                                              add_special_tokens=False)
    end_ids = sft_tok.tok.encode("<|im_end|>\n", add_special_tokens=False)
    target_ids = choice_content_ids + end_ids
    n_target = len(target_ids)

    # Message-level greedy truncation from oldest
    while True:
        context_ids = _encode_msgs_plain(sft_tok, msgs)
        total_len = len(context_ids) + len(asst_prefix_ids) + n_target
        if total_len <= max_len:
            break
        if len(msgs) <= 2:
            # emergency: hard-truncate context from the left
            budget = max_len - len(asst_prefix_ids) - n_target
            context_ids = context_ids[-budget:]
            break
        msgs.pop(0)

    full_ids = context_ids + asst_prefix_ids + target_ids
    device = torch.cuda.current_device()
    ids_t = torch.tensor([full_ids], dtype=torch.long, device=device)

    # Target positions in full_ids: the last n_target tokens.
    # Hidden at position i predicts token i+1, so hidden positions for
    # predicting target are [len(full_ids)-n_target-1, len(full_ids)-2].
    pred_start = len(full_ids) - n_target - 1
    pred_end = len(full_ids) - 1

    with torch.no_grad():
        hidden = model.model(input_ids=ids_t,
                             use_cache=False).last_hidden_state[0]
        h_pred = hidden[pred_start:pred_end]               # [n_target, H]
        labels = torch.tensor(target_ids, device=device)   # [n_target]

        total_nll = 0.0
        for i in range(0, n_target, lm_head_chunk):
            h = h_pred[i:i + lm_head_chunk]
            t = labels[i:i + lm_head_chunk]
            logits = model.lm_head(h).float()              # [c, V]
            total_nll += F.cross_entropy(logits, t, reduction="sum").item()

    return total_nll / max(n_target, 1), n_target


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    assert torch.cuda.is_available()
    rank, world_size, is_dist = init_dist()

    if rank == 0:
        print(f"[eval] loading MCQs + contexts ({args.mcq_version})")
    mcqs = load_questions(args.mcq_version)
    ctxs = load_contexts(args.mcq_version)
    rng = random.Random(args.seed)
    mcqs = [q for q in mcqs if q["shared_context_id"] in ctxs]
    n_take = len(mcqs) if args.num_mcqs <= 0 else min(args.num_mcqs, len(mcqs))
    sample = rng.sample(mcqs, n_take)
    if rank == 0:
        print(f"[eval] sampled {len(sample)} of {len(mcqs)} eligible MCQs "
              f"(seed={args.seed}, world_size={world_size})")

    sft_tok = SFTTokenizer(model_name=MODEL_NAME, max_len=args.max_seq_len)
    apply_liger()
    if rank == 0:
        print(f"[eval] loading UserSim from {args.usersim_model}")
    model = load_model(args.usersim_model)

    my_mcqs = sample[rank::world_size]
    local_results: list[dict] = []
    t0 = time.time()
    for i, q in enumerate(my_mcqs):
        t_mcq = time.time()
        ctx_msgs = build_context_messages(
            ctxs[q["shared_context_id"]],
            int(q["end_index_in_shared_context"]),
            args.last_n_sessions,
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
        # Pick lowest mean NLL
        best = min(scores, key=lambda x: x["mean_nll"])
        pred_letter = best["label"]
        correct = q["correct_answer"].strip("()").lower()
        is_correct = (pred_letter == correct)

        local_results.append({
            "global_idx": sample.index(q),
            "persona_id": q["persona_id"],
            "question_id": q["question_id"],
            "qtype_canonical": q["qtype_canonical"],
            "topic": q["topic"],
            "question": question,
            "correct_answer": correct,
            "pred_answer": pred_letter,
            "is_correct": is_correct,
            "scores": scores,
            "wall_s": time.time() - t_mcq,
        })
        # Delta between best and correct — diagnostic
        score_map = {s["label"]: s["mean_nll"] for s in scores}
        gt_nll = score_map.get(correct, float("nan"))
        gap = best["mean_nll"] - gt_nll
        print(f"  [rank{rank}] {i+1}/{len(my_mcqs)} "
              f"pid={q['persona_id']} qtype={q['qtype_canonical']} "
              f"pred={pred_letter} correct={correct} "
              f"{'OK' if is_correct else 'MISS'} "
              f"gt_nll={gt_nll:.3f} best_nll={best['mean_nll']:.3f} "
              f"gap={gap:+.3f} ({time.time()-t_mcq:.1f}s)", flush=True)

    # Gather
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

    n = len(all_results)
    n_correct = sum(1 for r in all_results if r["is_correct"])
    acc = n_correct / max(n, 1)
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"UserSim:     {args.usersim_model}")
    print(f"Metric:      per-token NLL of choice under UserSim (lowest wins)")
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

    # Diagnostic: average NLL gap (best - correct) — how close the model
    # is when it misses. Small positive gap = near miss; large = far miss.
    print("\n--- NLL gap distribution (best - correct, only when wrong) ---")
    wrong = [r for r in all_results if not r["is_correct"]]
    if wrong:
        gaps = []
        for r in wrong:
            m = {s["label"]: s["mean_nll"] for s in r["scores"]}
            gaps.append(m[r["pred_answer"]] - m[r["correct_answer"]])
        gaps.sort()
        ng = len(gaps)
        # gaps are all <= 0 (we picked min), so negative of gap is how much more likely wrong was
        margins = [-g for g in gaps]
        margins.sort()
        print(f"  wrong answers: {ng}")
        print(f"  margin (NLL_correct - NLL_pred) p25={margins[ng//4]:.3f} "
              f"p50={margins[ng//2]:.3f} p75={margins[3*ng//4]:.3f} "
              f"max={margins[-1]:.3f}")
        print(f"  (smaller margin = closer call; larger = confident but wrong)")

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
