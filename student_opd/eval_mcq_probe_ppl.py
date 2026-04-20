"""Probe-PPL MCQ eval (paradigm II redesign — new direction).

Diagnosis behind this eval: the existing MCQ-PPL (§9.5, eval_mcq_ppl.py /
eval_opd.py) scores ASSISTANT tokens, but our SFT objective only trains USER
tokens. Subject-mismatched. This script flips it: the choice becomes part of
the CONDITIONING, and we score two canonical USER-voice probes — one that
affirms, one that corrects — and pick the choice that maximises
   margin(choice) = nll_correct - nll_affirm                       (↑ better)
i.e. the choice that most makes the user want to affirm rather than correct.

Test plan for this first sniff:
  (1) margin-distribution: is margin(correct_choice) > margin(wrong_choice)
      on average? — validates that the signal exists.
  (2) accuracy: argmax-margin decision — does it beat random (0.25) and
      ideally approach / beat MCQ-PPL (§11 R1b benchmark).

Launch (single-GPU, fastest to iterate):
    CUDA_VISIBLE_DEVICES=0 python \
        dynamic_usersim/student_opd/eval_mcq_probe_ppl.py \
        --persona-id 4 --num-mcqs 30 --context-mode demo-only \
        --out-json dynamic_usersim/outputs/probe_pid4_base_sniff.json

4-GPU DP (match eval_opd):
    torchrun --standalone --nnodes=1 --nproc_per_node=4 \
        dynamic_usersim/student_opd/eval_mcq_probe_ppl.py \
        --persona-id 4 --num-mcqs 30 ...

With LoRA (later, once R1b adapters are on KCL):
    ... --lora-path $SCRATCHDIR/P-OPSD/student_lora_r1b/pid4 --lora-mode dual
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_TSFT = _HERE.parent / "teacher_sft"
_DP = _HERE.parent / "data_prep"
for p in (_HERE, _TSFT, _DP):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from config import MAX_SEQ_LEN, MODEL_NAME  # noqa: E402
from load_personamem import load_contexts, load_questions  # noqa: E402
from tokenize_teacher_sft import SFTTokenizer  # noqa: E402
from eval_mcq import parse_choices  # noqa: E402
from eval_mcq_ppl import apply_liger, init_dist, _encode_msgs_plain  # noqa: E402
from eval_opd import build_eval_context, load_model_with_lora  # noqa: E402


# ---------------------------------------------------------------------------
# Probes (v1 — fixed generic; length-symmetric for clean comparison)
# ---------------------------------------------------------------------------
PROBES = {
    "affirm":  "Yes, that's exactly what I had in mind.",
    "correct": "Hmm, that's not quite right.",
}


# ---------------------------------------------------------------------------
# Probe scoring core
# ---------------------------------------------------------------------------
def score_probe_ppl(
    model: torch.nn.Module,
    sft_tok: SFTTokenizer,
    context_msgs: list[dict],
    question: str,
    choice_text: str,
    probe_text: str,
    max_len: int,
    lm_head_chunk: int,
) -> tuple[float, int]:
    """Return (mean_nll, n_probe_tokens).

    Input build:
        [ctx_msgs] + user(question)
        + <|im_start|>assistant\\n + choice_text + <|im_end|>\\n
        + <|im_start|>user\\n + probe_text
        ----- scored = probe_text only -----

    Scored tokens are the probe's content tokens ONLY (no closing im_end) —
    we just want P(this user-opener | ctx + choice), not the full turn.

    Message-level greedy truncation from oldest if total > max_len; the
    assistant/user role markers + choice + probe are always preserved.
    """
    msgs = [dict(m) for m in context_msgs]
    if msgs and msgs[-1]["role"] == "user":
        msgs[-1]["content"] = msgs[-1]["content"].rstrip() + "\n\n" + question
    else:
        msgs.append({"role": "user", "content": question})

    tok = sft_tok.tok
    asst_open_ids = tok.encode("<|im_start|>assistant\n", add_special_tokens=False)
    choice_ids = tok.encode(choice_text, add_special_tokens=False)
    asst_close_ids = tok.encode("<|im_end|>\n", add_special_tokens=False)
    user_open_ids = tok.encode("<|im_start|>user\n", add_special_tokens=False)
    probe_ids = tok.encode(probe_text, add_special_tokens=False)

    n_probe = len(probe_ids)
    tail_len = (len(asst_open_ids) + len(choice_ids) + len(asst_close_ids)
                + len(user_open_ids) + n_probe)

    # Greedy drop oldest ctx msgs until fits
    while True:
        context_ids = _encode_msgs_plain(sft_tok, msgs)
        if len(context_ids) + tail_len <= max_len:
            break
        if len(msgs) <= 2:
            budget = max_len - tail_len
            if budget < 0:
                budget = 0
            context_ids = context_ids[-budget:]
            break
        msgs.pop(0)

    full_ids = (
        context_ids
        + asst_open_ids + choice_ids + asst_close_ids
        + user_open_ids + probe_ids
    )
    device = torch.cuda.current_device()
    ids_t = torch.tensor([full_ids], dtype=torch.long, device=device)

    # Probe positions = last n_probe tokens.
    # Hidden at pos i predicts token i+1 → score hidden[N-n-1 : N-1]
    pred_start = len(full_ids) - n_probe - 1
    pred_end = len(full_ids) - 1

    with torch.no_grad():
        hidden = model.model(input_ids=ids_t,
                             use_cache=False).last_hidden_state[0]
        h_pred = hidden[pred_start:pred_end]                 # [n_probe, H]
        labels = torch.tensor(probe_ids, device=device)      # [n_probe]

        total_nll = 0.0
        for i in range(0, n_probe, lm_head_chunk):
            h = h_pred[i:i + lm_head_chunk]
            t = labels[i:i + lm_head_chunk]
            logits = model.lm_head(h).float()                # [c, V]
            total_nll += F.cross_entropy(logits, t, reduction="sum").item()

    return total_nll / max(n_probe, 1), n_probe


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", type=str, default=MODEL_NAME,
                    help=f"HF base (default {MODEL_NAME})")
    ap.add_argument("--lora-path", type=Path, default=None,
                    help="PEFT adapter dir (single) or parent with slow/+fast/ (dual)")
    ap.add_argument("--lora-mode", choices=["single", "dual"], default="single")
    ap.add_argument("--persona-id", type=str, required=True)
    ap.add_argument("--mcq-version", choices=["32k", "128k", "1M"], default="128k")
    ap.add_argument("--context-mode",
                    choices=["demo-only", "last-n", "full", "recent-turns"],
                    default="demo-only")
    ap.add_argument("--last-n-sessions", type=int, default=3)
    ap.add_argument("--recent-turns", type=int, default=2)
    ap.add_argument("--num-mcqs", type=int, default=30,
                    help="-1 = all MCQs for this persona (default 30 for sniff)")
    ap.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    ap.add_argument("--lm-head-chunk", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--probe-affirm", type=str, default=PROBES["affirm"])
    ap.add_argument("--probe-correct", type=str, default=PROBES["correct"])
    ap.add_argument("--out-json", type=Path, required=True)
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    assert torch.cuda.is_available()
    rank, world_size, is_dist = init_dist()

    if rank == 0:
        print(f"[probe] loading MCQs ({args.mcq_version}) for pid={args.persona_id}")
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
        print(f"[probe] evaluating {len(mcqs)} MCQs "
              f"(context={args.context_mode}, world_size={world_size})")
        print(f"[probe] affirm : {args.probe_affirm!r}")
        print(f"[probe] correct: {args.probe_correct!r}")

    sft_tok = SFTTokenizer(model_name=args.base_model, max_len=args.max_seq_len)
    apply_liger()
    if rank == 0:
        print(f"[probe] loading base: {args.base_model}"
              f"{' + LoRA' if args.lora_path else ' (no LoRA)'}")
    model = load_model_with_lora(
        args.base_model, args.lora_path, args.lora_mode,
    )

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
        correct = q["correct_answer"].strip("()").lower()

        per_choice: list[dict] = []
        for label, choice_text in choices:
            nll_a, n_a = score_probe_ppl(
                model, sft_tok, ctx_msgs, question, choice_text,
                args.probe_affirm, args.max_seq_len, args.lm_head_chunk,
            )
            nll_c, n_c = score_probe_ppl(
                model, sft_tok, ctx_msgs, question, choice_text,
                args.probe_correct, args.max_seq_len, args.lm_head_chunk,
            )
            margin = nll_c - nll_a        # higher = more affirm-preferring
            per_choice.append({
                "label": label,
                "is_gold": label == correct,
                "nll_affirm": nll_a,
                "nll_correct": nll_c,
                "n_affirm_tokens": n_a,
                "n_correct_tokens": n_c,
                "margin": margin,
            })

        best = max(per_choice, key=lambda x: x["margin"])
        pred = best["label"]
        is_correct = (pred == correct)

        gold_margin = next(c["margin"] for c in per_choice if c["is_gold"])
        wrong_margins = [c["margin"] for c in per_choice if not c["is_gold"]]
        margin_gap = gold_margin - max(wrong_margins)   # >0 iff correct

        local_results.append({
            "global_idx": mcqs.index(q),
            "persona_id": q["persona_id"],
            "question_id": q["question_id"],
            "qtype_canonical": q["qtype_canonical"],
            "topic": q["topic"],
            "correct_answer": correct,
            "pred_answer": pred,
            "is_correct": is_correct,
            "gold_margin": gold_margin,
            "max_wrong_margin": max(wrong_margins),
            "margin_gap": margin_gap,
            "scores": per_choice,
            "wall_s": time.time() - t_mcq,
        })
        print(f"  [rank{rank}] {i+1}/{len(my_mcqs)} "
              f"qtype={q['qtype_canonical']} "
              f"pred={pred} gt={correct} "
              f"{'OK' if is_correct else 'MISS'} "
              f"gold_m={gold_margin:+.3f} max_wrong_m={max(wrong_margins):+.3f} "
              f"gap={margin_gap:+.3f} ({time.time()-t_mcq:.1f}s)",
              flush=True)

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

    # Distribution of margins: gold-choice vs wrong-choice
    gold_margins = [r["gold_margin"] for r in all_results]
    wrong_margins_all: list[float] = []
    for r in all_results:
        for c in r["scores"]:
            if not c["is_gold"]:
                wrong_margins_all.append(c["margin"])

    def _stats(xs):
        if not xs:
            return {"n": 0}
        xs = sorted(xs)
        return {
            "n": len(xs),
            "mean": statistics.mean(xs),
            "std": statistics.pstdev(xs),
            "p25": xs[len(xs)//4],
            "p50": xs[len(xs)//2],
            "p75": xs[3*len(xs)//4],
            "min": xs[0], "max": xs[-1],
        }

    print("\n" + "=" * 66)
    print("TEST (1) — margin distribution: gold choice vs wrong choice")
    print("=" * 66)
    gold_s = _stats(gold_margins)
    wrong_s = _stats(wrong_margins_all)
    print(f"  gold-choice margins  n={gold_s['n']:4d}  "
          f"mean={gold_s['mean']:+.4f}  std={gold_s['std']:.4f}  "
          f"p25/p50/p75 = {gold_s['p25']:+.3f} / {gold_s['p50']:+.3f} / {gold_s['p75']:+.3f}")
    print(f"  wrong-choice margins n={wrong_s['n']:4d}  "
          f"mean={wrong_s['mean']:+.4f}  std={wrong_s['std']:.4f}  "
          f"p25/p50/p75 = {wrong_s['p25']:+.3f} / {wrong_s['p50']:+.3f} / {wrong_s['p75']:+.3f}")
    sep = gold_s["mean"] - wrong_s["mean"]
    # rough pooled d
    pooled = ((gold_s["std"] ** 2 + wrong_s["std"] ** 2) / 2) ** 0.5
    d = sep / pooled if pooled > 0 else float("nan")
    print(f"  Δmean (gold - wrong) = {sep:+.4f}   Cohen-d ≈ {d:+.3f}  "
          f"(>0.2 small / >0.5 medium / >0.8 large)")

    print("\n" + "=" * 66)
    print("TEST (2) — accuracy")
    print("=" * 66)
    print(f"  MCQs:     {n}  correct {n_correct}  acc {acc:.3f}  "
          f"(random = 0.25)")

    by_type_correct: dict[str, int] = defaultdict(int)
    by_type_total: dict[str, int] = defaultdict(int)
    for r in all_results:
        by_type_total[r["qtype_canonical"]] += 1
        if r["is_correct"]:
            by_type_correct[r["qtype_canonical"]] += 1
    print("\n  by qtype:")
    for qt in sorted(by_type_total):
        t = by_type_total[qt]
        c = by_type_correct[qt]
        print(f"    {qt:22s} {c:3d}/{t:3d}  {c/t:.3f}")

    # --- Dump ---
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "base_model": args.base_model,
            "lora_path": str(args.lora_path) if args.lora_path else None,
            "lora_mode": args.lora_mode,
            "persona_id": args.persona_id,
            "mcq_version": args.mcq_version,
            "context_mode": args.context_mode,
            "last_n_sessions": args.last_n_sessions,
            "recent_turns": args.recent_turns,
            "num_mcqs": args.num_mcqs,
            "seed": args.seed,
            "probe_affirm": args.probe_affirm,
            "probe_correct": args.probe_correct,
        },
        "summary": {
            "n": n,
            "n_correct": n_correct,
            "acc": acc,
            "wall_s": time.time() - t0,
            "margin_gold": gold_s,
            "margin_wrong": wrong_s,
            "margin_sep": sep,
            "cohen_d": d,
            "by_qtype": {
                qt: {"correct": by_type_correct[qt],
                     "total": by_type_total[qt],
                     "acc": by_type_correct[qt] / by_type_total[qt]}
                for qt in by_type_total
            },
        },
        "results": all_results,
    }
    args.out_json.write_text(json.dumps(payload, indent=2))
    print(f"\n[probe] wrote {args.out_json}")

    if is_dist:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
