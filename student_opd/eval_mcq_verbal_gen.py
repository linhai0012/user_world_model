"""Verbal feedback — Stage 1 (pure generation, vLLM-batched).

Feeds the UserSim (R1b-merged model, or base) with:
    [ctx (demo-only default)] + user(question) + assistant(choice) + <|im_start|>user\\n
→ model generates the next USER turn naturally. NO primer / no judgment ask;
this is exactly the SFT training distribution.

For each MCQ × 4 choices, save the generated reaction alongside:
  - choice text (a/b/c/d)
  - correct label
  - qtype, topic, question_id
  - a small window of the ORIGINAL conversation following end_index (for
    reviewer eyeballing: "what did the real user actually say next")

Output: JSONL, one record per MCQ (4 reactions nested). Compact enough to
commit to git for review (~500KB for 147 MCQs × 4 × ~150 tokens).

Two-stage rationale: Stage 1 generation is expensive (vLLM batched, ~2-5
min for 588 prompts on B200); Stage 2 scoring is cheap text processing and
can iterate on multiple scoring methods without regenerating.

Usage (after setup_kcl.sh + merge_dual_lora.py):
    python dynamic_usersim/student_opd/eval_mcq_verbal_gen.py \\
        --model $SCRATCHDIR/P-OPSD/merged_r1b/pid4 \\
        --persona-id 4 --num-mcqs -1 --context-mode demo-only \\
        --out-jsonl $PROBE_OUT/verbal_pid4_r1b.jsonl

    # Compare with base:
    python dynamic_usersim/student_opd/eval_mcq_verbal_gen.py \\
        --model $QWEN3_BASE \\
        --persona-id 4 --num-mcqs -1 \\
        --out-jsonl $PROBE_OUT/verbal_pid4_base.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_TSFT = _HERE.parent / "teacher_sft"
_DP = _HERE.parent / "data_prep"
for p in (_HERE, _TSFT, _DP):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from load_personamem import (  # noqa: E402
    load_contexts, load_questions, strip_role_prefix,
)
from eval_mcq import parse_choices  # noqa: E402
from eval_opd import build_eval_context  # noqa: E402


# ---------------------------------------------------------------------------
# Prompt construction (plain ChatML, matches SFT training format)
# ---------------------------------------------------------------------------
def build_verbal_prompt(ctx_msgs: list[dict], question: str,
                        choice_text: str) -> str:
    """Format:
        <|im_start|>system\\n{ctx system}<|im_end|>\\n
        ...prior turns...
        <|im_start|>user\\n{last user + question}<|im_end|>\\n
        <|im_start|>assistant\\n{choice_text}<|im_end|>\\n
        <|im_start|>user\\n       ← model continues from here
    """
    msgs = [dict(m) for m in ctx_msgs]
    if msgs and msgs[-1]["role"] == "user":
        msgs[-1]["content"] = msgs[-1]["content"].rstrip() + "\n\n" + question
    else:
        msgs.append({"role": "user", "content": question})
    msgs.append({"role": "assistant", "content": choice_text})

    parts = []
    for m in msgs:
        parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
    parts.append("<|im_start|>user\n")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Extract a small raw-context window around end_index for reviewer context
# ---------------------------------------------------------------------------
def extract_gt_window(raw_ctx: list[dict], end_index: int,
                      max_content_chars: int = 400) -> list[dict]:
    """Up to 3 messages *after* end_index — typically the real assistant
    reply + the real user's followup turn (not MCQ-polished).
    Content is truncated to keep JSONL small."""
    window = raw_ctx[end_index + 1: end_index + 4]
    out = []
    for m in window:
        content = m.get("content", "")
        content = strip_role_prefix(content, m.get("role", ""))
        if len(content) > max_content_chars:
            content = content[:max_content_chars].rstrip() + "…"
        out.append({"role": m.get("role", ""), "content": content})
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True,
                    help="HF model dir (base or pre-merged R1b); loaded by vLLM")
    ap.add_argument("--persona-id", type=str, required=True)
    ap.add_argument("--mcq-version", choices=["32k", "128k", "1M"],
                    default="128k")
    ap.add_argument("--context-mode",
                    choices=["demo-only", "last-n", "full", "recent-turns"],
                    default="demo-only")
    ap.add_argument("--last-n-sessions", type=int, default=3)
    ap.add_argument("--recent-turns", type=int, default=2)
    ap.add_argument("--num-mcqs", type=int, default=-1,
                    help="-1 = all MCQs for this persona")
    ap.add_argument("--seed", type=int, default=42)
    # vLLM / sampling
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=150)
    ap.add_argument("--n-samples", type=int, default=1,
                    help="vLLM n= per prompt (multiple reactions per choice)")
    ap.add_argument("--out-jsonl", type=Path, required=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    # Import vLLM lazily so import errors are surfaced with a clear message
    try:
        from vllm import LLM, SamplingParams
    except ImportError as e:
        raise SystemExit(
            "vLLM not installed in this env. `pip install vllm` or activate "
            "an env that has it. Original error: " + str(e)
        )

    # --- Load MCQs + contexts ---
    print(f"[gen] loading MCQs ({args.mcq_version}) for pid={args.persona_id}")
    mcqs = load_questions(args.mcq_version)
    ctxs = load_contexts(args.mcq_version)
    mcqs = [q for q in mcqs
            if q["shared_context_id"] in ctxs
            and q["persona_id"] == args.persona_id]
    random.Random(args.seed).shuffle(mcqs)
    if args.num_mcqs > 0:
        mcqs = mcqs[: args.num_mcqs]
    print(f"[gen] {len(mcqs)} MCQs")

    # --- Build all prompts (MCQ × 4 choices) ---
    prompts: list[str] = []
    prompt_meta: list[dict] = []
    for q in mcqs:
        ctx_msgs = build_eval_context(
            ctxs[q["shared_context_id"]],
            int(q["end_index_in_shared_context"]),
            args.context_mode, args.last_n_sessions, args.recent_turns,
        )
        question = q["user_question_or_message"]
        choices = parse_choices(q["all_options"])
        for label, choice_text in choices:
            prompts.append(build_verbal_prompt(ctx_msgs, question, choice_text))
            prompt_meta.append({"qid": q["question_id"],
                                "label": label,
                                "choice_text": choice_text})
    print(f"[gen] {len(prompts)} prompts to generate "
          f"(= {len(mcqs)} MCQs × 4 choices)")

    # --- vLLM load + generate ---
    print(f"[gen] loading model into vLLM: {args.model}")
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=True,
    )
    sp = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        n=args.n_samples,
        stop=["<|im_end|>", "<|im_start|>"],
        seed=args.seed,
    )

    t0 = time.time()
    outs = llm.generate(prompts, sp)
    dt = time.time() - t0
    print(f"[gen] generated in {dt:.1f}s "
          f"({len(prompts)/dt:.1f} prompts/s)")

    # --- Collate: per MCQ, 4 reactions ---
    by_qid: dict[str, dict] = {}
    for out, meta in zip(outs, prompt_meta):
        qid = meta["qid"]
        entry = by_qid.setdefault(qid, {"reactions": {}})
        reactions = [o.text.strip() for o in out.outputs]
        entry["reactions"][meta["label"]] = {
            "choice_text": meta["choice_text"],
            # list of n_samples strings; first is primary
            "reactions": reactions,
        }

    # --- Dump JSONL with per-MCQ metadata + GT followup window ---
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as f:
        for q in mcqs:
            qid = q["question_id"]
            if qid not in by_qid:
                continue
            end_idx = int(q["end_index_in_shared_context"])
            raw_ctx = ctxs[q["shared_context_id"]]
            correct_label = q["correct_answer"].strip("()").lower()

            rec = {
                "question_id": qid,
                "persona_id": q["persona_id"],
                "qtype_canonical": q["qtype_canonical"],
                "topic": q["topic"],
                "user_question": q["user_question_or_message"],
                "correct_answer": correct_label,
                "choices": [
                    {
                        "label": L,
                        "is_correct": L == correct_label,
                        "choice_text": by_qid[qid]["reactions"][L]["choice_text"],
                        "reactions": by_qid[qid]["reactions"][L]["reactions"],
                    }
                    for L in ["a", "b", "c", "d"]
                    if L in by_qid[qid]["reactions"]
                ],
                # Original conversation window for reviewer eyeballing.
                # gt_followup[0] is typically the REAL assistant reply (which
                # the MCQ "correct choice" is polished from); gt_followup[1]
                # is the REAL user followup — a natural baseline for what
                # our generated reactions should look like.
                "gt_followup": extract_gt_window(raw_ctx, end_idx),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[gen] wrote {args.out_jsonl} "
          f"({len(by_qid)} MCQs, {args.n_samples} samples/choice)")


if __name__ == "__main__":
    main()
