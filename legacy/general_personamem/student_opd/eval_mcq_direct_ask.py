"""Paradigm III eval: directly ask the user-sim which choice matches its
own experiences.

Key design: the assistant turn POSES the MCQ (as a clarifying question
from chatbot back to user), and the user-sim generates its answer in a
USER turn — which is the only role it's been trained on. So:

    <|im_start|>system
    {persona card (demo)}<|im_end|>
    <|im_start|>user
    {original user_question from MCQ}<|im_end|>
    <|im_start|>assistant
    Before I answer, I want to check which of these best reflects what
    you've shared with me before. Just pick a letter.

    A) {choice_a}
    B) {choice_b}
    C) {choice_c}
    D) {choice_d}<|im_end|>
    <|im_start|>user
    ← model generates here; expected format "{A/B/C/D}" or "A because..."

Regex-parse the FIRST ABCD letter out of the response. Compare to
correct_answer.

If the student LoRA still retains Qwen3-Instruct-2507's basic
instruction-following, it should pick a letter; if the LoRA has fully
collapsed to free-form user-voice continuation, it'll say "I also tried
X..." — we'll see in the parse_fail rate.

Output: JSONL (same dir convention as verbal eval) + summary print.

Usage (vLLM, native LoRA):
    python dynamic_usersim/student_opd/eval_mcq_direct_ask.py \\
        --model $QWEN3_BASE \\
        --lora-path $LORA_ROOT/pid4 --lora-mode dual \\
        --persona-id 4 --num-mcqs -1 --context-mode demo-only \\
        --out-jsonl dynamic_usersim/outputs/direct_ask_pid4_r1b.jsonl

    # Base (no LoRA) control:
    python dynamic_usersim/student_opd/eval_mcq_direct_ask.py \\
        --model $QWEN3_BASE \\
        --persona-id 4 --num-mcqs -1 ...
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_TSFT = _HERE.parent / "teacher_sft"
_DP = _HERE.parent / "data_prep"
for p in (_HERE, _TSFT, _DP):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from load_personamem import load_contexts, load_questions, strip_role_prefix  # noqa: E402
from eval_mcq import parse_choices  # noqa: E402
from eval_opd import build_eval_context  # noqa: E402


ASK_TEMPLATE = (
    "Before I answer, I want to check which of these best reflects what "
    "you've shared with me before. Just pick a letter.\n\n"
    "A) {a}\n"
    "B) {b}\n"
    "C) {c}\n"
    "D) {d}"
)


def build_prompt(ctx_msgs: list[dict], question: str,
                 choices: list[tuple[str, str]]) -> str:
    """Compose the full ChatML string ending at '<|im_start|>user\\n'."""
    msgs = [dict(m) for m in ctx_msgs]
    # append the user's original question (merged into trailing user turn
    # if last msg is user, else a new user turn — matches eval_mcq_ppl)
    if msgs and msgs[-1]["role"] == "user":
        msgs[-1]["content"] = msgs[-1]["content"].rstrip() + "\n\n" + question
    else:
        msgs.append({"role": "user", "content": question})
    # assistant turn that POSES the MCQ
    choice_map = {lbl.lower(): txt for lbl, txt in choices}
    msgs.append({"role": "assistant", "content": ASK_TEMPLATE.format(
        a=choice_map.get("a", ""),
        b=choice_map.get("b", ""),
        c=choice_map.get("c", ""),
        d=choice_map.get("d", ""),
    )})
    # serialize
    parts = []
    for m in msgs:
        parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
    parts.append("<|im_start|>user\n")
    return "".join(parts)


LETTER_RE = re.compile(r"\b([ABCDabcd])\b")


def parse_letter(text: str) -> str | None:
    m = LETTER_RE.search(text)
    return m.group(1).lower() if m else None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="HF base model or local path")
    ap.add_argument("--lora-path", default=None,
                    help="single or dual adapter dir; optional")
    ap.add_argument("--lora-mode", choices=["single", "dual"], default="single",
                    help="(for pre-merged dual use --model directly, no --lora-path)")
    ap.add_argument("--max-lora-rank", type=int, default=64)
    ap.add_argument("--persona-id", required=True)
    ap.add_argument("--mcq-version", choices=["32k", "128k", "1M"],
                    default="128k")
    ap.add_argument("--context-mode",
                    choices=["demo-only", "last-n", "full", "recent-turns"],
                    default="demo-only")
    ap.add_argument("--last-n-sessions", type=int, default=3)
    ap.add_argument("--recent-turns", type=int, default=2)
    ap.add_argument("--num-mcqs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="deterministic by default (we want the 'most likely' answer)")
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=40,
                    help="short — we just need a letter plus a few justification words")
    ap.add_argument("--out-jsonl", type=Path, required=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from vllm import LLM, SamplingParams
    except ImportError as e:
        raise SystemExit(f"vLLM missing: {e}")

    # --- Data ---
    mcqs = load_questions(args.mcq_version)
    ctxs = load_contexts(args.mcq_version)
    mcqs = [q for q in mcqs
            if q["shared_context_id"] in ctxs
            and q["persona_id"] == args.persona_id]
    random.Random(args.seed).shuffle(mcqs)
    if args.num_mcqs > 0:
        mcqs = mcqs[: args.num_mcqs]
    print(f"[direct-ask] {len(mcqs)} MCQs for pid={args.persona_id}")

    # --- Build prompts ---
    prompts: list[str] = []
    meta: list[dict] = []
    for q in mcqs:
        ctx_msgs = build_eval_context(
            ctxs[q["shared_context_id"]],
            int(q["end_index_in_shared_context"]),
            args.context_mode, args.last_n_sessions, args.recent_turns,
        )
        choices = parse_choices(q["all_options"])
        prompt = build_prompt(ctx_msgs, q["user_question_or_message"], choices)
        prompts.append(prompt)
        meta.append({
            "qid": q["question_id"],
            "qtype": q["qtype_canonical"],
            "topic": q["topic"],
            "user_question": q["user_question_or_message"],
            "correct": q["correct_answer"].strip("()").lower(),
            "choices": [{"label": lbl, "text": txt} for lbl, txt in choices],
        })

    # --- Load + generate ---
    llm_kwargs = dict(
        model=args.model, dtype="bfloat16",
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=True,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    lora_req = None
    if args.lora_path:
        from vllm.lora.request import LoRARequest
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = args.max_lora_rank
        lora_req = LoRARequest("student", 1, str(args.lora_path))
        print(f"[direct-ask] + LoRA: {args.lora_path}")
    llm = LLM(**llm_kwargs)
    sp = SamplingParams(
        temperature=args.temperature, top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=["<|im_end|>", "<|im_start|>"],
        seed=args.seed,
    )
    t0 = time.time()
    gen_kwargs = {"lora_request": lora_req} if lora_req else {}
    outs = llm.generate(prompts, sp, **gen_kwargs)
    print(f"[direct-ask] generated in {time.time()-t0:.1f}s")

    # --- Score + dump ---
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    n_correct = n_parse_fail = 0
    by_type_correct: dict[str, int] = defaultdict(int)
    by_type_total: dict[str, int] = defaultdict(int)
    by_type_parse_fail: dict[str, int] = defaultdict(int)
    with args.out_jsonl.open("w", encoding="utf-8") as f:
        for out, m in zip(outs, meta):
            resp = out.outputs[0].text.strip()
            pred = parse_letter(resp)
            is_correct = (pred == m["correct"])
            if pred is None:
                n_parse_fail += 1
                by_type_parse_fail[m["qtype"]] += 1
            elif is_correct:
                n_correct += 1
                by_type_correct[m["qtype"]] += 1
            by_type_total[m["qtype"]] += 1
            f.write(json.dumps({
                **m,
                "response": resp,
                "pred": pred,
                "is_correct": is_correct,
            }, ensure_ascii=False) + "\n")

    n = len(meta)
    print("\n" + "=" * 64)
    print(f"direct-ask MCQ accuracy (pid={args.persona_id})  n={n}")
    print("=" * 64)
    print(f"  correct:     {n_correct}  ({n_correct/n:.3f})")
    print(f"  parse fail:  {n_parse_fail} ({n_parse_fail/n:.3f})  "
          f"← high = model ignored the 'pick a letter' instruction")
    print(f"  random:      {0.25:.3f}")
    print()
    print(f"  {'qtype':<22} correct  parse_fail  total  acc")
    for qt in sorted(by_type_total):
        t = by_type_total[qt]
        c = by_type_correct[qt]
        pf = by_type_parse_fail[qt]
        print(f"  {qt:<22} {c:>7}  {pf:>10}  {t:>5}  {c/t:.3f}")
    print(f"\n[direct-ask] wrote {args.out_jsonl}")


if __name__ == "__main__":
    main()
