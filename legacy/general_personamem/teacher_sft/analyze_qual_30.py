"""Structured analysis of 30-case qualitative eval.

Consumes the JSON produced by eval_qualitative.py or
eval_qualitative_vllm.py (same schema). For each case, scores:
  - <think> leakage (base-only artifact)
  - Generic "I also..." opener (SFT monologue mode)
  - Content-word Jaccard overlap with ground-truth user turn
  - "Reaction word" presence in first 100 chars (yes/no/hmm/actually/...)

Aggregates: per-condition rates, argmax-overlap winner counts, top/bottom
sft_ctx examples, and the sft_ctx − sft_noctx context-effect distribution.

Usage:
    python dynamic_usersim/teacher_sft/analyze_qual_30.py \
        --in-json dynamic_usersim/outputs/eval3_r3_final.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


STOP = set(
    "a an the of to and or but for in on at with by as is are was were be been "
    "being have has had do does did can could would should will shall may might "
    "this that these those i you he she it we they me my your his her our their "
    "not no so if than then also too very really just so yes".split()
)

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z']+")

REACT_WORDS = {
    "yes", "no", "yeah", "nope", "not quite", "not really", "hmm",
    "actually", "agree", "disagree", "absolutely", "definitely",
    "exactly", "that sounds", "sounds good", "sounds great",
    "that's true", "that is true", "correct", "right", "wrong",
    "you're right",
}

ALSO_OPEN = re.compile(r"^\s*i (also|have also|'ve also|'m also)\b",
                       re.IGNORECASE)

CONDITIONS = ["base_noctx", "base_ctx", "sft_noctx", "sft_ctx"]


def content_tokens(s: str) -> set[str]:
    return {t.lower() for t in TOKEN_RE.findall(s)
            if t.lower() not in STOP and len(t) > 2}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(len(a | b), 1)


def has_react(s: str) -> bool:
    head = s[:100].lower()
    return any(w in head for w in REACT_WORDS)


def also_opener(s: str) -> bool:
    return bool(ALSO_OPEN.match(s))


def has_think(s: str) -> bool:
    return "<think>" in s or "</think>" in s


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-json", type=Path, required=True,
                    help="Output JSON from eval_qualitative{,_vllm}.py")
    ap.add_argument("--top-k", type=int, default=5,
                    help="How many top-overlap sft_ctx cases to print")
    ap.add_argument("--bottom-k", type=int, default=3,
                    help="How many worst-overlap sft_ctx cases to print")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    d = json.loads(args.in_json.read_text(encoding="utf-8"))
    cases = d["cases"]
    N = len(cases)
    if N == 0:
        print("no cases in input")
        return

    # Per-case analysis
    agg = {c: {"think": 0, "also": 0, "react": 0, "overlap": [], "len": []}
           for c in CONDITIONS}
    wins_overlap: Counter = Counter()
    for c in cases:
        gt_tok = content_tokens(c["ground_truth"])
        per = {}
        for cond in CONDITIONS:
            out = c["outputs"].get(cond, "")
            ov = jaccard(content_tokens(out), gt_tok)
            per[cond] = ov
            agg[cond]["overlap"].append(ov)
            agg[cond]["len"].append(len(out))
            if has_think(out):
                agg[cond]["think"] += 1
            if also_opener(out):
                agg[cond]["also"] += 1
            if has_react(out):
                agg[cond]["react"] += 1
        best = max(per, key=per.get)
        wins_overlap[best] += 1

    print(f"=== {N}-case analysis ({args.in_json.name}) ===\n")
    print(f"{'condition':<12} {'<think>%':>9} {'also_open%':>12} "
          f"{'react%':>8} {'mean_overlap':>14} {'mean_len':>9}")
    for cond in CONDITIONS:
        a = agg[cond]
        mean_ov = sum(a["overlap"]) / N
        mean_len = sum(a["len"]) / N
        print(f"{cond:<12} {100*a['think']/N:>8.1f}% "
              f"{100*a['also']/N:>11.1f}% "
              f"{100*a['react']/N:>7.1f}% "
              f"{mean_ov:>14.3f} {mean_len:>9.0f}")

    print(f"\n=== GT overlap winner counts (of {N}) ===")
    for cond in CONDITIONS:
        print(f"  {cond:<12}: {wins_overlap[cond]:>3}")

    # Top sft_ctx examples
    print(f"\n=== Top {args.top_k} sft_ctx by GT overlap ===")
    scored = []
    for c in cases:
        gt_tok = content_tokens(c["ground_truth"])
        ov = jaccard(content_tokens(c["outputs"].get("sft_ctx", "")), gt_tok)
        scored.append((ov, c))
    scored.sort(key=lambda t: t[0], reverse=True)
    for ov, c in scored[: args.top_k]:
        gt = c["ground_truth"][:150].replace("\n", " ")
        sc = c["outputs"].get("sft_ctx", "")[:200].replace("\n", " ")
        snc = c["outputs"].get("sft_noctx", "")[:200].replace("\n", " ")
        bc = c["outputs"].get("base_ctx", "")[:200].replace("\n", " ")
        print(f"\n[overlap={ov:.3f}] persona={c['persona_id']} "
              f"sess={c['session_idx']} turn={c['turn_idx']}")
        print(f"  prev {c['prev_role']}: {c['prev_snippet'][:160]!r}")
        print(f"  GT        : {gt!r}")
        print(f"  sft_ctx   : {sc!r}")
        print(f"  sft_noctx : {snc!r}")
        print(f"  base_ctx  : {bc!r}")

    print(f"\n=== Bottom {args.bottom_k} sft_ctx by GT overlap ===")
    for ov, c in scored[-args.bottom_k:]:
        gt = c["ground_truth"][:150].replace("\n", " ")
        sc = c["outputs"].get("sft_ctx", "")[:250].replace("\n", " ")
        print(f"\n[overlap={ov:.3f}] persona={c['persona_id']} "
              f"sess={c['session_idx']} turn={c['turn_idx']}")
        print(f"  GT     : {gt!r}")
        print(f"  sft_ctx: {sc!r}")

    # Context effect on SFT (sft_ctx - sft_noctx overlap)
    print("\n=== Context effect on SFT (sft_ctx overlap − sft_noctx overlap) ===")
    deltas = []
    for c in cases:
        gt_tok = content_tokens(c["ground_truth"])
        oc = jaccard(content_tokens(c["outputs"].get("sft_ctx", "")), gt_tok)
        onc = jaccard(content_tokens(c["outputs"].get("sft_noctx", "")), gt_tok)
        deltas.append(oc - onc)
    deltas.sort()
    pos = sum(1 for d in deltas if d > 0)
    neg = sum(1 for d in deltas if d < 0)
    zero = N - pos - neg
    print(f"  cases where ctx helps (delta > 0): {pos}/{N}")
    print(f"  cases where ctx hurts (delta < 0): {neg}/{N}")
    print(f"  ties: {zero}")
    print(f"  mean delta:   {sum(deltas)/N:+.3f}")
    print(f"  median delta: {deltas[N//2]:+.3f}")
    print(f"  p25/p75:      {deltas[N//4]:+.3f} / {deltas[3*N//4]:+.3f}")


if __name__ == "__main__":
    main()
