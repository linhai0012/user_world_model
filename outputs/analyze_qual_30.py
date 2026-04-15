"""Structured analysis of 30-case qualitative eval.

For each case, scores simple signals we care about:
  - <think> leakage (base-only artifact)
  - Generic "I also..." opener (SFT monologue mode)
  - Content word overlap with ground-truth user turn (Jaccard on unigrams,
    stop-words filtered)
  - "reaction-word" presence (yes/no/hmm/actually/not quite/agree) in first
    100 chars — does the output engage with the assistant turn?

Then:
  - Per-condition aggregate (<think> rate, also-opener rate, mean overlap,
    reaction-word rate)
  - Ranking which condition best matches GT per case (argmax overlap)
  - Win counts per condition across 30 cases
"""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
d = json.load((HERE / "eval_ckpt50_qual.json").open("r", encoding="utf-8"))
cases = d["cases"]
N = len(cases)

STOP = set("a an the of to and or but for in on at with by as is are was were be been "
           "being have has had do does did can could would should will shall may might "
           "this that these those i you he she it we they me my your his her our their "
           "not no so if than then also too very really just so yes".split())

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z']+")


def content_tokens(s: str) -> set:
    return {t.lower() for t in TOKEN_RE.findall(s) if t.lower() not in STOP and len(t) > 2}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(len(a | b), 1)


REACT_WORDS = {"yes", "no", "yeah", "nope", "not quite", "not really", "hmm",
                "actually", "agree", "disagree", "absolutely", "definitely",
                "exactly", "that sounds", "sounds good", "sounds great",
                "that's true", "that is true", "correct", "right", "wrong",
                "you're right"}


def has_react(s: str) -> bool:
    head = s[:100].lower()
    return any(w in head for w in REACT_WORDS)


ALSO_OPEN = re.compile(r"^\s*i (also|have also|'ve also|'m also)\b", re.IGNORECASE)


def also_opener(s: str) -> bool:
    return bool(ALSO_OPEN.match(s))


def has_think(s: str) -> bool:
    return "<think>" in s or "</think>" in s


# Per-case analysis
conditions = ["base_noctx", "base_ctx", "sft_noctx", "sft_ctx"]
agg = {c: {"think": 0, "also": 0, "react": 0, "overlap": [], "len": []}
       for c in conditions}
wins_overlap = Counter()
for c in cases:
    gt = c["ground_truth"]
    gt_tok = content_tokens(gt)
    per = {}
    for cond in conditions:
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
    # best condition by overlap (argmax)
    best = max(per, key=per.get)
    wins_overlap[best] += 1


print(f"=== 30-case analysis ===\n")
print(f"{'condition':<12} {'<think>%':>9} {'also_open%':>12} {'react%':>8} "
      f"{'mean_overlap':>14} {'mean_len':>9}")
for cond in conditions:
    a = agg[cond]
    n = N
    mean_ov = sum(a["overlap"]) / n
    mean_len = sum(a["len"]) / n
    print(f"{cond:<12} {100*a['think']/n:>8.1f}% {100*a['also']/n:>11.1f}% "
          f"{100*a['react']/n:>7.1f}% {mean_ov:>14.3f} {mean_len:>9.0f}")

print(f"\n=== GT overlap winner counts (of {N}) ===")
for cond in conditions:
    print(f"  {cond:<12}: {wins_overlap[cond]:>3}")

# Show the top-overlap sft_ctx examples
print(f"\n=== Top 5 sft_ctx outputs by GT overlap (content-word Jaccard) ===")
scored = []
for i, c in enumerate(cases):
    gt_tok = content_tokens(c["ground_truth"])
    ov = jaccard(content_tokens(c["outputs"].get("sft_ctx", "")), gt_tok)
    scored.append((ov, i, c))
scored.sort(key=lambda t: t[0], reverse=True)
scored = [(ov, c) for ov, _, c in scored]
for ov, c in scored[:5]:
    gt = c["ground_truth"][:150].replace("\n", " ")
    sc = c["outputs"].get("sft_ctx", "")[:200].replace("\n", " ")
    snc = c["outputs"].get("sft_noctx", "")[:200].replace("\n", " ")
    bc = c["outputs"].get("base_ctx", "")[:200].replace("\n", " ")
    print(f"\n[overlap={ov:.3f}] persona={c['persona_id']} sess={c['session_idx']} turn={c['turn_idx']}")
    print(f"  prev {c['prev_role']}: {c['prev_snippet'][:160]!r}")
    print(f"  GT        : {gt!r}")
    print(f"  sft_ctx   : {sc!r}")
    print(f"  sft_noctx : {snc!r}")
    print(f"  base_ctx  : {bc!r}")

# Also show worst sft_ctx cases (to understand failure modes)
print(f"\n=== Bottom 3 sft_ctx outputs by GT overlap ===")
for ov, c in scored[-3:]:
    gt = c["ground_truth"][:150].replace("\n", " ")
    sc = c["outputs"].get("sft_ctx", "")[:250].replace("\n", " ")
    print(f"\n[overlap={ov:.3f}] persona={c['persona_id']} sess={c['session_idx']} turn={c['turn_idx']}")
    print(f"  GT     : {gt!r}")
    print(f"  sft_ctx: {sc!r}")

# Pairwise delta: sft_ctx vs sft_noctx (does context help?)
print(f"\n=== Context effect on SFT (sft_ctx - sft_noctx overlap) ===")
deltas = []
for c in cases:
    gt_tok = content_tokens(c["ground_truth"])
    oc = jaccard(content_tokens(c["outputs"].get("sft_ctx", "")), gt_tok)
    onc = jaccard(content_tokens(c["outputs"].get("sft_noctx", "")), gt_tok)
    deltas.append(oc - onc)
deltas.sort()
pos = sum(1 for d in deltas if d > 0)
neg = sum(1 for d in deltas if d < 0)
zero = sum(1 for d in deltas if d == 0)
print(f"  cases where ctx helps (delta > 0): {pos}/{N}")
print(f"  cases where ctx hurts (delta < 0): {neg}/{N}")
print(f"  ties: {zero}")
print(f"  mean delta: {sum(deltas)/N:+.3f}")
print(f"  median delta: {deltas[N//2]:+.3f}")
print(f"  p25/p75: {deltas[N//4]:+.3f} / {deltas[3*N//4]:+.3f}")
