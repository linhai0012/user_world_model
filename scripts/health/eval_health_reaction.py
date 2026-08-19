"""Health reaction-text head eval: NLL of the user's first-person reaction under frozen Qwen3-4B,
across the base/+current/+profile/+current+prof ablation.

  python scripts/health/eval_health_reaction.py [--limit N]

The second UWM output head for health (project_summary.md §2: reaction_text alongside next_state).
Lower mean per-token NLL = the frozen model predicts the user's reaction better; the base→+context
contrast is the text-head analog of the state-head ablation. Reaction text is GPT-synthesized
(grounded in the real activity/HR/self-report) → a SOFT signal; the primary head stays the real
next-state MAE. Frozen base only (no training) for this first cut.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch
from domains.health.data import (FIELDS, SYS_REACTION, build_reaction_prompt, load_records,
                                participant_baselines)

BASE = "Qwen/Qwen3-4B-Instruct-2507"
CONDS = ["base", "+current", "+profile", "+current+prof"]


def nll(model, tok, sys_msg, user_msg, target, max_len=2048):
    base = [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}]
    full = tok.apply_chat_template(base + [{"role": "assistant", "content": target}],
                                   tokenize=False, add_generation_prompt=False)
    pre = tok.apply_chat_template(base, tokenize=False, add_generation_prompt=True)
    fi = tok(full, add_special_tokens=False)["input_ids"]
    pi = tok(pre, add_special_tokens=False)["input_ids"]
    b = 0
    for x, y in zip(pi, fi):
        if x != y:
            break
        b += 1
    if len(fi) > max_len:
        cut = len(fi) - max_len
        fi = fi[cut:]
        b = max(0, b - cut)
    if b >= len(fi):
        return None
    ids = torch.tensor([fi], device=model.device)
    labels = torch.tensor([[-100] * b + fi[b:]], device=model.device)
    with torch.no_grad():
        return model(input_ids=ids, labels=labels).loss.item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--conds", default=",".join(CONDS))
    args = ap.parse_args()

    train = load_records("train")
    test = load_records("test")
    if args.limit:
        test = test[:args.limit]
    test = [r for r in test if r.reaction_text]
    baselines = participant_baselines(train)
    pop_mean = {f: round(sum(b[f] for b in baselines.values()) / len(baselines)) for f in FIELDS}
    print(f"[reaction] test (with reaction) = {len(test)}", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()

    conds = [c.strip() for c in args.conds.split(",")]
    summary = {}
    for cond in conds:
        nlls = []
        for r in test:
            um = baselines.get(r.pid, pop_mean)
            v = nll(model, tok, SYS_REACTION, build_reaction_prompt(r, cond, um), r.reaction_text)
            if v is not None:
                nlls.append(v)
        summary[cond] = {"n": len(nlls), "mean_nll": round(statistics.mean(nlls), 4)}
        print(f"[reaction] {cond:14s} n={len(nlls)} mean_NLL={summary[cond]['mean_nll']}", flush=True)

    out = REPO / "experiments" / "results" / "health" / "health_reaction__nll.json"
    out.write_text(json.dumps({"task": "health_stage1_reaction_nll", "by_cond": summary}, indent=2))
    base = summary.get("base", {}).get("mean_nll")
    for c in conds:
        d = summary[c]["mean_nll"] - base if base is not None else None
        print(f"  {c:14s} {summary[c]['mean_nll']:.4f}" + (f"  Δvs_base={d:+.4f}" if d is not None else ""))
    print(f"[done] -> {out}")


if __name__ == "__main__":
    main()
