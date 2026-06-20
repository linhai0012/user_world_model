"""Education Stage-1 eval: NLL of the student's next turn under frozen Qwen3-4B, base vs +memory.

  python scripts/eval_edu_stage1.py [--course nlp|ai|both]

Lower mean per-token NLL = the frozen model predicts the student's next turn better. The
base/+memory contrast tests whether the conversation history (episodic memory) helps a frozen
reader predict the learner's reaction — the education text-head analog of the general-domain
memory ablation (project_summary.md §6). No training, no per-user weights (no learner identity
in the data); frozen base only. Paired comparison (same targets), so Δ is per-turn meaningful.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch
from common.edu_data import COURSES, build_edu_samples

BASE = "Qwen/Qwen3-4B-Instruct-2507"


def nll_of_target(model, tok, ctx: list[dict], target: str, max_len: int) -> float | None:
    """Mean per-token NLL of `target` rendered as the next user(student) turn after `ctx`."""
    full = tok.apply_chat_template(ctx + [{"role": "user", "content": target}],
                                   tokenize=False, add_generation_prompt=False)
    pre = tok.apply_chat_template(ctx + [{"role": "user", "content": ""}],
                                  tokenize=False, add_generation_prompt=False)
    fi = tok(full, add_special_tokens=False)["input_ids"]
    pi = tok(pre, add_special_tokens=False)["input_ids"]
    b = 0
    for x, y in zip(pi, fi):
        if x != y:
            break
        b += 1
    if len(fi) > max_len:                      # keep the target; drop oldest context tokens
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
    ap.add_argument("--course", default="both")
    ap.add_argument("--max-len", type=int, default=8192)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()

    courses = list(COURSES) if args.course == "both" else [args.course]
    summary = {}
    for course in courses:
        # build paired base/memory samples (same targets, same order)
        paired = {c: build_edu_samples(course, c) for c in ("base", "memory")}
        n = len(paired["base"])
        res = {}
        nll_by_cond = {}
        for cond in ("base", "memory"):
            nlls = [v for s in paired[cond]
                    if (v := nll_of_target(model, tok, s["messages"], s["target"], args.max_len)) is not None]
            nll_by_cond[cond] = nlls
            res[cond] = {"n": len(nlls), "mean_nll": round(statistics.mean(nlls), 4)}
            print(f"[edu:{course}] {cond:7s} n={len(nlls)} mean_NLL={res[cond]['mean_nll']}", flush=True)
        # paired delta (per-sample memory-base) over the aligned samples
        paired_d = [m - b for b, m in zip(nll_by_cond["base"], nll_by_cond["memory"])]
        win = sum(d < 0 for d in paired_d)
        res["paired"] = {"n": len(paired_d), "mean_delta": round(statistics.mean(paired_d), 4),
                         "memory_better_frac": round(win / len(paired_d), 3)}
        print(f"[edu:{course}] Δ(memory-base) mean={res['paired']['mean_delta']:+.4f} "
              f"memory_better={win}/{len(paired_d)} "
              f"({'memory HELPS' if res['paired']['mean_delta'] < 0 else 'memory does NOT help'})",
              flush=True)
        summary[course] = res

    out = REPO / "experiments" / "results" / "edu_stage1__nll.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"[done] -> {out}")


if __name__ == "__main__":
    main()
