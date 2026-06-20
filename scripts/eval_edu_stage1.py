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
from common.edu_data import COURSES, build_edu_eval_items

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
    conds = ("base", "memory", "foreign")
    summary = {}
    for course in courses:
        items = build_edu_eval_items(course)
        rows = []   # per item: {cond: nll, turn}
        for it in items:
            row = {"turn": it["turn"]}
            ok = True
            for c in conds:
                v = nll_of_target(model, tok, it[c], it["target"], args.max_len)
                if v is None:
                    ok = False
                    break
                row[c] = v
            if ok:
                rows.append(row)
        res = {c: {"n": len(rows), "mean_nll": round(statistics.mean(r[c] for r in rows), 4)}
               for c in conds}
        for c in conds:
            print(f"[edu:{course}] {c:7s} n={res[c]['n']} mean_NLL={res[c]['mean_nll']}", flush=True)
        # paired deltas
        dmb = [r["memory"] - r["base"] for r in rows]
        dfb = [r["foreign"] - r["base"] for r in rows]
        dmf = [r["memory"] - r["foreign"] for r in rows]
        res["deltas"] = {
            "memory_minus_base": round(statistics.mean(dmb), 4),
            "foreign_minus_base": round(statistics.mean(dfb), 4),
            "memory_minus_foreign": round(statistics.mean(dmf), 4),
            "memory_better_than_base_frac": round(sum(d < 0 for d in dmb) / len(dmb), 3),
            "memory_better_than_foreign_frac": round(sum(d < 0 for d in dmf) / len(dmf), 3),
        }
        # per-turn-depth: does memory help more at deeper turns?
        depth = {}
        for label, lo, hi in (("early(turn<=4)", 0, 4), ("late(turn>4)", 5, 10**9)):
            sub = [r for r in rows if lo <= r["turn"] <= hi]
            if sub:
                depth[label] = {"n": len(sub),
                                "memory_minus_base": round(statistics.mean(r["memory"] - r["base"] for r in sub), 4)}
        res["by_depth"] = depth
        print(f"[edu:{course}] Δmem-base={res['deltas']['memory_minus_base']:+.4f}  "
              f"Δforeign-base={res['deltas']['foreign_minus_base']:+.4f}  "
              f"Δmem-foreign={res['deltas']['memory_minus_foreign']:+.4f}  "
              f"| depth: {depth}", flush=True)
        summary[course] = res

    out = REPO / "experiments" / "results" / "edu_stage1__nll.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"[done] -> {out}")


if __name__ == "__main__":
    main()
