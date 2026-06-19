"""Health Stage-1 intrinsic-prediction eval: predict next-day wellness state (project_summary §6).

Mirrors the general-domain frozen-base ablation, but the target is a 6-field structured state
(a world-model state transition) rather than an MCQ. Conditions (what the model is conditioned
on to predict next-day state):

  persistence   no model — predict next = current (the strong baseline legacy couldn't beat)
  pop-mean      no model — predict the population mean state (control)
  base          frozen LLM, activity only (no current state, no profile)
  +current      frozen LLM + current-day state (memory of today)
  +profile      frozen LLM + the participant's baseline mean state (structured profile)
  +current+prof frozen LLM + both

Metric: per-field + overall MAE vs the true next-day state; report each arm's overall MAE and
how it compares to persistence. Frozen Qwen3-4B via vLLM (states as plain text → JSON out).

  source scripts/env.sh ; python scripts/eval_health_stage1.py [--limit N]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from common.health_data import (FIELDS, HealthRecord, clamp, load_records,
                                participant_baselines, render_state)

CONDS = ["persistence", "pop-mean", "base", "+current", "+profile", "+current+prof"]
SYS = ("You predict a specific user's NEXT-day wellness self-report after an activity. "
       "Wellness fields are integers: fatigue 1-5, mood 1-5, readiness 1-10, sleep_quality 1-5, "
       "soreness 1-5 (1=very sore,5=none), stress 1-5 (1=very stressed,5=very relaxed). "
       "Output ONLY a JSON object with exactly these 6 integer fields.")


def build_prompt(rec: HealthRecord, cond: str, baseline: dict) -> str:
    lines = [f"Activity: {rec.activity} for {rec.duration_min:.0f} min."]
    if cond in ("+profile", "+current+prof"):
        lines.append(f"This user's typical wellness baseline: {render_state(baseline)}.")
    if cond in ("+current", "+current+prof"):
        lines.append(f"Today's wellness (before tomorrow): {render_state(rec.state_n)}.")
    lines.append("Predict tomorrow's wellness as JSON "
                 '{"fatigue":_,"mood":_,"readiness":_,"sleep_quality":_,"soreness":_,"stress":_}.')
    return "\n".join(lines)


def parse_state(text: str, fallback: dict) -> dict:
    try:
        start = text.index("{"); obj = json.loads(text[start:text.index("}", start) + 1])
        return {f: clamp(f, int(round(float(obj[f])))) if f in obj else fallback[f] for f in FIELDS}
    except (ValueError, KeyError, TypeError):
        return dict(fallback)


def mae(preds: list[dict], golds: list[dict]) -> dict:
    out = {}
    for f in FIELDS:
        out[f] = round(sum(abs(p[f] - g[f]) for p, g in zip(preds, golds)) / len(golds), 3)
    out["overall"] = round(sum(out[f] for f in FIELDS) / len(FIELDS), 3)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--conds", default=",".join(CONDS))
    args = ap.parse_args()

    train = load_records("train")
    test = load_records("test")
    if args.limit:
        test = test[:args.limit]
    baselines = participant_baselines(train)
    pop_mean = {f: round(sum(b[f] for b in baselines.values()) / len(baselines)) for f in FIELDS}
    golds = [r.state_n1 for r in test]
    print(f"[health] train={len(train)} test={len(test)} participants={len(baselines)}", flush=True)

    conds = [c.strip() for c in args.conds.split(",")]
    llm_conds = [c for c in conds if c not in ("persistence", "pop-mean")]
    backend = None
    if llm_conds:
        import os
        os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
        from vllm import LLM, SamplingParams
        backend = LLM(model="Qwen/Qwen3-4B-Instruct-2507", dtype="bfloat16",
                      max_model_len=2048, gpu_memory_utilization=0.85)
        tok = backend.get_tokenizer()
        sp = SamplingParams(temperature=0.0, max_tokens=64)

    results = {}
    for cond in conds:
        t0 = time.time()
        if cond == "persistence":
            preds = [r.state_n for r in test]
        elif cond == "pop-mean":
            preds = [pop_mean for _ in test]
        else:
            prompts = [tok.apply_chat_template(
                [{"role": "system", "content": SYS},
                 {"role": "user", "content": build_prompt(r, cond, baselines.get(r.pid, pop_mean))}],
                tokenize=False, add_generation_prompt=True) for r in test]
            outs = backend.generate(prompts, sp)
            preds = [parse_state(o.outputs[0].text, test[i].state_n) for i, o in enumerate(outs)]
        m = mae(preds, golds)
        results[cond] = m
        print(f"[health] {cond:14s} overall_MAE={m['overall']}  ({time.time()-t0:.1f}s)", flush=True)

    persist = results.get("persistence", {}).get("overall")
    summary = {"task": "health_stage1_next_state", "n_test": len(test),
               "n_participants": len(baselines), "fields": FIELDS,
               "mae_by_cond": results, "persistence_overall": persist}
    out = REPO / "experiments" / "results" / "health_stage1__mae.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n[done] persistence overall MAE={persist}; results -> {out}")
    for c in conds:
        d = results[c]["overall"] - persist if persist is not None else None
        print(f"  {c:14s} {results[c]['overall']:.3f}  Δvs_persist={d:+.3f}" if d is not None else f"  {c}: {results[c]['overall']}")


if __name__ == "__main__":
    main()
