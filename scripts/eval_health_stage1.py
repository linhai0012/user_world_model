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

from common.health_data import (CONDS, FIELDS, SYS, build_prompt, load_records,
                                parse_state, participant_baselines)


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
        # Reuse the proven MCQ backend init (CONVENTIONS §3): it sets the B200
        # flash-attn env quirk before importing vllm and constructs LLM identically
        # to the working scoring path. (The earlier raw LLM() "Device string must
        # not be empty" was a no-GPU/login-node artifact — must run on a GPU node.)
        from common.backends import VLLMQwenBackend
        backend = VLLMQwenBackend(max_model_len=2048)
        tok = backend.tok

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
            texts = backend.generate(prompts, max_tokens=64)
            preds = [parse_state(texts[i], test[i].state_n) for i in range(len(test))]
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
