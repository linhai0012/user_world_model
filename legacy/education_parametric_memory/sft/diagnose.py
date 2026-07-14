"""Diagnose whether per-user SFT carries ANY learner signal: does its p_correct track the
persona's TRUE per-skill mastery better than base, and does it separate weak vs strong skills?"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WS = Path("/scratch/prj/cllm/cue_sft")
PRED = WS / os.environ.get("CUE_PRED_DIR", "predictions")
SNAPSHOTS = [int(x) for x in os.environ.get("CUE_SNAPSHOTS", "0,5,10,20,40").split(",")]

splits = json.loads((ROOT / "substrate/splits.json").read_text())
eval_ids = set(splits["eval"])
truth = {(r["persona"], r["snapshot"], r["question_id"]): r
         for r in (json.loads(l) for l in (ROOT / "substrate/eval_truth.jsonl").read_text().splitlines())}
preds = {f.stem: [json.loads(l) for l in f.read_text().splitlines() if l.strip()] for f in PRED.glob("*.jsonl")}
personas = sorted(k for k in preds if k.startswith("persona_"))


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    return cov / math.sqrt(vx * vy) if vx > 0 and vy > 0 else float("nan")


# base flat predictions
base_flat = {r["question_id"]: r for r in preds["__base__"]}

print(f"{'snap':>5} | {'corr(per_user p, true mastery)':>30} | {'corr(base p, mastery)':>22} | "
      f"{'weak_gap':>9} {'base_gap':>9}")
print("  (weak_gap = mean p_correct on STRONG-skill items minus WEAK-skill items; bigger = better separation)")
for snap in SNAPSHOTS:
    pu_p, pu_m = [], []
    weak, strong = [], []
    for p in personas:
        for r in preds[p]:
            if r["snapshot"] != snap or r["question_id"] not in eval_ids:
                continue
            t = truth.get((p, snap, r["question_id"]))
            if not t:
                continue
            pu_p.append(r["p_correct"]); pu_m.append(t["mastery"])
            (weak if t["mastery"] < 0.4 else strong if t["mastery"] > 0.6 else []).append(r["p_correct"])
    # base correlation (same eval items, base pred, truth mastery at this snap)
    b_p, b_m, b_weak, b_strong = [], [], [], []
    for p in personas:
        for qid in eval_ids:
            t = truth.get((p, snap, qid))
            if not t or qid not in base_flat:
                continue
            bp = base_flat[qid]["p_correct"]
            b_p.append(bp); b_m.append(t["mastery"])
            (b_weak if t["mastery"] < 0.4 else b_strong if t["mastery"] > 0.6 else []).append(bp)
    wgap = (sum(strong) / len(strong) - sum(weak) / len(weak)) if weak and strong else float("nan")
    bgap = (sum(b_strong) / len(b_strong) - sum(b_weak) / len(b_weak)) if b_weak and b_strong else float("nan")
    print(f"{snap:>5} | {pearson(pu_p, pu_m):>30.4f} | {pearson(b_p, b_m):>22.4f} | {wgap:>9.4f} {bgap:>9.4f}")
