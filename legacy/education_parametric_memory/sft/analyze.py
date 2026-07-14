"""Comprehensive A1/A0/A∅ comparison on THREE metrics, to separate real signal from metric noise:
  1. binary NLL (standard KT metric; underpowered when outcomes ~ Bernoulli(mastery≈0.5))
  2. theta-recovery MSE: predicted p_correct vs TRUE latent mastery (removes Bernoulli noise)
  3. mastery correlation
Env: CUE_PRED_DIR, CUE_SUBSTRATE (substrate dir), CUE_SNAPSHOTS."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WS = Path("/scratch/prj/cllm/cue_sft")
PRED = WS / os.environ.get("CUE_PRED_DIR", "predictions")
SUB = Path(os.environ.get("CUE_SUBSTRATE", str(ROOT / "substrate")))
SNAPSHOTS = [int(x) for x in os.environ.get("CUE_SNAPSHOTS", "0,5,10,20,40").split(",")]
EPS = 1e-4


def nll(p, y):
    p = min(max(p, EPS), 1 - EPS)
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs); vy = sum((y - my) ** 2 for y in ys)
    return cov / math.sqrt(vx * vy) if vx > 0 and vy > 0 else float("nan")


splits = json.loads((SUB / "splits.json").read_text())
eval_ids = set(splits["eval"])
truth = {(r["persona"], r["snapshot"], r["question_id"]): r
         for r in (json.loads(l) for l in (SUB / "eval_truth.jsonl").read_text().splitlines())}
preds = {f.stem: [json.loads(l) for l in f.read_text().splitlines() if l.strip()] for f in PRED.glob("*.jsonl")}
personas = sorted(k for k in preds if k.startswith("persona_"))
base_flat = {r["question_id"]: r for r in preds["__base__"]}
shared_flat = {r["question_id"]: r for r in preds["__shared__"]}


def pred_for(scope, p, snap, qid, by_snap):
    if scope == "per_user":
        return by_snap.get((p, snap, qid))
    return (shared_flat if scope == "shared" else base_flat).get(qid, {}).get("p_correct")


# index per-user preds
pu = {}
for p in personas:
    for r in preds[p]:
        pu[(p, r["snapshot"], r["question_id"])] = r["p_correct"]


def metrics(scope, snap):
    nlls, mses, ps, ms = [], [], [], []
    for p in personas:
        for qid in eval_ids:
            t = truth.get((p, snap, qid))
            if not t:
                continue
            pred = pu.get((p, snap, qid)) if scope == "per_user" else \
                (shared_flat if scope == "shared" else base_flat).get(qid, {}).get("p_correct")
            if pred is None:
                continue
            y = int(t["is_correct"]); mast = t["mastery"]
            nlls.append(nll(pred, y)); mses.append((pred - mast) ** 2)
            ps.append(pred); ms.append(mast)
    if not nlls:
        return None
    return (sum(nlls) / len(nlls), sum(mses) / len(mses), pearson(ps, ms))


print(f"PRED={PRED.name}  SUBSTRATE={SUB}")
print(f"\n{'metric':>16} {'snap':>5} {'A1 per_user':>12} {'A0 shared':>11} {'A∅ base':>9}  winner")
for mi, mname in enumerate(["binary_NLL", "theta_MSE", "mastery_corr"]):
    for snap in SNAPSHOTS:
        vals = {sc: metrics(sc, snap)[mi] for sc in ("per_user", "shared", "base")}
        if mname == "mastery_corr":
            win = max(vals, key=vals.get)
        else:
            win = min(vals, key=vals.get)
        print(f"{mname:>16} {snap:>5} {vals['per_user']:>12.4f} {vals['shared']:>11.4f} "
              f"{vals['base']:>9.4f}  {win}")
    print()
