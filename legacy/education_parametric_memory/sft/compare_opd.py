"""Side-by-side: hard-SFT vs OPD per-user on the three metrics (the paper's money table).

Reads two prediction dirs (A = hard baseline e.g. predictions_v4, B = OPD e.g. predictions_opd)
scored against the SAME substrate, and prints per-snapshot binary_NLL / theta_MSE / mastery_corr
for A1 per-user under each recipe, plus shared/base for reference. Pure CPU.

Env: CUE_PRED_A(predictions_v4) CUE_PRED_B(predictions_opd) CUE_SUBSTRATE(substrate) CUE_SNAPSHOTS.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WS = Path("/scratch/prj/cllm/cue_sft")
PRED_A = WS / os.environ.get("CUE_PRED_A", "predictions_v4")
PRED_B = WS / os.environ.get("CUE_PRED_B", "predictions_opd")
SUB = Path(os.environ.get("CUE_SUBSTRATE", str(ROOT / "substrate")))
SNAPSHOTS = [int(x) for x in os.environ.get("CUE_SNAPSHOTS", "0,13,26,52,104").split(",")]
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
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    return cov / math.sqrt(vx * vy) if vx > 0 and vy > 0 else float("nan")


splits = json.loads((SUB / "splits.json").read_text())
eval_ids = set(splits["eval"])
truth = {(r["persona"], r["snapshot"], r["question_id"]): r
         for r in (json.loads(l) for l in (SUB / "eval_truth.jsonl").read_text().splitlines())}


def load_pred(pdir):
    preds = {f.stem: [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
             for f in pdir.glob("*.jsonl")}
    personas = sorted(k for k in preds if k.startswith("persona_"))
    pu = {}
    for p in personas:
        for r in preds[p]:
            pu[(p, r["snapshot"], r["question_id"])] = r["p_correct"]
    return preds, personas, pu


def metrics_per_user(pu, personas, snap):
    nlls, mses, ps, ms = [], [], [], []
    for p in personas:
        for qid in eval_ids:
            t = truth.get((p, snap, qid))
            pred = pu.get((p, snap, qid))
            if not t or pred is None:
                continue
            y = int(t["is_correct"]); mast = t["mastery"]
            nlls.append(nll(pred, y)); mses.append((pred - mast) ** 2)
            ps.append(pred); ms.append(mast)
    if not nlls:
        return None
    return sum(nlls) / len(nlls), sum(mses) / len(mses), pearson(ps, ms)


_, persA, puA = load_pred(PRED_A)
_, persB, puB = load_pred(PRED_B)
common = sorted(set(persA) & set(persB))
print(f"A (hard) = {PRED_A.name}   B (OPD) = {PRED_B.name}   substrate={SUB.name}")
print(f"personas compared: {len(common)}\n")

for mi, mname in enumerate(["binary_NLL", "theta_MSE", "mastery_corr"]):
    better = "higher" if mname == "mastery_corr" else "lower"
    print(f"=== {mname}  (per-user A1; {better}=better) ===")
    print(f"{'snap':>6} {'A hard':>10} {'B OPD':>10} {'delta(B-A)':>11}  {'OPD better?':>11}")
    for snap in SNAPSHOTS:
        a = metrics_per_user(puA, common, snap)
        b = metrics_per_user(puB, common, snap)
        if not a or not b:
            continue
        va, vb = a[mi], b[mi]
        d = vb - va
        win = (vb > va) if mname == "mastery_corr" else (vb < va)
        print(f"{snap:>6} {va:>10.4f} {vb:>10.4f} {d:>+11.4f}  {'YES' if win else 'no':>11}")
    print()
