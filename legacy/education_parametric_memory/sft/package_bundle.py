"""Phase 5 — calibrate + package the REAL parametric bundle, and print the headline A1/A0/A-null.

Reads the predictions from run_all + the substrate ground truth; fits a shared isotonic
calibration per scope on the held-out CALIB split; computes NLL/Brier convergence curves on the
EVAL split; writes the 5-file bundle in the schema offline_replay imports. Pure CPU, no deps
beyond stdlib (isotonic via Pool-Adjacent-Violators).
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WS = Path("/scratch/prj/cllm/cue_sft")
PRED = WS / os.environ.get("CUE_PRED_DIR", "predictions")
OUT = WS / "bundles" / os.environ.get("CUE_BUNDLE", "cue-param-biology-v1")
SNAPSHOTS = [int(x) for x in os.environ.get("CUE_SNAPSHOTS", "0,5,10,20,40").split(",")]
EPS = 1e-4


# --------------------------------------------------------------- isotonic (PAV)
def isotonic_fit(pairs: list[tuple[float, float]]):
    """Fit a monotonic calibrator from (x=pred, y=outcome) pairs via Pool Adjacent Violators.
    Returns a function mapping a raw prob -> calibrated prob (piecewise-constant, interpolated)."""
    if not pairs:
        return lambda x: x
    pts = sorted(pairs)
    xs = [x for x, _ in pts]
    ys = [y for _, y in pts]
    w = [1.0] * len(ys)
    # PAV
    i = 0
    vals = ys[:]
    while i < len(vals) - 1:
        if vals[i] > vals[i + 1] + 1e-12:
            new = (vals[i] * w[i] + vals[i + 1] * w[i + 1]) / (w[i] + w[i + 1])
            vals[i] = new
            w[i] += w[i + 1]
            del vals[i + 1]; del w[i + 1]; del xs[i + 1]
            if i > 0:
                i -= 1
        else:
            i += 1
    # vals is the pooled monotone level set at the kept xs (left edges); build a step lookup
    knots_x, knots_y = xs, vals

    def cal(p: float) -> float:
        # nearest-knot (step) interpolation, clamped
        if p <= knots_x[0]:
            return knots_y[0]
        if p >= knots_x[-1]:
            return knots_y[-1]
        lo, hi = 0, len(knots_x) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if knots_x[mid] < p:
                lo = mid + 1
            else:
                hi = mid
        return knots_y[max(0, lo - 1)]
    return cal


# --------------------------------------------------------------- metrics
def nll(p: float, y: int) -> float:
    p = min(max(p, EPS), 1 - EPS)
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def brier(p: float, y: int) -> float:
    return (p - y) ** 2


# --------------------------------------------------------------- load
def load():
    splits = json.loads((ROOT / "substrate/splits.json").read_text())
    eval_ids, calib_ids = set(splits["eval"]), set(splits["calib"])
    truth = {}  # (persona, snapshot, qid) -> {is_correct, selected_option_id, mastery}
    for l in (ROOT / "substrate/eval_truth.jsonl").read_text().splitlines():
        r = json.loads(l)
        truth[(r["persona"], r["snapshot"], r["question_id"])] = r
    preds = {}  # learner -> list of rows
    for f in PRED.glob("*.jsonl"):
        preds[f.stem] = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    return splits, eval_ids, calib_ids, truth, preds


# --------------------------------------------------------------- main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    splits, eval_ids, calib_ids, truth, preds = load()
    personas = sorted(k for k in preds if k.startswith("persona_"))
    print(f"loaded predictions for {len(personas)} personas + "
          f"{'__shared__' if '__shared__' in preds else 'NO shared'} + "
          f"{'__base__' if '__base__' in preds else 'NO base'}")

    # control predictions are snapshot-flat -> one {qid: row} map each (base scored only at s0)
    def flat_control(key):
        flat = {}
        for r in preds.get(key, []):
            flat.setdefault(r["question_id"], r)
        return flat
    shared_flat, base_flat = flat_control("__shared__"), flat_control("__base__")

    # --- ONE shared isotonic map on pooled CALIB pairs (all scopes), per the cross-backend spec ---
    calib_pairs = []
    for p in personas:
        for r in preds[p]:
            if r["question_id"] in calib_ids:
                t = truth.get((p, r["snapshot"], r["question_id"]))
                if t:
                    calib_pairs.append((r["p_correct"], int(t["is_correct"])))
    for flat in (shared_flat, base_flat):
        for qid, r in flat.items():
            if qid in calib_ids:
                for p in personas:
                    for s in SNAPSHOTS:
                        t = truth.get((p, s, qid))
                        if t:
                            calib_pairs.append((r["p_correct"], int(t["is_correct"])))
    cal = isotonic_fit(calib_pairs)
    def C(p):  # noqa: E306
        return min(max(cal(p), EPS), 1 - EPS)
    print(f"fitted ONE shared isotonic map on {len(calib_pairs)} calib pairs")

    curves, adapters, pred_rows = [], [], []
    agg = {sc: {s: [] for s in SNAPSHOTS} for sc in ("per_user", "shared", "base")}
    agg_raw = {sc: {s: [] for s in SNAPSHOTS} for sc in ("per_user", "shared", "base")}

    # per-user: prediction depends on snapshot
    for p in personas:
        for snap in SNAPSHOTS:
            nlls, briers, choice_hit = [], [], []
            for r in preds[p]:
                if r["snapshot"] != snap:
                    continue
                pc = C(r["p_correct"])
                if r["question_id"] in eval_ids:
                    t = truth.get((p, snap, r["question_id"]))
                    if t:
                        y = int(t["is_correct"])
                        nlls.append(nll(pc, y)); briers.append(brier(pc, y))
                        agg["per_user"][snap].append((pc, y))
                        agg_raw["per_user"][snap].append((min(max(r["p_correct"], EPS), 1 - EPS), y))
                        top = max(r["option_logprobs"], key=r["option_logprobs"].get)
                        choice_hit.append(int(top == t.get("selected_option_id")))
                    pred_rows.append({**{k: r[k] for k in ("learner_id", "subject_id", "snapshot",
                                       "question_id", "format", "p_correct_raw", "option_logprobs", "scope")},
                                      "p_correct": round(pc, 4)})
            if nlls:
                mn, mb = sum(nlls) / len(nlls), sum(briers) / len(briers)
                curves.append({"learner_id": p, "subject_id": "biology_gcse", "snapshot": snap,
                               "nll": round(mn, 4), "brier": round(mb, 4),
                               "mean_mastery_stab": 0.0, "scope": "per_user"})
                adapters.append({"learner_id": p, "subject_id": "biology_gcse", "scope": "per_user",
                                 "version": f"v-s{snap}", "snapshot": snap, "n_rounds_trained": snap,
                                 "parent_version": None,
                                 "eval": {"nll": round(mn, 4), "brier": round(mb, 4),
                                          "probe_acc": round(sum(choice_hit) / len(choice_hit), 4)},
                                 "best_step_caveat": False})

    # controls: snapshot-flat prediction, evaluated against each persona's truth AT each snapshot
    for scope, key, flat in (("shared", "__shared__", shared_flat), ("base", "__base__", base_flat)):
        for snap in SNAPSHOTS:
            nlls, briers = [], []
            for qid, r in flat.items():
                if qid not in eval_ids:
                    continue
                pc = C(r["p_correct"])
                for p in personas:
                    t = truth.get((p, snap, qid))
                    if t:
                        y = int(t["is_correct"])
                        nlls.append(nll(pc, y)); briers.append(brier(pc, y))
                        agg[scope][snap].append((pc, y))
                        agg_raw[scope][snap].append((min(max(r["p_correct"], EPS), 1 - EPS), y))
            if nlls:
                curves.append({"learner_id": key, "subject_id": "biology_gcse", "snapshot": snap,
                               "nll": round(sum(nlls) / len(nlls), 4),
                               "brier": round(sum(briers) / len(briers), 4),
                               "mean_mastery_stab": 0.0, "scope": scope})
                # emit control prediction rows at every snapshot (flat value) for offline_replay fallback
                for qid, r in flat.items():
                    if qid in eval_ids:
                        pred_rows.append({"learner_id": key, "subject_id": "biology_gcse", "snapshot": snap,
                                          "question_id": qid, "format": "mcq",
                                          "p_correct_raw": r["p_correct_raw"],
                                          "option_logprobs": r["option_logprobs"], "scope": scope,
                                          "p_correct": round(C(r["p_correct"]), 4)})

    # --- headline aggregate (calibrated + raw) ---
    def agg_metric(a, scope, snap):
        prs = a[scope][snap]
        if not prs:
            return None
        return round(sum(nll(p, y) for p, y in prs) / len(prs), 4)

    headline = {"nll_calibrated": {}, "nll_raw": {}, "brier_calibrated": {}}
    for sc in ("per_user", "shared", "base"):
        headline["nll_calibrated"][sc] = {s: agg_metric(agg, sc, s) for s in SNAPSHOTS}
        headline["nll_raw"][sc] = {s: agg_metric(agg_raw, sc, s) for s in SNAPSHOTS}
        headline["brier_calibrated"][sc] = {
            s: (round(sum(brier(p, y) for p, y in agg[sc][s]) / len(agg[sc][s]), 4) if agg[sc][s] else None)
            for s in SNAPSHOTS}

    # --- bundle files ---
    round_seq = json.loads((ROOT / "substrate/round_sequence.json").read_text())
    manifest = {
        "bundle_id": "cue-param-biology-v1",
        "base_model": "Qwen3-4B-Instruct-2507",
        "subject_ids": ["biology_gcse"],
        "created_offline": time.strftime("%Y-%m-%d"),
        "lora_recipe": {"slow_mlp_r": 32, "fast_attn_r": 16, "objective": "user_turn_response",
                        "training": "direct SFT (no teacher)"},
        "raw_completion": True,
        "calibration": {"method": "isotonic", "fit_on": "held_out_calib_split"},
        "persona_set": "cue-personas-v1",
        "round_sequence": {"seed": round_seq["seed"], "personas": round_seq["personas"]},
        "snapshots": SNAPSHOTS,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    _wl(OUT / "predictions.jsonl", pred_rows)
    _wl(OUT / "curves.jsonl", curves)
    _wl(OUT / "adapters.jsonl", adapters)
    status = json.loads((WS / "status.json").read_text()) if (WS / "status.json").exists() else {}
    (OUT / "cost.json").write_text(json.dumps({
        "train_s_per_snapshot": None, "infer_ms_per_item": None, "gpu": "1xB200-183GB",
        "total_train_s": round(status.get("elapsed_min", 0) * 60, 1),
        "note": "measured on the rig; per-snapshot/infer breakdown in run_all.log"}, indent=2))

    (OUT / "headline.json").write_text(json.dumps(headline, indent=2))
    hc = headline["nll_calibrated"]
    print("\n=== HEADLINE: mean NLL on held-out eval, CALIBRATED (lower=better) ===")
    print(f"{'snapshot':>10} {'A1 per_user':>14} {'A0 shared':>12} {'A∅ base':>10}")
    for s in SNAPSHOTS:
        a1, a0, an = hc["per_user"][s], hc["shared"][s], hc["base"][s]
        print(f"{s:>10} {a1!s:>14} {a0!s:>12} {an!s:>10}")
    a1_0, a1_f = hc["per_user"][0], hc["per_user"][SNAPSHOTS[-1]]
    print(f"\nA1 convergence: snap0={a1_0} -> snap{SNAPSHOTS[-1]}={a1_f}  "
          f"(beats A0={hc['shared'][SNAPSHOTS[-1]]}, A∅={hc['base'][SNAPSHOTS[-1]]}? "
          f"{'YES' if a1_f < hc['shared'][SNAPSHOTS[-1]] and a1_f < hc['base'][SNAPSHOTS[-1]] else 'NO'})")
    print(f"\nbundle -> {OUT}")
    print(f"predictions={len(pred_rows)} curves={len(curves)} adapters={len(adapters)}")


def _wl(path, rows):
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
