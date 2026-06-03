"""Analyze the results of phllm_predictability_test.py --use_llm.

Loads the saved JSON and prints:
  1. Within-user MAE (per-case averaging, controls between-user variance)
  2. LLM prediction bias and variance vs ground truth
  3. Parse success rate
  4. Per-case breakdown
  5. A few raw LLM outputs for inspection
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")


def mae(preds, truths):
    pairs = [(p, t) for p, t in zip(preds, truths) if p is not None]
    if not pairs:
        return float("nan"), 0
    diffs = [abs(p - t) for p, t in pairs]
    return float(np.mean(diffs)), len(pairs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="output/phllm_predictability/results_n10_w14_seed42_llm.json")
    args = ap.parse_args()

    path = Path(args.results)
    if not path.exists():
        print(f"ERROR: {path} not found", file=sys.stderr)
        sys.exit(1)

    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["rows"]
    has_llm = any("rhr_llm" in r for r in rows)
    print(f"Loaded {len(rows)} rows from {path}")
    print(f"  use_llm: {has_llm}")
    print(f"  RHR target stats: mean={data['rhr_target_stats']['mean']:.2f}, "
          f"std={data['rhr_target_stats']['std']:.2f}")
    print(f"  HRV target stats: mean={data['hrv_target_stats']['mean']:.2f}, "
          f"std={data['hrv_target_stats']['std']:.2f}")

    # ── 1. Pooled MAE (recap) ──
    print("\n=== 1. Pooled MAE (all rows together) ===")
    print(f"{'predictor':>14s}  {'RHR MAE':>10s}  {'HRV MAE':>10s}  {'n':>5s}")
    for label, key in [("Persistence", "persist"), ("7-day mean", "mean7"), ("Qwen3-4B", "llm")]:
        if key == "llm" and not has_llm:
            continue
        rhr_p = [r.get(f"rhr_{key}") for r in rows]
        hrv_p = [r.get(f"hrv_{key}") for r in rows]
        rhr_t = [r["rhr_true"] for r in rows]
        hrv_t = [r["hrv_true"] for r in rows]
        rm, rn = mae(rhr_p, rhr_t)
        hm, hn = mae(hrv_p, hrv_t)
        print(f"{label:>14s}  {rm:>10.3f}  {hm:>10.3f}  {rn:>5d}")

    # ── 2. Within-user MAE ──
    print("\n=== 2. Within-user MAE (per-case mean, then average across cases) ===")
    by_case = defaultdict(list)
    for r in rows:
        by_case[r["case_id"]].append(r)

    print(f"{'predictor':>14s}  {'RHR MAE':>10s}  {'HRV MAE':>10s}  {'cases':>6s}")
    for label, key in [("Persistence", "persist"), ("7-day mean", "mean7"), ("Qwen3-4B", "llm")]:
        if key == "llm" and not has_llm:
            continue
        case_rhr_maes = []
        case_hrv_maes = []
        for cid, case_rows in by_case.items():
            rm, _ = mae([r.get(f"rhr_{key}") for r in case_rows],
                        [r["rhr_true"] for r in case_rows])
            hm, _ = mae([r.get(f"hrv_{key}") for r in case_rows],
                        [r["hrv_true"] for r in case_rows])
            if not np.isnan(rm):
                case_rhr_maes.append(rm)
            if not np.isnan(hm):
                case_hrv_maes.append(hm)
        rm_avg = float(np.mean(case_rhr_maes)) if case_rhr_maes else float("nan")
        hm_avg = float(np.mean(case_hrv_maes)) if case_hrv_maes else float("nan")
        print(f"{label:>14s}  {rm_avg:>10.3f}  {hm_avg:>10.3f}  {len(by_case):>6d}")

    # ── 3. LLM bias / variance ──
    if has_llm:
        print("\n=== 3. LLM prediction bias and variance ===")
        rhr_llm = np.array([r["rhr_llm"] for r in rows if r.get("rhr_llm") is not None])
        hrv_llm = np.array([r["hrv_llm"] for r in rows if r.get("hrv_llm") is not None])
        rhr_t = np.array([r["rhr_true"] for r in rows if r.get("rhr_llm") is not None])
        hrv_t = np.array([r["hrv_true"] for r in rows if r.get("hrv_llm") is not None])

        print(f"  RHR  truth: mean={rhr_t.mean():.2f}, std={rhr_t.std():.2f}")
        print(f"  RHR  LLM:   mean={rhr_llm.mean():.2f}, std={rhr_llm.std():.2f}")
        print(f"  RHR  bias (LLM - truth): mean={(rhr_llm - rhr_t).mean():+.3f}")
        print(f"  RHR  pred std / truth std: {rhr_llm.std()/rhr_t.std():.3f}  "
              f"(<<1 = mode collapse to population mean)")
        print()
        print(f"  HRV  truth: mean={hrv_t.mean():.2f}, std={hrv_t.std():.2f}")
        print(f"  HRV  LLM:   mean={hrv_llm.mean():.2f}, std={hrv_llm.std():.2f}")
        print(f"  HRV  bias (LLM - truth): mean={(hrv_llm - hrv_t).mean():+.3f}")
        print(f"  HRV  pred std / truth std: {hrv_llm.std()/hrv_t.std():.3f}")

        # ── 4. Parse success ──
        n_total = len(rows)
        n_rhr_ok = sum(1 for r in rows if r.get("rhr_llm") is not None)
        n_hrv_ok = sum(1 for r in rows if r.get("hrv_llm") is not None)
        print(f"\n=== 4. Parse success ===")
        print(f"  Total rows: {n_total}")
        print(f"  RHR parsed: {n_rhr_ok} ({n_rhr_ok/n_total*100:.1f}%)")
        print(f"  HRV parsed: {n_hrv_ok} ({n_hrv_ok/n_total*100:.1f}%)")

        # ── 5. Win rate vs persistence ──
        print(f"\n=== 5. LLM vs Persistence head-to-head (per-row) ===")
        for field in ("rhr", "hrv"):
            wins = ties = losses = 0
            for r in rows:
                if r.get(f"{field}_llm") is None:
                    continue
                e_llm = abs(r[f"{field}_llm"] - r[f"{field}_true"])
                e_per = abs(r[f"{field}_persist"] - r[f"{field}_true"])
                if e_llm < e_per - 1e-6:
                    wins += 1
                elif e_llm > e_per + 1e-6:
                    losses += 1
                else:
                    ties += 1
            n = wins + ties + losses
            print(f"  {field.upper()}: LLM beats persist {wins}/{n} ({wins/n*100:.1f}%), "
                  f"loses {losses} ({losses/n*100:.1f}%), ties {ties}")

        # ── 6. Per-case breakdown ──
        print(f"\n=== 6. Per-case MAE breakdown ===")
        print(f"{'case_id':>10s}  {'n':>3s}  "
              f"{'RHR_per':>8s} {'RHR_m7':>8s} {'RHR_llm':>8s}  "
              f"{'HRV_per':>8s} {'HRV_m7':>8s} {'HRV_llm':>8s}")
        for cid, case_rows in sorted(by_case.items()):
            n = len(case_rows)
            r_per, _ = mae([r["rhr_persist"] for r in case_rows], [r["rhr_true"] for r in case_rows])
            r_m7, _ = mae([r["rhr_mean7"] for r in case_rows], [r["rhr_true"] for r in case_rows])
            r_llm, _ = mae([r.get("rhr_llm") for r in case_rows], [r["rhr_true"] for r in case_rows])
            h_per, _ = mae([r["hrv_persist"] for r in case_rows], [r["hrv_true"] for r in case_rows])
            h_m7, _ = mae([r["hrv_mean7"] for r in case_rows], [r["hrv_true"] for r in case_rows])
            h_llm, _ = mae([r.get("hrv_llm") for r in case_rows], [r["hrv_true"] for r in case_rows])
            print(f"{cid:>10s}  {n:>3d}  "
                  f"{r_per:>8.2f} {r_m7:>8.2f} {r_llm:>8.2f}  "
                  f"{h_per:>8.2f} {h_m7:>8.2f} {h_llm:>8.2f}")

        # ── 7. Sample raw outputs ──
        print(f"\n=== 7. Sample raw LLM outputs (first 3, middle, last) ===")
        idxs = [0, 1, 2, len(rows)//2, len(rows)-1]
        for i in idxs:
            if i >= len(rows):
                continue
            r = rows[i]
            print(f"\n--- row {i} (case={r['case_id']}, target_day={r['target_day']}) ---")
            print(f"  truth:     RHR={r['rhr_true']}, HRV={r['hrv_true']}")
            print(f"  persist:   RHR={r['rhr_persist']}, HRV={r['hrv_persist']:.1f}")
            print(f"  llm pred:  RHR={r.get('rhr_llm')}, HRV={r.get('hrv_llm')}")
            raw = r.get("llm_raw", "")
            print(f"  llm raw:   {raw[:200]}")


if __name__ == "__main__":
    main()
