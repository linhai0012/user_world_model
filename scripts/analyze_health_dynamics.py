"""Diagnose whether the health domain has any LEARNABLE activity-conditioned next-day dynamics,
or whether the task is structurally capped. Pure data analysis (no model). Prints to screen.

  python scripts/analyze_health_dynamics.py

Analyses:
  A. State stability — per-field |Δ| distribution, fraction of zero-change.
  B. readiness deep-dive — value dist, |Δ| dist, count/examples of big swings (|Δ|>=4).
  C. Same-day multi-activity — records sharing one (pid,date) -> one daily transition (activity
     has no discriminative power within a day); duplicate-transition count.
  D. Irreducible variance — given today's state, how spread is tomorrow? (the ceiling any model
     conditioning on (state,activity) can reach). Per-field, conditioned on today's value.
  E. Activity signal — does activity TYPE shift Δreadiness after controlling for today's readiness?
     If Run/Sport systematically drop next-day readiness vs Walk, there is learnable signal.
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from common.health_data import FIELDS, WELLNESS_FIELDS, _data_root


def parse(s):
    if isinstance(s, dict):
        return {f: int(s[f]) for f in FIELDS if f in s}
    try:
        import ast
        d = ast.literal_eval(s)
        return {f: int(d[f]) for f in FIELDS if f in d}
    except Exception:
        return None


def load_raw(split):
    rows = []
    with (_data_root() / f"{split}.jsonl").open() as fh:
        for line in fh:
            r = json.loads(line)
            sn, sn1 = parse(r.get("wellness_day_n")), parse(r.get("wellness_day_n1"))
            if not sn or not sn1 or len(sn) < 6 or len(sn1) < 6:
                continue
            rows.append({"pid": r.get("participant_id", "?"),
                         "date": str(r.get("start_time", ""))[:10],
                         "activity": r.get("activity_name", "?"),
                         "sn": sn, "sn1": sn1})
    return rows


def act_group(a):
    a = a.lower()
    if "walk" in a or "hike" in a:
        return "Walk/Hike"
    if "run" in a or "treadmill" in a:
        return "Run/Treadmill"
    if "bike" in a or "cycl" in a or "elliptical" in a:
        return "Bike/Ellip"
    if "sport" in a or "weight" in a or "workout" in a or "aerobic" in a:
        return "Sport/Weights"
    return "other"


def hist(vals, lo, hi):
    c = Counter(vals)
    return "  ".join(f"{v}:{c.get(v,0)}" for v in range(lo, hi + 1))


def main():
    rows = load_raw("train") + load_raw("test")
    n = len(rows)
    print(f"# Health dynamics diagnosis — n={n} transitions (train+test), 16 participants\n")

    # ---- A. per-field stability ----
    print("=" * 90 + "\nA. STATE STABILITY — per-field |Δ|=|next-today| (the persistence residual)\n" + "=" * 90)
    print(f"{'field':14s} {'range':>6s} {'mean|Δ|':>8s} {'%no-change':>11s} {'%|Δ|>=2':>9s} {'std(today)':>11s}")
    for f in FIELDS:
        lo, hi = WELLNESS_FIELDS[f]
        deltas = [abs(r['sn1'][f] - r['sn'][f]) for r in rows]
        today = [r['sn'][f] for r in rows]
        print(f"{f:14s} {f'{lo}-{hi}':>6s} {statistics.mean(deltas):8.3f} "
              f"{100*sum(d==0 for d in deltas)/n:10.1f}% {100*sum(d>=2 for d in deltas)/n:8.1f}% "
              f"{statistics.pstdev(today):11.3f}")

    # ---- B. readiness deep-dive ----
    print("\n" + "=" * 90 + "\nB. READINESS DEEP-DIVE (10-pt field, the biggest error source)\n" + "=" * 90)
    rd_today = [r['sn']['readiness'] for r in rows]
    rd_next = [r['sn1']['readiness'] for r in rows]
    rd_delta = [r['sn1']['readiness'] - r['sn']['readiness'] for r in rows]
    print("today readiness hist (1-10):", hist(rd_today, 1, 10))
    print("next  readiness hist (1-10):", hist(rd_next, 1, 10))
    print("Δreadiness hist (-9..9):    ", hist(rd_delta, -9, 9))
    big = [r for r in rows if abs(r['sn1']['readiness'] - r['sn']['readiness']) >= 4]
    print(f"\nbig swings |Δreadiness|>=4: {len(big)}/{n} ({100*len(big)/n:.1f}%)")
    drops = [r for r in big if r['sn1']['readiness'] < r['sn']['readiness']]
    print(f"  of which DROPS (next<today): {len(drops)}  (e.g. crash days)")
    print("  sample big drops (today->next readiness | activity | full Δ):")
    for r in sorted(big, key=lambda r: r['sn1']['readiness'] - r['sn']['readiness'])[:6]:
        d = {f: r['sn1'][f] - r['sn'][f] for f in FIELDS}
        print(f"    {r['pid']} {r['date']} {act_group(r['activity']):14s} "
              f"readiness {r['sn']['readiness']}->{r['sn1']['readiness']}  Δall={d}")

    # ---- C. same-day multi-activity ----
    print("\n" + "=" * 90 + "\nC. SAME-DAY MULTI-ACTIVITY — does 'activity' even individuate a transition?\n" + "=" * 90)
    by_day = defaultdict(list)
    for r in rows:
        by_day[(r['pid'], r['date'])].append(r)
    days = len(by_day)
    multi = {k: v for k, v in by_day.items() if len(v) > 1}
    recs_in_multi = sum(len(v) for v in multi.values())
    print(f"unique (pid,date) days = {days};  transitions = {n};  -> {n/days:.2f} records per day")
    print(f"days with >1 activity: {len(multi)}/{days} ({100*len(multi)/days:.1f}%); "
          f"records living on such days: {recs_in_multi}/{n} ({100*recs_in_multi/n:.1f}%)")
    # within a multi-activity day, the target (sn1) is identical across activities -> verify
    identical = sum(1 for v in multi.values()
                    if all(x['sn1'] == v[0]['sn1'] and x['sn'] == v[0]['sn'] for x in v))
    print(f"multi-activity days where ALL records share the SAME (today,next) state: "
          f"{identical}/{len(multi)} -> activity is NOT individuating the transition (same daily label)")
    # unique transitions vs records
    uniq_tr = len({(tuple(sorted(r['sn'].items())), tuple(sorted(r['sn1'].items()))) for r in rows})
    print(f"distinct (today_state -> next_state) transitions: {uniq_tr}/{n} "
          f"({100*uniq_tr/n:.0f}% unique; the rest are exact duplicates)")

    # ---- D. irreducible variance given today's state (per-field) ----
    print("\n" + "=" * 90 + "\nD. IRREDUCIBLE VARIANCE — given TODAY's field value, how spread is TOMORROW?\n"
          "   (std of next | today; this is the floor for any model conditioning on state)\n" + "=" * 90)
    print(f"{'field':14s} {'std(next|today) avg':>20s}   (lower = more predictable from today alone)")
    for f in FIELDS:
        groups = defaultdict(list)
        for r in rows:
            groups[r['sn'][f]].append(r['sn1'][f])
        # pooled within-group std (weighted), the conditional spread
        within = [statistics.pstdev(v) for v in groups.values() if len(v) >= 5]
        print(f"{f:14s} {statistics.mean(within) if within else float('nan'):20.3f}")

    # ---- E. activity signal on Δreadiness, controlling for today's readiness ----
    print("\n" + "=" * 90 + "\nE. ACTIVITY SIGNAL — Δreadiness by activity group, controlled for today's readiness\n"
          "   (if Run/Sport systematically drop next-day readiness vs Walk -> learnable causal signal)\n" + "=" * 90)
    # control: only mid-range today readiness (6-8) to reduce regression-to-mean confound
    print(f"{'activity group':16s} {'n':>5s} {'meanΔreadiness':>15s} {'std':>7s}   [today readiness 6-8 only]")
    sub = [r for r in rows if 6 <= r['sn']['readiness'] <= 8]
    g = defaultdict(list)
    for r in sub:
        g[act_group(r['activity'])].append(r['sn1']['readiness'] - r['sn']['readiness'])
    for grp in ["Walk/Hike", "Run/Treadmill", "Bike/Ellip", "Sport/Weights", "other"]:
        v = g.get(grp, [])
        if v:
            print(f"{grp:16s} {len(v):5d} {statistics.mean(v):15.3f} {statistics.pstdev(v):7.3f}")
    alld = [d for v in g.values() for d in v]
    print(f"{'ALL (6-8)':16s} {len(alld):5d} {statistics.mean(alld):15.3f} {statistics.pstdev(alld):7.3f}")
    print("\n-> If the per-activity means are all ~equal and << std, activity carries little signal\n"
          "   beyond regression-to-mean; the model's ceiling is then ~predicting the mean drift.")


if __name__ == "__main__":
    main()
