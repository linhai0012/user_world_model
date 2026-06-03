"""Cross-round MCQ-PPL comparison for Phase 2 / Phase 2b experiments.

Reads `dynamic_usersim/outputs/mcqppl_128k_*.json` and prints three tables:
  1. Raw accuracy per persona × per (round, ckpt)
  2. Closure % vs (teacher_k3 − base_demo) gap
  3. Per-persona best-step ckpt per round + macro-avg best closure

Currently configured for the rounds we've actually run:
  - P2  : single-LoRA (rank 32, all modules), demo-only student input
  - R1  : dual-LoRA s32f16 (slow MLP + fast Attn, slow_lr 1e-5), recent2 ctx
  - R1b : dual-LoRA s32f16 (slow_lr 5e-5), demo-only ctx, ckpts in scratch

Usage:
    python dynamic_usersim/student_opd/compare_rounds.py
    # show only specific personas:
    python dynamic_usersim/student_opd/compare_rounds.py --personas 4 12

Add new rounds by appending to ROUNDS — (label, name_prefix, [steps]).
The name_prefix matches the JSON filename:
    mcqppl_128k_pid{pid}_{name_prefix}_step{N}.json    (intermediate ckpt)
    mcqppl_128k_pid{pid}_{name_prefix}_final.json      (end-of-training)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"

NAMES = {0: "Kanoa", 4: "Lisa", 12: "Jordan", 14: "Leilani"}
DEFAULT_PERSONAS = [0, 4, 12, 14]

# (round_label, json_name_prefix, steps_to_show)
ROUNDS = [
    ("P2",  "student_demo",                       [200, 400, 600, 800, "final"]),
    ("R1",  "student_dual_s32f16_recent2",        [600, 800, "final"]),
    ("R1b", "student_dual_v2_demo",               [200, 400, 600, 800, 1000, "final"]),
    ("R2a", "student_dual_v3_gated_k3_demo",      [200, 400, 600, 800, 1000, "final"]),
    ("R2c", "student_dual_v3_joint_k3_demo",      [200, 400, 600, 800, 1000, "final"]),
]


VERSION = "128k"  # set by main() via --version


def acc(name: str, pid: int) -> float | None:
    p = OUT_DIR / f"mcqppl_{VERSION}_pid{pid}_{name}.json"
    return json.loads(p.read_text())["accuracy"] if p.exists() else None


def step_label(s) -> str:
    return "fin" if s == "final" else str(s)


def step_filename(s) -> str:
    return "final" if s == "final" else f"step{s}"


def make_cols() -> list[tuple[str, str]]:
    """Flatten ROUNDS into [(short_label, json_name), ...]; skip steps with no data anywhere."""
    cols: list[tuple[str, str]] = []
    for rlabel, prefix, steps in ROUNDS:
        for s in steps:
            jname = f"{prefix}_{step_filename(s)}"
            # Skip if no persona has this file (don't waste column width)
            if not any((OUT_DIR / f"mcqppl_{VERSION}_pid{pid}_{jname}.json").exists()
                       for pid in DEFAULT_PERSONAS):
                continue
            cols.append((f"{rlabel}{step_label(s)}", jname))
    return cols


def print_accuracy_table(personas: list[int], cols: list[tuple[str, str]]) -> None:
    print("=" * 132)
    print(" ACCURACY  (rounds side by side, all 128k MCQs per persona)")
    print("=" * 132)
    hdr = f"{'persona':<11}{'base':>7}{'tch_k3':>8}  " + \
          " ".join(f"{lbl:>7}" for lbl, _ in cols)
    print(hdr)
    print("-" * len(hdr))

    for pid in personas:
        b, t = acc("base_demo", pid), acc("teacher_k3", pid)
        cells = [f"{NAMES.get(pid, '?'):<8}{pid:<3}",
                 f"{b:>7.3f}" if b is not None else f"{'-':>7}",
                 f"{t:>8.3f}" if t is not None else f"{'-':>8}", ""]
        for _, name in cols:
            a = acc(name, pid)
            cells.append(f"{a:>7.3f}" if a is not None else f"{'-':>7}")
        print(" ".join(cells))

    # AVG row
    print("-" * len(hdr))
    bs = [v for v in (acc("base_demo", p) for p in personas) if v is not None]
    ts = [v for v in (acc("teacher_k3", p) for p in personas) if v is not None]
    cells = [f"{'AVG':<11}",
             f"{sum(bs)/len(bs):>7.3f}" if bs else f"{'-':>7}",
             f"{sum(ts)/len(ts):>8.3f}" if ts else f"{'-':>8}", ""]
    for _, name in cols:
        vs = [v for v in (acc(name, p) for p in personas) if v is not None]
        cells.append(f"{sum(vs)/len(vs):>7.3f}" if vs else f"{'-':>7}")
    print(" ".join(cells))


def print_closure_table(personas: list[int], cols: list[tuple[str, str]]) -> None:
    print()
    print("=" * 132)
    print(" CLOSURE %  ((student - base_demo) / (teacher_k3 - base_demo))")
    print("=" * 132)
    hdr = f"{'persona':<11}{'gap':>7}  " + \
          " ".join(f"{lbl:>7}" for lbl, _ in cols)
    print(hdr)
    print("-" * len(hdr))

    closures = {lbl: [] for lbl, _ in cols}
    for pid in personas:
        b, t = acc("base_demo", pid), acc("teacher_k3", pid)
        if b is None or t is None or t == b:
            continue
        gap = t - b
        cells = [f"{NAMES.get(pid, '?'):<8}{pid:<3}", f"{gap:>+7.3f}", ""]
        for lbl, name in cols:
            a = acc(name, pid)
            if a is None:
                cells.append(f"{'-':>7}")
            else:
                cl = 100 * (a - b) / gap
                closures[lbl].append(cl)
                cells.append(f"{cl:>+6.1f}%")
        print(" ".join(cells))

    print("-" * len(hdr))
    cells = [f"{'AVG':<11}", f"{'-':>7}", ""]
    for lbl, _ in cols:
        cells.append(f"{sum(closures[lbl])/len(closures[lbl]):>+6.1f}%"
                     if closures[lbl] else f"{'-':>7}")
    print(" ".join(cells))


def print_best_per_round(personas: list[int]) -> None:
    print()
    print("=" * 132)
    print(" PER-PERSONA BEST CKPT PER ROUND  (best-step closure %)")
    print("=" * 132)
    hdr = f"{'persona':<11}{'base':>7}{'tch_k3':>8}  " + \
          "".join(f"{r[0] + ' best':>16}" for r in ROUNDS)
    print(hdr)
    print("-" * len(hdr))

    best_closures = {r[0]: [] for r in ROUNDS}
    for pid in personas:
        b, t = acc("base_demo", pid), acc("teacher_k3", pid)
        if b is None or t is None or t == b:
            continue
        cells = [f"{NAMES.get(pid, '?'):<8}{pid:<3}",
                 f"{b:>7.3f}", f"{t:>8.3f}", ""]
        for rlabel, prefix, steps in ROUNDS:
            scored = []
            for s in steps:
                a = acc(f"{prefix}_{step_filename(s)}", pid)
                if a is not None:
                    scored.append((s, a))
            if not scored:
                cells.append(f"{'-':>16}")
                continue
            best_s, best_a = max(scored, key=lambda x: x[1])
            cl = 100 * (best_a - b) / (t - b)
            best_closures[rlabel].append(cl)
            cells.append(f"{cl:>+6.1f}% @{str(best_s):<5}")
        print(" ".join(cells))

    print("-" * len(hdr))
    cells = [f"{'AVG best':<11}", f"{'-':>7}", f"{'-':>8}", ""]
    for rlabel, _, _ in ROUNDS:
        cls = best_closures[rlabel]
        cells.append(f"{sum(cls)/len(cls):>+6.1f}%        " if cls
                     else f"{'-':>16}")
    print(" ".join(cells))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--personas", type=int, nargs="+", default=DEFAULT_PERSONAS)
    ap.add_argument("--version", choices=["32k", "128k", "1M"], default="128k",
                    help="PersonaMem-v1 MCQ context version. JSONs are looked up "
                         "as mcqppl_<version>_pid{N}_<cond>.json.")
    args = ap.parse_args()

    global VERSION
    VERSION = args.version

    print(f"### MCQ-PPL version: {VERSION} ###")
    cols = make_cols()
    print_accuracy_table(args.personas, cols)
    print_closure_table(args.personas, cols)
    print_best_per_round(args.personas)


if __name__ == "__main__":
    main()
