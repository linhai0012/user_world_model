"""Analyze gated-KL training dynamics from training_log.json.

Reads `training_log.json` files written by train_opd_dual.py with
--gated-kl, prints per-step gate-ratio trajectory and per-session
aggregate gate ratio (Phase 2b plan §5.4: per-session mean gate_ratio
== Titans-style "surprise" signal).

Usage:
    # Default: read all 4 personas from $SCRATCHDIR/P-OPSD/student_lora/
    python dynamic_usersim/student_opd/analyze_gate_ratio.py

    # Override tag (default dual_v3_gated_k3) or root:
    python ... --tag dual_v3_gated_k10
    python ... --root /path/to/lora_root
    python ... --personas 4 14
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

NAMES = {0: "Kanoa", 4: "Lisa", 12: "Jordan", 14: "Leilani"}


def buckets_for_steps(max_step: int) -> list[tuple[int, int]]:
    """Pick reasonable bucket edges depending on training length."""
    if max_step <= 600:
        return [(0, 50), (50, 200), (200, 400), (400, max_step + 1)]
    return [(0, 50), (50, 200), (200, 400), (400, 600),
            (600, 800), (800, max_step + 1)]


def per_step_summary(steps: list[dict]) -> None:
    print(f"  per-step trajectory (avg over windows):")
    if not steps:
        print("    (no steps recorded)")
        return
    max_step = max(s["step"] for s in steps)
    for lo, hi in buckets_for_steps(max_step):
        chunk = [s for s in steps if lo <= s["step"] < hi]
        if not chunk:
            continue
        avg = lambda k: sum(s.get(k, 0) for s in chunk) / len(chunk)
        gates = [s.get("gate_ratio", 1.0) for s in chunk]
        n_zero = sum(1 for g in gates if g == 0.0)
        print(f"    [{lo:4d},{hi:4d}): n={len(chunk):3d}  "
              f"loss={avg('loss'):.3f}  "
              f"gate={sum(gates)/len(gates):.3f}  "
              f"slow_gn={avg('slow_gnorm'):.3f}  "
              f"fast_gn={avg('fast_gnorm'):.3f}  "
              f"resp_len={avg('min_len'):.0f}  "
              f"fully_gated_skip={n_zero}")


def per_session_summary(steps: list[dict], top_n: int = 5) -> None:
    """Aggregate gate_ratio by (context_id, session_idx) and report
    high/low gate sessions."""
    by_sess: dict[tuple, list[float]] = defaultdict(list)
    for s in steps:
        ctx = s.get("context_id")
        sess = s.get("session_idx")
        if ctx is None or sess is None:
            continue
        by_sess[(ctx, sess)].append(s.get("gate_ratio", 1.0))

    if not by_sess:
        print("  (no per-session metadata — older trainer?)")
        return

    sess_means = {k: sum(v) / len(v) for k, v in by_sess.items()}
    sorted_asc = sorted(sess_means.items(), key=lambda kv: kv[1])
    n_zero = sum(1 for v in sess_means.values() if v == 0.0)
    n_high = sum(1 for v in sess_means.values() if v > 0.5)

    print(f"  per-session gate ratio: n_sessions={len(sess_means)}, "
          f"mean={sum(sess_means.values())/len(sess_means):.3f}, "
          f"sessions w/ gate=0 (full skip): {n_zero}, "
          f"sessions w/ gate>0.5: {n_high}")
    print(f"  lowest-gate (student already knows = redundant) sessions:")
    for (ctx, sess), v in sorted_asc[:top_n]:
        n = len(by_sess[(ctx, sess)])
        print(f"    ctx={ctx[:8]} sess={sess:2d}  gate={v:.3f}  (n_samples={n})")
    print(f"  highest-gate (teacher has novel info = useful) sessions:")
    for (ctx, sess), v in sorted_asc[-top_n:][::-1]:
        n = len(by_sess[(ctx, sess)])
        print(f"    ctx={ctx[:8]} sess={sess:2d}  gate={v:.3f}  (n_samples={n})")


def analyze_one(persona_dir: Path) -> None:
    log_path = persona_dir / "training_log.json"
    if not log_path.exists():
        print(f"  no training_log.json at {log_path}")
        return
    d = json.loads(log_path.read_text())
    if not d.get("gated_kl", False):
        print(f"  not a gated-KL run (gated_kl=False); ungated run "
              f"won't have meaningful gate ratios")
    steps = d.get("per_step_losses", [])
    n_skipped = d.get("n_fully_gated_skipped", 0)
    print(f"  total steps recorded: {len(steps)}  "
          f"fully_gated_skipped: {n_skipped}  "
          f"margin: {d.get('kl_gate_margin', 'n/a')}")
    per_step_summary(steps)
    per_session_summary(steps)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="dual_v3_gated_k3",
                    help="Persona-dir suffix tag (default dual_v3_gated_k3 "
                         "for Round 2a). Use dual_v3_gated_k10 for 2b.")
    ap.add_argument("--root", type=Path, default=None,
                    help="Parent dir holding lora_pid{N}_<tag>_ep1_r3teacher/ "
                         "subdirs. Default $SCRATCHDIR/P-OPSD/student_lora.")
    ap.add_argument("--personas", type=int, nargs="+",
                    default=[0, 4, 12, 14])
    ap.add_argument("--epochs", type=int, default=1)
    args = ap.parse_args()

    if args.root is None:
        scratch = os.environ.get("SCRATCHDIR")
        if scratch is None:
            raise SystemExit("$SCRATCHDIR not set; pass --root explicitly")
        args.root = Path(scratch) / "P-OPSD" / "student_lora"

    print(f"=== gate-ratio analysis  tag={args.tag}  root={args.root} ===")
    for pid in args.personas:
        persona_dir = args.root / f"lora_pid{pid}_{args.tag}_ep{args.epochs}_r3teacher"
        print(f"\n--- pid={pid} {NAMES.get(pid, '?')}  ({persona_dir}) ---")
        if not persona_dir.exists():
            print(f"  (dir missing)")
            continue
        analyze_one(persona_dir)


if __name__ == "__main__":
    main()
