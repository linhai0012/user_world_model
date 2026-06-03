"""
Step 3: Build the final training dataset.

V3: Day-to-day wellness prediction (world model).

  INPUT:  User State (Day N) + Activity summary + optional sRPE + optional baseline HR
  OUTPUT: <text_start> user feedback <text_end>
          <state_start> <fatigue_2> <mood_4> ... <state_end>

State tokens in output = Day N+1 morning wellness (ground truth).
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np

from config import (
    CACHE_DIR, OUTPUT_DIR, RANDOM_SEED,
    TOKEN_TS_START, TOKEN_TS_END,
    TOKEN_TEXT_START, TOKEN_TEXT_END,
    TOKEN_STATE_START, TOKEN_STATE_END,
    WELLNESS_FIELDS,
    hr_to_token, encode_wellness_state, get_all_special_tokens,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  Format one daily sample
# ═══════════════════════════════════════════════════════════════════════════════

def _format_wellness_text(w: dict) -> str:
    """Format wellness dict as human-readable text for input."""
    parts = []
    for field in ("fatigue", "mood", "sleep_quality", "soreness", "stress"):
        val = w.get(field)
        if val is not None:
            parts.append(f"{field.replace('_', ' ').title()}: {val}/5")
    readiness = w.get("readiness")
    if readiness is not None:
        parts.append(f"Readiness: {readiness}/10")
    return ", ".join(parts)


def _format_wellness_compact(w: dict) -> str:
    """Compact wellness format for history lines. e.g. 'Fat3 Mood4 Rdy7 Slp3 Sor3 Str4'"""
    abbrev = {"fatigue": "Fat", "mood": "Mood", "readiness": "Rdy",
              "sleep_quality": "Slp", "soreness": "Sor", "stress": "Str"}
    parts = []
    for field, ab in abbrev.items():
        val = w.get(field)
        if val is not None:
            parts.append(f"{ab}{val}")
    return " ".join(parts)


def _format_history(history: list[dict]) -> str:
    """Format recent history as compact text lines."""
    if not history:
        return ""
    lines = []
    for h in history:
        offset = h["day_offset"]
        w_str = _format_wellness_compact(h["wellness"])
        w_next_str = _format_wellness_compact(h["wellness_next"])

        act = h.get("activity")
        if act:
            act_names = ", ".join(act["activities"])
            act_str = f"{act_names} {int(act['duration_min'])}min"
            if act.get("calories"):
                act_str += f" {act['calories']}cal"
            if act.get("perceived_exertion"):
                act_str += f" RPE{act['perceived_exertion']}"
        else:
            act_str = "Rest"

        lines.append(f"Day {offset}: {w_str} | {act_str} -> {w_next_str}")
    return "\n".join(lines)


def format_sample(record: dict[str, Any]) -> dict[str, Any] | None:
    """
    Convert one daily record into a V3 training sample.

    Input: Day N state + activity + optional sRPE + optional baseline HR
    Output: user feedback text + Day N+1 state tokens
    """
    event_desc = record.get("event_description", "").strip()
    user_fb = record.get("user_feedback", "").strip()
    wn = record.get("wellness_day_n")
    wn1 = record.get("wellness_day_n1")

    if not event_desc or not user_fb or not wn or not wn1:
        return None

    # ── Build input ────────────────────────────────────────────────────
    input_parts = []

    # User ID
    pid = record.get("participant_id", "")
    if pid:
        input_parts.append(f"### User: {pid}")

    # Recent history (past 7 days of transitions)
    history = record.get("history", [])
    if history:
        history_text = _format_history(history)
        input_parts.append(f"### Recent History:\n{history_text}")

    # Day N wellness
    input_parts.append(f"### User State (Day N):\n{_format_wellness_text(wn)}")

    # Optional baseline HR
    hr_baseline = record.get("hr_baseline")
    if hr_baseline and len(hr_baseline) >= 3:
        baseline_tokens = " ".join(hr_to_token(int(round(bpm))) for bpm in hr_baseline)
        input_parts.append(
            f"### Baseline HR:\n{TOKEN_TS_START} {baseline_tokens} {TOKEN_TS_END}"
        )

    # Activity summary
    act_names = record.get("activity_names", [])
    act_str = ", ".join(act_names) if act_names else "Unknown"
    dur = record.get("total_duration_min", 0)
    cal = record.get("total_calories", 0)
    steps = record.get("total_steps", 0)

    act_parts = [f"{act_str}, total {int(round(dur))} minutes"]
    if cal:
        act_parts.append(f"{cal} calories")
    if steps:
        act_parts.append(f"{steps} steps")

    # sRPE
    srpe = record.get("srpe")
    if srpe and srpe.get("perceived_exertion"):
        act_parts.append(f"Perceived Exertion: {srpe['perceived_exertion']}/10")

    input_parts.append(f"### Activity (Day N):\n{', '.join(act_parts)}")

    input_text = "\n".join(input_parts)

    # ── Build output ───────────────────────────────────────────────────
    state_tokens = encode_wellness_state(wn1)
    if not state_tokens:
        return None

    output_text = (
        f"{TOKEN_TEXT_START} {user_fb} {TOKEN_TEXT_END} "
        f"{TOKEN_STATE_START} {' '.join(state_tokens)} {TOKEN_STATE_END}"
    )

    # Full training sequence
    full_sequence = f"{input_text}\n\n### Response:\n{output_text}"

    return {
        "input_text": input_text,
        "output_text": output_text,
        "full_sequence": full_sequence,
        "participant_id": record.get("participant_id", ""),
        "date": record.get("date", ""),
        "activity_names": act_names,
        "total_duration_min": dur,
        # Ground truth for evaluation
        "wellness_day_n": wn,
        "wellness_day_n1": wn1,
        "has_srpe": srpe is not None,
        "has_baseline_hr": hr_baseline is not None and len(hr_baseline) >= 3,
        "history_days": len(history),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Build splits
# ═══════════════════════════════════════════════════════════════════════════════

def split_dataset(
    samples: list[dict],
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
    test_ratio: float = 0.20,
) -> tuple[list, list, list]:
    """
    Temporal split: for each participant, sort by date, then
    first 70% → train, next 10% → val, last 20% → test.

    Every participant has data in all splits. This simulates the
    real use case: train on a user's history, predict their future.
    """
    by_pid: dict[str, list] = {}
    for s in samples:
        pid = s["participant_id"]
        by_pid.setdefault(pid, []).append(s)

    train, val, test = [], [], []
    for pid in sorted(by_pid):
        # Sort by date within each participant
        pid_samples = sorted(by_pid[pid], key=lambda s: s.get("date", ""))
        n = len(pid_samples)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))

        train.extend(pid_samples[:n_train])
        val.extend(pid_samples[n_train:n_train + n_val])
        test.extend(pid_samples[n_train + n_val:])

    random.seed(RANDOM_SEED)
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    train_pids = sorted(set(s["participant_id"] for s in train))
    val_pids = sorted(set(s["participant_id"] for s in val))
    test_pids = sorted(set(s["participant_id"] for s in test))
    log.info(f"Temporal split: train={len(train)} ({len(train_pids)} pids), "
             f"val={len(val)} ({len(val_pids)} pids), "
             f"test={len(test)} ({len(test_pids)} pids)")
    return train, val, test


# ═══════════════════════════════════════════════════════════════════════════════
#  Save utilities
# ═══════════════════════════════════════════════════════════════════════════════

def save_jsonl(samples: list[dict], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def compute_stats(samples: list[dict]) -> dict:
    if not samples:
        return {}

    input_lens = [len(s["input_text"].split()) for s in samples]
    output_lens = [len(s["output_text"].split()) for s in samples]

    activities = {}
    for s in samples:
        for a in s.get("activity_names", []):
            activities[a] = activities.get(a, 0) + 1

    has_srpe = sum(1 for s in samples if s.get("has_srpe"))
    has_baseline = sum(1 for s in samples if s.get("has_baseline_hr"))

    # Day N+1 state distribution
    state_dist = {}
    for field in WELLNESS_FIELDS:
        vals = [s["wellness_day_n1"].get(field) for s in samples
                if isinstance(s.get("wellness_day_n1", {}).get(field), (int, float))]
        if vals:
            unique, counts = np.unique(vals, return_counts=True)
            state_dist[field] = {
                "mean": round(float(np.mean(vals)), 2),
                "distribution": {int(u): int(c) for u, c in zip(unique, counts)},
            }

    return {
        "total_samples": len(samples),
        "input_word_count": {
            "min": min(input_lens), "max": max(input_lens),
            "mean": round(sum(input_lens) / len(input_lens), 1),
        },
        "output_word_count": {
            "min": min(output_lens), "max": max(output_lens),
            "mean": round(sum(output_lens) / len(output_lens), 1),
        },
        "has_srpe_pct": round(has_srpe / len(samples) * 100, 1),
        "has_baseline_hr_pct": round(has_baseline / len(samples) * 100, 1),
        "activity_distribution": dict(sorted(activities.items(), key=lambda x: -x[1])),
        "unique_participants": len(set(s["participant_id"] for s in samples)),
        "day_n1_state_distribution": state_dist,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load synthesized daily records (step1 daily + step2 v3)
    synth_path = CACHE_DIR / "synthesized_daily_records.json"
    if not synth_path.exists():
        raise FileNotFoundError(f"Run step2 (v3) first. Missing {synth_path}")
    records = json.loads(synth_path.read_text())
    log.info(f"Loaded {len(records)} synthesized daily records")

    # Format all samples
    samples = []
    for r in records:
        s = format_sample(r)
        if s:
            samples.append(s)
    log.info(f"Formatted {len(samples)} valid training samples")

    if not samples:
        log.error("No valid samples produced. Check upstream data.")
        return

    # Split
    train, val, test = split_dataset(samples)

    # Save JSONL splits
    save_jsonl(train, OUTPUT_DIR / "train.jsonl")
    save_jsonl(val, OUTPUT_DIR / "val.jsonl")
    save_jsonl(test, OUTPUT_DIR / "test.jsonl")
    log.info("Saved train.jsonl, val.jsonl, test.jsonl")

    # Save special tokens list
    tokens = get_all_special_tokens()
    tokens_path = OUTPUT_DIR / "special_tokens.json"
    tokens_path.write_text(json.dumps(tokens, indent=2))
    log.info(f"Saved {len(tokens)} special tokens → {tokens_path}")

    # Save stats
    all_stats = {
        "overall": compute_stats(samples),
        "train": compute_stats(train),
        "val": compute_stats(val),
        "test": compute_stats(test),
    }
    stats_path = OUTPUT_DIR / "dataset_stats.json"
    stats_path.write_text(json.dumps(all_stats, indent=2, ensure_ascii=False))
    log.info(f"Stats → {stats_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("  DATASET CONSTRUCTION COMPLETE (V3 — Day-to-Day Wellness)")
    print("=" * 60)
    print(f"  Total samples:  {len(samples)}")
    print(f"  Train / Val / Test:  {len(train)} / {len(val)} / {len(test)}")
    print(f"  With sRPE: {sum(1 for s in samples if s.get('has_srpe'))}/{len(samples)}")
    print(f"  With baseline HR: {sum(1 for s in samples if s.get('has_baseline_hr'))}/{len(samples)}")
    print(f"  Special tokens: {len(tokens)}")
    print(f"  Output dir:     {OUTPUT_DIR.resolve()}")
    print("=" * 60)

    # Show one example
    ex = train[0]
    print("\n── Example training sample (V3) ──")
    print(f"INPUT:\n{ex['input_text']}\n")
    preview = ex["output_text"]
    if len(preview) > 400:
        preview = preview[:400] + " ..."
    print(f"OUTPUT:\n{preview}\n")


if __name__ == "__main__":
    run()
