#!/usr/bin/env python3
"""
Build V2 training data: text + HR + next-day wellness generation.

V2 format with:
  - User ID in input for personalization
  - Current wellness state in input (Day N)
  - Activity begin/end tokens in output HR
  - Output order: HR -> text -> next-day wellness state
  - Time-based split per user (70/10/20), snapped to day boundaries

Output: output/{train,val,test}.jsonl + output/special_tokens.json
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from config import (
    OUTPUT_DIR, CACHE_DIR,
    TOKEN_TS_START, TOKEN_TS_END,
    TOKEN_TEXT_START, TOKEN_TEXT_END,
    TOKEN_STATE_START, TOKEN_STATE_END,
    ACTIVITY_BEGIN_TOKENS, ACTIVITY_END_TOKENS, USER_TOKENS,
    hr_to_token, normalize_activity_name, encode_wellness_state,
    get_all_special_tokens,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

# Time-based split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.10
# TEST_RATIO = 0.20 (remainder)


def _format_state_block(wellness: dict | None) -> str:
    """Encode a wellness dict as a delimited state-token block.
    Empty (Day N missing) → '<state_start> <state_end>'."""
    if not wellness:
        return f"{TOKEN_STATE_START} {TOKEN_STATE_END}"
    toks = encode_wellness_state(wellness)
    if not toks:
        return f"{TOKEN_STATE_START} {TOKEN_STATE_END}"
    return f"{TOKEN_STATE_START} {' '.join(toks)} {TOKEN_STATE_END}"


def format_v2_sample(parsed: dict, synthesized: dict,
                     wellness_by_date: dict[str, dict[str, dict]]) -> dict | None:
    """Build one V2 training sample from parsed + synthesized records.

    Looks up Day N (current) and Day N+1 (next day) wellness via the
    wellness_by_date index. Drops the sample if Day N+1 is missing.
    """
    hr_baseline = parsed.get("hr_baseline")
    hr_sequence = parsed.get("hr_sequence")
    user_fb = synthesized.get("user_feedback", "").strip()
    event_desc = synthesized.get("event_description", "").strip()
    duration = parsed.get("duration_min", 0)
    pid = parsed.get("participant_id", "")
    start_time = parsed.get("start_time", "")

    if not hr_sequence or not user_fb:
        return None

    # ── Resolve Day N / Day N+1 wellness ─────────────────────────────
    try:
        dt = datetime.fromisoformat(start_time)
    except (ValueError, TypeError):
        return None
    date_n = dt.date()
    date_n1 = date_n + timedelta(days=1)
    pid_wellness = wellness_by_date.get(pid, {})
    wn = pid_wellness.get(date_n.isoformat())     # may be None
    wn1 = pid_wellness.get(date_n1.isoformat())   # required
    if not wn1:
        return None  # drop: cannot train wellness output

    # ── Input ────────────────────────────────────────────────────────
    input_parts = []

    # Learned user token for personalization
    if pid and pid in USER_TOKENS:
        input_parts.append(USER_TOKENS[pid])

    # Current state (Day N) — possibly empty
    input_parts.append(f"### Current State:\n{_format_state_block(wn)}")

    if hr_baseline and len(hr_baseline) >= 3:
        baseline_tokens = " ".join(hr_to_token(int(round(b))) for b in hr_baseline)
        input_parts.append(
            f"### Baseline HR:\n{TOKEN_TS_START} {baseline_tokens} {TOKEN_TS_END}"
        )

    if event_desc:
        input_parts.append(f"### Event:\n{event_desc}")

    # Compute recovery length from actual data
    exercise_tokens = min(int(round(duration)), len(hr_sequence))
    recovery_tokens = len(hr_sequence) - exercise_tokens

    input_parts.append(f"### Exercise Tokens: {exercise_tokens}")
    input_parts.append(f"### Recovery Tokens: {recovery_tokens}")

    input_text = "\n".join(input_parts)

    # ── Output (HR -> text -> next-day wellness) ─────────────────────
    activity = normalize_activity_name(parsed.get("activity_name", "other"))
    begin_tok = ACTIVITY_BEGIN_TOKENS[activity]
    end_tok = ACTIVITY_END_TOKENS[activity]
    end_pos = exercise_tokens

    hr_parts = [begin_tok]
    for i, b in enumerate(hr_sequence):
        hr_parts.append(hr_to_token(int(round(b))))
        if i + 1 == end_pos:
            hr_parts.append(end_tok)

    output_text = (
        f"{TOKEN_TS_START} {' '.join(hr_parts)} {TOKEN_TS_END}\n"
        f"{TOKEN_TEXT_START} {user_fb} {TOKEN_TEXT_END}\n"
        f"{_format_state_block(wn1)}"
    )

    full_sequence = f"{input_text}\n\n### Response:\n{output_text}"

    return {
        "input_text": input_text,
        "output_text": output_text,
        "full_sequence": full_sequence,
        "participant_id": pid,
        "start_time": start_time,
        "activity_name": parsed.get("activity_name", ""),
        "duration_min": duration,
        "exercise_tokens": exercise_tokens,
        "recovery_tokens": recovery_tokens,
        "hr_seq_len": len(hr_sequence),
        "hr_baseline_len": len(hr_baseline) if hr_baseline else 0,
        "wellness_day_n": wn,        # may be None
        "wellness_day_n1": wn1,
        "has_current_state": wn is not None,
    }


def time_based_split(samples: list[dict],
                     train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO):
    """
    Split samples by time within each participant.

    Snaps split boundaries to day boundaries so multi-activity days are not
    split between train/val or val/test.
    """
    by_pid = defaultdict(list)
    for s in samples:
        by_pid[s["participant_id"]].append(s)

    train, val, test = [], [], []

    for pid in sorted(by_pid.keys()):
        user_samples = by_pid[pid]
        # Sort by date first, then by full timestamp
        user_samples.sort(key=lambda s: (s.get("start_time", "")[:10],
                                         s.get("start_time", "")))

        n = len(user_samples)
        target_train = int(n * train_ratio)
        target_val_end = int(n * (train_ratio + val_ratio))

        # Snap forward to next day boundary so all samples on the same date
        # land in the same split.
        def snap_forward(idx: int) -> int:
            if idx <= 0 or idx >= n:
                return idx
            current_day = user_samples[idx - 1].get("start_time", "")[:10]
            j = idx
            while j < n and user_samples[j].get("start_time", "")[:10] == current_day:
                j += 1
            return j

        train_end = snap_forward(target_train)
        val_end = snap_forward(target_val_end)
        if val_end <= train_end:
            val_end = train_end  # empty val if collapsed

        train.extend(user_samples[:train_end])
        val.extend(user_samples[train_end:val_end])
        test.extend(user_samples[val_end:])

        log.info(f"  {pid}: {n} total → "
                 f"{train_end} train / {val_end - train_end} val / "
                 f"{n - val_end} test")

    return train, val, test


def run():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load records ─────────────────────────────────────────────────
    parsed = json.loads((CACHE_DIR / "parsed_records.json").read_text())
    synthesized = json.loads((CACHE_DIR / "synthesized_records.json").read_text())
    wellness_idx_path = CACHE_DIR / "wellness_by_date.json"
    if not wellness_idx_path.exists():
        raise FileNotFoundError(
            f"{wellness_idx_path} not found. Run step1_parse_pmdata.py first "
            f"to generate the wellness_by_date index."
        )
    wellness_by_date = json.loads(wellness_idx_path.read_text())
    log.info(f"Loaded {len(parsed)} parsed, {len(synthesized)} synthesized records, "
             f"{len(wellness_by_date)} pids in wellness index")

    if len(parsed) != len(synthesized):
        log.warning(f"Record count mismatch: {len(parsed)} vs {len(synthesized)}")

    # ── Build samples ────────────────────────────────────────────────
    samples = []
    skipped_no_hr_or_text = 0
    skipped_no_next_wellness = 0
    samples_with_wn = 0
    samples_without_wn = 0
    for i in range(min(len(parsed), len(synthesized))):
        p = parsed[i]
        s = synthesized[i]
        # Pre-check for next-day wellness so we can count drops
        try:
            dt = datetime.fromisoformat(p.get("start_time", ""))
            date_n1 = (dt.date() + timedelta(days=1)).isoformat()
            pid_w = wellness_by_date.get(p.get("participant_id", ""), {})
            has_wn1 = date_n1 in pid_w
        except (ValueError, TypeError):
            has_wn1 = False

        sample = format_v2_sample(p, s, wellness_by_date)
        if sample is None:
            if not p.get("hr_sequence") or not s.get("user_feedback", "").strip():
                skipped_no_hr_or_text += 1
            elif not has_wn1:
                skipped_no_next_wellness += 1
            else:
                skipped_no_hr_or_text += 1
        else:
            samples.append(sample)
            if sample["has_current_state"]:
                samples_with_wn += 1
            else:
                samples_without_wn += 1

    log.info(f"Built {len(samples)} V2 samples")
    log.info(f"  with current state (Day N):    {samples_with_wn}")
    log.info(f"  without current state:         {samples_without_wn}")
    log.info(f"  skipped (no HR/text):          {skipped_no_hr_or_text}")
    log.info(f"  skipped (no Day N+1 wellness): {skipped_no_next_wellness}")

    # ── Time-based split (snapped to day boundaries) ─────────────────
    log.info("Splitting by time (per user: 70/10/20, snapped to day boundaries):")
    train, val, test = time_based_split(samples)

    # ── Save ─────────────────────────────────────────────────────────
    def save_jsonl(data, path):
        with open(path, "w", encoding="utf-8") as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    save_jsonl(train, OUTPUT_DIR / "train.jsonl")
    save_jsonl(val, OUTPUT_DIR / "val.jsonl")
    save_jsonl(test, OUTPUT_DIR / "test.jsonl")

    tokens = get_all_special_tokens()
    with open(OUTPUT_DIR / "special_tokens.json", "w") as f:
        json.dump(tokens, f, indent=2)

    log.info("=" * 60)
    log.info(f"  V2 data built (with next-day wellness):")
    log.info(f"    Train: {len(train)}")
    log.info(f"    Val:   {len(val)}")
    log.info(f"    Test:  {len(test)}")
    log.info(f"  Saved to {OUTPUT_DIR}/")
    log.info("=" * 60)


if __name__ == "__main__":
    run()
