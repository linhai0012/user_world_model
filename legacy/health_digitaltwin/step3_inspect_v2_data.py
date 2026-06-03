#!/usr/bin/env python3
"""
Quick inspection of V2 training data. Shows sample formats and statistics.

Usage:
    python step3_inspect_v2_data.py
"""
import json
import logging
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR = Path("output")


def inspect_split(name: str, path: Path):
    if not path.exists():
        log.warning(f"{name}: file not found at {path}")
        return

    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line.strip()))

    log.info(f"\n{'='*60}")
    log.info(f"  {name}: {len(samples)} samples")
    log.info(f"{'='*60}")

    if not samples:
        return

    # Activity distribution
    acts = Counter(s.get("activity_name", "?") for s in samples)
    log.info(f"  Activity distribution:")
    for act, count in acts.most_common():
        log.info(f"    {act:>20s}: {count:>4d} ({100*count/len(samples):.1f}%)")

    # Participant distribution
    pids = Counter(s.get("participant_id", "?") for s in samples)
    log.info(f"  Participants: {len(pids)}")
    for pid, count in sorted(pids.items()):
        log.info(f"    {pid}: {count}")

    # Time range per participant
    log.info(f"  Time ranges:")
    by_pid = {}
    for s in samples:
        pid = s.get("participant_id", "?")
        t = s.get("start_time", "")
        if pid not in by_pid:
            by_pid[pid] = {"min": t, "max": t}
        if t < by_pid[pid]["min"]:
            by_pid[pid]["min"] = t
        if t > by_pid[pid]["max"]:
            by_pid[pid]["max"] = t
    for pid in sorted(by_pid.keys()):
        r = by_pid[pid]
        log.info(f"    {pid}: {r['min'][:10]} to {r['max'][:10]}")

    # HR sequence lengths
    hr_lens = [s.get("hr_seq_len", 0) for s in samples]
    log.info(f"  HR seq lengths: {min(hr_lens)}-{max(hr_lens)} "
             f"(mean {sum(hr_lens)/len(hr_lens):.0f})")

    # Show 3 example samples
    log.info(f"\n  --- Example Samples ---")
    for i in [0, len(samples)//2, len(samples)-1]:
        s = samples[i]
        log.info(f"\n  [{name} sample {i}] "
                 f"user={s['participant_id']} "
                 f"activity={s['activity_name']} "
                 f"time={s.get('start_time','')[:16]}")
        log.info(f"  INPUT:\n    {s['input_text'][:300]}")
        log.info(f"  OUTPUT:\n    {s['output_text'][:300]}")


def main():
    for name, fname in [("Train", "train.jsonl"), ("Val", "val.jsonl"), ("Test", "test.jsonl")]:
        inspect_split(name, OUTPUT_DIR / fname)


if __name__ == "__main__":
    main()
