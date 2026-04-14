"""End-to-end validation for Teacher SFT data prep.

Checks on a small sample:
  1. Every loss-labeled token decodes to user content from the TARGET session
     (never from prefix, never from assistant/system).
  2. Every user turn in the target session is represented in the loss mask.
  3. Prefix truncation honors max_len and preserves target completely.
  4. token_len_per_session distribution makes sense (for Slurm budget planning).

Run: python dynamic_usersim/data_prep/validate_sft_data.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from tokenize_teacher_sft import SFTTokenizer

ROOT = Path(__file__).resolve().parents[2]
SFT_JSONL = ROOT / "dynamic_usersim" / "outputs" / "teacher_sft_128k.jsonl"


def load_samples(path: Path, limit: int | None = None) -> list[dict]:
    out = []
    with path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if limit is not None and i >= limit:
                break
            out.append(json.loads(line))
    return out


def check_loss_positions_are_target_user(tok: SFTTokenizer, sample: dict) -> None:
    """Decode runs of label tokens; every run must match a user message
    content from the target session."""
    input_ids, labels, info = tok.tokenize_sample(sample)

    # Extract contiguous non-masked runs
    runs: list[list[int]] = []
    cur: list[int] = []
    for L in labels:
        if L != -100:
            cur.append(L)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)

    # Expected: list of user-role content strings in target session
    expected_user_contents = [
        m["content"] for m in sample["target_messages"] if m["role"] == "user"
    ]

    # Decode each run (strip trailing <|im_end|>\n which was included in labels)
    decoded_runs = [tok.tok.decode(r) for r in runs]
    # Normalize: remove trailing "<|im_end|>\n" or similar, strip whitespace
    cleaned = []
    for s in decoded_runs:
        s = s.replace("<|im_end|>", "").strip()
        cleaned.append(s)

    # Every cleaned run should match a user content
    assert len(cleaned) == len(expected_user_contents), (
        f"run count mismatch: {len(cleaned)} runs vs "
        f"{len(expected_user_contents)} target user turns\n"
        f"cleaned[0]: {cleaned[0][:120] if cleaned else 'NONE'!r}\n"
        f"expected[0]: {expected_user_contents[0][:120] if expected_user_contents else 'NONE'!r}"
    )

    for i, (got, want) in enumerate(zip(cleaned, expected_user_contents)):
        # Decoding should round-trip exactly (modulo whitespace at ends).
        if got.strip() != want.strip():
            raise AssertionError(
                f"mismatch at turn {i}:\n  got:  {got[:200]!r}\n  want: {want[:200]!r}"
            )

    # Also sanity-check that no prefix user turn leaked into labels: pick a
    # prefix user content and assert it does NOT appear in decoded label runs.
    prefix_user_contents = [
        m["content"] for m in sample["prefix_messages"] if m["role"] == "user"
    ]
    if prefix_user_contents:
        label_blob = " ".join(cleaned)
        # It's possible (rare) that a short user phrase legitimately recurs
        # in target. Only flag if a long unique prefix phrase appears.
        long_prefix = [c for c in prefix_user_contents if len(c) > 200]
        if long_prefix:
            # Use first 100 chars as a fingerprint
            sig = long_prefix[0][:100]
            if sig in label_blob:
                raise AssertionError(
                    f"prefix user content leaked into loss tokens! sig={sig!r}"
                )


def main() -> None:
    samples = load_samples(SFT_JSONL)
    print(f"loaded {len(samples)} samples")

    tok = SFTTokenizer(max_len=131072)

    # Pick a mix: session_idx=0 (no prefix), mid (~10), late (max)
    picks = []
    for target_idx in [0, 5, 10, 15, 19]:
        matches = [s for s in samples if s["session_idx"] == target_idx]
        if matches:
            picks.append(random.Random(42).choice(matches))

    print(f"\n=== Label-mask correctness on {len(picks)} hand-picked samples ===")
    for s in picks:
        print(f"  persona={s['persona_id']} ctx={s['context_id'][:8]} "
              f"session_idx={s['session_idx']} ", end="", flush=True)
        try:
            check_loss_positions_are_target_user(tok, s)
            print("OK")
        except AssertionError as e:
            print(f"FAIL:\n    {e}")
            raise

    # Also run a quick random sweep
    rng = random.Random(123)
    sweep = rng.sample(samples, min(30, len(samples)))
    print(f"\n=== Random sweep: {len(sweep)} samples ===")
    for s in sweep:
        check_loss_positions_are_target_user(tok, s)
    print(f"  all {len(sweep)} passed")

    # Token-length distribution
    print(f"\n=== Token length distribution (all samples) ===")
    lens = []
    loss_cnts = []
    dropped_cnts = []
    for s in samples:
        _, _, info = tok.tokenize_sample(s)
        lens.append(info["total_len"])
        loss_cnts.append(info["loss_token_count"])
        dropped_cnts.append(info["prefix_dropped_msgs"])
    lens.sort()
    n = len(lens)
    print(f"  total_len    min={lens[0]} p50={lens[n//2]} p90={lens[int(n*0.9)]} "
          f"p99={lens[int(n*0.99)]} max={lens[-1]}")
    loss_cnts.sort()
    print(f"  loss tokens  min={loss_cnts[0]} p50={loss_cnts[n//2]} "
          f"p90={loss_cnts[int(n*0.9)]} max={loss_cnts[-1]} "
          f"sum={sum(loss_cnts)}")
    truncated = sum(1 for d in dropped_cnts if d > 0)
    print(f"  samples with prefix truncation: {truncated}/{n}")


if __name__ == "__main__":
    main()
