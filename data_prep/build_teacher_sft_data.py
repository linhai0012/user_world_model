"""Build Teacher SFT samples: progressive context accumulation.

For each (persona, shared_context) timeline and each session index t in it,
produce one training sample:
    prefix   = concat of sessions[max(0, t-K) : t]      (context, no loss)
    target   = sessions[t]                              (loss on user tokens)

Output: JSONL at the specified path. Each line:
    {
      "persona_id":   str,
      "context_id":   str,
      "session_idx":  int,
      "prefix_session_range": [start, end),
      "prefix_messages": [{role, content}, ...],  # flattened, order preserved
      "target_messages": [{role, content}, ...],  # role starts with 'system'
    }

Tokenization and label masking happen at train time (see tokenize_teacher_sft.py).

Usage:
    python dynamic_usersim/data_prep/build_teacher_sft_data.py \
        --version 128k \
        --out dynamic_usersim/outputs/teacher_sft_128k.jsonl \
        --context-window 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from load_personamem import build_context_timelines, strip_role_prefix


def build_samples(version: str, context_window: int) -> list[dict]:
    timelines = build_context_timelines(version)
    samples: list[dict] = []

    for (pid, ctx_id), sessions in timelines.items():
        for t in range(len(sessions)):
            start = max(0, t - context_window)
            prefix_sessions = sessions[start:t]
            target_session = sessions[t]

            prefix_messages: list[dict] = []
            for s in prefix_sessions:
                for m in s.messages:
                    prefix_messages.append({
                        "role": m["role"],
                        "content": strip_role_prefix(m["content"], m["role"]),
                    })

            target_messages: list[dict] = [
                {"role": m["role"],
                 "content": strip_role_prefix(m["content"], m["role"])}
                for m in target_session.messages
            ]

            samples.append({
                "persona_id": pid,
                "context_id": ctx_id,
                "session_idx": t,
                "prefix_session_range": [start, t],
                "prefix_messages": prefix_messages,
                "target_messages": target_messages,
            })

    return samples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=["32k", "128k", "1M"], default="128k")
    ap.add_argument("--out", type=Path, required=True,
                    help="output JSONL path")
    ap.add_argument("--context-window", type=int, default=20,
                    help="max prefix sessions (session-level cap; token-level "
                         "truncation applied later at tokenize time)")
    args = ap.parse_args()

    samples = build_samples(args.version, args.context_window)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for s in samples:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Stats
    n_user_turns = sum(
        sum(1 for m in s["target_messages"] if m["role"] == "user")
        for s in samples
    )
    prefix_lens = [len(s["prefix_messages"]) for s in samples]
    target_lens = [len(s["target_messages"]) for s in samples]
    first_turn_samples = sum(1 for s in samples if s["session_idx"] == 0)

    print(f"wrote {len(samples)} samples to {args.out}")
    print(f"  user turns in targets: {n_user_turns}")
    print(f"  prefix msgs: min={min(prefix_lens)} max={max(prefix_lens)} "
          f"mean={sum(prefix_lens)/len(prefix_lens):.1f}")
    print(f"  target msgs: min={min(target_lens)} max={max(target_lens)} "
          f"mean={sum(target_lens)/len(target_lens):.1f}")
    print(f"  samples with session_idx=0 (empty prefix): {first_turn_samples}")


if __name__ == "__main__":
    main()
