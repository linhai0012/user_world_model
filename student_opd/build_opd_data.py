"""Build Phase-2 OPD samples: per-user-turn.

Phase 1 (Teacher SFT) uses whole-session samples. Phase 2 (OPD) operates at
the user-turn granularity — each sample is one (chatbot_prev, user_response)
pair, with distinct views for teacher vs student:

  - Teacher view: demographics + K=3 prior sessions + current session up to
    and including chatbot_prev. Used to compute teacher logprobs on the
    student's rollout.
  - Student view: demographics + chatbot_prev only (no history). The LoRA
    must encode whatever the history contained.

Plan §1.4 + §1.6:
  - Iterate sessions per (persona, shared_context) timeline.
  - Skip session's first user turn (topic opener, unpredictable).
  - For each remaining user turn: record history_for_teacher and
    chatbot_prev for student input.

Output: one JSONL per persona under dynamic_usersim/outputs/
    opd_128k_pid{N}_k{K}.jsonl

Each line:
    {
      "persona_id":       str,
      "context_id":       str,
      "session_idx":      int,      # 0-based session within this context
      "user_turn_idx":    int,      # index within session body (excludes system)
      "n_prior_sessions": int,      # how many sessions fit in the K-window
      "demographics":     str,      # persona card text (system msg content)
      "history_messages": [{role, content}, ...],   # ends with chatbot_prev
      "chatbot_prev":     str,      # duplicate of last history msg content
      "user_response":    str,      # ground truth
    }

Usage:
    python dynamic_usersim/student_opd/build_opd_data.py \
        --personas 0 12 14 --context-window 3 --version 128k
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Reuse existing PersonaMem loader.
sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "data_prep")
)
from load_personamem import (  # noqa: E402
    build_context_timelines,
    strip_role_prefix,
)


def build_samples_for_persona(
    pid: str,
    timelines: dict,
    context_window: int,
) -> list[dict]:
    """Produce per-user-turn OPD samples for one persona.

    timelines: {(persona_id, context_id): [Session, ...]}
    """
    # Filter to this persona's timelines, sort by context_id for stability.
    my_ctxs = sorted(
        [(ctx, sess) for (p, ctx), sess in timelines.items() if p == pid]
    )

    out: list[dict] = []
    for ctx_id, sessions in my_ctxs:
        for t, session in enumerate(sessions):
            # K-window prior sessions (full messages, including system cards).
            prior_start = max(0, t - context_window)
            prior_sessions = sessions[prior_start:t]

            # Current session body (excludes system card; system lives separately
            # as demographics for the student view and as-is in the teacher
            # history).
            body = session.body  # messages after the persona card
            demographics = session.demographics

            # Walk body to locate user turns (skip first one per §1.6).
            user_turn_positions = [
                i for i, m in enumerate(body) if m["role"] == "user"
            ]
            if not user_turn_positions:
                continue
            # Skip the opener.
            user_turn_positions = user_turn_positions[1:]

            for body_idx in user_turn_positions:
                # The assistant turn immediately preceding this user turn.
                # In normal PersonaMem sessions user/assistant alternate, so
                # body[body_idx - 1] should be role=assistant.
                if body_idx == 0:
                    # First body message is a user turn (opener we already
                    # skipped at idx=0 via slicing above, so body_idx >= 1
                    # here — defensive check).
                    continue
                prev = body[body_idx - 1]
                if prev["role"] != "assistant":
                    # Malformed / unexpected — skip. (Should not happen on
                    # clean PersonaMem data.)
                    continue

                chatbot_prev = strip_role_prefix(prev["content"], "assistant")
                user_response = strip_role_prefix(body[body_idx]["content"], "user")

                # Build teacher history: prior sessions flattened +
                # current session's system card + current session body up to
                # and including chatbot_prev (exclusive of this user turn).
                history_messages: list[dict] = []
                for s in prior_sessions:
                    for m in s.messages:
                        history_messages.append({
                            "role": m["role"],
                            "content": strip_role_prefix(m["content"], m["role"]),
                        })
                # Current session's own system card (persona restatement)
                history_messages.append(
                    {"role": "system", "content": demographics}
                )
                # Current session body up to and including chatbot_prev
                for m in body[:body_idx]:
                    history_messages.append({
                        "role": m["role"],
                        "content": strip_role_prefix(m["content"], m["role"]),
                    })

                out.append({
                    "persona_id": pid,
                    "context_id": ctx_id,
                    "session_idx": t,
                    "user_turn_idx": body_idx,
                    "n_prior_sessions": len(prior_sessions),
                    "demographics": demographics,
                    "history_messages": history_messages,
                    "chatbot_prev": chatbot_prev,
                    "user_response": user_response,
                })

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--personas", nargs="+", default=["0", "12", "14"],
        help="persona_ids to build (default: 0 12 14)",
    )
    ap.add_argument(
        "--context-window", type=int, default=3,
        help="K — prior sessions in teacher history (default 3, matches R3)",
    )
    ap.add_argument(
        "--version", choices=["32k", "128k", "1M"], default="128k",
    )
    ap.add_argument(
        "--out-dir", type=Path, default=None,
        help="output directory; defaults to dynamic_usersim/outputs/",
    )
    args = ap.parse_args()

    if args.out_dir is None:
        repo_root = Path(__file__).resolve().parents[2]
        args.out_dir = repo_root / "dynamic_usersim" / "outputs"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    timelines = build_context_timelines(args.version)

    grand_stats: dict[str, dict] = {}
    for pid in args.personas:
        samples = build_samples_for_persona(pid, timelines, args.context_window)
        out_path = args.out_dir / (
            f"opd_{args.version}_pid{pid}_k{args.context_window}.jsonl"
        )
        with out_path.open("w", encoding="utf-8") as fh:
            for s in samples:
                fh.write(json.dumps(s, ensure_ascii=False) + "\n")

        # Stats
        hist_msg_counts = [len(s["history_messages"]) for s in samples]
        resp_lens = [len(s["user_response"]) for s in samples]
        n_prior = [s["n_prior_sessions"] for s in samples]
        sess_ids = sorted({(s["context_id"], s["session_idx"]) for s in samples})
        grand_stats[pid] = {
            "n_samples": len(samples),
            "n_sessions_with_samples": len(sess_ids),
            "hist_msgs_min": min(hist_msg_counts) if hist_msg_counts else 0,
            "hist_msgs_max": max(hist_msg_counts) if hist_msg_counts else 0,
            "hist_msgs_mean": (sum(hist_msg_counts) / len(hist_msg_counts)
                               if hist_msg_counts else 0),
            "n_prior_sess_mean": (sum(n_prior) / len(n_prior)
                                  if n_prior else 0),
            "user_resp_chars_p50": (
                sorted(resp_lens)[len(resp_lens) // 2] if resp_lens else 0
            ),
            "out_path": str(out_path),
        }
        print(f"[pid={pid}] wrote {len(samples)} samples -> {out_path}")
        print(f"    sessions covered: {len(sess_ids)}")
        print(f"    history_messages per sample: "
              f"min={grand_stats[pid]['hist_msgs_min']} "
              f"max={grand_stats[pid]['hist_msgs_max']} "
              f"mean={grand_stats[pid]['hist_msgs_mean']:.1f}")
        print(f"    n_prior_sessions mean: {grand_stats[pid]['n_prior_sess_mean']:.2f}")
        print(f"    user_response chars p50: {grand_stats[pid]['user_resp_chars_p50']}")

    # Dump a combined stats file for bookkeeping.
    stats_path = args.out_dir / (
        f"opd_{args.version}_stats_k{args.context_window}.json"
    )
    with stats_path.open("w", encoding="utf-8") as fh:
        json.dump(grand_stats, fh, indent=2)
    print(f"\nstats -> {stats_path}")


if __name__ == "__main__":
    main()
