"""PersonaMem-v1 loading and parsing utilities.

Single source of truth for:
  - reading shared_contexts_{32k,128k,1M}.jsonl
  - reading questions_{32k,128k,1M}.csv
  - splitting a shared context's message list into sessions
    (session boundary = system message index)
  - extracting persona demographics (first system message content)
  - canonicalizing the 7 question types across versions

All paths resolve relative to the repo root so callers don't need cwd setup.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "personamem_v1"

VERSIONS = ("32k", "128k", "1M")


# The raw CSVs use inconsistent strings across versions for the same 7 types.
# Observed variants (from inspect_mapping.py):
#   32k:  track_full_preference_evolution, recall_user_shared_facts,
#         recalling_the_reasons_behind_previous_updates, suggest_new_ideas,
#         generalizing_to_new_scenarios, provide_preference_aligned_recommendations,
#         recalling_facts_mentioned_by_the_user
#   128k: adds acknowledge_latest_user_preferences; uses revisit_reasons_behind_preference_updates,
#         generalize_to_new_scenarios, recall_user_shared_facts
#   1M:   track_full_preference_updates (note: 'updates' not 'evolution')
#
# We collapse to the 7 canonical names from plan §2.4.
QTYPE_CANONICAL: dict[str, str] = {
    # Type 1 — recall static facts/events
    "recall_user_shared_facts": "recall_facts",
    "recalling_facts_mentioned_by_the_user": "recall_facts",
    # Type 2 — suggest new items
    "suggest_new_ideas": "suggest_new",
    # Type 3 — acknowledge latest preference
    "acknowledge_latest_user_preferences": "acknowledge_latest",
    # Type 4 — track preference evolution
    "track_full_preference_evolution": "track_evolution",
    "track_full_preference_updates": "track_evolution",
    # Type 5 — reasons behind updates
    "revisit_reasons_behind_preference_updates": "reasons_behind",
    "recalling_the_reasons_behind_previous_updates": "reasons_behind",
    # Type 6 — preference-aligned recommendations
    "provide_preference_aligned_recommendations": "aligned_rec",
    # Type 7 — generalize to new scenarios
    "generalize_to_new_scenarios": "generalize",
    "generalizing_to_new_scenarios": "generalize",
}

CANONICAL_QTYPES = (
    "recall_facts",
    "suggest_new",
    "acknowledge_latest",
    "track_evolution",
    "reasons_behind",
    "aligned_rec",
    "generalize",
)


@dataclass
class Session:
    """A single conversation session inside a shared_context.

    persona_id:    '0'..'19'
    context_id:    sha hash from shared_contexts jsonl
    session_idx:   0-based position within its parent context
    messages:      list of {role, content} dicts; starts with a system
                   message (persona card), followed by user/assistant turns
    """

    persona_id: str
    context_id: str
    session_idx: int
    messages: list[dict]

    @property
    def demographics(self) -> str:
        """First system message content — the persona card text."""
        assert self.messages and self.messages[0]["role"] == "system", (
            f"session {self.persona_id}/{self.context_id}/{self.session_idx} "
            f"doesn't start with system"
        )
        return self.messages[0]["content"]

    @property
    def body(self) -> list[dict]:
        """Messages after the system card (user/assistant turns)."""
        return self.messages[1:]


def load_contexts(version: str) -> dict[str, list[dict]]:
    """ctx_id -> list of {role, content} messages."""
    assert version in VERSIONS
    out: dict[str, list[dict]] = {}
    with (DATA / f"shared_contexts_{version}.jsonl").open("r", encoding="utf-8") as fh:
        for line in fh:
            out.update(json.loads(line))
    return out


def load_questions(version: str) -> list[dict]:
    assert version in VERSIONS
    # Drop rows with negative end_index (observed only in 128k, ~44 rows).
    # These appear malformed — not safe for eval context slicing.
    path = DATA / f"questions_{version}.csv"
    rows: list[dict] = []
    dropped = 0
    with path.open("r", encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                if int(r["end_index_in_shared_context"]) < 0:
                    dropped += 1
                    continue
            except (ValueError, KeyError):
                dropped += 1
                continue
            r["qtype_canonical"] = QTYPE_CANONICAL.get(r["question_type"], None)
            if r["qtype_canonical"] is None:
                raise ValueError(f"unknown question_type: {r['question_type']!r}")
            rows.append(r)
    if dropped:
        print(f"[load_questions/{version}] dropped {dropped} rows with bad end_index")
    return rows


def split_sessions(messages: list[dict]) -> list[list[dict]]:
    """Split a shared_context's message list into sessions.

    A session starts at each system message. Every shared_context begins
    with role='system' at index 0, so the first session is [0 .. next_system).
    """
    if not messages or messages[0]["role"] != "system":
        raise ValueError("expected first message to be role=system")
    sys_indices = [i for i, m in enumerate(messages) if m["role"] == "system"]
    boundaries = sys_indices + [len(messages)]
    sessions: list[list[dict]] = []
    for a, b in zip(boundaries, boundaries[1:]):
        sessions.append(messages[a:b])
    return sessions


def _persona_id_from_session(session_msgs: list[dict], ctx_id: str,
                             ctx_to_pid: dict[str, str]) -> str:
    return ctx_to_pid[ctx_id]


def build_persona_sessions(version: str) -> dict[str, list[Session]]:
    """Return {persona_id: [Session, ...]} for all sessions in a version.

    Sessions from different shared_contexts for the same persona are
    concatenated (each context's sessions kept in their original order).
    Ordering across contexts is ctx_id string order — stable but arbitrary;
    callers that need temporal ordering should treat each (persona, ctx_id)
    as an independent timeline via session_idx alone.
    """
    questions = load_questions(version)
    ctx_to_pid: dict[str, str] = {}
    for r in questions:
        ctx_to_pid.setdefault(r["shared_context_id"], r["persona_id"])

    contexts = load_contexts(version)
    out: dict[str, list[Session]] = {}
    for ctx_id in sorted(contexts.keys()):
        if ctx_id not in ctx_to_pid:
            continue  # context not referenced by any question — skip
        pid = ctx_to_pid[ctx_id]
        sessions_msgs = split_sessions(contexts[ctx_id])
        for idx, msgs in enumerate(sessions_msgs):
            out.setdefault(pid, []).append(
                Session(persona_id=pid, context_id=ctx_id,
                        session_idx=idx, messages=msgs)
            )
    return out


def build_context_timelines(version: str) -> dict[tuple[str, str], list[Session]]:
    """Return {(persona_id, context_id): [Session, ...]} per-context timeline.

    Use this when doing progressive context accumulation: each context is
    an independent timeline (sessions within one ctx share chronology;
    across ctxs they don't).
    """
    questions = load_questions(version)
    ctx_to_pid = {r["shared_context_id"]: r["persona_id"] for r in questions}
    contexts = load_contexts(version)
    out: dict[tuple[str, str], list[Session]] = {}
    for ctx_id, msgs in contexts.items():
        if ctx_id not in ctx_to_pid:
            continue
        pid = ctx_to_pid[ctx_id]
        sessions = split_sessions(msgs)
        out[(pid, ctx_id)] = [
            Session(pid, ctx_id, i, s) for i, s in enumerate(sessions)
        ]
    return out


def strip_role_prefix(content: str, role: str) -> str:
    """Raw messages have 'User: ' / 'Assistant: ' prefixes even though role is
    already set. Strip them for cleaner text going into the model."""
    if role == "user" and content.startswith("User:"):
        return content[len("User:"):].lstrip()
    if role == "assistant" and content.startswith("Assistant:"):
        return content[len("Assistant:"):].lstrip()
    return content


if __name__ == "__main__":
    # Smoke test
    for v in VERSIONS:
        timelines = build_context_timelines(v)
        n_sess = sum(len(t) for t in timelines.values())
        sess_per_ctx = [len(t) for t in timelines.values()]
        print(f"{v}: {len(timelines)} contexts, {n_sess} total sessions, "
              f"sess/ctx min={min(sess_per_ctx)} max={max(sess_per_ctx)} "
              f"mean={sum(sess_per_ctx)/len(sess_per_ctx):.1f}")

        # Count sessions per persona
        per_pid: dict[str, int] = {}
        for (pid, _), sessions in timelines.items():
            per_pid[pid] = per_pid.get(pid, 0) + len(sessions)
        counts = sorted(per_pid.values())
        print(f"    sessions/persona: min={counts[0]} max={counts[-1]} "
              f"median={counts[len(counts)//2]}")

        # Show demographics of persona 0 in this version
        p0 = next(s for (pid, _), ss in timelines.items() if pid == "0" for s in ss)
        demo = p0.demographics[:200].replace("\n", " ")
        print(f"    persona 0 demographics preview: {demo!r}")
