"""Education-domain loader (private KCL Langfuse tutor chats) for Stage-1 next-student-turn
prediction — the education text/reaction head (project_summary.md §2, the edu-reaction head ≈
general's reply prediction).

Each JSONL record = one tutoring conversation; the dialogue lives in `input.messages`: a system
tutor-persona prompt + alternating assistant(=tutor) / user(=student) turns (a few tool turns,
dropped). **No userId / sessionId** in the data, so there is no cross-session learner identity:
the unit is the SESSION, and education runs the FROZEN base/+memory ablation only (no per-user
weights — no learner to personalize, 66 sessions). No exam data here → no structured state head.

Task: predict the STUDENT's next turn given the prior conversation.
  base    = system tutor-persona + the tutor's immediately preceding turn (minimal stimulus)
  memory  = system + the full prior dialogue (episodic memory of the session)
Target  = the student's actual next turn. Scored by NLL (lower = better) — the text-head analog
of the general-domain memory ablation.
"""
from __future__ import annotations

import json
from pathlib import Path

COURSES = {"nlp": "chat_nlp.jsonl", "ai": "chat_ai.jsonl"}
ROLES_KEEP = {"system", "user", "assistant"}


def _data_dir(data_dir: str | None = None) -> Path:
    if data_dir:
        return Path(data_dir)
    return Path(__file__).resolve().parents[1] / "data" / "education"


def load_edu_sessions(course: str, data_dir: str | None = None) -> list[list[dict]]:
    """Each session -> ordered [{role, content}] (system/user/assistant, non-empty; tool dropped)."""
    path = _data_dir(data_dir) / COURSES[course]
    sessions: list[list[dict]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            try:
                msgs = json.loads(r["input"])["messages"]
            except (KeyError, ValueError, TypeError):
                continue
            clean = [{"role": m["role"], "content": str(m.get("content") or "").strip()}
                     for m in msgs
                     if m.get("role") in ROLES_KEEP and str(m.get("content") or "").strip()]
            if any(m["role"] == "user" for m in clean):
                sessions.append(clean)
    return sessions


def build_edu_samples(course: str, cond: str = "memory", data_dir: str | None = None,
                      min_chars: int = 1) -> list[dict]:
    """[{messages, target, sess, turn}] — one per student(user) turn. Iteration is deterministic,
    so the `base` and `memory` builds are PAIRED (same targets in the same order)."""
    out: list[dict] = []
    for si, msgs in enumerate(load_edu_sessions(course, data_dir)):
        for i, m in enumerate(msgs):
            if m["role"] != "user":
                continue
            tgt = m["content"]
            if len(tgt) < min_chars:
                continue
            prior = msgs[:i]
            if not prior:
                continue
            system = [prior[0]] if prior[0]["role"] == "system" else []
            if cond == "base":
                last = [prior[-1]] if prior[-1]["role"] != "system" else []
                ctx = system + last
            else:  # memory
                ctx = prior
            out.append({"messages": ctx, "target": tgt, "sess": si, "turn": i})
    return out
