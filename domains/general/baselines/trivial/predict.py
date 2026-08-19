"""trivial — no memory (the LOWER reference / `base` arm, §8.2). Also the leak/floor check.

Returns no context messages; the backend scores each option as a continuation of just the
query. `build_context(mcq, data, params) -> list[dict]` (the shared message-list contract).
"""
from __future__ import annotations

from domains.general.data import MCQ, PersonaMem

METHOD = "trivial"


def build_context(mcq: MCQ, data: PersonaMem, params: dict) -> list[dict]:
    return []
