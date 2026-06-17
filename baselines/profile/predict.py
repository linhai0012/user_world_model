"""profile — inject the persona card (demographics) only (the `+profile` arm, §8.2).

The structured-profile condition: what the model can do with explicit demographics but NO
conversational history/memory. Anchor for protocol validation (legacy base-with-demographics
≈ 0.34 on 32k).
"""
from __future__ import annotations

from common.data import MCQ, PersonaMem

METHOD = "profile"


def build_context(mcq: MCQ, data: PersonaMem, params: dict) -> list[dict]:
    return data.card_msgs(mcq.ctx_id)
