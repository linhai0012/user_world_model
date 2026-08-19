"""oracle — inject the golden reference turns (the UPPER reference / `+memory` ceiling, §8.2).

The answer-bearing turns are present; this measures "given the relevant memory, can the model
use it?" `params`:
  mode: "slice" | "full"   # slice = ±window turns around the ref; full = whole history to ref
  window: int = 2
  with_profile: bool = False   # also prepend the persona card
  max_turns: int | None    # cap for mode=full
"""
from __future__ import annotations

from domains.general.data import MCQ, PersonaMem

METHOD = "oracle"


def build_context(mcq: MCQ, data: PersonaMem, params: dict) -> list[dict]:
    if params.get("mode", "slice") == "full":
        mem = data.full_msgs(mcq.ctx_id, mcq.end_index, max_turns=params.get("max_turns"))
    else:
        mem = data.slice_msgs(mcq.ctx_id, mcq.end_index, mcq.dist_ref_tokens,
                              window=params.get("window", 2))
    if params.get("with_profile"):
        mem = data.card_msgs(mcq.ctx_id) + mem
    return mem
