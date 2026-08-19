"""Per-user SFT data — the `+per-user weights` arm (project_summary.md §8.2, §5).

Builds (persona_card + preceding turn → THIS user's next turn) samples from a persona's own
PersonaMem conversations, and trains a per-user LoRA to PRODUCE that user's turns (the UserSim
framing: the model plays the user). No teacher needed — direct CE on the ground-truth user
turns (the "SFT backbone" of §5). The adapter encodes the user into weights; at eval it scores
MCQ options as user-continuations (MCQ-PPL), demographics-only context — matching the
`profile` baseline's input, so the only change vs baseline is the weights.

Format (matches the MCQ-PPL eval exactly so train/eval are aligned):
    messages = [system: persona_card, user: <preceding chatbot/context turn>]
    target   = assistant: <the user's actual next turn>          # CE on these tokens only
"""
from __future__ import annotations

from domains.general.data import PersonaMem


def build_user_samples(bench: str, persona_id: str, max_samples: int | None = None) -> list[dict]:
    """Return [{messages:[...], target:str}] for one persona, across all its contexts."""
    data = PersonaMem(bench)
    # which contexts belong to this persona (from the questions' ctx→persona map)
    ctx_ids = sorted({m.ctx_id for m in data.load_mcqs() if m.persona_id == persona_id})
    samples: list[dict] = []
    for ctx_id in ctx_ids:
        card = data.demographics(ctx_id)
        turns = [{"role": t["role"], "content": data._clean(t["content"], t["role"])}
                 for t in data.contexts[ctx_id] if t["role"] != "system"]
        for i in range(1, len(turns)):
            if turns[i]["role"] != "user":
                continue
            prev = turns[i - 1]["content"]
            tgt = turns[i]["content"]
            if not tgt.strip():
                continue
            msgs = [{"role": "system", "content": card}] if card else []
            msgs.append({"role": "user", "content": prev})
            samples.append({"messages": msgs, "target": tgt})
            if max_samples and len(samples) >= max_samples:
                return samples
    return samples
