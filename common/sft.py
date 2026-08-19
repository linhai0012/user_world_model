"""Domain-agnostic SFT tokenization — shared by every domain's per-user-weights arm.

One generic `{messages, target}` → `{input_ids, labels}` converter, so general / health /
education all train on the *same* loss shape (CE on the target span only) and any difference
between domains comes from the data, not the tokenizer. Each domain builds its own samples
(`domains/<domain>/peruser_data.py`); this is the only piece they share.
"""
from __future__ import annotations


def tokenize_sample(tok, sample: dict, max_len: int = 4096) -> dict | None:
    """Chat-format the sample, return {input_ids, labels} with labels masked to the target span.
    The target is the assistant turn (in UserSim framing, the user's own next utterance)."""
    base = sample["messages"]
    prefix = tok.apply_chat_template(base, tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(
        base + [{"role": "assistant", "content": sample["target"]}],
        tokenize=False, add_generation_prompt=False)
    pre_ids = tok(prefix, add_special_tokens=False)["input_ids"]
    full_ids = tok(full, add_special_tokens=False)["input_ids"]
    # common-prefix boundary (robust to template quirks)
    b = 0
    for x, y in zip(pre_ids, full_ids):
        if x != y:
            break
        b += 1
    if len(full_ids) > max_len:                 # keep the target; drop oldest prompt tokens
        cut = len(full_ids) - max_len
        full_ids = full_ids[cut:]
        b = max(0, b - cut)
    if b >= len(full_ids):
        return None
    labels = [-100] * b + full_ids[b:]
    return {"input_ids": full_ids, "labels": labels}
