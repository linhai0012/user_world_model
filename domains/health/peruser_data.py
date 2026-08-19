"""Per-user SFT data for the HEALTH domain (the `+per-user weights` arm, project_summary §8.2).

Builds (base-context activity prompt -> next-day state JSON) samples from ONE participant's own
records, and trains a per-user LoRA to PRODUCE that user's next-state transitions. No teacher —
direct CE on the REAL ground-truth next state (the "SFT backbone" of §5; the next state is the
genuine PMData self-report, not synthetic).

Context = `base` (activity only): the user's dynamics live in the LoRA *weights*, so the eval
comparison is per-user-weights vs frozen-base under identical (minimal) context — the clean
"do per-user weights add value over a frozen base" test (mirrors the general-domain per-user arm).

Format (matches eval_health_stage1.py token-for-token via the shared build_prompt/target_json,
so train and eval are aligned; CONVENTIONS §3):
    messages = [system: SYS, user: build_prompt(rec, "base", baseline)]
    target   = assistant: target_json(rec.state_n1)        # CE on these tokens only
"""
from __future__ import annotations

from domains.health.data import (SYS, build_prompt, load_records,
                                participant_baselines, target_json)
from common.sft import tokenize_sample

__all__ = ["build_health_user_samples", "tokenize_sample", "participant_record_counts",
           "cond_tag"]


def cond_tag(cond: str) -> str:
    """Filesystem-safe tag for a condition, used in adapter dir names (one set per cond)."""
    return cond.replace("+", "").strip() or "base"


def participant_record_counts(split: str = "train") -> dict[str, int]:
    """pid -> #records in the split (to pick data-rich participants for per-user training)."""
    counts: dict[str, int] = {}
    for r in load_records(split):
        counts[r.pid] = counts.get(r.pid, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def build_health_user_samples(pid: str, split: str = "train", cond: str = "base",
                              max_samples: int | None = None) -> list[dict]:
    """Return [{messages, target}] for ONE participant's records in `split`.
    `cond` defaults to `base` (activity-only context); baselines are train-derived so the
    `+profile` variant is also available without leaking the eval split."""
    baselines = participant_baselines(load_records("train"))
    base = baselines.get(pid, {})
    samples: list[dict] = []
    for r in load_records(split):
        if r.pid != pid:
            continue
        msgs = [{"role": "system", "content": SYS},
                {"role": "user", "content": build_prompt(r, cond, base)}]
        samples.append({"messages": msgs, "target": target_json(r.state_n1)})
        if max_samples and len(samples) >= max_samples:
            break
    return samples
