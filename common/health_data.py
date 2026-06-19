"""Health-domain loader (PMData digital-twin) for the Stage-1 intrinsic-prediction eval.

The UWM world-model target for health: given a user + an activity, predict the user's **next
wellness state** (project_summary §1/§6 — text reaction + next state; we take the structured
state here). Each record carries the current-day state (`wellness_day_n`) and the next-day
state (`wellness_day_n1`) plus the activity — i.e. a clean state-transition.

Data: `$UWM_DATA/health/digitaltwin/output/{train,val,test}.jsonl` (GPT-synthesized from PMData).
States are 6 ordinal fields. We keep everything in plain ints/text so a FROZEN base model can
predict the next state via generation (no custom state tokens, no training) — matching the
general-domain frozen-base ablation methodology.
"""
from __future__ import annotations

import ast
import json
import os
import statistics
from dataclasses import dataclass
from pathlib import Path

# (lo, hi) per field — verbatim from legacy/health_digitaltwin/config.py
WELLNESS_FIELDS = {
    "fatigue": (1, 5), "mood": (1, 5), "readiness": (1, 10),
    "sleep_quality": (1, 5), "soreness": (1, 5), "stress": (1, 5),
}
FIELDS = list(WELLNESS_FIELDS)


def _data_root(data_dir: str | None = None) -> Path:
    if data_dir:
        return Path(data_dir)
    base = os.environ.get("UWM_DATA")
    if not base:
        raise RuntimeError("UWM_DATA unset — set it (inf_elandi) first.")
    return Path(base) / "health" / "digitaltwin" / "output"


@dataclass
class HealthRecord:
    pid: str
    activity: str
    duration_min: float
    state_n: dict[str, int]    # current-day wellness (6 fields)
    state_n1: dict[str, int]   # next-day wellness (the prediction target)


def _parse_state(s: str) -> dict | None:
    try:
        d = ast.literal_eval(s) if isinstance(s, str) else s
        return {f: int(d[f]) for f in FIELDS if f in d} if isinstance(d, dict) else None
    except (ValueError, SyntaxError, TypeError, KeyError):
        return None


def load_records(split: str, data_dir: str | None = None) -> list[HealthRecord]:
    """Records with BOTH current and next state (the transition is defined)."""
    path = _data_root(data_dir) / f"{split}.jsonl"
    out: list[HealthRecord] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            sn, sn1 = _parse_state(r.get("wellness_day_n")), _parse_state(r.get("wellness_day_n1"))
            if not sn or not sn1 or len(sn) < 6 or len(sn1) < 6:
                continue
            try:
                dur = float(r.get("duration_min", 0) or 0)
            except ValueError:
                dur = 0.0
            out.append(HealthRecord(pid=r.get("participant_id", "?"),
                                    activity=r.get("activity_name", "activity"),
                                    duration_min=dur, state_n=sn, state_n1=sn1))
    return out


def participant_baselines(records: list[HealthRecord]) -> dict[str, dict[str, int]]:
    """Per-participant mean current-state (rounded) — the structured `profile` for `+profile`."""
    by_pid: dict[str, list[dict]] = {}
    for r in records:
        by_pid.setdefault(r.pid, []).append(r.state_n)
    base: dict[str, dict[str, int]] = {}
    for pid, states in by_pid.items():
        base[pid] = {f: round(statistics.mean(s[f] for s in states)) for f in FIELDS}
    return base


def render_state(state: dict[str, int]) -> str:
    return ", ".join(f"{f}={state[f]}" for f in FIELDS)


def clamp(field: str, v: int) -> int:
    lo, hi = WELLNESS_FIELDS[field]
    return max(lo, min(hi, v))
