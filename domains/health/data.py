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
import re
import statistics
from dataclasses import dataclass, field
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
    event_text: str = ""       # NL description of the activity (GPT-synth, grounded)
    reaction_text: str = ""    # first-person body-feel reaction (GPT-synth) — text head


_REACTION_RE = re.compile(r"<text_start>(.*?)<text_end>", re.S)


def _extract_reaction(output_text: str | None) -> str:
    m = _REACTION_RE.search(output_text or "")
    return m.group(1).strip() if m else ""


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
            event = ""
            m = re.search(r"### Event:\s*(.*?)\s*### ", r.get("input_text", ""), re.S)
            if m:
                event = m.group(1).strip()
            out.append(HealthRecord(pid=r.get("participant_id", "?"),
                                    activity=r.get("activity_name", "activity"),
                                    duration_min=dur, state_n=sn, state_n1=sn1,
                                    event_text=event,
                                    reaction_text=_extract_reaction(r.get("output_text"))))
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


# --- shared prompt / target / parse (used by BOTH eval and per-user training, so
#     train and eval are token-for-token aligned; CONVENTIONS §3) ---

CONDS = ["persistence", "pop-mean", "base", "+current", "+profile", "+current+prof"]

SYS = ("You predict a specific user's NEXT-day wellness self-report after an activity. "
       "Wellness fields are integers: fatigue 1-5, mood 1-5, readiness 1-10, sleep_quality 1-5, "
       "soreness 1-5 (1=very sore,5=none), stress 1-5 (1=very stressed,5=very relaxed). "
       "Output ONLY a JSON object with exactly these 6 integer fields.")


def build_prompt(rec: "HealthRecord", cond: str, baseline: dict) -> str:
    """The user-turn for predicting next-day state. `cond` selects which conditioning
    blocks are injected (the Stage-1 ablation). `base` = activity only; the per-user
    arm trains/evals under `base` (the user lives in the LoRA weights, not the prompt)."""
    lines = [f"Activity: {rec.activity} for {rec.duration_min:.0f} min."]
    if cond in ("+profile", "+current+prof"):
        lines.append(f"This user's typical wellness baseline: {render_state(baseline)}.")
    if cond in ("+current", "+current+prof"):
        lines.append(f"Today's wellness (before tomorrow): {render_state(rec.state_n)}.")
    lines.append("Predict tomorrow's wellness as JSON "
                 '{"fatigue":_,"mood":_,"readiness":_,"sleep_quality":_,"soreness":_,"stress":_}.')
    return "\n".join(lines)


def target_json(state: dict[str, int]) -> str:
    """Canonical next-state JSON string — the CE target for the per-user arm."""
    return json.dumps({f: int(state[f]) for f in FIELDS})


# --- reaction-text head (the second UWM output head, project_summary §2) ---

SYS_REACTION = ("You are a specific user describing, in the first person, how you felt during and "
                "after a physical activity — consistent with your body and wellness.")


def build_reaction_prompt(rec: "HealthRecord", cond: str, baseline: dict) -> str:
    """User-turn for predicting the first-person reaction. Same conditioning blocks as the state
    head, so the two heads share the base/+current/+profile ablation."""
    lines = [f"Activity: {rec.activity} for {rec.duration_min:.0f} min."]
    if rec.event_text:
        lines.append(rec.event_text)
    if cond in ("+profile", "+current+prof"):
        lines.append(f"Your typical wellness baseline: {render_state(baseline)}.")
    if cond in ("+current", "+current+prof"):
        lines.append(f"Today's wellness (before the activity): {render_state(rec.state_n)}.")
    lines.append("In one short first-person paragraph, describe how you felt during and after it.")
    return "\n".join(lines)


def parse_state(text: str, fallback: dict) -> dict:
    """Parse the first {...} JSON object from a generation into a clamped 6-field state;
    fall back per-field to `fallback` (the current state) on any malformation."""
    try:
        start = text.index("{")
        obj = json.loads(text[start:text.index("}", start) + 1])
        return {f: clamp(f, int(round(float(obj[f])))) if f in obj else fallback[f]
                for f in FIELDS}
    except (ValueError, KeyError, TypeError):
        return dict(fallback)
