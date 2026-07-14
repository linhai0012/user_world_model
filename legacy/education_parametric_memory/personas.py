"""Persona theta (the hidden ground truth) + the controllable latent->answer generator.

This is the oracle we author: each persona is a low-dimensional trait vector (IRT-style:
overall ability + per-topic offset + per-skill jitter) plus a misconception profile. Mastery
governs the correct-rate; the misconception profile governs WHICH distractor is chosen when
wrong — the learner-discriminative signal the per-user LoRA must reconstruct (and that the base
model cannot leak). Theta is hidden from the model but known to us, so we can score recovery.

The generator is **controllable / near-deterministic given (theta, seed)** (decision D-a): no
LLM, no GPU, fully reproducible. It improves on simulate.py, which picks a *uniform random*
wrong option and throws the misconception signal away.
"""
from __future__ import annotations

import hashlib
import math
import os
import random
from dataclasses import dataclass, field


def det_seed(*parts: object) -> int:
    """Deterministic 32-bit seed from arbitrary parts. Uses md5, NOT builtin hash() — Python
    randomises str hashing per process (PYTHONHASHSEED), which would make round_sequence
    irreproducible across machines/runs and break the cross-team replay contract."""
    digest = hashlib.md5("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(digest[:8], 16)

# --- trait hyperparameters (the persona_set generative model) ---
# Env-overridable: a more-spread population (larger SDs) un-stacks the 0.5-centered base baseline
# and is arguably more realistic for GCSE learners (v4 robustness config).
_ABILITY_SD = float(os.environ.get("CUE_ABILITY_SD", "0.85"))
_TOPIC_SD = float(os.environ.get("CUE_TOPIC_SD", "0.55"))
_SKILL_SD = float(os.environ.get("CUE_SKILL_SD", "0.35"))
_LEARN_RATE = 0.08          # mastery gained per practice of a skill
_REPAIR_AT = 0.70           # once mastery crosses this, a held misconception starts to fade
_REPAIR_DECAY = 0.5         # multiplicative fade of misconception strength on repair
_WRONG_BIAS = 0.70          # share of error mass on the held-misconception distractor


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _topic_of(skill_id: str) -> str:
    return "__".join(skill_id.split("__")[:-1])


# --------------------------------------------------------------------- theta
@dataclass
class Persona:
    """Static traits — the hidden theta. State that evolves over a stream lives in PersonaState."""
    pid: str
    ability: float
    topic_offset: dict[str, float]
    skill_jitter: dict[str, float]
    held: dict[str, tuple[str, float]]   # skill_id -> (misconception_tag, base_strength)

    def base_mastery(self, skill_id: str) -> float:
        return _sigmoid(self.ability
                        + self.topic_offset.get(_topic_of(skill_id), 0.0)
                        + self.skill_jitter.get(skill_id, 0.0))


@dataclass
class PersonaState:
    """A persona at a point in its learning trajectory (mutated as it practises skills)."""
    persona: Persona
    mastery: dict[str, float]
    misc_strength: dict[str, float] = field(default_factory=dict)  # skill_id -> current strength

    @classmethod
    def initial(cls, p: Persona, skills: list[str]) -> "PersonaState":
        return cls(
            persona=p,
            mastery={s: p.base_mastery(s) for s in skills},
            misc_strength={s: stg for s, (_tag, stg) in p.held.items()},
        )

    def mastery_of(self, skill_id: str) -> float:
        return self.mastery.get(skill_id, self.persona.base_mastery(skill_id))

    def misconception_for(self, skill_id: str) -> str | None:
        """The misconception tag this persona still actively holds for the skill (None if repaired)."""
        if self.misc_strength.get(skill_id, 0.0) <= 0.05:
            return None
        held = self.persona.held.get(skill_id)
        return held[0] if held else None

    def practice(self, skill_id: str) -> None:
        """One practice of a skill: mastery rises; a held misconception fades once mastery is high."""
        m = self.mastery_of(skill_id)
        self.mastery[skill_id] = min(0.98, m + _LEARN_RATE * (1.0 - m))
        if self.mastery[skill_id] >= _REPAIR_AT and skill_id in self.misc_strength:
            self.misc_strength[skill_id] *= _REPAIR_DECAY


# --------------------------------------------------------------------- sampling
def skill_misconceptions(questions: list[dict]) -> dict[str, list[str]]:
    """Per skill, the misconception tags available in its MCQ distractors / short-answer notes."""
    out: dict[str, set[str]] = {}
    for q in questions:
        s = q["skill_id"]
        tags = out.setdefault(s, set())
        for o in q.get("options", []):
            if not o.get("is_correct") and o.get("misconception_tag"):
                tags.add(o["misconception_tag"])
        for mc in q.get("misconceptions", []):
            tags.add(mc)
    return {s: sorted(t) for s, t in out.items()}


def make_persona_set(questions: list[dict], n: int = 24, seed: int = 7) -> list[Persona]:
    """Sample N personas from the trait model. Deterministic given seed."""
    skills = sorted({q["skill_id"] for q in questions})
    topics = sorted({_topic_of(s) for s in skills})
    tags_by_skill = skill_misconceptions(questions)

    personas: list[Persona] = []
    for i in range(n):
        rng = random.Random(seed * 1000 + i)
        ability = rng.gauss(0.0, _ABILITY_SD)
        topic_offset = {t: rng.gauss(0.0, _TOPIC_SD) for t in topics}
        skill_jitter = {s: rng.gauss(0.0, _SKILL_SD) for s in skills}
        p = Persona(pid=f"persona_{i:02d}", ability=ability,
                    topic_offset=topic_offset, skill_jitter=skill_jitter, held={})
        # Assign held misconceptions: more likely where this persona is weak.
        for s in skills:
            avail = tags_by_skill.get(s, [])
            if not avail:
                continue
            m = p.base_mastery(s)
            p_hold = max(0.0, min(0.9, 0.9 * (1.0 - m)))
            if rng.random() < p_hold:
                tag = rng.choice(avail)
                strength = round(rng.uniform(0.55, 0.9), 3)
                p.held[s] = (tag, strength)
        personas.append(p)
    return personas


# --------------------------------------------------------------------- generator g(theta, q)
def answer_mcq(state: PersonaState, q: dict, rng: random.Random) -> dict:
    """Controllable MCQ answer: correct w.p. mastery; else bias toward the held-misconception option."""
    skill = q["skill_id"]
    m = state.mastery_of(skill)
    opts = q["options"]
    correct_id = next((o["id"] for o in opts if o.get("is_correct")), opts[0]["id"])
    if rng.random() < m:
        return {"selected_option_id": correct_id, "is_correct": True}

    wrong = [o for o in opts if not o.get("is_correct")]
    held_tag = state.misconception_for(skill)
    fav = next((o for o in wrong if o.get("misconception_tag") == held_tag), None)
    if fav is not None and rng.random() < _WRONG_BIAS:
        choice = fav["id"]
    else:
        choice = rng.choice([o["id"] for o in wrong]) if wrong else correct_id
    return {"selected_option_id": choice, "is_correct": False}


def answer_short(state: PersonaState, q: dict, rng: random.Random) -> dict:
    """Controllable short-answer: a templated rendering consistent with theta (text + is_correct).

    Ground-truth is_correct is theta-driven; the text is a controllable surface form (correct ->
    mark-scheme-aligned; wrong -> expresses a held misconception). The shared grader can still be
    run for realism, but the experiment's ground truth is theta, not the grader's read of tone.
    """
    skill = q["skill_id"]
    m = state.mastery_of(skill)
    if rng.random() < m:
        return {"answer_text": f"[correct] {q.get('mark_scheme', '')[:160]}", "is_correct": True}
    held_tag = state.misconception_for(skill)
    misc = held_tag or (q.get("misconceptions") or ["(no clear reasoning)"])[0]
    return {"answer_text": f"[misconception: {misc}]", "is_correct": False}


def answer(state: PersonaState, q: dict, rng: random.Random) -> dict:
    return answer_mcq(state, q, rng) if q.get("format") == "mcq" else answer_short(state, q, rng)
