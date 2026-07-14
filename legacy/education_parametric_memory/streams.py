"""Round sequences, answer-stream simulation, snapshot states, held-out eval, discrimination gate.

This turns persona theta (personas.py) into the data the offline SFT harness consumes:
  - round_sequence : a fixed per-persona question order (the fairness contract, replayed by both
                     the offline harness and the live sim).
  - simulate_stream: the persona answers its round_sequence in order, LEARNING as it practises
                     (mastery rises, misconceptions repair) — the "starts weak, improves" arc the
                     snapshots {0,5,10,20,40} capture.
  - eval_at_snapshot: held-out items answered at the persona's state at round k = the ground
                     truth the LoRA (trained on rounds 0..k) must predict.
  - discrimination : the identifiability gate — keep only items whose answer varies with theta.

All deterministic given seeds. No LLM, no GPU.
"""
from __future__ import annotations

import math
import random
import statistics

from personas import Persona, PersonaState, answer, det_seed


def _skill_of(question_id: str) -> str:
    return question_id.rsplit("#", 1)[0]


# --------------------------------------------------------------------- round sequence
def build_round_sequence(persona: Persona, train_pool: list[str], length: int, seed: int) -> list[str]:
    """A fixed, reproducible question order for one persona. Tiles the pool (shuffled) to `length`,
    so a small bank still yields a long stream (revisits = spaced practice)."""
    rng = random.Random(det_seed(persona.pid, "seq", seed))
    out: list[str] = []
    while len(out) < length:
        block = train_pool[:]
        rng.shuffle(block)
        out.extend(block)
    return out[:length]


# --------------------------------------------------------------------- stream + snapshots
def simulate_stream(state: PersonaState, round_sequence: list[str], q_by_id: dict[str, dict],
                    seed: int) -> list[dict]:
    """Answer each round in order; practise the skill AFTER answering (so round k reflects state
    before that practice). Returns one record per round."""
    rng = random.Random(det_seed(state.persona.pid, "stream", seed))
    records: list[dict] = []
    for k, qid in enumerate(round_sequence):
        q = q_by_id[qid]
        a = answer(state, q, rng)
        records.append({"round": k, "question_id": qid, "skill_id": q["skill_id"],
                        "format": q.get("format"), **a})
        state.practice(q["skill_id"])
    return records


def eval_at_snapshot(persona: Persona, all_skills: list[str], round_sequence: list[str],
                     q_by_id: dict[str, dict], eval_ids: list[str], snapshot: int,
                     seed: int) -> list[dict]:
    """Replay the stream up to `snapshot` rounds, then answer the held-out eval items at that state.
    This is the per-(persona, snapshot) ground truth the bundle's predictions are scored against."""
    state = PersonaState.initial(persona, all_skills)
    rng = random.Random(det_seed(persona.pid, "stream", seed))
    for qid in round_sequence[:snapshot]:
        q = q_by_id[qid]
        answer(state, q, rng)          # advance state (answers discarded; we only want the state)
        state.practice(q["skill_id"])
    erng = random.Random(det_seed(persona.pid, "eval", snapshot, seed))
    out = []
    for qid in eval_ids:
        q = q_by_id[qid]
        a = answer(state, q, erng)
        out.append({"question_id": qid, "skill_id": q["skill_id"], "snapshot": snapshot,
                    "mastery": round(state.mastery_of(q["skill_id"]), 4), **a})
    return out


# --------------------------------------------------------------------- discrimination gate
def discrimination(q: dict, personas: list[Persona], all_skills: list[str], seed: int,
                   reps: int = 3) -> float:
    """Identifiability score for an item: how much its answer varies across the persona_set at
    their initial state. 0 = degenerate (everyone same -> no theta signal -> drop). For MCQ we use
    the entropy of the chosen-option distribution; for short-answer the variance of is_correct."""
    chosen: list[str] = []
    correct: list[float] = []
    for p in personas:
        st = PersonaState.initial(p, all_skills)
        rng = random.Random(det_seed(p.pid, q["_id"], seed))
        for _ in range(reps):
            a = answer(st, q, rng)
            correct.append(1.0 if a["is_correct"] else 0.0)
            chosen.append(a.get("selected_option_id", "1" if a["is_correct"] else "0"))
    if q.get("format") == "mcq":
        counts: dict[str, int] = {}
        for c in chosen:
            counts[c] = counts.get(c, 0) + 1
        total = sum(counts.values()) or 1
        ent = -sum((n / total) * math.log(n / total) for n in counts.values())
        return round(ent, 4)
    return round(statistics.pvariance(correct) if len(correct) > 1 else 0.0, 4)


def gate(questions: list[dict], personas: list[Persona], all_skills: list[str], seed: int,
         min_disc: float = 0.35) -> tuple[list[dict], list[dict]]:
    """Split a candidate bank into (kept, dropped) by discrimination >= min_disc."""
    kept, dropped = [], []
    for q in questions:
        d = discrimination(q, personas, all_skills, seed)
        (kept if d >= min_disc else dropped).append({**q, "_discrimination": d})
    return kept, dropped
