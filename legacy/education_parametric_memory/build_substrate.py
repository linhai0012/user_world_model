"""Build the shared experimental substrate (the A-vs-B fairness contract).

Consumes the offline question bank; emits everything both the parametric (ours) and agentic
(collaborator's) arms must replay identically:

  substrate/persona_set.json     N personas with hidden theta (the oracle)
  substrate/splits.json          per-skill train / eval / calib question ids (eval+calib HELD OUT)
  substrate/round_sequence.json  per-persona fixed question order (replayed by both arms)
  substrate/streams.jsonl        per (persona, round) the persona's answer = the SFT training trace
  substrate/eval_truth.jsonl     per (persona, snapshot, eval_q) the persona's TRUE answer = NLL/Brier ground truth

All deterministic (personas.det_seed). No LLM, no GPU.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import os

from personas import Persona, PersonaState, make_persona_set
from streams import build_round_sequence, simulate_stream, eval_at_snapshot, gate

SNAPSHOTS = [int(x) for x in os.environ.get("CUE_SNAPSHOTS", "0,5,10,20,40").split(",")]
N_PERSONAS = int(os.environ.get("CUE_N_PERSONAS", "24"))
SEED = int(os.environ.get("CUE_SEED", "7"))
TRAIN_PER_SKILL = 8
EVAL_PER_SKILL = 3
ROUND_LEN = int(os.environ.get("CUE_ROUND_LEN", "40"))


def load_bank(path: Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def split_bank(bank: list[dict]) -> dict[str, list[str]]:
    by_skill: dict[str, list[dict]] = {}
    for q in bank:
        by_skill.setdefault(q["skill_id"], []).append(q)
    train, evalp, calib = [], [], []
    for sk, qs in by_skill.items():
        qs = sorted(qs, key=lambda q: q["_id"])  # stable
        train += [q["_id"] for q in qs[:TRAIN_PER_SKILL]]
        evalp += [q["_id"] for q in qs[TRAIN_PER_SKILL:TRAIN_PER_SKILL + EVAL_PER_SKILL]]
        calib += [q["_id"] for q in qs[TRAIN_PER_SKILL + EVAL_PER_SKILL:]]
    return {"train": train, "eval": evalp, "calib": calib}


def persona_to_dict(p: Persona) -> dict:
    return {"pid": p.pid, "ability": p.ability, "topic_offset": p.topic_offset,
            "skill_jitter": p.skill_jitter,
            "held": {s: {"tag": t, "strength": st} for s, (t, st) in p.held.items()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path, default=Path(__file__).parent / "question_bank" / "biology_gcse.jsonl")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "substrate")
    ap.add_argument("--min-disc", type=float, default=0.2)
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    bank = load_bank(args.bank)
    q_by_id = {q["_id"]: q for q in bank}
    skills = sorted({q["skill_id"] for q in bank})

    personas = make_persona_set(bank, n=N_PERSONAS, seed=SEED)

    # discrimination gate: drop degenerate items (no theta signal)
    kept, dropped = gate(bank, personas, skills, seed=SEED, min_disc=args.min_disc)
    kept_ids = {q["_id"] for q in kept}
    kept_bank = [q for q in bank if q["_id"] in kept_ids]
    disc_map = {q["_id"]: q["_discrimination"] for q in kept}

    splits = split_bank(kept_bank)

    rounds = {p.pid: {"subject_id": "biology_gcse",
                      "rounds": build_round_sequence(p, splits["train"], length=ROUND_LEN, seed=SEED)}
              for p in personas}

    stream_rows, eval_rows = [], []
    for p in personas:
        seq = rounds[p.pid]["rounds"]
        for r in simulate_stream(PersonaState.initial(p, skills), seq, q_by_id, seed=SEED):
            q = q_by_id[r["question_id"]]
            opt_text = next((o["text"] for o in q["options"] if o["id"] == r.get("selected_option_id")), "")
            stream_rows.append({"persona": p.pid, "round": r["round"], "question_id": r["question_id"],
                                "skill_id": r["skill_id"], "selected_option_id": r.get("selected_option_id"),
                                "is_correct": r["is_correct"], "answer_text": opt_text})
        held = splits["eval"] + splits["calib"]   # truth for both: eval (report) + calib (fit isotonic)
        for snap in SNAPSHOTS:
            for r in eval_at_snapshot(p, skills, seq, q_by_id, held, snap, SEED):
                eval_rows.append({"persona": p.pid, **r})

    (out / "persona_set.json").write_text(json.dumps(
        {"persona_set": "cue-personas-v1", "seed": SEED, "snapshots": SNAPSHOTS,
         "personas": [persona_to_dict(p) for p in personas]}, indent=2))
    (out / "splits.json").write_text(json.dumps(
        {**splits, "discrimination": disc_map, "n_dropped_degenerate": len(dropped)}, indent=2))
    (out / "round_sequence.json").write_text(json.dumps(
        {"persona_set": "cue-personas-v1", "seed": SEED, "personas": rounds}, indent=2))
    with (out / "streams.jsonl").open("w") as fh:
        for r in stream_rows:
            fh.write(json.dumps(r) + "\n")
    with (out / "eval_truth.jsonl").open("w") as fh:
        for r in eval_rows:
            fh.write(json.dumps(r) + "\n")

    print(json.dumps({
        "personas": len(personas),
        "bank_total": len(bank), "kept_after_gate": len(kept_bank), "dropped_degenerate": len(dropped),
        "splits": {k: len(v) for k, v in splits.items()},
        "stream_rows": len(stream_rows), "eval_truth_rows": len(eval_rows),
        "out": str(out),
    }, indent=2))


if __name__ == "__main__":
    main()
