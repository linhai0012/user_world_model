# Shared experimental substrate — the A-vs-B fairness contract

This directory is the **held-constant interaction stream** both the parametric (ours) and
agentic (collaborator's) arms must run on, so the headline comparison is apples-to-apples
(fixes review B1). One side owns it (us — we have the persona generator); the other consumes it.

## Files (all deterministic, `personas.det_seed`)

| File | What it is | Both arms must… |
| --- | --- | --- |
| `persona_set.json` | N=24 personas with hidden θ (ability + per-topic offset + per-skill jitter + held misconceptions). The oracle ground truth. | use the SAME personas |
| `splits.json` | per-skill `train` (104) / `eval` (39, HELD OUT, scored) / `calib` (30, HELD OUT, isotonic) question ids + per-item discrimination | score on the SAME eval; fit calibration on the SAME calib |
| `round_sequence.json` | per-persona fixed question order over the train pool (seed 7) | **replay the SAME order per persona** |
| `streams.jsonl` | per (persona, round) the persona's answer = `{question_id, selected_option_id, is_correct, answer_text}` — the canonical interaction trace | `update()` the SAME graded rounds (don't re-generate answers) |
| `eval_truth.jsonl` | per (persona, snapshot, held-out q) the persona's TRUE answer = the NLL/Brier ground truth | score predictions against the SAME truth |

Questions live in `../question_bank/biology_gcse.jsonl` (173 MCQs, content-stable ids
`{skill_id}#g{md5}`). Snapshots: `[0,5,10,20,40]`.

## What the agentic arm returns (so A and B are row-aligned)

One `predictions.jsonl` in the **same schema** the parametric bundle uses
(`../../docs/backend/parametric-bundle-schema.md`), one row per `(learner, snapshot, eval_q)`:

```jsonc
{ "learner_id": "persona_07", "subject_id": "biology_gcse", "snapshot": 20,
  "question_id": "biology_gcse__cell_biology__osmosis#g1a2b3c", "format": "mcq",
  "p_correct_raw": <agentic raw estimate>,          // pre-calibration
  "p_correct": <calibrated>,                          // same isotonic METHOD, fit on calib split
  "option_logprobs": { "a": .., "b": .., "c": .., "d": .. },  // or per-skill p replicated per option
  "scope": "agentic" }
```

Plus, for the cost axis (§7), the agentic per-round LLM-call count + latency + model id.

## The contract points that bite if skipped

1. **Same persona_set + round_sequence + streams** — if agentic runs on its own personas/order,
   the comparison is apples-to-oranges. Consume these files; don't regenerate.
2. **Per-question, not per-skill.** `agentic.py::predict_correct` currently estimates per *skill*;
   emit a `p_correct` per *eval question* (replicating the skill value across that skill's items is
   fine — but the rows must be per-question to align with parametric's per-item predictions).
3. **One shared calibration METHOD** (isotonic on the held-out calib split) applied to each arm's
   own raw signal — so NLL/Brier are comparable (fixes B3).
4. **Same grader / same outcomes.** The `is_correct` ground truth is in `eval_truth.jsonl`
   (θ-driven). Both arms are scored against it; neither grades by simulated tone.

## How the comparison is computed (both arms)

For each `(persona P, snapshot S)`: NLL/Brier of the arm's `p_correct` on the EVAL items vs
`eval_truth[P,S]`. Aggregate across personas → the A1 (per-user) / A0 (shared) / A∅ (base) /
agentic curves over snapshots. See `../sft/package_bundle.py` for the parametric side's exact
metric code; mirror it.
