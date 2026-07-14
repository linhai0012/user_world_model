# `parametric_offline/` — Mode-A bundle contract tooling (de-risk)

This directory is **Lin Hai's side** of Cue: the *contract tooling* for the parametric
(per-user LoRA) artifact bundle. It exists to **lock the cross-team contract before the real
GPU harness does**, per `ONBOARDING-LINHAI.md` §10 and `docs/backend/parametric-bundle-schema.md`.

> **Scope.** This holds the **sample bundle + importer + verification** only — NOT the GPU
> trainer. The real per-user/shared LoRA training + scoring runs offline on Lin's rig and is
> kept out of this repo (ONBOARDING §11–12); it only ever *produces* a bundle in this exact
> shape, which `import_bundle.py` then loads. Numbers in `sample_bundle/` are **synthetic**.

## Files

| File | What |
| --- | --- |
| `make_sample_bundle.py` | generates `sample_bundle/` (3 personas + A0 + A∅) over the **real** seeded `biology_gcse` ids, synthetic metrics |
| `sample_bundle/` | the 5-file bundle: `manifest.json` · `predictions.jsonl` · `curves.jsonl` · `adapters.jsonl` · `cost.json` |
| `import_bundle.py` | the **missing bridge**: bundle JSONL → Mongo (`parametric_replay` + `learner_adapters` + `parametric_curves` + `parametric_bundles`) |
| `personas.py` | the data layer: persona **theta** (low-dim IRT-style traits + misconception profile) + the **controllable generator** g(theta, q)→answer (decision D-a/D-b) |
| `streams.py` | `round_sequence` builder, answer-stream simulation (with learning dynamics), per-snapshot held-out eval, and the **discrimination gate** (identifiability filter) |
| (test) `backend/tests/test_offline_replay_bundle.py` | end-to-end proof: seed → import → `offline_replay` replays it, GPU-free |
| (test) `backend/tests/test_data_layer.py` | proves theta varies, wrong-answers follow the held misconception, the gate separates signal from degenerate items, streams learn + are reproducible |

### Data layer (the persona substrate)

`personas.py` + `streams.py` turn hidden persona theta into identifiable answer streams, the
input the SFT harness trains on. **Reproducibility is a contract requirement** — both the
offline harness and the live sim must replay the *same* `round_sequence`, so all seeding goes
through `personas.det_seed()` (md5-based), **never builtin `hash()`** (Python randomises str
hashing per process; using it makes round_sequence differ across machines). Verified
byte-identical across processes.

Decisions locked (D-a..D-d): controllable generator (no LLM), low-dim trait theta, biology-only
v1 with N=24 personas, target ~12 MCQ + 3 short per skill.

## The real SFT run (`sft/`, GPU) — DONE

The lean-SFT realisation: train a per-user dual-rate LoRA (Qwen3-4B-Instruct-2507) to produce
each learner's answers, score held-out items by choice-perplexity, package the real bundle.

- `sft/sft_core.py` — dual-rate LoRA SFT + choice-PPL scoring (sdpa; env `CUE_SKILL_PROMPT`).
- `sft/run_all.py` — A1 per-user / A0 shared / A∅ base → `predictions/` (resumable; env
  `CUE_SNAPSHOTS/EPOCHS/PRED_DIR`).
- `sft/package_bundle.py` — PAV isotonic calibration + 5-file bundle + headline (env `CUE_BUNDLE`).
- `sft/verify_bundle.py` — real bundle → live `offline_replay` honesty-gate check. **PASS** (v1, v4).
- `sft/analyze.py` / `diagnose.py` / `ceiling_test.py` — the 3-metric comparison, mastery-tracking,
  and the memorise-vs-transfer ceiling test.

Run config via env, e.g.: `CUE_PRED_DIR=predictions_v2 CUE_SNAPSHOTS=0,13,26,52,104 CUE_EPOCHS=3
vllm_env/bin/python sft/run_all.py`. Base model + outputs live on scratch
(`/scratch/prj/cllm/cue_sft/`), never in the repo.

**→ Results & findings: [`RESULTS.md`](RESULTS.md).** Headline: per-user parametric memory
recovers learner mastery that the shared LoRA and base do not (and beats base too under a powered
eval); lean SFT memorises but transfers weakly. Honesty gate passes on a real bundle.

## Run it

```bash
backend/.venv/bin/python parametric_offline/make_sample_bundle.py        # (re)generate the sample bundle
cd backend && .venv/bin/python -m pytest tests/test_offline_replay_bundle.py -v   # prove the contract
# against a live Mongo instead of the test mock:
backend/.venv/bin/python parametric_offline/import_bundle.py --bundle parametric_offline/sample_bundle
```

## The contract, as the code actually reads it

`offline_replay` (the live provider) reads the `parametric_replay` collection. The importer
absorbs three mismatches between the schema doc and that code so the bundle files keep the
documented shape:

1. **`learner_id` → `user_id`.** The schema names the key `learner_id`; the provider queries
   `user_id`. The importer maps it.
2. **Derived `skill_id`.** `offline_replay.mastery()` reads a `skill_id` field, but
   `predictions.jsonl` rows only carry `question_id`. The importer parses
   `skill_id = question_id.rsplit("#",1)[0]` onto every replay row.
3. **Controls import verbatim.** `__shared__` (A0) and `__base__` (A∅) rows load as-is; the
   provider's miss-policy falls back to `__shared__`, then to the stub.

## Known contract gaps (surfaced by this de-risk — decide before the real bundle)

- **🔴 `offline_replay` snapshot matching is exact, not nearest-≤ (live-app bug).** The chat
  loop does `parametric_snapshot += 1` per answer (1,2,3,…), but bundle snapshots are discrete
  `[0,5,10,20,40]`. `OfflineReplayProvider._snapshot()` returns the raw pointer and the queries
  do an **exact** `snapshot:` match → at pointer=7 it finds nothing and silently degrades to the
  stub. The schema doc says "nearest snapshot ≤ it" but that mapping is **not implemented**. The
  de-risk test sidesteps this by setting the session pointer to an exact snapshot; the **real
  running demo needs the fix below.** It touches the friend's live-app file, so it is left
  **unapplied** — Lin's call whether to apply/push.

  ```python
  # backend/app/services/usermodel/providers/offline_replay.py — proposed
  async def _snapshot(self, db, user_id, subject_id) -> int:
      sess = await db[SESSIONS].find_one({"user_id": user_id, "subject_id": subject_id})
      raw = int((sess or {}).get("parametric_snapshot", 0))
      snaps = await db[PARAMETRIC_BUNDLES].find_one({"subject_ids": subject_id}) or {}
      avail = sorted(snaps.get("snapshots", []))
      return max([s for s in avail if s <= raw], default=(avail[0] if avail else 0))
  ```

- **🟠 `mastery()` returns a representative, not an aggregate.** It does
  `out.setdefault(skill_id, p_correct)` over all rows at a snapshot, so per skill it keeps the
  *first* question's `p_correct`, not the skill mean. Fine for the sample; the real bundle should
  either ship dedicated per-skill **probe** rows (schema D5) or `offline_replay` should average.

- **🟠 `question_id` is positional (`{skill_id}#q{i}`, `i` = index in `questions.json`).**
  Inserting/reordering a question shifts every later id and breaks `round_sequence`. Recommend a
  **content-stable id** (explicit `id` field or content hash) in `taxonomy.py` before freezing
  the contract. The sample generator dodges this by reading ids from `taxonomy` at build time.

- **🟡 No live consumer yet for `curves` / `adapters` / `cost`.** Imported into
  `parametric_curves` / `learner_adapters` / `parametric_bundles` for the compare view + eval to
  pick up later.

- **🟡 Seed MCQs all have correct option `a`.** A position-bias leak for the eventual real
  harness — randomise option order when expanding the bank.

## Swapping in the real bundle

Replace the synthetic numbers in `make_sample_bundle.py` with the GPU rig's output (real
`p_correct`/`option_logprobs` from SFT'd per-user/shared LoRAs scored by choice-perplexity; real
`nll`/`brier`/`cost`). The file layout, field names, importer, and `offline_replay` path stay
identical — that is the whole point of locking the contract now.
