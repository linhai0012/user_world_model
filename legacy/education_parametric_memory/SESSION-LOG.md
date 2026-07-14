# Session log — Mode A parametric memory: design → full SFT run

**Single "start here" doc for resuming.** This session went from conceptual design of the
parametric learner model all the way to a complete, GPU-real lean-SFT run on a B200, with
adversarial verification. Companion docs: [`RESULTS.md`](RESULTS.md) (numbers),
[`substrate/README.md`](substrate/README.md) (the A-vs-B contract for the agentic collaborator),
[`notes/knowledge-tracing-lineage.md`](notes/knowledge-tracing-lineage.md) (KT positioning).

---

## Part 1 — Conceptual foundation locked (the discussion)

The early session pinned down *what the parametric model is* before any code:

1. **Lin's scope** = the **offline Mode-A parametric bundle only**. The live app (engine, chat UI,
   baseline+agentic backends, the 3 ParametricProviders, the mock service) is already built by the
   friend's team; the live demo replays our bundle via `offline_replay`, GPU-free.
2. **A_t, not H_t.** The parametric model is the **agent's belief about the learner** (a
   *predictor*), NOT the user-simulator that answers (`backend/app/services/simulate.py`, already
   built). It is decoupled from the symbolic proactive engine (`proactive.py`, pure Python ranking,
   not a model) and from the general Claude used for question-gen/grading.
3. **Per round the model is touched 3 separate ways:** *predict* (`predict_correct`, read-only, =
   the eval metric) · *update* (a separate write = the offline LoRA retrain) · *calibrate* (offline
   isotonic, once). Don't conflate them.
4. **Prediction target = the full `option_logprobs` distribution (which distractor the learner
   picks)**, not binary correctness — because tutoring MCQs have an objectively-correct answer the
   base model already knows (strong prior + high A∅ leak floor); the learner-specific signal lives
   in the *errors*.
5. **Latent-variable / inverse-problem framing.** Persona θ = (misconception/mastery) is a
   structured ground truth we author (oracle to us, hidden to the model); observable = the answer
   stream; the per-user LoRA reconstructs θ's *behaviour* into weights = amortized neural KT/IRT —
   NOT symbolic θ recovery (that's BKT's job). It works only if θ is *identifiable* from
   observations (items must discriminate).
6. **How to build the latent model** (Lin asked SAE vs Concept-Bottleneck): **rejected both as the
   backbone** — SAE is a post-hoc probe, CBM collapses the 3-way contrast + is out of scope.
   **Decision (走A):** keep Mode A an **opaque per-user LoRA**; put interpretability into a
   **post-hoc θ-recovery probe** (linear/SAE) as an appendix, runnable on all 3 backends.
7. **Training fork:** **lean direct SFT (no teacher)** — honours the §9 "lean adapter only"
   guardrail; the ground-truth answer is the target so the OPD teacher is unnecessary.
8. **The "Qwen knows everything" worry resolved:** base competence ≠ learner mastery; the learner's
   "starts weak, improves" trajectory comes from the ground-truth stream, decoupled from the base.

Data-design decisions **D-a..D-d** (all approved): controllable θ→answer generator (no LLM,
deterministic) · low-dim IRT-style θ (ability + per-topic offset + per-skill jitter + misconception
profile) · biology-only v1, N=24 · target ~12 MCQ + 3 short per skill.

## Part 2 — Built (de-risk first, then the harness)

Before the GPU node, we **de-risked the cross-team contract**: a sample bundle + the missing
JSONL→Mongo importer + a test proving the live `offline_replay` serves it. Then on the B200:

- **Question bank** — a Workflow fanned out 13 skill-experts + adversarial verifiers →
  **173 GCSE-Biology MCQs** with misconception-tagged distractors, content-stable ids
  (`{skill_id}#g{md5}`). (`build_question_bank.py`, `question_bank/biology_gcse.jsonl`.)
- **Data layer** — `personas.py` (θ + controllable generator that biases wrong answers toward the
  held misconception, fixing simulate.py's uniform-random wrong pick) + `streams.py` (round_seq,
  stream sim with learning dynamics, per-snapshot eval, discrimination gate). **Contract bug caught
  & fixed:** builtin `hash()` of strings is per-process random → round_sequence irreproducible
  across machines → replaced with md5 `det_seed` (verified byte-identical across processes).
- **Substrate** — `build_substrate.py` → `substrate/` (24 personas, splits train104/eval39/calib30,
  round_sequence, streams, eval_truth). This is the **shared A-vs-B contract** for the agentic arm.
- **SFT harness** (`sft/`) — `sft_core.py` (dual-rate LoRA SFT + choice-PPL) · `run_all.py`
  (A1/A0/A∅ → predictions, resumable) · `package_bundle.py` (PAV isotonic + 5-file bundle) ·
  `verify_bundle.py` (honesty gate) · `analyze.py`/`diagnose.py`/`ceiling_test.py`.

## Part 3 — The autonomous run (process)

Env: **1×B200 183GB**, conda `vllm_env`, Qwen3-4B-Instruct-2507, **no API key** (bank via Workflow
subagents; generator deterministic; train/score local). Each full config ≈ 35–52 min.

| run | config | purpose |
| --- | --- | --- |
| smoke | 1 persona | validate train+score on GPU (passed) |
| **v1** | round_len 40, std prompt | baseline |
| **v2** | round_len 104 (denser) | does more per-skill data help transfer? |
| ceiling | 3 personas, 15 epochs | memorise vs transfer? |
| **v3** | denser + skill-conditioned prompt | does a skill key aid transfer? |
| **v4** | denser + spread population | is "base wins" a population artifact? |

Three metrics computed each config: **binary NLL** (standard KT), **θ-recovery MSE** (predicted
p_correct vs true mastery — removes Bernoulli noise), **mastery correlation**.

## Part 4 — Results (synthesis; full tables in RESULTS.md)

| | v1 | v2 | v3 | v4 (spread) |
| --- | --- | --- | --- | --- |
| mastery-corr peak | 0.10 | 0.11 | 0.13 | **0.18** |
| per-user **> shared** (recovery) | ✅ | ✅ | ✅ | ✅ |
| per-user **> base** (θ-MSE) | early | no | no | **early/mid** |
| per-user > base (binary NLL) | early | no | no | no |

1. **Per-user parametric memory reliably recovers learner mastery that the shared LoRA and base do
   NOT** (corr 0.10–0.18 vs ≈0; θ-MSE < shared every config). This **reverses the pilot's NLL-based
   shared>per-user null** — once you measure *latent recovery* not binary NLL.
2. **Binary-outcome NLL is underpowered** (an oracle barely beats chance when mastery≈0.5,
   Bernoulli outcomes). With a powered eval (spread population + θ-recovery, v4), **per-user beats
   the base too**. The earlier "base wins" was a 0.5-centered-population artifact.
3. **Lean SFT memorises (train choice-acc 1.0) but transfers weakly** (ceiling test); denser data &
   skill-conditioning didn't fix it. OPD/teacher (the cut recipe) is the future-work transfer fix.

**Honesty gate: PASS** — real bundles `cue-param-biology-v1` and `-v4` import into Mongo and are
served by the live `offline_replay` (per-user rows exact, walk-up → shared fallback). Deployed demo
can replay the real run; never the stub.

## Part 5 — Where everything is

- **Code** (repo, untracked, **NOT pushed**): `parametric_offline/` — `sft/`, `personas.py`,
  `streams.py`, `build_*.py`, `RESULTS.md`, `substrate/` (+ its contract README), `notes/`. Plus
  `backend/tests/test_offline_replay_bundle.py`, `backend/tests/test_data_layer.py`.
- **Outputs** (scratch, 284 MB): `/scratch/prj/cllm/cue_sft/` — `bundles/{v1,v2,v4}/`,
  `predictions*/`, `logs/`, `archive_v1/` (archived substrates), HF model cache on
  `/scratch/users/k2480198/.cache/huggingface`.
- **Repo venv:** `backend/.venv` (py3.12, from conda `sdpo`) for CPU/tests; `vllm_env` for GPU.

## Part 6 — How to resume

```bash
# CPU (tests, packaging, analysis):  backend/.venv/bin/python
# GPU (train/score):                 /users/k2480198/.conda/envs/vllm_env/bin/python  (HF_HOME=/scratch/users/k2480198/.cache/huggingface)
# A run is parameterised entirely by env:
CUE_SNAPSHOTS=0,13,26,52,104 CUE_ROUND_LEN=104 CUE_EPOCHS=3 \
CUE_SKILL_PROMPT=0 CUE_ABILITY_SD=1.5 CUE_SKILL_SD=0.2 \
CUE_PRED_DIR=predictions_vX CUE_BUNDLE=cue-param-biology-vX
#   1) build_substrate.py   2) sft/run_all.py   3) sft/package_bundle.py   4) sft/verify_bundle.py
#   analyse: sft/analyze.py  (CUE_PRED_DIR, CUE_SUBSTRATE, CUE_SNAPSHOTS)
```
Project memory (`cue-linhai-role`, `cue-parametric-design-decisions`) carries the same state.

## Part 7 — Open decisions for next session

1. **Push?** `parametric_offline/` + 2 tests + `ONBOARDING-LINHAI.md` are untracked, nothing pushed
   (per Lin's instruction). Decide what/when to push (trust-boundary: Lin pushes).
2. **Paper headline config** — recommend **v4** (spread population, the cleanest positive) — and
   which bundle ships for the live demo.
3. **`offline_replay` nearest-≤ snapshot patch** — exact-match fails on non-listed snapshots (it
   surfaced in v4 verify); patch is written in `README.md`, unapplied (friend's live-app file).
4. **OPD/teacher recipe?** The transfer ceiling suggests revisiting the recipe we deliberately cut —
   reopens the lean-SFT-vs-OPD decision; weigh against scope.
5. **Agentic alignment** — hand `substrate/` + its README to the collaborator so both arms run the
   identical persona_set/round_sequence and return predictions in the same schema.
6. **Power the eval** — adopt θ-recovery as a reported metric and a spread persona population in the
   shared substrate; consider adding chemistry for generality, and multiple seeds.
