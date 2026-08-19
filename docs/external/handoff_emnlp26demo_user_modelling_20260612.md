# Handoff: user-modelling & learn-from-user-feedback research → emnlp26_demo

> **Audience**: the Claude Code session working in `exp-mark/emnlp26_demo` ("colearn").
> **Author side**: Lin Hai (linhai0012), owner of the *user modelling* and *learn from
> user feedback* responsibilities in the demo.
> **Date**: 2026-06-12.
>
> This document summarizes (1) what Lin's three research repos have tried on **public
> datasets** — code + results, and (2) which pieces of that code can be ported into the
> demo, mapped to concrete files/seams in `emnlp26_demo` (every mapping below was
> verified by actually opening both the demo files and the source files).

## 0. The three source repos

| Repo | GitHub | Local checkout (KCL CREATE) | One-liner |
| --- | --- | --- | --- |
| `user_world_model` | `linhai0012/user_world_model` (private) | `/cephfs/volumes/hpc_home/k2480198/.../user_world_model` (also `~/user_world_model`) | Active project: all-purpose per-user world model (three stores: per-user LoRA + structured profile + episodic memory). Empirical substance in `legacy/general_personamem/` (PersonaMem-v1 UserSim) and `legacy/health_digitaltwin/` (PMData). |
| `P-OPSD` | `linhai0012/P-OPSD` (private) | `~/P-OPSD` | Predecessor: parametric personalization via on-policy distillation on PersonaMem-v2 (+ the PersonaMem-v1 `dynamic_usersim/` track). Four experiment logs, last commit 2026-04-28. |
| `parametric-memory-pilot` | `linhai0012/parametric-memory-pilot` (private) | `~/parametric-memory-pilot` | Closed pilot (2026-05-08→13): "models can SEE but can't USE user memory"; retrieval baseline (A-MEM) vs golden ICL vs GRPO-trained parametric memory, on 3 public benchmarks. |

Caveat that applies everywhere: model checkpoints and heavy artifacts live on KCL
CREATE / Isambard cluster scratch, **not** in the repos. The reusable things are
prompts, pipeline/eval code, and the documented results/negative results.

---

## Part 1 — What was tried on public datasets (code + results)

### 1.1 Public datasets used

- **PersonaMem-v1** (HF `bowen-upenn/PersonaMem`) — 20 personas × {32k, 128k, 1M}
  context versions; 589 / 2,727 / 2,674 MCQs; 7 question types.
- **PersonaMem-v2** (HF `bowen-upenn/PersonaMem-v2`) — ~1,000 personas, 32k chat
  histories, 7 preference types, 5,000-MCQ benchmark split.
- **HorizonBench** (HF `stellalisy/HorizonBench`) — 346 users, structured
  preference-evolution records, 3–5-option MCQs.
- **PMData** (Simula, CC BY 4.0) — 16 participants, Fitbit + wellness self-reports
  (health-domain digital-twin track; less relevant to the demo).

### 1.2 Track A — per-user UserSim via on-policy distillation (PersonaMem-v1)

*Code*: `user_world_model/legacy/general_personamem/{data_prep,teacher_sft,student_opd}/`
(~40 .py, 10.7k lines) — same lineage as `P-OPSD/dynamic_usersim/`.
*Question*: can a specific user's preferences be compressed into a small per-user LoRA
so a 4B model simulates that user with **zero conversation history at inference**?

Setup: Qwen3-4B-Instruct-2507 teacher SFT'd to predict the *user's* next utterance
(user-token-only loss, K=3-session context); then per-persona dual-rate LoRA (~68M
params: slow MLP r32 + fast Attn r16) distilled from the teacher via on-policy
reverse KL; student sees only a persona card.

Headline results (documented in `legacy/general_personamem/EXPERIMENTS.md` §9–§13):

- Teacher SFT (R3): MCQ-PPL 34.5% → 49.1% (+14.6pp) on 32k MCQs; a persona-swap probe
  shows user identity IS partially parameterized (35/40 sign-consistent, p≈7e-7) but
  ~99% of the gain is generic user-style.
- Best per-user recipe (R1b): across **all 20 personas**, closes **+95.7%** (best-step;
  +71.7% final-step) of the (teacher-with-history − no-context-base) gap with zero
  context: base 30.6% → student 38.8% → teacher_k3 39.8% on 128k MCQs; 8/20 students
  beat the context-bearing teacher.
- Cross-version generalization: 128k-trained LoRAs get **+128% closure on the 1M
  version** (different events, same personas) → the LoRA learns a transferable
  "persona fingerprint", not episodic events.
- OPSD variant (§13): teacher additionally sees the **ground-truth user reply** in its
  context while scoring rollouts — i.e. distilling from *realized user feedback*.
  Converges 2–3× faster; much better on discrimination question types
  (track_evolution 1.00, generalize 0.82) but worse generation; complementary to R1b
  (+11pp per-qtype oracle-ensemble headroom).

Load-bearing **negative results** (these constrain what the demo should do):

- **Verbal-feedback paradigm failed twice, structurally**: reading a user-simulator's
  *expressed enthusiasm* as approval is anti-correlated with actual preference
  alignment (`EXPERIMENTS.md` §4.4, `verbal_eval_summary.md`). → Never score demo
  conditions by a simulated student's tone; always grade simulated answers with the
  same grader as human answers.
- **No model-internal confidence gate can filter a wrong teacher** (§11.3–11.7, R2a/R2c):
  entropy/argmax gates cannot distinguish "confidently right" from "confidently
  wrong" — an **external outcome signal** is required. → In the demo, automatic
  evolution triggers must key off graded outcomes, not LLM self-confidence.
- **User-only-loss LoRA destroys instruction-following** (direct-ask parse failures
  5.4% → 36–48%) — paradigm III (just ask the LoRA'd model the MCQ) abandoned.
- Metrics genuinely diverge per user (NLL vs LLM-judge vs MCQ-PPL) → never report a
  single metric for a user model.

### 1.3 Track B — parametric personalization OPD/GRPO (PersonaMem-v2, `P-OPSD`)

*Code*: `P-OPSD/{prepare_splits.py, grpo_baseline/, opd/, scripts/}`.
*Question*: same compression idea, but for answering preference MCQs about the user
(agent-modeling), with GRPO/SFT teachers.

- Reproduced the PersonaMem-v2 paper's pipeline: SFT 29% → TRL GRPO (paper-faithful
  rule+judge reward) **47.4%** on the 5,000-MCQ benchmark (paper SOTA 53.8%). A first
  GRPO attempt with embedding-similarity reward **mode-collapsed** — documented
  reward-hacking case study.
- Per-user LoRA OPD, 100 personas: C0 39.6% → C2 **50.7%** (query-only, +11.1pp, 6.3σ)
  vs C1 oracle 59.3%; v2-agent track: 48.0% → 58.8% (+10.8pp).
- **Pivotal control — the shared-LoRA null**: ONE LoRA trained on the pooled data hits
  **76.3%**, nearly matching the teacher (77.6%) and crushing per-user LoRAs (58.8%).
  Plus a leakage discovery: an earlier shared LoRA scored 69.8% with **no persona
  input at all** (benchmark-pattern memorization). → Any "per-user adaptation wins"
  claim needs a shared-model control and a no-input control.
- **Stage 1 self-supervised pipeline** (most demo-relevant): LLM-extracted **persona
  cards** — atomic facts, type enum, **verbatim evidence quotes**, post-hoc string-match
  verification (82.7% pass), sensitive-value placeholders; 100 personas / 6,186 facts /
  $8 (`scripts/extract_persona_cards.py`). And **self-generated MCQs** whose v3 had a
  catastrophic leak (99.9% answerable with no persona — preferences leaked into queries,
  correct option longest in 98.5% of cases), fixed in v5 with form-level constraints +
  structural-twin distractors → 62.3% no-persona floor (`scripts/generate_stage1b_mcqs.py`
  lineage). The leak→fix iteration is the single most reusable *lesson* for the demo's
  question generation.
- Persona-coherence analysis (E1–E4): only life-narrative (therapy_background)
  preferences have exploitable within-user structure; demographically-orthogonal
  preferences are noise by construction.

### 1.4 Track C — see≠use, retrieval baseline, GRPO on augmented MCQs (`parametric-memory-pilot`)

*Code*: `parametric-memory-pilot/{augmenter.py, prompts.py, scripts/, baselines/a_mem/}`.
*Three phases on PersonaMem-v1 + v2 + HorizonBench (584 sampled MCQs, SEED=42)*:

- **Phase 1 ("see ≠ use")**: even with the dataset's own golden reference *visible in
  the prompt*, every model leaves a big gap: gpt-5.4-mini ≤77.5%, Qwen3-4B 32.6–63%
  across the 3 datasets (12/12 cells below 80%). Manual failure taxonomy F1–F9
  (trait fabrication, snippet-literal trap, evolution confusion, …) in
  `EXPERIMENTS.md` §8.
- **Phase 2 (A-MEM retrieval baseline)**: A-MEM (vector + 1-hop graph, published
  agentic-memory system) **loses to golden-reference ICL in 12/12 cells**, mean
  −16.6pp; worst on long histories (v2, −23 to −28pp). "Memory = retrieval" is
  empirically insufficient; the bottleneck is the substrate, not the query.
- **Phase 3 (GRPO on augmented MCQs)**: training Qwen3-4B on auto-generated MCQs built
  from the same memory reaches **frontier-ICL parity on 2 of 3 datasets**
  (v1: 63→77%, +14pp, parity with gpt-5.4-mini; HorizonBench via transfer: 32.6→40.8%).
  Composition failures documented: fusing two specialist checkpoints fails
  catastrophically; each augmenter wins on exactly one source format.
- **The two calibration gates** (`scripts/audit_augmenter_quality.py`) — run before any
  training on generated MCQs: Gate 1 *no-memory* accuracy ≤40% (catches answer leaks);
  Gate 2 *with-memory* accuracy ∈[50,80] (catches ungroundable/trivial items). P-OPSD
  burned weeks on the 99.9% leak that Gate 1 would have caught in 5 minutes.
- Post-pilot §18: GRPO gains depend on memory **presentation format**, not information
  content (+9.5pp golden format vs +1.5pp A-MEM note format, same facts).

---

## Part 2 — What ports into `emnlp26_demo`, and where

The demo's own division of labor (from its docs): the user model is `learner_memory`
(agent state A_t, BKT in `services/tracing.py` + `services/memory.py`); the designed
learn-from-feedback seam is `services/evolve.py` (M5 stub) + the never-read
`guidelines` collection; `services/diagnostics.py` (M6) and `/simulate/persona` (M7,
router lines commented out) are also open.

Four verified mappings, in suggested build order:

> **Implementation status (2026-06-12, branch `linhai/user-modelling-research-handoff`):**
> all four mappings below are IMPLEMENTED and offline-tested (FakeLLM + mongomock; full
> backend suite green). Code: `services/quality_gates.py`, `services/evolve.py`,
> `services/profile.py`, `services/simulate.py` + `routes/simulate.py`, plus the wiring
> into tasking/quizgen/events/admin and new Settings (documented in `deploy/.env.example`).
> The §2.4 "optional stretch" also landed: `SIM_LLM_*` settings point the simulated student
> at a dedicated OpenAI-compatible endpoint (e.g. vLLM serving a merged UserSim LoRA),
> driven in raw-completion mode because user-turn-trained LoRAs lose instruction-following.
> Sections below are kept as the design rationale / review guide.

### 2.1 Quality gates for generated questions ← `parametric-memory-pilot`  (effort: S)

- **Demo target**: new `backend/app/services/quality_gates.py`; pre-activation smoke
  test required by `docs/backend/06-learning-engine.md` §5, plus an optional inline
  leak check in `quizgen.py::_build_mcq` (which currently has none).
- **Source**: `scripts/audit_augmenter_quality.py` (Gate 1 / Gate 2 protocol + system
  prompts, reusable nearly verbatim); `augmenter.py::find_leak` /
  `extract_trait_keywords` (pure-string distractor-leak check, direct copy, no LLM
  call, zero latency).
- **Adapt**: drop all vLLM/B200/OpenAI-CLI plumbing → ~150 lines of async code on the
  demo's `get_llm_client()` + `complete_json`; re-anchor thresholds for 4-option GCSE
  MCQs (Gate 1 ≤~40%, Gate 2 ∈[60,95]) as `Settings` fields; **add a FakeLLM branch**
  (mandatory, demo CI convention, `services/llm/client.py:47-59`).
- **Why first**: smallest piece; immediately raises live MCQ quality; hard prerequisite
  for safely activating evolved guidelines (2.2). Gate rationale: the research repos
  hit the generated-question leak failure twice; the demo's misconception-tagged
  distractors have the same risk surface.

### 2.2 `evolve_guideline` implementation (M5) ← pilot driver pattern + research constraints  (effort: M)

- **Demo target**: `backend/app/services/evolve.py` (the `NotImplementedError` stub).
  Note the consumption plumbing **already exists but is dead**: `quizgen.generate_next`
  accepts and injects `guideline_text` (quizgen.py:178/251, 280-281) but **no caller
  passes it**; `events.py:58` hardcodes `guideline_version: None`; the `GUIDELINES`
  collection constant + indexes exist in `db/mongo.py:73,101-102` and nothing reads it.
  `_build_mcq` ignores `guideline_text` (short-answer path only) — extend it.
- **Source / what transfers**: the batch driver shape from
  `parametric-memory-pilot/scripts/build_augmented_training.py` (loop recent records →
  render example blocks → one strict-JSON LLM call → sidecar audit log), plus three
  hard design constraints from documented negative results:
  1. Guideline updates = **whole-document rewrites** of a single active version, on
     batch boundaries — never incremental per-answer accretion (pilot §16:
     continued-training / data-mix fusion fails).
  2. **Never auto-activate** a new guideline without the 2.1 gates (pilot §9
     anti-attribution inversion: training amplified exactly the wrong signal).
  3. The automatic `mastery_mismatch` trigger must key off **external outcomes**
     already in `attempts` (grading.mastery_evidence vs BKT-predicted correctness) —
     not LLM self-confidence (user_world_model §11.7 structural result).
- **Concrete steps**: fix the interface
  (`evolve_guideline(db, subject_id, *, trigger, window) -> {why_updated, new_guideline_text, version}`);
  add an active-guideline lookup called from `tasking.next_item`/`next_item_streamed`
  and passed into `generate_next`; set `events` `guideline_version` from the active
  doc; admin routes `GET /admin/guidelines` + `POST /admin/guidelines/evolve`; FakeLLM
  branch; gates before activation.
- This closes the demo's designed learning-from-feedback loop and is the biggest
  outstanding paper item (the "agent side" of co-evolution).

### 2.3 Evidence-grounded learner profile ← `P-OPSD` persona cards  (effort: M)

- **Demo target**: new `backend/app/services/profile.py` + a `learner_profile`
  collection; upgrades the user model from {scalar BKT mastery + flat
  `label_summaries` strings} to typed, evidence-linked learner facts (recurring
  misconception patterns, study habits, affect/confidence), consumed by `quizgen`'s
  `student_topic_profile` payload (quizgen.py:272-308) and a Progress-page card.
- **Source**: `P-OPSD/scripts/extract_persona_cards.py` (+`postprocess_persona_cards.py`):
  atomic one-fact-per-item schema with type enum (rename `state|preference|event|value`
  → `misconception_pattern|habit|affect|goal`), **verbatim evidence quote per fact**,
  evidence-pointer scheme, post-hoc string-match verification that drops hallucinated
  facts (82.7% verification rate at 100-persona scale), sensitive-info placeholders.
  Composes naturally with the demo's existing verbatim-quote discipline in
  `grading.py` (`label_evidence_quotes`).
- **Adapt**: input = Mongo `attempts` (answer_text + grading) instead of HF episodes;
  LLM plumbing = demo's async `get_llm_client()`; run on **batch boundaries**
  (session close / every K attempts), never on the live answer path; FakeLLM branch;
  new collection constant + index in `db/mongo.py`.
- Independent of 2.1/2.2 — parallelizable.

### 2.4 `/simulate/persona` (M7) — synthetic student + offline eval harness  (effort: M)

- **Demo target**: new `backend/app/services/simulate.py` + `api/routes/simulate.py`
  (seam explicitly reserved at `api/router.py:27-29`; endpoint contract in
  `docs/backend/04-services-and-api.md`). Cold-start population of `learner_memory` +
  the volume of events/attempts that M6 diagnostics need.
- **Source patterns** (prompt/harness reuse, NOT model reuse):
  - persona-conditioned next-user-turn generation format from
    `user_world_model/legacy/general_personamem/student_opd/eval_mcq_verbal_gen.py`
    (`build_verbal_prompt`: persona header + prior turns + stimulus, model continues
    AS the user, no judgment-ask primer);
  - compact persona-card→prompt-header rendering from
    `P-OPSD/scripts/extract_persona_cards.py::render_demographics`;
  - harness skeleton from `parametric-memory-pilot/scripts/run_eval.py`
    (deterministic seed, parallel workers, resume-via-done-IDs, per-run summary JSON).
- **Key design rules**: simulated answers must flow through the **normal submit path**
  (`tasking._grade → _trace_and_persist`) so BKT/events/attempts stay uniform; the
  FakeLLM already supports a `mastery_signal=<x>` scripted-grading hook
  (client.py:71-77) — emit it for deterministic offline simulation; and per the
  verbal-feedback negative result, **grade simulated answers with the same grader as
  human answers, never by the simulated student's tone**.
- ~~Optional stretch~~ → **implemented**: `SIM_LLM_BASE_URL` / `SIM_LLM_MODEL` /
  `SIM_LLM_API_KEY` give the simulated student a dedicated OpenAI-compatible endpoint
  (e.g. vLLM serving a merged R1b UserSim LoRA: `vllm serve <merged-dir>
  --served-model-name usersim-lora-merged`), in raw-completion mode
  (`SIM_LLM_RAW_COMPLETION=1`, no JSON demanded — user-turn-trained LoRAs lose
  instruction-following). Unchanged caveat: those LoRAs encode PersonaMem chat
  personas, not GCSE students — texture, not fidelity; the offline FakeLLM path stays
  the default.

### What NOT to port (and why)

- **All GPU training pipelines** (per-user OPD/OPSD LoRA trainers, teacher SFT, GRPO
  stacks): the demo is a single FastAPI+Mongo node on Bedrock, framed as a demo paper
  ("interactive system + evaluation, NOT a method paper"). Per-user weight training
  adds GPU serving + latency + a claims burden — and the research itself warns the
  headline per-user gains are confounded (shared-LoRA null 76.3% vs per-user 58.8%).
- **PPL-based probes** (`eval_mcq_ppl.py`, `eval_persona_swap.py`): need token
  logprobs; Bedrock Converse doesn't expose them.
- **A-MEM/O-MEM retrieval stacks**: wrong substrate for the demo's structured
  per-(user,skill) Mongo memory — and A-MEM lost to golden ICL 12/12 anyway.
- **The verbal-feedback paradigm** (reading simulated reactions as approval): proven
  structurally anti-correlated, twice.
- **Blind-A/B records as a training signal for evolve**: the A/B is the paper's human-
  eval instrument; learning from it would contaminate the headline evaluation (also
  flagged in the demo's own open questions).

### Gaps needing fresh code (no research coverage)

- The evolve LLM prompt itself + guidelines CRUD/admin UI (M5) — research gives the
  driver pattern and guardrails only.
- `diagnostics.py` (M6): Brier calibration of BKT predictions vs outcomes,
  mastery-stabilization curves — pure math over `events`+`attempts` (~150 lines).
  Research lesson to encode: triangulate metrics; use the frozen condition as the
  baseline the way the health track used persistence baselines.
- Transfer-probe authoring + scheduling (`is_probe` exists in schema; no probe set or
  scheduler).
- FakeLLM branches for every new LLM task (profile extraction, simulation, gates,
  evolve) — demo CI convention.
- Optional upgrade path the demo docs themselves name: a DKT-style sequential model
  behind the `(obs, w)` tracing seam — none of the three research repos has
  knowledge-tracing code; the attempts log already captures the sequence.

---

## Appendix — where the numbers live

- `user_world_model/legacy/general_personamem/EXPERIMENTS.md` (§1–§13, 115KB) + 358
  committed result JSONs in `legacy/general_personamem/outputs/`; status snapshot in
  `PROJECT_STATUS_2026-05-29.md` (Chinese).
- `P-OPSD/EXPERIMENTS.md`, `EXPERIMENTS_V2_AGENT.md`, `EXPERIMENTS_STAGE1.md`,
  `EXPERIMENTS_STAGE2a.md`, `dynamic_usersim/EXPERIMENTS.md`. README is stale — trust
  the EXPERIMENTS files.
- `parametric-memory-pilot/EXPERIMENTS.md` (TL;DR at top) +
  `outputs/eval_runs/master_results_table.md`.
- Known headline caveats: the +95.7% closure uses per-persona best-step (oracle early
  stopping; final-step +71.7%) and a rich persona-card input (minimal-demographics
  ablation never run); P-OPSD per-user numbers are reinterpreted by the shared-LoRA
  null; all PersonaMem closure numbers compare against the same-model teacher — an
  external token-memory baseline (Mem0/Zep/RAG) was identified as the key gap and is
  the active (not yet started) work in `user_world_model`.
