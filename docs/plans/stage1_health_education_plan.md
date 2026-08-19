# Stage-1 cross-domain plan — Health + Education (handoff to a fresh session)


> **Historical plan (2026-06-17).** Kept as written. Code paths in it predate the 2026-08-09
> reorg: `common/health_data.py` → `domains/health/data.py`, `common/edu_data.py` →
> `domains/education/data.py`, `scripts/<x>.py` → `scripts/<domain>/<x>.py`.

> 2026-06-17. Decided with the user: after general/PersonaMem (done), **validate the other two
> UWM domains**. Scope = **Stage-1 intrinsic prediction** (project_summary §6): predict the user's
> reaction/state, with a **base / +profile / +memory** ablation, frozen base — mirroring the
> general-domain methodology. **Health first**, then Education.
>
> Build plan for a NEW Claude Code session. Read `CONVENTIONS.md` (§2 storage, §4 concurrency) and
> the `EXPERIMENTS.md` 2026-06-17 general-domain entry first.

## Operating context (READ FIRST)
- **Storage**: `inf_embrace_llm` quota is FULL → active scratch = **`/scratch/prj/inf_elandi/k2480198/uwm`** (env.sh updated). **Save LoRA adapters (~130MB), never 8GB merged models; delete transient merged models after eval.** Touch only `user_world_model` + its `inf_elandi` scratch.
- **Env (use explicit — env.sh propagation was flaky in backgrounded jobs)**:
  `UWM_DATA=/scratch/prj/inf_elandi/k2480198/uwm/data HF_HOME=/scratch/prj/inf_elandi/k2480198/uwm/hf_cache VLLM_ATTENTION_BACKEND=FLASH_ATTN`; python = `/scratch/users/k2480198/conda/.conda/envs/vllm_env/bin/python`.
- Node: KCL CREATE, 1× B200, vLLM 0.13 / torch 2.9 cu128. (HF *training* on B200 must use `attn_implementation="sdpa"` — the flash-attn wheel has no sm_100 kernel for the HF path; vLLM serving is fine.)
- Base model `Qwen/Qwen3-4B-Instruct-2507` cached at `$HF_HOME`.

---

## HEALTH — mostly built; finish + run

**Target**: predict next-day wellness **state** (6 ordinal fields: fatigue/mood/sleep_quality/
soreness/stress 1–5, readiness 1–10) — the UWM world-model state transition. Data:
`$UWM_DATA/health/digitaltwin/output/{train,val,test}.jsonl` (PMData→GPT-synth; each record has
`wellness_day_n`→`wellness_day_n1` + activity). 1197 train / 314 test / 16 participants.

**Already built this session (committed):**
- `common/health_data.py` — loader (validated), per-participant baselines, plain-text state rendering, field clamps.
- `scripts/eval_health_stage1.py` — conditions {persistence, pop-mean, base, +current, +profile, +current+prof}; per-field+overall MAE vs persistence; frozen Qwen3-4B generates a 6-field JSON next-state.

**Result so far (no-LLM baselines, validated):** persistence overall MAE **0.544**, pop-mean **0.538**
→ wellness is very stable (pop-mean already ≈ persistence); beating persistence is the hard bar
(matches `legacy/health_digitaltwin/EXPERIMENTS.md`: "model loses to persistence on all fields").
The Stage-1 *question*: can +current/+profile get a frozen model NEAR persistence, and does
memory/profile help vs base?

**TODO (new session):**
1. **Fix the vLLM init bug** in `eval_health_stage1.py`: a raw `LLM(...)` call threw
   `RuntimeError: Device string must not be empty`. The MCQ path `common/backends.VLLMQwenBackend`
   works — easiest fix: reuse it (or copy its exact init: `VLLM_ATTENTION_BACKEND` setdefault
   *before* importing vllm; don't pass an empty device). Likely a stray `CUDA_VISIBLE_DEVICES=""`
   or a vLLM-0.13 arg — reproduce + fix.
2. Run all conditions on test (314 recs × ~4 LLM arms — generation only, no model save, fast).
3. Record in `EXPERIMENTS.md`: per-condition MAE vs persistence; honest read (does profile/memory
   help; can anything beat persistence). Commit + push.
4. (Optional) add the **text-reaction** target (NLL / LLM-judge of `output_text`), and a
   **+per-user weights** arm later (adapter-only save; reuse `train_peruser.py` pattern) — but the
   state-prediction ablation is the Stage-1 core.

**Caveats:** persistence is a brutally strong baseline (dataset property; GPT-4o zero-shot failed
identically in legacy) — frame as "does conditioning approach/beat persistence", not absolute acc.
States as plain text (no custom tokens) → frozen base needs no training for base/profile/memory.

---

## EDUCATION — greenfield; bigger build

**Data**: `data/education/{chat_nlp.jsonl (48), chat_ai.jsonl (18)}` — private KCL student↔AI-tutor
Langfuse traces (system prompt + `{role,content}` turns + `output` + `_reconstructed_turns` +
`_category`). **No labels, no benchmark, no prior code/results.** Learner-reaction signals present
(clarity/confusion, correctness feedback, engagement) per `data/education/README.md`.

**Stage-1 task to define (design decision for the new session):** on chat-only data the cleanest
tractable target is **predict the learner's next turn / reaction** given (tutor turn + prior
context), base/+profile/+memory:
- **reaction text**: NLL or LLM-judge similarity of predicted vs real student turn (≈ general's
  PPL/judge) — the "edu-reaction" head (project_summary §2).
- conditions: base (tutor turn only) / +profile (learner summary/level) / +memory (prior turns).
- **No exam data here** → the "edu-exam" structured-state head needs public sets
  (EEDI/EdNet/ASSISTments — not downloaded; defer or fetch).

**TODO (new session):**
1. `common/edu_data.py`: parse Langfuse traces → per-session ordered turns (use
   `_reconstructed_turns`); student turns = targets, tutor turns = context; per-learner profile from
   `userId`/early turns. **Inspect 2–3 sessions first** — schema is rich/nested; confirm reconstruction.
2. Build the next-student-turn prediction eval (NLL + optional judge), base/+profile/+memory.
3. Small data (66 sessions) → report cautiously; pair with **StudyChat** (public analog) for breadth.

---

## Suggested order
1. Health TODO 1–3 (fix vLLM → run → record) — quick, mostly built.
2. Education TODO 1 (parse+inspect) → 2 (eval) — the real build.
3. Keep `EXPERIMENTS.md` current; commit/push per milestone (driver, path-scoped git; never
   tree-wide destructive git). Storage-safe throughout.

## Pointers
- General harness to mirror: `common/{data,backends,scorer}.py`, `scripts/run_campaign.py`.
- Reusable health code: `legacy/health_digitaltwin/` (config wellness fields, metrics state-MAE,
  step1_parse_pmdata, generate_and_eval) + its `EXPERIMENTS.md` (prior digital-twin results).
- Results → `experiments/results/<...>.json` (committed); large artifacts → `$UWM_SCRATCH` only.
