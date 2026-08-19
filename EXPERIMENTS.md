# EXPERIMENTS — user_world_model (active work, from 2026-06)

> Experiment log for the **active all-purpose framework build**. Design spec:
> `project_summary.md`. Operational conventions: `CONVENTIONS.md`.
> Prior per-domain logs are frozen under:
> - `legacy/general_personamem/EXPERIMENTS.md` — Phase 0–2b + OPSD (PersonaMem prototype)
> - `legacy/health_digitaltwin/EXPERIMENTS.md` — digital-twin health prototype

> ## How to read this log
>
> The project is at an **early, exploratory stage**. Every entry below is *one* dataset,
> *one* recipe, usually *one* base model and one seed. Entries record **what was observed
> in that setup**, not what has been established about the method:
>
> - A number that beats its control is a **lead worth pursuing**, not a demonstrated win —
>   it has not been reproduced, tuned, or shown to hold on other data.
> - A number that fails to beat its control is a **null under that recipe**, not evidence
>   that the arm cannot work — nearly every null here has an untried stronger recipe behind
>   it (per-user best-step selection, soft-label distillation, better data), and the two
>   places per-user weights *have* moved a metric used exactly such recipes.
>
> So: no entry below settles whether a component of the design works. Phrase future entries
> the same way — report the setup, the control, and the margin, and let "success"/"failure"
> stay open until something has been run more than once.

## Current focus

General domain first (`project_summary.md` §8): an ablation skeleton
`base` vs `+profile` vs `+memory` vs `+per-user weights` on PersonaMem, with
comparable token-based-memory baselines (oracle / trivial / FluxMem / Mem0 / …)
under one eval contract (`CONVENTIONS.md` §3 — accuracy + per-qtype).

## Log

<!-- newest first; one entry per milestone. Mirror headline numbers from
     experiments/results/<domain>/<run_id>__*.json; full data stays on cluster scratch. -->

### 2026-06-20 — EDUCATION Stage-1 (domain #3): in-context memory partially helps (genuine vs priming, via controls)

**Domain #3 (education) Stage-1**, text/reaction head only (predict the student's next turn;
project_summary.md §2 edu-reaction ≈ general's reply). Data: private KCL Langfuse tutor chats,
66 sessions (48 NLP + 18 AI) → **433 student turns**. Code: `domains/education/data.py`,
`scripts/education/eval_edu_stage1.py`. Result: `experiments/results/education/edu_stage1__nll.json`.

**Data reality (differs from `data/education/README.md`):** dialogue lives in `input.messages`
(system tutor-persona + alternating assistant=tutor / user=student; a few tool turns); **no
`userId`/`sessionId`** → no cross-session learner identity, so the unit is the SESSION and
education runs the **FROZEN base/+memory ablation only** (no per-user weights — nobody to
personalize, 66 sessions). No exam data → no structured state head (edu-exam deferred, needs
EEDI/EdNet). Student turns are genuine learner text (typos, follow-ups, "WHAT IS PPO").

**Task:** NLL of the student's next turn under frozen Qwen3-4B. `base` = system + the tutor's
immediately preceding turn; `memory` = system + full prior dialogue. Paired (same targets).

| course | base | memory | shuffled (same turns, scrambled) | foreign (other session) | Δmem−base | Δmem−shuf | Δmem−foreign |
|---|---:|---:|---:|---:|---:|---:|---:|
| nlp (n=346) | 4.397 | 3.414 | 3.641 | 3.003 | −0.983 | **−0.227** | +0.412 |
| ai (n=87) | 3.964 | 3.075 | 3.354 | 2.958 | −0.888 | **−0.279** | +0.117 |

**Observation (two controls):** raw "memory" lowers next-turn NLL ~0.9 nats (~25%).
Two controls decompose it:
- **shuffled** (the SAME real turns in scrambled order — content+length-matched): memory is ahead
  by ~0.23–0.28 nats. So at equal content/length, the *ordered, coherent* real history still
  helps in this setup — the effect is not fully explained by priming.
- **foreign** (a different session's dialogue): appears to help *more* than memory (Δmem−foreign
  +0.1/+0.4), but foreign injects a whole, **longer** session → a length confound. The
  length-matched shuffled control is the cleaner comparison of the two.
- Decomposition (under these two controls): of the ~0.9-nat raw gain, ~0.25 tracks
  ordered-relevance (memory−shuffled) and the rest tracks in-domain priming/length
  (shuffled−base ≈ −0.6/−0.7). Depth is consistent with that (late −1.18 vs early −0.43).

**Cross-domain read (provisional).** Across the three Stage-1 setups run so far, education is the
only one where the effect survives its control (memory > shuffled), while general (frozen+memory
≈ no gain under MCQ-PPL, "see≠use"; retrieval below oracle) and health (per-user ≈ shared-LoRA;
the +current gain appears at the population level) do not show one under the recipes tried. The
pattern to carry forward is methodological rather than conclusive: **naive context/weights gains
have repeatedly shrunk once a trivial/shared/foreign/shuffled control was added**, so every arm
needs its control — and, symmetrically, none of these nulls has yet been tested against the
stronger recipes (per-persona stopping, soft-label distillation) that moved the metric elsewhere.

**Caveats:** no learner identity → no per-user arm in education (frozen reader only); the surviving
margin is small (~0.25 nats) and unreplicated; 66 sessions, NLL only — LLM-judge generation lens +
public **StudyChat** breadth are TODO.

### 2026-06-20 — HEALTH Stage-1 cross-domain replication: frozen "see≠use" + per-user weights (state head)

**Domain #2 (health) Stage-1 intrinsic-prediction**, same ablation skeleton as general
(`project_summary.md` §6/§8.2), target = next-day wellness **state** (6 ordinal self-report
fields; the real PMData self-report is the GT). Frozen Qwen3-4B-Instruct-2507 via vLLM 0.13 on
1× H200; per-user LoRAs via HF SFT (sdpa), adapter-only (CONVENTIONS §2). Data:
`$UWM_DATA/health/digitaltwin/output` — 1197 train / 314 test / 16 participants. Code:
`domains/health/data.py`, `domains/health/peruser_data.py`, `scripts/health/{eval_health_stage1,
train_health_peruser,eval_health_peruser}.py`. Results: `experiments/results/health/health_stage1__mae.json`,
`health_peruser_{base,current}__mae.json`.

**Frozen baselines (overall MAE vs next state; lower better; n=314):**

| arm | MAE | Δ vs persistence |
|---|---:|---:|
| persistence (next = today) | **0.544** | — |
| pop-mean | 0.538 | −0.006 |
| frozen base (activity only) | 1.122 | +0.578 |
| frozen +current (today's state in prompt) | 1.260 | +0.716 |
| frozen +profile (baseline in prompt) | 1.141 | +0.597 |
| frozen +current+prof | 1.136 | +0.592 |

→ **Same shape as general's "see≠use" in this setup:** the frozen LLM is ~2× worse than the
trivial persistence baseline, and injected context does not improve it — **+current is the worst
arm** (the frozen model does not anchor on today's state). Under this prompt/model, conditioning a
frozen base by prompting alone does not get near the trivial bar; whether a different prompt
format or a stronger base would is untested.

**Per-user weights (LoRA per participant, 12 data-rich pids ≥39 train recs; pooled n=301):**

| arm | base context | +current context |
|---|---:|---:|
| persistence | 0.532 | 0.532 |
| per-user-mean lookup (control) | **0.521** | 0.521 |
| frozen base | 1.112 | 1.262 |
| **per-user LoRA** | 0.548 | 0.506 |
| **shared LoRA (1 model, all 12 pids pooled)** | 0.537 | **0.499** |

**Observations:**
1. **Training put something about the user into the weights that prompting did not** — base-context
   per-user (0.548) halves the frozen base (1.112) and reaches trivial-baseline territory from the
   activity alone (user in weights, not prompt). Ahead of frozen base on 11/12 pids.
2. **But at base context it does not clear the per-user-mean lookup** (0.548 > 0.521) — consistent
   with the LoRA mostly capturing the user's *level* rather than activity-conditioned dynamics.
   The control is what shows this (cf. P-OPSD shared-LoRA / general naive-per-user); a stronger
   recipe was not tried here.
3. **`+current` per-user clears every trivial bar in this run** (0.506 < umean 0.521 < persistence
   0.532) — the frozen model got *worse* when given the current state (1.112→1.262) while the LoRA
   *trained* with it got *better* (0.548→0.506). Suggestive that training can make usable what
   prompting could not; margins are small (see 4) and this is a single recipe/seed.
4. **Caveats (honest):** margins are small (0.506/0.521/0.532 within 0.03 — the dataset has weak
   day-to-day dynamics); the pooled +current win is partly driven by hyper-stable users (p05:
   0.808→0.142 once it can anchor on today; umean 0.992 for p05); per-pid vs umean is mostly small,
   winning where current-state anchoring matters. p12 (8 test/39 train) overfits at base context
   (1.729), rescued by +current (0.542).
5. **Shared-LoRA control (the one that matters most here):** ONE LoRA trained on all 12 pids
   pooled matches/slightly beats the per-user LoRAs (shared 0.499 vs per-user 0.506 at +current;
   0.537 vs 0.548 at base). → **On this dataset with this recipe, per-user weights showed no gain
   over a single population model.** The improvement over the frozen base is therefore attributable
   to a *population-level* skill ("learn the task / learn to anchor on the current state") rather
   than to per-user signal — plausible given how weak the between-user-distinguishable dynamics are
   here (see the dynamics diagnosis TODO). Note what this does *not* establish: the per-user LoRA
   used fixed 3-epoch CE with no per-pid stopping, so this is a null for *that* recipe on *this*
   data, not a finding that per-user weights are uninformative in health. (Same shape as P-OPSD's
   shared-LoRA null and general's high-variance naive per-user — the point to keep is: always run
   the shared control.) Results: `health_shared_{base,current}__mae.json`.
6. **Field-level (per-field MAE, persistence → shared+current model):** the dynamic signal is
   concentrated in **readiness** (10-pt field; 1.236 → 1.146, by far the largest error) plus
   sleep_quality (0.615→0.515), fatigue (0.385→0.336), soreness (0.385→0.342). **mood (0.246) and
   stress (0.326) are near-constant** and the model does NOT help (slightly *hurts* stress
   0.326→0.405). So the modest overall gain is essentially "predict readiness/sleep change a bit
   better"; the stable fields offer no headroom.
7. **Shared-LoRA ablation grid (pooled MAE, the population model at each cond):** base 0.537 →
   +profile 0.518 → **+current 0.499** → +current+prof 0.508. The **current state is the useful
   conditioning** (−0.038); the structured **profile adds little alone (−0.019) and nothing on top
   of current** (+current+prof 0.508 > +current 0.499) — the baseline profile is redundant with
   today's state for predicting tomorrow. Best health model overall = shared+current 0.499.
   Results: `health_shared_{profile,currentprof}__mae.json`. (Per-user grid not run — shared-null
   showed personalization doesn't help, so the grid is only meaningful at the population level.)
8. **Reaction-text head (health's 2nd UWM head; frozen NLL of the first-person reaction, n=314):**
   base 4.193 → +current 4.094 (−0.099) → **+profile 4.009 (−0.184)** → +current+prof 4.084.
   Unlike the state head (+current HURT), for the free-text reaction head context HELPS modestly
   (profile most) — consistent with the cross-domain pattern that context aids text generation but
   not structured-state prediction. ⚠️ The reaction is GPT-synthesized grounded in the real state,
   so context→reaction NLL gains are partly circular; treat as a soft signal. Code:
   `scripts/health/eval_health_reaction.py`; result `health_reaction__nll.json`.

### 2026-06-17 — baseline harness live; dual-lens token-memory ablation on PersonaMem-v1 (pm32k)

**What landed.** First runnable general-domain harness (`common/` shared lib + `baselines/`
+ `scripts/general/run_campaign.py`), validated end-to-end on 1× B200 (vllm_env, vLLM 0.13, FLASH_ATTN).
Methods implement the shared `build_context(mcq,data,params) -> list[dict]` contract; one shared
backend scores them, so arms are comparable (CONVENTIONS §3). Code commit `ae14d87`.

**Two eval lenses** (both report accuracy + per-qtype on the 589 pm32k MCQs):
- **PPL** — MCQ choice-perplexity (the legacy UserSim protocol: argmin mean-NLL of each option
  as the assistant continuation; faithful to `legacy/.../teacher_sft/eval_mcq_ppl.py`). The lens
  for the eventual per-user-weights (UserSim) arm.
- **gen (reader)** — the model *answers* the MCQ given the injected memory (how A-MEM/Mem0-style
  token-memory baselines are evaluated). The lens for the token-memory arms.

**Protocol validation + a bug caught.** profile (demographics-only) PPL = 0.353 ≈ legacy base-32k
0.345, and pm128k profile-PPL = **0.304 ≈ legacy base-128k 0.306** (exact anchor) → the MCQ-PPL
protocol is faithful to `legacy/.../eval_mcq_ppl.py`. First oracle was mislocated: it sliced ±2 turns around `end_index`
(the *query* position), but `distance_to_ref_in_tokens` shows the reference sits far earlier
(~2k tokens from the start of a 27k context). Fixed by locating the reference via **token
distance** walked back from the query (`domains/general/data.py::_ref_index`, uses the Qwen tokenizer).

**Headline — pm32k ablation (acc; random=0.25):**

| arm | PPL lens | reader lens |
|---|---:|---:|
| profile (+demographics) | 0.353 | 0.379 |
| trivial (no memory, = base/floor) | 0.389 | 0.435 |
| naiverag BM25 top-5 (retrieval) | — | 0.584 |
| oracle-full (whole 27k context) | — | 0.584 |
| naiverag dense top-5 (MiniLM) | — | 0.601 |
| oracle-slice (golden ±2-turn ref) | 0.326 | **0.683** |

**Observations (general-domain, frozen Qwen3-4B-Instruct-2507):**
1. **Under the PPL lens, injected memory did not help this frozen base** (trivial 0.389 ≥ oracle
   0.326); it scored below no-memory. Consistent with the legacy "base context-benefit ≈ 0" and
   the pilot's see≠use — a reason to *try training* (the per-user-weights arm) alongside context
   injection, not a demonstration that context injection cannot work.
2. **The same model, asked to answer, does use the golden reference** (+25pp, 0.435 → 0.683). The
   PPL-vs-reader divergence is the notable part: which lens you pick changes the sign of the story.
3. **Demographics alone hurt** (−0.06 both lenses): the persona card lacks the specific fact and
   pulls the model toward persona-matching distractors (the snippet-literal / wrong-facet trap).
4. **Needle ≫ haystack**: the focused golden slice (0.683) beats the full 27k context (0.584) by
   ~10pp — long-context dilution → motivates retrieval.
5. **Token-memory baseline (the #1 documented gap, now filled for pm32k):** realistic retrieval
   recovers most of the oracle ceiling but leaves an **~8pp gap** (dense 0.601 / BM25 0.584 vs
   oracle 0.683), and retrieval ≈ full-context. This is the bar the `+per-user weights` arm must
   beat. (dense MiniLM is CPU-bound: 12 min vs BM25 12 s.)

**pm128k (2710 MCQs, reader lens) — the longer-context, more memory-dependent regime:**

| arm | reader acc | Δ vs base |
|---|---:|---:|
| profile (+demographics) | 0.327 | −0.06 |
| trivial (base) | 0.385 | — |
| oracle-slice (golden ±2-turn) | 0.487 | +0.10 |
| naiverag BM25 top-5 | 0.494 | +0.11 |

- base is lower than pm32k (0.385 vs 0.435) → a bit more headroom, but PersonaMem-v1 is still
  largely answerable without persona. demographics still hurt.
- **On pm128k, retrieval ≈ oracle-slice** (0.494 vs 0.487), unlike pm32k where oracle led by
  ~8pp. The aggregate tie hides a **qtype crossover**: naiverag beats oracle on `recall_facts`
  (0.561 vs 0.479) and `acknowledge_latest` (0.565 vs 0.452, the 128k-only "latest preference"
  qtype, n=866) — retrieval finds the specific/recent turn better than a fixed ±2 slice — while
  oracle wins `aligned_rec` (0.645 vs 0.504) and `generalize` (0.657 vs 0.460), where the
  contiguous reference context carries the reasoning. Suggests the ±2 oracle window is undersized
  for dispersed 128k references (a wider/adaptive oracle is a TODO), and that retrieval and a
  fixed golden slice are complementary across qtypes.

**Cross-version trend (reader lens, oracle ±2-slice vs BM25 retrieval as context grows):**

| | pm32k | pm128k | pm1m |
|---|---:|---:|---:|
| trivial (base) | 0.435 | 0.385 | 0.356 |
| oracle ±2-slice | 0.683 | 0.487 | 0.379 |
| naiverag BM25 | 0.601 | 0.494 | 0.438 |

The fixed ±2-turn oracle **degrades as context grows** (0.68→0.49→0.38) — at 1M the window is a
poor approximation of a dispersed reference — while **retrieval scales** and overtakes it
(pm32k oracle≫rag → pm128k tie → pm1m rag>oracle by +6pp). Two reads: (a) retrieval is the
robust token-memory method as context grows; (b) my fixed-window oracle is an imperfect ceiling
at long context (widen/adapt the window, or trust the token-located slice less at 1M — TODO).
Base accuracy also falls with version (0.44→0.39→0.36) → more memory headroom at 1M.

**Net token-memory-baseline result (the #1 documented gap, now filled for pm32k+pm128k+pm1m):**
realistic retrieval lifts a frozen reader **+11–17pp** over no-memory, demographics **hurt**,
and there remains a gap to the focused oracle on pm32k (8pp) that closes on pm128k. This is the
quantified bar the `+per-user weights` arm must beat.

**`+per-user weights` arm — 7 personas (naive minimal recipe).** Per-persona LoRA via direct CE
on that persona's own user-turns (no teacher; r32 all-proj, 3 epochs; merged for serving —
`scripts/general/train_peruser.py`), evaluated apples-to-apples on that persona's pm128k MCQs. Δ =
per-user − base, per persona:

| pid | base PPL | PU PPL | ΔPPL | base rdr | PU rdr | Δrdr | naiverag rdr |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.409 | 0.338 | −0.071 | 0.370 | 0.351 | −0.019 | 0.422 |
| 4 | 0.320 | 0.469 | **+0.150** | 0.408 | 0.381 | −0.027 | 0.503 |
| 1 | 0.457 | 0.400 | −0.057 | 0.471 | 0.236 | −0.236 | 0.586 |
| 3 | 0.255 | 0.483 | **+0.228** | 0.379 | 0.469 | +0.090 | 0.538 |
| 5 | 0.348 | 0.376 | +0.028 | 0.382 | 0.258 | −0.124 | 0.489 |
| 12 | 0.434 | 0.469 | +0.035 | 0.443 | 0.336 | −0.106 | 0.575 |
| 16 | 0.374 | 0.288 | −0.086 | 0.345 | 0.237 | −0.108 | 0.532 |

**PPL (UserSim) lens — high-variance, slight positive tilt, not a reliable win:** per-user beats
base on **4/7** personas; **meanΔ = +0.032**, but **range [−0.086, +0.228], sd 0.110** — the
sign flips persona to persona (pid3 +0.228, pid16 −0.086). This is exactly the legacy R1b
heterogeneity (per-persona dynamics differ qualitatively; R1b needed **per-persona best-step
selection** — 95.7% best-step vs 71.7% final — which this fixed-3-epoch recipe does NOT do). So
naive per-user CE is a near coin-flip with a small mean gain; the upside is real (pid3/pid4
+15–23pp) but unreliable without per-persona stopping or the OPD recipe.

**Reader/generation lens — per-user consistently loses:** meanΔ = **−0.076** (6/7 negative), and
per-user reader (0.24–0.47) is far below naiverag reader (**0.42–0.59**). The user-turn LoRA
degrades instruction-following (legacy "user-only-loss LoRA hurts direct-ask"), so the correct
lens for the per-user arm is PPL, and **token-memory (retrieval) is the robust winner whenever
the model must answer**.

**Reads:** (1) **token-memory/retrieval is the robust reference** across personas + lenses;
(2) **naive per-user weights**: small +0.032 mean PPL gain but high variance — *not* a reliable
win as-is; (3) the documented path to a reliable per-user win is **OPD distillation + dual-rate
LoRA + per-persona best-step** (R1b), and/or fewer epochs — the clear next experiment. n=7
personas, 1 bench (pm128k), 1 recipe; aggregate claims need the per-persona-stopping variant.

**Cross-version (memorization vs learning).** The pm128k-trained pid0/pid4 LoRAs evaluated on
those personas' **pm1m** MCQs (different events, PPL lens): pid0 ΔPPL **+0.045** (n=176), pid4
**+0.037** (n=163) — **both positive on unseen events**. So even naive CE captures a small
*transferable* persona signal (not just memorized turns), echoing R1b's "LoRA learns the persona
fingerprint, not the events." Striking: pid0 *hurt* in-version (−0.071) yet *helps* cross-version
(+0.045), and pid4's +0.150 in-version shrinks to +0.037 cross-version — i.e. part of the
in-version effect is event-specific, but a small (~+0.04) transferable component survives for
both. Small but real, and consistent with the persona-swap probe (~99% generic style, a thin
persona-specific layer).

**Caveats.** The `+per-user weights` arm is the naive recipe (7 personas, fixed 3-epoch CE, no
per-persona stopping — high variance, meanΔPPL +0.032; above). PersonaMem-v1 is substantially
answerable with no persona (trivial reader 0.385–0.435 ≫ 0.25), so headroom is modest.
`suggest_new` stays low everywhere (open-ended qtype). dense retrieval is CPU-bound (12 min vs
BM25 12 s on pm32k) so pm128k used BM25 only. Results JSON + per-record preds under
`experiments/results/<domain>/` + `$UWM_RUNS/<run_id>/`.

### 2026-06-03 — repo reorganized to all-purpose; operating layer restored
- Upstream `810e9dc` moved the PersonaMem prototype to `legacy/general_personamem/`
  and digital-twin code to `legacy/health_digitaltwin/`; added `data/education/`,
  `project_summary.md`, `docs/`.
- Restored/aligned the operating layer: method-agnostic `CLAUDE.md`,
  project-specific `CONVENTIONS.md`, and the `baselines/ common/ experiments/ scripts/`
  scaffolding (general-domain active build).
- No experiments run yet under the new framework.
