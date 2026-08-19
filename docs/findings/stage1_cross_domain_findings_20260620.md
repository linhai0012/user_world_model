# Stage-1 cross-domain findings — general / health / education (2026-06-20)

> Synthesis of the unified-framework Stage-1 intrinsic-prediction ablation
> (`project_summary.md` §6) run across all three domains. One recipe (frozen base →
> +context → +per-user-weights, with trivial/shared/foreign **controls**), instantiated
> **independently** on three unrelated tasks with non-overlapping users. Headline numbers
> mirror `EXPERIMENTS.md` (2026-06-17 general; 2026-06-20 health + education) and
> `experiments/results/{health,education}/*.json`.

> **Status: exploratory.** Each cell below is one dataset, one recipe, one seed — a snapshot of
> what these particular setups produced on 2026-06-20. Nothing here establishes that a component
> of the design works or does not work; the nulls in particular are nulls *for the recipe tried*,
> and the stronger recipes that have moved these metrics elsewhere (per-persona best-step
> selection, soft-label distillation) have not been run in health or in the new general harness.
> Read this as a map of where to look next, not as a verdict.

## The one-line read

**Across these three setups, "context / memory / per-user weights help" gains mostly shrank once a
control was added — and the part that survived is where to look next.** Stuffing a frozen model
with profile/memory, or training naive per-user LoRAs, repeatedly failed to clear trivial
baselines, a shared-population model, or content/length-matched context controls. What did
survive: in education, ordered relevant dialogue history stayed ahead of its own scrambled version
(memory < shuffled by ~0.25 nats). The working hypothesis this suggests — not a conclusion — is
that whether personalization shows up depends on the mechanism *and* on having a task with
per-user-distinguishable structure, and that both need checking before either is written off.

## The unified ablation, three domains

| | **general** (PersonaMem-v1) | **health** (PMData digital-twin) | **education** (KCL tutor chats) |
|---|---|---|---|
| UWM head | reply / preference (MCQ-PPL, reader acc) | next-state MAE (+ reaction NLL) | next student-turn NLL |
| target GT | real MCQ answer | **real** wellness self-report | **real** student turn |
| users | 20 personas | 16 participants (12 data-rich) | none (no learner id) |
| frozen base + context | no gain observed (trivial 0.389 ≥ oracle 0.326, "see≠use") | no gain observed (1.12 vs persist 0.54; +current worst at 1.26) | partial gain (−0.9 NLL; ~0.25 survives the control) |
| per-user weights | naive recipe high-variance (meanΔ +0.03, sd 0.11); the recipe that moved it was OPD+dual-LoRA+best-step (R1b, legacy) | ≈ shared-LoRA (0.506 vs 0.499) → no per-user gain *under this recipe* | not run (no learner identity in the data) |
| control used | reader-vs-PPL; retrieval vs oracle | per-user-mean lookup; **shared-LoRA** | **shuffled** (content+len-matched) + foreign |
| what the control showed | retrieval below oracle (~8pp); demographics scored below no-context | the gain tracks the population model, not the individual | memory ahead of shuffled — small; foreign is length-confounded |

## Per-domain, in one paragraph each

**General (prior work, 2026-06-17).** Frozen Qwen3-4B + injected memory did not beat no-memory
under MCQ-PPL (the model "sees but doesn't use"); the *same* model asked to *answer* did use the
golden reference (+25pp), so the PPL/reader split is the notable part. Realistic retrieval was the
most robust token-memory method here (+11–17pp for a reader) but stayed ~8pp below a focused
oracle; demographics-only profile scored below no-context. Naive per-user LoRA came out near a
coin-flip (meanΔPPL +0.032, sign flips per persona) — the legacy recipe that moved this metric used
OPD distillation + dual-rate LoRA + per-persona best-step, which the new harness has not rerun.
→ some per-user signal appears present; how much depends on a recipe not yet tested here.

**Health (this session).** Predict next-day wellness state (6 real self-report fields). The frozen
LLM was ~2× worse than persistence (1.11 vs 0.54) and conditioning did not help — **+current was
the worst arm** (the frozen model did not anchor on today's state): the same shape as "see≠use". A
per-user LoRA halved the frozen base (0.548) but did not clear a per-user-mean lookup (0.521),
i.e. it looks like it learned the level; training with +current cleared every trivial bar (0.506) —
yet a **single shared LoRA over all users matched/beat it (0.499)**, so what improved tracks a
*population* skill ("learn to use the current state") rather than the individual. Signal is
concentrated in `readiness`; `mood`/`stress` are near-constant. The reaction *text* head moved the
other way from the state head (context helped NLL a little, profile most) — worth noting, but the
reaction text is GPT-synthesized, so that comparison is soft.

**Education (this session).** Predict the student's next turn (NLL); 66 KCL Langfuse tutor chats,
433 student turns, **no learner identity** → frozen base/+memory only. Real conversation memory
lowered NLL ~0.9 nats. Two controls decompose it: vs **shuffled** (the same turns scrambled —
content+length-matched) memory stayed ahead by ~0.25 nats → the *ordered, coherent* history was
still doing something; vs **foreign** (a different, longer session) memory looked worse, but that
is a length confound. So ~0.25 nats tracks ordered-relevance and the rest (~0.6) tracks in-domain
priming/length (depth is consistent: late −1.18 vs early −0.43). Of the three setups, this is the
only one where the effect survived its control — on 66 sessions, unreplicated.

## What this suggests for the project (axis A: cross-domain UWM method paper)

1. **The controls are as much of the contribution as the methods.** per-user-mean, shared-LoRA,
   foreign-memory, reader-vs-PPL — each one changed the sign of a naive positive. Any
   "personalization/memory helps" claim should ship with its control (P-OPSD learned this the hard
   way; the same thing happened again in health and education).
2. **Where per-user signal shows up is still open, and currently looks most promising in general.**
   Health's between-user dynamics look weak on this dataset (shared ≈ per-user); education's data
   has no learner identity at all — but neither has been tested with the recipe that moved the
   metric in the legacy general work or in the edu-exam track. A defensible current framing:
   *lead* the per-user story with general, treat health/education as cross-domain stress tests, and
   revisit both once a stronger recipe has been tried in each.
3. **The open question these Stage-1 runs set up:** can the full mechanism (structured profile +
   RAG memory + per-user weights via staged init) clear these controls in a domain that has
   per-user-distinguishable structure? Answering that — rather than adding more naive arms — is the
   Stage-1→method bridge.

## Honest caveats / TODO

- Health reaction text and the digital-twin event text are GPT-synthesized (grounded in real
  state/HR) → soft signals; the real-GT heads are health next-state (MAE) and the student turn.
- Education: the **shuffled** control (same turns, scrambled) is content+length-matched and is the
  clean test (memory > shuffled → genuine but small relevance); the **foreign** control is *not*
  length-matched (a whole, longer session), so its "beats memory" is a length artifact. An
  LLM-judge generation lens + public **StudyChat** breadth remain TODO.
- Health per-user used a fixed 3-epoch CE (no per-pid best-step — general's R1b showed that
  matters); shared-null makes this moot for *personalization* but a stronger recipe could still
  lift the population model.
- Breadth: pair education with public **StudyChat**; health has no public analog wired in yet.
- Not run (deprioritized after the shared-null): the per-user +profile / +current+prof grid.
