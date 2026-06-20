# Stage-1 cross-domain findings — general / health / education (2026-06-20)

> Synthesis of the unified-framework Stage-1 intrinsic-prediction ablation
> (`project_summary.md` §6) run across all three domains. One recipe (frozen base →
> +context → +per-user-weights, with trivial/shared/foreign **controls**), instantiated
> **independently** on three unrelated tasks with non-overlapping users. Headline numbers
> mirror `EXPERIMENTS.md` (2026-06-17 general; 2026-06-20 health + education) and
> `experiments/results/{health_*,edu_*}__*.json`.

## The one-line result

**Across all three domains, the naive "context / memory / per-user weights help" claim does NOT
survive the right control.** Apparent gains from stuffing a frozen model with profile/memory, or
from per-user LoRAs, repeatedly collapse against trivial baselines, a shared-population model, or
a foreign-context control. This is the same lesson three times — and it is exactly the motivation
for the project thesis: real personalization needs the *right mechanism*, not stuffed context.

## The unified ablation, three domains

| | **general** (PersonaMem-v1) | **health** (PMData digital-twin) | **education** (KCL tutor chats) |
|---|---|---|---|
| UWM head | reply / preference (MCQ-PPL, reader acc) | next-state MAE (+ reaction NLL) | next student-turn NLL |
| target GT | real MCQ answer | **real** wellness self-report | **real** student turn |
| users | 20 personas | 16 participants (12 data-rich) | none (no learner id) |
| frozen base + context | **no help** (trivial 0.389 ≥ oracle 0.326, "see≠use") | **no help** (1.12 vs persist 0.54; +current *hurts* 1.26) | "helps" (−0.9 NLL) **but = foreign** |
| per-user weights | naive unreliable (meanΔ +0.03, sd 0.11); reliable only w/ OPD+dual-LoRA+best-step (R1b +95.7%) | **= shared-LoRA** (0.506 vs 0.499) → no personalization | n/a (no learner identity) |
| decisive control | reader-vs-PPL; retrieval vs oracle | per-user-mean lookup; **shared-LoRA null** | **foreign-memory** |
| control verdict | retrieval < oracle (~8pp gap); demographics *hurt* | personalization adds nothing; gain is population-level | gain is priming/length, not relevance |

## Per-domain, in one paragraph each

**General (prior work, 2026-06-17).** Frozen Qwen3-4B + injected memory does not beat no-memory
under MCQ-PPL (the model "sees but doesn't use"); the *same* model asked to *answer* exploits the
golden reference (+25pp), so the PPL/reader split is itself a result. Realistic retrieval is the
robust token-memory method (+11–17pp for a reader) but stays ~8pp below a focused oracle;
demographics-only profile *hurts*. Naive per-user LoRA is a near coin-flip (meanΔPPL +0.032, sign
flips per persona); a reliable per-user win needs OPD distillation + dual-rate LoRA + per-persona
best-step. → per-user signal exists here but is thin and needs the right recipe.

**Health (this session).** Predict next-day wellness state (6 real self-report fields). A frozen
LLM is ~2× worse than persistence (1.11 vs 0.54) and conditioning does not help — **+current even
hurts** (the frozen model won't anchor on today's state): "see≠use" again. A per-user LoRA halves
the frozen base (0.548) but only learns the user's *mean* (loses to a per-user-mean lookup 0.521);
training with +current does beat every trivial bar (0.506) — yet a **single shared LoRA over all
users matches/beats it (0.499)**, so the gain is a *population* skill ("learn to use the current
state"), not personalization. Signal is concentrated in `readiness`; `mood`/`stress` are
near-constant. The reaction *text* head behaves oppositely to the state head (context helps NLL a
little, profile most) — consistent with "context aids text generation, not structured state."

**Education (this session).** Predict the student's next turn (NLL); 66 KCL Langfuse tutor chats,
433 student turns, **no learner identity** → frozen base/+memory only. Real conversation memory
lowers NLL a lot (~0.9 nats) — but a **foreign** session's dialogue as "memory" lowers it as much
or more (memory is +0.4/+0.1 nats *worse* than foreign), and the effect scales with context length
(late turns −1.18 vs early −0.43). So the apparent "memory helps" is **in-domain priming / length**,
not retrieval of this learner's relevant history.

## What this means for the project (axis A: cross-domain UWM method paper)

1. **The controls are the contribution as much as the methods.** per-user-mean, shared-LoRA-null,
   foreign-memory, reader-vs-PPL — each one deflated a naive positive. Any "personalization/memory
   helps" claim in the paper must ship with its control (P-OPSD learned this the hard way; now
   replicated in health and education).
2. **Per-user signal is domain-dependent and currently strongest in general.** Health has weak
   between-user-distinguishable dynamics (shared = per-user); education has no learner identity at
   all. So the per-user-weights story should be *led* by general (with the OPD recipe), and
   health/education framed as **cross-domain stress tests** that show where/why it does or doesn't
   transfer — not as independent personalization wins.
3. **The open question the Stage-1 baselines set up:** can the project's full mechanism
   (structured profile + RAG memory + per-user weights via staged init) beat these controls in a
   domain with genuine per-user signal? That is the Stage-1→method bridge.

## Honest caveats / TODO

- Health reaction text and the digital-twin event text are GPT-synthesized (grounded in real
  state/HR) → soft signals; the real-GT heads are health next-state (MAE) and the student turn.
- Education foreign control is **not length-matched** (it conflates relevance with length); a
  length-matched foreign control + an LLM-judge generation lens are TODO. Foreign helping ≥ real
  memory already rules out a relevance-only story.
- Health per-user used a fixed 3-epoch CE (no per-pid best-step — general's R1b showed that
  matters); shared-null makes this moot for *personalization* but a stronger recipe could still
  lift the population model.
- Breadth: pair education with public **StudyChat**; health has no public analog wired in yet.
- Not run (deprioritized after the shared-null): the per-user +profile / +current+prof grid.
