# Project Summary — All-Purpose User World Model

> One extensible **per-user world model** that simulates how a *specific* user
> reacts, serving personalized agents across **general**, **health**, and
> **education** domains — and designed so new domains can be added without
> changing the architecture.

**Status (2026-06-03):** architecture converged; repo reorganized for the new
direction. The prior general-domain prototype is archived under
[`legacy/general_personamem/`](legacy/general_personamem/); reusable
health-domain code is imported under
[`legacy/health_digitaltwin/`](legacy/health_digitaltwin/); private education
dialogue data is under [`data/education/`](data/education/). The new framework
implementation has **not** started yet — this document and
[`docs/uwm_framework_discussion.html`](docs/uwm_framework_discussion.html) are
the spec to build from.

---

## 1. Vision

Build **one** user world model that powers personalized agents in three domains,
unified at the *representation* layer (profile + memory), not the weight layer:

| Domain | What we simulate | Data |
|---|---|---|
| **general** | user's reply / preference reaction to an assistant | PersonaMem-v1/v2 |
| **health** | user's textual reaction + next wellness state after an activity (NOT raw physiological signals) | PMData + LLM synthesis |
| **education** | learner's reaction to a tutor + exam/assessment behavior | private KCL course chats + public anchors (StudyChat, MathDial, EEDI…) |

The core keeps the original repo's bet — a **per-user LLM** that simulates the
user — and adds three pieces: a **user profile**, a **user memory**, and
**structured output** (reaction text + user-state description).

## 2. Unified I/O schema

```
input  = ( user_profile , retrieved user_memory , context , agent_action )
output = { reaction_text ,            # what the user says (all domains)
           next_state }               # user state AFTER the action (optional, structured)
```

- `next_state` makes it a **world model** (predicts a *state transition*, not just an utterance) and **closes the loop**: predicted state → write back to `profile.dynamic` → multi-step rollout.
- **Education splits symmetrically:** *edu-reaction* (≈ general) + *edu-exam* (≈ health's structured state). So health's `state` head and education's `exam` head are two instances of one mechanism.

## 3. Core architecture — three stores

**Guiding principle (keeps the project on-theme, avoids agentic-memory creep):
no component makes an LLM-driven *write* decision. All writes are dumb; all
intelligence is on the read/simulate side — the world model itself.**

| Component | Stores | Write | Read |
|---|---|---|---|
| **per-user weights (LoRA)** | *disposition*: style, preference priors, constitution/knowledge baseline | periodic batch re-distillation | parametric |
| **user profile** | structured JSON: static invariants + dynamic current state | **deterministic recompute** (script / feature-extraction; KT for mastery) | model reads & interprets |
| **user memory** | *episodic* facts/events | **append-only** (every turn / activity / exam attempt) | RAG by current context |

- **Boundary between profile and weights = verbalizability.** Anything writable as a slot → profile (so it's cold-start-injectable, editable, inspectable); what can't be verbalized (style, taste, gestalt) → weights.
- **Invariants (gender/race/birth_year) go in profile, not weights** — cold-start must inject them without training; "store the root, derive the view" (e.g. `age = now − birth_year`) removes most "update" problems.
- **Health dynamic profile = raw physiological features** (model learns to interpret, Health-LLM-style serialization); self-reports (fatigue/mood) are training *targets*, not inputs.

## 4. Initialization — staged population → group → user

Follows **PROPER** (arXiv 2503.01303, our prior work): residual *freeze-and-merge*
— each level is merged into the backbone and frozen, the next level learns only
the delta. All LoRA post-training; **no from-scratch pretraining**, so it is cheap.

- **population** — broad shared user-LM. Needs *breadth* (pool WildChat + cross-domain data), not just PersonaMem's 20 personas. The existing teacher SFT (R3) ≈ this stage already (matches UserLM, arXiv 2510.06552). Open ablation: Base vs Instruct.
- **group** — **learned** clusters (PROPER's user-aware router + diversity loss), *not* hand-defined demographic buckets. New contribution: the structured profile enables **cold-start group assignment with zero user data** (PROPER's router needs a trained user embedding).
- **user** — residual LoRA on the frozen group backbone + soft β-mix of group experts (> hard nearest-group).
- **Two modes:** *cold-start* (profile → nearest group backbone + empty memory) and *online* (streaming updates). They are one continuum (cold → warm → hot as data accrues).

## 5. Training objective

Distillation's role **shrank** once memory was added: the original OPD existed
because the student was context-starved (no history); memory now feeds history
directly, so direct SFT (proven by the digital-twin) becomes the backbone.

- **SFT (CE on ground-truth) is the backbone** at all stages — no teacher-ceiling risk.
- **Distillation is optional**, added only when the teacher has genuine privilege SFT cannot capture: oracle/full memory vs the student's bounded RAG; post-hoc labels (health wellness / edu outcome); or a stronger teacher. (OPSD's GT-injection ≈ smoothed on-policy SFT — adds on-policy benefit, little extra knowledge.)
- **Structured state output → pure SFT** (exact supervised label); **free-text reaction → CE + optional KL**.
- **Online:** `Loss = CE(GT) + λ·KL(student ‖ frozen group teacher)`. The KL term is an **anti-forgetting trust-region anchor** (LwF-style), not imitation — so teacher quality doesn't matter there. Novel episodic content is appended to memory and **gated out of the weight loss** via a novelty / memory-servability signal; weights learn disposition + *the skill of using memory*, never the fact itself.

## 6. Two-stage evaluation

- **Stage 1 — intrinsic prediction (recall + extrapolation).** Does the model predict the user accurately? Text reaction (MCQ-PPL / NLL / judge) + structured state (per-field error; answer accuracy / distractor match). *recall* tests **memory**; *extrapolation* tests the **disposition weights** — the split is a falsifiable test of the "facts→memory, style→weights" hypothesis.
- **Stage 2 — agent collaboration (downstream utility).** Does **agent + UWM** beat the agent alone? Main channel = UWM as a **planner**: agent proposes N candidate actions → UWM predicts each action's reaction/next-state → agent picks the best-predicted outcome. (Generalizes the repo's PPL candidate-scoring to full agent loops; cf. UserLM showing realistic simulators change measured agent performance.)

## 7. Repo layout

```
README.md / project_summary.md      this spec
docs/                               framework discussion (html) + reference pdfs
data/education/                     private KCL course chat data (chat_nlp/chat_ai)
legacy/
  README.md                         entry point describing all legacy resources
  general_personamem/               archived general-domain OPD/OPSD prototype (the prior repo)
  health_digitaltwin/               reusable code imported from LLM-based-Digital-Twins
```

## 8. Status & next steps

- **Done:** general-domain prototype (R1b, +95.7% best-step closure on 20 personas) → archived in `legacy/general_personamem/`; health pipeline + state-token model imported in `legacy/health_digitaltwin/`; design converged.
- **Next:**
  1. Pick the first domain to build the full stack on (recommend **general** first — most mature, fullest eval — then **health** as the cross-domain check).
  2. Ablation skeleton in general: `base` vs `+profile` vs `+memory` vs `+per-user weights` — quantify the marginal value of parametric per-user adaptation on top of explicit profile+memory (a publishable question on its own).
  3. Health: strip the HR-token branch from the digital-twin V3, wire in profile/memory, validate cross-domain.

## 9. Key references

- **PROPER** — staged population→group→user personalization · [arXiv 2503.01303](https://arxiv.org/abs/2503.01303)
- **UserLM "Flipping the Dialogue"** — population user-LM recipe · [arXiv 2510.06552](https://arxiv.org/abs/2510.06552)
- **LLM-based-Digital-Twins** — sibling health project (imported in `legacy/health_digitaltwin/`)
- **Health-LLM** [arXiv 2401.06866] · **PH-LLM** [arXiv 2406.06474] — signal→self-report-state
- **Datasets** — PersonaMem-v1 (`bowen-upenn/PersonaMem`); PMData (Simula); StudyChat, MathDial, EEDI/NeurIPS2020
