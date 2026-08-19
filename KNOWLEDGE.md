# KNOWLEDGE — reading notes & direction decisions

> Paper notes and research-direction decisions for this repo. Experiment records live in
> [`EXPERIMENTS.md`](EXPERIMENTS.md); the design spec is [`project_summary.md`](project_summary.md);
> cross-run syntheses are in [`docs/findings/`](docs/findings/).
>
> **Project stage: early and exploratory.** Entries here are working positions — the reason a
> direction was taken at a point in time, plus what would change it. They are not settled
> conclusions, and a decision recorded here can be reopened by one experiment.

---

## 1. Direction decisions

### 2026-06-20 — three separate domains, one recipe; per-user modelling is in scope from the start

The axis is a **cross-domain user-world-model method**: one recipe (architecture + eval contract +
two heads, `reaction_text` + `next_state`) instantiated **independently** on general / health /
education. The domains stay fully separate — non-overlapping users, unrelated tasks, no pooling.
"Unified" refers to the method, not to a single model over merged data.

**Per-user weights are in scope in every domain from the beginning**, starting from the simplest
per-user model and validating upward, rather than deferring them until profile+memory are settled.
Staged init (population → group → user, PROPER-style) is the destination, not the entry point.

*What would change this:* if per-user arms keep failing to clear shared/trivial controls across
domains *after* stronger recipes have been tried, the paper's centre of gravity moves from
"per-user weights" to "profile + memory + the control methodology".

**Not this repo's axis:** the EMNLP-26 demo (`emnlp26_demo` / "colearn", `emnlp2026_demo2` / "Cue")
and the co-evolution / CoELoVE direction. Those are borrowed references kept in `docs/external/`
for background; they should not steer the general/health/education work programme.

### 2026-06-17 — ship every arm with its control

Repeatedly, a naive positive changed sign once a control was added (per-user-mean lookup,
shared-LoRA, foreign/shuffled context, reader-vs-PPL). Working position: a gain measured only
against "no context" or "frozen base" is not yet evidence, and results are reported with the
control they were measured against. The symmetric half matters just as much — **a null under one
recipe is not evidence that an arm cannot work**, so nulls are recorded with the recipe that
produced them.

### 2026-06-20 — report θ-recovery, not binary-outcome NLL, for choice-style heads

From the edu-exam track (`legacy/education_parametric_memory/`): binary-outcome NLL is
underpowered — near mastery ≈ 0.5 an oracle barely beats chance under Bernoulli noise, and this
produced a "shared beats per-user" reading that reversed once `mastery_corr` / θ-MSE were used
instead. Carry the metric lesson into any domain whose head is a choice among known options.

---

## 2. Reference notes

| Work | Why it matters here |
|---|---|
| **PROPER** — staged population→group→user personalization · [arXiv 2503.01303](https://arxiv.org/abs/2503.01303) | Our prior work; the residual freeze-and-merge staging this project's init plan follows. Its router needs a trained user embedding — the structured profile is our proposed way to get cold-start group assignment without one. |
| **UserLM, "Flipping the Dialogue"** · [arXiv 2510.06552](https://arxiv.org/abs/2510.06552) (PDF in `docs/refs/`) | Independent recipe for a population-level user-LM; close to our teacher-SFT (R3) stage. Also evidence that a more realistic user simulator changes measured agent performance — the motivation for Stage-2. |
| **Health-LLM** [arXiv 2401.06866] · **PH-LLM** [arXiv 2406.06474] | Signal → self-report-state prediction; the serialization style the health dynamic profile borrows (model reads raw features rather than pre-digested summaries). |
| **PersonaMem-v1** (`bowen-upenn/PersonaMem`) | The general-domain benchmark. Note for interpretation: a large fraction is answerable with no persona at all (trivial reader 0.385–0.435 ≫ 0.25 random), so headroom on it is modest. |
| **PMData** (Simula) | Source of the health domain via the digital-twin pipeline. Its day-to-day wellness dynamics look weak (persistence is a strong bar) — a property to verify before reading any health null as a method result. |
| **StudyChat / MathDial / EEDI / EdNet** | Candidate public education anchors. Currently unused: the private KCL chats have no learner identity, so a public set with one is the obvious way to enable an education per-user arm. |

---

## 3. Open questions being carried

1. Does the full mechanism (profile + RAG memory + per-user weights) clear the trivial/shared
   controls in a domain that has per-user-distinguishable structure?
2. Does the legacy general-domain recipe (OPD distillation + dual-rate LoRA + per-persona
   best-step) reproduce inside the new harness, against the token-memory baselines it never had?
3. Is health's weak per-user signal a property of PMData or of the recipe? (The dynamics
   diagnosis in `scripts/health/analyze_health_dynamics.py` is meant to separate these; its output
   has not been recorded yet.)
4. Can education get a learner identity — a public set, or a defensible session-level proxy —
   without which the domain cannot host a per-user arm at all?
5. What does Stage-2 (UWM as a planner scoring candidate agent actions) actually measure, and on
   which domain is it cheapest to try first?
