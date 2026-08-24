# CoPAI B1/B2 — Conceptual Suggestions (Review Notes)

*Date: 2026-08-24. Scope: conceptual/structural suggestions only — assumes the mechanical fixes (Gantt person-months, undefined β_k, WP3 staffing row, template sections, typos/bibliography) are already handled. Items are ordered by expected reviewer impact; each is a self-contained one-paragraph insertion with ready-to-paste draft text. Placeholders are marked ⟨⟩.*

---

## S1. State the design principle explicitly: personalise the model, share the policy

**Where:** B2, T4.2, immediately after Eq. (5); one clause in B1 SQ3.

**Draft text:**

> **Design principle: personalise the model, share the policy.** CoPAI deliberately separates user-specific knowledge from decision-making. All user-specific learning resides in the modelling components: the cognitive-state posterior c_t (WP2), the personal memory M (WP2), and the progressively personalised UWM parameters (WP3). The policy π_θ is shared across all users: its parameters encode pedagogical knowledge — the mapping from learner states to appropriate support actions — which we hypothesise to be largely learner-general, while everything specific to an individual enters the policy exclusively through the state s_t = (c_t, γ_t, m_t); no user identifier is provided. This separation yields (i) sample efficiency and cold start: pedagogical knowledge is estimated jointly from all users, and a new user is served immediately through their state estimate rather than through untrained user-specific weights; (ii) consistency and auditability: users presenting identical evidence receive identical support, and any difference in treatment is traceable to a difference in inferred state; (iii) privacy and unlearning: honouring a data-deletion request requires deleting a user's state and memory, not unlearning policy weights. Where individual variation cannot be captured by state values on shared dimensions, it is absorbed first by the residual subspace of the cognitive profile (T2.1) and then by the user-level UWM parameters (T3.3) — never by user-specific policy weights. Should the T4.4 ablation reveal residual user-grouped signal in policy performance after state conditioning, lightweight per-user adapters can be added; we treat their predicted redundancy as a testable hypothesis.

**B1 SQ3 clause:** after "It selects each action based on the current task, the user profile, the interaction regime, ..." append: *", with all user-specific knowledge entering through the inferred state rather than through user-specific policy parameters."*

**Why:** Pre-empts three predictable reviewer questions at once — scalability ("one policy per user?"), privacy/GDPR, and the state-aliasing objection ("users with identical states but opposite feedback") — and converts the per-user-parameter question from a vulnerability into a falsifiable design decision tested in T4.4.

---

## S2. Domain generalisation: add an instantiation template

**Where:** B2, T4.4 (or end of WP2 Novelty); roughly half a page including the table.

**Draft text:**

> The CoPAI framework is domain-agnostic in its formal core (coupled latent-state dynamics, V-information diagnostics, regime detection, constrained policy optimisation); domain knowledge enters through exactly three plug-in points: (i) the grounded outcome y and its probe design, (ii) the CBM concept space and its anchoring instruments, and (iii) the action space. Instantiating CoPAI in a new domain therefore follows a fixed recipe: specify the capability the user has chosen to develop (defining y), assemble candidate state dimensions from three sources — predictors of y, moderators of action effects on y, and sensors for the failure modes (dependency, echo chamber) — and let the V-information probes prune dimensions that carry no usable information about grounded outcomes. Education is the primary domain not because the framework is education-specific, but because education's mature psychometric infrastructure is the only setting in which this construction procedure can itself be validated (T2.4); the health instantiation (T4.4) then tests the recipe's transfer.

| Plug-in | Education (primary) | Health (T4.4, offline) | General assistant (outlook) |
|---|---|---|---|
| Grounded outcome y & probes | unaided problem-solving; in/near/far transfer items | sustained self-management without prompting; validated longitudinal instruments | unaided task completion; verification of agent errors |
| CBM blocks & instruments | mastery, reasoning style, epistemic state, learning dynamics; CSI, NFC, REI | self-efficacy, planning, activation; PAM, SDT scales | goals/values, trust calibration, reliance habits (construct theory largely open) |
| Action space | hint, scaffold level, questioning, control transfer, task choice | prompt timing, plan granularity, autonomy support | assistance allocation: solve vs. verify vs. withhold |
| Failure-mode sensors | help-seeking spikes, ρ decline, Δ_far collapse | prompting dependence, adherence collapse on withdrawal | verifier-role migration, assisted–unaided gap growth |

**Why:** Converts "generalises beyond education" from a claim into a procedure, and answers "why education first" with a methodological argument rather than convenience.

---

## S3. Align B1 Eq. (2) with B2 Eq. (4)

**Where:** B1, Section 3, Normative objective.

**Suggestion:** Replace the weighted-sum objective max_A E[I_H + λ·I_A] with the constrained form used in B2 — maximise I_H subject to a floor on interaction quality — or explicitly gate the I_A term by the coupling/role-shift conditions described in SQ3.

**Draft text:**

> max_A E[I_H] subject to interaction quality remaining above a usability floor; I_A is monitored as a descriptive quantity and contributes to the objective only when coupled with learner growth (C > 0, rising ρ).

**Why:** The current B1 text states that I_A "is not an intrinsically positive objective", yet the weighted sum rewards I_A unconditionally — a dependency trajectory (high I_A, I_H ≤ 0) still earns λ·I_A. B2's Eq. (4) already has the correct form; the two documents should not disagree on the central objective.

---

## S4. Make the interaction-quality floor per-user

**Where:** B2, WP4, Constrained objective paragraph; one sentence.

**Draft text:**

> The constraint is enforced per user rather than only in expectation over the population, so that no individual learner's interaction quality is traded away for aggregate performance; τ is calibrated from pilot retention data as the quality level below which disengagement is observed.

**Why:** An expectation-level floor permits sacrificing individual users while satisfying the constraint on average — an easy target for the panel's ethics reading, and a one-sentence fix.

---

## S5. WP5: statistical power, comparator ethics, and adaptive-design framing

**Where:** B2, T5.2; three or four sentences.

**Draft text:**

> With n ≥ 200 per arm and baseline-covariate adjustment, the trial detects standardised effects of d ≈ ⟨0.30⟩ on the primary endpoint at 80% power (α = .05, adjusted for the pairwise comparisons), with the recruitment buffer absorbing ⟨20⟩% attrition; effect sizes for far-transfer outcomes are historically small, and the pilot will refine this calculation before pre-registration. Arm (b) instantiates current practice — a standard preference-tuned assistant from the same base model family — rather than a system engineered to maximise satisfaction, preserving equipoise relative to the tools students already use. The policy version deployed in the trial is updated only at pre-registered cohort boundaries, following the adaptive-intervention tradition of micro-randomised designs.

**Why:** The primary endpoint is delayed solo transfer — the outcome family with the smallest effect sizes in the education literature. A trial this central to the proposal cannot omit a power statement; arm (b) additionally needs an explicit equipoise argument, since the proposal's own cited evidence suggests satisfaction-optimised tutoring can harm learning.

**Note:** the numeric placeholders (d ≈ 0.30, 20% attrition) must be replaced by an actual power calculation before insertion.

---

## S6. An explicit identifiability go/no-go gate

**Where:** B2, T2.4 or the Workplan section; two sentences.

**Draft text:**

> An interim identifiability check at M15, using a reduced psychometric battery on accumulated pilot data, tests whether UCA-inferred profiles predict instrument scores above chance before WP3–WP4 commit to the full concept space. If identifiability is not established, the concept space is descoped to the behaviourally grounded mastery block plus proxy diagnostics (cf. R1/R3), and concept-level attribution in WP4 degrades to interval-level credit.

**Why:** The load-bearing validation (T2.4) currently reports at M30, after WP3 (M13) and WP4 (M19) have committed to the CBM space. ERC panels reward high-risk projects that specify falsification points; this converts a timeline vulnerability into a governance feature.

---

## S7–S11. One-sentence stitches

**S7 (T3.1, evaluation):**

> Because the policy consumes comparative judgements between candidate actions, UWM evaluation includes ordinal accuracy — whether the model ranks candidate actions correctly by their effect on the learner state — alongside next-response prediction against Deep Knowledge Tracing baselines.

**S8a (T3.3, missing arrow WP2→WP3):**

> PID insights (T2.2) serve as structured priors for user-level UWM adaptation, so that consolidated knowledge about a learner initialises rather than duplicates the world model's individual parameters.

**S8b (T3.1 or T3.2, estimator disagreement):**

> Persistent disagreement between the UCA estimate and the UWM posterior is itself treated as a signal — triggering probe scheduling and flagging potential model error — turning the redundancy between the two estimators into a consistency check.

**S9 (T2.1, residual-subspace training signal):**

> Systematic cross-user disagreement in feedback at matched cognitive states serves as the discovery signal for residual dimensions: where users whose grounded profiles coincide respond oppositely to the same action, a dimension is missing from the state.

**S10 (T5.3, CogWell-Bench external usage):**

> CogWell-Bench v1 supports three usage modes for external teams: applying its metric suite to their own interaction logs; dynamic evaluation against the released UWM surrogate as a simulated-user harness; and static prediction tasks over the de-identified trajectories (e.g., forecasting dependency outcomes from early sessions).

**S11 (WP4, wording consistency):**

> Throughout WP4, c_t denotes the deployment-time active estimate — the UWM posterior calibrated to the independent UCA reference — resolving the current mixed attribution ("WP2 cognitive state" in the objective, "UWM estimates" in T4.1).

---

## Open decisions (for PI)

1. **S1, final sentence** — whether to cite ongoing preliminary evidence supporting the predicted redundancy of per-user adapters (strengthens the claim, but exposes unpublished work).
2. **S5 numbers** — the power-calculation placeholders need to be computed and filled before insertion.
3. **S10, second usage mode** — releasing the UWM surrogate as a simulated-user evaluation harness is a new commitment (an additional WP3 deliverable in effect); include only if the PI agrees to it.
