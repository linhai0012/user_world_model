# UserSim — Verbal / Generative Evaluation Attempts

Summary of every non-PPL evaluation we've tried on the UserSim (teacher
SFT across R1 / R3, and Phase 2 per-user student LoRAs), focusing on
setups where the model **generates a user-voice response** and that
response is then **judged** (by an LLM or by programmatic metrics).

The project has two deployment paradigms for the UserSim:
- **(I) NLL-based scoring** (plan §9.5): teacher/student assign
  conditional PPL to each MCQ choice; lowest wins. Currently the primary
  working end-to-end metric.
- **(II) Verbal-feedback Agent** (plan §9.3): UserSim generates a
  natural reaction to each choice, an Agent reads them and picks. **No
  working implementation** as of now.

This doc covers everything that touches paradigm (II) plus related
generative-quality checks.

---

## 1. Three LLM-judge attempts

### Attempt 1 — Agent v1 (R1, `eval_mcq.py`, EXPERIMENTS §4.4.1)

**Pipeline**

```
MCQ question + 4 choices (a,b,c,d)
  → UserSim generates 4 reactions (one per choice)
  → OpenAI Agent reads (question, choice, reaction) × 4
  → Agent outputs "Answer: (X)"
```

**Result** (n=20): base Qwen3-4B = **0.45 (9/20)**, R1 SFT ckpt-50 =
**0.30 (6/20)** — SFT was **worse** than base.

**Diagnosis**: base had three hidden advantages the Agent exploited:
1. `<think>` blocks leaked in from HF `apply_chat_template` (26% of
   reactions contaminated; Qwen3-4B hybrid's template hard-codes
   `<think>\n\n</think>\n\n`).
2. Prompt echoing — base frequently copied the MCQ question verbatim,
   giving the Agent a side-channel to read.
3. Length — base reactions were longer (p50 916 vs 583 chars),
   providing more surface for the Agent to score.

### Attempt 2 — Agent v2 (R1, same pipeline, EXPERIMENTS §4.4.2)

**Fixes applied**:
- Replaced `apply_chat_template` with manual ChatML encoding (no
  `<think>` injection; matches training format).
- Appended a reaction primer
  `"\n\nWhat do you think — does that match what you're looking for?"`
  to each choice to solicit a direct reaction rather than monologue.

**Result** (n=20): base = 0.45 (9/20), R1 SFT = **0.20 (4/20) — worse
than v1**.

**Structural diagnosis (the important finding)**: SFT's reactions turn
out **anti-correlated** with choice correctness.

- Teacher SFT was trained to predict *natural user continuations* given
  full history.
- In PersonaMem's data, user turns often start new threads (`"I also
  tried X..."`), regardless of what the assistant just said.
- Evaluated on MCQs, SFT therefore gives its **least-engaged response
  to the most preference-aligned choice** (which doesn't need
  extension), and its **most-engaged response to off-target choices**
  (where new content flows naturally).
- The Agent picks the most-engaged reaction → picks the wrong choice.

**Concrete example** (MCQ persona=6, qtype=suggest_new, correct=d):
- Choice **d** (correct, flashcards + spaced repetition): SFT reacts
  `"I'm curious about different study techniques. I've been trying to
  find effective methods."` — generic.
- Choice **b** (pomodoro timer, wrong): SFT reacts `"Yes, I like it!
  I also created a detailed study guide using mind maps."` — specific,
  engaged.
- Agent picks b. Correct was d.

**Conclusion**: the UserSim-reaction-then-Agent paradigm does not work
with a user-only-loss teacher SFT. The anti-correlation is structural,
not fixable by prompt engineering.

### Attempt 3 — Similarity judge (Phase 2, `eval_user_gen_judge.py`, EXPERIMENTS §10.4)

**Different task**: not MCQ decision — the LLM rates how similar the
model's generation is to the real user's GT response.

**Pipeline**

```
For each real user turn in held-out data:
  → each condition's model generates one user-turn continuation
  → GPT-4o-mini judges similarity on 1-5 scale:
      5 = essentially same content, preferences, style
      4 = same topic and preferences, stylistic differences
      3 = related but diverges
      2 = same area, different preferences
      1 = unrelated or contradictory
```

Five conditions per persona (see EXPERIMENTS §10.1 for setup):
base_demo, base_full, teacher_demo, teacher_full, student_demo.

**Result** (n=50 per persona, Phase 2 student at step 400, 4 personas):

| condition     | pid 0 | pid 4 | pid 12 | pid 14 | AVG  |
|---------------|------:|------:|-------:|-------:|-----:|
| base_demo     | 1.38  | 1.14  | 1.14   | 1.16   | 1.21 |
| base_full     | 1.90  | 1.86  | 1.96   | 2.02   | 1.94 |
| teacher_demo  | 1.52  | 1.70  | 1.72   | 1.72   | 1.66 |
| teacher_full  | 2.14  | 2.12  | 2.38   | 2.48   | **2.28** |
| **student_demo** | 1.52  | **1.76** | **2.22** | **1.98** | **1.87** |

- Student beats teacher_demo on **3/4 personas** (Lisa / Jordan / Leilani)
  on semantic similarity.
- Student reaches 93% of teacher_full on Jordan, 80% on Leilani.
- **This is the only metric in the project where LoRA's 0-context
  output is qualitatively comparable to teacher's full-context output.**

**Important caveat**: this metric is *quality*, not *decision*. You
cannot directly convert similarity-to-GT into an MCQ answer because
there is no "GT user response" defined for an MCQ's correct choice.
So this result validates that the LoRA produces persona-appropriate
generations but does NOT revive paradigm (II) as an MCQ-decision path.

---

## 2. Two programmatic (non-LLM) generative analyses

### §4.3 qualitative-30 (R1 era)

30 random cases × 4 conditions ({base, SFT} × {no-ctx, with-ctx}), scored
programmatically for:
- `<think>` leakage rate
- `"I also..."` opener rate (PersonaMem-typical template)
- Content-word Jaccard overlap with GT user utterance
- Reaction-word presence in first 100 chars

**Key findings**:
- SFT eliminates `<think>` entirely (0% vs base 17-37%).
- SFT beats base on GT overlap in 20/30 cases (0.076 vs 0.054 mean).
- Context selectively helps SFT on domain-specific recall cases (e.g.,
  cooking fusion, movie snacks) but hurts on average (−0.009 Jaccard).

### §9.5 qualitative-30 (R3 era, vLLM batched)

Same 30-case protocol on R3 final.

**Key findings vs R1**:
- `<think>` now 0% across **all** conditions — R3's Instruct-2507 base
  is architecturally non-thinking, not a training effect.
- Mean GT overlap uniformly higher (sft_ctx 0.096 vs R1 0.067).
- Context now helps SFT slightly on average (+0.004 vs R1's −0.009).
- **`"I also..."` opener rate nearly doubled**: sft_noctx 6.7% → 20.0%,
  sft_ctx 16.7% → 26.7%. PersonaMem's data-distribution pattern bleeds
  through more strongly under Instruct-2507's stronger chat priors.
  **This is a warning sign for paradigm (II)** — any future attempt at
  Agent-based MCQ decision would see even more mode-collapsed reactions.

---

## 3. Comparison table

| Attempt | Era | LLM role | UserSim task | Result |
|---------|-----|----------|--------------|--------|
| §4.4.1 Agent v1 | R1 | MCQ decision-maker | Generate reactions to 4 choices | **SFT worse** (base 0.45 → SFT 0.30); contamination artifacts |
| §4.4.2 Agent v2 | R1 | MCQ decision-maker | Same + primer | **SFT worse** (SFT 0.20); anti-correlated reactions — structural failure |
| §10.4 Similarity judge | Phase 2 | Quality rater | Generate one user continuation | **Student > teacher_demo on 3/4 personas** (partial success, but not a decision path) |
| §4.3 qual-30 (R1) | R1 | — (programmatic) | 30 cases × 4 conds | SFT wins 20/30 Jaccard; `<think>` eliminated |
| §9.5 qual-30 (R3) | R3 | — (programmatic) | Same on R3 | Better overlap; **`"I also..."` rate doubled** — worsens paradigm (II) outlook |

---

## 4. Status of paradigm (II) and open directions

**Current state**: Attempts 1 + 2 are the only direct try at paradigm
(II); both failed under user-only-loss training. R3 qualitative (§9.5)
suggests a retry on R3 would be worse, not better, so we have not
attempted a v3.

**Open research directions** (none implemented):

1. **Structured preference markers** — train UserSim to emit
   `<pref>X</pref>` tags or explicit correction sentences rather than
   free-form continuation. Agent reads tags instead of inferring
   enthusiasm.
2. **Pairwise comparison framing** — have UserSim compare (choice_a,
   choice_b) head-to-head rather than react to each independently.
3. **Add assistant-turn loss during teacher training** — currently
   teacher SFT loss is user-only, which gives the teacher no incentive
   to calibrate assistant-choice preferences. A small assistant-turn
   component could make verbal reactions carry correct-choice signal.
4. **Multi-turn UserSim ↔ Agent dialogue** — one-shot reaction
   aggregates too much ambiguity; let the Agent ask clarifying
   questions.
5. **UserSim direct action** — have UserSim itself output `"I'd choose
   (b)"` explicitly, bypassing the Agent-interpretation layer.

Each is a legitimate ablation. All are deferred until paradigm (I)
main track stabilizes.

---

## 5. What this means for the paper

- Paradigm (I) has **a working implementation**, meaningful numbers
  (54-76% gap closure on MCQ-PPL), and clean ablations.
- Paradigm (II) has **two documented failure modes** (contamination
  artifacts, anti-correlated reactions) and **one adjacent success**
  (LLM-judged similarity to GT).
- The honest framing: "We demonstrated paradigm (I) works; paradigm
  (II) remains an open problem — we report two documented failures
  under the current UserSim training objective, and a redesign involving
  structured output, multi-turn interaction, or teacher-loss adjustment
  is future work."
- The similarity-judge result (§10.4) is the **strongest single piece
  of evidence** that the per-user LoRA has internalized persona-specific
  generation, and is the right qualitative demonstration to accompany
  paradigm (I)'s MCQ-PPL quantitative numbers.
