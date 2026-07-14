# Note — Where Cue's parametric user model sits in the Knowledge-Tracing lineage

> Background / positioning note for Mode A. **Not** a claim that Cue contributes a new KT
> algorithm — KT is (a) the lineage that justifies our design choices, (b) the home of the
> `baseline` (BKT) control, and (c) the related-work frame for the paper. Citations below are
> from memory (knowledge cutoff 2026-01) — **verify exact years/venues before paper use.**

## 1. Why KT is the right frame

Cue's user model is a hidden-state estimator: a latent learner state theta (mastery +
misconceptions) is never observed; we see only graded answers; we reconstruct theta from them.
That *is* knowledge tracing. Our three backends are three points in KT's representation space
(§4), and several of our design choices are KT concepts under new names (§5).

## 2. The lineage (concept by concept)

| Era | Model | Core idea | What it added | Limitation |
| --- | --- | --- | --- | --- |
| 1960s | **IRT** (Rasch; Birnbaum) | `P(correct)=σ(a·(θ−b))`: latent ability θ, item difficulty b, **discrimination a**, guess c | the latent-trait framing; **item discrimination**; **multidim-IRT** = θ as a skill vector | **static** — no learning over time |
| 1994 | **BKT** (Corbett & Anderson) | per-skill HMM, binary known/unknown, `{p_init,p_learn,p_slip,p_guess}` | models **learning over time**; interpretable | binary; skills independent; classic = no forgetting |
| 2006–09 | **AFM / PFA** (Cen; Pavlik) | logistic over opportunity / success / failure counts | additive, interpretable, easy to fit | no latent dynamics; hand-built features |
| 2015 | **DKT** (Piech et al., NeurIPS) | LSTM over the (skill,correct) sequence → next `P(correct)` | first **deep** KT; distributed state; accuracy jump | loses per-skill interpretability; prediction-consistency issues (Yeung 2018) |
| 2017 | **DKVMN** (Zhang et al., WWW) | key–value memory: keys=concepts, values=evolving mastery | restores some per-concept readout | still opaque internals |
| 2019–20 | **SAKT / AKT / SAINT** | attention/Transformers over history | **AKT**: Rasch (IRT) embeddings + monotonic attention decay (=forgetting); **SAINT**: enc–dec, separate exercise/response streams | data-hungry; marginal gains (see §3) |
| 2016–19 | **HLR / DAS3H** (Settles; Choffin) | forgetting curves / spaced-repetition half-life; temporal features on IRT/PFA | explicit **forgetting** | orthogonal to mastery estimation |
| 2022–23 | **pyKT / simpleKT** (Liu et al.) | fair-eval benchmark; a deliberately simple strong baseline | **honesty check**: many deep-KT gains evaporate under fair eval | — |
| 2023→ | **generative / LLM-KT** | predict the *actual response*, LLMs as the model of the student | beyond binary `P(correct)` | new, unsettled |

## 3. The honesty lesson the field already learned

`pyKT` + `simpleKT` showed that, under fair evaluation, elaborate deep-KT models often fail to
beat a well-tuned simple baseline. This is the same discipline as our **shared-LoRA null**
(per-user must beat shared + no-input) — so we keep BKT (baseline) and the A0/A∅ controls, and
report **final-step** numbers. A "parametric memory wins" claim has to survive this bar.

## 4. Where Cue's three backends sit (the load-bearing positioning)

Classic KT = **one shared model + per-student latent state inferred from the interaction
sequence (in context)** — the student lives in the *hidden state / attention context*. Cue
adds a different axis: *where does the per-student information physically live?*

| Backend | Per-student info lives in… | KT analogue |
| --- | --- | --- |
| classic DKT/AKT | shared weights + **context/sequence state** | the mainstream |
| `baseline` (BKT) | shared structure + a **per-skill state table** | explicit KT |
| **`parametric` (ours)** | **per-user weights (a LoRA)** | per-student *parametric memory* |
| `agentic` | **free-form text notes** | per-student *token memory* |

So Cue's headline (parametric vs agentic) is a concrete instance of **context/token-memory vs
parametric-memory** for the learner model, with BKT and classic KT as reference points. This is
the framing that makes the demo a *system* contribution rather than a KT-algorithm contribution.

## 5. Our design choices = KT concepts under new names

- **discrimination gate** (`streams.discrimination`) ≈ IRT's **item discrimination `a`** — keep
  items that separate learners; drop degenerate ones (no theta signal).
- **persona theta** (`personas.py`: ability + topic offset + skill jitter) ≈ **multidimensional
  IRT ability**; the learning dynamics (mastery rises, misconception repairs) ≈ **BKT transit**.
- **shared forgetting decay** (the app's `decay.py`) ≈ **HLR / DAS3H**.
- **held-out eval + isotonic calibration** = the fair-eval discipline `pyKT` argued for.

## 6. What the parametric per-user LoRA does *differently* from classic KT (the mild novelty)

1. **Per-user parametric memory, not shared-model+context.** Classic KT trains one model and
   carries the student in a hidden state / attention window. We bake the student into **separate
   weights** (a per-user LoRA) — the *parametric vs context memory* thesis, made interactive.
2. **Generative response prediction, not binary `P(correct)`.** We model the **full answer
   distribution** — *which* distractor a learner picks — via choice-perplexity / `option_logprobs`.
   Mainstream KT collapses the response to right/wrong and throws the distractor (the
   misconception signal) away. Predicting the chosen wrong option is where the learner-specific
   signal lives (and where the base model's correct-answer prior cannot leak).
3. **Amortized recovery.** The LoRA is an amortized neural estimator of theta — the high-capacity,
   generative end of the IRT→DKT line — but evaluated as *reconstruction of behaviour*, not
   recovery of a symbolic state (that's BKT's job).

## 7. Scope guard

We are not proposing a new tracer. BKT is the control; KT is the related-work frame; the
contribution is the interactive proactive **system** + the parametric-vs-agentic-memory contrast
+ honest evaluation. Keep the framing in §4 — it is what separates "a demo" from "yet another KT
model."

## 8. Citations to verify before paper use
IRT: Rasch 1960; Birnbaum 1968. BKT: Corbett & Anderson 1994 (UMUAI). AFM: Cen et al. 2006;
PFA: Pavlik et al. 2009. DKT: Piech et al. 2015 (NeurIPS); reg.: Yeung & Yeung 2018. DKVMN:
Zhang et al. 2017 (WWW). SAKT: Pandey & Karypis 2019 (EDM). AKT: Ghosh et al. 2020 (KDD). SAINT:
Choi et al. 2020 (L@S); SAINT+: Shin et al. 2021. GKT: Nakagawa et al. 2019. HLR: Settles &
Meeder 2016 (ACL). DAS3H: Choffin et al. 2019 (EDM). pyKT: Liu et al. 2022 (NeurIPS D&B);
simpleKT: Liu et al. 2023 (ICLR).
