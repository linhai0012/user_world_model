# `education_parametric_memory/` — entry point

> **The education-domain instantiation of the framework's *per-user weights* store.**
> A per-user LoRA that compresses a learner's graded answer history into weights and predicts
> **how that specific learner answers new items** — i.e. a user world model for *edu-exam*.
> Imported from the sibling repo `parametric_user_memory` (snapshot 2026-07-14).
> **Not archived / not superseded** — the canonical, still-active home is
> `/users/k2480198/parametric_user_memory` (local git, `main`). This copy is reference material,
> in the same sense as [`../health_digitaltwin/`](../health_digitaltwin/).

---

## 1. Why it is in this repo

It fills three slots the framework ([`../../project_summary.md`](../../project_summary.md)) had
specified but not yet instantiated:

| Framework concept (`project_summary.md`) | What this code base provides |
|---|---|
| **Store 3 — per-user weights (LoRA)**: *disposition*, periodic batch re-distillation, **parametric** | A working per-user dual-rate LoRA, batch-retrained at snapshots, that carries the learner's disposition (ability + misconception profile) in weights |
| **Domain: education** — "learner's reaction to a tutor + **exam/assessment behavior**" | The **edu-exam** half: predicting a learner's MCQ answer (which distractor they pick), on AQA GCSE Biology |
| **Stage-1 intrinsic eval** — "structured state (per-field error; **answer accuracy / distractor match**)" | Exactly this: the prediction target is the full **option distribution**, and the latent state (per-skill mastery) is scored by θ-recovery |
| **§8.2 ablation** — `base` vs `+profile` vs `+memory` vs **`+per-user weights`** | The controls are built in: **A∅** = untrained base (no-input) · **A0** = one *shared* LoRA (pooled) · **A1** = per-user LoRA. The headline is A1's lift over A0/A∅ |

It is also the **education-domain sibling of [`../general_personamem/`](../general_personamem/)**:
both train a per-user student by **distillation** (OPD family), so the two are directly comparable
across domains — general (PersonaMem preferences) vs education (MCQ assessment).

> **A note on framing.** In this repo's language the per-user LoRA *is* the user world model (it
> predicts the user's reaction). In the downstream tutoring demos it is consumed as the agent's
> *belief* about the learner (`A_t`), kept deliberately separate from the ground-truth simulator
> (`H_t`) used to score it — so the evaluation cannot be circular.

---

## 2. The reusable part: a pipeline that makes "did the model recover the user?" answerable

The hard problem for any user world model is that the **user's latent state is unobservable**, so
you cannot check whether the model recovered it. This code base solves that by **authoring** the
latent state and rendering the observable behaviour from it — a deterministic, cross-machine
reproducible pipeline (all seeding via md5 `det_seed`, never Python's per-process `hash()`).

```
knowledge structure (skill tree + misconception tags)
  ├─► ① question bank        173 GCSE-Bio MCQs · 13 skills · every distractor misconception-tagged
  └─► ② persona θ (hidden)   24 learners: ability + topic_offset + skill_jitter + held misconceptions
                 ↓  (questions ⊥ personas | knowledge structure)
        ③ controllable generator g(θ,q) → answer      P(correct)=mastery; errors concentrate on the held distractor
                 ↓
        ④ answer streams + identifiability gate       fixed round order; drop non-discriminative items
                 ↓
        ⑤ substrate            persona_set · splits 104/39/30 · round_sequence · streams (2,496) · eval_truth (8,280)
                 ↓
        ⑥ per-user LoRA train + choice-PPL scoring    snapshots [0,13,26,52,104] → artifact bundle
```

**This is the transferable methodology**, not just the biology instance: author a low-dimensional
latent θ, render behaviour with a controllable generator, gate for identifiability, hold out clean
evaluation truth. Any domain where you can write down a latent user state can reuse it.

---

## 3. The model and the four training recipes

Same skeleton throughout — base **Qwen3-4B-Instruct-2507**, **dual-rate LoRA** (slow MLP r32/α64
lr1e-5 + fast Attn r16/α32 lr2e-4), cumulative snapshot training, scored by **choice-perplexity**.
**Only the training target differs:**

| Recipe | Target | Family |
|---|---|---|
| **hard** | token-CE on the learner's chosen-option **text** (open generation; never sees the 4 options) | plain SFT |
| **oracle-g** | soft-CE toward the generator's exact option categorical `g(θ,q)` | **OPD** (privileged) |
| **oracle-hybrid** | soft-CE toward `0.5·g(θ,q) + 0.5·one-hot(realized)` | **OPD** (privileged) |
| **LLM-teacher** | soft-CE toward `0.5·llm_card + 0.5·one-hot(realized)`; the teacher is a frozen base LM reading a learner card (misconception + coarse mastery bucket), **no oracle θ** | **OPD** (semi-oracle) |

The OPD loss is `−Σ_o p_teacher(o)·log q_student(o)` over the 4 options (forward-KL / Hinton
soft-CE). Two structural reasons it beats hard SFT: **(i) information** — a soft `P(correct)=mastery`
target carries far more bits than one Bernoulli sample; **(ii) format match** — it trains a
*closed-set* distribution over the options, structurally identical to the choice-PPL evaluation,
whereas hard SFT trains open generation and never sees the distractors.

---

## 4. Results (held-out eval, pooled @ snapshot 104)

| Recipe | mastery_corr ↑ | θ-MSE ↓ | |
|---|---:|---:|---|
| hard (lean SFT) | 0.159 | 0.130 | baseline; peaks ~0.175 then declines |
| **LLM-teacher** | **0.216** | 0.108 | the *deployable* number (no oracle θ) |
| oracle-hybrid | 0.494 | 0.051 | **privileged upper bound** |
| oracle-g | 0.570 | 0.054 | **privileged upper bound** |

- **The robust finding:** per-user weights recover learner mastery that a **shared** LoRA and the
  **base** model do not (θ-MSE below shared in every config) — i.e. the `+per-user weights` arm of
  the §8.2 ablation pays off, and the lean-SFT ceiling is an *information/recipe* limit, not a
  student-capacity limit.
- **Key methodological result (worth carrying into the framework's eval):** **binary-outcome NLL is
  underpowered** — an oracle barely beats chance when mastery ≈ 0.5 under Bernoulli noise. Report
  **θ-recovery** (`mastery_corr`, `θ-MSE`) instead. This reversed an earlier "shared beats per-user"
  null that was purely a metric artifact.
- **Honest limits.** oracle-g/hybrid see the authored θ → *upper bounds*, not results. Misconception
  recovery is **near-noise** under the current bank (tags are near-unique per question; eval
  wrong-answers hit the held misconception only 3.9% ≈ uniform-random), so the honest headline is
  *mastery* recovery. Item-level 0.57 ≈ 0.77 (skill-tracking) × 0.74 (read-out decay).

Full write-ups: [`RESULTS.md`](RESULTS.md) (lean-SFT runs) · [`RESULTS-OPD.md`](RESULTS-OPD.md)
(the four-way OPD ablation) · [`notes/data-and-model-deep-dive.md`](notes/data-and-model-deep-dive.md)
(the complete data + training/eval deep-dive, honest limitations, and the TODO).

---

## 5. File map

| Path | What it is |
|---|---|
| `personas.py` | persona **θ** (ability + topic_offset + skill_jitter + held misconceptions) and the **controllable generator** `g(θ,q) → answer` |
| `streams.py` | `round_sequence` builder, stream simulation with learning dynamics, per-snapshot held-out eval, and the **discrimination / identifiability gate** |
| `build_question_bank.py` | canonicalise the generated MCQ bank (content-stable ids `{skill}#g{md5}`) |
| `build_substrate.py` | persona_set / splits / round_sequence / streams / eval_truth → `substrate/` |
| `sft/sft_core.py` | dual-rate LoRA **SFT** + **choice-PPL** scoring |
| `sft/run_all.py` | the hard-SFT run: A1 per-user · A0 shared · A∅ base → predictions |
| `sft/run_all_opd.py` | the **OPD** run — `CUE_TARGET=g \| hybrid \| realized_opt`; contains `teacher_dist` (= `g(θ,q)`), `_target_vec`, `_distill_step` |
| `sft/run_all_opd_llm.py` | the **LLM-teacher** OPD run (frozen base + privileged learner card; no oracle θ) |
| `sft/compare_opd.py`, `sft/analyze.py` | the 3-metric comparison (`binary_NLL` / `θ-MSE` / `mastery_corr`) |
| `sft/package_bundle.py`, `sft/verify_bundle.py` | isotonic calibration + the artifact bundle; the honesty-gate check |
| `sft/ceiling_*.py`, `sft/diagnose.py` | memorise-vs-transfer ceiling tests and recipe ablations |
| `question_bank/biology_gcse.jsonl` | **173** GCSE-Biology MCQs, 13 skills, misconception-tagged distractors |
| `substrate/` | the **held-constant evaluation contract**: `persona_set` · `splits` (104/39/30) · `round_sequence` · `streams` (2,496) · `eval_truth` (8,280) + its `README.md` |
| `results/bundles/` | per-run `manifest` / `curves` / `cost` / `headline` (the small summary JSONs) |
| `import_bundle.py`, `make_sample_bundle.py`, `sample_bundle/` | serving path: bundle → Mongo, for GPU-free replay in the downstream tutoring demos |
| `README.md`, `PACKAGE-README.md`, `SESSION-LOG.md`, `ONBOARDING-LINHAI.md` | the original repo docs |

---

## 6. Reuse for the new framework

- **The whole `personas.py` + `streams.py` + substrate methodology** — the cheapest way to get a
  *falsifiable* Stage-1 intrinsic eval in any new domain: author θ, generate behaviour, hold out truth.
  It directly implements the framework's "extrapolation tests the disposition weights" split.
- **`sft/run_all_opd.py`'s `_distill_step`** — a clean, small soft-label distillation step over a
  closed answer set. Reusable wherever the user's reaction is a choice among known options.
- **The 3-metric eval** (`compare_opd.py`) — and specifically the lesson to **not** headline binary NLL.
- **The A∅ / A0 / A1 control ladder** — drop-in for the §8.2 `base` vs `+per-user weights` ablation.
- **The bundle + replay path** (`import_bundle.py`) — how to serve a per-user parametric model to a
  live CPU-only agent without a GPU (record & replay), if the framework ever needs a live demo.

## 7. What was **NOT** copied

- `results/predictions/` (~11 MB) — the four runs' raw per-persona prediction dumps, needed only to
  *regenerate* the tables in `RESULTS-OPD.md` on CPU. Per this repo's convention (code + small files
  only), they stay in the canonical repo: `/users/k2480198/parametric_user_memory/results/predictions/`.
- Model weights / adapters and run logs — never in git; they live on scratch
  (`/scratch/prj/cllm/cue_sft/`).
- The `.git/` history of the source repo.

## 8. Provenance

Source: `/users/k2480198/parametric_user_memory` (local git, branch `main`, no remote).
Built for the EMNLP-2026 tutoring demos (`emnlp26_demo` "colearn" and `emnlp2026_demo2` "Cue"),
where the bundle is replayed GPU-free as the parametric user-model backend.
