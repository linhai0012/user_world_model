# Parametric User Memory (Mode A) — offline per-user LoRA for MCQ tutoring

A self-contained offline pipeline that **compresses a learner's graded MCQ history into a per-user
LoRA adapter** (the agent's belief `A_t` about the learner), trained and evaluated entirely offline,
GPU-free to inspect. It studies whether a *parametric* user model recovers a learner's latent
mastery from their answer stream, and compares training recipes (lean SFT vs **privileged
distillation / OPD**).

> **Scope / origin.** This is the **MCQ-centric parametric user-memory** component (knowledge-tracing
> flavour). It was developed during the parametric-vs-agentic memory exploration; it lands here in the
> MCQ-tutoring demo because its unit of interaction is the **multiple-choice item**. Some of the
> historical notes (`SESSION-LOG.md`, `notes/data-and-model-deep-dive.md`) still reference the sibling
> proactive-agent demo where the exploration ran — the code and data here are self-contained and
> MCQ-only.

## Headline result (held-out eval, pooled mastery_corr ↑ / theta_MSE ↓)

| recipe | what it trains on | mastery_corr @snap104 | theta_MSE @snap104 |
| --- | --- | ---: | ---: |
| hard (lean SFT) | the learner's chosen-option **text** | 0.159 | 0.130 |
| LLM-teacher (semi-oracle*) | `0.5·llm-card + 0.5·realized` | 0.216 | 0.108 |
| oracle-hybrid | `0.5·g(θ) + 0.5·realized` | 0.494 | 0.051 |
| **oracle-g** | the generator's option distribution `g(θ)` | **0.570** | 0.054 |

- **Privileged distillation lifts the lean-SFT transfer ceiling ~3×** (mastery_corr 0.16 → 0.49–0.57);
  the gain comes from `g`'s soft `P(correct)=mastery` target carrying far more information than one
  Bernoulli outcome. This is an **oracle upper bound** (distills the authored θ).
- *LLM-teacher is **semi-oracle** (still uses the exact misconception card + a true-mastery bucket); a
  fully deployable teacher number is lower and TBD. See `RESULTS-OPD.md` and the deep-dive.
- **Mastery/ability recovery is the robust signal; misconception recovery is near-noise under the
  current data** (each misconception is a unique-per-question tag → not transferable; see the deep-dive
  §3). The honest headline is mastery recovery.

## Layout

```
personas.py / streams.py        persona θ generative model + answer-stream / snapshot simulation
build_question_bank.py          canonicalise the Workflow-generated MCQ bank
build_substrate.py              persona_set / round_sequence / streams / eval_truth  -> substrate/
sft/                            training + scoring harnesses:
  sft_core.py                     dual-rate LoRA + choice-PPL scoring
  run_all.py / run_all_opd.py     full runs (hard SFT / OPD: CUE_TARGET=g|hybrid|realized_opt)
  run_all_opd_llm.py              deployable LLM-teacher OPD
  ceiling_*.py                    transfer / recipe ablations
  compare_opd.py / analyze.py     3-metric (binary_NLL / theta_MSE / mastery_corr) comparison
  package_bundle.py / verify_bundle.py   calibrated artifact bundle + honesty-gate check
question_bank/biology_gcse.jsonl  173 GCSE-biology MCQs (13 skills, misconception-tagged distractors)
substrate/                       the shared offline contract (persona_set, round_sequence, streams, eval_truth, splits)
sample_bundle/  import_bundle.py  example artifact bundle + JSONL->Mongo importer
results/                         the four runs' predictions + bundle summaries (CPU-reproducible numbers)
RESULTS.md  RESULTS-OPD.md        SFT results / the OPD four-way ablation
notes/data-and-model-deep-dive.md  full data + training/eval deep-dive, TODO, and future work
```

## Reproduce the numbers (CPU, no GPU)

```bash
# the 3-metric four-way comparison in RESULTS-OPD.md is regenerated from results/predictions/:
CUE_PRED_A=results/predictions/hard_v4 CUE_PRED_B=results/predictions/oracle_g \
CUE_SUBSTRATE=$PWD/substrate CUE_SNAPSHOTS=0,13,26,52,104 python sft/compare_opd.py
```
(GPU is only needed to *regenerate* predictions via `run_all*.py` on Qwen3-4B; the analysis is CPU-only.)

See `notes/data-and-model-deep-dive.md` for the complete walk-through, honest limitations, the TODO
(incl. the **knowledge-structure redesign** that would make misconception recovery learnable), and
the out-of-scope future work.
