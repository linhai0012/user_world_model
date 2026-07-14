# Legacy resources — entry point

This folder holds the three sibling code bases the all-purpose User World Model
builds on. **Nothing here is the new framework** — it is reference material to
reuse. See [`../project_summary.md`](../project_summary.md) for the new design.

> Two are archived prototypes (`general_personamem`, `health_digitaltwin`); the third,
> `education_parametric_memory`, is a snapshot of an **active** sibling repo — it is the
> *education* domain's starting point and the first working instance of the framework's
> **per-user weights** store.

---

## `general_personamem/` — prior general-domain prototype

The original `user_world_model` repo: a **per-user simulator via on-policy
distillation** on PersonaMem-v1. This is the *general domain* of the new
framework, and the source of most design lessons (why memory is needed, the
OPSD≈SFT analysis, the failed gating experiments).

| Path | What it is |
|---|---|
| `data_prep/` | PersonaMem-v1 loading, episode segmentation, K-session windows, SFT tokenization |
| `teacher_sft/` | Phase-1 teacher SFT (Qwen3-4B-Instruct-2507, K=3 context, user-token loss) → R3 — **≈ the new framework's population stage** |
| `student_opd/` | Phase-2/2b per-user dual-LoRA student via OPD/OPSD; all eval scripts (MCQ-PPL, NLL, judge, verbal) and Slurm launchers (`run_*.{sh,slurm}`) |
| `outputs/` | committed reference eval results (JSON / xlsx); adapter configs |
| `EXPERIMENTS.md` | full experiment log (Phase 0–2b; R1 / R3 / Phase 2 / gating / OPSD) — **read for the empirical basis of the new design** |
| `phase2b_experiment_plan.md`, `verbal_eval_summary.md`, `mcq_examples.*` | design notes & inspection samples |
| `README.md` | the original repo README (headline results, quick start, HF model URLs) |

**Reuse for the new framework:** `build_opd_data.py` (add a `retrieved_memory`
field), `train_opd_dual.py` / `train_opsd_dual.py` (objective becomes CE+optional-KL),
`load_personamem.py`, and the whole eval suite (becomes Stage-1 intrinsic eval).

---

## `health_digitaltwin/` — reusable code from LLM-based-Digital-Twins

A **subset** copied from the sibling project
`github.com/linhai0012/LLM-based-Digital-Twins` (full repo has more — HR/MoE
stages, large eval artifacts — not copied here; go to the original for those).
This is the *health domain* starting point.

| File | Role · reuse |
|---|---|
| `config.py` | **wellness fields, state tokens, `encode_wellness_state`** — 100% reusable for the structured-state output |
| `step1_parse_pmdata.py` | parse PMData (Fitbit + PMSys wellness CSV) → records — reusable for profile feature extraction |
| `step2_synthesize_text.py` | GPT synthesizes event desc + user feedback grounded in real signals — reusable for reaction-text targets |
| `step3_build_training_data.py` / `step3_build_v2_data.py` / `step3_inspect_v2_data.py` | format training data — adapt: **drop the HR-token branch**, keep state tokens |
| `dataset.py`, `model_setup.py`, `trainer.py` | Dataset/collator, vocab extension, α-weighted text/state loss — ~70% reusable |
| `metrics.py` | `compute_state_metrics`, `compute_consistency` — **Stage-1 structured-state metrics** |
| `generate_and_eval.py`, `train.py`, `run_pipeline.py`, `train_config.py`, `accelerate_config.yaml` | training/eval entry points & config — reference |
| `phllm_predictability_test.py`, `phllm_predictability_analyze.py` | multi-day sliding-window approach — reference for the online/temporal aspect |
| `README.md`, `EXPERIMENTS.md`, `MoE_Implementation_Spec.md` | original docs & results |
| `reference/special_tokens.json`, `reference/dataset_stats.json` | small reference artifacts |

**What was NOT copied** (still in the original repo): HR-specific `stage0_*` /
`stage1_*` (heart-rate generation + MoE), bulk `output/*.json|*.png`, and the
`train/val/test.jsonl` data. The new framework targets **text reaction + state**,
not raw HR generation, so those are out of scope (per the design in
`project_summary.md`).

---

## `education_parametric_memory/` — education domain · the *per-user weights* store

**→ Read [`education_parametric_memory/ENTRY.md`](education_parametric_memory/ENTRY.md) first.**

A snapshot (2026-07-14) of the **active** sibling repo
`/users/k2480198/parametric_user_memory`. A **per-user LoRA** that compresses a learner's graded
answer history into weights and predicts **how that learner answers new MCQs** — the *edu-exam*
half of the education domain, and the first working instance of the framework's third store
(**per-user weights**, `project_summary.md` §3). Directly comparable to `general_personamem/`:
both train the per-user student by **distillation (OPD)**, one on preferences, one on assessment.

| Path | What it is |
|---|---|
| `personas.py` / `streams.py` | persona **θ** + the **controllable generator** `g(θ,q)→answer`; round_sequence, stream simulation, **identifiability gate** |
| `build_question_bank.py` / `build_substrate.py` | the deterministic data-construction pipeline |
| `question_bank/` · `substrate/` | 173 misconception-tagged GCSE-Bio MCQs · the held-constant eval contract (splits 104/39/30, streams 2496, eval_truth 8280) |
| `sft/sft_core.py` · `sft/run_all.py` | dual-rate LoRA **SFT** + **choice-PPL** scoring; the A1/A0/A∅ control ladder |
| `sft/run_all_opd.py` · `sft/run_all_opd_llm.py` | the **OPD** runs (soft-label distillation): oracle-`g` / `hybrid` / **LLM-teacher** |
| `sft/compare_opd.py` · `RESULTS.md` · `RESULTS-OPD.md` | the 3-metric comparison and the full results |
| `notes/data-and-model-deep-dive.md` | the complete data + training/eval deep-dive, honest limits, TODO |
| `import_bundle.py` · `sample_bundle/` | serve a per-user parametric model to a **CPU-only** live agent (record & replay) |

**Headline (held-out, @snap104):** per-user weights recover learner mastery that a *shared* LoRA and
the *base* model do not. `mastery_corr` — hard SFT **0.159** → deployable LLM-teacher **0.216** →
privileged oracle **0.494–0.570**. **Methodological lesson worth adopting framework-wide: binary-outcome
NLL is underpowered — report θ-recovery (`mastery_corr` / `θ-MSE`) instead.** Honest limits (oracle =
upper bound; misconception recovery near-noise) are in `ENTRY.md` §4.

**Reuse for the new framework:** the whole author-θ → generate-behaviour → hold-out-truth
**methodology** (a falsifiable Stage-1 intrinsic eval in any domain); the closed-set soft-label
`_distill_step`; the A∅/A0/A1 ladder for the §8.2 `+per-user weights` ablation.

**What was NOT copied:** `results/predictions/` (~11 MB raw prediction dumps — regenerate the tables
from them in the canonical repo), model weights/adapters and run logs (on scratch), and the source
repo's `.git/`.
