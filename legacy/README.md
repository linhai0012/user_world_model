# Legacy resources — entry point

This folder archives the two prior code bases the all-purpose User World Model
builds on. **Nothing here is the new framework** — it is reference material to
reuse. See [`../project_summary.md`](../project_summary.md) for the new design.

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
