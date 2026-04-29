# user_world_model

> Per-user simulator via on-policy distillation — train per-user LoRAs that score
> agent candidates by predicting how a specific user would react, without any
> chat-history context at inference time.

This repository contains the **user-modeling track** of the
P-OPSD project. The companion *agent-modeling track* lives in
[P-OPSD](https://github.com/linhai0012/P-OPSD) (private/sister
repo) — together they explore two complementary entry points to
personalized assistants: train an agent that responds aligned to
the user, vs. train a user simulator that an agent queries.

---

## What this is

We train a **per-user LoRA** as a parametric user simulator. At
deployment, an agent proposes N candidate responses to a query;
the simulator scores each by perplexity of the next user turn under
the per-user adapter; the agent picks `argmin PPL`. The user
simulator never sees chat history at inference — all per-user
information is compressed into the LoRA weights.

Trained on **PersonaMem-v1** (20 personas, 32k / 128k / 1M context
versions; cross-version test isolates persona knowledge from event
memorization).

### Headline result

On the full 20-persona benchmark, R1b dual-LoRA student reaches:

| Setup                          | Acc on 781 / 128k MCQ |
|--------------------------------|---------------------:|
| Random (4-way MCQ)             | 25.0%                |
| Base Qwen3-4B-Instruct (no ctx)| 30.6%                |
| **R1b student** (demo only)    | **38.8%**            |
| Teacher_k3 (+ K=3 history)     | 39.8%                |

**+95.7% best-step gap closure across 20 personas** — student
matches teacher with zero conversation context. On the 1M
cross-version OOD test (4 focal personas), student exceeds
teacher: **+128% closure**, with 3/4 personas individually beating
the K=3-history teacher.

See [`EXPERIMENTS.md`](EXPERIMENTS.md) for full ablation table,
per-qtype breakdown, and comparison vs reverse-KL / gated-KL
variants.

---

## Repository layout

```
data_prep/         PersonaMem-v1 loading, episode segmentation,
                   K-session window construction, SFT tokenization
teacher_sft/       Teacher SFT (Instruct-2507 base, K=3 progressive
                   context, user-token loss only) — produces R3 final
student_opd/       Per-user dual-LoRA student trained via on-policy
                   distillation; eval scripts (MCQ-PPL, NLL, judge)
outputs/           Eval result JSONs and review xlsx (committed
                   reference results; NOT training artifacts —
                   those live on cluster scratch)
EXPERIMENTS.md     Full experiment log (Phase 0–2b, R1 / R3 / Phase 2)
phase2b_experiment_plan.md   Round 1b → 2c design notes
verbal_eval_summary.md       Three failed verbal-feedback attempts
mcq_examples.md / .xlsx      Per-qtype MCQ inspection samples
```

---

## Setup

Tested on Python 3.12, NVIDIA H100/H200/B200, CUDA 12.x.

```bash
# 1. Conda + base deps
conda create -n user_world_model python=3.12
conda activate user_world_model
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# 2. PersonaMem-v1 dataset
huggingface-cli download bowen-upenn/PersonaMem --repo-type dataset

# 3. Verify install
python -c "import torch, transformers, peft, vllm; print('OK')"
```

Outputs and checkpoints are gitignored — keep them on cluster
scratch (`$SCRATCHDIR` on Isambard-AI Phase 2 / KCL CREATE).

---

## Quick start

Reproduce the R1b headline result (per-persona):

```bash
# Build OPD training data (K=3 teacher window, demo-only student input)
python data_prep/build_opd_data.py --version 128k --pid 4

# Train R1b dual LoRA on persona 4 (Lisa)
bash student_opd/run_opsd_train_interactive.sh 4

# Evaluate MCQ-PPL on 128k holdout
bash student_opd/run_round1b_mcq_eval.sh 4
```

Full per-persona + cross-version sweep is in
`student_opd/run_round1b_extend.{sh,slurm}`.

---

## Models on HuggingFace

R1b LoRAs uploaded to HF for reproduction:
- Per-persona LoRAs: `lzhang472/user_world_model-r1b-pid{0..19}`
- Teacher SFT R3 final: `lzhang472/user_world_model-teacher-r3`

(Update with concrete URLs after upload.)

---

## Citation

If this work is useful, please cite:

```bibtex
@misc{zhang2026userworldmodel,
  title  = {Per-user simulator via on-policy distillation for
            zero-context personalized agents},
  author = {Zhang, Lin},
  year   = {2026},
  note   = {Working paper. KCLNLP},
}
```

---

## License

MIT — see [LICENSE](LICENSE).

---

## Contact

`lzhang472@gmail.com` · KCLNLP · 2026

Issues / discussions welcome — work in progress.
