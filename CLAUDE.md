# Project Context — user_world_model

## What this repo is

**All-purpose user world model**: one extensible per-user model that
simulates how a specific user reacts, serving personalized agents across
**general** (PersonaMem), **health** (PMData / digital-twin), and
**education** (course tutor chats) domains. Core = per-user LLM +
structured profile + append-only memory + structured output (reaction
text + user-state). See `project_summary.md` and
`docs/uwm_framework_discussion.html` for the design.

The prior general-domain prototype (per-user on-policy distillation on
PersonaMem-v1) is archived under `legacy/general_personamem/`; reusable
health code under `legacy/health_digitaltwin/` — see `legacy/README.md`.

Companion to https://github.com/linhai0012/P-OPSD (agent-modeling
track on PersonaMem-v2), developed independently.

## Target platforms

- **Bristol BriCS (Isambard-AI Phase 2)**: ARM aarch64, NVIDIA GH200,
  Slurm scheduler, max 24h walltime
- **KCL CREATE**: A100 / H100 / H200 / B200; NVIDIA `vmm` driver

Both clusters share the same conda + HF cache layout below.

## Storage rules (HARD)

- **Code in `$HOME`** (the git checkout)
- **Data + models + checkpoints in `$SCRATCHDIR`**
- New scripts MUST default `--output-dir` to scratch, not repo root
- HF cache: `export HF_HOME=$SCRATCHDIR/hf_cache` in every Slurm script

A root `outputs/` directory is reserved for **committed reference eval
results** (small JSONs, review xlsx) of the *new* framework. It is NOT for
training artifacts. New eval scripts should default their output to
`$SCRATCHDIR/user_world_model_outputs/...` and only the headline JSON
should ever land in `outputs/`. (The prior prototype's eval results are
archived under `legacy/general_personamem/outputs/`.)

## Server environment

- Python: 3.12 via Miniforge (conda-forge)
- PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cu128`
- Multi-node: `module load brics/nccl` on Isambard

## HuggingFace account

`lzhang472` — used for both model uploads and dataset access.

## Coding standards

- Distributed training: PyTorch FSDP or DDP
- Always include `--time` and `--gpus` in Slurm scripts
- Slurm launchers: keep `.sh` and `.slurm` variants in sync (Slurm
  version = SBATCH-wrapped version of the interactive `.sh`). The prior
  prototype's launchers are in
  `legacy/general_personamem/student_opd/run_*.{sh,slurm}`

## Collaboration rules

- Do NOT modify code unless the user explicitly asks for changes
- Analysis, diagnosis, and suggestions are always welcome — but only
  write/edit code when instructed
