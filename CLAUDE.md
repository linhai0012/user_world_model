# Project Context — user_world_model

## What this repo is

Per-user simulator via on-policy distillation. Trained on
PersonaMem-v1 (20 personas × 32k / 128k / 1M context versions).

Companion to https://github.com/linhai0012/P-OPSD (agent-modeling
track on PersonaMem-v2). The two tracks share design heritage but
are now developed independently.

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

The `outputs/` directory in this repo is reserved for **committed
reference eval results** (small JSONs, review xlsx). It is NOT for
training artifacts. New eval scripts should default their output to
`$SCRATCHDIR/user_world_model_outputs/...` and only the headline
JSON should ever land in `outputs/`.

## Server environment

- Python: 3.12 via Miniforge (conda-forge)
- PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cu128`
- Multi-node: `module load brics/nccl` on Isambard

## HuggingFace account

`lzhang472` — used for both model uploads and dataset access.

## Coding standards

- Distributed training: PyTorch FSDP or DDP
- Always include `--time` and `--gpus` in Slurm scripts
- Slurm launchers live in `student_opd/run_*.{sh,slurm}` — keep both
  variants in sync (Slurm version = SBATCH-wrapped version of the
  interactive `.sh`)

## Collaboration rules

- Do NOT modify code unless the user explicitly asks for changes
- Analysis, diagnosis, and suggestions are always welcome — but only
  write/edit code when instructed
