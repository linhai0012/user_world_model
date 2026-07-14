# Project Context (method-agnostic — reusable across projects)

> This file is the **reusable operating protocol** for working in this repo,
> meant to drop into sibling projects with little change. It is kept
> **method/project-agnostic on purpose** — the specific research method, design,
> and layout live in the repo's own docs:
>
> - **What this repo is / the design** → `project_summary.md`, `README.md`
> - **Naming / storage / code / multi-agent concurrency (read first)** → `CONVENTIONS.md`

## Target platforms

- **KCL CREATE**: x86_64; A100 / H100 / H200 / B200; Slurm
- **Bristol BriCS (Isambard-AI Phase 2)**: ARM aarch64, NVIDIA GH200; Slurm, max 24h walltime

## Storage rules (HARD)

- **Code + small experiment files in the git checkout** (configs, status,
  headline JSON, reports, small committed reference data)
- **Data + models + checkpoints + large logs on cluster scratch** — `source
  scripts/env.sh`, which sets the scratch root, `HF_HOME`, and the conda env
  (exact paths / quota are project-specific; see `CONVENTIONS.md` §2)
- New scripts MUST default `--output-dir` to scratch, not the repo root

## Server environment

- Conda env + Python pinned in `scripts/env.sh` (`source` it to activate)
- PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cu128`
- Multi-node: `module load brics/nccl` on Isambard

## HuggingFace account

`lzhang472` — used for both model uploads and dataset access.

## Coding standards

- Distributed training: PyTorch FSDP or DDP
- Always include `--time` and `--gpus` in Slurm scripts
- Slurm launchers live in `scripts/`; keep `.sh` / `.slurm` variants in sync

## Multi-agent concurrency (HARD) — full rules in CONVENTIONS.md §4

Multiple Claude Code processes may run on different compute nodes but share ONE
working tree on shared cluster storage (edits are instantly visible; git is NOT
the inter-agent sync layer).
- One **driver** session owns shared code and is the only one that runs git.
- **Runners** write only their own work paths (`baselines/<method>/`,
  `experiments/.../<run_id>.*`, the scratch run dir); they never run git.
- **Never** run tree-wide destructive git (`reset --hard`, `checkout -- .`,
  `clean -fd`, `stash`, `rebase`) — it wipes other sessions' uncommitted work.
- Claim a work area with `scripts/claim_run.sh <area>` before editing it.

## Per-project files (maintained at repo root)

- `EXPERIMENTS.md` — milestones, config summaries, key metrics (update after significant runs)
- `KNOWLEDGE.md` — paper-reading notes + research-direction decisions
- `CONVENTIONS.md` — this repo's concrete naming / storage / layout (project-specific)

## Collaboration rules

- Do NOT modify code unless the user explicitly asks for changes
- Analysis, diagnosis, and suggestions are always welcome — but only
  write/edit code when instructed
