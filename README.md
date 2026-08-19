# user_world_model

> An all-purpose, extensible **per-user world model** that simulates how a
> specific user reacts — serving personalized agents across **general**,
> **health**, and **education** domains.

The model keeps the core bet of the original per-user simulator (compress a
user into a small model that predicts their reactions) and adds three pieces:
a structured **user profile**, an append-only **user memory** (RAG-retrieved),
and **structured output** (reaction text **+** a user-state description). It is
initialized in stages (population → group → user) and updated either by
cold-start fast-init or by online streaming learning.

## Start here

- **[`project_summary.md`](project_summary.md)** — the framework in one read (vision, schema, three-store architecture, staged init, training objective, two-stage eval, status, next steps).
- **[`CONVENTIONS.md`](CONVENTIONS.md)** — how we name, store, code and coordinate. Every new session reads this first.
- **[`EXPERIMENTS.md`](EXPERIMENTS.md)** — what has actually been run, newest first.
- **[`docs/README.md`](docs/README.md)** — index of design / plans / findings / references.
- **[`legacy/README.md`](legacy/README.md)** — the three imported prior code bases and how to reuse them.

## Layout

```
project_summary.md          framework spec (read this first)
CONVENTIONS.md              operational conventions (naming / storage / code / concurrency)
EXPERIMENTS.md              experiment log · KNOWLEDGE.md  reading notes & direction decisions

domains/                    ONE SUBPACKAGE PER DOMAIN — no cross-domain imports
  general/                    PersonaMem: data.py · peruser_data.py · scorer.py · baselines/
  health/                     PMData digital-twin: data.py · peruser_data.py
  education/                  KCL tutor chats: data.py
common/                     shared infra only: backends.py · runmeta.py · sft.py
scripts/                    env.sh · claim_run.sh · gen_runs_md.py
  general/ health/ education/   per-domain entry points (+ .slurm launchers)
experiments/                configs/ runs/ reports/ · results/{general,health,education}/
data/education/             private KCL course tutor-chat data (small, in-repo)
docs/                       design/ plans/ findings/ status/ refs/ external/
legacy/                     imported prior code bases (general / health / education)
```

Large artifacts — models, datasets, checkpoints, logs — live on cluster scratch
(`$UWM_SCRATCH`), never in git. `source scripts/env.sh` sets it up.

## Status — early exploratory stage

Architecture converged (2026-06-03). A Stage-1 intrinsic-prediction ablation has been run once
in each of the three domains (see `EXPERIMENTS.md`, 2026-06-17 / 06-20), each on a single
dataset with a single recipe. **Those numbers are provisional observations, not settled results**:
no arm here is established as working or as ruled out, and the immediate priority is exactly the
experiments that would tell the two apart. Stage-2 (agent collaboration) has not started.

The prior general-domain prototype (per-user on-policy distillation on PersonaMem-v1) lives in
`legacy/general_personamem/`; the education per-user-weights instance in
`legacy/education_parametric_memory/` — see their own `README.md` / `ENTRY.md` for those results
and their limits.

## Setup

```bash
conda create -n user_world_model python=3.12 && conda activate user_world_model
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Training artifacts and large data live on cluster scratch (`$SCRATCHDIR`), not
in git. See [`CLAUDE.md`](CLAUDE.md) for platform/storage conventions.

## License

MIT — see [LICENSE](LICENSE). · `lzhang472@gmail.com` · KCLNLP · 2026
