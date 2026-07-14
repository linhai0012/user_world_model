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
- **[`docs/uwm_framework_discussion.html`](docs/uwm_framework_discussion.html)** — the same design as a visual discussion deck (open in a browser).
- **[`legacy/README.md`](legacy/README.md)** — what the archived prior work is and how to reuse it.

## Layout

```
project_summary.md          framework spec (read this first)
CONVENTIONS.md              operational conventions (naming / storage / code / concurrency)
docs/                       discussion deck (html) + reference papers
data/education/             private KCL course tutor-chat data
common/ baselines/ experiments/ scripts/   active general-domain build (baselines + run registry)
legacy/general_personamem/  archived general-domain OPD/OPSD prototype
legacy/health_digitaltwin/  reusable code imported from LLM-based-Digital-Twins
```

## Status

Architecture converged (2026-06-03); repo reorganized; new-framework
implementation not yet started. The prior general-domain prototype (per-user
on-policy distillation on PersonaMem-v1, **+95.7%** best-step gap closure across
20 personas) lives in `legacy/general_personamem/` — see its `README.md` and
`EXPERIMENTS.md` for full results.

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
