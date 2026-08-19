# docs/ — index

| dir | what is in it |
|---|---|
| `design/`   | the framework design as a visual discussion deck (`uwm_framework_discussion.html`). The written spec is [`../project_summary.md`](../project_summary.md). |
| `plans/`    | build plans handed between sessions (scope, TODO, operating context at the time of writing) |
| `findings/` | cross-run syntheses — what a set of experiments looked like *at that date*, with its controls and caveats |
| `status/`   | dated project-status snapshots (historical; paths inside them may predate a reorg) |
| `refs/`     | external reference papers |
| `external/` | material **borrowed from other projects** — background only, not this repo's work programme |

## Reading order for a new session

1. [`../CONVENTIONS.md`](../CONVENTIONS.md) — naming / storage / code / concurrency (read first)
2. [`../project_summary.md`](../project_summary.md) — the design
3. [`../EXPERIMENTS.md`](../EXPERIMENTS.md) — what has actually been run, newest first
4. `findings/` — the cross-domain reads of those runs
5. `../legacy/README.md` — the three imported prior code bases and what is reusable

## A note on `external/`

`coevolution_design_notes_*.md`, `handoff_emnlp26demo_*.md` and `HiME_EMNLP_demo.pdf` come from
the EMNLP-26 demo / co-evolution线 of work. They are kept here because they share vocabulary with
this project, **not** because they define its direction — this repo's axis is the cross-domain
user-world-model method. Do not let demo requirements steer the general/health/education work.

## Status of everything in here

The project is at an **early, exploratory stage**. Documents in `findings/` and `status/` record
what a particular setup produced on a particular date; they are provisional readings, not settled
conclusions, and later runs may move them in either direction.
