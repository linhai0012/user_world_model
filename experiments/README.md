# experiments/ — run registry (see CONVENTIONS.md §1)

| dir | contents |
|-----|----------|
| `configs/` | `<run_id>.yaml` — fully-specified, reproducible run config |
| `runs/`    | `<run_id>.yaml` — live status board (one file per run, merge-safe) |
| `results/<domain>/` | headline results, committed (small JSON) — `general/` `health/` `education/` |
| `reports/` | cross-run analysis (md / xlsx) |

Large artifacts (logs, predictions, memory stores, checkpoints) live in
`$UWM_SCRATCH/runs/<run_id>/`, **never here**. Copy `_TEMPLATE.yaml` in
`configs/` and `runs/` to start a new run. Regenerate the top-level index
with `python scripts/gen_runs_md.py`.

## Honest state of the registry (2026-08-09)

The `configs/` + `runs/` + `RUNS.md` machinery is **scaffolded but not in use**: every run to
date (general sweeps, health Stage-1, education Stage-1) was launched directly from
`scripts/<domain>/*.py` and never registered, so `runs/` holds only the template and `RUNS.md`
renders an empty table. `results/<domain>/*.json` is therefore the actual record of what ran —
each file carries its own `run_id`/params, and `EXPERIMENTS.md` mirrors the headline numbers.

Two ways forward, whichever the project prefers — pick one rather than leaving the mismatch:
either backfill a `runs/<run_id>.yaml` per existing result and register new runs going forward,
or drop `configs/`+`runs/`+`gen_runs_md.py` and make `results/<domain>/` + `EXPERIMENTS.md` the
declared source of truth (and simplify CONVENTIONS.md §1/§5 to match).

Naming, as actually practised: general runs follow the CONVENTIONS §1 grammar
(`bl__<method>__qwen3-4b__<bench>[__variant]__<metric>.json`); health and education runs predate
it and use `<domain-ish>_<arm>__<metric>.json` (e.g. `health_shared_current__mae.json`,
`edu_stage1__nll.json`). Unify when the registry question above is settled.
