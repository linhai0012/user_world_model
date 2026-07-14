# experiments/ — run registry (see CONVENTIONS.md §1)

| dir | contents |
|-----|----------|
| `configs/` | `<run_id>.yaml` — fully-specified, reproducible run config |
| `runs/`    | `<run_id>.yaml` — live status board (one file per run, merge-safe) |
| `results/` | `<run_id>__<metric>.json` — committed headline results (small) |
| `reports/` | cross-run analysis (md / xlsx) |

Large artifacts (logs, predictions, memory stores, checkpoints) live in
`$UWM_SCRATCH/runs/<run_id>/`, **never here**. Copy `_TEMPLATE.yaml` in
`configs/` and `runs/` to start a new run. Regenerate the top-level index
with `python scripts/gen_runs_md.py`.
