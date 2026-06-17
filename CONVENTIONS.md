# CONVENTIONS — user_world_model

> **Operational single source of truth** for how we name, store, code, and
> coordinate. (Project *design/vision* — the three-store all-purpose framework —
> lives in [`project_summary.md`](project_summary.md).)
> **Every new session / every Claude Code process reads this first.**
>
> Why this file exists: multiple Claude Code processes run on different
> compute nodes but all `cd` into **this same repo on shared CephFS** —
> edits are instantly visible to everyone, with no per-node isolation. So
> coordination discipline lives here, not in git. Private agent memory is
> NOT shared across sessions; this file is.

Design spec: [`project_summary.md`](project_summary.md) (all-purpose framework) ·
prior general-domain prototype: `legacy/general_personamem/` ·
general-track snapshot: `PROJECT_STATUS_2026-05-29.md`

---

## 0. Environment (KCL CREATE)

- Single-GPU jobs on **H200 143GB**; partition e.g. `tier1_charity_gpu`.
- Conda env **`vllm_env`** (py3.10, torch2.9+cu128, vllm0.13, transformers4.57,
  peft0.18, datasets4.8, openai2.24). Activate via `source scripts/env.sh`.
- `/users/k2480198`, `/scratch/...`, `/cephfs/...` are ONE shared CephFS,
  visible identically from every node (`/users/k2480198` == the cephfs home,
  same inode).
- Secrets read from the environment (`OPENAI_API_KEY`, `HF_TOKEN`); you
  `export` them per session — code never stores or reads them from disk.

---

## 1. Naming

Run-id grammar (fields joined by `__`, words inside a field by `-`):

```
{family}__{method}__{model}__{bench}[__{variant}]
```

| field   | values |
|---------|--------|
| family  | `bl` (baseline) · `ours` · `probe` |
| method  | `oracle` · `trivial` · `fluxmem` · `mem0` · `zep` · `amem` · `naiverag` |
| model   | `qwen3-4b` · `gpt41` |
| bench   | `pm32k` · `pm128k` · `pm1m` · `locomo` |
| variant | optional: `v1`, `topk5`, `seed0`, … |

Examples: `bl__oracle__qwen3-4b__pm128k` · `bl__fluxmem__gpt41__pm128k__v1`

The run-id keys every artifact:

| artifact | path |
|----------|------|
| config (reproducible spec) | `experiments/configs/<run_id>.yaml` |
| status (task board, 1/run) | `experiments/runs/<run_id>.yaml` |
| headline result (small)    | `experiments/results/<run_id>__<metric>.json` |
| large artifacts            | `$UWM_SCRATCH/runs/<run_id>/` |
| index (generated)          | `RUNS.md` ← `scripts/gen_runs_md.py` |

`<metric>` ∈ `acc`, `acc_by_qtype`, `ppl`. **One YAML file per run** so two
agents never edit the same file (merge-safe).

---

## 2. Storage

`/scratch/prj/cllm` **and** `/scratch/prj/inf_embrace_llm` are **FULL** (2026-06-17) — active
space moved to `inf_elandi`. (Other writable fallbacks if needed: `/scratch/prj/eventnlu`,
`/scratch/users/k2480198`.)

```
UWM_SCRATCH=/scratch/prj/inf_elandi/k2480198/uwm  # active (inf_embrace_llm filled up)
HF_HOME=$UWM_SCRATCH/hf_cache

$UWM_SCRATCH/
  data/    PersonaMem + LoCoMo (raw + processed)
  models/  base weights, merged checkpoints
  hf_cache/
  shared/  build-once, reuse-across-runs (embeddings, indices)
  runs/<run_id>/   per-run logs, preds, memstore, ckpt
```

**Rule:** the repo holds only code + `experiments/` small files (config /
status / headline JSON) + reports + small committed reference data (e.g.
`data/education/`). Models, large data, checkpoints, large logs, memory
stores → `$UWM_SCRATCH`, never git.

Quotas: home 50GB · user-scratch 200GB · `cllm` (full) · `inf_embrace_llm`
932GB (**full as of 2026-06-17** — couldn't write 1GB) · `inf_elandi` (active,
hundreds of GB free). **Save LoRA adapters (~130MB), not merged 8GB models, and
delete transient merged models after eval — quota is shared across the project.**

---

## 3. Code & evaluation

This doc governs the **active general-domain work** (`project_summary.md` §8
builds *general* first). The `baselines/` here — oracle / trivial / token-memory —
are the general-domain instantiation of `project_summary.md` §8.2's ablation
(`base` vs `+profile` vs `+memory` vs `+per-user weights`); the eval contract
below is its `project_summary.md` §6 Stage-1 intrinsic-prediction layer.

```
ACTIVE general-domain work (governed by this doc):
  common/      shared lib: data loaders (PersonaMem/LoCoMo), scorers
               (accuracy + ppl), backends (vllm_qwen / openai_gpt), run-meta I/O
  baselines/   oracle/ trivial/ tokenmem/{fluxmem,mem0,zep,amem,naiverag}/
  experiments/ configs/ runs/ results/ reports/
  scripts/     env.sh · claim_run.sh · gen_runs_md.py · run_baseline.slurm

Framework & legacy (see project_summary.md):
  project_summary.md / docs/   all-purpose design (profile + memory + per-user weights)
  data/education/              private KCL course-chat data
  legacy/general_personamem/   frozen Phase 0–2b + OPSD prototype (the prior repo:
                               data_prep/ teacher_sft/ student_opd/ outputs/ EXPERIMENTS.md)
  legacy/health_digitaltwin/   reusable health-domain code (LLM-based-Digital-Twins)
```

**Evaluation contract (makes baselines comparable):**

- All methods implement `predict(mcq, context) -> {pred_choice, per_choice_score?}`.
- **Primary metric = accuracy** on PersonaMem single-choice; always also
  report **per-qtype** over the 7 categories. PPL is a secondary,
  model-internal-only metric.
- One loader in `common/` per `bench`; same split, same decoding (temp 0).
- Backends: `qwen3-4b` via local vLLM; `gpt41` via OpenAI API.

**Coding rules:** argparse `--run-id` + read `experiments/configs/<run_id>.yaml`;
no hard-coded paths (resolve from `scripts/env.sh` vars); deterministic seed;
write `run_meta.json` into the run dir; any `--output-dir` defaults to
`$UWM_SCRATCH`, never the repo.

---

## 4. Concurrency — multiple CC processes, one shared tree

All CC processes share this working tree; edits are instant. **git is the
driver's versioning/backup tool, NOT the inter-agent sync mechanism.**
Coordination state lives in `.uwm/` (in-repo, gitignored, instantly visible):

```
.uwm/locks/<area>/owner.txt   # mkdir = atomic claim; owner.txt = node/session/pid/time
.uwm/heartbeat/<session>      # optional liveness
```

**Roles**

- **driver** (the main session): owns all shared code — `common/`, `scripts/`,
  the four root docs (`CONVENTIONS` / `EXPERIMENTS` / `CLAUDE` / `README`).
  **The only process that runs git.**
- **runner** (auxiliary sessions): writes only `baselines/<its method>/`,
  `experiments/{configs,runs,results}/<its run_id>.*`, and
  `$UWM_SCRATCH/runs/<run_id>/`. **Never runs git.**

**Rules**

1. Write only within your owned paths. Need a `common/` change → ask the driver.
2. Claim before editing: `scripts/claim_run.sh <area>` (atomic mkdir lock);
   release with `--release <area>`. Can't claim → someone's there; pick another.
3. **Forbidden for everyone — tree-wide destructive git** (wipes others'
   uncommitted work): `git reset --hard`, `git checkout -- .`, `git clean -fd`,
   `git stash`, `git rebase`.
4. Only the driver commits, with **path-scoped** `git add <paths>` (never
   `git add -A` / `git add .`), and only files whose run is `status: done`.
5. Prefer creating new files over editing shared ones (1 run = 1 yaml,
   1 method = 1 dir).
6. Destructive ops (rm/mv/bulk-edit): list first, stay within owned paths,
   confirm — per global CLAUDE.md rule 7.
7. `experiments/runs/<run_id>.yaml` is the truth source for "who runs what";
   register there + `mkdir .uwm/locks/<run_id>` before starting.

---

## 5. Typical run workflow

1. `cp experiments/configs/_TEMPLATE.yaml experiments/configs/<run_id>.yaml` — fill it.
2. `cp experiments/runs/_TEMPLATE.yaml experiments/runs/<run_id>.yaml` — `status: planned`.
3. `source scripts/env.sh` → `scripts/claim_run.sh <run_id>`.
4. Run (e.g. `sbatch scripts/run_baseline.slurm --run-id <run_id>`); artifacts → `$UWM_RUNS/<run_id>/`.
5. On finish: write `experiments/results/<run_id>__acc.json`; set run yaml `status: done` + `acc`.
6. **driver:** `python scripts/gen_runs_md.py` → `git add` the small files → commit → push.
