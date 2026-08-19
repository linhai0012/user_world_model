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
general-track snapshot: `docs/status/PROJECT_STATUS_2026-05-29.md`

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
| headline result (small)    | `experiments/results/<domain>/<run_id>__<metric>.json` |
| large artifacts            | `$UWM_SCRATCH/runs/<run_id>/` |
| index (generated)          | `RUNS.md` ← `scripts/gen_runs_md.py` |

`<metric>` ∈ `acc`, `acc_by_qtype`, `ppl`, `mae`, `nll`. `<domain>` ∈ `general`,
`health`, `education`. **One YAML file per run** so two agents never edit the same
file (merge-safe).

> **Where practice differs from this grammar (2026-08-09).** The grammar above is
> PersonaMem-shaped and is followed by the general-domain runs. Health and education
> runs predate it and use `<arm>__<metric>.json` (`health_shared_current__mae.json`,
> `edu_stage1__nll.json`); the `configs/` + `runs/` registry has not been used by any
> run so far, so `RUNS.md` is empty. See `experiments/README.md` for the two ways to
> close that gap — don't assume the registry reflects what has run.

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

This doc governs all three domains. Each is `project_summary.md` §8.2's ablation
(`base` vs `+profile` vs `+memory` vs `+per-user weights`) instantiated once, and the
eval contract below is its `project_summary.md` §6 Stage-1 intrinsic-prediction layer.

**Layout rule: domain code is separated, infrastructure is shared.** The domains have
non-overlapping users and unrelated tasks — "unified" means the same recipe run three
times, not one model over pooled data — so `domains/<domain>/` never imports from
another domain. Anything truly common goes to `common/`, and keeping that set small is
what makes cross-domain comparisons about the domains rather than about the plumbing.

```
ACTIVE work (governed by this doc):
  domains/general/     data.py · peruser_data.py · scorer.py ·
                       baselines/{trivial,profile,oracle,tokenmem/{naiverag,fluxmem,mem0,zep,amem}}
  domains/health/      data.py · peruser_data.py
  domains/education/   data.py
  common/              backends.py (vllm_qwen / openai_gpt) · runmeta.py · sft.py (generic SFT tokenizer)
  scripts/             env.sh · claim_run.sh · gen_runs_md.py
    general/ health/ education/    per-domain entry points + .slurm launchers
  experiments/         configs/ runs/ reports/ · results/{general,health,education}/

Framework & legacy (see project_summary.md):
  project_summary.md / docs/   all-purpose design (profile + memory + per-user weights)
  data/education/              private KCL course-chat data (small, in-repo)
  legacy/general_personamem/   frozen Phase 0–2b + OPSD prototype (the prior repo)
  legacy/health_digitaltwin/   reusable health-domain code (LLM-based-Digital-Twins)
  legacy/education_parametric_memory/  snapshot of the ACTIVE sibling per-user-weights
                               repo (~/parametric_user_memory) — the edu-exam track
```

**Evaluation contract (makes arms comparable):**

- Within a domain, every arm differs **only** in what it is given or trained on — same
  loader, same split, same decoding (temp 0), same scoring path. That is the whole point
  of the shared `common/`.
- **general**: methods implement `build_context(mcq, data, params)`; primary metric =
  accuracy on PersonaMem single-choice, always also per-qtype over the 7 categories; PPL
  is secondary and model-internal. Backends: `qwen3-4b` via local vLLM, `gpt41` via OpenAI.
- **health**: per-field + overall MAE of the next-day state, always reported against the
  trivial bars (persistence, pop-mean, per-user-mean) — an LLM number without them is
  uninterpretable. Reaction text: NLL.
- **education**: NLL of the student's next turn, always with its context controls
  (shuffled = content+length-matched, foreign = other session).
- **Ship every claim with its control.** A gain measured only against "no context" or
  "frozen base" is not yet evidence; the shared/trivial/shuffled controls are what decide
  whether it survives. This applies in both directions — a null under one recipe is not
  evidence that the arm cannot work.

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
  each `domains/<domain>/*.py`, and the root docs (`CONVENTIONS` / `EXPERIMENTS` /
  `KNOWLEDGE` / `CLAUDE` / `README`). **The only process that runs git.**
- **runner** (auxiliary sessions): writes only
  `domains/general/baselines/<its method>/`, `experiments/configs|runs/<its run_id>.yaml`,
  `experiments/results/<domain>/<its run_id>__*.json`, and `$UWM_SCRATCH/runs/<run_id>/`.
  **Never runs git.**

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
4. Run — e.g. `sbatch scripts/general/run_baseline.slurm --run-id <run_id>`, or directly
   `python scripts/<domain>/<script>.py`; artifacts → `$UWM_RUNS/<run_id>/`.
5. On finish: write `experiments/results/<domain>/<run_id>__<metric>.json`; set run yaml
   `status: done` + the headline metric.
6. **driver:** `python scripts/gen_runs_md.py` → `git add` the small files → commit → push,
   and mirror the headline numbers into `EXPERIMENTS.md` with the controls they were
   measured against.

> Steps 1–2 and 6's `gen_runs_md.py` describe the intended registry; no run has used it yet
> (`experiments/README.md`). Either adopt it from the next run on, or retire it — but keep
> step 5 and the `EXPERIMENTS.md` entry either way, since those are the actual record.
