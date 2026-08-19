# domains/ — one subpackage per domain

The project's organizing principle: **three independent domains, one recipe applied to each.**
Users are non-overlapping, the tasks are unrelated; "unified" means the *same* method
instantiated three times, **not** one model over pooled data. So each domain owns its data
loader and its per-user-weights sample builder, and **nothing here imports across domains**.

| domain | what we predict | data | code |
|---|---|---|---|
| `general/` | the user's reply / preference reaction (PersonaMem MCQ) | `$UWM_DATA/personamem` | `data.py` · `peruser_data.py` · `scorer.py` · `baselines/` |
| `health/` | next-day wellness state (+ reaction text) | `$UWM_DATA/health/digitaltwin` | `data.py` · `peruser_data.py` |
| `education/` | the student's next turn | `data/education/` (in-repo, private) | `data.py` |

Shared, domain-agnostic pieces live in [`../common/`](../common/): model backends
(`backends.py`), run config / run-meta I/O (`runmeta.py`), and the generic SFT tokenizer
(`sft.py`) that keeps every domain's per-user loss the same shape.

Entry points are in [`../scripts/`](../scripts/), mirrored by domain
(`scripts/general/`, `scripts/health/`, `scripts/education/`); headline results land in
`experiments/results/<domain>/`.

> `general/baselines/` is general-only by construction — those are PersonaMem MCQ methods
> (trivial / profile / oracle / token-memory). A runner owns one method subdir and claims it
> via `scripts/claim_run.sh domains/general/baselines/<method>` (CONVENTIONS.md §4).
>
> A second education track exists as reference material, not as code here:
> `legacy/education_parametric_memory/` (the edu-exam / per-user-weights instance; canonical
> repo `~/parametric_user_memory`).
