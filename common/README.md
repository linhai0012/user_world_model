# common/ — shared, domain-agnostic infrastructure (driver-owned; CONVENTIONS.md §4)

Only things that are genuinely the same across **general / health / education** belong here.
Anything that knows about a specific dataset, prompt, or metric belongs in
[`../domains/<domain>/`](../domains/).

| file | role |
|---|---|
| `backends.py` | model backends — `vllm_qwen` (local) and `openai_gpt`; batched MCQ-PPL scoring and generation |
| `runmeta.py`  | `experiments/configs/<run_id>.yaml` loading (`${UWM_*}` expansion) + `run_meta.json` into the scratch run dir |
| `sft.py`      | the generic `{messages, target} → {input_ids, labels}` tokenizer, so every domain's per-user-weights arm trains on the same loss shape (CE on the target span only) |

Keeping this set small is deliberate: it is what makes a cross-domain comparison a comparison
of *domains*, not of incidentally-different plumbing.

> Only the driver session edits this directory. Runners that need a change here request it
> from the driver (CONVENTIONS.md §4 rule 1).
