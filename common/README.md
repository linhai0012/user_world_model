# common/ — shared library (driver-owned; see CONVENTIONS.md §4)

PersonaMem / LoCoMo loaders (reuse `data_prep/` where possible), scorers
(accuracy + ppl), model backends (`vllm_qwen`, `openai_gpt`), and run-meta
I/O. Every baseline imports from here so all runs share **one** data loader
and **one** scoring path — that is what makes the baselines comparable.

> Only the driver session edits this directory. Runners that need a change
> here request it from the driver (CONVENTIONS.md §4 rule 1).
