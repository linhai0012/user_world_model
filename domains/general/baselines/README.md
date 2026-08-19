# domains/general/baselines/ — one subdir per method (runner-owned per method; CONVENTIONS.md §4)

| dir | role |
|-----|------|
| `oracle/`   | inject the official MCQ reference into the prompt — **upper reference** |
| `trivial/`  | no memory, answer directly — **lower reference** |
| `tokenmem/` | retrieval-into-context memory systems: `fluxmem/` `mem0/` `zep/` `amem/` `naiverag/` |

All implement the common `predict(mcq, context)` interface and are scored by
the shared `common/` scorer. **Primary metric = accuracy** (+ per-qtype).
Models: `qwen3-4b` (local vLLM) and `gpt41` (OpenAI API).

> A runner owns the method subdir it is implementing and claims it via
> `scripts/claim_run.sh domains/general/baselines/<method>` before editing.
