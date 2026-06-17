# EXPERIMENTS — user_world_model (active work, from 2026-06)

> Experiment log for the **active all-purpose framework build**. Design spec:
> `project_summary.md`. Operational conventions: `CONVENTIONS.md`.
> Prior per-domain logs are frozen under:
> - `legacy/general_personamem/EXPERIMENTS.md` — Phase 0–2b + OPSD (PersonaMem prototype)
> - `legacy/health_digitaltwin/EXPERIMENTS.md` — digital-twin health prototype

## Current focus

General domain first (`project_summary.md` §8): an ablation skeleton
`base` vs `+profile` vs `+memory` vs `+per-user weights` on PersonaMem, with
comparable token-based-memory baselines (oracle / trivial / FluxMem / Mem0 / …)
under one eval contract (`CONVENTIONS.md` §3 — accuracy + per-qtype).

## Log

<!-- newest first; one entry per milestone. Mirror headline numbers from
     experiments/results/<run_id>__*.json; full data stays on cluster scratch. -->

### 2026-06-17 — baseline harness live; dual-lens token-memory ablation on PersonaMem-v1 (pm32k)

**What landed.** First runnable general-domain harness (`common/` shared lib + `baselines/`
+ `scripts/run_campaign.py`), validated end-to-end on 1× B200 (vllm_env, vLLM 0.13, FLASH_ATTN).
Methods implement the shared `build_context(mcq,data,params) -> list[dict]` contract; one shared
backend scores them, so arms are comparable (CONVENTIONS §3). Code commit `ae14d87`.

**Two eval lenses** (both report accuracy + per-qtype on the 589 pm32k MCQs):
- **PPL** — MCQ choice-perplexity (the legacy UserSim protocol: argmin mean-NLL of each option
  as the assistant continuation; faithful to `legacy/.../teacher_sft/eval_mcq_ppl.py`). The lens
  for the eventual per-user-weights (UserSim) arm.
- **gen (reader)** — the model *answers* the MCQ given the injected memory (how A-MEM/Mem0-style
  token-memory baselines are evaluated). The lens for the token-memory arms.

**Protocol validation + a bug caught.** profile (demographics-only) PPL = 0.353 ≈ legacy base-32k
0.345, and pm128k profile-PPL = **0.304 ≈ legacy base-128k 0.306** (exact anchor) → the MCQ-PPL
protocol is faithful to `legacy/.../eval_mcq_ppl.py`. First oracle was mislocated: it sliced ±2 turns around `end_index`
(the *query* position), but `distance_to_ref_in_tokens` shows the reference sits far earlier
(~2k tokens from the start of a 27k context). Fixed by locating the reference via **token
distance** walked back from the query (`common/data.py::_ref_index`, uses the Qwen tokenizer).

**Headline — pm32k ablation (acc; random=0.25):**

| arm | PPL lens | reader lens |
|---|---:|---:|
| profile (+demographics) | 0.353 | 0.379 |
| trivial (no memory, = base/floor) | 0.389 | 0.435 |
| naiverag BM25 top-5 (retrieval) | — | 0.584 |
| oracle-full (whole 27k context) | — | 0.584 |
| naiverag dense top-5 (MiniLM) | — | 0.601 |
| oracle-slice (golden ±2-turn ref) | 0.326 | **0.683** |

**Findings (general-domain, frozen Qwen3-4B-Instruct-2507):**
1. **Frozen base + injected memory does NOT help under PPL** (trivial 0.389 ≥ oracle 0.326);
   memory even hurts. Matches the legacy "base context-benefit ≈ 0" and the pilot's see≠use —
   and motivates *training* (the per-user-weights arm) rather than context injection.
2. **A reader DOES use the golden reference** (+25pp, 0.435 → 0.683): the same model, asked to
   answer, exploits the exact reference. The PPL-vs-reader divergence is itself a result.
3. **Demographics alone hurt** (−0.06 both lenses): the persona card lacks the specific fact and
   pulls the model toward persona-matching distractors (the snippet-literal / wrong-facet trap).
4. **Needle ≫ haystack**: the focused golden slice (0.683) beats the full 27k context (0.584) by
   ~10pp — long-context dilution → motivates retrieval.
5. **Token-memory baseline (the #1 documented gap, now filled for pm32k):** realistic retrieval
   recovers most of the oracle ceiling but leaves an **~8pp gap** (dense 0.601 / BM25 0.584 vs
   oracle 0.683), and retrieval ≈ full-context. This is the bar the `+per-user weights` arm must
   beat. (dense MiniLM is CPU-bound: 12 min vs BM25 12 s.)

**pm128k (2710 MCQs, reader lens) — the longer-context, more memory-dependent regime:**

| arm | reader acc | Δ vs base |
|---|---:|---:|
| profile (+demographics) | 0.327 | −0.06 |
| trivial (base) | 0.385 | — |
| oracle-slice (golden ±2-turn) | 0.487 | +0.10 |
| naiverag BM25 top-5 | 0.494 | +0.11 |

- base is lower than pm32k (0.385 vs 0.435) → a bit more headroom, but PersonaMem-v1 is still
  largely answerable without persona. demographics still hurt.
- **On pm128k, retrieval ≈ oracle-slice** (0.494 vs 0.487), unlike pm32k where oracle led by
  ~8pp. The aggregate tie hides a **qtype crossover**: naiverag beats oracle on `recall_facts`
  (0.561 vs 0.479) and `acknowledge_latest` (0.565 vs 0.452, the 128k-only "latest preference"
  qtype, n=866) — retrieval finds the specific/recent turn better than a fixed ±2 slice — while
  oracle wins `aligned_rec` (0.645 vs 0.504) and `generalize` (0.657 vs 0.460), where the
  contiguous reference context carries the reasoning. Suggests the ±2 oracle window is undersized
  for dispersed 128k references (a wider/adaptive oracle is a TODO), and that retrieval and a
  fixed golden slice are complementary across qtypes.

**Cross-version trend (reader lens, oracle ±2-slice vs BM25 retrieval as context grows):**

| | pm32k | pm128k | pm1m |
|---|---:|---:|---:|
| trivial (base) | 0.435 | 0.385 | 0.356 |
| oracle ±2-slice | 0.683 | 0.487 | 0.379 |
| naiverag BM25 | 0.601 | 0.494 | 0.438 |

The fixed ±2-turn oracle **degrades as context grows** (0.68→0.49→0.38) — at 1M the window is a
poor approximation of a dispersed reference — while **retrieval scales** and overtakes it
(pm32k oracle≫rag → pm128k tie → pm1m rag>oracle by +6pp). Two reads: (a) retrieval is the
robust token-memory method as context grows; (b) my fixed-window oracle is an imperfect ceiling
at long context (widen/adapt the window, or trust the token-located slice less at 1M — TODO).
Base accuracy also falls with version (0.44→0.39→0.36) → more memory headroom at 1M.

**Net token-memory-baseline result (the #1 documented gap, now filled for pm32k+pm128k+pm1m):**
realistic retrieval lifts a frozen reader **+11–17pp** over no-memory, demographics **hurt**,
and there remains a gap to the focused oracle on pm32k (8pp) that closes on pm128k. This is the
quantified bar the `+per-user weights` arm must beat.

**`+per-user weights` arm — first result (PRELIMINARY, minimal recipe).** Per-persona LoRA via
direct CE on that persona's own user-turns (no teacher; r32 all-proj, 3 epochs; merged for
serving — `scripts/train_peruser.py`), evaluated apples-to-apples on that persona's pm128k MCQs.
**Persona 0 (154 MCQs):**

| arm | PPL | reader |
|---|---:|---:|
| base (no mem) | 0.409 | 0.370 |
| +profile | 0.331 | 0.370 |
| +memory (naiverag) | 0.279 | 0.422 |
| **+per-user weights** | 0.338 | 0.351 |

The naive per-user LoRA **does NOT beat token-memory**: under PPL it lands ≈ +profile and below
base; under the reader lens it is the lowest (below base, well below retrieval 0.422). The
merged LoRA *did* shift PPL (base 0.409→0.338), so training had an effect — just not a
beneficial one for MCQ discrimination. Consistent with the legacy intuition: the persona-swap
probe found ~99% of user-SFT gain is generic style, not persona-specific; and R1b's +95.7%
closure used a **different** recipe (OPD distillation from a context-bearing teacher + dual-rate
slow/fast LoRA + per-persona best-step selection), **not** direct CE. **Caveats:** single noisy
persona (n=154); 3-epoch CE may overfit the user's verbose style at the cost of discrimination
(the reader-lens drop echoes the legacy "user-only-loss LoRA hurts instruction-following");
persona 4 re-running after a mid-train SIGKILL. **Read:** a per-user-weights win on
PersonaMem-v1 needs the OPD recipe (or fewer epochs / dual-rate), not naive SFT — that is the
documented next step; the token-memory baselines stand as the reference bar.

**Caveats.** The `+per-user weights` arm has only a preliminary single-persona, minimal-recipe
result (above). PersonaMem-v1 is substantially
answerable with no persona (trivial reader 0.385–0.435 ≫ 0.25), so headroom is modest.
`suggest_new` stays low everywhere (open-ended qtype). dense retrieval is CPU-bound (12 min vs
BM25 12 s on pm32k) so pm128k used BM25 only. Results JSON + per-record preds under
`experiments/results/` + `$UWM_RUNS/<run_id>/`.

### 2026-06-03 — repo reorganized to all-purpose; operating layer restored
- Upstream `810e9dc` moved the PersonaMem prototype to `legacy/general_personamem/`
  and digital-twin code to `legacy/health_digitaltwin/`; added `data/education/`,
  `project_summary.md`, `docs/`.
- Restored/aligned the operating layer: method-agnostic `CLAUDE.md`,
  project-specific `CONVENTIONS.md`, and the `baselines/ common/ experiments/ scripts/`
  scaffolding (general-domain active build).
- No experiments run yet under the new framework.
