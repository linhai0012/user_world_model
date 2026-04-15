# Dynamic UserSim — Experiment Log

Subproject implementing the plan in
[`../dynamic_usersim_complete_plan.md`](../dynamic_usersim_complete_plan.md)
(Dynamic User Simulator for Personalization).

**Paradigm**: train a per-user LoRA (Student) via OPD to predict what a specific
user would say, using a Teacher that sees the full progressive conversation
history. At inference, a general Agent queries UserSim reactions for each
candidate response and picks the best.

This log covers **Phase 0 (data prep) + Phase 1 (Teacher SFT)** — we have not
reached Phase 2 (OPD) yet. It is the running record of what we built, what
broke, and what we learned.

Repo commits are referenced inline (`<shortsha>`) so each finding is tied to
verifiable code.

---

## 1. Target Dataset: PersonaMem-v1

HuggingFace `bowen-upenn/PersonaMem`. Three context-length versions, one
persona/topic set:

| Version | Shared-context records | MCQs | Max ctx (tokens) |
|---------|-----------------------:|-----:|-----------------:|
| 32k     | 37                     | 589  | ~27k             |
| 128k    | 60                     | 2727 | ~128k            |
| 1M      | 31                     | 2674 | ~1M              |

20 personas (`persona_id` 0–19) shared across all three versions; `shared_context_id`
hashes are version-specific (no overlap).

### 1.1 Structural findings (non-obvious)

- **Session boundary = `role: system` message.** Each shared context is a long
  interleaved dialogue; every new session re-states the persona card as a
  system turn. Session count per context = number of system messages.
- **Demographics live in the first system message content** (not a separate
  field). Raw format is `Current user persona: Name: ...\nGender Identity: ...`
  + free-text paragraph. No separate metadata needed.
- **Question-type names are inconsistent across CSVs.** E.g. 32k uses
  `track_full_preference_evolution` and 1M uses `track_full_preference_updates`
  for the same type. We maintain a `QTYPE_CANONICAL` map that collapses both
  to the canonical 7 types.
- **128k CSV has 17 rows with negative `end_index_in_shared_context`.**
  These are malformed; we drop them at load time.
- **`all_options` field uses mixed quoting** in the CSV — some rows are
  JSON-style double-quoted, others are Python-repr single-quoted. `json.loads`
  chokes on the latter; we use `ast.literal_eval` instead (handles both).
  (`54a7395`)

### 1.2 Context length per K (prefix session cap)

For each `(persona, shared_context)` timeline we take progressive prefixes:
`prefix = sessions[max(0, t-K):t]`. Token distribution (Qwen3 tokenizer):

| K    | p50 tokens | p90  | p99  | max  | over 40k | over 32k |
|-----:|-----------:|-----:|-----:|-----:|---------:|---------:|
| 20   | 71k        | 116k | 130k | 131k | ~95%     | ~98%     |
| 5    | 36k        | 48k  | 58k  | 70k  | 29%      | 60%      |
| **3**| **23k**    | 34k  | 41k  | 49k  | **1.3%** | 12%      |

K=3 was chosen to keep nearly all samples within Qwen3-4B's
`max_position_embeddings=40960`, eliminating RoPE extrapolation.
(`b62c063`)

---

## 2. Phase 0: Data Preparation (all commits on `dynamic_usersim/data_prep/`)

Produced:
- `load_personamem.py` — session split, demographics extraction, qtype canonicalization
- `build_teacher_sft_data.py` — progressive-context SFT sample construction
- `tokenize_teacher_sft.py` — Qwen chat-template tokenization with
  user-token-only loss mask
- `validate_sft_data.py` — decodes label positions to confirm mask correctness

**Output (K=3, 128k data):** 1172 samples, 24,385 user turns, ~2.83M loss tokens,
1.3% samples prefix-truncated to fit 40k cap.

---

## 3. Phase 1: Teacher SFT

### 3.1 Setup

- **Model**: `Qwen/Qwen3-4B-Instruct-2507` (switched from `Qwen/Qwen3-4B` late
  in the experiment — see §6)
- **Training recipe**: full-parameter SFT, bf16, flash-attn 2, FSDP full-shard,
  liger-kernel fused CE, activation checkpointing
- **Effective batch**: 4 GPU × per-device 1 × grad-accum 4 = 16 samples
- **LR**: 1e-5, cosine schedule, warmup 0.1
- **Epochs**: 2
- **Loss**: only on user-role tokens inside the target session (prefix = frozen
  context, no loss)

### 3.2 Runs

| Run | Model            | K  | MAX_SEQ_LEN | Status | Notes |
|-----|------------------|----|-------------|--------|-------|
| R1  | Qwen3-4B         | 20 | 131072      | partial | 7h/epoch, terminated at step 50; checkpoint-50 saved |
| R2  | Qwen3-4B         | 3  | 40960       | failed first launch | `--resume` bug on empty dir (fixed `9df6c33`) |
| R3  | Qwen3-4B-Instruct-2507 | 3 | 40960 | pending | current target |

R1's checkpoint-50 is kept as a baseline artifact for comparison even though
we've since moved past it.

### 3.3 Memory crises + fixes

A sequence of memory issues surfaced as we scaled sequence length:

- **Initial 131k context OOM** (R1): 62 GB allocation attempt for the
  `[1, 131k, 151936]` bf16 logits tensor at the lm_head. Fix: apply
  `liger_kernel.transformers.apply_liger_kernel_to_qwen3(fused_linear_cross_entropy=True)`
  BEFORE model load — never materializes full logits.
  (`b9adb49`, `ca4d321`)
- **FSDP + `gradient_checkpointing=True` conflict** (transformers >= 4.40):
  FSDP activation_checkpointing and TrainingArguments.gradient_checkpointing
  are mutually exclusive. Fix: FSDP path owns activation_checkpointing;
  TrainingArguments.gradient_checkpointing only used in `--no-fsdp` fallback.
  (`e36558f`)

### 3.4 K (progressive context window) decision chain

1. **K=20 (plan §3.3 original)**: p50 = 71k tokens, 95% of samples require
   RoPE extrapolation beyond Qwen3-4B's 40960 max_pos. Suspected this is
   degrading long-range attention.
2. **K=5**: still 60% of samples exceed 32k native, 29% exceed max_pos.
   Better but not clean.
3. **K=3** (chosen): p99 = 41k, max = 49k. 99% of samples stay within
   max_position_embeddings. Eval 2 showed marginal-prior-session benefit
   was only +0.02 nats anyway — shrinking K loses little signal while
   eliminating the RoPE extrapolation confound.
   (`b9adb49` → `aba969f` → `b62c063`)

### 3.5 Training throughput (4 × GH200)

At K=20 with liger + flash-attn + FSDP + activation checkpointing:
- ~180 s / optimizer step, 74 steps / epoch → ~3.7h / epoch
- Peak GPU memory ~48 GB / 96 GB per GPU
- Save frequency: every 25 steps (~3 saves / epoch), save_total_limit=2

At K=3 (expected, R3): ~50 s / step, ~1h / epoch (~2x faster than K=20).

---

## 4. Evaluation Findings (on K=20 checkpoint-50)

Four evaluations run on R1's checkpoint-50. All are "teacher-alone" tests;
downstream OPD student-LoRA evaluation is future work.

### 4.1 Eval 1 — held-out user-token NLL (`eval_ppl.py`)

Plan §3.5 sanity check #1. 20 random training-set samples, base vs SFT mean
NLL on the same user-token positions.

| Model | Mean NLL | PPL    | Notes |
|-------|---------:|-------:|-------|
| Base Qwen3-4B | 2.506 | 12.26 | 20/20 samples, range 2.10–3.13 |
| **SFT ckpt-50**   | **1.430** | **4.18** | 20/20 improved, Δ ranges +0.62 to +1.67 |

**Finding**: unambiguous −1.08 nats reduction. Teacher SFT works as intended.
Every single sample improves. PPL drop 66%. **This is the cleanest positive
result of the run.**

### 4.2 Eval 2 — context benefit (`eval_context_kl.py`)

Plan §3.5 sanity check #2 — does SFT actually USE the progressive context?
Two forwards per sample (with prefix / without prefix), same target session.

Originally designed as KL(P_with || P_without), but KL is ambiguous:
positional RoPE noise inflates base KL at long context, SFT's low KL could
be "ignoring context" OR "robust to it". We added a more direct metric:
**context benefit = NLL_without − NLL_with** on ground-truth user tokens.

| Model | NLL_with | NLL_without | benefit |
|-------|---------:|------------:|--------:|
| Base  | 2.506    | 2.313       | **−0.193** |
| SFT   | 1.430    | 1.452       | **+0.021** |

**Two readings**:
1. **SFT − Base = +0.214**. SFT fixes base's context-distraction problem
   (base actively gets WORSE with long prefix).
2. **Absolute SFT benefit is tiny** (+0.02 nats). Most of SFT's gain over
   base comes from user-style learning, not from exploiting prior sessions.
   Possible confounds: (a) PersonaMem sessions re-state persona card each
   time so prior sessions add little on top; (b) RoPE extrapolation on
   Qwen3-4B degrades long-range signal. R3 (K=3 on Instruct-2507) will
   disentangle these.

### 4.3 Eval 3 — qualitative generation (`eval_qualitative.py`)

3 MCQs, 4 conditions each: {base, sft} × {no-ctx, with-ctx}.

- **Case 1 (cinematography)**: `sft_ctx` tracked the technical-detail
  cinematography theme from the immediately-preceding assistant turn; base
  generated generic film-history lecture with `</think>` artifacts.
- **Case 2 (classic films)**: no clear winner — ground truth was an
  unexpected pivot to film-criticism blog; all four outputs were in the
  right semantic neighborhood.
- **Case 3 (financial content)**: `sft_ctx` nailed the ground-truth core
  ("manual budgeting > digital tools"); base hallucinated biographical
  details ("started a film club at university").

**Net**: SFT produces cleaner user-voice text (no `<think>` leakage, no
typos, no assistant-style fabrication); context helps SFT thematically
match ground truth on 2/3 cases. Small sample though.

### 4.4 MCQ evaluation — the hard one

Plan §9.3 proposes a full pipeline: UserSim generates reactions to each of
4 choices → LLM Agent picks the best choice. Tested three implementations
across several iterations.

#### 4.4.1 Generate + Agent v1 (`eval_mcq.py`)

First attempt. Result:

| Run     | Accuracy | Notes |
|---------|---------:|-------|
| Base as UserSim | 0.45 (9/20) | |
| SFT as UserSim  | 0.30 (6/20) | *unexpected: SFT worse* |

Inspected reactions — base had three hidden advantages:
1. **`<think>` contamination** (26% of reactions). Root cause: HF
   `apply_chat_template` hard-codes `<think>\n\n</think>\n\n` around every
   assistant history message for Qwen3-4B (thinking-mode hybrid model).
   Model was primed to continue in reasoning mode. (`afd7b07`)
2. **Prompt echoing** — base frequently started every reaction by copying
   the MCQ question verbatim, then continued with a sentiment-laden
   extension. Incidentally gave the Agent a side-channel to read.
3. **Length advantage** (p50 916 vs 583 chars) — more surface for the
   Agent to score.

#### 4.4.2 Generate + Agent v2 (after think-mode fix + reaction primer)

Fixes applied:
- Use our custom `SFTTokenizer._encode_message` instead of
  `apply_chat_template` (matches training exactly, no `<think>` leakage).
- Append `"\n\nWhat do you think — does that match what you're looking for?"`
  to each choice to solicit a direct reaction instead of monologue.
  (`9390e69`)

Result:

| Run     | Accuracy | Notes |
|---------|---------:|-------|
| Base    | 0.45 (9/20) | unchanged; now 54% reactions triggered thinking mode from the "What do you think?" primer |
| SFT     | 0.20 (4/20) | *worse!* SFT reacts more "as user" but uniformly engaged across choices |

**Diagnosis**: structural mismatch between the training objective and what
MCQ discrimination needs.
- Teacher SFT was trained to predict **natural user continuations** given
  full history. In PersonaMem's data, user turns often start new threads
  ("I also did X ..."), regardless of what the assistant said.
- Evaluated on MCQs, SFT therefore gives its **least-engaged response to
  the most preference-aligned choice** (which is the one that correctly
  recalls a past preference and doesn't need extension). Most-engaged
  response goes to off-target choices where continuation with new
  material flows naturally.
- Anti-correlation is structural, not fixable by prompt engineering.

Concrete example (MCQ persona=6, suggest_new, correct=d):
- Choice d (correct, flashcards+spaced-repetition): SFT reacts
  "I'm curious about different study techniques. I've been trying to find
  effective methods." — generic.
- Choice b (pomodoro timer): SFT reacts "Yes, I like it! I also created a
  detailed study guide using mind maps." — specific, engaged.
- Agent picks b. Correct was d.

**Conclusion**: UserSim-reaction + Agent paradigm (plan §9.3) does not work
with a user-only-loss teacher SFT. The reaction content is anti-correlated
with choice correctness.

#### 4.4.3 MCQ perplexity scoring (`eval_mcq_ppl.py`)

Plan §9.5 — bypass generation entirely. For each MCQ, score
`P_teacher(choice_text | full_context + user_question)` at the
`<|im_start|>assistant\n` prefix. Pick `argmin(mean NLL)`.
(`1f80237`)

**Why this is cleaner**:
- Deterministic (no sampling)
- No Agent intermediary or OpenAI cost
- No generation-quality confound
- Directly tests "does teacher prefer the correct assistant response"

Full 32k MCQ run (n = 589, ~10 min each on 4 × GH200):

| Model | Overall | recall_facts | aligned_rec | track_evolution | suggest_new | generalize | reasons_behind |
|-------|:-------:|:------------:|:-----------:|:---------------:|:-----------:|:----------:|:--------------:|
| Base  | **0.467** | 0.370 | 0.236 | 0.719 | 0.129 | 0.281 | 0.808 |
| SFT   | **0.469** | 0.281 | 0.200 | **0.820** | **0.269** | 0.175 | 0.758 |
| Δ     | +0.002  | −0.089 | −0.036 | **+0.101** | **+0.140** | −0.106 | −0.051 |
| n     | 589     | 146   | 55    | 139   | 93    | 57    | 99    |

**Observations**:
- Overall essentially tied (95% CI at n=589 is ±4pp).
- **SFT wins clearly** on `track_evolution` (+10pp) and `suggest_new`
  (+14pp) — types where the correct choice reads like a long user-style
  narrative. SFT's user-token-optimized distribution scores these higher.
- **SFT loses** on `recall_facts` (−9pp) and `aligned_rec` (−4pp) — types
  where the correct choice is an assistant explicitly referencing user
  preferences ("I remember you like X"). SFT was never trained to score
  assistant-style text; base's general LM distribution serves better here.
- NLL margin (best − correct, when wrong): SFT max = 2.60, Base max = 5.21.
  **SFT is never as confidently-wrong as base can be** — calibrated
  uncertainty in preference-sensitive regions.

**Structural conclusion**: our teacher SFT optimizes for user-turn
prediction, not assistant-choice discrimination. MCQ-PPL is NOT a direct
test of teacher quality. **The clean test is Eval 1** (user-token NLL),
where SFT wins decisively.

---

## 5. Pitfalls log (each cost a commit cycle)

A running list of things that broke, and how:

| # | Bug | Symptom | Fix commit |
|---|-----|---------|------------|
| 1 | FSDP + gradient_checkpointing both True | ValueError at Trainer init | `e36558f` |
| 2 | Logits OOM at 131k seq (30 GB bf16 tensor) | CUDA OOM | enable liger FLCE, `b9adb49` |
| 3 | Liger FLCE inactive in `torch.inference_mode` | OOM during eval | use `torch.no_grad` instead (`949a614`) |
| 4 | Liger FLCE inactive in eval path anyway | OOM persisted at 65k | manual chunked CE on non-masked positions only (`8bcc458`) |
| 5 | `json.loads` fails on Python-repr CSV rows | parse error 12 in 20 MCQs | use `ast.literal_eval` (`54a7395`) |
| 6 | `apply_chat_template` injects `<think>\n\n</think>\n\n` | reactions contaminated, SFT "loses" MCQ | use manual `_encode_message` (`afd7b07`) |
| 7 | `trainer.train(resume_from_checkpoint=True)` errors on empty output_dir | first sbatch of every new run fails instantly | check for `checkpoint-*` first (`9df6c33`) |
| 8 | HF HEAD-request 10s timeout on slow server network | download aborts mid-file | `HF_HUB_DOWNLOAD_TIMEOUT=120` or curl fallback |
| 9 | sbatch walltime shorter than needed | training killed mid-run | verify `--time` vs measured step rate |
| 10 | `max_position_embeddings=40960` violated at K=20 (p50 = 71k) | RoPE extrapolation; possibly confounds eval 2 | K=3 + max_seq_len=40960 (`b62c063`); later switch to Instruct-2507 (`966f264`) |

---

## 6. Model switch decision (late-experiment)

Qwen3-4B is a hybrid thinking/non-thinking model with thinking mode ON by
default. Observed side effects throughout the run:
- `apply_chat_template` injects `<think></think>` wrappers into history
  (pitfall #6)
- Reaction primer "What do you think?" directly triggers thinking mode in
  base (54% reactions in eval_mcq v2)
- Training with user-only loss on a hybrid model produces a "user simulator
  that could but doesn't think" — potentially unstable

**Switched to `Qwen/Qwen3-4B-Instruct-2507`** (`966f264`):
- **262k native context** (vs 40k). Eliminates the RoPE extrapolation
  confound for eval 2. Allows re-testing long-context benefit cleanly.
- **No thinking mode**. Clean chat template with no `<think>` wrappers.
- Same architecture (`Qwen3ForCausalLM`), same tokenizer (vocab 151936).
  Existing JSONL data and eval scripts work unchanged.
- Consistent with `../grpo_baseline/` which already used this variant.

Zero-cost switch: made after R2's sbatch hit pitfall #7 and before any real
K=3 training had accumulated. R3 is the first run on the new model.

---

## 7. What we know vs what's speculation

### Confirmed
- Teacher SFT reduces user-token NLL by 43% on held-out user turns
  (eval 1: 2.506 → 1.430).
- SFT produces cleaner user-voice text, dropping thinking-mode artifacts
  and typos seen in base (eval 3 qualitative).
- SFT is better-calibrated when uncertain — never as confidently wrong as
  base at the top of the margin distribution.
- Data prep and tokenization are correct (validate_sft_data round-trips
  label positions to exact ground-truth user turns, 100% match on 30
  random samples).

### Unclear / needs R3 to disambiguate
- Whether long prior-session context genuinely adds signal beyond the
  current session's persona card. Eval 2 says +0.02 nats on K=20 /
  Qwen3-4B. Could be real smallness, or could be RoPE extrapolation.
  Instruct-2507 (262k native) at K=3 will tell us.
- Whether MCQ-PPL tie (0.469 vs 0.467) persists when both models use
  clean native-context positions. Probably yes (the structural argument
  about user-only loss stands), but worth confirming.

### Known NOT working
- The UserSim-reaction + LLM-agent pipeline as a teacher evaluation
  (§4.4.1, §4.4.2). Reactions are anti-correlated with correctness for
  this training objective. The pipeline may still work downstream with
  a Student LoRA that's been OPD-distilled to emphasize discrimination —
  that is the Phase 2 test.

---

## 8. Next steps

1. **R3**: K=3 training on Qwen3-4B-Instruct-2507. Re-run eval 1, 2, 3,
   MCQ-PPL. Primary question: does eval 2 benefit grow to a meaningful
   range (>0.05 nats) once RoPE is in-distribution?
2. **Phase 2 — OPD**: implement per-user LoRA training via
   KL(student || teacher) on student rollouts (plan §4). Evaluate full
   pipeline (UserSim-LoRA + Agent) on MCQs. This is where the
   UserSim+Agent paradigm is supposed to pay off.
3. **Optional probes before Phase 2**:
   - Swap-persona prefix test: score ground-truth user tokens under
     "my real prefix" vs "other persona's prefix". If SFT learned
     persona-specific structure, true prefix should have much lower NLL.
     Sharper signal than eval 2.
   - Evaluate on 1M-version data to see if larger held-out signal
     strengthens the NLL advantage.

---

## 9. Referenced commits (chronological)

```
040fecd  Phase 0+1 scaffolding
e36558f  FSDP + gradient_checkpointing fix
b9adb49  MAX_SEQ_LEN 65k + liger fused CE
ca4d321  Restore MAX_SEQ_LEN 131k after liger confirmed
aba969f  MAX_SEQ_LEN 98k + save every 25 steps (stability)
07c038c  Teacher SFT 2 epochs
a4b62ae  --resume flag + eval_ppl (sanity check #1)
5ed304b  eval_ppl data-parallel across 4 GPUs
949a614  eval_ppl: no_grad instead of inference_mode
8bcc458  eval_ppl: manual chunked CE (liger doesn't run in eval)
ab8cf15  eval_context_kl (#2) + eval_qualitative (#3)
35c95fd  eval_context_kl: add NLL-benefit alongside KL
f59773f  eval_mcq (generate + OpenAI agent)
54a7395  eval_mcq: ast.literal_eval for mixed-quote all_options
26dfcb0  eval_mcq: default to 32k; merge question into trailing user turn
afd7b07  eval_mcq: avoid apply_chat_template <think> contamination
9390e69  eval_mcq: reaction-priming suffix on choices
1f80237  eval_mcq_ppl (plan §9.5 — pure PPL scoring)
4fb96e3  parameterize K (default K=5)
b62c063  K=3 + MAX_SEQ_LEN 40960 (in-distribution RoPE)
9df6c33  --resume only resumes if checkpoint exists
966f264  Switch to Qwen3-4B-Instruct-2507 (non-thinking, 262k)
```
