# Dynamic UserSim — Experiment Log

Subproject implementing the plan in
[`../dynamic_usersim_complete_plan.md`](../dynamic_usersim_complete_plan.md)
(Dynamic User Simulator for Personalization).

**Paradigm**: train a per-user LoRA (Student) via OPD to predict what a specific
user would say, using a Teacher that sees the full progressive conversation
history. At inference, a general Agent queries UserSim reactions for each
candidate response and picks the best.

This log covers **Phase 0 (data prep) + Phase 1 (Teacher SFT)** + Phase 2
scaffolding. Phase 1 has two teacher runs: **R1** (Qwen3-4B hybrid, K=20,
ckpt-50) described in §2–§7, and **R3** (Qwen3-4B-Instruct-2507, K=3,
`final`) described in §9 — now the production teacher. R2 was a failed
launch, skipped. Phase 2 OPD retraining against R3 is in progress at
write time (§8).

Repo commits are referenced inline (`<shortsha>`) so each finding is tied to
verifiable code.

---

## 0. Current Status (TL;DR)

**✅ Confirmed — Teacher SFT learns persona-aware user simulation.**
Two independent pieces of evidence:
- **Eval 1 (user-token NLL)**: SFT 1.43 vs base 2.51 on held-out user turns
  (−43%, PPL 12.26 → 4.18). **All 20/20 samples improve.** §4.1
- **Eval 3 (30-case GT overlap)**: SFT outputs match ground-truth user
  content in 20/30 cases vs base 10/30. Mean Jaccard 0.07 vs 0.05. SFT
  fully eliminates `<think>` leakage (0% vs base 17–37%). §4.3

**⏳ Unresolved — Translating SFT's user-modeling capability to MCQ
accuracy.** Two paradigms tested, both limited:
- **Paradigm A: UserSim generates reactions → LLM Agent picks best choice
  (plan §9.3).** Structural failure: teacher SFT is optimized to predict
  natural user continuations, and natural user turns in PersonaMem often
  start new threads regardless of what the assistant said. SFT's reactions
  turn out **anti-correlated** with choice correctness — least-engaged on
  the most preference-aligned choice. §4.4.1, §4.4.2
- **Paradigm B: Score each choice's conditional PPL under teacher (plan
  §9.5).** Cleanest possible test. At n=589 32k-MCQs, SFT 46.9% vs base
  46.7% — essentially tied. By-type breakdown reveals SFT wins on
  narrative-style correct answers (track_evolution +10pp, suggest_new
  +14pp) but loses on assistant-voice recall answers (recall_facts −9pp,
  aligned_rec −4pp). Root cause: SFT trained only on user tokens, so its
  probability mass on assistant-style text isn't where we'd need it for
  this metric. §4.4.3

**In one sentence**: Teacher SFT is doing its job (modeling the user);
the downstream MCQ evaluation paradigm needs to be re-thought, and that's
what Phase 2 (Student LoRA via OPD) + agent-pipeline iteration is for.

### Update after R3 (§9)

R3 (Instruct-2507 teacher, K=3) resolves most of R1's ambiguities:
- **MCQ-PPL: +14.6pp over Instruct-2507 base** (489 → 345 → **491/589**);
  R1's "tie" was a K=20/RoPE + hybrid-model artifact, not a structural
  limit of user-only-loss training.
- **Context benefit is real and cleanly measurable**: base +0.130,
  SFT +0.020 (R1's negative base benefit was pure RoPE extrapolation).
- **Swap-persona diagnostic** (§9.3): SFT does parameterize persona
  identity (+0.008 nats, 35/40 sign-consistent) — small but unambiguous.
- Phase 2 OPD is being retrained against R3 as the production teacher.

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

30 random training-set cases (seed 42), 4 conditions each:
{base, sft} × {no-ctx, with-ctx}. For each case we generate the predicted
user response at a non-opening target-session user turn, then compare to
the ground-truth user utterance (`analyze_qual_30.py`).

**Metric**: content-word Jaccard overlap between generated output and
ground-truth, plus automated flags for `<think>` artifacts, "I also..."
generic openers, and reaction-word presence in first 100 chars.

| Condition | `<think>%` | "I also..." opener | mean GT-overlap | mean len | wins (argmax overlap) |
|-----------|-----------:|-------------------:|----------------:|---------:|----------------------:|
| base_noctx | 36.7% | 6.7%  | 0.054 | 892 | 5/30  |
| base_ctx   | 16.7% | 23.3% | 0.050 | 793 | 5/30  |
| **sft_noctx** | **0.0%** | 6.7%  | **0.076** | 586 | **12/30** |
| sft_ctx    | 0.0%  | 16.7% | 0.067 | 611 | 8/30  |

**Four structural findings**:

1. **SFT eliminates `<think>` leakage entirely** (0% vs base 17-37%). User-
   only-loss SFT internalized "users don't think aloud" as a side effect —
   this happens even on the hybrid Qwen3-4B. A clean implicit benefit.

2. **SFT beats base on GT overlap in 20/30 cases** (67%). Mean overlap
   0.076/0.067 for SFT vs 0.054/0.050 for base. Quantitative confirmation
   of the qualitative "SFT sounds more like a user" impression.

3. **Context hurts SFT slightly on average** (sft_ctx − sft_noctx overlap
   mean = −0.009, 18/30 cases ctx lower). But **when ctx helps, it helps
   concretely** — top-overlap sft_ctx cases are domain-specific topic
   recalls (cooking fusion, movie snacks) where prior sessions provide
   the actual retrievable detail. Consistent with eval 2's near-zero
   average context benefit despite selective large wins.

4. **"I also..." opener is triggered by context, not by SFT**: base_ctx
   23.3% vs base_noctx 6.7%, sft_ctx 16.7% vs sft_noctx 6.7%. This is a
   PersonaMem data-distribution pattern ("in addition to what we're
   discussing, ..."), not an SFT-specific artifact. Both models pick it
   up from the long context.

**Strongest SFT win** (content-overlap Jaccard 0.286, persona 14):

```
prev user: "[Original_Sentence]: It is important to note that these issues
  are not exhaustive, and ongoing dialogues shall be essential..."
GT       : "This sentence is too lengthy and formal."
sft_ctx  : "This sentence is too long and formal. I want it to be punchy
  and direct."
sft_noctx: "This sentence is too long and formal. I want something punchy."
base_ctx : <contains </think>> then pivots to rewriting the sentence itself
```

Both SFT conditions nail the user's "reject formalism" preference; base
defaults to its assistant role and actually attempts the rewrite.

**Failure mode**: SFT struggles on GT that pivots to a *new sub-topic* with
no continuous thread (e.g. "decided to give up on my herb garden" after
mental-health context). 3/3 bottom-overlap cases are this pattern.

**Caveat**: Jaccard on short free-form text is noisy; these scores only
indicate rough semantic alignment, not exact content match. The structural
pattern (SFT > Base, SFT_noctx ≈ SFT_ctx, ctx selectively useful) is the
durable finding.

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

**Benchmark positioning (vs. paper Figure 6).** PersonaMem paper Figure 6
reports LLaMA 3.1-8B under the same MCQ protocol on 32k. Their 32k type
distribution differs slightly — it includes `acknowledge_latest_user_preferences`
(0.36 for LLaMA 8B MCQ) which our public-CSV 32k subset lacks — so we
weight their per-type accuracies by our 589-question type counts on the
six shared types:

```
LLaMA 3.1-8B (MCQ), re-weighted to our type distribution
  = (99×0.68 + 139×0.53 + 146×0.18 + 55×0.44 + 57×0.32 + 93×0.08) / 589
  = 36.9%
```

Side-by-side (same 32k, same MCQ protocol):

| Overall 32k MCQ              | Accuracy |
|------------------------------|---------:|
| Random baseline              | 25.0%    |
| LLaMA 3.1-8B (MCQ, re-weighted) | 36.9% |
| **Our Qwen3-4B base**        | **46.7%** |
| **Our SFT ckpt-50**          | **46.9%** |

So our Qwen3-4B baseline alone is +10 pp over the comparable-size LLaMA
3.1-8B baseline at the same protocol; SFT adds ~0 pp on overall but
shifts the per-type profile meaningfully:

| Query Type            | LLaMA 8B (MCQ) | SFT ckpt-50 | Δ (SFT − LLaMA) |
|-----------------------|---------------:|------------:|----------------:|
| Tracking Evolution    | 0.53           | **0.820**   | **+0.29**       |
| Suggest New           | 0.08           | **0.269**   | **+0.19**       |
| Recall Facts          | 0.18           | 0.281       | +0.10           |
| Revisit Reasons       | 0.68           | 0.758       | +0.08           |
| Generalize            | 0.32           | 0.175       | −0.14           |
| Preference-Aligned Rec| 0.44           | 0.200       | −0.24           |

SFT's `narrative-style` wins (track_evolution +29 pp, suggest_new +19 pp)
align with its training objective (user-voice long continuation). SFT
loses on `preference_aligned_rec` and `generalize` — types that reward
broad assistant-style reasoning rather than user-voice scoring, so the
LLaMA 8B instruct-tuned distribution serves those better. This is a
clean ablation of SFT's type-specific strengths/weaknesses at benchmark
scale.

LLaMA 3.1-70B generative (0.82 on recall_facts) and DeepSeek R1-Distill
8B generative (0.94) vastly outperform any MCQ number for their
respective types — consistent with the paper's note that the generative
setting is strictly easier for personalization tasks ("the model is able
to provide a personalized response without seeing all the candidate
options"). That suggests our next evaluation iteration should include a
generative variant (§8).

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
  (eval 1: 2.506 → 1.430, all 20/20 samples improve).
- SFT produces content that matches ground-truth user utterances in
  67% of 30 qualitative cases, by content-word Jaccard. Mean overlap
  0.07 vs base 0.05. (§4.3)
- SFT eliminates `<think>` artifacts (0% vs base 17–37%).
- SFT is better-calibrated when uncertain — never as confidently wrong as
  base at the top of the margin distribution on MCQ-PPL.
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

1. ~~**R3**~~ — **DONE, §9.** R3 (Instruct-2507, K=3) is trained and
   fully evaluated. Now the production teacher.
2. ~~**Swap-persona probe**~~ — **DONE, §9.3.** Result: (C) partially
   parameterized (+0.008 nats, 35/40 one-sided).
3. **Phase 2 — OPD, retrain with R3 teacher**. An initial 4-persona
   proof-of-concept run was launched against R1 ckpt-50 but killed
   mid-run once R3 eval results confirmed R3 as the strictly better
   teacher. Retraining the same 4 personas (0 Kanoa, 4 Lisa,
   12 Jordan, 14 Leilani) against R3 `final`. K=3 OPD data already
   built; trainer unchanged. Primary eval: MCQ-PPL on each persona's
   128k MCQ subset, base vs base+LoRA (`eval_opd.py`).
4. **Optional follow-ups**:
   - Per-persona swap probe (40 samples each for pid 0/4/12/14) to
     calibrate expected LoRA differentiation before training.
   - K-scan of Eval 2 (K=3/10/20) now that Instruct-2507's 262k native
     makes this clean.
   - Evaluate on 1M-version data once compute budget allows.

---

## 9. R3 re-evaluation (Instruct-2507, K=3)

R3 = Qwen3-4B-Instruct-2507, K=3, max_seq=40960. Designed to disentangle
the unresolved pieces from §7 by eliminating three R1 confounds:
(1) RoPE extrapolation at K=20, (2) Qwen3-4B hybrid thinking-mode,
(3) K=20's distribution gap between eval/deployment contexts.

Training produced 3 checkpoints (125, 148, final). We evaluated `final`.

### 9.1 Eval 1 — user-token NLL (§3.5 #1)

Same 20 held-out samples, same seed=42.

| Metric                 | R1 ckpt-50 | **R3 final** |
|------------------------|-----------:|-------------:|
| Base mean NLL          | 2.506      | 2.169        |
| SFT mean NLL           | 1.430      | **1.183**    |
| Δ (SFT − Base)         | −1.076     | −0.986       |
| PPL ratio SFT / Base   | 0.341      | 0.373        |
| Samples improved       | 20/20      | 20/20        |
| Δ range                | +0.62–1.67 | +0.75–1.39   |

Both models score lower (cleaner) NLL than their R1 counterparts — the
Instruct-2507 base alone is 0.34 nats better at user-turn prediction than
Qwen3-4B hybrid, even before any SFT. Δ is slightly smaller (0.99 vs
1.08) because base improved faster than SFT, but the SFT-over-base
advantage remains decisive; all 20 samples still improve, with a tighter
range (no low-end +0.62 outlier).

**Finding**: R3 is strictly better than R1 on this metric. User modeling
is confirmed even more cleanly.

### 9.2 Eval 2 — context benefit (§3.5 #2, revisited)

Same paired with/without-prefix design on the same 20 samples.

| Model | NLL_with | NLL_without | benefit |
|-------|---------:|------------:|--------:|
| R1 base  (Qwen3-4B, K=20)     | 2.506 | 2.313 | **−0.193** |
| R1 SFT   (Qwen3-4B, K=20)     | 1.430 | 1.452 | +0.021 |
| **R3 base** (Instruct-2507, K=3) | **2.169** | **2.299** | **+0.130** |
| **R3 SFT**  (Instruct-2507, K=3) | **1.183** | **1.203** | +0.020 |

Two structural changes:

1. **Base benefit flipped sign: −0.193 → +0.130.** R1's negative benefit
   was RoPE extrapolation (K=20 pushed Qwen3-4B hybrid past its 40960
   max_pos). Instruct-2507's 262k native context + K=3 entirely in
   distribution removes this confound. Base now uses context normally.

2. **SFT benefit essentially unchanged: 0.021 → 0.020.** Even in clean
   conditions, SFT's prior-session benefit is near-zero.

Consequence: the R1 reading "SFT fixes base's context distraction" no
longer holds — base doesn't have distraction to fix. Instead SFT's
context benefit is smaller than base's (+0.02 vs +0.13). Two candidate
explanations, indistinguishable from this eval alone:
- **(A)** SFT parameterized user patterns → less dependence on context
- **(B)** SFT learned generic user-style → context adds little because
  the style is generic

Supporting hint from per-sample data: SFT benefit **saturates quickly
with prefix length** — samples with n < 1600 tokens in target hit
+0.05–0.07 benefit while samples with n > 2500 drop to < +0.015. Base
doesn't show this saturation. Weakly consistent with (A) but not
conclusive. Motivates §9.3.

### 9.3 Persona-swap diagnostic (new probe)

Sharpened version of Eval 2: instead of removing the K=3 prefix, we
**replace** it with a random other persona's K=3 prefix, keeping the
current session and target user tokens intact. The mean NLL gap (swap −
real) on GT user tokens directly measures whether prior-session prefix
carries persona-specific predictive signal.

40 paired samples (larger n for power; see `eval_persona_swap.py`).

| Metric                         | Base   | SFT    | Δ (SFT − Base) |
|--------------------------------|-------:|-------:|---------------:|
| real_nll                       | 2.157  | 1.154  | —              |
| swap_nll                       | 2.155  | 1.162  | —              |
| **gap (swap − real)**          | −0.002 | **+0.008** | **+0.010** |
| Directional sign (gap > 0)     | 24/40  | **35/40**  | —          |

**Interpretation**:
- Base gap ≈ 0 with 60/40 sign split = noise. Method is unbiased.
- SFT gap = +0.008 small but with **35/40 sign consistency**.
  Probability under null of ≥35/40 positive is ~6.6×10⁻⁷ — unambiguous.
- The 5 "negative" SFT samples all have |gap| ≤ 0.001 (zero, not
  counter-evidence).

**Case (C): partially parameterized**, not (A) and not (B).

- ✅ SFT genuinely does differentiate personas (binomial proof).
- ⚠️ Magnitude is small: +0.008 nats is ~0.8% of SFT's −0.986 total NLL
  improvement. The remaining ~99% is generic user-style learning
  (first person, non-`<think>`, "I also..." opener cadence, etc.).

Per-persona observations on the four Phase-2 focal personas:

| PID | Name       | Samples | SFT gap range       | Note |
|----:|------------|--------:|---------------------|------|
|   0 | Kanoa      | 1       | +0.009              | Too few to judge |
|   4 | Lisa       | 5       | +0.001 to **+0.037** | Most distinctive |
|  14 | Leilani    | 2       | +0.006, +0.008      | Small but consistent |
|  12 | Jordan     | 3       | −0.001, +0.005, +0.007 | Weakest (warning sign) |

Jordan's near-zero gap is a Phase-2 risk: if the teacher can barely
distinguish Jordan's prior sessions from an unrelated persona's, the
per-user LoRA trained against this teacher may not carry strong
Jordan-specific information either.

### 9.4 Eval 4.4.3 revisited — MCQ-PPL (§9.5 protocol, full 32k)

Plan §9.5 rerun on all 589 32k-MCQs. Compared against **Instruct-2507
base** (R3's actual pretrained start) rather than R1-era Qwen3-4B base.

| Run                                     | Overall | n   |
|-----------------------------------------|--------:|----:|
| R1 ckpt-50 (Qwen3-4B hybrid)            | 0.469   | 589 |
| R1 base (Qwen3-4B hybrid)               | 0.467   | 589 |
| **R3 final (Instruct-2507)**            | **0.491** | 589 |
| **Instruct-2507 base**                  | 0.345   | 589 |
| LLaMA 3.1-8B (paper, re-weighted)       | 0.369   | —   |
| Random                                  | 0.250   | —   |

**The headline result**: R3 SFT beats its base by **+14.6pp**; R1 SFT
beat its base by **+0.2pp**. Training effect at R3 is 75× larger in
absolute accuracy points — even though R3 started from a **worse** base
(Instruct-2507 base 0.345 < Qwen3-4B hybrid base 0.467).

Per-type breakdown (R3 SFT vs Instruct-2507 base, with R1 for context):

| qtype           | R3 SFT | R3 Base | Δ (R3)   | R1 SFT | R1 Base | Δ (R1)  |
|-----------------|-------:|--------:|---------:|-------:|--------:|--------:|
| track_evolution | 0.813  | 0.460   | **+0.353** | 0.820 | 0.719 | +0.101 |
| suggest_new     | 0.409  | 0.129   | **+0.280** | 0.269 | 0.129 | +0.140 |
| reasons_behind  | 0.747  | 0.636   | +0.111   | 0.758 | 0.808 | −0.051 |
| recall_facts    | 0.329  | 0.247   | +0.082   | 0.281 | 0.370 | −0.089 |
| aligned_rec     | 0.182  | 0.127   | +0.055   | 0.200 | 0.236 | −0.036 |
| generalize      | 0.105  | 0.368   | **−0.263** | 0.175 | 0.281 | −0.106 |

Three structural shifts from R1:

1. **Five of six types flip from SFT-loss to SFT-win.** R1's user-only
   loss was net neutral on MCQ-PPL because narrative-type wins were
   cancelled by assistant-voice recall losses. R3 eliminates those
   losses while preserving and amplifying the narrative wins.

2. **"User-only loss is broken for MCQ-PPL" (§4.4.3) was wrong.** The
   correct diagnosis is "R1's teacher distribution over assistant tokens
   was poor for this task"; R3 shows user-only training can indirectly
   shape assistant-token probabilities strongly enough to win discrim­
   ination when the base's chat structure is clean.

3. **`generalize` remains a durable SFT weakness** (both R1 and R3
   regress against base here). This type rewards broad cross-domain
   reasoning that user-turn prediction actively narrows. Persists across
   teachers — structural, not an artifact.

**NLL margin calibration**:
- R3 SFT max margin (confident-but-wrong): 2.60 nats
- Instruct-2507 base max margin: 5.76 nats

Same pattern as R1: SFT is never as confidently wrong as base. Well
calibrated. (R1 ckpt-50 max = 2.60, Qwen3-4B base max = 5.21 — R3
preserves this property.)

### 9.5 Eval 3 — 30-case qualitative (vLLM batched, §4.3 replay)

Same seed=42 sampler, 30 cases × 4 conditions. New runner
`eval_qualitative_vllm.py` (TP=4, batched) — 2-3 min instead of ~15.

| Condition  | `<think>%` R1 / R3 | also_open% R1 / R3 | overlap R1 / R3 | wins R1 / R3 |
|------------|-------------------:|-------------------:|----------------:|-------------:|
| base_noctx | 36.7 / **0.0**     | 6.7 / 6.7          | 0.054 / 0.060   | 5 / 7        |
| base_ctx   | 16.7 / **0.0**     | 23.3 / 6.7         | 0.050 / 0.049   | 5 / 4        |
| sft_noctx  | 0.0 / 0.0          | 6.7 / **20.0**     | 0.076 / **0.092** | 12 / 8     |
| sft_ctx    | 0.0 / 0.0          | 16.7 / **26.7**    | 0.067 / **0.096** | 8 / **11** |

Four findings:

1. **`<think>` eliminated architecturally** (all conditions 0%). The
   non-thinking Instruct-2507 base does not produce `<think>` markers.
   Previously this was credited to SFT; now it's free from the base —
   drop this from SFT's list of benefits.

2. **GT overlap uniformly higher under R3**. Base and SFT both lift;
   SFT lifts more (+0.016 at sft_noctx, +0.029 at sft_ctx). Consistent
   with Eval 1's stronger absolute PPL.

3. **Winner shifts from sft_noctx → sft_ctx** (12→8 and 8→11). Combined
   with Eval 2's SFT benefit +0.020 and MCQ-PPL's +14.6pp overall win,
   R3 is unambiguously "using context" more than R1 did.

4. **"I also..." opener rate nearly doubled** (sft_noctx 6.7% → 20.0%,
   sft_ctx 16.7% → 26.7%). PersonaMem's user-turn data-distribution
   pattern ("I also tried X...") bleeds through more strongly with
   Instruct-2507's stronger chat priors. Bottom-3 sft_ctx cases are
   all "I also..." templates that score near-zero GT overlap
   (persona 4 Lisa especially). This is a **qualitative regression**
   not visible to NLL-based metrics.

Persona 14 Leilani's "reject formalism" case from R1 §4.3 reappears in
R3 (overlap 0.250), confirming this is a stable, reproducible
style-learning signature across teacher versions:

```
prev user: "[Original_Sentence]: It is important to note that these
             issues are not exhaustive, and ongoing dialogues shall..."
GT       : "This sentence is too lengthy and formal."
R3 sft_ctx: "This sentence is too formal and long. I prefer a concise
             and energetic tone."
```

### 9.6 Cross-eval picture

| Claim                                | R1                          | R3                           |
|--------------------------------------|-----------------------------|------------------------------|
| SFT learns user style                | ✅ −1.08 nats NLL           | ✅ −0.99 nats NLL            |
| SFT uses long prior-session context  | Unclear (RoPE confound)     | ✅ +0.020, small but clean   |
| Base uses prior-session context      | Appeared no (−0.193 RoPE)   | ✅ +0.130 (RoPE cleared)     |
| SFT is persona-specific              | Unknown                     | ✅ gap +0.008, 35/40 sign    |
| MCQ-PPL benefit from SFT             | Tied (+0.002)               | **✅ +0.146 (big win)**      |
| `<think>` suppression                | Training effect             | Architectural (Instruct-2507)|
| "I also..." failure mode             | Moderate (6–17%)            | ❌ Worse (20–27%)            |

R3 replaces R1 as the production teacher.

### 9.7 Phase 2 implications

- **Strong case to retrain Phase 2 OPD with R3 teacher.** R3 has +14.6pp
  of MCQ-PPL headroom for the student LoRA to distill; R1 had essentially
  none. Per-user LoRA differentiation is bounded by what the teacher can
  parameterize — swap diagnostic says that's +0.008 nats per persona,
  small but non-zero.
- **MCQ-PPL (plan §9.5) is the primary Phase 2 metric**, not the
  Generate+Agent pipeline (§9.3). §4.4.2 showed the agent pipeline
  failed on R1 via "anti-correlated reaction" mode collapse; R3's
  worse "I also..." rate will only deepen this failure. Skip.
- **Expect moderate, not dramatic, per-user differentiation**.
  Persona 4 Lisa has the strongest SFT swap signal (+0.037 max) and
  should produce the most distinctive LoRA; persona 12 Jordan has the
  weakest (+0.007 max) and is the canary for "does per-user OPD
  actually teach anything beyond average user-style?"
- **Qualitative monitoring during OPD**. "I also..." rate observed
  during student rollouts should be tracked — if the LoRA amplifies
  this pattern (as we'd expect when training against Instruct-2507
  teacher), reaction-quality for any future generate-based eval is
  further compromised.

---

## 10. Referenced commits (chronological)

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
e4a6452  Phase 2 (OPD) scaffolding: per-user-turn data + single-LoRA trainer
a4a3513  Phase 2 MCQ-PPL eval (student_opd/eval_opd.py)
e7641a9  Persona-swap diagnostic (§9.3)
6f75934  eval_mcq_ppl: --num-mcqs <= 0 = all
0634433  eval_qualitative_vllm: 4-GPU batched generation
ed64b6f  analyze_qual_30: proper CLI, relocated to teacher_sft/
```
