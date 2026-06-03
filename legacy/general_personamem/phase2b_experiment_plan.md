# Phase 2b Experiment Plan: Dual LoRA + Gated KL

Builds on Phase 2 (single LoRA OPD, §10 of EXPERIMENTS.md).
All design decisions from the discussion session on 2026-04-16.

---

## 1. Summary of changes from Phase 2

| Dimension | Phase 2 (current) | Phase 2b (this plan) |
|---|---|---|
| LoRA structure | Single LoRA, rank 32, all modules | Dual LoRA: MLP slow + Attention fast |
| Student input | demographics + chatbot_prev only | demographics + last 2-3 turns + chatbot_prev |
| KL loss | Standard reverse KL on all tokens | Gated reverse KL (skip where student more confident) |
| Teacher OPD context | K=3 (same as teacher SFT) | K=10 (larger window, no retraining needed) |

Gated KL subsumes surprise detection — no separate session-level
gating mechanism needed (see §5.4). These four changes are introduced
in two rounds to enable clean ablation.

---

## 2. Experimental rounds

### Round 1: Dual LoRA only

Changes: dual LoRA structure + student 2-turn context.
KL loss: ungated reverse KL. Teacher context: K=3 (unchanged).
Goal: isolate the effect of dual LoRA on stability.

**Outcome (R1)**: dual LoRA avg best-step closure crashed to 26%
(vs P2 single-LoRA 76%). Two confounded culprits — (1) 2-turn context
poisonous for some personas (Lisa base_recent2 −2.7pp), (2) slow_lr
1e-5 too low (MLP barely trains). **Round 1b** ablated both at once
(--student-ctx demo + --slow-lr 5e-5) and recovered to 78% best-step
closure (Jordan exceeded teacher_k3 at +109% on n=113).

### Round 2 (split into 2a then 2b for clean ablation)

**Round 2a**: dual LoRA + **entropy-gated reverse KL** + Teacher K=3.
Inherits R1b best settings (slow_lr 5e-5, student_ctx demo).
Outcome: **failed its core hypothesis**. Gate dropped to ~7% by step 30
(Instruct-2507 overconfidence dominates entropy comparison); Leilani
`acknowledge_latest` accelerated decay (0.20 → 0.05 by step 200, worse
than R1b's 0.075 at final). track_evolution and recall_facts also
regressed across all 4 personas. See chat log + qtype analysis.

**Round 2c** (after 2a failure): replace entropy gate with **joint gate**:
gate = (H_teacher < tau) AND (argmax_teacher != argmax_student). Tests
whether requiring "teacher confident AND actual disagreement" recovers
R2a's lost gains. Same training infrastructure as R2a (dual LoRA,
slow_lr 5e-5, student_ctx demo, K=3).

**Round 2b** (after 2c): + Teacher K=10. Goal: raise distance>3 ceiling.
Decoupled from gating choice so K bump is cleanly attributed.

Gated KL provides automatic selective update at all granularities
(token, sample, session) — see §5.4 — so no Round 3 surprise-detection
mechanism is needed.

---

## 3. Dual LoRA architecture

### 3.1 Design rationale

MLP layers store factual knowledge (Knowledge Neurons, ROME/MEMIT
literature). Attention layers do contextual routing. In user modeling:

- MLP → "Laura likes Mediterranean food, has gluten allergy"
  (factual preferences, change slowly, need protection)
- Attention → "when asked about food, attend to dietary info"
  (routing patterns, adapt quickly per session/topic)

Placing two LoRAs on **non-overlapping modules** eliminates gradient
interference, consolidation complexity, and rank mismatch issues.

### 3.2 Configuration

```yaml
LoRA_slow:
  adapter_name: "slow"
  target_modules: [gate_proj, up_proj]     # MLP only
  rank: 32
  alpha: 64
  dropout: 0.0
  lr: 1e-5                                 # 20x lower than fast

LoRA_fast:
  adapter_name: "fast"
  target_modules: [q_proj, k_proj, v_proj, o_proj]  # Attention only
  rank: 16
  alpha: 32
  dropout: 0.0
  lr: 2e-4                                 # same as Phase 2

Inference: base + LoRA_slow + LoRA_fast (PEFT additive)
Training:  both receive gradient every step, two separate optimizers
```

Parameter count (Qwen3-4B, hidden=3584, intermediate=18944, 36 layers):

```
LoRA_slow (gate+up):  ~52M  (large matrices, high capacity for facts)
LoRA_fast (q,k,v,o):  ~16.5M (smaller, compact routing adjustments)
Combined:             ~68M  (~1.7% of 4B)
Phase 2 single LoRA:  ~52M  (for comparison)
```

### 3.3 Implementation

```python
from peft import LoraConfig, get_peft_model

# LoRA_slow on MLP
config_slow = LoraConfig(
    r=32, lora_alpha=64,
    target_modules=["gate_proj", "up_proj"],
    lora_dropout=0.0,
)

# LoRA_fast on Attention
config_fast = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.0,
)

model = get_peft_model(base_model, config_slow, adapter_name="slow")
model.add_adapter("fast", config_fast)
model.set_adapter(["slow", "fast"])

# Two optimizers
optimizer_slow = AdamW(
    [p for n, p in model.named_parameters() if "slow" in n and p.requires_grad],
    lr=1e-5
)
optimizer_fast = AdamW(
    [p for n, p in model.named_parameters() if "fast" in n and p.requires_grad],
    lr=2e-4
)
```

No consolidation mechanism. No freeze/unfreeze. No alternating updates.
LR difference is the entire mechanism.

---

## 4. Student input: 2-turn intra-session context

### 4.1 Rationale

Current Phase 2 gives student only `[demographics] + [chatbot_prev]`.
Student doesn't know what topic the conversation is about, making many
rollouts generic/off-topic even for a human.

Fix: add last 2-3 turns (4-6 messages) of the current session.
This tells student "we're talking about Barcelona travel" without
revealing long-term preference history (that's LoRA's job).

### 4.2 Student input format (Phase 2b)

```
[Demographics] Laura, 32-year-old female Pacific Islander, AI startup
founder, passionate about leveraging cultural nuances...

[Recent conversation]
User: I'm thinking about taking a trip to Barcelona next month.
Chatbot: That sounds exciting! Barcelona has so much to offer.
User: Yeah, I've been looking at flights. Any suggestions for
      neighborhoods to stay in?
Chatbot: That sounds like a great plan! Would you like some
         specific suggestions?

[User says]
```

### 4.3 Data construction change

```python
def build_student_input(sample, n_context_turns=2):
    """
    Include last n_context_turns user-chatbot exchanges from current
    session before chatbot_prev.
    """
    current_session_turns = sample["current_session_turns"]
    target_turn_idx = sample["target_turn_idx"]

    # Collect up to n_context_turns exchanges before the target
    context_start = max(0, target_turn_idx - n_context_turns * 2)
    recent_turns = current_session_turns[context_start:target_turn_idx]

    # chatbot_prev is the last message in recent_turns (or standalone)
    return format_prompt(
        demographics=sample["demographics"],
        recent_turns=recent_turns,  # includes chatbot_prev at the end
    )
```

Teacher input unchanged — still gets full K-session progressive context.

### 4.4 MCQ eval adaptation

At eval time, extract last 2-3 turns before the MCQ's `end_index`
from the shared_context. This matches the training format.

```python
def build_mcq_student_input(mcq, shared_context, n_context_turns=2):
    messages = shared_context["messages"][:mcq["end_index"]]
    # Take last few turns as context
    recent = messages[-(n_context_turns * 2 + 1):]  # +1 for the user query
    return format_prompt(
        demographics=extract_demographics(shared_context),
        recent_turns=recent,
    )
```

---

## 5. Gated KL loss (Round 2)

### 5.1 Rationale

Pure KL loss means student is always pulled toward teacher's
distribution. When teacher is uninformed (current K=3 window doesn't
contain relevant info), this actively erases student's accumulated
knowledge from earlier windows.

Gated KL: only distill on tokens where teacher is more confident
(lower entropy) than student. When student already knows more than
teacher on a token, skip the KL — protect accumulated knowledge.

### 5.2 Implementation

**KL direction note**: Phase 2 / R1 / R1b / R2 all use **reverse KL**
= `KL(student || teacher)` (the modern RLHF / Thinking Machines on-policy
distillation convention; classical Hinton-2015 KD literature confusingly
labels the same direction "forward KL"). See
https://thinkingmachines.ai/blog/on-policy-distillation/ for rationale
(mode-seeking, unhackable reward, exposure-bias reduction). DO NOT
`F.kl_div(student_logprobs, teacher_probs)` — that computes the OPPOSITE
direction (`KL(teacher || student)`), incompatible with Phase 2 / R1.

```python
def gated_reverse_kl(s_lp, t_lp, mode="entropy", margin=0.0, tau=1.0):
    """
    s_lp, t_lp: log-softmax student/teacher logits, shape [T, V].
    Per-token reverse KL = sum_v P_s(v) * (log P_s(v) - log P_t(v)),
    masked by gate. Returns (loss, gate_ratio).

    Two gate modes:
      "entropy" (R2a): gate = (H_t + margin < H_s). Independent comparison
        of two scalars; assumes "more confident = more knowledge to teach".
      "joint" (R2c): gate = (H_t < tau) AND (argmax_t != argmax_s).
        Joint measure of (a) teacher confidence absolute level AND
        (b) actual disagreement at top-1. Addresses two R2a failure modes:
        Instruct-base overconfidence drowning out the entropy signal, and
        gating opening on tokens where student & teacher already agree.
    """
    kl_per_token = (s_lp.exp() * (s_lp - t_lp)).sum(dim=-1)   # [T]
    H_t = -(t_lp.exp() * t_lp).sum(dim=-1)                    # [T]

    if mode == "entropy":
        H_s = -(s_lp.exp() * s_lp).sum(dim=-1)
        gate = (H_t + margin < H_s).float()
    elif mode == "joint":
        top_t = t_lp.argmax(dim=-1)
        top_s = s_lp.argmax(dim=-1)
        gate = ((H_t < tau) & (top_t != top_s)).float()
    else:
        raise ValueError(f"unknown gate mode: {mode}")

    gate_sum = gate.sum()
    if gate_sum.item() == 0.0:
        return None, 0.0

    loss = (gate * kl_per_token).sum() / gate_sum
    gate_ratio = (gate_sum / gate.numel()).item()
    return loss, gate_ratio
```

**R2a (entropy) outcome**: gate ratio dropped to ~0.07 by step 30
(Instruct-2507 base is naturally peaked, so H_s < H_t for ~85% of tokens
even at init — the gate gets drowned out). `acknowledge_latest` for
Leilani went from 0.20 (R1b @200) to 0.05 (R2a @200) — gate accelerated
the very failure it was supposed to prevent, because gradient concentration
on the few open tokens amplified teacher's wrong direction.

**R2c (joint) hypothesis**: requiring both teacher confidence (H_t < 1.0)
AND top-1 disagreement should:
- recover the track_evolution / recall_facts gains R2a lost (where teacher
  is genuinely confident-and-correct AND student disagrees)
- avoid wasting gradient on tokens where student already matches teacher
- not exacerbate Instruct-base overconfidence (no H_s comparison)

Joint gate still cannot distinguish "teacher confidently correct" from
"teacher confidently wrong" — that's a fundamental no-ground-truth limit.
Leilani's `acknowledge_latest` may still degrade under R2c.

### 5.3 Monitoring

Log `gate_ratio` per step. Expected trajectory:

```
Early training:   gate_ratio ≈ 0.9  (teacher confident almost everywhere)
Mid training:     gate_ratio ≈ 0.6  (student catching up on many tokens)
Late training:    gate_ratio ≈ 0.3  (student surpasses teacher on most tokens)
```

Monotonically decreasing gate_ratio = student is accumulating knowledge.
Flat gate_ratio = student stopped learning.
Increasing gate_ratio = student is forgetting (red flag).

### 5.4 Unification: gated KL subsumes surprise detection

Gated KL and surprise detection are the same principle at different
granularities. Both ask: "does this signal contain information the
student doesn't already have?"

```
Gated KL (token level):
  "Is teacher more confident than student on this token?"
  Metric: H(teacher) vs H(student)

Surprise detection (session level):
  "Is user behavior surprising to the student in this session?"
  Metric: NLL(actual_response | student)
```

The connection: gate_ratio aggregated to session level IS the surprise
signal.

```python
# Per-session gate_ratio = continuous surprise score
session_gate_ratios = []
for sample in session_samples:
    _, gate_ratio = gated_kl_loss(student_logits, teacher_logits)
    session_gate_ratios.append(gate_ratio)

session_surprise = mean(session_gate_ratios)
# High gate_ratio → teacher knows more → session has new info → "surprising"
# Low gate_ratio  → student already knows → session is redundant → "not surprising"
```

Gated KL handles all granularities automatically:
- **Low-info session**: most tokens gated out → effective loss ≈ 0
  (equivalent to surprise detection's SKIP)
- **High-info session**: most tokens pass gate → full update
  (equivalent to surprise detection's FULL_UPDATE)
- **Mixed session**: some tokens pass, some don't
  (equivalent to surprise detection's TARGETED_UPDATE, but more granular)

No separate surprise detection mechanism needed. No threshold tuning.
No SKIP/TARGETED/FULL decision tree. One mechanism, zero hyperparameters
beyond the entropy comparison itself.

**Monitoring**: log per-session mean gate_ratio over time. This curve
replaces the "surprise curve" from the original plan (§6 of
dynamic_usersim_complete_plan.md) with a strictly more informative
signal that also captures the teacher-student relative relationship.

---

## 6. Teacher K=10 at OPD time (Round 2)

### 6.1 Rationale

R3 teacher was SFT-trained at K=3, but the base model (Instruct-2507)
has 262k native context. Increasing teacher's OPD-time context to K=10
gives it access to ~50k tokens of history, well within native range.

No teacher retraining needed — SFT didn't destroy base's long-context
ICL ability (R3 eval 2 shows base context benefit +0.130 is preserved).

### 6.2 Implementation

Only change `build_opd_data.py`'s K parameter from 3 to 10:

```python
# In build_opd_data.py
K_TEACHER_OPD = 10  # was 3

# Teacher sees sessions[max(0, t-K):t] + current session up to target turn
# Student still sees only demographics + 2-turn context (no change)
```

Teacher forward pass will be slower (~50k tokens vs ~23k), roughly 2x.
Still feasible on GH200.

### 6.3 Expected effect

- MCQs with distance_to_ref 4-10: teacher now sees relevant info →
  richer distillation signal → student should improve on these
- MCQs with distance_to_ref > 10: teacher still can't see → but
  gated KL protects student's accumulated knowledge from earlier windows
- Overall: combination of K=10 + gated KL should enable student to
  **exceed teacher_k3** on distance > 3 MCQs

---

## 7. Evaluation plan

### 7.1 Same 4 focal personas

| PID | Name | Phase 2 best closure | Key characteristic |
|---|---|---|---|
| 0 | Kanoa | 50% (slow monotonic) | Mixed identity, hard to parameterize |
| 4 | Lisa | 84% (clean monotonic) | Most distinctive, all metrics agree |
| 12 | Jordan | 92% peak, unstable | Strong signal but oscillates |
| 14 | Leilani | 61% peak, then decay | Style-preference conflict |

### 7.2 Checkpoints

Save every 200 steps (same as Phase 2). Evaluate at steps
{200, 400, 600, 800, final} for learning curve comparison.

### 7.3 Metrics (same three as Phase 2 §10)

1. **MCQ-PPL** (primary): per-persona 128k MCQs, argmin mean NLL
2. **Logit NLL on GT**: 100 samples/persona, 5 conditions
3. **LLM judge on generations**: 50 samples/persona, gpt-4o-mini

Plus new monitoring:
4. **gate_ratio curve** (Round 2 only): logged per step

### 7.4 Conditions for MCQ-PPL

```
base_demo:       Instruct-2507, demographics + 2-turn context
teacher_k3:      R3, K=3 context (Phase 2 teacher ceiling)
teacher_k10:     R3, K=10 context (Round 2 only, new ceiling)
student_dual_R1: base + LoRA_slow + LoRA_fast, demographics + 2-turn
student_dual_R2: same but trained with gated KL + teacher K=10
student_single:  Phase 2 results (for comparison)
```

### 7.5 New analysis: distance_to_ref stratification

For ALL conditions, break down MCQ-PPL by `distance_to_ref_in_blocks`:

```
Bucket 1: distance 1-3   (within teacher K=3 window)
Bucket 2: distance 4-7   (within K=10 but outside K=3)
Bucket 3: distance 8+    (outside both windows)
```

Expected pattern:

```
                  Bucket 1    Bucket 2    Bucket 3
teacher_k3:       strong      weak        very weak
teacher_k10:      strong      strong      weak
student (R2):     moderate    moderate    moderate  (LoRA accumulation)
```

Student exceeding teacher_k3 on Bucket 2-3 = cross-session advantage
confirmed (Dimension 2 of our three-dimension framework).

### 7.6 MCQ type breakdown

Map results to three-dimension framework:

```
Dimension 1 (Continual Memory, vs SFT):
  Primary: Type 3 (latest prefs), Type 1 (recall facts)
  Metric:  retention across training steps

Dimension 2 (Cross-Session, vs Teacher):
  Primary: Type 4 (track evolution), Type 1 @ distance>3
  Metric:  distance_to_ref stratification

Dimension 3 (Generalization, vs Token Memory):
  Primary: Type 7 (generalize), Type 2 (suggest new), Type 6 (aligned recs)
  Metric:  cross-topic held-out (future, not in this round)
```

---

## 8. Ablation table (target)

```
| Method                           | Lisa | Kanoa | Jordan | Leilani | Avg closure | Notes |
|----------------------------------|------|-------|--------|---------|-------------|-------|
| Phase 2: single LoRA (best step) | 84%  | 50%   | 92%*   | 61%*    | 76%*        | * = unstable |
| Phase 2: single LoRA (final)     | 84%  | 50%   | 45%    | -13%    | 51%         | Jordan/Leilani regress |
| R1: dual LoRA (final)            |  ?   |  ?    |  ?     |  ?      |  ?          | stability test |
| R2: dual + gated KL + K=10 (final)|  ? |  ?    |  ?     |  ?      |  ?          | ceiling test |
```

Success criteria:
- R1 dual LoRA **final** ≥ Phase 2 single LoRA **best step** (76%)
  → dual-rate eliminates need for early stopping
- R2 **exceeds teacher_k3** on distance>3 MCQs for ≥2 personas
  → cross-session accumulation advantage confirmed

---

## 9. Implementation order

### Step 1: Data changes (before any training)

- [ ] Modify `build_opd_data.py` to include 2-turn intra-session context
      in student input format
- [ ] Modify `eval_opd.py` / `eval_mcq_ppl` to use 2-turn context for
      student conditions at eval time
- [ ] Verify: student input now contains demographics + 2 recent turns +
      chatbot_prev (validate on 5 samples visually)

### Step 2: Round 1 training (Dual LoRA, standard KL, K=3 teacher)

- [ ] Implement dual LoRA setup in `train_opd.py` (two LoRA configs,
      two optimizers, both `.step()` per iteration)
- [ ] Sanity check: both adapters' parameters are updating (log grad
      norms for slow and fast separately)
- [ ] Train 4 personas × 1 epoch, save every 200 steps
- [ ] Run full eval suite: MCQ-PPL + NLL + judge on all checkpoints
- [ ] Compare learning curves to Phase 2 single LoRA:
      Focus on Jordan (still oscillating?) and Leilani (still decaying?)

### Step 3: Analysis of Round 1

- [ ] Distance_to_ref stratification on Round 1 results
- [ ] Per-type breakdown mapped to 3-dimension framework
- [ ] Decision: does Round 1 already solve stability? If yes, Round 2
      is about ceiling-breaking. If no, investigate before proceeding.

### Step 4: Round 2 training (Dual LoRA + Gated KL + K=10)

- [ ] Implement `gated_kl_loss` function in `train_opd.py`
- [ ] Add per-step gate_ratio logging (this IS the surprise signal — §5.4)
- [ ] Add per-session mean gate_ratio logging (replaces surprise curve)
- [ ] Change teacher OPD context K from 3 to 10 in data construction
- [ ] Rebuild OPD data with K=10 teacher contexts
- [ ] Train 4 personas × 1 epoch
- [ ] Run full eval suite
- [ ] Compare to Round 1 and Phase 2

### Step 5: Key analyses

- [ ] Distance_to_ref stratification: student vs teacher_k3 vs teacher_k10
- [ ] Gate_ratio curve analysis: is it monotonically decreasing?
- [ ] Per-session gate_ratio heatmap: which sessions have high/low info?
      (high gate_ratio sessions = where teacher has novel info for student;
       low gate_ratio sessions = student already knows this content)
- [ ] Per-type breakdown for all rounds
- [ ] Identify if student > teacher_k3 on any distance bucket / type

### Step 6: SFT baseline (for Dimension 1 comparison)

- [ ] Train SFT LoRA baseline: same 4 personas, same data, CE on GT
      user tokens instead of OPD KL. Same dual LoRA config.
- [ ] Same eval suite, same checkpoints
- [ ] Compare retention: Phase 1 MCQ accuracy after full training
      (SFT expected to forget more than OPD)

---

## 10. Compute estimate

Per persona per round:
- OPD training: ~1h (K=3) or ~2h (K=10) on single GH200
- MCQ-PPL eval per checkpoint: ~15 min
- 5 checkpoints × 15 min = 1.25h eval
- NLL + judge eval: ~30 min total

Per round (4 personas):
- Round 1 (K=3): ~4h training + 5h eval ≈ 9h
- Round 2 (K=10): ~8h training + 5h eval ≈ 13h
- SFT baseline: ~4h training + 5h eval ≈ 9h

Total: ~31h of GPU time across 4 GH200s (parallelized: ~8h wall time).

---

## 11. Key references from discussion

- Knowledge Neurons / ROME / MEMIT: MLP stores factual knowledge,
  attention does routing → motivates MLP-slow / Attention-fast split
- Born-Again Networks (Furlanello et al., ICML 2018): student can
  exceed teacher via dark knowledge accumulation
- ExOPD (arxiv 2602.12125): reward extrapolation enables student to
  surpass teacher in multi-teacher OPD
- Titans (arxiv 2501.00663): surprise-driven memory, entropy-based
  gating for memory management
- Nested Learning (arxiv 2512.24695): multi-level optimization with
  fast inner loop and slow outer loop → motivates dual-rate LR
- Gated KL unification: our entropy gate is analogous to Titans'
  surprise metric but operates at deployment-level OPD. Key insight:
  token-level gated KL subsumes session-level surprise detection —
  gate_ratio aggregated per session IS the surprise signal, eliminating
  the need for a separate mechanism (§5.4)
