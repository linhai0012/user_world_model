# MoE Architecture Implementation Specification

## 1. Motivation

### 1.1 Project Goal

Build an LLM-based user digital twin that **jointly generates** natural-language feedback, physiological time series (heart rate), and wellness indicators in a single autoregressive sequence. The model serves as a lightweight personal health simulator within a personalised health agent pipeline.

### 1.2 Why MoE?

Our theoretical framework posits that all observable user signals — text feedback, HR time series, wellness scores — are projections of a **shared latent psychophysiological state** through modality-specific emission functions:

```
S_t (shared latent state)
  ├→ f_text(S_t)    → text feedback       (semantic space)
  ├→ f_hr(S_t)      → HR time series      (cardiovascular space)
  └→ f_wellness(S_t) → wellness scores    (subjective rating space)
```

This maps directly onto a Transformer with:
- **Shared self-attention** = computing the shared hidden state S_t (all modality tokens attend to each other)
- **Modality-specific FFN experts** = the projection functions f_text(·), f_hr(·), f_wellness(·)

A single dense FFN forces one set of weights to handle both natural language generation and numerical time series generation — two fundamentally different computational tasks. MoE resolves this by routing tokens to specialised experts based on their modality.

### 1.3 What We Have Already

| Version | Experiment | Key Result |
|---------|-----------|------------|
| V1 (A1) | Omnimodal text+HR generation (dense Qwen3-4B) | Parse 73.8%, HR MAE 18.38 |
| V2 (B1) | + Baseline HR conditioning | HR MAE 17.19 (near noise floor 16.32), consistency 0.77 |
| V2.1 (C1) | + Delta encoding | Failed — cumulative error drift, MAE doubled |
| V3 (E1c) | Day-to-day wellness prediction | All fields lose to persistence baseline |
| V3+E-ctx | + 7-day history context | No improvement — task is information-limited |

The current dense model has reached the noise floor for HR prediction (MAE 17.19 vs floor 16.32). The MoE architecture is the next step to: (a) improve generation quality via modality specialisation, (b) enable plan-then-generate for better numerical generation, and (c) provide a modular architecture for adding new modalities (EDA, skin temperature, etc.).

---

## 2. Model Architecture

### 2.1 Base Model

**Qwen3-4B** (decoder-only transformer, `enable_thinking=False`). We modify its FFN layers into an MoE structure while keeping the self-attention layers shared.

### 2.2 Tokenisation

**Existing tokens** (from V2):
- 4 structural tokens: `<text_start>`, `<text_end>`, `<ts_start>`, `<ts_end>`
- 161 HR tokens: `<hr_40>` to `<hr_200>` (1-bpm bins)
- HR embedding initialisation: linear gradient in original embedding space (ordinal-aware)

**New tokens for MoE version** (if plan-then-generate is implemented):
- 2 plan tokens: `<plan_start>`, `<plan_end>`
- Wellness state tokens (from V3): `<state_start>`, `<state_end>`, plus per-field tokens

### 2.3 MoE Layer Design

Replace each FFN in Qwen3-4B with a Mixture-of-Experts block:

```
Input token hidden state h
        │
   Shared Self-Attention (unchanged from Qwen3-4B)
        │
   ┌────┴────┐
   │  Router  │  ← soft gating: σ(W_r · h)  → weights [w_text, w_ts, w_shared]
   └────┬────┘
        │
   ┌────┼────────────┐
   ▼    ▼            ▼
 FFN_text  FFN_ts  FFN_shared
   │    │            │
   └────┼────────────┘
        │
   Weighted sum: output = Σ w_i · FFN_i(h)
```

**Expert inventory:**

| Expert | Role | Initialisation |
|--------|------|---------------|
| `FFN_text` | Text generation specialist | Copy from Qwen3-4B pretrained FFN weights |
| `FFN_ts` | HR time series specialist | From Qwen3-4B, then Stage 0 HR pretraining |
| `FFN_shared` | Cross-modal bridging | Average of FFN_text and FFN_ts initial weights |

**Router design:**
- Soft router: all experts contribute (weighted sum), not top-k sparse
- Input: the hidden state h after attention
- Output: 3 weights via softmax (or sigmoid + normalise)
- Regularisation: routing entropy loss to encourage specialisation without collapse

**What stays shared (NOT split into experts):**
- Self-attention (Q, K, V projections + output projection) — this computes the shared latent state
- RMSNorm / LayerNorm layers
- Token embeddings and final LM head

**Design rationale (vs MSE-ITT which also splits attention):**
MSE-ITT splits QKV projections and LayerNorm by modality because they process two very different token types (natural language vs discretised stock returns) in an interleaved sequence where bottom-layer isolation helps. Our setting is different: the text and HR tokens appear sequentially (text first, then HR), so the attention mechanism already handles the causal ordering. We start with FFN-only MoE as the minimal intervention, and can later ablate attention splitting.

### 2.4 Input/Output Format

```
### Baseline HR:
<ts_start> <hr_65> <hr_63> <hr_64> ... <ts_end>
### Event:
Morning jog, 35 minutes, moderate pace
### Expected Duration: 35 min

### Response:
<plan_start>
Based on baseline ~65 bpm, moderate jog should peak 135-145.
Recovery expected within 8-10 minutes given user history.
<plan_end>
<text_start> That was a good run today... felt strong throughout... <text_end>
<ts_start> <hr_68> <hr_95> <hr_130> <hr_142> ... <hr_78> <hr_70> <ts_end>
```

### 2.5 Loss Function

```
L_total = α · L_text + β · L_hr + γ · L_plan + λ · L_routing

where:
  L_text     = CE loss on text tokens (including plan tokens)
  L_hr       = CE loss on HR tokens
  L_plan     = CE loss on plan tokens (optional, can merge with L_text)
  L_routing  = entropy regularisation on router weights
  
Default: α = 0.5, β = 0.5, λ = 0.01
```

The routing loss encourages:
- Text tokens → high weight on FFN_text
- HR tokens → high weight on FFN_ts
- Transition tokens (near modality boundaries) → allow FFN_shared

---

## 3. Training Pipeline

### 3.1 Stage 0: HR Pretraining

**Goal:** Give FFN_ts a "motor skill" for generating plausible HR sequences before joint training.

**Data:** Raw HR records from PMData (~20M+ HR values across all participants). Format as sequences of HR tokens for next-token prediction.

**Procedure:**
1. Start from Qwen3-4B
2. **Freeze:** all attention layers, FFN_text, embeddings
3. **Train:** only FFN_ts (copied from pretrained FFN) on HR-only sequences
4. Loss: standard CE on HR tokens
5. Training: ~5-10 epochs on HR data, lr ~1e-4

**Output:** Pretrained FFN_ts weights that understand HR token statistics (smoothness, typical ranges, acceleration/deceleration patterns).

### 3.2 Stage 1: MoE Assembly + Joint Training

**Goal:** Train the full MoE model on the complete multimodal task.

**Procedure:**
1. Assemble MoE architecture:
   - FFN_text ← Qwen3-4B pretrained weights (frozen or LoRA)
   - FFN_ts ← Stage 0 pretrained weights (frozen or LoRA)
   - FFN_shared ← average(FFN_text, FFN_ts)
   - Router ← random init (small weights, near-uniform initial routing)
2. **Freeze:** attention layers (optionally add LoRA adapters)
3. **Train:** Router + FFN_shared + LoRA on FFN_text and FFN_ts
4. Data: full training set with format from §2.4
5. Loss: L_total from §2.5
6. Training: 8-32 epochs (following V3 findings that small datasets need more epochs), lr ~1e-5

### 3.3 Data

**PMData (primary):**
- 2,438 samples (per-activity), 16 participants
- Split by participant: 12 train / 1 val / 3 test
- Activities: Walk 56%, Run 14%, Treadmill 10%, Bike 6%, others

**LifeSnaps (planned extension):**
- 71 participants, 4 months, Fitbit Sense
- EMA mood + Big Five personality + STAI anxiety
- To be explored for additional modality (mood/personality-conditioned generation)

### 3.4 Evaluation

**Metrics carried over from V1/V2:**

| Category | Metric | Description |
|----------|--------|-------------|
| Format | Parse success rate | % of outputs with valid structure |
| HR accuracy | MAE, RMSE | Against ground-truth HR sequence |
| HR dynamics | Trend accuracy | 3-class (up/down/flat) per timestep |
| Text quality | Word overlap | Token-level overlap with reference |
| Cross-modal | Consistency score | Rule-based text↔HR alignment check |
| Text quality | LLM-as-Judge | GPT-based evaluation of text naturalness |

**New metrics for MoE:**

| Category | Metric | Description |
|----------|--------|-------------|
| Routing | Expert utilisation | % of tokens routed to each expert |
| Routing | Modality alignment | Do HR tokens preferentially route to FFN_ts? |
| Routing | Routing entropy | Per-layer entropy of router weights |
| Ablation | Expert knockout | Performance when disabling one expert |

**Key baselines:**
- V2 (B1) dense model: MAE 17.19, consistency 0.77
- Noise floor: MAE 16.32 (data intrinsic variability)
- Persistence baseline (for wellness): predict tomorrow = today

---

## 4. Reference Implementation: MSE-ITT

MSE-ITT (Koval, Andrews & Yan, 2025) is the closest prior work — it applies modality-specific parameters to a pretrained LLM (Llama3-8B) for processing interleaved text and time series (stock returns). Their code is at `github.com/rosskoval/mlm_text_ts`.

### 4.1 Architecture Summary

MSE-ITT adds **dedicated TS-specific parameter copies** at each Llama layer:

```python
# Per layer, optionally adds:
self.ts_self_attn  = LlamaAttention(...)    # separate QKV for TS tokens
self.ts_mlp        = LlamaMLP(...)          # separate FFN for TS tokens  
self.ts_input_layernorm = LlamaRMSNorm(...) # separate LN for TS tokens
```

Routing is **hard** — based on a `modality_ids` tensor (0=text, 1=TS):

```python
# Forward pass pattern (repeated for attn, LN, MLP):
output = self.mlp(hidden_states)                # text MLP on all tokens
if self.config.separate_ts_mlp_params:
    mlp_ts = self.ts_mlp(hidden_states[ts_idx]) # TS MLP on TS positions
    output[ts_idx] = mlp_ts                     # overwrite TS positions
```

### 4.2 Selective Cross-Modal Attention

MSE-ITT's most distinctive feature: **bottom-half layers use modality-isolated attention, top-half layers use full cross-modal attention.**

```python
# Bottom layers: separate attention per modality using flash_attn_varlen_func
# Each modality's tokens only attend to tokens of the same modality
for mod in modality_ids.unique():
    mod_mask = (modality_ids == mod)
    out_flat = flash_attn_varlen_func(q_mod, k_mod, v_mod, ...)

# Top layers: standard full causal attention across all tokens
attn_output = _flash_attention_forward(q, k, v, attention_mask, ...)
```

This "isolate-then-fuse" design lets each modality build its own representation before cross-modal reasoning.

### 4.3 TS Encoding

MSE-ITT uses **quantile binning** rather than fixed bins:

```python
# Data-driven bin edges from training distribution
quantiles = torch.linspace(0, 1, vocab_size + 1)
self.bin_edges = torch.quantile(flat_train_ts, quantiles)
# Default vocab_size = 32 bins
```

Embeddings are initialised by sampling from the LLM's token embedding distribution (multivariate normal fit).

### 4.4 SALMON Pre-training

Two-objective CLM loss on interleaved sequences:
- `lm_head` for text next-token prediction (initialised from pretrained LM head)
- `ts_head` for TS next-token prediction (randomly initialised linear layer)

**Salient Token Weighting (STW):** identifies text tokens that benefit from TS context by comparing predictions with and without TS tokens visible:

```python
# Two forward passes:
# (1) Full multimodal → text_loss
# (2) TS tokens masked out → contrast_loss  
# Weight = exp(contrast_loss - text_loss)  — higher when TS context helps
```

### 4.5 Training Strategy

- TS-specific parameters trained with **LoRA** (not full fine-tuning)
- Supports QLoRA (4-bit quantisation)
- Two-phase: SALMON pre-training → classification fine-tuning

### 4.6 Key Differences from Our Design

| Aspect | MSE-ITT | Our MoE |
|--------|---------|---------|
| Task | Classification (stock direction) | Generation (text + HR + wellness) |
| Routing | Hard (modality_ids lookup) | Soft (learned router weights) |
| Experts | 2 (text + TS), no shared expert | 3 (text + TS + shared) |
| Attention | Bottom: isolated; Top: cross-modal | All layers: shared (start simple) |
| QKV split | Yes (separate ts_self_attn) | No (shared attention only, FFN split) |
| LN split | Yes (separate ts_layernorm) | No (shared LN) |
| TS encoding | Quantile binning (32 bins) | Fixed 1-bpm bins (161 tokens) |
| TS init | Sample from LLM embedding distribution | Linear gradient in embedding space |
| Pre-training | SALMON (CLM + STW alignment) | Stage 0 HR-only pretraining |
| Personalisation | None | In-context (user history as few-shot) |
| Base model | Llama3-8B | Qwen3-4B |

### 4.7 What to Reference from MSE-ITT Code

Key files: `mse_itt.py` (model), `salmon.py` (pre-training head)

Useful implementation patterns:
1. **`MSE_ITT_Layer.forward()`** (line 435-628): The modality-routing pattern in the forward pass — how to efficiently index and overwrite tensor positions by modality
2. **`init_ts_weights()`** (line 376-397): Loading pretrained text weights into TS-specific modules
3. **`TSEncoder`** (line 248-302): Quantile binning implementation (consider adapting for HR)
4. **`create_peft_config()`** (line 819-853): LoRA configuration targeting only TS-specific parameters
5. **`create_interleaved_sequence()`** (line ~1020-1184): How to construct interleaved multimodal input sequences with modality tracking tensors

---

## 5. Implementation Priorities

### Phase 1: Minimal MoE (FFN-only split)
1. Implement `MoEFFN` module: 3 experts + soft router
2. Replace Qwen3-4B FFN layers with MoEFFN
3. Implement modality-aware loss (L_text + L_hr + L_routing)
4. Train and evaluate against V2 (B1) baseline

### Phase 2: Stage 0 HR Pretraining
1. Build HR-only training data from raw PMData
2. Implement Stage 0 training script (freeze attention, train FFN_ts only)
3. Verify FFN_ts produces smoother/more realistic HR sequences

### Phase 3: Plan-then-Generate
1. Modify data construction to insert plan text before HR generation
2. Add plan tokens to tokeniser
3. Evaluate whether plan improves HR accuracy and consistency

### Phase 4: Evaluation & Ablations
1. Full evaluation suite (HR MAE, consistency, routing analysis)
2. Ablations: remove each expert, vary number of experts, hard vs soft routing
3. Compare with MSE-ITT-style hard routing as baseline
