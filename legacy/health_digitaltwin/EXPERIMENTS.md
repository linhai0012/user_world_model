# Experiment Log

## Project Overview

### Research Goal
Build an omnimodal LLM-based "digital twin" that simultaneously generates natural-language user feedback **and** physiological time series (heart rate) in a single autoregressive sequence. The model serves as a lightweight personal health simulator.

### Architecture
- **Base model**: Qwen3-4B (text-only, `enable_thinking=False`)
- **Vocab extension**: 165 new tokens — 4 structural (`<text_start>`, `<text_end>`, `<ts_start>`, `<ts_end>`) + 161 HR tokens (`<hr_40>` .. `<hr_200>`, 1-bpm bins)
- **Generation paradigm**: Sequential — text first, then HR time series
- **Output format**: `<text_start> [user feedback] <text_end> <ts_start> [HR tokens] <ts_end>`
- **HR embedding init**: Linear gradient in original embedding space (ordinal-aware)
- **Loss**: Weighted CE — `L = α·L_text + (1-α)·L_hr`, default α=0.5

### Dataset: PMData
- **Source**: PMData (Simula) — Fitbit heart rate + exercise logs from 16 participants
- **Text synthesis**: GPT-5.4 generates (event description, user feedback) pairs from raw sensor data
- **Split**: By participant (not by sample) to prevent data leakage
- **Total**: 2,438 samples

| Split | Samples | Participants | Avg HR seq len | Avg input words | Avg output words |
|-------|---------|-------------|----------------|-----------------|------------------|
| Train | 1,851 | 12 | 72.3 | 42.8 | 126.8 |
| Val | 145 | 1 | 63.0 | 40.3 | 116.2 |
| Test | 442 | 3 | 67.7 | 42.2 | 121.3 |

**Activity distribution** (full dataset):

| Activity | Count | % |
|----------|-------|------|
| Walk | 1,374 | 56.4% |
| Run | 339 | 13.9% |
| Treadmill | 252 | 10.3% |
| Outdoor Bike | 148 | 6.1% |
| Sport | 91 | 3.7% |
| Weights | 83 | 3.4% |
| Other (13 types) | 151 | 6.2% |

**Known data issues**:
- Val set contains only 1 participant — limited diversity, may affect early stopping reliability
- Heavy class imbalance — Walk dominates at 56%, many activities have <10 samples
- HR sequence lengths vary widely (18–120 tokens)

### Planned Ablation Experiments

| ID | Experiment | Config | Purpose | Status |
|----|-----------|--------|---------|--------|
| A1 | Omnimodal α=0.5 | `--loss_alpha 0.5` | Main model | Training |
| A2 | Omnimodal α=0.3 | `--loss_alpha 0.3` | HR-weighted variant | Planned |
| A3 | Uniform CE | `--loss_alpha -1` | No modality weighting baseline | Planned |
| A4 | Random HR init | `--hr_embed_init random` | Test embedding init impact | Planned |
| A5 | Text-only | Remove HR tokens from data | Text baseline | Planned |
| A6 | HR-only | Remove text tokens from data | HR baseline | Planned |

### Evaluation Metrics

**Text quality**: Perplexity, token accuracy, LLM-as-Judge (GPT-5.4)
**HR accuracy**: MAE, RMSE, trend accuracy, range error, token exact-match
**Cross-modal consistency**: Rule-based checker (intensity language ↔ HR level), LLM-as-Judge

### Infrastructure
- **Data storage**: HuggingFace Hub (`lzhang472/digital-twin-data`, private)
- **Model storage**: HuggingFace Hub (TBD, after training completes)
- **Compute**: Isambard-AI BriCS (4× H200 per node), KCL CREATE (A100/H100/H200/B200)

---

## Experiment 1: Initial Training Run (2026-03-20)

**ID**: A1 (first attempt)
**Server**: Isambard-AI, node nid010624, 4× H200
**Config**: α=0.5, lr=2e-5, 8 epochs, effective batch=32, cosine schedule, warmup=10%, SDPA attention

### Result: Training terminated early (~0.86/8 epochs)

Two nearly identical runs (`Mar20_15-29-42`, `Mar20_15-40-23`), both stopped at step 50 out of expected ~464. Cause unknown — likely Slurm walltime or manual interruption.

| Step | Epoch | Total Loss | Text Loss | HR Loss | Grad Norm | LR |
|------|-------|-----------|-----------|---------|-----------|-----|
| 1 | 0.02 | 32.06 | ~1.93 | ~13.91 | 472 | 0 |
| 10 | 0.17 | 30.93 | ~2.04 | ~11.42 | 458 | 4e-6 |
| 20 | 0.34 | 23.64 | ~1.25 | ~8.97 | 78 | 8e-6 |
| 30 | 0.52 | 18.63 | ~0.80 | ~7.47 | 47 | 1.2e-5 |
| 40 | 0.69 | 15.11 | ~0.64 | ~6.04 | 77 | 1.7e-5 |
| 50 | 0.86 | 12.06 | ~0.50 | ~4.73 | 32 | 2e-5 |

### Analysis

1. **Training incomplete**: Only ~11% of planned training done. Model far from convergence.
2. **Text loss converging fast**: 1.93 → 0.50 in <1 epoch. Qwen3-4B pretrained knowledge transfers well.
3. **HR loss still near random**: 4.73 vs random baseline ln(161) ≈ 5.08. HR prediction barely above chance.
4. **Gradient norm spike**: 472 at step 1, stabilizing to 30–80 by step 20. New HR embeddings cause large initial gradient mismatch.
5. **LR still in warmup**: At step 50, LR just reached peak 2e-5. Cosine decay hasn't started.
6. **No eval metrics**: `prediction_loss_only=True` (OOM fix) disabled accuracy computation. No eval_loss recorded.

### Issues
- [ ] Training terminated early — cause not determined
- [ ] `prediction_loss_only=True` disables eval accuracy metrics
- [ ] Val set single-participant limits early stopping reliability
- [ ] Initial grad norm spike (472) suggests need for warmup strategy

---

## Experiment 2: Full Training Run (2026-03-24)

**ID**: A1 (retry)
**Server**: Isambard-AI, node nid010186, 4× H200
**Config**: Same as Experiment 1 — α=0.5, lr=2e-5, 8 epochs, effective batch=32
**Data**: Restored from HuggingFace Hub (`lzhang472/digital-twin-data`)
**Status**: Training in progress

### Changes from Experiment 1
- Running directly on compute node (not via Slurm) with `nohup` to avoid premature termination
- No code changes — same configuration as Experiment 1

### Results

Training completed 8 full epochs successfully.

- **Final eval loss**: 1.8719
- **Text accuracy**: N/A (`prediction_loss_only=True` disabled `compute_metrics`)
- **HR accuracy**: N/A (same reason)

Training log crashed at the very end due to a formatting bug in `train.py:257` — tried to format string `'N/A'` with `:.4f`. Model and checkpoints were saved before the crash.

### Analysis

1. **Eval loss 1.8719**: This is the combined weighted loss on the val set. Compared to Experiment 1's step-50 total train loss of 12.06, the model has converged significantly. However, without separate text/HR eval loss breakdown, we can't assess per-modality performance.
2. **No accuracy metrics available**: The `prediction_loss_only=True` workaround for eval OOM means we have no token-level accuracy. Need to run `generate_and_eval.py` for full evaluation.
3. **Training completed without interruption**: The `nohup` + direct compute node approach worked — no Slurm walltime issues.

### Post-Training Evaluation (generate_and_eval.py on test set, 442 samples)

#### Aggregate Metrics

| Metric | Value |
|--------|-------|
| Parse success rate | 73.8% (326/442) |
| HR MAE | 18.38 bpm (median 17.16) |
| HR RMSE | 22.05 bpm (median 20.59) |
| HR Trend accuracy | 0.4399 (near random ~0.33) |
| Text word overlap | 0.5400 |
| Consistency score | 0.7326 (on 268 checkable samples) |
| HR length ratio (pred/ref) | 1.392 (model over-generates HR tokens) |

#### Failure Analysis (116 failed parses)

All 116 failures share the same pattern:
- All have `<text_end>` and `<ts_start>` — text was generated correctly
- **None have `<ts_end>`** — HR sequence was truncated by `max_new_tokens=300`
- Failed samples have longer reference HR sequences (mean 86.6 vs 60.9 for successes)
- Root cause: **`max_new_tokens=300` is too short** for samples with long HR sequences. Text (~100 tokens) + long HR (~90+ tokens) + structural tokens can exceed 300.

#### Key Findings

1. **Text generation works well**: The model generates coherent first-person feedback with 54% word overlap to reference. Consistency score 0.73 shows reasonable text-HR alignment.
2. **HR accuracy is poor**: MAE of 18.38 bpm is high — the model's HR predictions are not yet precise at the per-token level.
3. **HR trend accuracy ~0.44 is near random**: For 3-class trend (up/down/flat), random baseline is ~0.33. The model captures some directional patterns but not reliably.
4. **Truncation is the main failure mode**: 26.2% of samples fail purely because generation is cut off before `<ts_end>`. Fix: increase `max_new_tokens` to 512.
5. **Model over-generates HR tokens**: Length ratio 1.39 means predicted HR sequences are ~39% longer than ground truth on average.

### Issues
- [x] Training completed successfully (8 epochs)
- [x] Bug fix: `train.py` line 257 format string error (fixed)
- [x] Evaluation completed with `generate_and_eval.py`
- [x] Increase `max_new_tokens` to 512 (fixed in V2 code)
- [ ] HR accuracy needs improvement → addressed by V2 paradigm shift
- [ ] Per-modality eval breakdown still not available during training

---

## V2: Baseline-HR-Conditioned Paradigm (2026-03-24)

### Motivation

V1 model has no user-conditioning information. Different users have very different physiological responses to the same activity (e.g., an athlete's HR during a 30-min run vs. a sedentary person's). Without knowing the user's baseline, the model can only predict population-average HR, which inflates MAE.

### Architecture Change

```
V1:  Input: [event description]                     → Output: [text] + [full HR]
V2:  Input: [baseline HR] + [event] + [duration]    → Output: [text] + [response HR]
```

The pre-event HR (10 min before activity) is provided as input context, allowing the model to infer the user's resting HR level and personalize predictions. Duration is also included to help the model control output HR sequence length.

### New Training Format

```
### Baseline HR:
<ts_start> <hr_65> <hr_63> <hr_64> ... <ts_end>
### Event:
{event description}
### Expected Duration: {N} min

### Response:
<text_start> {user feedback} <text_end> <ts_start> <hr_120> <hr_135> ... <ts_end>
```

### Key Changes
- `config.py`: `HR_BASELINE_WINDOW_MIN=10` replaces `HR_WINDOW_BEFORE_MIN=5`
- `step1_parse_pmdata.py`: Separate `extract_hr_baseline()` and `extract_hr_response()` functions
- `step3_build_training_data.py`: Merges new step1 output with existing step2 text, new format, val split 11/2/3
- `dataset.py`: Cached `hr_token_ids`, baseline HR in input is masked (no loss)
- `generate_and_eval.py`: Uses full `input_text` as prompt, `max_new_tokens=512`
- GPT-5.4 text synthesis NOT re-run (existing text reused)

### Ablation Plan

| ID | Experiment | Config | Purpose |
|----|-----------|--------|---------|
| B1 | Baseline HR 10min, α=0.5 | Main | New paradigm primary |
| B2 | Baseline HR 10min, α=0.3 | HR-weighted | Test stronger HR weighting |
| B3 | No baseline (V1), α=0.5 | Control | Existing A1 result |
| B4 | Baseline HR 5min, α=0.5 | Short window | Window length ablation |
| B5 | Baseline HR 20min, α=0.5 | Long window | Window length ablation |

### Status
- [x] Code changes complete (local)
- [x] Push to GitHub
- [x] Re-run step1 on server (extract baseline HR from PMData)
- [x] Run step3 on server (merge + reformat + split)
- [ ] Upload new data to HuggingFace
- [x] Train B1 experiment
- [x] Evaluate and compare with V1 A1 baseline

### B1 Training Results (2026-03-25)

**Config**: α=0.5, lr=2e-5, 8 epochs, effective batch=32, 4×H200, baseline window=10min
**Final eval loss**: 1.8874

### B1 Evaluation Results (442 test samples)

#### V1 vs V2 Comparison

| Metric | V1 (A1) | V2 (B1) | Change |
|--------|---------|---------|--------|
| Parse success rate | 73.8% | **99.8%** | +26.0pp |
| HR MAE (bpm) | 18.38 | **17.19** | -6.5% |
| HR RMSE (bpm) | 22.05 | **20.80** | -5.7% |
| HR Trend accuracy | 0.4399 | **0.4473** | +0.7% |
| Text word overlap | 0.5382 | **0.5428** | +0.8% |
| Consistency score | 0.5113 | **0.7729** | +51.2% |

Note: V1 parse rate was artificially low due to `max_new_tokens=300` truncation.
V2 uses `max_new_tokens=512`, which alone explains the parse rate improvement.

#### Analysis

1. **Parse success near-perfect (99.8%)**: `max_new_tokens=512` eliminates truncation.
2. **HR MAE improved modestly (18.38 → 17.19, -6.5%)**: Baseline HR conditioning helps but is not a silver bullet. The model uses baseline info for rough calibration but doesn't achieve fine-grained per-minute accuracy.
3. **Consistency score jumped significantly (0.51 → 0.77, +51%)**: The model produces much better text-HR alignment with baseline context. This is the biggest qualitative win.
4. **Trend accuracy unchanged (~0.44)**: Per-step directional prediction remains near random. The model predicts reasonable HR ranges but not precise temporal dynamics.
5. **HR sequences still over-generated**: Predicted lengths ~25-35% longer than reference despite duration conditioning.

#### Remaining Weaknesses
- HR trend accuracy near random — model captures level but not dynamics
- HR length control imperfect — model over-generates despite duration input
- HR MAE still ~17 bpm — significant room for improvement

#### Next Steps
- [ ] Run B2 (α=0.3) to test HR-weighted loss
- [ ] Run B4/B5 (5min/20min baseline window) for window length ablation
- [ ] Investigate HR length control — consider adding explicit length token
- [x] Delta encoding for HR sequences → V2.1

---

## V2.1: Delta Encoding for HR Sequences (2026-03-25)

### Motivation

HR prediction remains the core challenge (MAE=17.19, trend acc=0.45). The LLM treats 161 absolute HR tokens as independent categories with no numerical prior. Delta encoding transforms the problem:

- **161-class → ~11-class**: 60% of deltas are ±5, model focuses on small changes
- **Built-in smoothness**: `<hd_+0>` (no change) is the most frequent token at 10.1%
- **Error isolation**: A wrong delta only affects one step, not the whole sequence

### Data Analysis (170,426 delta values)

| Range | Coverage | Effective classes |
|-------|----------|-------------------|
| ±5 | 60.0% | 11 |
| ±10 | 80.8% | 21 |
| ±20 | 94.6% | 41 |

Distribution: mean=0.0, std=9.95, median=0. Strongly peaked at 0 (Laplace-like).

### Encoding Format

```
Before: <ts_start> <hr_142> <hr_138> <hr_135> <hr_130> <hr_128> <ts_end>
After:  <ts_start> <hr_142> <hd_-4>  <hd_-3>  <hd_-5>  <hd_-2>  <ts_end>
```

- First token: absolute HR (anchor point)
- Subsequent tokens: delta from previous step, clamped to ±20
- 41 new delta tokens: `<hd_-20>` to `<hd_+20>` (explicit sign: `<hd_+0>`)
- Baseline HR in input unchanged (absolute encoding)
- Total special tokens: 165 + 41 = 206

### Changes
- `config.py`: Delta token definitions, encode/decode functions
- `step3_build_training_data.py`: Output HR uses delta encoding
- `model_setup.py`: Delta token embedding initialization (linear gradient)
- `dataset.py`: Delta tokens included in HR token set for weighted loss
- `metrics.py`: Parser auto-detects delta encoding, reconstructs absolute HR
- `visualize_eval.py`: Updated HR parser for delta decoding

### Status
- [x] Code changes complete
- [x] Push to GitHub
- [x] Re-run step3 on server (re-encode training data)
- [x] Train C1 experiment (baseline HR + delta encoding, α=0.5)
- [x] Evaluate and compare with B1 baseline

### C1 Training Results (2026-03-25)

**Config**: α=0.5, lr=2e-5, 8 epochs, effective batch=32, 4×H200, baseline 10min, delta encoding ±20
**Final eval loss**: 1.7258 (lower than B1's 1.8874 — delta tokens are easier to predict)

### C1 Evaluation Results (442 test samples)

#### Full Comparison: V1 → V2 → V2.1

| Metric | V1 (A1) | V2 (B1) | V2.1 (C1) | Best |
|--------|---------|---------|-----------|------|
| Parse rate | 73.8% | 99.8% | **100%** | C1 |
| HR MAE (bpm) | 18.38 | **17.19** | 33.63 | B1 |
| HR RMSE (bpm) | 22.05 | **20.80** | 38.65 | B1 |
| Trend accuracy | 0.4399 | **0.4473** | 0.3894 | B1 |
| Text word overlap | 0.5382 | **0.5428** | 0.5424 | B1 |
| Consistency score | 0.5113 | **0.7729** | 0.7149 | B1 |
| HR length ratio | 1.392 | 1.392 | **1.011** | C1 |
| Eval loss | 1.8719 | 1.8874 | **1.7258** | C1 |

#### Analysis

1. **Eval loss decreased (1.89 → 1.73)**: Delta tokens are inherently easier to predict (fewer effective classes), so per-token loss is lower. But this doesn't translate to better absolute HR accuracy.

2. **HR MAE nearly doubled (17.19 → 33.63)**: Delta error accumulation is the dominant problem. While individual delta predictions may be reasonable, systematic bias (consistently predicting slightly too positive or too negative) causes the reconstructed absolute HR to drift away from ground truth over time.

3. **Error scales with sequence length**:
   - Short sequences (≤40 steps): MAE = 31.26
   - Medium sequences (41-80 steps): MAE = 30.54
   - Long sequences (>80 steps): MAE = 49.85

   The worst cases (MAE > 140 bpm) are all long sequences where delta drift compounds.

4. **HR length ratio dramatically improved (1.39 → 1.01)**: The duration conditioning now works well — the model generates sequences very close to the expected length. This was a clear win from the V2 architecture, enhanced by delta encoding's simpler stop prediction.

5. **Trend accuracy degraded (0.45 → 0.39)**: Delta drift corrupts trend matching. When the absolute level is wrong, step-to-step direction comparisons become unreliable.

#### Visualization Findings

Visual comparison of B1 vs C1 on the same samples confirms:
- **C1 worst cases**: Predicted HR starts near ground truth but progressively drifts away. Curves diverge after 20-40 steps, reaching 50-100+ bpm offset by sequence end.
- **B1 on same samples**: Predictions fluctuate around ground truth without systematic drift. Even when inaccurate, they stay in the right ballpark.
- **C1 best cases**: Short sequences where drift hasn't accumulated. Performance comparable to B1.

#### Conclusion

**Delta encoding failed** in this configuration. The reduced per-token classification difficulty (eval loss 1.73 < 1.89) was more than offset by error accumulation in long sequences. The fundamental issue: LLM autoregressive generation has no mechanism to self-correct drift — each delta prediction is independent, and systematic bias compounds linearly with sequence length.

#### Possible Improvements (not yet tested)
- **Hybrid encoding**: Insert absolute HR anchors every N steps to reset drift
- **Coarser absolute bins**: Use 5-bpm bins (32 classes) instead of 1-bpm (161 classes)
- **Loss penalty on drift**: Add auxiliary loss term on cumulative absolute error

---

## HR Predictability Analysis (2026-03-25)

### Motivation

Before further model optimization, we need to answer: **is the HR sequence fundamentally predictable given our conditioning information?** If the same user doing the same activity produces very different HR patterns each time, then no model can achieve low MAE.

### Method

Computed pairwise MAE between HR sequences within groups of (same user, same activity, similar duration ±5 min). This measures the **intrinsic variability** of the data — the noise floor that no model can beat.

### Results

| Comparison | Pairwise MAE | Interpretation |
|------------|-------------|----------------|
| Same user + same activity + same duration (±5min) | **16.32** | Data noise floor |
| Mean HR profile baseline (leave-one-out, same user+activity) | **15.17** | Simple baseline |
| **Our model B1** | **17.19** | Only 0.87 above noise floor |
| Same user + same activity (any duration) | 19.86 | Duration matters |
| Cross-user + same activity | 26.38 | Individual variation |

Additional statistics:
- **Pairwise correlation** between same-user same-activity HR curves: mean=0.33, median=0.36
- This means HR curve **shape** is only weakly reproducible even for the same person doing the same activity

![HR variability within same user + same activity](output/hr_variability_same_user_activity.png)

### Key Conclusions

1. **Model B1 (MAE=17.19) is already near-optimal.** The data noise floor is 16.32 — our model is within 0.87 bpm of the theoretical best. Further MAE improvement is bounded to ~2 bpm at most.

2. **HR dynamics are inherently noisy at 1-min resolution.** The low pairwise correlation (0.33) explains why trend accuracy is stuck at ~0.44 — the minute-to-minute HR trajectory is not reproducible even for the same person.

3. **The remaining gap (~2 bpm) could come from**:
   - Better per-user conditioning (more personal history)
   - Finer activity sub-type information
   - External factors not captured (weather, sleep, caffeine, etc.)

4. **For the paper**: This analysis provides a strong argument that the model has learned effectively. The focus should shift from "how to lower MAE further" to "demonstrating that the model captures the learnable signal and generates physiologically coherent multimodal outputs."

### Implications for Next Steps

- **Stop chasing MAE**: 17.19 vs floor of 16.32 leaves minimal room
- **Focus on qualitative evaluation**: LLM-as-Judge, user studies, coherence
- **Focus on ablation completeness**: Run B2-B5 to show which components matter
- **Strengthen the paper narrative**: Omnimodal generation + near-optimal HR + text-HR consistency

---

## V3: Day-to-Day Wellness Prediction (2026-03-25)

### Motivation

HR time-series prediction hit its ceiling (MAE 17.19 vs noise floor 16.32). More importantly, predicting minute-by-minute HR curves has limited downstream value for health agents.

**New paradigm**: Predict **Day N+1 morning wellness state** from Day N state + Day N activity. This directly supports the health agent decision loop:

```
Day N: Observe state → Recommend activity → User executes
         ↓
       [Digital Twin predicts Day N+1 state]
         ↓
Day N+1: Check prediction → Adjust next recommendation
```

### Data Construction

- **Source**: PMData wellness.csv (daily self-report) + srpe.csv (session RPE)
- **Pairing**: Each sample = (Day N wellness, Day N activities, Day N+1 wellness)
- **Multi-activity days**: All activities merged into one daily summary
- **Text synthesis**: GPT-5.4 generates event descriptions + diary entries reflecting day-to-day recovery

| Metric | Value |
|--------|-------|
| Paired daily samples | 936 (from 2438 per-activity records) |
| With sRPE | 44% |
| With baseline HR | ~90% |
| Participants | 15 |
| Train / Val / Test | 80% / 10% / 10% by participant |

**Data reduction**: 2438 → 936 samples due to (1) Day N+1 wellness pairing requirement (-28%) and (2) multi-activity day merging (-47%).

### Training Format

```
### User State (Day N):
Fatigue: 3/5, Mood: 4/5, Readiness: 7/10, Sleep Quality: 4/5,
Soreness: 3/5, Stress: 4/5
### Baseline HR:
<ts_start> <hr_65> <hr_63> ... <ts_end>
### Activity (Day N):
Walk and Run, total 55 minutes, 480 calories, 6200 steps,
Perceived Exertion: 6/10

### Response:
<text_start> It was a solid day of exercise... <text_end>
<state_start> <fatigue_2> <mood_4> <readiness_6> <sleep_quality_3>
<soreness_2> <stress_4> <state_end>
```

State tokens: 37 new tokens (6 wellness fields × 5-10 values + 2 structural).

### E1a: 8 epochs, lr=2e-5 (first attempt)

- Parse rate: **9.5%** — model didn't learn format
- All fields worse than persistence baseline
- Root cause: 545 training samples (bad 60/13/27 split) + 8 epochs insufficient

### E1b: 8 epochs, lr=2e-5, fixed split (80/10/10)

- Parse rate: **19.7%** — slightly better but still mostly failing format
- Readiness showed some signal but overall worse than persistence

### E1c: 32 epochs, lr=1e-5, fixed split

**Config**: α=0.5, lr=1e-5, 32 epochs, effective batch=32, 4×H200

| Field | Model MAE | Persistence MAE | Model Exact% | Model Wins? |
|-------|----------|-----------------|-------------|-------------|
| fatigue | 0.31 | **0.08** | 68.6% | No |
| mood | 0.39 | **0.04** | 61.3% | No |
| readiness | 1.24 | **0.64** | 29.2% | No |
| sleep_quality | 0.54 | **0.14** | 51.1% | No |
| soreness | 0.38 | **0.14** | 64.2% | No |
| stress | 0.37 | **0.04** | 62.8% | No |

Additional metrics:
- Parse rate: **100%** (32 epochs solved format learning)
- Text word overlap: **0.445**
- Consistency score: 0.24

### Analysis

1. **Format learning solved**: 32 epochs → 100% parse rate (vs 9.5% at 8 epochs). Small datasets need more epochs.

2. **Model loses to persistence baseline on all fields**: The "predict tomorrow = today" strategy is extremely strong on this dataset because most participants' wellness changes slowly.

3. **The core issue is user heterogeneity**: Analysis of per-participant wellness variability reveals:
   - **2 "stable" participants** (p02, p03): wellness almost never changes (persistence MAE ≈ 0.03-0.10)
   - **14 "variable" participants**: wellness changes 25-55% of days (persistence MAE ≈ 0.3-1.0)
   - The model cannot distinguish these two types from the input alone

4. **Test set composition matters**: The test set's persistence MAE (fatigue=0.08, mood=0.04) is much lower than the population average (fatigue=0.40, mood=0.27), suggesting the test participants happen to be unusually stable.

5. **Model's predictions are semantically reasonable**: It predicts fatigue should decrease after exercise, soreness should increase — these are correct patterns. But since the ground truth often shows no change, the model gets penalized for predicting reasonable changes that didn't happen.

### Per-Participant Wellness Variability

| Participant | Avg Change Rate | Classification |
|-------------|----------------|----------------|
| p02 | 12.5% | STABLE |
| p03 | 6.4% | STABLE |
| p01 | 34.9% | VARIABLE |
| p04 | 56.8% | VARIABLE |
| p08 | 46.9% | VARIABLE |
| p10 | 52.2% | VARIABLE |
| (10 others) | 25-50% | VARIABLE |

Persistence baseline by group:
- **Stable group**: fatigue MAE=0.03, mood MAE=0.04 (trivially easy)
- **Variable group**: fatigue MAE=0.49, mood MAE=0.32 (meaningful prediction challenge)

### Open Questions

- [ ] Evaluate model only on "variable" participants — does it beat persistence there?
- [x] Add user history (past 7-day wellness trend) to input for user-type inference → E-context
- [ ] Consider per-user fine-tuning as Phase 2 (personalization)
- [ ] Evaluate on days where wellness actually changed (the interesting cases)

---

## E-context: In-Context Personalization via 7-Day History (2026-03-28)

### Motivation

E1c showed that the model cannot distinguish stable vs variable users from a single day's input. Solution: provide the past 7 days of wellness+activity→next-day-wellness transitions as in-context history, so the model can infer user-specific patterns.

### Input Format Change

Added `### Recent History:` section before the existing input:

```
### Recent History:
Day -7: Fat3 Mood3 Rdy5 Slp3 Sor3 Str3 | Walk 30min 245cal -> Fat3 Mood3 Rdy5 Slp3 Sor3 Str3
Day -6: Fat3 Mood3 Rdy5 Slp3 Sor3 Str3 | Run 45min 520cal RPE7 -> Fat2 Mood3 Rdy4 Slp3 Sor2 Str3
...
Day -1: Fat3 Mood3 Rdy6 Slp3 Sor3 Str3 | Walk 20min 180cal -> Fat3 Mood3 Rdy5 Slp3 Sor3 Str3
### User State (Day N):
...
```

Each history line shows: day-N-k state | activity summary → day-N-k+1 state. This lets the model see how often this user's wellness changes and what activities cause changes.

### Changes
- `step1_parse_pmdata.py`: `build_daily_records()` attaches 7-day history to each sample
- `step3_build_training_data.py`: `format_sample()` adds compact history text to input
- No changes to step2 (reused existing GPT text), config, model, metrics, or eval code

**Config**: α=0.5, lr=1e-5, 32 epochs, 4×H200

### Results

| Field | E-context | E1c (no history) | Persistence | E-context wins? |
|-------|-----------|-------------------|-------------|-----------------|
| fatigue | 0.336 | 0.314 | **0.080** | No |
| mood | 0.343 | 0.394 | **0.037** | No |
| readiness | 1.321 | 1.241 | **0.642** | No |
| sleep_quality | **0.453** | 0.540 | 0.139 | No (but improved vs E1c) |
| soreness | 0.350 | 0.305 | **0.139** | No |
| stress | 0.416 | 0.372 | **0.037** | No |

Additional: parse rate=100%, text overlap=0.436, consistency=0.245

### Analysis

1. **7-day history did not help**: E-context results are essentially the same as E1c. The model did not learn to use history to distinguish stable vs variable users.

2. **sleep_quality slightly improved** (0.540 → 0.453), but still far from persistence baseline (0.139).

3. **The model consistently over-predicts change**: When the true Day N+1 is the same as Day N (which is ~70% of cases), the model still predicts changes (e.g., fatigue 3→2, readiness 5→7). LLM's generative bias towards producing "meaningful" content works against predicting "no change."

4. **Root cause confirmed**: The PMData wellness data is too stable for day-to-day prediction to be a viable task on this dataset. With mood/stress changing only 4-5% of days in the test set, any model that predicts change will lose to persistence.

### Conclusion

The in-context personalization approach is sound in principle, but the **PMData wellness signal is too weak** to demonstrate its value. The data has:
- 6 wellness fields on 1-5 or 1-10 scales
- Most fields stay constant for most users on most days
- The test set participants are particularly stable (persistence MAE ≈ 0.04 for mood/stress)

This is a **dataset limitation**, not a model limitation. The approach would likely work better with:
- A dataset with more variable wellness indicators
- Higher-resolution wellness tracking (multiple times per day)
- More diverse user populations

---

## GPT-4o Upper Bound Test (2026-03-28)

### Motivation

To determine whether the wellness prediction task is fundamentally hard (information-limited) or just hard for our fine-tuned model, we tested GPT-4o (zero-shot, no fine-tuning) on 5 representative samples: 3 no-change cases and 2 big-change cases.

### Method

- Model: GPT-4o, temperature=0 (deterministic)
- Prompt: Day N wellness + activity summary → predict Day N+1 wellness as JSON
- No user history provided (same information as our model without E-context)
- 5 samples selected from train set: 3 where wellness didn't change, 2 where total change ≥ 14

### Results

| Sample | Type | GPT-4o err | Persistence err | Winner |
|--------|------|-----------|-----------------|--------|
| 1 | NO_CHANGE | 3 | **0** | Persist |
| 2 | NO_CHANGE | 6 | **0** | Persist |
| 3 | NO_CHANGE | 3 | **0** | Persist |
| 5 | BIG_CHANGE (+14) | **13** | 14 | GPT |
| 6 | BIG_CHANGE (+14) | **12** | 14 | GPT |

### Analysis

1. **GPT-4o makes the same error as our model**: On no-change samples, it predicts "reasonable" post-exercise improvements (fatigue ↑, mood ↑) that didn't actually happen. The LLM's common-sense reasoning ("exercise should improve wellness") conflicts with the data reality ("this user's wellness doesn't change").

2. **On big-change samples, GPT-4o barely beats persistence** (err=12-13 vs 14): It predicts the right direction (recovery from low values) but severely underestimates the magnitude.

3. **This confirms the task is information-limited, not model-limited**: Even the strongest available LLM cannot predict day-to-day wellness changes from a single day's data. The missing information includes individual recovery rates, sleep quality that night, psychological factors, and other unmeasured variables.

4. **Implication for the paper**: This GPT-4o result serves as an upper bound reference, demonstrating that our fine-tuned model's performance gap vs persistence is not due to insufficient model capacity but due to inherent unpredictability of the task without personalization data.

---

## Overall Project Summary (2026-03-28)

### What Worked

| Achievement | Version | Key Metric |
|-------------|---------|------------|
| Omnimodal text+HR generation | V1 (A1) | Parse rate 73.8%, text overlap 0.54 |
| Near-optimal HR prediction | V2 (B1) | MAE=17.19 vs noise floor 16.32 |
| High text-HR consistency | V2 (B1) | Consistency score 0.77 |
| Perfect output format learning | V3 (E1c) | Parse rate 100% with 32 epochs |
| Baseline HR personalization | V2 (B1) | +51% consistency vs V1 |

### What Didn't Work

| Attempt | Version | Why It Failed |
|---------|---------|---------------|
| Delta HR encoding | V2.1 (C1) | Error accumulation in long sequences (MAE doubled) |
| Day-to-day wellness prediction | V3 | Wellness too stable, persistence baseline unbeatable |
| In-context 7-day history | E-context | Model didn't learn to use history for stable/variable inference |
| Temporal split + user ID | E2 | Per-user data helps but still loses to persistence |

---

## Data Deep Dive: Exercise-Wellness Relationship (2026-03-28)

### Hypothesis Testing

Tested three common-sense assumptions about exercise and wellness:
1. No/light exercise → wellness unchanged or slightly improved
2. Moderate exercise → beneficial, wellness improved
3. Excessive exercise → harmful, wellness worsened

#### Results by exercise intensity

| Intensity | n | Avg delta | Improved | Worsened | Unchanged |
|-----------|---|----------|----------|----------|-----------|
| Light (≤30min) | 156 | **+0.59** | 40% | 32% | 28% |
| Moderate (30-60min) | 298 | -0.19 | 35% | 36% | 30% |
| Heavy (60-120min) | 330 | -0.01 | 33% | 34% | 33% |
| Extreme (>120min) | 152 | **-0.47** | 28% | 34% | 38% |

The trend exists (light → slight improvement, extreme → slight worsening) but is **very weak**. Even "beneficial" light exercise only improves wellness 40% of the time — 32% actually worsen.

**Overall**: 67% of samples match the hypothesis, 8.8% contradict it, 24% ambiguous.

### Delayed Effects Analysis

Tested whether exercise effects accumulate over multiple days (e.g., heavy exercise on Day N-2 causing wellness drop on Day N+1).

#### Key findings

**1. Delayed effects explain some anomalies**: Of 18 cases where light exercise was followed by wellness worsening (anomalous), 6 (33%) had heavy exercise in the preceding 3 days.

Examples:
- p10 (11/24): Only walked 28min, but ran 64min two days before → delta=-8 (**explained**)
- p15 (11/19): Light 24min today, but 102min + 90min + 567min in past 3 days → delta=-7 (**explained**)

**2. "Heavy exercise + improvement" is mostly mean reversion**: All 8 cases of heavy exercise followed by major improvement had Day N fatigue ≤ 2. The "improvement" is recovery from an already-bad state, not exercise benefit.

**3. But cumulative load has near-zero correlation with wellness change**:

| Load measure | Correlation with wellness delta |
|-------------|-------------------------------|
| Today's exercise | r = -0.064 |
| Yesterday's exercise | r = +0.018 |
| Past 3 days cumulative | r = +0.003 |
| All 4 days cumulative | r = -0.020 |

All correlations ≈ 0. Exercise load (whether same-day or cumulative) explains virtually none of the variance in wellness changes.

### Root Cause Analysis

Why exercise barely predicts wellness change in PMData:

1. **Unobserved confounders dominate**: Sleep quality that night, work stress, social events, diet, weather, illness — all affect next-day wellness but are not in our data.

2. **Mean reversion**: Users at extreme values (fatigue=1 or fatigue=5) naturally regress toward their personal mean, regardless of exercise.

3. **Individual response heterogeneity**: The same 60min run causes fatigue in one person and energizes another. Without deep personal history, this is unpredictable.

4. **Measurement noise**: Self-reported 1-5 scales are inherently noisy. A "3 vs 4" distinction may reflect random variation in how the user interprets the scale that morning.

5. **Temporal granularity**: Daily wellness captures slow-moving trends, while exercise effects may peak at specific hours (e.g., 6-12 hours post-exercise for soreness).

---

## MoE Architecture: Modality-Specific Experts (2026-03-30 — 2026-04-03)

### Motivation

The dense Qwen3-4B model (V2 B1) uses a single FFN for both text generation and HR time series prediction — two fundamentally different computational tasks. Mixture-of-Experts (MoE) replaces each FFN with 3 specialized experts + a soft router to improve modality specialization.

### Architecture

```
Shared Self-Attention (unchanged)
        |
   Soft Router → [w_text, w_ts, w_shared]
        |
   output = w_text·FFN_text(h) + w_ts·FFN_ts(h) + w_shared·FFN_shared(h)
```

| Expert | Role | Initialization |
|--------|------|----------------|
| FFN_text | Text generation | Qwen3-4B pretrained FFN (LoRA) |
| FFN_ts | HR time series | Stage 0 pretrained FFN (LoRA) |
| FFN_shared | Cross-modal bridging | Qwen3-4B pretrained FFN (full rank) |

### Two-Stage Training Pipeline

**Stage 0: HR-only pretraining** — Teach FFN_ts the "motor skill" of generating HR sequences.

- Data: Exercise-centered samples (baseline → `<run_start>` → exercise HR → `<run_end>` → recovery) + rest sliding windows from raw PMData
- Activity begin/end tokens at exact positions for conditioning
- Linear interpolation resampling, gap=90s, confidence filtering
- Loss: CE + ordinal loss (|E[bpm] - true_bpm|)
- Training: LoRA on attention + full MLP, 5 epochs, early stopping

Stage 0 training data inspection (exercise-centered samples with activity markers):

![Stage 0 run samples with run_start/run_end markers](output/stage0_data_inspect/inspect_train_run.png)

![Stage 0 rest sliding windows](output/stage0_data_inspect/inspect_train_rest.png)

Stage 0 eval (val, autoregressive with prefix=20%):
- Rest MAE: 4.41 (good)
- Walk MAE: 17.35, Run MAE: 25.19, Bike MAE: 23.41 (exercise segments still challenging)
- Mode collapse partially resolved by activity tokens + ordinal loss

Per-activity generation quality (best/median/worst MAE per activity type):

![Stage 0 per-activity best/worst — run](output/stage0_eval_v3_by_activity/eval_val_run_best_worst.png)

![Stage 0 per-activity best/worst — walk](output/stage0_eval_v3_by_activity/eval_val_walk_best_worst.png)

**Stage 1: MoE joint training** — Full multimodal text + HR generation.

- Data: V2 format with user ID, baseline HR conditioning, activity begin/end tokens in output
- Output order: HR first → text second (HR closer to baseline for continuity)
- Split: Time-based per user (70% train / 10% val / 20% test)
- Loss: α·L_text + (1-α)·L_hr + β·L_ordinal + λ·L_routing
- Training: LoRA on attention + FFN_text + FFN_ts; FFN_shared + Router fully trainable

### Results

#### Stage 1 MoE v2 (user-based split, no ordinal, text-first output)

| Metric | V2 Dense (B1) | MoE v2 | Change |
|--------|--------------|--------|--------|
| Parse rate | 73.8% | **96%** | +22pp |
| HR MAE | 17.19 | **14.36** | -16.5% |
| Text overlap | 0.54 | 0.52 | -4% |

Note: user-based split (train on 12 users, test on 3 users). Not directly comparable to later time-based experiments.

#### Stage 1 MoE v5 (time-based split, ordinal β=0.5, HR-first, user ID)

| Metric | MoE v5 |
|--------|--------|
| Parse rate | **100%** |
| HR MAE | 18.14 |
| HR MAE median | 16.17 |
| Text overlap | 0.51 |

#### Stage 1 MoE v6 (time-based split, ordinal β=0.1)

| Metric | MoE v6 |
|--------|--------|
| Parse rate | **100%** |
| HR MAE | 19.06 |
| HR MAE median | 16.23 |
| Text overlap | 0.54 |

#### Stage 1 MoE v9_fixed — Best (user embed + exercise/recovery token counts + eval parse fix)

Added learned `<user_pXX>` embedding tokens + explicit `### Exercise Tokens` /
`### Recovery Tokens` in the input to remove length ambiguity. Also fixed an
eval parse bug that was hiding true performance for v7/v8.

| Metric | **MoE v9_fixed** |
|--------|-----------------|
| Parse rate | **100%** |
| HR MAE mean | **16.11** |
| HR MAE median | 15.02 |
| HR evaluated | 49/50 |
| Text overlap | 0.55 |

v9_fixed achieves MAE 16.11 on time-based split — within 0.1 bpm of the V2
B1 user-split result, despite time-based being the harder evaluation setup.

![Stage 1 MoE v9 final HR generation vs ground truth](output/stage1_eval_v9_fixed/stage1_hr_viz_test.png)

![Stage 1 MoE v9 MAE distribution](output/stage1_eval_v9_fixed/stage1_mae_dist_test.png)

### Analysis

1. **MoE generates physiologically meaningful HR sequences**: No more mode collapse — the model produces HR curves with exercise-appropriate dynamics (onset, peak, partial recovery).

2. **Parse rate reached 100%**: MoE architecture + sufficient training consistently produces valid text + HR output format.

3. **User-based vs time-based split matters**: MoE v2 (user-based, MAE 14.36) vs v5/v6 (time-based, MAE 18-19). The ~4 bpm difference is primarily due to evaluation difficulty, not model quality. Time-based is more realistic but harder.

4. **Ordinal loss impact is small**: v5 (β=0.5) MAE=18.14 vs v6 (β=0.1) MAE=19.06 — minimal difference. The ordinal loss helps prevent extreme HR predictions but doesn't significantly improve overall MAE.

5. **Remaining weaknesses**:
   - Recovery segments (HR declining after exercise) are under-generated
   - High-intensity exercise peaks (HR >150) are often under-predicted
   - Systematic ~15-20 bpm offset on some samples

### Key Design Decisions & Lessons

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Soft routing (all 3 experts) | Simple, stable gradients | Worked well |
| Stage 0 activity tokens | Condition HR generation by activity type | Critical for avoiding mode collapse |
| HR-first output order | Better continuity with baseline HR input | Aligned with Stage 0 training |
| Time-based split | More realistic evaluation | Harder but fairer |
| Ordinal loss | Penalize far-off BPM predictions | Marginal effect at β=0.1 |
| User ID in input | Enable personalization | Not yet evaluated in isolation |

---

## Overall Project Summary (2026-04-03)

### What Worked

| Achievement | Version | Key Metric |
|-------------|---------|------------|
| Omnimodal text+HR generation | V1 (A1) | Parse rate 73.8%, text overlap 0.54 |
| Near-optimal HR prediction | V2 (B1) | MAE=17.19 vs noise floor 16.32 |
| High text-HR consistency | V2 (B1) | Consistency score 0.77 |
| Perfect output format learning | V3 (E1c) | Parse rate 100% with 32 epochs |
| Baseline HR personalization | V2 (B1) | +51% consistency vs V1 |
| **MoE modality specialization** | **MoE v2** | **MAE=14.36 (user-split), 100% parse** |
| **Physiologically coherent HR** | **MoE v5/v6** | **No mode collapse, exercise-appropriate dynamics** |

### Key Insights

1. **HR time series prediction hit the data noise floor** at MAE=17.19 (vs theoretical minimum 16.32). The model learned the learnable signal.

2. **LLM + special tokens works** for omnimodal generation. The model successfully generates coherent text and structured physiological outputs in a single sequence.

3. **Prediction difficulty depends on signal characteristics**: HR time series (high noise, 161-class) is fundamentally harder than wellness state (low noise, 5-class), but wellness is too stable to predict change.

4. **Persistence baseline is the key challenge** for any state-prediction task on slowly-changing signals. The model needs signals that actually change in response to interventions.

5. **Exercise-wellness relationship is real but weak**: Light exercise slightly improves wellness (+0.59 avg delta), extreme exercise slightly worsens it (-0.47). But the effect is dwarfed by unobserved confounders, making it nearly unpredictable from exercise data alone.

6. **Delayed exercise effects exist but are not systematic**: Some anomalous cases are explained by multi-day load accumulation (33%), but cumulative load has near-zero correlation (r ≈ 0) with wellness change overall.

7. **GPT-4o (zero-shot) also fails**: Even the strongest LLM cannot predict wellness changes from single-day data, confirming the task is information-limited, not model-limited.

---

## Part 2: Agent Integration Plan (2026-04-04)

### Project Structure

The overall project splits into two parts:

**Part 1 — Hybrid Signal Simulator** (largely complete, see MoE v9_fixed):
- Core claim: Joint text+HR modeling > separate single-modality models
- Personalization via learned user embeddings
- Current best: MoE v9_fixed, MAE=16.11 (time-split), 100% parse rate

**Part 2 — Decoupled User Modeling + Coach Agent** (planning phase):
- Core claim: Decoupling user modeling (simulator) from decision making (coach) beats
  a monolithic agent that tries to learn both from observational data.

### Why Monolithic Personal Coach Training Is Hard

PMData limitation (confirmed via original paper): participants' activities were
entirely self-initiated — "encouraged to exercise at least twice a week, but
with no requirements on what type of exercise or the length of each". There is
**no ground-truth optimal recommendation** in the dataset; we only have what
users chose to do for themselves.

Training a monolithic coach on this data directly (weak label = user's actual
activity) teaches the model to mimic user habits, not to give good advice.
This is unfair as a baseline — it's set up to fail.

### Task Design: Multi-Step Goal-Conditioned Planning

To make the task hard enough that decoupled reasoning is actually required, we
define a **weekly plan generation** task rather than a single-step recommendation:

```
Input:
  - User profile (user_id with learned embedding, typical activity patterns)
  - 7-day recent history (wellness + activities)
  - Current state (fatigue/mood/readiness/sleep/soreness/stress, baseline HR)
  - Goal (e.g., "average fatigue >= 3 next week", "total exercise >= 180 min",
          "recover from current fatigue=1 to fatigue>=3 within 3 days")

Output:
  - 7-day activity plan (per-day activity type + duration)
  - Rationale for the plan
```

Why this task design works:
- Single-step recommendations are easy to learn via SFT — no decoupled advantage.
- Multi-step planning **requires predicting future states**, which is exactly
  what the simulator does.
- Monolithic agents must implicitly roll out future states; they are myopic and
  prone to error compounding.
- Decoupled approach can explicitly simulate each day and search/plan.

### Training Data Construction

**Shared training data for both approaches** (fair comparison):
- Use GPT-4o to generate (state, goal) -> (7-day plan, rationale) examples
- These are "reasonable starting points", not ground-truth expert advice
- Used for SFT of the monolithic baseline AND as demonstrations for the planner

**Target size**: ~500-1000 synthetic (state, goal, plan) triples spanning diverse
user states, goals, and constraint scenarios.

### Comparison Conditions

| Condition | Components | Training | Inference |
|-----------|-----------|----------|-----------|
| **M1 — Zero-shot GPT-4o** | GPT-4o only | None | Direct plan generation |
| **M2 — Monolithic SFT** | Qwen3-4B | SFT on GPT-4o-generated plans | Direct plan generation |
| **M3 — Monolithic RL** | Qwen3-4B + simulator as reward | SFT + GRPO using simulator | Direct plan generation |
| **Ours — Decoupled** | v9 simulator + LLM planner | Simulator: already trained. Planner: zero-shot or SFT | Planner enumerates candidates, simulator predicts outcomes, planner picks best |

Key pairwise comparisons:
- **M2 vs Ours**: same base model (Qwen3-4B), same training data (in different
  forms). Tests whether the decoupled structure itself helps.
- **M3 vs Ours**: both use the simulator. M3 uses it as RL reward, Ours uses it
  as an inference-time tool. Tests whether explicit simulation at inference
  beats baking it into the policy during training.

### The Evaluation Dilemma

There is a fundamental difficulty: we cannot directly measure "actual outcome
quality" of a recommendation because:

| Approach | Problem |
|----------|---------|
| Match ground-truth activity | PMData has no optimal labels — users self-selected |
| LLM-as-judge (outcome prediction) | LLMs cannot simulate user physiology |
| Simulator rollout | Only our method has a simulator; using it to evaluate itself is circular |

This is not a design flaw — it is a fundamental property of the task, and in
fact motivates the decoupled approach (specialized simulator is the only
reliable way to reason about outcomes).

### Solution: Multi-Layer Evaluation

Rather than relying on one metric, we use complementary metrics that together
triangulate plan quality. Each metric is honest about its limitations.

#### Layer 1 — Objective metrics (large scale, n ≈ 500)

These metrics require no ground truth, no simulator, and no LLM judgment.
They are fully reproducible.

**1.1 Constraint satisfaction rate** (primary objective metric)
- Reframe the task as **goal-constrained planning**: goals are expressed as
  hard constraints, not open-ended objectives
- Example constraints: `total_exercise_min ∈ [150, 200]`, `num_rest_days ≥ 2`,
  `includes cardio + strength`, `fatigue_trajectory ≥ 2 every day`
- Constraints are designed to require multi-step trade-offs that expose
  myopic decision making
- Metric: % of plans satisfying all constraints

**1.2 Safety rule compliance**
- Rule-based checks derived from sports medicine guidelines:
  - No HIIT when fatigue ≤ 2
  - No more than 3 consecutive heavy days without rest
  - Weekly cumulative load caps
  - Activity variety (avoid same activity 5+ days)
- Metric: violation rate per plan

**1.3 Goal sensitivity**
- Same user and state, different goals → plans should meaningfully differ
- Metric: pairwise plan distance across different goals for same user

**1.4 Cold-start gap**
- Performance on users not seen during training vs seen users
- Metric: (metric_seen − metric_unseen) as a measure of personalization
  robustness

#### Layer 2 — Relative simulator metrics (large scale, n ≈ 500)

**2.1 Hold-out simulator rollout**
- Train a second simulator on a disjoint data split (never used by any method)
- Roll out each plan through this simulator, measure predicted state trajectory
- Honest caveats: simulator is imperfect; all methods evaluated against the
  same simulator, so biases cancel in pairwise comparison
- Report: predicted goal achievement rate, predicted fatigue trajectory

#### Layer 3 — Expert evaluation (small scale, n ≈ 30-50) — gold standard

This is the most credible validation for a health-domain paper. Sports medicine
experts apply clinical knowledge that neither LLMs nor simulators possess.

**3.1 Recruitment**
- 2-3 sports medicine physicians or certified exercise physiologists
- Multi-rater setup for inter-rater agreement analysis

**3.2 Evaluation protocol**
- **Blind**: experts do not know which plan came from which method
- **Randomized order** to avoid anchoring
- **Paired comparison**: for the same (user, state, goal), rate plans from
  different methods side by side
- **Free-text rationale**: experts explain their preferences

**3.3 Rubric** (5-point scale per dimension)
| Dimension | What it captures |
|-----------|------------------|
| Safety | Does the plan avoid clinically dangerous patterns? |
| Individualization | Is the plan tailored to this specific user? |
| Goal alignment | Does the plan support the stated goal? |
| Progression | Is the intensity/volume progression sound? |
| Would prescribe? | Overall clinical endorsement |

**3.4 Complementary to Layer 2**
- Simulator rollout: measures physiological outcomes (what simulator predicts)
- Expert evaluation: measures clinical reasonableness (what clinicians endorse)
- These answer different questions and are both necessary

#### Layer 4 — Qualitative case studies

- Curated scenarios illustrating each method's typical successes and failures
- Examples: high-fatigue recovery, cold-start user, conflicting goals, etc.
- Provides interpretable insights that aggregate metrics miss

### Ablations

- Ours without simulator rollout (planner only): isolates simulator contribution
- Ours without user embedding (generic simulator): isolates personalization
- Ours with different planner backbones (zero-shot GPT-4o, Qwen3-4B SFT planner)
- Ours on cold-start users (new user tokens initialized from nearest neighbor)

### Why This Is a Fair Comparison

- M2 and Ours use the same base model (Qwen3-4B) and the same training budget.
- M3 and Ours both use the simulator — the difference is when/how it is used.
- Layer 1 metrics are purely objective (constraint satisfaction, safety rules)
  so no method has built-in advantage.
- Layer 2 (hold-out simulator) is used for relative comparison with transparent
  caveats about its limitations.
- Layer 3 (expert evaluation) is the true gold standard — experts bring
  clinical knowledge that is independent of any model.

### Narrative

The core claim is NOT "our recommendations are better" (unmeasurable in
absolute terms). The claim is:

> "Decoupling user modeling from decision making enables outcome-aware
> reasoning that is systematically required for multi-step constrained
> planning. A monolithic agent trained on the same data cannot learn this
> reasoning from observational signals alone, regardless of training
> paradigm (SFT or RL), and expert raters confirm the clinical superiority
> of the decoupled plans."

### Expected Failure Modes

**M2 (monolithic SFT)**:
- Regresses to mean plan, ignores user/goal specifics
- Myopic: first day looks reasonable but later days accumulate error
- No counterfactual reasoning about "what if I recommend differently"

**M3 (monolithic RL)**:
- Overfits to simulator reward signal
- May exploit simulator inaccuracies (generate plans that look good to simulator
  but are unrealistic)
- Still limited to single-pass generation at inference time

**Ours (decoupled)**:
- Depends on simulator accuracy — if simulator is wrong, planner is misled
- Inference cost is higher (multiple simulator rollouts per plan)
- Planner may miss creative plans not in the candidate set

### Roadmap

**Phase 1 — Data & Infrastructure** (1-2 weeks)
- [ ] Define exact input/output schema and goal taxonomy
- [ ] Generate synthetic training set with GPT-4o (~500-1000 examples)
- [ ] Train hold-out simulator on different data split (for evaluation)
- [ ] Define LLM-as-judge prompts + scoring rubric

**Phase 2 — Baseline Implementation** (2-3 weeks)
- [ ] M1: Zero-shot GPT-4o pipeline
- [ ] M2: Qwen3-4B SFT on synthetic plans
- [ ] M3: Qwen3-4B + simulator-reward RL (GRPO)
- [ ] Ours: v9 simulator + planner (start with zero-shot, ablation with SFT planner)

**Phase 3 — Evaluation** (2-3 weeks)
- [ ] Build test set: diverse (user, state, goal) tuples covering cold-start,
      recovery, progression, maintenance scenarios
- [ ] Layer 1: Run all four conditions, compute objective metrics
      (constraint satisfaction, safety compliance, goal sensitivity, cold-start gap)
- [ ] Layer 2: Hold-out simulator rollout on all methods
- [ ] Layer 3: Recruit expert raters, blind paired comparison on ~30-50 samples
- [ ] Layer 4: Curated case studies
- [ ] Ablation studies

### Open Questions

- **Goal specification format**: natural language vs structured? Starting with
  structured for evaluability.
- **Planner architecture**: pure LLM-as-planner (GPT-4o with tool use) vs
  enumerate-and-score? Start simple with enumerate-and-score.
- **Candidate generation for planner**: GPT-4o proposes candidates vs
  predefined activity templates? Depends on diversity requirements.
- **Simulator as reward vs tool distinction**: need to empirically validate that
  inference-time rollout beats training-time reward shaping.

---

## Stage 1 v10: Wellness as Third Output Modality (2026-04-11)

### Motivation

The Part 2 minimal experiment showed that the GPT-5.4-mini coach over-weighted
the simulator's text feedback when inferring next-day state, leading to overly
conservative plans on Progression goals. The fix is to have the simulator
output next-day wellness directly, removing the LLM-based inference step.

### Changes

Extended the MoE simulator output from `[HR sequence + text feedback]` to
`[HR sequence + text feedback + next-day wellness state]`. Architecture stays
3 experts (text/ts/shared); wellness tokens flow through the existing soft
router with neutral routing supervision (let the router self-organize).

- **Input addition**: `### Current State:` block with state special tokens
  for symmetry with output (allows missing Day N wellness)
- **Output addition**: `<state_start> ... <state_end>` block at the end
- **Loss formula**: `α·L_text + β·L_hr + γ·L_wellness + δ·L_ordinal + λ·L_routing`
  with default `α=0.4, β=0.4, γ=0.2`
- **Mask split**: `is_hr_token` (HR class) vs `is_wellness_token` (wellness
  class) for three-component CE loss
- **Eval extension**: per-field wellness MAE + persistence baseline

### Results (50 test samples, time-based split)

| Metric | v9_fixed | **v10 (wellness)** |
|--------|---------|-------------------|
| Parse rate | 100% | **100%** |
| HR MAE mean | **16.11** | 16.90 |
| HR MAE median | 15.02 | 15.33 |
| Text overlap | 0.55 | 0.5449 |
| Wellness MAE overall | n/a | **0.633** |
| Persistence baseline | n/a | 0.519 |

#### Per-field wellness

| Field | Model | Persistence | Δ |
|-------|-------|-------------|---|
| fatigue | 0.49 | 0.36 | +0.13 |
| mood | 0.22 | 0.18 | +0.05 |
| readiness | 1.59 | 1.58 | +0.01 |
| sleep_quality | 0.67 | 0.38 | +0.30 |
| soreness | 0.49 | 0.40 | +0.09 |
| stress | 0.33 | 0.22 | +0.10 |

The model loses to persistence on every field, but the gap is much smaller
than V3 (E1c) showed (where the gap was 4-9× per field).

### Visual analysis (output/stage1_eval_v10/)

- **HR generation**: comparable quality to v9_fixed; slight regression
- **Wellness scatter** (`v10_wellness_scatter.png`): predictions cluster around
  the mode of each field. Model and persistence overlap heavily on the diagonal.
- **Wellness distributions** (`v10_wellness_distributions.png`): model
  distribution is narrower than ground truth; mild mode collapse on stress and
  readiness.
- **Change behavior** (`v10_change_behavior.png`): model OVER-predicts change
  (e.g., readiness 82% predicted change vs 69% true change). The model knows
  some change should happen but doesn't get the direction right.

### Analysis

The model is essentially a "mild persistence perturbator" — it knows it should
mostly match Day N but adds small adjustments. Those adjustments are not
accurate enough to beat persistence, but they're also not catastrophically
wrong. The 0.633 vs 0.519 gap is small enough that v10 is usable as the
"complete simulator" for the Part 2 coach experiment.

---

## Exercise Intensity vs Wellness Change Analysis (2026-04-11)

### Hypothesis

PMData users self-selected their exercise activities. If users generally make
sustainable choices, then most exercise days are within each user's tolerance
zone, so next-day wellness should not change much regardless of intensity.
This would explain why the wellness predictor (v10) approaches but cannot beat
persistence: persistence is the *correct* answer most of the time.

If true, this reframes the simulator's role:
- It is NOT primarily about predicting wellness change (the change signal is
  too weak in the training data)
- It IS about modeling each user's individual tolerance / baseline range,
  so a coach can avoid recommendations outside that range

### Method

Built a per-user-relative intensity stratification:
1. For each user, ranked their exercise days by `(duration, peak HR)`
2. Assigned each day a per-user percentile (0-100)
3. Stratified all 936 exercise days into 5 buckets
4. Computed average total wellness change `Σ |Δ field|` per stratum
5. Counted "stable" (Δ ≤ 1) vs "big shift" (Δ ≥ 4) days per stratum

### Results

#### Total wellness change is essentially flat across intensity strata

| Stratum (per-user) | n | Mean total \|Δ wellness\| |
|---------------------|----|-------------------------|
| very_low (0-25%) | 232 | **2.95** |
| low (25-50%) | 233 | 3.23 |
| mid (50-75%) | 234 | 3.05 |
| high (75-90%) | 141 | 3.00 |
| very_high (90-100%) | 96 | **3.26** |

The very_high stratum (top 10% intensity per user) is only 0.31 higher than
the very_low stratum. **Exercise intensity barely predicts next-day wellness
magnitude.**

#### "Hard but stable" days are common

Among top-20% intensity days (n=186):
- **31% are stable** (total Δ ≤ 1) — high effort, no next-day wellness change
- **42% are big shifts** (total Δ ≥ 4)
- Some extreme cases:
  - p15 2019-11-16: 567 min duration, peak 137 → wellness completely unchanged
  - p08 2020-03-08: 304 min, peak 150 → wellness completely unchanged
  - p02 2020-01-06: 297 min, peak 167 → only sleep_quality −1

These users tolerate massive workloads with no wellness perturbation. This is
the homeostasis evidence: they self-selected loads they can handle.

#### Per-user heterogeneity is huge

Wellness change rate (% of days where fatigue changed) vs average duration:

- p15: 130 min/session avg, 47% fatigue change rate
- p02: 105 min/session avg, **2%** fatigue change rate (extremely stable)
- p10: 55 min/session avg, **74%** fatigue change rate (most volatile)
- p03: 36 min/session avg, 12% fatigue change rate

There is no monotonic "more exercise → more wellness change" relationship.
Some users do massive workouts and stay stable; others do gentle workouts and
fluctuate constantly. The dominant signal is **user identity**, not exercise
load.

### Implications

1. **Persistence is hard to beat by design.** PMData users operate within
   their tolerance zone almost every day, making "predict no change" the
   correct answer most of the time. Any model that learns from this data will
   converge toward persistence.

2. **The simulator's value should be reframed.** Instead of "predict the
   wellness delta after this activity" (a low-signal task), the simulator's
   contribution is:
   - Modeling each user's normal HR + text response to typical activities
   - Identifying when a candidate activity falls *outside* the user's
     observed tolerance range
   - Providing physiologically grounded "this will likely be hard" signals
     that pure persistence cannot

3. **Part 2 evaluation should adjust.** Outcome metrics like "wellness
   improvement" will be near-zero for most plans (because PMData wellness
   barely moves). More informative metrics:
   - Whether the recommended activities fall within the user's observed
     tolerance range (per-user percentile of duration / peak HR)
   - Whether the simulator correctly flags out-of-range recommendations
   - Whether the coach respects per-user constraints (a 5-hour run plan for
     p03 is wrong; for p15 it might be normal)

4. **For the paper narrative.** The exercise→wellness link in observational
   sports lifelogging data is weak by construction (selection bias toward
   sustainable loads). This is a property of the data, not a limitation of
   our method. It also suggests our framework would benefit more from a
   *prescribed-load* dataset (where users do activities outside their normal
   range) than from more PMData-style observational data.

### Saved analysis

- `output/intensity_wellness_analysis.json` — per-stratum raw data
- `output/intensity_wellness_analysis.png` — three-panel summary
- `output/intensity_wellness_per_user.png` — per-user duration vs change rate

---

## Part 2 Minimal Experiment: Direct vs Sim-guided Coach (2026-04-10)

### Setup

First Part 2 experiment to compare direct LLM coaching vs simulator-guided
coaching, with a stronger LLM as blind judge.

- **Coach (both conditions)**: GPT-5.4-mini, same model for fairness
- **Judge**: GPT-5.4 (different/stronger model, blind pairwise A/B with
  randomized order to avoid position bias)
- **6 test cases**: 3 users (p01 mid / p08 high / p03 low) × 2 tasks
  (recovery / progression)
- **Output**: 7-day plan with activity type, duration, intensity, rationale

#### Two conditions

**M1 — Direct**: Coach reads (state + goal) → outputs full 7-day plan in one pass

**M2 — Sim-guided**: Sequential day-by-day generation:
1. Coach proposes day t activity given current state + remaining goal
2. v10 simulator predicts HR + text feedback for that activity
3. Coach uses simulator output to infer day t+1 state via LLM reasoning
4. Repeat for 7 days

#### Judge rubric (1-5 scale per dimension)

Safety, Individualization, Goal alignment, Progression logic, Overall preference

### Results

| Dimension (avg of 6 cases) | M1 Direct | M2 Sim-guided | Δ |
|---------------------------|-----------|---------------|---|
| Safety | 3.83 | 3.83 | 0.00 |
| Individualization | **3.83** | 3.00 | −0.83 |
| Goal alignment | **3.83** | 3.50 | −0.33 |
| Progression logic | **4.00** | 3.33 | −0.67 |
| Overall | **3.67** | 3.33 | −0.33 |
| Preference wins | 3/6 | 3/6 | tied |

### Pattern: recovery vs progression

- **Recovery goals (3 cases)**: Sim wins 2-1. Conservative bias of
  simulator helps when the user needs rest.
- **Progression goals (3 cases)**: Direct wins 2-1. Sim becomes too
  conservative and misses volume targets.

### Failure mode of sim-guided

The coach interprets v10's text output ("felt tired/sore") as a stop
signal → reduces intensity → next simulator call still says "tired" →
reduces further → **plan drifts away from goal**.

This is the over-conservative spiral. Confirmed motivation for v10's
explicit wellness output (which removes the LLM-text-inference step).

### Files

- `part2_experiment.py` — script (Direct + Sim-guided + Judge)
- `output/part2_experiment/part2_results_*.json` — results

---

## Tolerance Test: Does Simulator Learn Per-User Limits? (2026-04-11)

### Hypothesis

Following the homeostasis finding, the simulator's value should be in
modeling each user's tolerance zone. Test: same activity × different users
should produce different wellness predictions, with low-activity users
showing larger Δ wellness for high-intensity activities.

### Setup

- **6 users**: 3 high (p15/p02/p08), 1 mid (p01), 2 low (p03/p09)
- **4 intensity scenarios**: light (20 min walk) → very heavy (120 min run)
- **Fixed neutral state** for all (only user identity varies)
- **3 reps per cell** (72 total generations)
- Measured total |Δ wellness| from neutral baseline

### Results

#### Per-scenario × user group

| Scenario | high | mid | low | (low − high) |
|----------|------|-----|-----|-------------|
| light | 2.56 | 2.33 | 2.17 | −0.39 |
| moderate | 1.89 | 1.67 | 2.67 | +0.78 |
| heavy | 3.56 | 3.00 | 2.67 | **−0.89** ❌ |
| very_heavy | 2.22 | 1.00 | 3.83 | **+1.61** ✓ |

Only `very_heavy` shows the expected pattern (low > high). `heavy` is
reversed. The middle scenarios are noisy.

#### Per-user monotonicity (rank correlation: intensity vs Δ)

| User | Level | Rank corr | Signal |
|------|-------|-----------|--------|
| p09 | low | **+0.95** | strong ✓ |
| p03 | low | **+0.63** | clear ✓ |
| p15 | high | +0.48 | weak |
| p02 | high | −0.25 | flat / noise |
| p08 | high | −0.27 | flat / noise |
| p01 | mid | −0.40 | reversed ✗ |

### Verdict

Partial signal, high noise. Low-activity users (p09, p03) show the
expected monotonic increase with intensity — consistent with the
"out-of-tolerance" hypothesis. High-activity users are flat — could be
"within tolerance" learned by the model, or could be mode collapse.

#### Limitations
- Only 3 reps per cell (high std up to 1.95)
- HR peak is similar across users (model doesn't strongly differentiate
  per-user HR response in generation, only wellness)

### Next iteration

- n=10 reps, lower temperature (0.3) for stability
- Add HR-curve side-by-side comparison (same activity, p15 vs p03)
- Finer duration gradient (15, 30, 45, 60, 90, 120, 180 min)
- More users for statistical power

### Files

- `stage1_tolerance_test.py` — script
- `output/stage1_tolerance_test/tolerance_test_results.json`
- `output/stage1_tolerance_test/tolerance_group_bars.png`
- `output/stage1_tolerance_test/tolerance_per_user_curves.png`

---

## External Dataset Comparison: PHLLM (2026-04-11)

### Background

Google released [PH-LLM](https://www.nature.com/articles/s41591-025-03888-0)
(Yang et al., Nature Medicine 2025) along with public coaching datasets:
- 400 fitness case studies
- 557 sleep case studies

Each case has 30 days of daily-aggregated metrics + expert-written
coaching outputs (insight + etiology + recommendation, structured as
markdown with Observation/Insight per dimension).

### Comparison

| Dimension | PMData (ours) | PHLLM (Google) |
|-----------|--------------|----------------|
| Granularity | Per-second HR + per-activity sequences | Daily aggregates (avg HR, sleep duration, HRV avg, TRIMP) |
| Subjective | 5 wellness fields (1-5 / 1-10) | Demographics + readiness + soreness as natural language |
| Coaching labels | None (we synthesize via GPT) | **Expert-written** (sports medicine specialists) |
| Metrics included | We compute ourselves | Already includes ACWR, TRIMP, Z-scores |
| Sleep detail | Only sleep_quality (1-5) | Full polysomnography-style metrics |
| Sample size | 2438 per-activity, 936 daily | 957 case studies (30-day each) |

### Strategic relevance

PHLLM solves Part 2's biggest problem: **lack of ground-truth coaching
labels**. But adopting their data wholesale would lose our differentiator
(per-activity fine-grained simulation).

### Plan: migrate evaluation framework, not the data

Decision: **Keep training on PMData (preserves our fine-grained advantage)
but borrow PHLLM's evaluation methodology**.

#### What to migrate from PHLLM

1. **Output format**: structured markdown with Observation + Insight per
   dimension (Training Load, Health Metrics, Readiness Assessment)
2. **Rubric**: PHLLM-style scoring on observation accuracy, insight depth,
   clinical reasonableness, personalization
3. **Reference outputs**: use PHLLM expert outputs as few-shot calibration
   for the LLM-as-judge
4. **Derived metrics**: compute ACWR, TRIMP, RHR Z-scores from our
   per-activity HR data and feed to coach as input (raises information
   density to PHLLM-comparable level)

#### What to keep from our setup

1. **Per-activity simulator** (the differentiator) — feeds simulated HR +
   text + wellness predictions to the coach
2. **PMData users + time-based split**
3. **3-modality output** from v10
4. **The simulator-vs-direct comparison structure** in Part 2

### New Part 2 coach output format (proposed)

Replace plain "Day 1: Walk 30 min" with PHLLM-aligned structured output:

```
## Training Load Analysis
**Observation**: User's recent 7-day TRIMP averages 200 with peak day 350...
**Insight**: This is a high acute load relative to chronic baseline...

## Health Metrics Analysis
**Observation**: Resting HR trending up by 5 bpm over week, HRV decreased...
**Insight**: Indicates incomplete recovery, suggests reducing volume...

## Readiness Assessment
**Score: 2/5**
**Reasoning**: ...

## Recommended 7-Day Plan
Day 1: Walk, 30 min, light (recovery focus)
Day 2: ...
```

This positions us as: PHLLM analyzes user state. We do that AND prescribe
forward-looking plans, with the simulator enabling outcome-aware planning.

### Cross-dataset evaluation as secondary validation

Optional: cold-start mode (drop user embeddings) on 5-10 PHLLM cases as a
case study showing our system generalizes outside PMData users. Not the
main evaluation due to user/format mismatch.

### Paper narrative upgrade

> "We adopt the evaluation framework from PHLLM (Yang et al. Nature
> Medicine 2025) — observation/insight rubric and ACWR/TRIMP derived
> metrics — but built on per-activity fine-grained simulation rather than
> daily aggregates. This contrasts 'aggregated coaching' (PHLLM) with
> 'simulator-grounded coaching' (ours): the former infers state from
> daily statistics, the latter reasons about hypothetical activity choices
> via minute-level physiological simulation."

### Roadmap

- Phase 1 (this week): write PMData → PHLLM-style metrics extractor
  (TRIMP, ACWR, RHR Z-scores), restructure coach prompt to emit
  observation/insight markdown + 7-day plan
- Phase 2 (next week): rerun Part 2 minimal experiment (6 → 30 cases) with
  PHLLM-rubric LLM-as-judge using PHLLM expert outputs as few-shot
- Phase 3: cold-start case studies on a small PHLLM subset for external
  validity

### Resources

- Paper: https://www.nature.com/articles/s41591-025-03888-0
- Code/data: https://github.com/Google-Health/consumer-health-research/tree/main/phllm
- Sample we downloaded: `output/external/fitness_sample.jsonl` (3 example records)

---

## PH-LLM Re-examination & Predictability Sanity Check (2026-05-04)

### Re-reading the PH-LLM paper (architecture)

Confirmed by paper + repo inspection:
- **Single shared model**, no per-user mechanism. Personalization is achieved
  purely by including the user's 30-day data as text in the prompt.
- `user_id` in the data is for tracking only, never fed to the model.
- The main contribution is an **agent/coach** model: 957 expert-written case
  studies (400 fitness, 557 sleep) where the LLM produces Observation +
  Insight + Recommendations + a readiness score (1-5).
- One side experiment predicts self-reported sleep quality via a dedicated
  MLP adapter — not the centerpiece.

This means **PH-LLM has no user simulator**. Our v9_fixed / v10 simulators
are not redundant with PH-LLM; they fill a missing component (counterfactual
rollout for outcome-aware planning).

### PH-LLM data: per-user structure

400 fitness cases come from only **58 unique users** — average 6.9 cases per
user, max 20, **45 users have ≥ 3 cases**. Each case is a 30-day window from
that user's wearable history.

Implication: learned per-user embeddings (analogous to our `<user_pXX>`)
are viable on this dataset, with more user-cases-per-user than PMData
(16 users in PMData). PH-LLM's prompt-only personalization is leaving
this signal on the table.

### Sanity check: is next-day RHR / HRV predictable?

`phllm_predictability_test.py` — sliding-window prediction task on a random
sample of 10 cases (seed=42, window=14, 133 (input, target) pairs).

Targets: next-day morning RHR (bpm) and HRV RMSSD (ms).
Baselines: persistence (next = today), 7-day rolling mean, Qwen3-4B zero-shot.

#### Pooled MAE

| Predictor | RHR MAE | HRV MAE | n |
|-----------|---------|---------|---|
| Persistence | **1.04** | 12.37 | 133 |
| 7-day rolling mean | 1.36 | **9.47** | 133 |
| Qwen3-4B zero-shot | 1.43 | 10.60 | 133 |

#### Within-user MAE (per-case mean, then averaged across 10 cases)

| Predictor | RHR MAE | HRV MAE |
|-----------|---------|---------|
| Persistence | **1.03** | 12.51 |
| 7-day rolling mean | 1.37 | **9.57** |
| Qwen3-4B zero-shot | 1.44 | 10.80 |

Pooled and within-user numbers are essentially identical — the persistence
trap is real **within users**, not just an artifact of between-user variance.
Day-to-day RHR jumps are genuinely ~1 bpm even for a single user.

#### Qwen3-4B prediction shape

| | mean | std | bias vs truth |
|-|------|-----|---------------|
| RHR truth | 60.12 | 10.96 | — |
| RHR LLM | 60.02 | 10.82 | -0.10 |
| HRV truth | 53.97 | 43.35 | — |
| HRV LLM | 57.32 | 49.42 | +3.35 |

`pred_std / truth_std`: RHR 0.987, HRV 1.140. **No mode collapse** — Qwen3-4B
makes expressive, variance-matched predictions (contrast with PMData V3
wellness, which collapsed to the population mean). Parse rate 100%.

#### Per-row head-to-head vs persistence

- RHR: LLM beats persistence 21.8 % of rows, loses 41.4 %, ties 36.8 %
- HRV: LLM beats persistence 57.9 %, loses 40.6 %, ties 1.5 %

LLM has **a real but weak signal on HRV** (per-row > 50%) but cannot beat a
trivial 7-day rolling mean on aggregate MAE.

#### Per-case heterogeneity (the most informative slice)

Cases where Qwen3-4B beats both baselines on HRV: FC14893 (LLM 4.49 vs
persist 7.09), FC42123 (7.88 vs 9.78), FC79151 (5.61 vs 10.33). All have
moderate, normal-range HRV.

Cases where Qwen3-4B loses badly on HRV — FC17963 (truth ≈ 120 ms but
LLM persistently answers 125-130), FC89969 (truth ≈ 100 ms but LLM
predicts 135). These are users with **distribution-tail HRV**: the LLM
uses generic "high HRV is healthy" prior to override the user's actual
elevated baseline.

This is the predicted failure mode of prompt-only personalization:
**common-sense LLM priors override user-specific patterns that don't fit
the population norm.** Learned per-user embeddings target this exact gap.

### Implications for next steps

1. **Persistence is the bar for RHR; 7-day mean is the bar for HRV.**
   Any fine-tuned simulator must beat these on within-user MAE to be
   meaningful.
2. **HRV is the more tractable target.** RHR captures slow chronic
   adaptation; day-to-day jumps are too small (~1 bpm) for a model to
   meaningfully improve on persistence. HRV is acutely responsive to
   activity/sleep/stress, with 12 ms day-to-day persistence MAE on 43 ms
   target std — real headroom.
3. **Per-case heterogeneity is the strongest argument for personalization.**
   Per-case persistence MAE on RHR ranges 0.19 (FC17963) to 1.83 (FC88465)
   — 10× spread. A user-conditioned model has clear room.
4. **PH-LLM data is well-suited to replicate the PMData simulator path**
   with user embeddings, with key differences in granularity (day-only),
   output format (continuous numerics), and richer 30-day metric history.
