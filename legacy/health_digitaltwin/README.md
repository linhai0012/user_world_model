# LLM-Based User Digital Twin

An omnimodal LLM that simultaneously generates natural-language user feedback
**and** physiological time series (heart rate) in response to activity events.

## Idea

Current LLM-based user simulators only produce text. Meanwhile, transformers
have shown strong time-series forecasting ability. We combine both into a
single model with a unified token space — text tokens and quantized heart-rate
tokens are generated autoregressively in one sequence, creating a lightweight
"digital twin" that can power personal health agents.

## Architecture

```
                         Qwen3-4B (frozen vocab + 165 new tokens)
                         ┌─────────────────────────────────────┐
                         │  Standard Transformer Decoder        │
  "User did a 30min run" │  (causal attention, next-token pred) │
  ──────────────────────▶│                                     │──▶ Output sequence
                         │  Extended vocab:                     │
                         │    151,936 original text tokens      │
                         │  +       4 structural tokens         │
                         │  +     161 HR tokens (<hr_40>..<hr_200>)
                         └─────────────────────────────────────┘

  Output sequence (Paradigm A — Sequential):
  ┌─────────────────────────────────────────────────────────────────────┐
  │ <text_start> Feeling tired, legs are heavy ... <text_end>          │
  │ <ts_start> <hr_142> <hr_138> <hr_130> ... <hr_88> <hr_82> <ts_end>│
  └─────────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
digital_twin_pipeline/
│
│── config.py                    # Shared constants: HR bins, special tokens, paths
│── train_config.py              # Training hyperparameters (dataclass)
│
│── step1_parse_pmdata.py        # Step 1: Parse PMData → structured records
│── step2_synthesize_text.py     # Step 2: GPT-5.4 synthesizes event desc + user feedback
│── step3_build_training_data.py # Step 3: Quantize HR, format omnimodal sequences, split
│── run_pipeline.py              # Orchestrator for steps 1-3
│
│── model_setup.py               # Load Qwen3-4B, extend vocab, HR embedding init
│── dataset.py                   # PyTorch Dataset + Collator for causal LM training
│── trainer.py                   # Custom Trainer with α-weighted text/HR loss
│── metrics.py                   # Evaluation: HR accuracy, text quality, consistency
│── train.py                     # Main training entry point
│── generate_and_eval.py         # Post-training generation + full evaluation
│
│── accelerate_config.yaml       # 4×H100 multi-GPU config
└── README.md
```

## Pipeline Overview

```
 Phase 1: Data Construction               Phase 2: Training              Phase 3: Evaluation
 ─────────────────────────                 ────────────────               ──────────────────

 PMData (raw Fitbit exports)               Qwen3-4B                      Trained model
 ┌──────────────────┐                      ┌──────────────┐              ┌──────────────┐
 │ heart_rate.json   │   Step 1            │ Extend vocab │   train.py   │ generate     │
 │ exercise.json     │──parse──▶ records   │ Init HR embed│──────────▶  │ + evaluate   │
 │ wellness.csv      │          │          │ Weighted loss│              │ per-sample   │
 └──────────────────┘          │          └──────────────┘              └──────┬───────┘
                               ▼                                               │
                         GPT-5.4 API        train.jsonl ──────────▶ train      │
                         ┌──────────┐       val.jsonl   ──────────▶ eval       ▼
                  Step 2 │ Synthesize│       test.jsonl  ──────────────▶  eval_results.json
                         │ text pairs│
                         └────┬─────┘
                              │
                   Step 3: quantize HR + format + split
                              │
                              ▼
                     output/train.jsonl
                     output/val.jsonl
                     output/test.jsonl
                     output/special_tokens.json
```

## Quick Start

### 0. Install dependencies

```bash
pip install openai numpy                          # for data pipeline
pip install torch transformers accelerate tensorboard  # for training
pip install flash-attn --no-build-isolation        # optional, for speed
```

### 1. Download PMData

```bash
wget https://datasets.simula.no/downloads/pmdata.zip
unzip pmdata.zip -d ./pmdata
```

Expected structure:
```
pmdata/
  p01/
    Fitbit/
      heart_rate-YYYY-MM-DD.json
      exercise-YYYY-MM-DD.json
    PMSys/
      wellness.csv
      srpe.csv
  p02/ ... p16/
```

### 2. Set environment variables

```bash
export OPENAI_API_KEY="sk-..."
export PMDATA_ROOT="./pmdata"
```

### 3. Build training data

```bash
# Full pipeline (parse → synthesize via GPT-5.4 → build omnimodal data)
python run_pipeline.py --all

# Or step by step
python run_pipeline.py --step 1     # parse PMData
python run_pipeline.py --step 2     # synthesize text (~$3 API cost)
python run_pipeline.py --step 3     # quantize HR + format + split

# Quick test with 20 samples
python run_pipeline.py --all --limit 20
```

### 4. Train

```bash
# Full training on 4×H100
accelerate launch --config_file accelerate_config.yaml train.py

# Quick debug on 1 GPU
python train.py --debug
```

### 5. Evaluate

```bash
python generate_and_eval.py \
    --model_path ./checkpoints/final \
    --test_file ./output/test.jsonl
```

## Key Design Decisions

### Model: Qwen3-4B (text-only, thinking disabled)

We use the base Qwen3-4B with `enable_thinking=False`. No vision encoder
overhead, no unnecessary `<think>` blocks. The model treats HR tokens as a
"foreign language" in the same vocab space — no separate encoder/decoder needed.

### Heart Rate Quantization

HR is quantized to 1-bpm bins → 161 discrete tokens (`<hr_40>` to `<hr_200>`).
At 1-minute sampling, a 60-minute window = 60 HR tokens. The total sequence
(text + HR) fits comfortably within 768 tokens.

### HR Embedding Initialization

New HR token embeddings are not randomly initialized. Instead, they are placed
along a linear gradient in the original embedding space:

```
<hr_40>  →  mean - 0.25·std
<hr_120> →  mean
<hr_200> →  mean + 0.25·std
```

This encodes the ordinal structure of heart rate (adjacent bpm values start
close together) and accelerates convergence.

### Weighted Loss

The cross-entropy loss is split by modality:

```
L = α · L_text + (1 - α) · L_hr
```

Both `loss_text` and `loss_hr` are logged separately to TensorBoard.
Set `--loss_alpha -1` for uniform CE (baseline).

### Data Split

Train/val/test is split **by participant** (not by sample) to prevent
data leakage. All sessions from the same person stay in the same split.

## Training Configuration

| Parameter                  | Default   | Notes                                 |
|---------------------------|-----------|---------------------------------------|
| Model                     | Qwen3-4B  | ~4B params, text-only                 |
| Loss α                    | 0.5       | Equal weight to text and HR           |
| Epochs                    | 8         | Monitor val loss for early stopping   |
| Learning rate             | 2e-5      | Full fine-tune range                  |
| Effective batch size      | 32        | 2 per GPU × 4 accum × 4 GPUs         |
| Max sequence length       | 768       | Covers text (~100) + HR (~120) tokens |
| Precision                 | bf16      | Native on H100                        |
| Gradient checkpointing    | On        | Saves memory                          |
| HR embedding init         | linear    | Ordinal-aware initialization          |

## Evaluation Metrics

### Text Quality
- **Perplexity**: on text tokens during eval
- **Token accuracy**: next-token prediction accuracy on text
- **LLM-as-Judge** (manual): GPT-5.4 rates naturalness, physiological plausibility

### HR Accuracy
- **MAE / RMSE**: point-wise error vs ground truth HR sequence
- **Trend accuracy**: fraction of step-to-step directions (up/down/flat) matching
- **Range error**: |predicted HR range - true HR range|
- **Token accuracy**: exact-match rate on HR tokens

### Cross-Modal Consistency
- **Rule-based checker**: "intense" language ↔ high HR, "relaxed" ↔ low HR, etc.
- **LLM-as-Judge** (manual): GPT-5.4 evaluates if text and HR tell the same story

### Ablation Experiments

| Experiment       | Config                          | Purpose                         |
|-----------------|---------------------------------|---------------------------------|
| Omnimodal α=0.5 | `--loss_alpha 0.5`             | Main model                      |
| Omnimodal α=0.3 | `--loss_alpha 0.3`             | HR-weighted variant             |
| Uniform CE      | `--loss_alpha -1`              | No modality weighting baseline  |
| Random HR init  | `--hr_embed_init random`       | Test embedding init impact      |
| Text-only       | Remove HR tokens from data      | Text baseline                   |
| HR-only         | Remove text tokens from data    | HR baseline                     |

## Output Files

```
output/
  train.jsonl                 # Training data (JSONL, ~80% by participant)
  val.jsonl                   # Validation data (~10%)
  test.jsonl                  # Test data (~10%)
  special_tokens.json         # 165 special tokens to add to tokenizer
  dataset_stats.json          # Data statistics
  eval_results.json           # Post-training evaluation report
  cache/
    parsed_records.json       # Intermediate: parsed PMData
    synthesized_records.json  # Intermediate: + GPT-generated text

checkpoints/
  final/                      # Best model + tokenizer
  checkpoint-*/               # Periodic checkpoints
```

## Training Sample Format

Each line in `train.jsonl`:

```json
{
  "input_text": "### Event:\nThe user completed a 35-minute outdoor run ...",
  "output_text": "<text_start> Legs feel heavy but mentally refreshed ... <text_end> <ts_start> <hr_142> <hr_145> ... <hr_82> <ts_end>",
  "full_sequence": "### Event:\n...\n\n### Response:\n<text_start> ... <ts_end>",
  "raw_hr": [142, 145, 138, ..., 88, 82],
  "participant_id": "p03",
  "activity_name": "Run",
  "duration_min": 35.2,
  "hr_seq_len": 70
}
```

## Cost Estimates

| Item                | Cost        |
|---------------------|-------------|
| GPT-5.4 data synthesis (~1,500 samples) | ~$3 |
| Training on 4×H100 (~20 min)           | minimal  |
| Full ablation suite (6 runs)            | ~2 hours |

## Citation

If you use this code, please cite:

```bibtex
@misc{digitaltwin2026,
  title={LLM-Based Hybrid User Simulator with Dual-Modal Generation},
  year={2026},
}
```

## License

Apache 2.0 (code). PMData is CC BY 4.0. Qwen3 is Apache 2.0.
