"""Sanity check #1 for Teacher SFT (plan §3.5): base vs SFT perplexity.

Compare mean NLL on the SAME user-token positions (target session only)
between the stock Qwen3-4B base model and the SFT checkpoint. A meaningful
drop confirms the teacher is actually leveraging progressive context.

Single-GPU, no FSDP needed (Qwen3-4B bf16 ~ 8 GB weights; 98 k context with
liger fused CE fits easily on one H100 96 GB).

Usage (on server):
    python dynamic_usersim/teacher_sft/eval_ppl.py \
        --sft-checkpoint $SCRATCHDIR/P-OPSD/teacher_sft_128k/checkpoint-50 \
        --num-samples 20

The samples are drawn from the same training JSONL; this measures training-
set fitting (not generalization). For a mid-training sanity check that is
the right metric.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

_HERE = Path(__file__).resolve().parent
_DP = _HERE.parent / "data_prep"
for p in (_HERE, _DP):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from config import DEFAULT_TRAIN_JSONL, MAX_SEQ_LEN, MODEL_NAME  # noqa: E402
from tokenize_teacher_sft import SFTTokenizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default=MODEL_NAME)
    ap.add_argument("--sft-checkpoint", type=Path, required=True,
                    help="Path to a checkpoint-N dir from training.")
    ap.add_argument("--train-jsonl", type=Path, default=DEFAULT_TRAIN_JSONL)
    ap.add_argument("--num-samples", type=int, default=20)
    ap.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-json", type=Path, default=None,
                    help="Optional: dump per-sample + aggregate results.")
    return ap.parse_args()


def load_samples(path: Path) -> list[dict]:
    out = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            out.append(json.loads(line))
    return out


def apply_liger() -> None:
    """Match training numerics by patching Qwen3 with liger fused kernels."""
    try:
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3
        apply_liger_kernel_to_qwen3(
            rope=True, swiglu=True, cross_entropy=False,
            fused_linear_cross_entropy=True, rms_norm=True,
        )
        print("[eval] liger applied")
    except ImportError:
        print("[eval] liger not installed — falling back (may OOM at long ctx)")


def load_model(path: str | Path) -> torch.nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        str(path),
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.eval()
    model.to("cuda")
    return model


def nll_on_sample(model: torch.nn.Module, tok: SFTTokenizer,
                  sample: dict) -> tuple[float, int, int]:
    """Return (total_nll, n_loss_tokens, total_seq_len) for one sample.

    model.forward(input_ids, labels) returns mean NLL over non-(-100) positions.
    We multiply back by token count to accumulate correctly across samples.
    """
    input_ids, labels, _ = tok.tokenize_sample(sample)
    ids_t = torch.tensor([input_ids], dtype=torch.long, device="cuda")
    lab_t = torch.tensor([labels], dtype=torch.long, device="cuda")

    with torch.inference_mode():
        out = model(input_ids=ids_t, labels=lab_t)
    loss_mean = out.loss.item()
    n_loss = (lab_t != -100).sum().item()
    return loss_mean * n_loss, n_loss, ids_t.size(1)


def evaluate_model(label: str, model: torch.nn.Module, tok: SFTTokenizer,
                   samples: list[dict]) -> dict:
    per_sample: list[dict] = []
    total_nll = 0.0
    total_tokens = 0
    t0 = time.time()
    for i, s in enumerate(samples):
        t_s = time.time()
        nll, n_loss, seq_len = nll_on_sample(model, tok, s)
        mean_nll = nll / max(n_loss, 1)
        per_sample.append({
            "persona_id": s["persona_id"],
            "context_id": s["context_id"][:8],
            "session_idx": s["session_idx"],
            "seq_len": seq_len,
            "loss_tokens": n_loss,
            "mean_nll": mean_nll,
            "ppl": float(torch.exp(torch.tensor(mean_nll)).item()),
            "wall_s": time.time() - t_s,
        })
        total_nll += nll
        total_tokens += n_loss
        print(f"  [{label}] sample {i+1}/{len(samples)} "
              f"seq_len={seq_len} loss_tokens={n_loss} "
              f"mean_nll={mean_nll:.4f} ppl={per_sample[-1]['ppl']:.3f} "
              f"({time.time()-t_s:.1f}s)")
    mean_nll = total_nll / max(total_tokens, 1)
    ppl = float(torch.exp(torch.tensor(mean_nll)).item())
    wall = time.time() - t0
    print(f"[{label}] aggregate  total_loss_tokens={total_tokens}  "
          f"mean_nll={mean_nll:.4f}  ppl={ppl:.3f}  ({wall:.1f}s)")
    return {
        "label": label,
        "per_sample": per_sample,
        "total_loss_tokens": total_tokens,
        "mean_nll": mean_nll,
        "ppl": ppl,
        "wall_s": wall,
    }


def main() -> None:
    args = parse_args()
    assert torch.cuda.is_available(), "CUDA required"

    rng = random.Random(args.seed)
    all_samples = load_samples(args.train_jsonl)
    samples = rng.sample(all_samples, args.num_samples)
    print(f"[eval] sampled {len(samples)} from {len(all_samples)} "
          f"(seed={args.seed})")

    tok = SFTTokenizer(model_name=args.base_model, max_len=args.max_seq_len)

    apply_liger()

    # ---- Base ----
    print(f"\n[eval] loading base: {args.base_model}")
    base = load_model(args.base_model)
    base_res = evaluate_model("base", base, tok, samples)
    del base
    gc.collect()
    torch.cuda.empty_cache()

    # ---- SFT ----
    print(f"\n[eval] loading SFT: {args.sft_checkpoint}")
    sft = load_model(args.sft_checkpoint)
    sft_res = evaluate_model("sft ", sft, tok, samples)
    del sft
    gc.collect()
    torch.cuda.empty_cache()

    # ---- Compare ----
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"base  mean_nll = {base_res['mean_nll']:.4f}  "
          f"ppl = {base_res['ppl']:.3f}")
    print(f"sft   mean_nll = {sft_res['mean_nll']:.4f}  "
          f"ppl = {sft_res['ppl']:.3f}")
    delta = base_res["mean_nll"] - sft_res["mean_nll"]
    ratio = sft_res["ppl"] / base_res["ppl"]
    print(f"Δ nll  = {delta:+.4f}  (positive = SFT improved)")
    print(f"ppl ratio sft/base = {ratio:.3f}  "
          f"({'SFT better' if ratio < 1 else 'NO improvement'})")

    # Per-sample paired look
    print("\n--- per-sample NLL ---")
    print(f"{'sample':>6}  {'persona':>7}  {'ses':>3}  "
          f"{'base_nll':>10}  {'sft_nll':>10}  {'delta':>8}")
    for a, b in zip(base_res["per_sample"], sft_res["per_sample"]):
        print(f"{a['persona_id']:>6}  {a['context_id']:>7}  "
              f"{a['session_idx']:>3}  {a['mean_nll']:>10.4f}  "
              f"{b['mean_nll']:>10.4f}  {a['mean_nll']-b['mean_nll']:>+8.4f}")

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps({
            "args": {k: str(v) for k, v in vars(args).items()},
            "base": base_res, "sft": sft_res,
            "delta_nll": delta, "ppl_ratio": ratio,
        }, indent=2))
        print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
