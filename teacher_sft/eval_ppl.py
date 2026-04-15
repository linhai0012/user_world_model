"""Sanity check #1 for Teacher SFT (plan §3.5): base vs SFT perplexity.

Compare mean NLL on the SAME user-token positions (target session only)
between the stock Qwen3-4B base model and the SFT checkpoint. A meaningful
drop confirms the teacher is actually leveraging progressive context.

Data-parallel across N GPUs: each rank loads a full copy of the model
(~8 GB bf16 — plenty of room on one H100) and handles samples[rank::N].
Results are all-reduced; rank 0 prints the summary.

Launch:
    torchrun --standalone --nnodes=1 --nproc_per_node=4 \
        dynamic_usersim/teacher_sft/eval_ppl.py \
        --sft-checkpoint $SCRATCHDIR/P-OPSD/teacher_sft_128k/checkpoint-50 \
        --num-samples 20

Single-GPU fallback (omit torchrun):
    python dynamic_usersim/teacher_sft/eval_ppl.py --sft-checkpoint ...

The samples are drawn from the same training JSONL; this measures training-
set fitting (not generalization). For a mid-training sanity check that is
the right metric.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
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


def init_dist() -> tuple[int, int, bool]:
    """Returns (rank, world_size, is_distributed). Torchrun-aware."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        return rank, world_size, True
    torch.cuda.set_device(0)
    return 0, 1, False


def load_model(path: str | Path) -> torch.nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        str(path),
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.eval()
    model.to(torch.cuda.current_device())
    return model


def nll_on_sample(model: torch.nn.Module, tok: SFTTokenizer,
                  sample: dict, lm_head_chunk: int = 4096
                  ) -> tuple[float, int, int]:
    """Return (total_nll, n_loss_tokens, total_seq_len) for one sample.

    Why manual CE instead of model.forward(labels=...):
      Standard HF forward materializes logits [1, seq, vocab] for CE, which
      at 65k+ seq and 152k vocab is 20-40 GB bf16 plus CE intermediates —
      OOMs on H100 96GB. Liger's FusedLinearCrossEntropy only activates on
      the training path, not eval. We bypass by running the transformer to
      get hidden states, then projecting only the label != -100 positions
      through lm_head (a few thousand rows, not tens of thousands).
    """
    input_ids, labels, _ = tok.tokenize_sample(sample)
    device = torch.cuda.current_device()
    ids_t = torch.tensor([input_ids], dtype=torch.long, device=device)
    lab_t = torch.tensor([labels], dtype=torch.long, device=device)

    with torch.no_grad():
        # Call the underlying transformer, skipping lm_head + CE.
        outputs = model.model(input_ids=ids_t, use_cache=False)
        hidden = outputs.last_hidden_state  # [1, seq, hidden]

        # Causal shift: hidden at position i predicts token at position i+1
        shift_hidden = hidden[0, :-1, :]    # [seq-1, hidden]
        shift_labels = lab_t[0, 1:]          # [seq-1]
        valid = shift_labels != -100
        n_loss = int(valid.sum().item())
        if n_loss == 0:
            return 0.0, 0, ids_t.size(1)

        hidden_valid = shift_hidden[valid]   # [n_loss, hidden]
        labels_valid = shift_labels[valid]   # [n_loss]

        # Chunked lm_head + CE on valid positions only.
        total_nll = 0.0
        for i in range(0, n_loss, lm_head_chunk):
            h = hidden_valid[i:i + lm_head_chunk]      # [c, hidden]
            t = labels_valid[i:i + lm_head_chunk]       # [c]
            logits = model.lm_head(h)                   # [c, vocab]
            # fp32 for CE numerical stability (matches training default)
            loss = F.cross_entropy(logits.float(), t, reduction="sum")
            total_nll += loss.item()

    return total_nll, n_loss, ids_t.size(1)


def evaluate_model(label: str, model: torch.nn.Module, tok: SFTTokenizer,
                   samples: list[dict], rank: int, world_size: int,
                   is_dist: bool) -> dict:
    """Each rank processes samples[rank::world_size]; results are reduced.

    Per-sample detail is gathered to rank 0 for the printed table.
    """
    my_samples = samples[rank::world_size]
    local_per_sample: list[dict] = []
    local_nll = 0.0
    local_tokens = 0
    t0 = time.time()
    for i, s in enumerate(my_samples):
        t_s = time.time()
        nll, n_loss, seq_len = nll_on_sample(model, tok, s)
        mean_nll = nll / max(n_loss, 1)
        local_per_sample.append({
            "global_idx": samples.index(s),  # for ordering in the gathered table
            "persona_id": s["persona_id"],
            "context_id": s["context_id"][:8],
            "session_idx": s["session_idx"],
            "seq_len": seq_len,
            "loss_tokens": n_loss,
            "mean_nll": mean_nll,
            "ppl": float(torch.exp(torch.tensor(mean_nll)).item()),
            "wall_s": time.time() - t_s,
        })
        local_nll += nll
        local_tokens += n_loss
        print(f"  [rank{rank} {label}] {i+1}/{len(my_samples)} "
              f"seq_len={seq_len} loss_tokens={n_loss} "
              f"mean_nll={mean_nll:.4f} ppl={local_per_sample[-1]['ppl']:.3f} "
              f"({time.time()-t_s:.1f}s)", flush=True)

    # ---- Reduce aggregates ----
    if is_dist:
        t = torch.tensor([local_nll, float(local_tokens)],
                         dtype=torch.float64, device=torch.cuda.current_device())
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        total_nll, total_tokens = t[0].item(), int(t[1].item())

        # Gather per-sample details to rank 0
        gathered = [None] * world_size if rank == 0 else None
        dist.gather_object(local_per_sample, gathered, dst=0)
        if rank == 0:
            all_per_sample = [x for lst in gathered for x in lst]
            all_per_sample.sort(key=lambda d: d["global_idx"])
        else:
            all_per_sample = local_per_sample
    else:
        total_nll, total_tokens = local_nll, local_tokens
        all_per_sample = local_per_sample

    mean_nll = total_nll / max(total_tokens, 1)
    ppl = float(torch.exp(torch.tensor(mean_nll)).item())
    wall = time.time() - t0
    if rank == 0:
        print(f"[{label}] aggregate  total_loss_tokens={total_tokens}  "
              f"mean_nll={mean_nll:.4f}  ppl={ppl:.3f}  "
              f"(wall {wall:.1f}s across {world_size} ranks)")
    return {
        "label": label,
        "per_sample": all_per_sample,
        "total_loss_tokens": total_tokens,
        "mean_nll": mean_nll,
        "ppl": ppl,
        "wall_s": wall,
    }


def main() -> None:
    args = parse_args()
    assert torch.cuda.is_available(), "CUDA required"

    rank, world_size, is_dist = init_dist()

    # All ranks use the same seed -> identical sample subset; we split by stride.
    rng = random.Random(args.seed)
    all_samples = load_samples(args.train_jsonl)
    samples = rng.sample(all_samples, args.num_samples)
    if rank == 0:
        print(f"[eval] sampled {len(samples)} from {len(all_samples)} "
              f"(seed={args.seed}, world_size={world_size})")

    tok = SFTTokenizer(model_name=args.base_model, max_len=args.max_seq_len)

    apply_liger()

    # ---- Base ----
    if rank == 0:
        print(f"\n[eval] loading base: {args.base_model}")
    base = load_model(args.base_model)
    base_res = evaluate_model("base", base, tok, samples, rank, world_size, is_dist)
    del base
    gc.collect()
    torch.cuda.empty_cache()

    # ---- SFT ----
    if rank == 0:
        print(f"\n[eval] loading SFT: {args.sft_checkpoint}")
    sft = load_model(args.sft_checkpoint)
    sft_res = evaluate_model("sft ", sft, tok, samples, rank, world_size, is_dist)
    del sft
    gc.collect()
    torch.cuda.empty_cache()

    if rank != 0:
        if is_dist:
            dist.barrier()
            dist.destroy_process_group()
        return

    # ---- Compare (rank 0 only) ----
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"base  mean_nll = {base_res['mean_nll']:.4f}  "
          f"ppl = {base_res['ppl']:.3f}")
    print(f"sft   mean_nll = {sft_res['mean_nll']:.4f}  "
          f"ppl = {sft_res['ppl']:.3f}")
    delta = base_res["mean_nll"] - sft_res["mean_nll"]
    ratio = sft_res["ppl"] / base_res["ppl"]
    print(f"delta nll = {delta:+.4f}  (positive = SFT improved)")
    print(f"ppl ratio sft/base = {ratio:.3f}  "
          f"({'SFT better' if ratio < 1 else 'NO improvement'})")

    # Per-sample paired look (both tables are sorted by global_idx, so they align)
    print("\n--- per-sample NLL ---")
    print(f"{'persona':>7}  {'ctx':>8}  {'ses':>3}  "
          f"{'base_nll':>10}  {'sft_nll':>10}  {'delta':>8}")
    for a, b in zip(base_res["per_sample"], sft_res["per_sample"]):
        print(f"{a['persona_id']:>7}  {a['context_id']:>8}  "
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

    if is_dist:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
