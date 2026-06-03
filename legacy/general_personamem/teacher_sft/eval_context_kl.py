"""Sanity check #2 for Teacher SFT (plan §3.5): context-use KL divergence.

Measures how much the model's predicted-user-response distribution CHANGES
when given the full progressive-context prefix vs just the target session
alone. If the SFT teacher actually leverages context, KL should be large.
The base model's KL is the baseline ("natural" sensitivity to context) —
SFT should clearly exceed it.

For each sample we do two forwards (same model): one on the full
[prefix + target] sequence, one on the target-only sequence. At every
valid user-token position inside target, we compare the two predicted
distributions via KL(P_with || P_without).

Data-parallel across N GPUs via torchrun (same pattern as eval_ppl.py).

Usage:
    torchrun --standalone --nnodes=1 --nproc_per_node=4 \
        dynamic_usersim/teacher_sft/eval_context_kl.py \
        --sft-checkpoint $SCRATCHDIR/P-OPSD/teacher_sft_128k/checkpoint-50 \
        --num-samples 20
"""

from __future__ import annotations

import argparse
import copy
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
    ap.add_argument("--sft-checkpoint", type=Path, required=True)
    ap.add_argument("--train-jsonl", type=Path, default=DEFAULT_TRAIN_JSONL)
    ap.add_argument("--num-samples", type=int, default=20)
    ap.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--lm-head-chunk", type=int, default=1024,
                    help="Positions to project through lm_head per chunk. "
                         "Lower if OOM (two logit tensors live at once here).")
    return ap.parse_args()


def load_samples(path: Path) -> list[dict]:
    return [json.loads(L) for L in path.open("r", encoding="utf-8")]


def apply_liger() -> None:
    try:
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3
        apply_liger_kernel_to_qwen3(
            rope=True, swiglu=True, cross_entropy=False,
            fused_linear_cross_entropy=True, rms_norm=True,
        )
        if int(os.environ.get("RANK", 0)) == 0:
            print("[eval] liger applied")
    except ImportError:
        if int(os.environ.get("RANK", 0)) == 0:
            print("[eval] liger not available; continuing")


def init_dist() -> tuple[int, int, bool]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        ws = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        return rank, ws, True
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


def kl_on_sample(model: torch.nn.Module, tok: SFTTokenizer, sample: dict,
                 lm_head_chunk: int) -> dict:
    """Return dict with KL and NLL metrics from two forward passes.

    Runs forwards on (full prefix + target) and (target-only), then at each
    valid target user-token position computes:
      - KL(P_with || P_without) between the two predicted distributions
      - NLL_with, NLL_without on the actual ground-truth token

    Keys: sum_kl, sum_nll_with, sum_nll_without, n_positions, full_seq_len,
    target_seq_len.

    Why both: pure KL (a) is inflated for base by long-context positional
    noise in RoPE and (b) is reduced for SFT if SFT learns to ignore
    irrelevant prefix — in both cases KL is ambiguous about "context use".
    NLL-on-ground-truth directly measures predictive benefit from context:
    delta = NLL_without - NLL_with is "nats saved by seeing prefix".
    """
    # Tokenize with full prefix
    full_ids, full_labels, _ = tok.tokenize_sample(sample)

    # Tokenize target-only: same sample but empty prefix
    target_only = copy.copy(sample)
    target_only["prefix_messages"] = []
    target_only["prefix_session_range"] = [0, 0]
    nc_ids, nc_labels, _ = tok.tokenize_sample(target_only)

    full_len = len(full_ids)
    tgt_len = len(nc_ids)
    target_start = full_len - tgt_len
    assert target_start >= 0, f"target_start={target_start} (full={full_len} tgt={tgt_len})"
    # Structural sanity: the target tail of the full sequence must equal the
    # standalone target tokenization. (Both run through the same tokenizer
    # with identical chunk boundaries — prefix doesn't affect target encoding.)
    assert full_ids[target_start:] == nc_ids, (
        "target tail mismatch — tokenization not prefix-invariant as expected"
    )

    device = torch.cuda.current_device()
    full_t = torch.tensor([full_ids], dtype=torch.long, device=device)
    nc_t = torch.tensor([nc_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        h_with = model.model(input_ids=full_t, use_cache=False).last_hidden_state[0]
        h_without = model.model(input_ids=nc_t, use_cache=False).last_hidden_state[0]

    # Collect valid shift positions + their target labels
    with_idx: list[int] = []
    without_idx: list[int] = []
    tgt_tokens: list[int] = []
    for p in range(1, tgt_len):
        if nc_labels[p] != -100:
            with_idx.append(target_start + p - 1)
            without_idx.append(p - 1)
            tgt_tokens.append(nc_labels[p])
    n = len(with_idx)
    if n == 0:
        return {"sum_kl": 0.0, "sum_nll_with": 0.0, "sum_nll_without": 0.0,
                "n_positions": 0, "full_seq_len": full_len,
                "target_seq_len": tgt_len}

    idx_with = torch.tensor(with_idx, device=device)
    idx_without = torch.tensor(without_idx, device=device)
    tgt_t = torch.tensor(tgt_tokens, device=device)
    hv_with = h_with.index_select(0, idx_with)        # [n, H]
    hv_without = h_without.index_select(0, idx_without)  # [n, H]

    # Chunked lm_head + KL + NLL over valid positions
    total_kl = 0.0
    total_nll_with = 0.0
    total_nll_without = 0.0
    with torch.no_grad():
        for i in range(0, n, lm_head_chunk):
            a = hv_with[i:i + lm_head_chunk]
            b = hv_without[i:i + lm_head_chunk]
            t = tgt_t[i:i + lm_head_chunk]
            logits_w = model.lm_head(a).float()                # [c, V]
            logits_nw = model.lm_head(b).float()               # [c, V]
            log_p = F.log_softmax(logits_w, dim=-1)
            log_q = F.log_softmax(logits_nw, dim=-1)
            # KL(P_with || P_without)
            p = log_p.exp()
            total_kl += (p * (log_p - log_q)).sum(dim=-1).sum().item()
            # NLL on ground-truth tokens
            total_nll_with += -log_p.gather(1, t.unsqueeze(1)).sum().item()
            total_nll_without += -log_q.gather(1, t.unsqueeze(1)).sum().item()
            del log_p, log_q, p, logits_w, logits_nw

    return {"sum_kl": total_kl,
            "sum_nll_with": total_nll_with,
            "sum_nll_without": total_nll_without,
            "n_positions": n,
            "full_seq_len": full_len,
            "target_seq_len": tgt_len}


def evaluate(label: str, model: torch.nn.Module, tok: SFTTokenizer,
             samples: list[dict], rank: int, world_size: int, is_dist: bool,
             lm_head_chunk: int) -> dict:
    my_samples = samples[rank::world_size]
    per_sample: list[dict] = []
    local_kl = 0.0
    local_nll_with = 0.0
    local_nll_without = 0.0
    local_n = 0
    t0 = time.time()
    for i, s in enumerate(my_samples):
        ts = time.time()
        r = kl_on_sample(model, tok, s, lm_head_chunk)
        n = r["n_positions"]
        mean_kl = r["sum_kl"] / max(n, 1)
        mean_nll_w = r["sum_nll_with"] / max(n, 1)
        mean_nll_nw = r["sum_nll_without"] / max(n, 1)
        per_sample.append({
            "global_idx": samples.index(s),
            "persona_id": s["persona_id"],
            "context_id": s["context_id"][:8],
            "session_idx": s["session_idx"],
            "full_len": r["full_seq_len"],
            "target_len": r["target_seq_len"],
            "n_positions": n,
            "mean_kl": mean_kl,
            "mean_nll_with": mean_nll_w,
            "mean_nll_without": mean_nll_nw,
            "context_benefit_nll": mean_nll_nw - mean_nll_w,
            "wall_s": time.time() - ts,
        })
        local_kl += r["sum_kl"]
        local_nll_with += r["sum_nll_with"]
        local_nll_without += r["sum_nll_without"]
        local_n += n
        print(f"  [rank{rank} {label}] {i+1}/{len(my_samples)} "
              f"full={r['full_seq_len']} tgt={r['target_seq_len']} n={n} "
              f"kl={mean_kl:.4f} nll_w={mean_nll_w:.3f} nll_nw={mean_nll_nw:.3f} "
              f"benefit={mean_nll_nw-mean_nll_w:+.3f} "
              f"({time.time()-ts:.1f}s)", flush=True)

    if is_dist:
        t = torch.tensor([local_kl, local_nll_with, local_nll_without,
                          float(local_n)],
                         dtype=torch.float64,
                         device=torch.cuda.current_device())
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        total_kl = t[0].item()
        total_nll_with = t[1].item()
        total_nll_without = t[2].item()
        total_n = int(t[3].item())
        gathered = [None] * world_size if rank == 0 else None
        dist.gather_object(per_sample, gathered, dst=0)
        if rank == 0:
            all_per = [x for lst in gathered for x in lst]
            all_per.sort(key=lambda d: d["global_idx"])
        else:
            all_per = per_sample
    else:
        total_kl = local_kl
        total_nll_with = local_nll_with
        total_nll_without = local_nll_without
        total_n = local_n
        all_per = per_sample

    mean_kl = total_kl / max(total_n, 1)
    mean_nll_w = total_nll_with / max(total_n, 1)
    mean_nll_nw = total_nll_without / max(total_n, 1)
    wall = time.time() - t0
    if rank == 0:
        print(f"[{label}] aggregate  n={total_n}  "
              f"kl={mean_kl:.4f}  nll_with={mean_nll_w:.4f}  "
              f"nll_without={mean_nll_nw:.4f}  "
              f"benefit={mean_nll_nw-mean_nll_w:+.4f}  "
              f"(wall {wall:.1f}s)")
    return {"label": label, "per_sample": all_per,
            "total_positions": total_n,
            "mean_kl": mean_kl,
            "mean_nll_with": mean_nll_w,
            "mean_nll_without": mean_nll_nw,
            "context_benefit_nll": mean_nll_nw - mean_nll_w,
            "wall_s": wall}


def main() -> None:
    args = parse_args()
    assert torch.cuda.is_available()

    rank, world_size, is_dist = init_dist()
    rng = random.Random(args.seed)
    all_samples = load_samples(args.train_jsonl)
    samples = rng.sample(all_samples, args.num_samples)
    if rank == 0:
        print(f"[eval] sampled {len(samples)} from {len(all_samples)} "
              f"(seed={args.seed}, world_size={world_size})")

    tok = SFTTokenizer(model_name=args.base_model, max_len=args.max_seq_len)
    apply_liger()

    if rank == 0:
        print(f"\n[eval] loading base: {args.base_model}")
    base = load_model(args.base_model)
    base_res = evaluate("base", base, tok, samples, rank, world_size,
                        is_dist, args.lm_head_chunk)
    del base
    gc.collect()
    torch.cuda.empty_cache()

    if rank == 0:
        print(f"\n[eval] loading SFT: {args.sft_checkpoint}")
    sft = load_model(args.sft_checkpoint)
    sft_res = evaluate("sft ", sft, tok, samples, rank, world_size,
                       is_dist, args.lm_head_chunk)
    del sft
    gc.collect()
    torch.cuda.empty_cache()

    if rank != 0:
        if is_dist:
            dist.barrier()
            dist.destroy_process_group()
        return

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\n[primary] context benefit = NLL_without - NLL_with  "
          "(larger = model uses context more)")
    b_ben = base_res["context_benefit_nll"]
    s_ben = sft_res["context_benefit_nll"]
    print(f"  base  nll_with={base_res['mean_nll_with']:.4f}  "
          f"nll_without={base_res['mean_nll_without']:.4f}  "
          f"benefit={b_ben:+.4f}")
    print(f"  sft   nll_with={sft_res['mean_nll_with']:.4f}  "
          f"nll_without={sft_res['mean_nll_without']:.4f}  "
          f"benefit={s_ben:+.4f}")
    if b_ben > 0 and s_ben > 0:
        print(f"  sft benefit / base benefit = {s_ben/b_ben:.2f}x")
    print(f"  (SFT {'actually uses context' if s_ben > b_ben else 'DOES NOT use context more than base'})")

    print("\n[secondary] KL(P_with || P_without) — distributional shift "
          "(noisy; interpret with care)")
    print(f"  base  mean_kl = {base_res['mean_kl']:.4f}")
    print(f"  sft   mean_kl = {sft_res['mean_kl']:.4f}")

    print("\n--- per-sample ---")
    print(f"{'pid':>3} {'ctx':>8} {'ses':>3} {'n':>5}  "
          f"{'base_w':>7} {'base_nw':>7} {'b_ben':>7}  "
          f"{'sft_w':>7} {'sft_nw':>7} {'s_ben':>7}")
    for a, b in zip(base_res["per_sample"], sft_res["per_sample"]):
        print(f"{a['persona_id']:>3} {a['context_id']:>8} "
              f"{a['session_idx']:>3} {a['n_positions']:>5}  "
              f"{a['mean_nll_with']:>7.3f} {a['mean_nll_without']:>7.3f} "
              f"{a['context_benefit_nll']:>+7.3f}  "
              f"{b['mean_nll_with']:>7.3f} {b['mean_nll_without']:>7.3f} "
              f"{b['context_benefit_nll']:>+7.3f}")

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps({
            "args": {k: str(v) for k, v in vars(args).items()},
            "base": base_res, "sft": sft_res,
        }, indent=2))
        print(f"\nwrote {args.out_json}")

    if is_dist:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
