"""Sanity check: does SFT parameterize persona identity?

Sharper variant of eval_context_kl.py. Instead of "with prefix vs without",
this script compares:
  NLL(user_tokens | REAL persona's K=3 prefix + current session)
  NLL(user_tokens | RANDOM OTHER persona's K=3 prefix + current session)

Only the K=3 prior-session prefix is swapped; the current session
(including its in-line persona system card and all turns up to the target
user turn) is kept intact. This isolates the persona-specific signal in
prior sessions.

Interpretation:
  gap = NLL(swap) - NLL(real)

  gap > 0 and large → real persona's prior sessions contain persona-specific
                      predictive signal that the model actually uses
                      (parameters + prior-session content jointly encode
                      persona identity; (A) in Phase-2 ambiguity)
  gap ≈ 0          → prior-session content is generic; SFT's NLL drop over
                      base comes from user-style learning, not persona
                      identity ((B) in the ambiguity — concerning for Phase 2)

We run both base and SFT. The SFT-minus-base delta-gap is the clearest
signal: SFT should have a larger gap if *training* taught it to separate
personas based on prior-session history.

Usage:
    torchrun --standalone --nnodes=1 --nproc_per_node=4 \
        dynamic_usersim/teacher_sft/eval_persona_swap.py \
        --sft-checkpoint $SCRATCHDIR/P-OPSD/teacher_sft_128k_k3/final \
        --num-samples 40 \
        --out-json dynamic_usersim/outputs/personaswap_r3_final.json
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
from collections import defaultdict
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
    ap.add_argument("--num-samples", type=int, default=40,
                    help="More than Eval 1/2 because per-sample gaps are "
                         "expected to be small; paired-t needs n>=30 for "
                         "decent power.")
    ap.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--swap-seed", type=int, default=1234,
                    help="Separate seed for swap-partner selection so the "
                         "sample set can overlap with Eval 1/2 while the "
                         "swap mapping is reproducible.")
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--lm-head-chunk", type=int, default=2048)
    return ap.parse_args()


# ---- Boilerplate (mirrors eval_context_kl.py) ----

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
        str(path), torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", trust_remote_code=True,
    )
    model.config.use_cache = False
    model.eval()
    model.to(torch.cuda.current_device())
    return model


# ---- Core: NLL on a sample with a given prefix ----

def nll_on_sample(model: torch.nn.Module, tok: SFTTokenizer, sample: dict,
                  lm_head_chunk: int) -> tuple[float, int, int]:
    """Return (sum_nll_on_user_tokens, n_positions, full_seq_len).

    Same mechanic as eval_context_kl.py's kl_on_sample but only NLL-with.
    Any valid user label position in the target gets its log-prob contributed.
    """
    full_ids, full_labels, _ = tok.tokenize_sample(sample)
    device = torch.cuda.current_device()
    full_t = torch.tensor([full_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        h = model.model(input_ids=full_t,
                        use_cache=False).last_hidden_state[0]  # [L, H]

    # Shift: position p-1 predicts token at position p. Gather only where
    # label (at p) is not -100.
    idx: list[int] = []
    tgt: list[int] = []
    for p in range(1, len(full_ids)):
        if full_labels[p] != -100:
            idx.append(p - 1)
            tgt.append(full_labels[p])
    n = len(idx)
    if n == 0:
        return 0.0, 0, len(full_ids)

    idx_t = torch.tensor(idx, device=device)
    tgt_t = torch.tensor(tgt, device=device)
    h_pred = h.index_select(0, idx_t)

    total = 0.0
    with torch.no_grad():
        for i in range(0, n, lm_head_chunk):
            logits = model.lm_head(h_pred[i:i + lm_head_chunk]).float()
            log_p = F.log_softmax(logits, dim=-1)
            tt = tgt_t[i:i + lm_head_chunk]
            total += -log_p.gather(1, tt.unsqueeze(1)).sum().item()
            del logits, log_p
    return total, n, len(full_ids)


# ---- Swap-partner selection ----

def build_swap_partners(samples: list[dict], pool: list[dict],
                        seed: int) -> list[dict]:
    """For each sample, pick a swap partner from a DIFFERENT persona.

    pool: full training jsonl (all personas) — gives diverse choices beyond
    the 20-sample eval set.
    Deterministic given (sample global_idx, seed).
    """
    by_persona: dict[str, list[dict]] = defaultdict(list)
    for s in pool:
        by_persona[s["persona_id"]].append(s)

    partners: list[dict] = []
    rng = random.Random(seed)
    all_pids = sorted(by_persona.keys())
    for s in samples:
        other_pids = [p for p in all_pids if p != s["persona_id"]]
        chosen_pid = rng.choice(other_pids)
        partner = rng.choice(by_persona[chosen_pid])
        partners.append(partner)
    return partners


def swap_prefix(sample: dict, partner: dict) -> dict:
    """Return a copy of `sample` with prefix_messages replaced by partner's."""
    out = copy.copy(sample)
    out["prefix_messages"] = partner["prefix_messages"]
    out["prefix_session_range"] = partner["prefix_session_range"]
    # Keep target_messages unchanged — so user-tokens we score are the same.
    return out


# ---- Evaluation over samples ----

def evaluate_condition(label: str, model: torch.nn.Module, tok: SFTTokenizer,
                       samples: list[dict], rank: int, world_size: int,
                       is_dist: bool, lm_head_chunk: int) -> dict:
    """Compute mean NLL on user tokens over `samples` for one model.

    Returns per-sample and aggregate stats.
    """
    my_samples = samples[rank::world_size]
    per_sample: list[dict] = []
    local_nll = 0.0
    local_n = 0
    t0 = time.time()
    for i, s in enumerate(my_samples):
        ts = time.time()
        total, n, full_len = nll_on_sample(model, tok, s, lm_head_chunk)
        mean_nll = total / max(n, 1)
        per_sample.append({
            "global_idx": samples.index(s),
            "persona_id": s["persona_id"],
            "context_id": s["context_id"][:8],
            "session_idx": s["session_idx"],
            "full_len": full_len,
            "n_positions": n,
            "mean_nll": mean_nll,
            "wall_s": time.time() - ts,
        })
        local_nll += total
        local_n += n
        print(f"  [rank{rank} {label}] {i+1}/{len(my_samples)} "
              f"pid={s['persona_id']} full={full_len} n={n} "
              f"nll={mean_nll:.4f} ({time.time()-ts:.1f}s)", flush=True)

    if is_dist:
        t = torch.tensor([local_nll, float(local_n)], dtype=torch.float64,
                         device=torch.cuda.current_device())
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        total_nll = t[0].item()
        total_n = int(t[1].item())
        gathered = [None] * world_size if rank == 0 else None
        dist.gather_object(per_sample, gathered, dst=0)
        if rank == 0:
            all_per = [x for lst in gathered for x in lst]
            all_per.sort(key=lambda d: d["global_idx"])
        else:
            all_per = per_sample
    else:
        total_nll = local_nll
        total_n = local_n
        all_per = per_sample

    mean_nll = total_nll / max(total_n, 1)
    if rank == 0:
        print(f"[{label}] aggregate n={total_n} mean_nll={mean_nll:.4f} "
              f"(wall {time.time()-t0:.1f}s)")
    return {"label": label, "per_sample": all_per,
            "total_positions": total_n, "mean_nll": mean_nll}


def run_model(model_label: str, model_path: str | Path, tok: SFTTokenizer,
              real_samples: list[dict], swap_samples: list[dict],
              rank: int, world_size: int, is_dist: bool,
              lm_head_chunk: int) -> dict:
    """Load one model, eval on real + swap sample sets, free it."""
    if rank == 0:
        print(f"\n[eval] loading {model_label}: {model_path}")
    model = load_model(model_path)
    real_res = evaluate_condition(
        f"{model_label}/real", model, tok, real_samples,
        rank, world_size, is_dist, lm_head_chunk,
    )
    swap_res = evaluate_condition(
        f"{model_label}/swap", model, tok, swap_samples,
        rank, world_size, is_dist, lm_head_chunk,
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {"real": real_res, "swap": swap_res}


# ---- Main ----

def main() -> None:
    args = parse_args()
    assert torch.cuda.is_available()
    rank, world_size, is_dist = init_dist()

    # Load samples
    all_samples = load_samples(args.train_jsonl)
    rng = random.Random(args.seed)
    samples = rng.sample(all_samples, args.num_samples)
    # Deterministic swap partners, orthogonal seed
    partners = build_swap_partners(samples, all_samples, seed=args.swap_seed)
    swapped = [swap_prefix(s, p) for s, p in zip(samples, partners)]

    if rank == 0:
        # Sanity: no swap partner has the same persona as its original
        assert all(p["persona_id"] != s["persona_id"]
                   for s, p in zip(samples, partners))
        print(f"[eval] sampled {len(samples)} of {len(all_samples)} "
              f"(seed={args.seed})")
        # Report swap persona distribution
        pair_pids = [(s["persona_id"], p["persona_id"])
                     for s, p in zip(samples, partners)]
        print(f"[eval] sample personas: "
              f"{sorted({s['persona_id'] for s in samples})}")
        print(f"[eval] swap-partner personas: "
              f"{sorted({p['persona_id'] for p in partners})}")
        print(f"[eval] first 5 pairs (real_pid -> swap_pid): "
              f"{pair_pids[:5]}")

    tok = SFTTokenizer(model_name=args.base_model, max_len=args.max_seq_len)
    apply_liger()

    base_out = run_model("base", args.base_model, tok, samples, swapped,
                         rank, world_size, is_dist, args.lm_head_chunk)
    sft_out = run_model("sft", args.sft_checkpoint, tok, samples, swapped,
                        rank, world_size, is_dist, args.lm_head_chunk)

    if rank != 0:
        if is_dist:
            dist.barrier()
            dist.destroy_process_group()
        return

    # ---- Summary ----
    print("\n" + "=" * 68)
    print("SUMMARY — persona swap diagnostic")
    print("=" * 68)

    def line(tag: str, real: dict, swap: dict) -> None:
        gap = swap["mean_nll"] - real["mean_nll"]
        print(f"  {tag:<5} real_nll={real['mean_nll']:.4f}  "
              f"swap_nll={swap['mean_nll']:.4f}  "
              f"gap={gap:+.4f}  (positive = real persona helps)")

    line("base", base_out["real"], base_out["swap"])
    line("sft ", sft_out["real"], sft_out["swap"])
    base_gap = base_out["swap"]["mean_nll"] - base_out["real"]["mean_nll"]
    sft_gap = sft_out["swap"]["mean_nll"] - sft_out["real"]["mean_nll"]
    delta = sft_gap - base_gap
    print(f"\n  SFT gap − Base gap = {delta:+.4f}   "
          f"(positive = SFT is MORE persona-sensitive than base; "
          f"this is the Phase-2-relevant signal)")

    # Per-sample paired comparison
    print("\n--- per-sample (paired) ---")
    print(f"{'pid':>3} {'pid_swp':>7} {'ctx':>8} {'ses':>3} {'n':>5}  "
          f"{'base_real':>9} {'base_swap':>9} {'b_gap':>7}  "
          f"{'sft_real':>9} {'sft_swap':>9} {'s_gap':>7}")
    partners_by_idx = {samples.index(s): p for s, p in zip(samples, partners)}
    base_real = base_out["real"]["per_sample"]
    base_swap = base_out["swap"]["per_sample"]
    sft_real = sft_out["real"]["per_sample"]
    sft_swap = sft_out["swap"]["per_sample"]
    gaps_base = []
    gaps_sft = []
    for br, bs, sr, ss in zip(base_real, base_swap, sft_real, sft_swap):
        gi = br["global_idx"]
        partner = partners_by_idx[gi]
        bg = bs["mean_nll"] - br["mean_nll"]
        sg = ss["mean_nll"] - sr["mean_nll"]
        gaps_base.append(bg)
        gaps_sft.append(sg)
        print(f"{br['persona_id']:>3} {partner['persona_id']:>7} "
              f"{br['context_id']:>8} {br['session_idx']:>3} "
              f"{br['n_positions']:>5}  "
              f"{br['mean_nll']:>9.4f} {bs['mean_nll']:>9.4f} {bg:>+7.3f}  "
              f"{sr['mean_nll']:>9.4f} {ss['mean_nll']:>9.4f} {sg:>+7.3f}")

    # Sign counts
    def sign_summary(gaps: list[float], tag: str) -> None:
        pos = sum(1 for g in gaps if g > 0)
        neg = sum(1 for g in gaps if g < 0)
        zero = len(gaps) - pos - neg
        print(f"  {tag} gap>0: {pos}/{len(gaps)}  gap<0: {neg}/{len(gaps)}  "
              f"gap=0: {zero}/{len(gaps)}")

    print("\n--- sign distribution (gap = swap - real) ---")
    sign_summary(gaps_base, "base")
    sign_summary(gaps_sft, "sft ")

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps({
            "args": {k: str(v) for k, v in vars(args).items()},
            "sample_persona_ids": [s["persona_id"] for s in samples],
            "swap_partner_persona_ids": [p["persona_id"] for p in partners],
            "base": base_out, "sft": sft_out,
            "base_gap_mean": base_gap, "sft_gap_mean": sft_gap,
            "delta_gap_sft_minus_base": delta,
        }, indent=2))
        print(f"\nwrote {args.out_json}")

    if is_dist:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
