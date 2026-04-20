"""Sanity check for OPSD design — does the R3 teacher actually USE an
injected GT user_response to sharpen its distribution on that same
sequence, and which injection layout is in-distribution for it?

Three GT-injection conditions, all scored on the SAME target tokens
(the sample's GT user_response) so the only variable is WHERE GT lives:

  A (baseline / natural):
      teacher_input = [K=3 history] + <|im_start|>user\\n
      target = gt_ids  (teacher has NEVER seen GT in this forward)
      → measures teacher's natural difficulty on this user turn.

  B (system-augmented — our preferred OPSD variant):
      Mutate the FIRST system message in history to:
        "{persona_card}\\n\\n[Recent indication of user's preference: {GT}]"
      Everything else identical to A.
      target = gt_ids  (GT appears in the system block AND as the target)

  C (prior-turn injection — original pseudo-code proposal):
      teacher_input = [K=3 history] + <|im_start|>user\\n{GT}<|im_end|>\\n
                    + <|im_start|>user\\n
      target = gt_ids
      → teacher sees GT as a completed prior user turn, then asked to
        predict another user turn starting with GT again.

For each target token position i (up to --max-target-tokens):
  rank_i = position of gt_ids[i] when teacher's logits at pred-pos i are
           sorted descending (0 = argmax)
  p_i    = softmax P(gt_ids[i])
  H_i    = entropy of softmax

Aggregate over all positions × all samples. Output a comparison table.

Decision rule for going ahead with OPSD (system-augmented variant):
  - B mean P(GT) significantly > A mean P(GT)        (teacher IS attending)
  - B rank@1 rate significantly > A rank@1 rate
  - C degrades vs B (e.g., rank spikes or entropy spikes on some positions)
    → confirms user's hypothesis that raw GT-as-prior-turn is OOD.

No training. Pure eval. ~1-2 minutes on a single H100/GH200.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# ChatML helpers (match training format — see tokenize_teacher_sft.py)
# ---------------------------------------------------------------------------
def encode_msg(tok, role: str, content: str) -> list[int]:
    prefix = tok.encode(f"<|im_start|>{role}\n", add_special_tokens=False)
    body = tok.encode(content, add_special_tokens=False)
    end = tok.encode("<|im_end|>\n", add_special_tokens=False)
    return prefix + body + end


def build_prefix_ids(tok, msgs: list[dict], trailing_role: str) -> list[int]:
    ids: list[int] = []
    for m in msgs:
        ids.extend(encode_msg(tok, m["role"], m["content"]))
    ids.extend(tok.encode(f"<|im_start|>{trailing_role}\n",
                          add_special_tokens=False))
    return ids


def augment_system_with_gt(history_msgs: list[dict], gt: str) -> list[dict]:
    """Return a copy of history_msgs with the FIRST system message
    augmented to include a 'Recent indication of user's preference' block
    containing GT.  Other messages untouched."""
    out = [dict(m) for m in history_msgs]
    for m in out:
        if m["role"] == "system":
            m["content"] = (
                m["content"].rstrip()
                + f"\n\n[Recent indication of user's preference: {gt.strip()}]"
            )
            return out
    # Fallback: if no system message (shouldn't happen in PersonaMem),
    # prepend one.
    out.insert(0, {"role": "system",
                   "content": f"[Recent indication of user's preference: {gt.strip()}]"})
    return out


# ---------------------------------------------------------------------------
# Per-position scoring
# ---------------------------------------------------------------------------
@torch.no_grad()
def score_target_positions(
    model, prefix_ids: list[int], target_ids: list[int],
    device: torch.device, top_k: int = 50,
) -> dict:
    """Return per-token arrays: rank, p, entropy."""
    full = torch.tensor([prefix_ids + target_ids], dtype=torch.long,
                        device=device)
    logits = model(input_ids=full, use_cache=False).logits[0]
    # Next-token: position (len(prefix)-1) predicts target[0]
    start = len(prefix_ids) - 1
    end = start + len(target_ids)
    sel = logits[start:end].float()             # [T, V]
    probs = F.softmax(sel, dim=-1)

    tgt = torch.tensor(target_ids, device=device)
    p_tgt = probs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)   # [T]
    # Rank: how many tokens have prob strictly greater than target's
    rank = (probs > p_tgt.unsqueeze(-1)).sum(dim=-1).long()    # [T], 0-based
    entropy = -(probs.clamp_min(1e-12).log() * probs).sum(dim=-1)   # [T]

    return {
        "rank": rank.cpu().tolist(),
        "p": p_tgt.cpu().tolist(),
        "entropy": entropy.cpu().tolist(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True,
                    help="local path to R3 teacher (e.g. $SCRATCHDIR/P-OPSD/teacher_sft_128k_k3/final)")
    ap.add_argument("--data", required=True,
                    help="opd_128k_pid{N}_k3.jsonl")
    ap.add_argument("--n-samples", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-target-tokens", type=int, default=80,
                    help="cap scored target positions per sample (saves memory)")
    ap.add_argument("--max-seq-len", type=int, default=32768)
    ap.add_argument("--attn-impl", default="sdpa",
                    choices=["flash_attention_2", "sdpa", "eager"])
    ap.add_argument("--out-json", default="dynamic_usersim/outputs/sanity_gt_injection.json")
    return ap.parse_args()


def summarize(vals: list[float]) -> dict:
    vals = sorted(vals)
    n = len(vals)
    return {
        "n": n,
        "mean": statistics.mean(vals) if vals else float("nan"),
        "median": vals[n // 2] if vals else float("nan"),
        "p25": vals[n // 4] if vals else float("nan"),
        "p75": vals[3 * n // 4] if vals else float("nan"),
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"[sanity] loading teacher from {args.teacher}")
    tok = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.teacher, torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_impl, trust_remote_code=True,
    ).to(device).eval()

    # Load OPD training samples
    with open(args.data, encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]
    random.Random(args.seed).shuffle(samples)
    samples = samples[: args.n_samples]
    print(f"[sanity] {len(samples)} samples from {args.data}")

    # Per-condition running buffers
    ranks = {"A": [], "B": [], "C": []}
    probs = {"A": [], "B": [], "C": []}
    ents  = {"A": [], "B": [], "C": []}

    for si, sample in enumerate(samples):
        gt = sample["user_response"]
        history = sample["history_messages"]
        target_ids = tok.encode(gt, add_special_tokens=False)
        if args.max_target_tokens > 0:
            target_ids = target_ids[: args.max_target_tokens]
        if not target_ids:
            continue

        # ---- A: natural ----
        prefix_A = build_prefix_ids(tok, history, trailing_role="user")

        # ---- B: system-augmented ----
        history_B = augment_system_with_gt(history, gt)
        prefix_B = build_prefix_ids(tok, history_B, trailing_role="user")

        # ---- C: prior-turn injection ----
        # history ends with assistant (chatbot_prev). Insert a full user turn
        # with GT, then open a new user turn.
        prefix_C = (
            build_prefix_ids(tok, history, trailing_role="user")[:-1]  # strip trailing '\n' from user\n? no — prefix ends with tokens of "<|im_start|>user\n"
        )
        # Cleaner: rebuild prefix_C explicitly
        prefix_C = []
        for m in history:
            prefix_C.extend(encode_msg(tok, m["role"], m["content"]))
        # Completed GT user turn
        prefix_C.extend(encode_msg(tok, "user", gt))
        # New user turn trailing role
        prefix_C.extend(tok.encode("<|im_start|>user\n", add_special_tokens=False))

        # Safety: cap total length
        for prefix in (prefix_A, prefix_B, prefix_C):
            if len(prefix) + len(target_ids) > args.max_seq_len:
                overflow = len(prefix) + len(target_ids) - args.max_seq_len
                # drop oldest tokens (left-side truncation)
                del prefix[:overflow]

        for cond, prefix in [("A", prefix_A), ("B", prefix_B), ("C", prefix_C)]:
            out = score_target_positions(model, prefix, target_ids, device)
            ranks[cond].extend(out["rank"])
            probs[cond].extend(out["p"])
            ents[cond].extend(out["entropy"])

        if (si + 1) % 5 == 0 or si == len(samples) - 1:
            print(f"  [{si+1}/{len(samples)}] processed; "
                  f"cumulative positions A/B/C = "
                  f"{len(ranks['A'])}/{len(ranks['B'])}/{len(ranks['C'])}",
                  flush=True)

    # -----------------------------------------------------------------
    # Print comparison table
    # -----------------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"{'metric':<22} {'A natural':>15} {'B sys-aug':>15} {'C prior-turn':>15}")
    print("=" * 78)

    def line(label: str, fmt: str, a, b, c):
        print(f"{label:<22} {fmt.format(a):>15} {fmt.format(b):>15} {fmt.format(c):>15}")

    # rank@1 rate
    rank1 = {c: sum(1 for r in ranks[c] if r == 0) / max(len(ranks[c]), 1)
             for c in ranks}
    rank5 = {c: sum(1 for r in ranks[c] if r < 5) / max(len(ranks[c]), 1)
             for c in ranks}
    rank50 = {c: sum(1 for r in ranks[c] if r < 50) / max(len(ranks[c]), 1)
              for c in ranks}
    mean_p = {c: statistics.mean(probs[c]) if probs[c] else float("nan")
              for c in probs}
    median_rank = {c: sorted(ranks[c])[len(ranks[c]) // 2] if ranks[c] else -1
                   for c in ranks}
    mean_ent = {c: statistics.mean(ents[c]) if ents[c] else float("nan")
                for c in ents}
    # Mean NLL = -mean(log p)
    def _nll(ps):
        ls = [-math.log(max(p, 1e-20)) for p in ps]
        return statistics.mean(ls) if ls else float("nan")
    mean_nll = {c: _nll(probs[c]) for c in probs}

    line("rank@1 (↑ better)", "{:.3f}", rank1["A"], rank1["B"], rank1["C"])
    line("rank@5",            "{:.3f}", rank5["A"], rank5["B"], rank5["C"])
    line("rank@50",           "{:.3f}", rank50["A"], rank50["B"], rank50["C"])
    line("median rank (↓)",   "{}",     median_rank["A"], median_rank["B"], median_rank["C"])
    line("mean P(GT) (↑)",    "{:.4f}", mean_p["A"], mean_p["B"], mean_p["C"])
    line("mean NLL (↓)",      "{:.3f}", mean_nll["A"], mean_nll["B"], mean_nll["C"])
    line("mean entropy (↓)",  "{:.3f}", mean_ent["A"], mean_ent["B"], mean_ent["C"])
    print("=" * 78)

    # Save raw
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "args": vars(args),
        "n_samples": len(samples),
        "total_positions": {c: len(ranks[c]) for c in ranks},
        "rank1": rank1, "rank5": rank5, "rank50": rank50,
        "mean_P_gt": mean_p, "mean_NLL": mean_nll, "mean_entropy": mean_ent,
        "median_rank": median_rank,
        "rank_distribution": ranks,
        "p_distribution": probs,
    }, indent=2, ensure_ascii=False))
    print(f"\n[sanity] wrote {out_path}")

    # Interpretation hints
    print("\n--- expected pattern if OPSD-B is viable ---")
    print("  P(GT) :   A (baseline) <  B (sys-aug)  — teacher uses GT hint")
    print("  P(GT) :   B  vs  C     — if B > C, system channel is cleaner")
    print("  rank@1:   A < B         — teacher frequently argmaxes GT with hint")
    print("  entropy:  B < A         — distribution sharpens with GT hint")


if __name__ == "__main__":
    main()
