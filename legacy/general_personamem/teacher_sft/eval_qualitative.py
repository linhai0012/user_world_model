"""Sanity check #3 for Teacher SFT (plan §3.5): qualitative generation.

For a handful of samples, pick a user turn inside the target session and
generate the predicted user response under four conditions:

    base + no_context    base + context    sft + no_context    sft + context

Print all four side-by-side alongside the ground-truth user response.
Purpose is a vibe check: does SFT-with-context produce responses that
reference specific details from the conversation history (hobbies, prior
preferences, named people/places) that no-context cannot know?

Single GPU, small N (default 3 samples x 1 turn each). Uses the model's
own generate() with temperature=0.7 top_p=0.9.

Usage (server):
    python dynamic_usersim/teacher_sft/eval_qualitative.py \
        --sft-checkpoint $SCRATCHDIR/P-OPSD/teacher_sft_128k/checkpoint-50 \
        --num-samples 3 --max-new-tokens 200
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import random
import sys
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
    ap.add_argument("--sft-checkpoint", type=Path, required=True)
    ap.add_argument("--train-jsonl", type=Path, default=DEFAULT_TRAIN_JSONL)
    ap.add_argument("--num-samples", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    ap.add_argument("--out-json", type=Path, default=None)
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
        print("[eval] liger applied")
    except ImportError:
        print("[eval] liger not available; continuing")


def load_model(path: str | Path) -> torch.nn.Module:
    m = AutoModelForCausalLM.from_pretrained(
        str(path),
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    m.config.use_cache = True   # enable for generation
    m.eval()
    m.to(torch.cuda.current_device())
    return m


def pick_target_user_turn(sample: dict, rng: random.Random) -> int | None:
    """Return message index within target_messages for a user turn that is
    NOT the first turn (first is a greeting; per plan §4.1.6 they're
    unpredictable). Returns None if no valid turn exists."""
    tgt = sample["target_messages"]
    # User turns, skipping index 1 (which is the session opener right after
    # the system card at index 0)
    candidates = [i for i, m in enumerate(tgt)
                  if m["role"] == "user" and i >= 3]
    if not candidates:
        return None
    return rng.choice(candidates)


def build_prompt_tokens(tok: SFTTokenizer, sample: dict, user_turn_idx: int,
                       with_context: bool) -> tuple[list[int], str]:
    """Build input_ids ending just before the target user turn at user_turn_idx.

    Returns (input_ids, ground_truth_user_content).
    """
    s = copy.copy(sample)
    if not with_context:
        s["prefix_messages"] = []
        s["prefix_session_range"] = [0, 0]
    # Truncate target to everything BEFORE user_turn_idx, then append the
    # user role prefix so the model continues as that user.
    s = copy.copy(s)
    s["target_messages"] = sample["target_messages"][:user_turn_idx]

    # tokenize_sample expects target to have at least one message
    if not s["target_messages"]:
        # fall back: include the system card only
        s["target_messages"] = [sample["target_messages"][0]]

    ids, _, _ = tok.tokenize_sample(s)

    # Append "<|im_start|>user\n" to prime generation as the user
    role_prefix_ids = tok.tok.encode("<|im_start|>user\n", add_special_tokens=False)
    ids = ids + role_prefix_ids

    gt = sample["target_messages"][user_turn_idx]["content"]
    return ids, gt


def generate(model: torch.nn.Module, tok: SFTTokenizer, ids: list[int],
             max_new_tokens: int, temperature: float, top_p: float) -> str:
    device = torch.cuda.current_device()
    # Trim from left if too long for model's max ctx
    if len(ids) > model.config.max_position_embeddings:
        ids = ids[-model.config.max_position_embeddings:]
    ids_t = torch.tensor([ids], dtype=torch.long, device=device)
    im_end = tok.tok.encode("<|im_end|>", add_special_tokens=False)[0]
    with torch.no_grad():
        out = model.generate(
            input_ids=ids_t,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=im_end,
            pad_token_id=tok.tok.pad_token_id or tok.tok.eos_token_id,
        )
    gen_ids = out[0, ids_t.size(1):].tolist()
    text = tok.tok.decode(gen_ids, skip_special_tokens=False)
    # Strip trailing markers if present
    for marker in ["<|im_end|>", "<|endoftext|>"]:
        if marker in text:
            text = text.split(marker, 1)[0]
    return text.strip()


def main() -> None:
    args = parse_args()
    assert torch.cuda.is_available()
    torch.cuda.set_device(0)

    rng = random.Random(args.seed)
    all_samples = load_samples(args.train_jsonl)

    # Pick samples that have enough target user turns for a meaningful turn
    filtered = [s for s in all_samples
                if sum(1 for m in s["target_messages"]
                       if m["role"] == "user") >= 3]
    samples = rng.sample(filtered, args.num_samples)
    print(f"[eval] sampled {len(samples)} from {len(filtered)} eligible "
          f"(total {len(all_samples)})")

    tok = SFTTokenizer(model_name=args.base_model, max_len=args.max_seq_len)
    apply_liger()

    # Pre-pick turn indices and build prompts
    cases: list[dict] = []
    for s in samples:
        turn_idx = pick_target_user_turn(s, rng)
        if turn_idx is None:
            continue
        # The immediately preceding message tells us what the model is reacting to
        prev = s["target_messages"][turn_idx - 1]
        prev_snippet = prev["content"][:200].replace("\n", " ")
        cases.append({
            "persona_id": s["persona_id"],
            "context_id": s["context_id"][:8],
            "session_idx": s["session_idx"],
            "turn_idx": turn_idx,
            "prev_role": prev["role"],
            "prev_snippet": prev_snippet,
            "sample": s,
        })

    def run_condition(label: str, model: torch.nn.Module,
                      with_ctx: bool) -> None:
        print(f"\n==== [{label}]  (with_context={with_ctx}) ====")
        for c in cases:
            ids, gt = build_prompt_tokens(tok, c["sample"], c["turn_idx"], with_ctx)
            text = generate(model, tok, ids, args.max_new_tokens,
                            args.temperature, args.top_p)
            c.setdefault("outputs", {})[f"{label}_{'ctx' if with_ctx else 'noctx'}"] = text
            c["ground_truth"] = gt
            print(f"  persona={c['persona_id']} sess={c['session_idx']} "
                  f"turn={c['turn_idx']} (prompt_len={len(ids)})")
            print(f"  prev ({c['prev_role']}): {c['prev_snippet']!r}")
            print(f"  -> {text[:300]!r}")

    print("\n[eval] loading base")
    base = load_model(args.base_model)
    run_condition("base", base, with_ctx=False)
    run_condition("base", base, with_ctx=True)
    del base
    gc.collect()
    torch.cuda.empty_cache()

    print("\n[eval] loading SFT")
    sft = load_model(args.sft_checkpoint)
    run_condition("sft", sft, with_ctx=False)
    run_condition("sft", sft, with_ctx=True)
    del sft
    gc.collect()
    torch.cuda.empty_cache()

    # ---- Side-by-side print ----
    print("\n" + "=" * 70)
    print("SIDE-BY-SIDE (truncated to 300 chars)")
    print("=" * 70)
    for c in cases:
        print(f"\n--- persona={c['persona_id']} ctx={c['context_id']} "
              f"session={c['session_idx']} turn={c['turn_idx']} ---")
        print(f"prev {c['prev_role']}: {c['prev_snippet'][:200]!r}")
        print(f"GROUND TRUTH:     {c['ground_truth'][:300]!r}")
        for k in ["base_noctx", "base_ctx", "sft_noctx", "sft_ctx"]:
            v = c["outputs"].get(k, "")
            print(f"  {k:14s}: {v[:300]!r}")

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        # drop non-serializable "sample"
        for c in cases:
            c.pop("sample", None)
        args.out_json.write_text(json.dumps(
            {"args": {k: str(v) for k, v in vars(args).items()}, "cases": cases},
            indent=2, ensure_ascii=False))
        print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
