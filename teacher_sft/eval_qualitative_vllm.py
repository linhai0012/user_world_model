"""Qualitative generation, vLLM + tensor-parallel edition.

Drop-in faster replacement for eval_qualitative.py (§4.3): same input JSON
schema consumed by analyze_qual_30.py. Each of the 30 cases is generated
under 4 conditions — {base, sft} × {no-ctx, with-ctx} — but batched via
vLLM continuous batching across all 4 GPUs. Typical wall clock: ~2-3 min
vs ~15 min on HF transformers.

Reproducibility caveat: vLLM uses its own CUDA-level sampling, so output
text will differ from the HF version even at matching temperature/seed.
We're using the same rng to pick turn indices (deterministic) — only
token-sampling RNG differs.

Usage:
    python dynamic_usersim/teacher_sft/eval_qualitative_vllm.py \
        --sft-checkpoint $SCRATCHDIR/P-OPSD/teacher_sft_128k_k3/final \
        --num-samples 30 --tensor-parallel-size 4 \
        --out-json dynamic_usersim/outputs/eval3_r3_final_vllm.json
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
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

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
    ap.add_argument("--num-samples", type=int, default=30)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    ap.add_argument("--tensor-parallel-size", type=int, default=4)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--out-json", type=Path, default=None)
    return ap.parse_args()


def load_samples(path: Path) -> list[dict]:
    return [json.loads(L) for L in path.open("r", encoding="utf-8")]


def pick_target_user_turn(sample: dict, rng: random.Random) -> int | None:
    """Same logic as eval_qualitative.py: skip session opener (idx 1)."""
    tgt = sample["target_messages"]
    candidates = [i for i, m in enumerate(tgt)
                  if m["role"] == "user" and i >= 3]
    if not candidates:
        return None
    return rng.choice(candidates)


def build_prompt_tokens(tok: SFTTokenizer, sample: dict, user_turn_idx: int,
                        with_context: bool) -> tuple[list[int], str]:
    """Same as eval_qualitative.py — tokens end with '<|im_start|>user\\n'."""
    s = copy.copy(sample)
    if not with_context:
        s["prefix_messages"] = []
        s["prefix_session_range"] = [0, 0]
    s = copy.copy(s)
    s["target_messages"] = sample["target_messages"][:user_turn_idx]
    if not s["target_messages"]:
        s["target_messages"] = [sample["target_messages"][0]]

    ids, _, _ = tok.tokenize_sample(s)

    # Append "<|im_start|>user\n" to prime generation as the user
    role_prefix_ids = tok.tok.encode("<|im_start|>user\n",
                                      add_special_tokens=False)
    ids = ids + role_prefix_ids

    gt = sample["target_messages"][user_turn_idx]["content"]
    return ids, gt


def generate_batch_vllm(
    model_path: str | Path,
    prompts_ids: list[list[int]],
    sampling: SamplingParams,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: int,
) -> list[str]:
    """Load vLLM, batch-generate, free, return list of texts."""
    print(f"[vllm] loading {model_path}")
    llm = LLM(
        model=str(model_path),
        tensor_parallel_size=tensor_parallel_size,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        enforce_eager=False,
    )
    inputs = [TokensPrompt(prompt_token_ids=ids) for ids in prompts_ids]
    outputs = llm.generate(inputs, sampling)

    texts: list[str] = []
    for o in outputs:
        text = o.outputs[0].text
        for marker in ("<|im_end|>", "<|endoftext|>"):
            if marker in text:
                text = text.split(marker, 1)[0]
        texts.append(text.strip())

    del llm, outputs
    gc.collect()
    torch.cuda.empty_cache()
    return texts


def main() -> None:
    args = parse_args()
    assert torch.cuda.is_available()

    rng = random.Random(args.seed)
    all_samples = load_samples(args.train_jsonl)

    filtered = [s for s in all_samples
                if sum(1 for m in s["target_messages"]
                       if m["role"] == "user") >= 3]
    samples = rng.sample(filtered, args.num_samples)
    print(f"[eval] sampled {len(samples)} from {len(filtered)} eligible "
          f"(total {len(all_samples)})")

    tok = SFTTokenizer(model_name=args.base_model, max_len=args.max_seq_len)

    # Pre-build all prompts (4 conditions × N samples = 4N prompts per model)
    cases: list[dict] = []
    for s in samples:
        turn_idx = pick_target_user_turn(s, rng)
        if turn_idx is None:
            continue
        prev = s["target_messages"][turn_idx - 1]
        prev_snippet = prev["content"][:200].replace("\n", " ")

        ids_noctx, gt = build_prompt_tokens(tok, s, turn_idx, with_context=False)
        ids_ctx, _ = build_prompt_tokens(tok, s, turn_idx, with_context=True)

        # Guard against overlong prompts for the context case
        if len(ids_ctx) > args.max_seq_len:
            ids_ctx = ids_ctx[-args.max_seq_len:]
        if len(ids_noctx) > args.max_seq_len:
            ids_noctx = ids_noctx[-args.max_seq_len:]

        cases.append({
            "persona_id": s["persona_id"],
            "context_id": s["context_id"][:8],
            "session_idx": s["session_idx"],
            "turn_idx": turn_idx,
            "prev_role": prev["role"],
            "prev_snippet": prev_snippet,
            "ground_truth": gt,
            "ids_noctx": ids_noctx,
            "ids_ctx": ids_ctx,
            "outputs": {},
        })

    # Order: first N = noctx (one per case), next N = ctx.
    noctx_ids = [c["ids_noctx"] for c in cases]
    ctx_ids = [c["ids_ctx"] for c in cases]
    all_prompts = noctx_ids + ctx_ids
    print(f"[eval] built {len(all_prompts)} prompts per model "
          f"({len(noctx_ids)} noctx + {len(ctx_ids)} ctx)")

    im_end_id = tok.tok.encode("<|im_end|>", add_special_tokens=False)[0]
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        stop_token_ids=[im_end_id],
        seed=args.seed,
    )

    N = len(cases)  # first N prompts are noctx, next N are ctx

    # ---- Run base ----
    base_texts = generate_batch_vllm(
        args.base_model, all_prompts, sampling,
        args.tensor_parallel_size, args.gpu_memory_utilization,
        args.max_seq_len,
    )
    assert len(base_texts) == 2 * N, (len(base_texts), 2 * N)
    for i, c in enumerate(cases):
        c["outputs"]["base_noctx"] = base_texts[i]
        c["outputs"]["base_ctx"] = base_texts[N + i]

    # ---- Run SFT ----
    sft_texts = generate_batch_vllm(
        args.sft_checkpoint, all_prompts, sampling,
        args.tensor_parallel_size, args.gpu_memory_utilization,
        args.max_seq_len,
    )
    assert len(sft_texts) == 2 * N, (len(sft_texts), 2 * N)
    for i, c in enumerate(cases):
        c["outputs"]["sft_noctx"] = sft_texts[i]
        c["outputs"]["sft_ctx"] = sft_texts[N + i]

    # ---- Print side-by-side ----
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
        # Drop non-serializable token ids; keep same schema as HF version
        for c in cases:
            c.pop("ids_noctx", None)
            c.pop("ids_ctx", None)
        args.out_json.write_text(json.dumps(
            {"args": {k: str(v) for k, v in vars(args).items()},
             "cases": cases},
            indent=2, ensure_ascii=False))
        print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
