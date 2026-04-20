"""Verbal feedback — Stage 1 (pure generation) — HF transformers backend.

Drop-in alternative to eval_mcq_verbal_gen.py (vLLM) that does NOT require
a pre-merged model on disk. Uses eval_opd.load_model_with_lora, which
merges dual LoRA IN GPU memory — fits within tight Slurm CPU-RAM caps
(proven to work on KCL with the 2GB-mem allocation that SIGKILL'd the
save_pretrained step in merge_dual_lora.py).

Same prompt format, same output JSONL structure as the vLLM version, so
downstream scoring scripts work on both interchangeably.

Speed trade-off: ~5-10× slower than vLLM (no continuous batching) but
tolerable — ~10-20 min for 147 MCQs × 4 choices on B200.

Usage:
    source setup_kcl.sh
    python dynamic_usersim/student_opd/eval_mcq_verbal_gen_hf.py \\
        --base-model $QWEN3_BASE \\
        --lora-path  $LORA_ROOT/pid4 --lora-mode dual \\
        --persona-id 4 --num-mcqs -1 --context-mode demo-only \\
        --attn-impl sdpa --no-liger \\
        --out-jsonl dynamic_usersim/outputs/verbal_pid4_r1b.jsonl

    # Base-only (no LoRA) for comparison:
    python dynamic_usersim/student_opd/eval_mcq_verbal_gen_hf.py \\
        --base-model $QWEN3_BASE \\
        --persona-id 4 --num-mcqs -1 --context-mode demo-only \\
        --attn-impl sdpa --no-liger \\
        --out-jsonl dynamic_usersim/outputs/verbal_pid4_base.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

_HERE = Path(__file__).resolve().parent
_TSFT = _HERE.parent / "teacher_sft"
_DP = _HERE.parent / "data_prep"
for p in (_HERE, _TSFT, _DP):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from load_personamem import (  # noqa: E402
    load_contexts, load_questions, strip_role_prefix,
)
from eval_mcq import parse_choices  # noqa: E402
from eval_opd import build_eval_context, load_model_with_lora  # noqa: E402
from eval_mcq_ppl import apply_liger  # noqa: E402
from eval_mcq_verbal_gen import (  # noqa: E402
    build_verbal_prompt, extract_gt_window,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", type=str, required=True)
    ap.add_argument("--lora-path", type=Path, default=None)
    ap.add_argument("--lora-mode", choices=["single", "dual"], default="single")
    ap.add_argument("--persona-id", type=str, required=True)
    ap.add_argument("--mcq-version", choices=["32k", "128k", "1M"],
                    default="128k")
    ap.add_argument("--context-mode",
                    choices=["demo-only", "last-n", "full", "recent-turns"],
                    default="demo-only")
    ap.add_argument("--last-n-sessions", type=int, default=3)
    ap.add_argument("--recent-turns", type=int, default=2)
    ap.add_argument("--num-mcqs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=42)
    # sampling
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=150)
    ap.add_argument("--n-samples", type=int, default=1,
                    help="reactions per choice (HF num_return_sequences)")
    # batching + runtime
    ap.add_argument("--batch-size", type=int, default=16,
                    help="prompts per forward; adjust for GPU memory")
    ap.add_argument("--max-input-len", type=int, default=4096,
                    help="truncate prompts longer than this (demo-only "
                         "stays well under this; full context may not)")
    ap.add_argument("--attn-impl",
                    choices=["flash_attention_2", "sdpa", "eager"],
                    default="sdpa")
    ap.add_argument("--no-liger", action="store_true")
    ap.add_argument("--out-jsonl", type=Path, required=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    assert torch.cuda.is_available()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # --- Data ---
    print(f"[gen] loading MCQs ({args.mcq_version}) for pid={args.persona_id}")
    mcqs = load_questions(args.mcq_version)
    ctxs = load_contexts(args.mcq_version)
    mcqs = [q for q in mcqs
            if q["shared_context_id"] in ctxs
            and q["persona_id"] == args.persona_id]
    random.Random(args.seed).shuffle(mcqs)
    if args.num_mcqs > 0:
        mcqs = mcqs[: args.num_mcqs]
    print(f"[gen] {len(mcqs)} MCQs → {len(mcqs)*4} prompts")

    # --- Model + tokenizer ---
    if not args.no_liger:
        apply_liger()
    else:
        print("[gen] liger skipped (--no-liger)")
    print(f"[gen] loading base={args.base_model}"
          f"{' + LoRA' if args.lora_path else ''}  attn={args.attn_impl}")
    model = load_model_with_lora(
        args.base_model, args.lora_path, args.lora_mode,
        attn_impl=args.attn_impl,
    )
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
    im_start_id = tok.convert_tokens_to_ids("<|im_start|>")
    eos_ids = [im_end_id, im_start_id]

    # --- Build prompts ---
    prompts: list[str] = []
    prompt_meta: list[dict] = []
    for q in mcqs:
        ctx_msgs = build_eval_context(
            ctxs[q["shared_context_id"]],
            int(q["end_index_in_shared_context"]),
            args.context_mode, args.last_n_sessions, args.recent_turns,
        )
        question = q["user_question_or_message"]
        choices = parse_choices(q["all_options"])
        for label, choice_text in choices:
            prompts.append(build_verbal_prompt(ctx_msgs, question, choice_text))
            prompt_meta.append({"qid": q["question_id"],
                                "label": label,
                                "choice_text": choice_text})

    # --- HF batched generate ---
    device = model.device
    all_reactions: list[list[str]] = []   # one list (n_samples strings) per prompt

    t0 = time.time()
    n_batches = (len(prompts) + args.batch_size - 1) // args.batch_size
    for bi in range(n_batches):
        batch = prompts[bi*args.batch_size : (bi+1)*args.batch_size]
        enc = tok(batch, return_tensors="pt", padding=True,
                  truncation=True, max_length=args.max_input_len)
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=args.temperature > 0,
                num_return_sequences=args.n_samples,
                pad_token_id=tok.pad_token_id,
                eos_token_id=eos_ids,
                use_cache=True,
            )
        # gen shape: (batch_size * n_samples, total_len)
        # Reshape to (batch_size, n_samples, total_len)
        bs = len(batch)
        gen = gen.view(bs, args.n_samples, -1)
        prompt_lens = enc["attention_mask"].sum(dim=1)   # (batch_size,)

        for j in range(bs):
            plen = int(prompt_lens[j].item())
            per_prompt = []
            for k in range(args.n_samples):
                new_ids = gen[j, k, plen:].tolist()
                text = tok.decode(new_ids, skip_special_tokens=False)
                for stop in ("<|im_end|>", "<|im_start|>"):
                    if stop in text:
                        text = text.split(stop)[0]
                # Some tokenizers render pad as special strings; strip
                text = text.replace("<|endoftext|>", "").strip()
                per_prompt.append(text)
            all_reactions.append(per_prompt)

        if (bi + 1) % 5 == 0 or bi == n_batches - 1:
            elapsed = time.time() - t0
            eta = elapsed / (bi + 1) * (n_batches - bi - 1)
            print(f"  batch {bi+1}/{n_batches}  elapsed={elapsed:.0f}s "
                  f"eta={eta:.0f}s", flush=True)

    print(f"[gen] done in {time.time()-t0:.1f}s")

    # --- Collate per MCQ ---
    by_qid: dict[str, dict] = {}
    for meta, reactions in zip(prompt_meta, all_reactions):
        qid = meta["qid"]
        entry = by_qid.setdefault(qid, {"reactions": {}})
        entry["reactions"][meta["label"]] = {
            "choice_text": meta["choice_text"],
            "reactions": reactions,
        }

    # --- Dump JSONL ---
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as f:
        for q in mcqs:
            qid = q["question_id"]
            if qid not in by_qid:
                continue
            end_idx = int(q["end_index_in_shared_context"])
            raw_ctx = ctxs[q["shared_context_id"]]
            correct_label = q["correct_answer"].strip("()").lower()
            rec = {
                "question_id": qid,
                "persona_id": q["persona_id"],
                "qtype_canonical": q["qtype_canonical"],
                "topic": q["topic"],
                "user_question": q["user_question_or_message"],
                "correct_answer": correct_label,
                "choices": [
                    {
                        "label": L,
                        "is_correct": L == correct_label,
                        "choice_text": by_qid[qid]["reactions"][L]["choice_text"],
                        "reactions": by_qid[qid]["reactions"][L]["reactions"],
                    }
                    for L in ["a", "b", "c", "d"]
                    if L in by_qid[qid]["reactions"]
                ],
                "gt_followup": extract_gt_window(raw_ctx, end_idx),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[gen] wrote {args.out_jsonl}  "
          f"({len(by_qid)} MCQs × 4 choices × {args.n_samples} samples)")


if __name__ == "__main__":
    main()
