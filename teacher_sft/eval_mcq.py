"""End-to-end MCQ evaluation: UserSim reactions + LLM agent decision.

Plan §9.3 pipeline:
  1. For each MCQ, build context up to end_index_in_shared_context
  2. For each of the 4 choices:
       context + "user: <question>" + "assistant: <choice>" -> UserSim
       UserSim generates the user's natural reaction (256 tokens)
  3. Feed all 4 (choice, reaction) pairs to an LLM agent
  4. Agent picks a/b/c/d
  5. Compare with correct_answer; compute accuracy

Run once with --usersim-model = base Qwen3-4B, once with the SFT checkpoint;
compare accuracies. Same MCQ subset, same agent — only UserSim differs.

4-GPU DP via torchrun: each rank handles mcqs[rank::world_size], generates
its reactions, calls the agent for its MCQs, then results are gathered on
rank 0 for the summary.

Agent uses OpenAI API (default gpt-4o-mini — fast, cheap, capable enough).

Launch:
    torchrun --standalone --nnodes=1 --nproc_per_node=4 \
        dynamic_usersim/teacher_sft/eval_mcq.py \
        --usersim-model $SCRATCHDIR/P-OPSD/teacher_sft_128k/checkpoint-50 \
        --num-mcqs 20 --mcq-version 128k \
        --out-json dynamic_usersim/outputs/mcq_sft_ckpt50.json

Run again with --usersim-model Qwen/Qwen3-4B for the base comparison.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

_HERE = Path(__file__).resolve().parent
_DP = _HERE.parent / "data_prep"
for p in (_HERE, _DP):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from config import MAX_SEQ_LEN, MODEL_NAME  # noqa: E402
from load_personamem import load_contexts, load_questions, strip_role_prefix  # noqa: E402
from tokenize_teacher_sft import SFTTokenizer  # noqa: E402


# ---------------------------------------------------------------------------
# OpenAI agent
# ---------------------------------------------------------------------------
DEFAULT_AGENT_MODEL = "gpt-4o-mini"
AGENT_MAX_TOKENS = 200
AGENT_RETRIES = 4
AGENT_TIMEOUT = 60


def get_openai_client():
    from openai import OpenAI
    # Key: api_key.txt at repo root (gitignored), or env var.
    repo_root = Path(__file__).resolve().parents[2]
    key_path = repo_root / "api_key.txt"
    api_key = (key_path.read_text().strip() if key_path.exists()
               else os.environ.get("OPENAI_API_KEY", ""))
    if not api_key:
        raise RuntimeError("No OpenAI key: put at api_key.txt or set OPENAI_API_KEY")
    return OpenAI(api_key=api_key, timeout=AGENT_TIMEOUT)


def call_agent(client, model: str, question: str,
               choices_and_reactions: list[tuple[str, str, str]]) -> tuple[str, str]:
    """Return (letter_or_'?', raw_response_text)."""
    prompt_lines = [
        f'A user asked: "{question}"',
        "",
        "Four different responses were proposed below. The user reacted to "
        "each one naturally. Based on the reactions, determine which "
        "response best matches the user's preferences and situation.",
        "",
    ]
    for label, choice, reaction in choices_and_reactions:
        prompt_lines.append(f"Response ({label}): {choice}")
        prompt_lines.append(f"User's reaction to ({label}): {reaction}")
        prompt_lines.append("")
    prompt_lines.append(
        "Analyze each reaction for signs of enthusiasm, agreement, personal "
        "connection, or correction. Then select the response that the user "
        "most clearly prefers. First give a brief (1-3 sentence) analysis, "
        "then output your final answer on its own line as: Answer: (X)"
    )
    prompt = "\n".join(prompt_lines)

    last_err: Exception | None = None
    for attempt in range(AGENT_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=AGENT_MAX_TOKENS,
                temperature=0.0,
            )
            text = resp.choices[0].message.content or ""
            m = re.search(r"[Aa]nswer:\s*\(?([a-dA-D])\)?", text)
            letter = m.group(1).lower() if m else "?"
            return letter, text
        except Exception as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
    return "?", f"[agent error: {last_err}]"


# ---------------------------------------------------------------------------
# MCQ data prep
# ---------------------------------------------------------------------------
def parse_choices(all_options: str) -> list[tuple[str, str]]:
    """Return [(label, choice_text_without_prefix), ...] for 4 choices.

    The CSV's all_options field is inconsistent: some rows use JSON style
    (double-quoted strings), others use Python repr (single-quoted with
    escaped apostrophes). ast.literal_eval handles both — unlike json.loads
    which rejects single-quoted sequences.
    """
    import ast
    try:
        arr = ast.literal_eval(all_options)
    except (ValueError, SyntaxError):
        # Last-ditch: try json.loads for anything literal_eval can't handle
        arr = json.loads(all_options)
    out: list[tuple[str, str]] = []
    for s in arr:
        m = re.match(r"^\s*\(([a-dA-D])\)\s*(.*)", s, flags=re.DOTALL)
        if m:
            out.append((m.group(1).lower(), m.group(2).strip()))
        else:
            out.append(("?", s.strip()))
    return out


def build_context_messages(raw_ctx: list[dict], end_index: int,
                           last_n_sessions: int | None) -> list[dict]:
    """Slice and clean shared_context messages up to end_index.

    Strips the 'User: '/'Assistant: ' prefixes that appear in raw content.
    If last_n_sessions is given, truncate to keep only the most recent N
    sessions (delimited by 'system' messages). Useful to match the K used
    during teacher SFT training (out-of-distribution context lengths can
    hurt a K-limited teacher's performance).
    """
    sliced = raw_ctx[:end_index]
    cleaned = [{"role": m["role"],
                "content": strip_role_prefix(m["content"], m["role"])}
               for m in sliced]
    if last_n_sessions is None:
        return cleaned
    # session boundary = each system message
    sys_idxs = [i for i, m in enumerate(cleaned) if m["role"] == "system"]
    if len(sys_idxs) <= last_n_sessions:
        return cleaned
    cut = sys_idxs[-last_n_sessions]
    return cleaned[cut:]


def _encode_msgs_plain(sft_tok: SFTTokenizer, msgs: list[dict]) -> list[int]:
    """Encode a message list using the SAME format as training.

    Avoids tok.apply_chat_template because Qwen3's default chat template
    hard-codes `<think>\\n\\n</think>\\n\\n` before every assistant
    message's content — this contaminates the prompt with thinking-mode
    cues that then leak into generated user turns (observed in v1 run:
    26% of base reactions contained stray <think> tags).

    Our training format (from SFTTokenizer._encode_message):
        <|im_start|>{role}\\n{content}<|im_end|>\\n
    No thinking markers anywhere.
    """
    ids: list[int] = []
    for m in msgs:
        chunk = sft_tok._encode_message(m["role"], m["content"], False)
        ids.extend(chunk.input_ids)
    return ids


def build_usersim_input_ids(sft_tok: SFTTokenizer, context_msgs: list[dict],
                            question: str, choice_text: str,
                            max_len: int, reserve: int = 300) -> list[int]:
    """Build prompt ending in '<|im_start|>user\\n' ready for generation.

    Conversation structure, preserving flow coherence:
      ... assistant -> user(question) -> assistant(choice) -> user(react)
    If the last context message is ALREADY user (true for ~68% of MCQs in
    128k, where PersonaMem's end_index sits inside a user monologue), we
    merge the MCQ question into that existing user turn instead of adding
    a separate one — avoids the awkward user->user->assistant flow while
    preserving all prior content.

    Trims oldest context messages greedily until total + reserve fits max_len.
    """
    msgs = [dict(m) for m in context_msgs]   # shallow copy
    if msgs and msgs[-1]["role"] == "user":
        msgs[-1]["content"] = msgs[-1]["content"].rstrip() + "\n\n" + question
    else:
        msgs.append({"role": "user", "content": question})
    msgs.append({"role": "assistant", "content": choice_text})

    user_prime = sft_tok.tok.encode("<|im_start|>user\n", add_special_tokens=False)
    while True:
        ids = _encode_msgs_plain(sft_tok, msgs) + user_prime
        if len(ids) + reserve <= max_len:
            return ids
        if len(msgs) <= 2:
            return ids[-(max_len - reserve):]  # last resort
        msgs.pop(0)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
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


def load_usersim(path: str | Path) -> torch.nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        str(path),
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    model.config.use_cache = True
    model.eval()
    model.to(torch.cuda.current_device())
    return model


def generate_reaction(model, tok, ids: list[int], max_new_tokens: int,
                      temperature: float, top_p: float) -> str:
    device = torch.cuda.current_device()
    ids_t = torch.tensor([ids], dtype=torch.long, device=device)
    im_end = tok.encode("<|im_end|>", add_special_tokens=False)[0]
    with torch.no_grad():
        out = model.generate(
            input_ids=ids_t,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=im_end,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    gen = out[0, ids_t.size(1):].tolist()
    text = tok.decode(gen, skip_special_tokens=False)
    for marker in ["<|im_end|>", "<|endoftext|>"]:
        if marker in text:
            text = text.split(marker, 1)[0]
    return text.strip()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--usersim-model", type=str, required=True,
                    help="HF name (e.g. Qwen/Qwen3-4B) or local checkpoint dir")
    ap.add_argument("--mcq-version", choices=["32k", "128k", "1M"], default="32k",
                    help="32k fits in Qwen3-4B's native RoPE range (fastest, "
                         "cleanest signal). 128k requires RoPE extrapolation.")
    ap.add_argument("--num-mcqs", type=int, default=20)
    ap.add_argument("--agent-model", default=DEFAULT_AGENT_MODEL,
                    help="OpenAI model id for the agent")
    ap.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--last-n-sessions", type=int, default=None,
                    help="Cap context to most recent N sessions (match SFT K).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-json", type=Path, required=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    assert torch.cuda.is_available()
    rank, world_size, is_dist = init_dist()

    # ---- Load MCQ + contexts ----
    if rank == 0:
        print(f"[eval] loading MCQs + contexts ({args.mcq_version})")
    mcqs = load_questions(args.mcq_version)
    ctxs = load_contexts(args.mcq_version)
    rng = random.Random(args.seed)
    # Keep only MCQs whose shared_context we have
    mcqs = [q for q in mcqs if q["shared_context_id"] in ctxs]
    sample = rng.sample(mcqs, min(args.num_mcqs, len(mcqs)))
    if rank == 0:
        print(f"[eval] sampled {len(sample)} of {len(mcqs)} eligible MCQs "
              f"(seed={args.seed}, world_size={world_size})")

    # ---- Tokenizer + liger + model ----
    # Use our SFTTokenizer — it owns the _encode_message format that matches
    # training exactly. tok is the underlying HF tokenizer for decode/pad id.
    sft_tok = SFTTokenizer(model_name=MODEL_NAME, max_len=args.max_seq_len)
    tok = sft_tok.tok
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    apply_liger()
    if rank == 0:
        print(f"[eval] loading UserSim from {args.usersim_model}")
    model = load_usersim(args.usersim_model)

    # Each rank handles a slice
    my_mcqs = sample[rank::world_size]

    # Set up OpenAI client on every rank (independent calls)
    client = get_openai_client()

    local_results: list[dict] = []
    t0 = time.time()
    for i, q in enumerate(my_mcqs):
        t_mcq = time.time()
        ctx_msgs = build_context_messages(
            ctxs[q["shared_context_id"]],
            int(q["end_index_in_shared_context"]),
            args.last_n_sessions,
        )
        question = q["user_question_or_message"]
        choices = parse_choices(q["all_options"])

        # Generate reactions (4 per MCQ)
        reactions: list[tuple[str, str, str]] = []
        for label, choice_text in choices:
            ids = build_usersim_input_ids(
                sft_tok, ctx_msgs, question, choice_text,
                max_len=args.max_seq_len,
                reserve=args.max_new_tokens + 50,
            )
            reaction = generate_reaction(
                model, tok, ids, args.max_new_tokens,
                args.temperature, args.top_p,
            )
            reactions.append((label, choice_text, reaction))

        # Agent decision
        pred_letter, agent_text = call_agent(
            client, args.agent_model, question, reactions,
        )
        correct = q["correct_answer"].strip("()").lower()
        is_correct = (pred_letter == correct)

        local_results.append({
            "global_idx": sample.index(q),
            "persona_id": q["persona_id"],
            "question_id": q["question_id"],
            "qtype_canonical": q["qtype_canonical"],
            "topic": q["topic"],
            "question": question,
            "correct_answer": correct,
            "pred_answer": pred_letter,
            "is_correct": is_correct,
            "reactions": [{"label": lbl, "choice": ch, "reaction": rx}
                          for lbl, ch, rx in reactions],
            "agent_response": agent_text,
            "wall_s": time.time() - t_mcq,
        })
        print(f"  [rank{rank}] {i+1}/{len(my_mcqs)} "
              f"pid={q['persona_id']} qtype={q['qtype_canonical']} "
              f"pred={pred_letter} correct={correct} "
              f"{'OK' if is_correct else 'MISS'} "
              f"({time.time()-t_mcq:.1f}s)", flush=True)

    # ---- Gather to rank 0 ----
    if is_dist:
        gathered = [None] * world_size if rank == 0 else None
        dist.gather_object(local_results, gathered, dst=0)
        if rank == 0:
            all_results = [x for lst in gathered for x in lst]
            all_results.sort(key=lambda d: d["global_idx"])
        else:
            all_results = local_results
    else:
        all_results = local_results

    del model
    gc.collect()
    torch.cuda.empty_cache()

    if rank != 0:
        if is_dist:
            dist.barrier()
            dist.destroy_process_group()
        return

    # ---- Report ----
    n = len(all_results)
    n_correct = sum(1 for r in all_results if r["is_correct"])
    acc = n_correct / max(n, 1)
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"UserSim:     {args.usersim_model}")
    print(f"Agent:       {args.agent_model}")
    print(f"MCQs:        {n}  (correct {n_correct}, acc {acc:.3f})")
    print(f"Wall clock:  {time.time()-t0:.1f}s")

    # By question type
    from collections import Counter, defaultdict
    by_type_correct: dict[str, int] = defaultdict(int)
    by_type_total: dict[str, int] = defaultdict(int)
    for r in all_results:
        by_type_total[r["qtype_canonical"]] += 1
        if r["is_correct"]:
            by_type_correct[r["qtype_canonical"]] += 1
    print("\n--- by question type ---")
    for qt in sorted(by_type_total):
        t = by_type_total[qt]
        c = by_type_correct[qt]
        print(f"  {qt:22s} {c:3d}/{t:3d}  {c/t:.3f}")

    print("\n--- per-MCQ ---")
    print(f"{'pid':>3} {'qtype':>18} {'topic':>22} {'correct':>7} {'pred':>4} {'ok':>3}")
    for r in all_results:
        print(f"{r['persona_id']:>3} {r['qtype_canonical']:>18} "
              f"{r['topic']:>22} {r['correct_answer']:>7} "
              f"{r['pred_answer']:>4} {'Y' if r['is_correct'] else 'N':>3}")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps({
        "args": {k: str(v) for k, v in vars(args).items()},
        "accuracy": acc,
        "n_total": n,
        "n_correct": n_correct,
        "results": all_results,
    }, indent=2, ensure_ascii=False))
    print(f"\nwrote {args.out_json}")

    if is_dist:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
