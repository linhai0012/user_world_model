"""Phase-2 eval (1b): semantic-level user-fit — generate + LLM judge.

Same five conditions as eval_user_nll.py:
  - base_demo, base_full, teacher_demo, teacher_full, student_demo

For each sample, generate one user-turn continuation under each condition
(vLLM, batched). Then a GPT-4o-mini judge scores {GT vs generation} on a
1-5 scale of similarity in content / style / preferences revealed.

vLLM setup:
  - Load base LLM once with `enable_lora=True` so both base_* conditions
    (no LoRA) and student_demo (with LoRA) share the same LLM instance.
  - Load teacher LLM separately.
  Total: 2 vLLM loads.

OpenAI judge:
  - gpt-4o-mini, temperature 0, 100 max tokens
  - Parallel with ThreadPoolExecutor (16 workers by default)
  - api_key.txt at repo root or OPENAI_API_KEY env var (same as eval_mcq.py)

Usage:
    python dynamic_usersim/student_opd/eval_user_gen_judge.py \\
        --persona-id 14 --teacher-path $R3 \\
        --lora-path dynamic_usersim/outputs/lora_pid14_r32_ep1_r3teacher \\
        --num-samples 50 --tensor-parallel-size 4 \\
        --out-json dynamic_usersim/outputs/user_gen_pid14.json
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
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from vllm.lora.request import LoRARequest

_HERE = Path(__file__).resolve().parent
_TSFT = _HERE.parent / "teacher_sft"
for p in (_HERE, _TSFT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from config import MAX_SEQ_LEN, MODEL_NAME  # noqa: E402


# (name, model_key, context_mode, needs_lora)
CONDITIONS = [
    ("base_demo",    "base",    "demo-only", False),
    ("base_full",    "base",    "full",      False),
    ("teacher_demo", "teacher", "demo-only", False),
    ("teacher_full", "teacher", "full",      False),
    ("student_demo", "base",    "demo-only", True),   # base LLM + LoRA
]


JUDGE_MODEL = "gpt-4o-mini"
JUDGE_TEMPLATE = """You are evaluating how well a language model simulates a specific \
user's response. Given the real user's response (ground truth) and a \
model's simulated response, rate similarity on a 1-5 scale:

5 - Essentially the same content, preferences, and style
4 - Same topic and expressed preferences, stylistic differences
3 - Related topic but diverges in content or preferences
2 - Same general area, but different preferences or reactions
1 - Unrelated, off-topic, or contradictory

Real user's response:
\"\"\"{gt}\"\"\"

Model's simulated response:
\"\"\"{gen}\"\"\"

First output a single digit 1-5, then on the next line give a one-sentence \
rationale. Format exactly:
Score: <1-5>
Reason: <one sentence>
"""


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--persona-id", type=str, required=True)
    ap.add_argument("--base-model", type=str, default=MODEL_NAME)
    ap.add_argument("--teacher-path", type=Path, required=True)
    ap.add_argument("--lora-path", type=Path, required=True)
    ap.add_argument("--data-path", type=Path, default=None)
    ap.add_argument("--num-samples", type=int, default=50,
                    help="50 is a decent default — generates fast, "
                         "judge API is cheap but non-trivial.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--tensor-parallel-size", type=int, default=4)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--lora-rank", type=int, default=32,
                    help="Must match --lora-rank used during training.")
    ap.add_argument("--judge-workers", type=int, default=16)
    ap.add_argument("--judge-model", type=str, default=JUDGE_MODEL)
    ap.add_argument("--skip-judge", action="store_true",
                    help="Generate only; skip OpenAI judge (for debugging).")
    ap.add_argument("--out-json", type=Path, required=True)
    return ap.parse_args()


# ---------- Prompt construction ----------

def build_prompt_ids(sample: dict, tokenizer, context_mode: str) -> list[int]:
    if context_mode == "demo-only":
        msgs = [
            {"role": "system", "content": sample["demographics"]},
            {"role": "assistant", "content": sample["chatbot_prev"]},
        ]
    elif context_mode == "full":
        msgs = sample["history_messages"]
    else:
        raise ValueError(context_mode)
    ids: list[int] = []
    for m in msgs:
        s = f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        ids.extend(tokenizer.encode(s, add_special_tokens=False))
    ids.extend(
        tokenizer.encode("<|im_start|>user\n", add_special_tokens=False)
    )
    return ids


# ---------- vLLM generation ----------

def truncate_left(ids: list[int], max_len: int) -> list[int]:
    return ids if len(ids) <= max_len else ids[-max_len:]


def generate_for_conditions(llm: LLM, sampling: SamplingParams,
                            prompts_ids: list[list[int]],
                            lora_request: LoRARequest | None) -> list[str]:
    inputs = [TokensPrompt(prompt_token_ids=ids) for ids in prompts_ids]
    kwargs = {}
    if lora_request is not None:
        kwargs["lora_request"] = lora_request
    outputs = llm.generate(inputs, sampling, **kwargs)
    texts: list[str] = []
    for o in outputs:
        text = o.outputs[0].text
        for marker in ("<|im_end|>", "<|endoftext|>"):
            if marker in text:
                text = text.split(marker, 1)[0]
        texts.append(text.strip())
    return texts


# ---------- OpenAI judge ----------

def get_openai_client(timeout: int = 60):
    from openai import OpenAI
    repo_root = Path(__file__).resolve().parents[2]
    key_path = repo_root / "api_key.txt"
    api_key = (key_path.read_text().strip() if key_path.exists()
               else os.environ.get("OPENAI_API_KEY", ""))
    if not api_key:
        raise RuntimeError("No OpenAI key: api_key.txt or OPENAI_API_KEY")
    return OpenAI(api_key=api_key, timeout=timeout)


SCORE_RE = re.compile(r"[Ss]core:\s*([1-5])")


def judge_one(client, model: str, gt: str, gen: str,
              max_retries: int = 3) -> tuple[int | None, str]:
    prompt = JUDGE_TEMPLATE.format(gt=gt[:1500], gen=gen[:1500])
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=120,
                temperature=0.0,
            )
            text = resp.choices[0].message.content or ""
            m = SCORE_RE.search(text)
            if not m:
                # fall back: first standalone digit 1-5
                m2 = re.search(r"\b([1-5])\b", text)
                score = int(m2.group(1)) if m2 else None
            else:
                score = int(m.group(1))
            return score, text
        except Exception as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
    return None, f"[judge error: {last_err}]"


def judge_all(tasks: list[dict], client, model: str, workers: int) -> None:
    """Populate each task dict with 'score' and 'reason' in-place."""
    def work(task):
        score, text = judge_one(client, model, task["gt"], task["gen"])
        task["score"] = score
        task["reason"] = text
        return task

    total = len(tasks)
    done = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(work, t) for t in tasks]
        for fut in as_completed(futures):
            fut.result()
            done += 1
            if done % 25 == 0 or done == total:
                print(f"  [judge] {done}/{total}  "
                      f"({time.time()-t0:.1f}s)", flush=True)


# ---------- Main ----------

def main() -> None:
    args = parse_args()

    # Data
    if args.data_path is None:
        repo_root = Path(__file__).resolve().parents[2]
        args.data_path = (
            repo_root / "dynamic_usersim" / "outputs"
            / f"opd_128k_pid{args.persona_id}_k3.jsonl"
        )
    all_samples = [
        json.loads(L)
        for L in args.data_path.open("r", encoding="utf-8")
    ]
    rng = random.Random(args.seed)
    if 0 < args.num_samples < len(all_samples):
        samples = rng.sample(all_samples, args.num_samples)
    else:
        samples = all_samples
    print(f"[eval] pid={args.persona_id}: {len(samples)} samples "
          f"of {len(all_samples)} total (seed={args.seed})")

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True
    )
    im_end_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]

    # Pre-tokenize prompts
    prompts_by_mode: dict[str, list[list[int]]] = {}
    for mode in ("demo-only", "full"):
        prompts_by_mode[mode] = [
            truncate_left(build_prompt_ids(s, tokenizer, mode),
                          args.max_seq_len)
            for s in samples
        ]

    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        stop_token_ids=[im_end_id],
        seed=args.seed,
    )

    # Store generations keyed by condition
    gens: dict[str, list[str]] = {}

    # ---- Pass 1: BASE LLM (handles base_demo / base_full / student_demo via LoRA) ----
    print(f"\n[vllm] loading BASE: {args.base_model}  (enable_lora=True)")
    t0 = time.time()
    llm_base = LLM(
        model=str(args.base_model),
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_seq_len,
        enable_lora=True,
        max_loras=1,
        max_lora_rank=args.lora_rank,
    )
    print(f"  loaded in {time.time()-t0:.1f}s")

    for cond_name, _, mode, needs_lora in CONDITIONS:
        if cond_name not in ("base_demo", "base_full", "student_demo"):
            continue
        print(f"\n[gen] {cond_name} ({mode}, lora={needs_lora})")
        t_cond = time.time()
        lora_req = None
        if needs_lora:
            lora_req = LoRARequest(
                lora_name=f"pid{args.persona_id}",
                lora_int_id=1,
                lora_path=str(args.lora_path),
            )
        gens[cond_name] = generate_for_conditions(
            llm_base, sampling, prompts_by_mode[mode], lora_req
        )
        print(f"  [gen {cond_name}] done in {time.time()-t_cond:.1f}s")

    del llm_base
    gc.collect()
    torch.cuda.empty_cache()

    # ---- Pass 2: TEACHER LLM ----
    print(f"\n[vllm] loading TEACHER: {args.teacher_path}")
    t0 = time.time()
    llm_teacher = LLM(
        model=str(args.teacher_path),
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_seq_len,
    )
    print(f"  loaded in {time.time()-t0:.1f}s")

    for cond_name, _, mode, _ in CONDITIONS:
        if cond_name not in ("teacher_demo", "teacher_full"):
            continue
        print(f"\n[gen] {cond_name} ({mode})")
        t_cond = time.time()
        gens[cond_name] = generate_for_conditions(
            llm_teacher, sampling, prompts_by_mode[mode], None
        )
        print(f"  [gen {cond_name}] done in {time.time()-t_cond:.1f}s")

    del llm_teacher
    gc.collect()
    torch.cuda.empty_cache()

    # ---- LLM judge ----
    per_condition: dict[str, list[dict]] = defaultdict(list)
    judge_tasks: list[dict] = []
    for cond_name, _, mode, needs_lora in CONDITIONS:
        for i, sample in enumerate(samples):
            entry = {
                "condition": cond_name,
                "sample_idx": i,
                "persona_id": sample["persona_id"],
                "context_id": sample["context_id"][:8],
                "session_idx": sample["session_idx"],
                "user_turn_idx": sample["user_turn_idx"],
                "chatbot_prev_snippet": sample["chatbot_prev"][:200],
                "gt": sample["user_response"],
                "gen": gens[cond_name][i],
                "score": None,
                "reason": "",
            }
            per_condition[cond_name].append(entry)
            if not args.skip_judge:
                judge_tasks.append(entry)

    if not args.skip_judge:
        print(f"\n[judge] scoring {len(judge_tasks)} (cond, sample) pairs "
              f"with {args.judge_model} ({args.judge_workers} workers)")
        client = get_openai_client()
        judge_all(judge_tasks, client, args.judge_model, args.judge_workers)

    # ---- Summary ----
    print("\n" + "=" * 72)
    print(f"SUMMARY — eval_user_gen_judge  pid={args.persona_id}  "
          f"n={len(samples)}")
    print("=" * 72)
    if not args.skip_judge:
        print(f"{'condition':<15}  {'mean_score':>10}  "
              f"{'score_dist (1..5)':>22}  {'valid/n':>8}")
        for cond_name, _, _, _ in CONDITIONS:
            entries = per_condition[cond_name]
            scores = [e["score"] for e in entries if e["score"] is not None]
            valid = len(scores)
            mean_s = sum(scores) / valid if valid else float("nan")
            dist = [sum(1 for s in scores if s == v) for v in range(1, 6)]
            print(f"{cond_name:<15}  {mean_s:>10.3f}  "
                  f"{str(dist):>22}  {valid:>4}/{len(entries):<3}")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps({
        "args": {k: str(v) for k, v in vars(args).items()},
        "n_samples": len(samples),
        "per_condition": {k: v for k, v in per_condition.items()},
    }, indent=2, ensure_ascii=False))
    print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
