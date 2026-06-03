"""LLM-judge of verbal Stage-1 outputs against the GT next-user-turn.

For each MCQ:
    For each CORRECT-choice reaction (one per model config):
        gpt-4o-mini rates similarity to gt_followup[1] on 1-5 scale
        (mirrors §10.4 eval_user_gen_judge.py rubric).

Inputs: N verbal jsonls produced by eval_mcq_verbal_gen.py. Each record has:
    choices[*].reactions[0]  — model's user-voice continuation
    gt_followup[1].content    — the real user's turn that came right after
                                 chatbot_response at end_index+1 (if present)

Outputs: one JSON per config with per-MCQ scores + summary means.

ONLY scores rows where:
  - the config emitted a non-empty reaction for the correct choice
  - gt_followup[1] exists (some end_index = end of context → no followup)

Usage:
    python dynamic_usersim/student_opd/judge_verbal.py \\
        --in-dir dynamic_usersim/outputs \\
        --configs base_vllm phase2_vllm r1b_vllm opsd \\
        --out-json dynamic_usersim/outputs/judge_verbal_pid4.json \\
        --api-key-file api_key.txt \\
        --workers 16
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


JUDGE_MODEL = "gpt-4o-mini"
JUDGE_TEMPLATE = """You are evaluating how well a language model simulates a specific user's response. Given the real user's response (ground truth) and a model's simulated response, rate similarity on a 1-5 scale:

5 - Essentially the same content, preferences, and style
4 - Same topic and expressed preferences, stylistic differences
3 - Related topic but diverges in content or preferences
2 - Same general area, but different preferences or reactions
1 - Unrelated, off-topic, or contradictory

Real user's response:
\"\"\"{gt}\"\"\"

Model's simulated response:
\"\"\"{gen}\"\"\"

First output a single digit 1-5, then on the next line give a one-sentence reason. Nothing else."""


FILE_MAP = {
    "base_vllm":   "verbal_pid4_base_vllm.jsonl",
    "phase2_vllm": "verbal_pid4_phase2_vllm.jsonl",
    "r1b_vllm":    "verbal_pid4_r1b_vllm.jsonl",
    "opsd":        "verbal_pid4_opsd.jsonl",
}


def load_jsonl(path: Path) -> dict[str, dict]:
    out = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["question_id"]] = r
    return out


def extract_task(qid: str, rec: dict) -> dict | None:
    """Return {qid, qtype, gt, gen, correct_label} or None if unusable."""
    correct = rec.get("correct_answer", "").strip("()").lower()
    correct_choice = next(
        (c for c in rec.get("choices", []) if c["label"] == correct), None
    )
    if correct_choice is None:
        return None
    reactions = correct_choice.get("reactions") or [""]
    gen = (reactions[0] or "").strip()
    if not gen:
        return None
    gt_followup = rec.get("gt_followup") or []
    # gt_followup[0] = real assistant reply, gt_followup[1] = real user followup
    gt = ""
    for m in gt_followup:
        if m.get("role") == "user":
            gt = m.get("content", "").strip()
            break
    if not gt:
        return None
    return {
        "qid": qid,
        "qtype": rec.get("qtype_canonical", ""),
        "topic": rec.get("topic", ""),
        "user_question": rec.get("user_question", ""),
        "correct_label": correct,
        "choice_text": correct_choice.get("choice_text", ""),
        "gt": gt,
        "gen": gen,
    }


def judge_one(task: dict, client, model: str, max_retries: int = 3) -> dict:
    prompt = JUDGE_TEMPLATE.format(gt=task["gt"], gen=task["gen"])
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.0,
            )
            text = resp.choices[0].message.content or ""
            m = re.search(r"\b([1-5])\b", text)
            score = int(m.group(1)) if m else None
            reason = text.strip()
            return {**task, "score": score, "raw": reason}
        except Exception as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
    return {**task, "score": None, "raw": f"[error: {last_err}]"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, default=Path("dynamic_usersim/outputs"))
    ap.add_argument("--configs", nargs="+",
                    default=["base_vllm", "phase2_vllm", "r1b_vllm", "opsd"])
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--api-key-file", type=Path, default=Path("api_key.txt"))
    ap.add_argument("--model", default=JUDGE_MODEL)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--num-mcqs", type=int, default=-1,
                    help="cap MCQs per config for a quick sniff (-1 = all)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    # API key
    key_env = os.environ.get("OPENAI_API_KEY")
    if not key_env and args.api_key_file.exists():
        key_env = args.api_key_file.read_text().strip()
    if not key_env:
        raise SystemExit("no OpenAI API key (checked OPENAI_API_KEY + " + str(args.api_key_file) + ")")
    from openai import OpenAI
    client = OpenAI(api_key=key_env)

    # Load + build task lists per config
    per_config_data = {}
    for cfg in args.configs:
        path = args.in_dir / FILE_MAP[cfg]
        if not path.exists():
            print(f"[judge] WARN missing: {path}")
            continue
        per_config_data[cfg] = load_jsonl(path)

    # Tasks = (cfg, qid) pairs where both reaction + gt are usable
    tasks = []
    for cfg, recs in per_config_data.items():
        items = list(recs.items())
        if args.num_mcqs > 0:
            items = items[: args.num_mcqs]
        for qid, rec in items:
            t = extract_task(qid, rec)
            if t is not None:
                t["config"] = cfg
                tasks.append(t)
    print(f"[judge] {len(tasks)} (config, MCQ) tasks queued "
          f"across {len(per_config_data)} configs")
    # Count usable per config
    by_cfg = defaultdict(int)
    for t in tasks:
        by_cfg[t["config"]] += 1
    for cfg, n in by_cfg.items():
        print(f"  {cfg:<14} {n}")

    # Parallel judge calls
    results: list[dict] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(judge_one, t, client, args.model) for t in tasks]
        done = 0
        for f in as_completed(futs):
            results.append(f.result())
            done += 1
            if done % 50 == 0 or done == len(futs):
                print(f"  [{done}/{len(futs)}]  {time.time()-t0:.0f}s elapsed",
                      flush=True)

    # Aggregate
    print("\n" + "=" * 64)
    print("SUMMARY — mean judge score (1=unrelated, 5=identical)")
    print("=" * 64)
    by_cfg_scores: dict[str, list[int]] = defaultdict(list)
    by_cfg_qtype: dict[tuple, list[int]] = defaultdict(list)
    n_parse_fail = defaultdict(int)
    for r in results:
        cfg = r["config"]
        if isinstance(r.get("score"), int):
            by_cfg_scores[cfg].append(r["score"])
            by_cfg_qtype[(cfg, r["qtype"])].append(r["score"])
        else:
            n_parse_fail[cfg] += 1

    print(f"{'config':<14} {'mean':>8} {'n':>6} {'1s':>4} {'2s':>4} {'3s':>4} {'4s':>4} {'5s':>4} {'parse_fail':>12}")
    for cfg in args.configs:
        scores = by_cfg_scores.get(cfg, [])
        n = len(scores)
        mean = sum(scores) / n if n else float("nan")
        dist = [sum(1 for s in scores if s == k) for k in range(1, 6)]
        pf = n_parse_fail.get(cfg, 0)
        print(f"{cfg:<14} {mean:>8.3f} {n:>6}  {dist[0]:>3} {dist[1]:>3} "
              f"{dist[2]:>3} {dist[3]:>3} {dist[4]:>3}  {pf:>10}")

    print("\n--- per qtype ---")
    qtypes = sorted({qt for cfg, qt in by_cfg_qtype.keys()})
    hdr = f"{'qtype':<22}" + "".join(f"{c:>12}" for c in args.configs)
    print(hdr)
    for qt in qtypes:
        cells = []
        for cfg in args.configs:
            ss = by_cfg_qtype.get((cfg, qt), [])
            m = sum(ss) / len(ss) if ss else float("nan")
            cells.append(f"{m:.2f} (n={len(ss)})" if ss else "        —  ")
        print(f"{qt:<22}" + "".join(f"{c:>12}" for c in cells))

    # Dump
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps({
        "model": args.model,
        "summary": {
            cfg: {
                "n": len(by_cfg_scores.get(cfg, [])),
                "mean": (sum(by_cfg_scores.get(cfg, [])) / len(by_cfg_scores.get(cfg, []))
                         if by_cfg_scores.get(cfg) else None),
                "distribution": [sum(1 for s in by_cfg_scores.get(cfg, []) if s == k)
                                 for k in range(1, 6)],
                "parse_fail": n_parse_fail.get(cfg, 0),
            }
            for cfg in args.configs
        },
        "results": results,
    }, indent=2, ensure_ascii=False))
    print(f"\n[judge] wrote {args.out_json}")


if __name__ == "__main__":
    main()
