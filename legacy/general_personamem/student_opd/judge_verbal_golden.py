"""LLM-judge of verbal Stage-1 outputs against the PersonaMem MCQ's
GOLDEN-SNIPPET — the past user utterance the MCQ was designed to
require recalling.

The MCQ CSV has `distance_to_ref_in_blocks` per row. Block = session
(bounded by system messages in PersonaMem). We use this to locate the
earlier session that contains the evidence for the correct answer, and
pull its first user turn(s) as the "golden snippet" GT.

Why this GT is better than gt_followup[1] (judge_verbal.py):
  - 100% coverage (every MCQ has distance_to_ref), not 13%.
  - Directly tests (1) recall: does our user-sim produce content that
    references the specific past user preference that the MCQ is built on?
  - Aligned with OPSD's optimization goal: compress user knowledge into
    student parameters. Golden snippet IS that knowledge.

Usage:
    python dynamic_usersim/student_opd/judge_verbal_golden.py \\
        --in-dir dynamic_usersim/outputs \\
        --configs base_vllm phase2_vllm r1b_vllm opsd \\
        --out-json dynamic_usersim/outputs/judge_verbal_golden_pid4.json \\
        --api-key-file api_key.txt \\
        --workers 16 \\
        --persona-id 4 --mcq-version 128k
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

_HERE = Path(__file__).resolve().parent
_DP = _HERE.parent / "data_prep"
for p in (_HERE, _DP):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from load_personamem import load_contexts, load_questions, strip_role_prefix  # noqa: E402


JUDGE_MODEL_DEFAULT = "gpt-4o-mini"
JUDGE_TEMPLATE = """You are evaluating whether a user-simulator correctly recalled a specific past preference / experience of the real user.

Below is (GOLDEN) an actual past utterance from this user (pulled from an earlier conversation with a chatbot). Then (GEN) is a response from a simulator trying to act like this user.

Rate on a 1-5 scale how well GEN's content/preferences are consistent with GOLDEN:

5 - GEN clearly references the exact preference/experience in GOLDEN, or says something that only makes sense if the user has that history.
4 - GEN is strongly related to GOLDEN's topic and consistent in preference direction.
3 - GEN is in a related area but could apply to many users, not specifically aligned with GOLDEN.
2 - GEN is on a different topic but not contradictory.
1 - GEN is unrelated or contradicts GOLDEN.

GOLDEN:
\"\"\"{gt}\"\"\"

GEN:
\"\"\"{gen}\"\"\"

First output a single digit 1-5, then on the next line give a one-sentence reason. Nothing else."""


FILE_MAP = {
    "base_vllm":   "verbal_pid4_base_vllm.jsonl",
    "phase2_vllm": "verbal_pid4_phase2_vllm.jsonl",
    "r1b_vllm":    "verbal_pid4_r1b_vllm.jsonl",
    "opsd":        "verbal_pid4_opsd.jsonl",
}


# ---------------------------------------------------------------------------
# Golden-snippet extraction
# ---------------------------------------------------------------------------
def split_sessions(raw_ctx: list[dict]) -> list[list[dict]]:
    """Each system message starts a new session."""
    sessions: list[list[dict]] = []
    current: list[dict] = []
    for m in raw_ctx:
        if m.get("role") == "system":
            if current:
                sessions.append(current)
            current = [m]
        else:
            current.append(m)
    if current:
        sessions.append(current)
    return sessions


def session_idx_for_msg(sessions: list[list[dict]], msg_idx: int) -> int:
    cum = 0
    for si, s in enumerate(sessions):
        if cum + len(s) > msg_idx:
            return si
        cum += len(s)
    return len(sessions) - 1


def extract_golden_snippet(
    raw_ctx: list[dict],
    end_index: int,
    distance_in_tokens: int,
    context_length_in_tokens: int,
    max_chars: int = 600,
    n_user_turns: int = 2,
) -> str:
    """Locate the reference position via distance_to_ref_in_tokens, then
    extract N user turns starting at (or closest after) that position.

    Token → char proxy: compute char_per_token = total_chars / total_tokens
    for this specific context, then ref_char_position = ref_token_position *
    char_per_token. Walk raw_ctx until cumulative chars reach that position;
    start collecting user turns from there.
    """
    msgs = raw_ctx[: end_index + 1]
    total_chars = sum(len(m.get("content", "")) for m in msgs)
    cpt = total_chars / max(context_length_in_tokens, 1)
    ref_tok_from_start = max(0, context_length_in_tokens - distance_in_tokens)
    ref_char_from_start = int(ref_tok_from_start * cpt)

    # Walk messages; find first message whose content straddles ref position.
    cum = 0
    start_idx = 0
    for i, m in enumerate(msgs):
        mlen = len(m.get("content", ""))
        if cum + mlen >= ref_char_from_start:
            start_idx = i
            break
        cum += mlen

    # From start_idx forward, collect first n user turns.
    user_contents: list[str] = []
    for m in msgs[start_idx:]:
        if m.get("role") == "user":
            c = strip_role_prefix(m.get("content", ""), "user").strip()
            if c:
                user_contents.append(c)
        if len(user_contents) >= n_user_turns:
            break
    snippet = " ".join(user_contents)
    if len(snippet) > max_chars:
        snippet = snippet[:max_chars].rstrip() + "…"
    return snippet


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------
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
            return {**task, "score": score, "raw": text.strip()}
        except Exception as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
    return {**task, "score": None, "raw": f"[error: {last_err}]"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path,
                    default=Path("dynamic_usersim/outputs"))
    ap.add_argument("--configs", nargs="+",
                    default=["base_vllm", "phase2_vllm", "r1b_vllm", "opsd"])
    ap.add_argument("--persona-id", default="4")
    ap.add_argument("--mcq-version", choices=["32k", "128k", "1M"],
                    default="128k")
    ap.add_argument("--max-snippet-chars", type=int, default=600)
    ap.add_argument("--n-user-turns", type=int, default=2)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--api-key-file", type=Path, default=Path("api_key.txt"))
    ap.add_argument("--model", default=JUDGE_MODEL_DEFAULT)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--dry-run-snippets", type=int, default=0,
                    help="if > 0, print N examples of (end_s, ref_s, snippet) and exit")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    # Load PersonaMem questions + shared contexts to get distance_to_ref
    print(f"[judge-golden] loading {args.mcq_version} MCQs + contexts")
    mcqs = load_questions(args.mcq_version)
    ctxs = load_contexts(args.mcq_version)
    # Index by qid for fast join
    mcq_by_qid = {q["question_id"]: q for q in mcqs
                   if q["persona_id"] == args.persona_id
                   and q["shared_context_id"] in ctxs}

    # Extract golden snippet for every MCQ we care about
    snippets: dict[str, str] = {}
    misses = 0
    sampled: list[tuple[str, int, int, int, str]] = []
    for qid, q in mcq_by_qid.items():
        raw_ctx = ctxs[q["shared_context_id"]]
        end_idx = int(q["end_index_in_shared_context"])
        dist_tok = int(q["distance_to_ref_in_tokens"])
        ctx_tok = int(q["context_length_in_tokens"])
        snip = extract_golden_snippet(
            raw_ctx, end_idx, dist_tok, ctx_tok,
            max_chars=args.max_snippet_chars,
            n_user_turns=args.n_user_turns,
        )
        snippets[qid] = snip
        if not snip:
            misses += 1
        if len(sampled) < args.dry_run_snippets:
            sampled.append((qid[:8], q["qtype_canonical"], q["topic"],
                            q["user_question_or_message"][:100],
                            dist_tok, ctx_tok,
                            snip[:220]))

    if args.dry_run_snippets > 0:
        print(f"\n--- first {args.dry_run_snippets} MCQs with golden snippet ---")
        for s in sampled:
            print(f"  qid={s[0]}  qtype={s[1]}  topic={s[2]}  dist_tok={s[4]}/{s[5]}")
            print(f"    user_Q: {s[3]!r}")
            print(f"    golden snippet: {s[6]!r}")
            print()
        print(f"{len(snippets)} total MCQs, {misses} with empty snippet")
        return

    print(f"[judge-golden] {len(snippets)} snippets extracted "
          f"({misses} empty = {misses/max(len(snippets),1)*100:.1f}%)")

    # API key
    key = os.environ.get("OPENAI_API_KEY")
    if not key and args.api_key_file.exists():
        key = args.api_key_file.read_text().strip()
    if not key:
        raise SystemExit("no OpenAI API key")
    from openai import OpenAI
    client = OpenAI(api_key=key)

    # Build tasks: per-config × per-MCQ, using correct-choice reaction
    tasks = []
    for cfg in args.configs:
        path = args.in_dir / FILE_MAP[cfg]
        if not path.exists():
            print(f"  WARN missing {path}"); continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                qid = rec["question_id"]
                gt = snippets.get(qid, "")
                if not gt:
                    continue
                correct = rec["correct_answer"].strip("()").lower()
                cc = next((c for c in rec["choices"] if c["label"] == correct), None)
                if cc is None:
                    continue
                gen = (cc["reactions"][0] or "").strip() if cc["reactions"] else ""
                if not gen:
                    continue
                tasks.append({
                    "config": cfg,
                    "qid": qid,
                    "qtype": rec.get("qtype_canonical", ""),
                    "correct_label": correct,
                    "choice_text": cc.get("choice_text", ""),
                    "gt": gt,
                    "gen": gen,
                })
    print(f"[judge-golden] {len(tasks)} (config, MCQ) pairs queued")
    by_cfg_n = defaultdict(int)
    for t in tasks:
        by_cfg_n[t["config"]] += 1
    for cfg, n in by_cfg_n.items():
        print(f"  {cfg:<14} {n}")

    # Parallel calls
    results = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(judge_one, t, client, args.model) for t in tasks]
        done = 0
        for f in as_completed(futs):
            results.append(f.result())
            done += 1
            if done % 100 == 0 or done == len(futs):
                print(f"  [{done}/{len(futs)}]  {time.time()-t0:.0f}s",
                      flush=True)

    # Aggregate
    print("\n" + "=" * 80)
    print("SUMMARY — golden-snippet judge (1=unrelated/contradict, 5=exact recall)")
    print("=" * 80)
    by_cfg: dict[str, list[int]] = defaultdict(list)
    by_ck: dict[tuple, list[int]] = defaultdict(list)
    fails = defaultdict(int)
    for r in results:
        cfg = r["config"]
        if isinstance(r.get("score"), int):
            by_cfg[cfg].append(r["score"])
            by_ck[(cfg, r["qtype"])].append(r["score"])
        else:
            fails[cfg] += 1

    print(f"{'config':<14} {'mean':>8} {'n':>6}  1s   2s   3s   4s   5s   parse_fail")
    for cfg in args.configs:
        sc = by_cfg.get(cfg, [])
        n = len(sc)
        m = sum(sc) / n if n else float("nan")
        dist = [sum(1 for s in sc if s == k) for k in range(1, 6)]
        pf = fails.get(cfg, 0)
        print(f"{cfg:<14} {m:>8.3f} {n:>6}  {dist[0]:>3}  {dist[1]:>3}  "
              f"{dist[2]:>3}  {dist[3]:>3}  {dist[4]:>3}   {pf:>10}")

    print("\n--- per qtype ---")
    qtypes = sorted({qt for cfg, qt in by_ck.keys()})
    print(f"{'qtype':<22}" + "".join(f"{c:>14}" for c in args.configs))
    for qt in qtypes:
        cells = []
        for cfg in args.configs:
            ss = by_ck.get((cfg, qt), [])
            mm = sum(ss) / len(ss) if ss else float("nan")
            cells.append(f"{mm:.2f}(n={len(ss)})" if ss else "      —     ")
        print(f"{qt:<22}" + "".join(f"{c:>14}" for c in cells))

    # Dump
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": args.model,
        "summary": {
            cfg: {
                "n": len(by_cfg.get(cfg, [])),
                "mean": (sum(by_cfg.get(cfg, [])) / len(by_cfg.get(cfg, []))
                         if by_cfg.get(cfg) else None),
                "distribution": [sum(1 for s in by_cfg.get(cfg, []) if s == k)
                                 for k in range(1, 6)],
                "parse_fail": fails.get(cfg, 0),
            }
            for cfg in args.configs
        },
        "results": results,
    }
    with args.out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\n[judge-golden] wrote {args.out_json}")


if __name__ == "__main__":
    main()
