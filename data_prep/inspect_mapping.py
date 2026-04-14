"""Follow-up inspection: persona <-> context <-> session mapping.

Resolves:
  - How many unique personas across each version?
  - How are contexts distributed per persona / topic?
  - What delimits a 'session' inside a shared_context message list?
  - Do 32k / 128k / 1M versions share personas / contexts?
  - What persona_ids and topics exist?
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "personamem_v1"


def load_questions(name: str) -> list[dict]:
    with (DATA / name).open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def load_contexts(name: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    with (DATA / name).open("r", encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            out.update(rec)
    return out


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def analyze_questions(tag: str, rows: list[dict]) -> None:
    section(f"QUESTIONS: {tag}  ({len(rows)} rows)")
    personas = Counter(r["persona_id"] for r in rows)
    topics = Counter(r["topic"] for r in rows)
    qtypes = Counter(r["question_type"] for r in rows)
    ctx_ids = Counter(r["shared_context_id"] for r in rows)

    print(f"unique persona_ids: {len(personas)}  -> {sorted(personas.keys(), key=int)}")
    print(f"unique topics ({len(topics)}): {sorted(topics.keys())}")
    print(f"unique shared_context_ids: {len(ctx_ids)}")
    print(f"question_types ({len(qtypes)}):")
    for k, v in qtypes.most_common():
        print(f"  {k:50s} {v}")

    # How many contexts per persona?
    per_persona_ctx: dict[str, set[str]] = defaultdict(set)
    per_persona_topic: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        per_persona_ctx[r["persona_id"]].add(r["shared_context_id"])
        per_persona_topic[r["persona_id"]].add(r["topic"])
    ctx_counts = [len(s) for s in per_persona_ctx.values()]
    topic_counts = [len(s) for s in per_persona_topic.values()]
    print(f"contexts_per_persona: min={min(ctx_counts)} max={max(ctx_counts)} "
          f"mean={sum(ctx_counts)/len(ctx_counts):.2f}")
    print(f"topics_per_persona:   min={min(topic_counts)} max={max(topic_counts)} "
          f"mean={sum(topic_counts)/len(topic_counts):.2f}")

    # end_index distribution
    end_idxs = sorted(set(int(r["end_index_in_shared_context"]) for r in rows))
    print(f"distinct end_index values: {len(end_idxs)}  "
          f"(range {min(end_idxs)}..{max(end_idxs)})")
    print(f"  first 20: {end_idxs[:20]}")
    print(f"  last  10: {end_idxs[-10:]}")

    # distance_to_ref_in_blocks — this is supposedly 'sessions back'
    dist_blocks = Counter(r["distance_to_ref_in_blocks"] for r in rows)
    print(f"distance_to_ref_in_blocks distribution: "
          f"{sorted(dist_blocks.items(), key=lambda x: int(x[0]))}")


def analyze_contexts(tag: str, ctx: dict[str, list[dict]], qrows: list[dict]) -> None:
    section(f"CONTEXTS: {tag}  ({len(ctx)} records)")

    # message count distribution
    lens = [len(v) for v in ctx.values()]
    print(f"messages_per_context: min={min(lens)} max={max(lens)} "
          f"mean={sum(lens)/len(lens):.1f}")

    # How many of the context_ids appear in questions csv?
    qctx_ids = set(r["shared_context_id"] for r in qrows)
    jctx_ids = set(ctx.keys())
    print(f"context ids in questions csv: {len(qctx_ids)}")
    print(f"context ids in jsonl:         {len(jctx_ids)}")
    print(f"intersection:                 {len(qctx_ids & jctx_ids)}")
    print(f"in jsonl but not csv:         {len(jctx_ids - qctx_ids)}")
    print(f"in csv but not jsonl:         {len(qctx_ids - jctx_ids)}")

    # For one specific persona, look at all its contexts
    first_pid = min((r["persona_id"] for r in qrows), key=int)
    pid_ctxs = [r["shared_context_id"] for r in qrows if r["persona_id"] == first_pid]
    pid_ctxs_unique = list(dict.fromkeys(pid_ctxs))  # preserve order, dedup
    print(f"\npersona_id={first_pid} has {len(pid_ctxs_unique)} unique context_ids:")
    for cid in pid_ctxs_unique[:5]:
        msgs = ctx.get(cid)
        n = len(msgs) if msgs else 0
        # count roles
        roles = Counter(m["role"] for m in (msgs or []))
        print(f"  {cid[:16]}...  len={n}  roles={dict(roles)}")


def find_session_delimiters(ctx: dict[str, list[dict]]) -> None:
    section("SESSION DELIMITER DETECTION")
    # Check first context's messages for patterns that might indicate
    # new sessions (e.g. system messages repeating, "Hi again" openers,
    # content lengths, role transitions)
    cid, msgs = next(iter(ctx.items()))
    print(f"inspecting context {cid[:16]}...  ({len(msgs)} messages)")
    roles = [m["role"] for m in msgs]
    role_seq = ""
    for r in roles:
        role_seq += {"system": "S", "user": "U", "assistant": "A"}.get(r, "?")
    # Print role sequence in chunks
    print("role sequence (S=system, U=user, A=assistant):")
    for i in range(0, len(role_seq), 80):
        print(f"  [{i:4d}]  {role_seq[i:i+80]}")

    # Look for 'system' messages after the first one
    sys_indices = [i for i, m in enumerate(msgs) if m["role"] == "system"]
    print(f"\nsystem message indices: {sys_indices}")

    # Also look for messages whose user content starts with 'Hi' / 'Hello'
    greetings = []
    for i, m in enumerate(msgs):
        if m["role"] == "user":
            c = m["content"].lstrip()
            # remove "User: " prefix if present
            if c.startswith("User:"):
                c = c[len("User:"):].lstrip()
            first_word = c[:30].split()[:3]
            first_word = " ".join(first_word)
            if any(c.lower().startswith(g) for g in
                   ["hi ", "hi,", "hi!", "hello", "hey", "good morning",
                    "good afternoon", "good evening"]):
                greetings.append((i, first_word))
    print(f"\n{len(greetings)} user messages start with greeting:")
    for i, g in greetings[:20]:
        print(f"  [{i:4d}]  {g!r}")

    # Dump a window around the 5th greeting to see what a boundary looks like
    if len(greetings) >= 2:
        target = greetings[1][0]
        print(f"\n--- context around message {target} (suspected session boundary) ---")
        for i in range(max(0, target - 2), min(len(msgs), target + 3)):
            m = msgs[i]
            c = m["content"][:180].replace("\n", " ")
            print(f"  [{i}] {m['role']:10s} {c!r}")


def cross_version_overlap() -> None:
    section("CROSS-VERSION PERSONA / CONTEXT OVERLAP")
    versions = ["32k", "128k", "1M"]
    pids_by_ver: dict[str, set[str]] = {}
    ctxs_by_ver: dict[str, set[str]] = {}
    for v in versions:
        qs = load_questions(f"questions_{v}.csv")
        pids_by_ver[v] = set(r["persona_id"] for r in qs)
        ctxs_by_ver[v] = set(r["shared_context_id"] for r in qs)

    for v in versions:
        print(f"{v}: personas={len(pids_by_ver[v])}  contexts={len(ctxs_by_ver[v])}")
    print(f"\npersona overlap 32k ∩ 128k: {len(pids_by_ver['32k'] & pids_by_ver['128k'])}")
    print(f"persona overlap 128k ∩ 1M: {len(pids_by_ver['128k'] & pids_by_ver['1M'])}")
    print(f"persona overlap 32k ∩ 1M:  {len(pids_by_ver['32k'] & pids_by_ver['1M'])}")
    print(f"\ncontext overlap 32k ∩ 128k: {len(ctxs_by_ver['32k'] & ctxs_by_ver['128k'])}")
    print(f"context overlap 128k ∩ 1M: {len(ctxs_by_ver['128k'] & ctxs_by_ver['1M'])}")


def dump_readme() -> None:
    section("README.md")
    print((DATA / "README.md").read_text(encoding="utf-8"))


def main() -> None:
    dump_readme()
    cross_version_overlap()

    for tag in ["32k", "128k", "1M"]:
        qrows = load_questions(f"questions_{tag}.csv")
        analyze_questions(tag, qrows)

    # Context structure on 32k (smallest)
    ctx32 = load_contexts("shared_contexts_32k.jsonl")
    q32 = load_questions("questions_32k.csv")
    analyze_contexts("32k", ctx32, q32)
    find_session_delimiters(ctx32)


if __name__ == "__main__":
    main()
