"""PersonaMem-v1 loader for the baseline suite — the ONE loader every method shares.

Adapted from `legacy/general_personamem/data_prep/load_personamem.py` (the original single
source of truth), repointed at `$UWM_DATA/personamem` per CONVENTIONS.md §2 and reshaped to
emit flat MCQ records + the memory views the baselines inject.

A baseline never re-parses the data; it only chooses *what context to inject* per MCQ
(see `baselines/*/predict.py`, i.e. domains/general/baselines/). The 4 memory views this exposes:
  - `demographics(ctx_id)`     → the persona card (first system message)  → the `+profile` arm
  - `reference_slice(...)`     → ±window turns around the reference turn   → `oracle` upper ref
  - `full_context(...)`        → all turns up to the reference            → long-context upper bound
  - (retrieval lives in the tokenmem baselines, built on `context_turns`)
"""
from __future__ import annotations

import ast
import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

VERSIONS = ("32k", "128k", "1M")
BENCH_TO_VERSION = {"pm32k": "32k", "pm128k": "128k", "pm1m": "1M"}

# Collapse the inconsistent raw question_type strings to 7 canonical names
# (verbatim from the legacy loader — keep in sync).
QTYPE_CANONICAL: dict[str, str] = {
    "recall_user_shared_facts": "recall_facts",
    "recalling_facts_mentioned_by_the_user": "recall_facts",
    "suggest_new_ideas": "suggest_new",
    "acknowledge_latest_user_preferences": "acknowledge_latest",
    "track_full_preference_evolution": "track_evolution",
    "track_full_preference_updates": "track_evolution",
    "revisit_reasons_behind_preference_updates": "reasons_behind",
    "recalling_the_reasons_behind_previous_updates": "reasons_behind",
    "provide_preference_aligned_recommendations": "aligned_rec",
    "generalize_to_new_scenarios": "generalize",
    "generalizing_to_new_scenarios": "generalize",
}
CANONICAL_QTYPES = (
    "recall_facts", "suggest_new", "acknowledge_latest", "track_evolution",
    "reasons_behind", "aligned_rec", "generalize",
)

_OPT_LETTER = re.compile(r"^\s*\(([a-zA-Z])\)\s*")


def _data_root(data_dir: str | None = None) -> Path:
    """Resolve the PersonaMem dir from arg or $UWM_DATA (CONVENTIONS.md §3: no hardcoded paths)."""
    if data_dir:
        return Path(data_dir)
    base = os.environ.get("UWM_DATA")
    if not base:
        raise RuntimeError("UWM_DATA unset — `source scripts/env.sh` first (CONVENTIONS.md §0).")
    return Path(base) / "personamem"


@dataclass
class MCQ:
    qid: str
    persona_id: str
    qtype: str               # canonical
    query: str               # user_question_or_message
    options: list[tuple[str, str]]   # [(letter, text), ...] letters lowercased
    gold_letter: str         # lowercased
    ctx_id: str
    end_index: int           # the QUERY position in the shared context's message list
    dist_ref_tokens: int     # tokens between the reference turn and the query (ref is EARLIER)


def _parse_options(raw: str) -> list[tuple[str, str]]:
    """all_options is a Python-list-repr string of '(a) ...' strings (single-quoted →
    json.loads fails; ast.literal_eval is the fix, per the P-OPSD data-pitfall log)."""
    try:
        items = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        items = ast.literal_eval(raw)
    out: list[tuple[str, str]] = []
    for s in items:
        m = _OPT_LETTER.match(s)
        if not m:
            continue
        letter = m.group(1).lower()
        text = s[m.end():].strip()
        out.append((letter, text))
    return out


def _gold_letter(correct_answer: str) -> str:
    m = _OPT_LETTER.match(correct_answer.strip())
    return m.group(1).lower() if m else correct_answer.strip().strip("()").lower()


class PersonaMem:
    """Loads one bench version once; serves MCQs + the memory views baselines inject."""

    def __init__(self, bench: str, data_dir: str | None = None):
        if bench not in BENCH_TO_VERSION:
            raise ValueError(f"bench {bench!r} not in {list(BENCH_TO_VERSION)}")
        self.bench = bench
        self.version = BENCH_TO_VERSION[bench]
        self.root = _data_root(data_dir)
        self._contexts: dict[str, list[dict]] | None = None

    # -- contexts (lazy; the 1M jsonl is large) --
    @property
    def contexts(self) -> dict[str, list[dict]]:
        if self._contexts is None:
            self._contexts = {}
            with (self.root / f"shared_contexts_{self.version}.jsonl").open(encoding="utf-8") as fh:
                for line in fh:
                    self._contexts.update(json.loads(line))
        return self._contexts

    # -- questions → MCQs --
    def load_mcqs(self, limit: int | None = None) -> list[MCQ]:
        path = self.root / f"questions_{self.version}.csv"
        out: list[MCQ] = []
        dropped = 0
        with path.open(encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                try:
                    end_idx = int(r["end_index_in_shared_context"])
                    if end_idx < 0:
                        dropped += 1
                        continue
                except (ValueError, KeyError):
                    dropped += 1
                    continue
                qt = QTYPE_CANONICAL.get(r["question_type"])
                if qt is None:
                    raise ValueError(f"unknown question_type {r['question_type']!r}")
                opts = _parse_options(r["all_options"])
                if not opts:
                    dropped += 1
                    continue
                try:
                    dref = int(r.get("distance_to_ref_in_tokens", 0) or 0)
                except ValueError:
                    dref = 0
                out.append(MCQ(
                    qid=r["question_id"], persona_id=r["persona_id"], qtype=qt,
                    query=r["user_question_or_message"].strip(),
                    options=opts, gold_letter=_gold_letter(r["correct_answer"]),
                    ctx_id=r["shared_context_id"], end_index=end_idx, dist_ref_tokens=dref,
                ))
                if limit and len(out) >= limit:
                    break
        if dropped:
            print(f"[data/{self.bench}] dropped {dropped} malformed rows; kept {len(out)}")
        return out

    # -- memory views (what the baselines inject) --
    @staticmethod
    def _clean(content: str, role: str) -> str:
        for pre in (("user", "User:"), ("assistant", "Assistant:")):
            if role == pre[0] and content.startswith(pre[1]):
                return content[len(pre[1]):].lstrip()
        return content

    def demographics(self, ctx_id: str) -> str:
        """Persona card = first system message content. The `+profile` arm."""
        msgs = self.contexts[ctx_id]
        if msgs and msgs[0]["role"] == "system":
            return msgs[0]["content"].strip()
        return ""

    def _render(self, msgs: list[dict]) -> str:
        return "\n".join(
            f"{m['role'].capitalize()}: {self._clean(m['content'], m['role'])}" for m in msgs
        )

    def reference_slice(self, ctx_id: str, end_index: int, window: int = 2) -> str:
        """±`window` turns around the reference turn — the golden upper-reference (oracle)."""
        msgs = self.contexts[ctx_id]
        lo, hi = max(0, end_index - window), min(len(msgs), end_index + window + 1)
        return self._render(msgs[lo:hi])

    def full_context(self, ctx_id: str, end_index: int, max_turns: int | None = None) -> str:
        """All turns up to and including the reference — long-context upper bound."""
        msgs = self.contexts[ctx_id][: end_index + 1]
        if max_turns:
            msgs = msgs[-max_turns:]
        return self._render(msgs)

    def context_turns(self, ctx_id: str, end_index: int) -> list[dict]:
        """Raw prior turns (for retrieval baselines to index/retrieve over)."""
        return [
            {"role": m["role"], "content": self._clean(m["content"], m["role"])}
            for m in self.contexts[ctx_id][: end_index + 1]
        ]

    # -- message-list views (the backend scores options as continuations of these) --
    def card_msgs(self, ctx_id: str) -> list[dict]:
        """The persona card as a system message — the `+profile` arm. [] if absent."""
        demo = self.demographics(ctx_id)
        return [{"role": "system", "content": demo}] if demo else []

    # tokenizer for token-accurate reference location (the pilot showed char//4 is too coarse)
    _tok = None

    @classmethod
    def _tokenizer(cls):
        if cls._tok is None:
            from transformers import AutoTokenizer
            cls._tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
        return cls._tok

    _tok_len_cache: dict = {}   # ctx_id -> [per-message token length] (computed once/context)

    def _msg_token_lens(self, ctx_id: str) -> list[int]:
        if ctx_id not in self._tok_len_cache:
            tok = self._tokenizer()
            self._tok_len_cache[ctx_id] = [
                len(tok(m["content"], add_special_tokens=False)["input_ids"])
                for m in self.contexts[ctx_id]
            ]
        return self._tok_len_cache[ctx_id]

    def _ref_index(self, ctx_id: str, end_index: int, dist_ref_tokens: int) -> int:
        """Locate the reference TURN: walk back from the query (end_index) accumulating token
        counts until we've passed `dist_ref_tokens`. Per-context token lengths are cached so a
        shared context is tokenized once (essential for pm128k/pm1m where 1 context backs many
        MCQs)."""
        lens = self._msg_token_lens(ctx_id)
        acc, j = 0, end_index
        while j > 0 and acc < dist_ref_tokens:
            acc += lens[j]
            j -= 1
        return max(0, j)

    def slice_msgs(self, ctx_id: str, end_index: int, dist_ref_tokens: int,
                   window: int = 2) -> list[dict]:
        """±window turns around the TOKEN-located reference (the golden upper-reference)."""
        ref = self._ref_index(ctx_id, end_index, dist_ref_tokens)
        msgs = self.contexts[ctx_id]
        lo, hi = max(0, ref - window), min(len(msgs), ref + window + 1)
        return [{"role": m["role"], "content": self._clean(m["content"], m["role"])}
                for m in msgs[lo:hi] if m["role"] != "system"]

    def full_msgs(self, ctx_id: str, end_index: int, max_turns: int | None = None) -> list[dict]:
        """All turns up to the reference (cleaned, persona card dropped) — long-context arm."""
        turns = [{"role": m["role"], "content": self._clean(m["content"], m["role"])}
                 for m in self.contexts[ctx_id][: end_index + 1] if m["role"] != "system"]
        return turns[-max_turns:] if max_turns else turns
