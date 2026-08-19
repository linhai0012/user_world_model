"""naiverag — retrieve top-k prior turns into context (the realistic token-memory baseline).

The honest "memory = retrieval" instantiation the project's key gap targets: index the user's
own prior turns, retrieve the top-k most relevant to the current query, inject them as the
memory. Contrast vs `oracle` (which is handed the exact reference) and vs `trivial`/`profile`.

`params`:
  top_k: int = 5
  retriever: "bm25" | "dense"     # bm25 = lexical (CPU, default); dense = all-MiniLM-L6-v2
  with_profile: bool = False      # also prepend the persona card
"""
from __future__ import annotations

import re

from domains.general.data import MCQ, PersonaMem

METHOD = "naiverag"
_WORD = re.compile(r"[A-Za-z0-9']+")
_DENSE = {}   # lazy model cache


def _tokenize(s: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(s)]


def _bm25_topk(query: str, turns: list[dict], k: int) -> list[int]:
    from rank_bm25 import BM25Okapi
    corpus = [_tokenize(t["content"]) for t in turns]
    if not any(corpus):
        return list(range(min(k, len(turns))))
    bm = BM25Okapi(corpus)
    scores = bm.get_scores(_tokenize(query))
    return sorted(range(len(turns)), key=lambda i: scores[i], reverse=True)[:k]


def _dense_topk(query: str, turns: list[dict], k: int) -> list[int]:
    import numpy as np
    if "m" not in _DENSE:
        from sentence_transformers import SentenceTransformer
        _DENSE["m"] = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    m = _DENSE["m"]
    emb = m.encode([t["content"] for t in turns] + [query], normalize_embeddings=True)
    tv, qv = emb[:-1], emb[-1]
    sims = tv @ qv
    return sorted(range(len(turns)), key=lambda i: sims[i], reverse=True)[:k]


def build_context(mcq: MCQ, data: PersonaMem, params: dict) -> list[dict]:
    turns = data.context_turns(mcq.ctx_id, mcq.end_index)
    if not turns:
        return data.card_msgs(mcq.ctx_id) if params.get("with_profile") else []
    k = params.get("top_k", 5)
    idx = (_dense_topk if params.get("retriever") == "dense" else _bm25_topk)(mcq.query, turns, k)
    picked = [turns[i] for i in sorted(idx)]   # chronological order
    return (data.card_msgs(mcq.ctx_id) + picked) if params.get("with_profile") else picked
