"""Shared scorer (CONVENTIONS.md §3): accuracy + per-qtype over the 7 PersonaMem categories.

Primary metric = accuracy; PPL is secondary/model-internal. Writes the headline results JSON
keyed by run-id (`experiments/results/<run_id>__acc.json`).
"""
from __future__ import annotations

import json
from pathlib import Path

from domains.general.data import CANONICAL_QTYPES


def score(records: list[dict]) -> dict:
    """records = [{qtype, gold, pred}, ...]  → accuracy overall + per-qtype."""
    n = len(records)
    correct = sum(1 for r in records if r["pred"] == r["gold"])
    by_qtype: dict[str, dict] = {q: {"n": 0, "correct": 0} for q in CANONICAL_QTYPES}
    for r in records:
        b = by_qtype.setdefault(r["qtype"], {"n": 0, "correct": 0})
        b["n"] += 1
        b["correct"] += int(r["pred"] == r["gold"])
    return {
        "n": n,
        "accuracy": round(correct / n, 4) if n else None,
        "by_qtype": {
            q: {"n": b["n"], "accuracy": round(b["correct"] / b["n"], 4) if b["n"] else None}
            for q, b in by_qtype.items()
        },
    }


def write_results(run_id: str, results_dir: Path | str, summary: dict, metric: str = "acc") -> Path:
    out = Path(results_dir) / f"{run_id}__{metric}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return out
