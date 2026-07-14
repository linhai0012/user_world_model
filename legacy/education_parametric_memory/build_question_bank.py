"""Turn the raw workflow MCQ output into the canonical offline question bank.

Fixes the question_id fragility flagged earlier: ids are **content-stable**
(`{skill_id}#g{md5(stem)[:6]}`), not positional, so adding/reordering questions never shifts
existing ids — a hard requirement for the round_sequence contract. Output matches the seeded
question doc shape (so personas.py / streams.py / sft_core.py consume it unchanged).

    python parametric_offline/build_question_bank.py --raw /tmp/cue_bank_result.json
    -> parametric_offline/question_bank/biology_gcse.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SUBJECT = "biology_gcse"


def _topic_id(skill_id: str) -> str:
    return "__".join(skill_id.split("__")[:2])


def _qid(skill_id: str, stem: str) -> str:
    return f"{skill_id}#g{hashlib.md5(stem.strip().encode()).hexdigest()[:6]}"


def build(raw_path: Path, out_path: Path) -> dict:
    raw = json.loads(Path(raw_path).read_text())["bank"]
    seen_ids: set[str] = set()
    seen_stems: set[str] = set()
    docs: list[dict] = []
    dropped = {"structure": 0, "dup": 0}
    for m in raw:
        opts = m.get("options", [])
        n_correct = sum(1 for o in opts if o.get("is_correct"))
        n_tagged = sum(1 for o in opts if not o.get("is_correct") and o.get("misconception"))
        if len(opts) != 4 or n_correct != 1 or n_tagged < 3 or "skill_id" not in m:
            dropped["structure"] += 1
            continue
        stem = m["stem"].strip()
        skill_id = m["skill_id"]
        if stem.lower() in seen_stems:
            dropped["dup"] += 1
            continue
        seen_stems.add(stem.lower())
        qid = _qid(skill_id, stem)
        if qid in seen_ids:
            dropped["dup"] += 1
            continue
        seen_ids.add(qid)
        docs.append({
            "_id": qid,
            "subject_id": SUBJECT,
            "topic_id": _topic_id(skill_id),
            "skill_id": skill_id,
            "skill_ids": [skill_id],
            "format": "mcq",
            "stem": stem,
            "options": [
                {"id": chr(ord("a") + j), "text": o["text"],
                 "is_correct": bool(o.get("is_correct")),
                 "misconception_tag": o.get("misconception")}
                for j, o in enumerate(opts)
            ],
            "is_probe": False,
            "source": "workflow_gen",
        })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for d in docs:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")

    from collections import Counter
    per_skill = Counter(d["skill_id"].split("__")[-1] for d in docs)
    return {"written": len(docs), "dropped": dropped, "per_skill": dict(per_skill),
            "out": str(out_path)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("/tmp/cue_bank_result.json"))
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).parent / "question_bank" / "biology_gcse.jsonl")
    args = ap.parse_args()
    stats = build(args.raw, args.out)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
