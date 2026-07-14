"""Import a parametric artifact bundle into Mongo for the `offline_replay` provider.

This is the **missing bridge**: the live app's `offline_replay` provider READS the
`parametric_replay` collection, but nothing in the repo WRITES it. This script is that writer.
It also absorbs the contract mismatches between the schema doc and the provider's code, so the
bundle files can stay in the documented shape:

  - field mapping    bundle `learner_id`  ->  Mongo `user_id`   (provider queries `user_id`)
  - derived field    each prediction row gets `skill_id`        (parsed from `question_id`, so
                     `offline_replay.mastery()` — which reads `skill_id` — works without a
                     separate mastery file the schema never defined)
  - controls         `__shared__` (A0) and `__base__` (A-null) rows import verbatim; the
                     provider's miss-policy falls back to `__shared__`.

Idempotent: wipes the subject's prior replay/adapter/curve rows, then reloads.

Importable:  `await import_bundle(db, Path("parametric_offline/sample_bundle"))`
CLI:         backend/.venv/bin/python parametric_offline/import_bundle.py \
                 --bundle parametric_offline/sample_bundle
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from app.db.mongo import LEARNER_ADAPTERS, PARAMETRIC_REPLAY

# Collections with no constant in app/db/mongo.py yet (no live consumer — imported for the
# compare view / eval to pick up later, and to expose the available snapshots for nearest-<=).
PARAMETRIC_CURVES = "parametric_curves"
PARAMETRIC_BUNDLES = "parametric_bundles"


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _skill_id_of(question_id: str) -> str:
    """question_id == "{skill_id}#q{i}"  ->  skill_id. (Contract: see README §question_id.)"""
    return question_id.rsplit("#", 1)[0]


async def import_bundle(db, bundle_dir: Path) -> dict:
    bundle_dir = Path(bundle_dir)
    manifest = json.loads((bundle_dir / "manifest.json").read_text())
    cost = json.loads((bundle_dir / "cost.json").read_text())
    predictions = _read_jsonl(bundle_dir / "predictions.jsonl")
    adapters = _read_jsonl(bundle_dir / "adapters.jsonl")
    curves = _read_jsonl(bundle_dir / "curves.jsonl")

    subject_ids = manifest.get("subject_ids", [])

    # --- idempotent wipe for these subjects ---
    for coll in (PARAMETRIC_REPLAY, LEARNER_ADAPTERS, PARAMETRIC_CURVES):
        await db[coll].delete_many({"subject_id": {"$in": subject_ids}})
    await db[PARAMETRIC_BUNDLES].delete_many({"bundle_id": manifest.get("bundle_id")})

    # --- predictions -> parametric_replay (the core read path) ---
    replay_docs = []
    for r in predictions:
        qid = r["question_id"]
        replay_docs.append({
            "user_id": r["learner_id"],                 # learner_id -> user_id
            "subject_id": r["subject_id"],
            "snapshot": int(r["snapshot"]),
            "question_id": qid,
            "skill_id": _skill_id_of(qid),              # derived so mastery() works
            "format": r.get("format"),
            "p_correct": float(r["p_correct"]),
            "p_correct_raw": r.get("p_correct_raw"),
            "option_logprobs": r.get("option_logprobs"),
            "scope": r.get("scope"),
        })
    if replay_docs:
        await db[PARAMETRIC_REPLAY].insert_many(replay_docs)

    # --- adapters -> learner_adapters (metadata; mark the max snapshot per learner active) ---
    max_snap: dict[tuple, int] = {}
    for a in adapters:
        key = (a["learner_id"], a["subject_id"])
        max_snap[key] = max(max_snap.get(key, -1), int(a["snapshot"]))
    adapter_docs = []
    for a in adapters:
        key = (a["learner_id"], a["subject_id"])
        adapter_docs.append({
            "user_id": a["learner_id"],
            "subject_id": a["subject_id"],
            "scope": a.get("scope"),
            "version": a.get("version"),
            "snapshot": int(a["snapshot"]),
            "n_rounds_trained": a.get("n_rounds_trained"),
            "parent_version": a.get("parent_version"),
            "eval": a.get("eval", {}),
            "best_step_caveat": bool(a.get("best_step_caveat", False)),
            "active": int(a["snapshot"]) == max_snap[key],
        })
    if adapter_docs:
        await db[LEARNER_ADAPTERS].insert_many(adapter_docs)

    # --- curves -> parametric_curves (no live consumer yet; for compare/eval) ---
    curve_docs = [{**c, "user_id": c["learner_id"]} for c in curves]
    if curve_docs:
        await db[PARAMETRIC_CURVES].insert_many(curve_docs)

    # --- manifest + cost -> parametric_bundles (traceability + available snapshots) ---
    await db[PARAMETRIC_BUNDLES].insert_one({
        "bundle_id": manifest.get("bundle_id"),
        "subject_ids": subject_ids,
        "snapshots": manifest.get("snapshots", []),
        "persona_set": manifest.get("persona_set"),
        "base_model": manifest.get("base_model"),
        "lora_recipe": manifest.get("lora_recipe"),
        "calibration": manifest.get("calibration"),
        "created_offline": manifest.get("created_offline"),
        "cost": cost,
    })

    return {
        "bundle_id": manifest.get("bundle_id"),
        "subjects": subject_ids,
        "replay_rows": len(replay_docs),
        "adapters": len(adapter_docs),
        "curves": len(curve_docs),
        "learners": sorted({d["user_id"] for d in replay_docs}),
        "snapshots": manifest.get("snapshots", []),
    }


async def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, default=Path(__file__).parent / "sample_bundle")
    args = ap.parse_args()
    from app.config import get_settings
    from app.db import mongo

    settings = get_settings()
    db = mongo.connect(settings.mongo_uri, settings.mongo_db)
    stats = await import_bundle(db, args.bundle)
    mongo.close()
    print("imported bundle:", json.dumps(stats, indent=2))


if __name__ == "__main__":
    asyncio.run(_main())
