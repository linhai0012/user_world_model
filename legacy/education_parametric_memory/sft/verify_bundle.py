"""Phase 6 — verify the REAL bundle flows through the live `offline_replay` seam (honesty gate).

Imports the real bundle into an in-memory Mongo and checks that OfflineReplayProvider serves the
bundle's own calibrated p_correct (NOT the stub) for real (learner, snapshot, question) rows.
If this passes, the live demo's parametric arm replays linhai's real run — the gate the schema
requires (never `stub` under a paper claiming real parametric results). Run with backend/.venv.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from mongomock_motor import AsyncMongoMockClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                      # import_bundle
sys.path.insert(0, str(ROOT.parent / "backend"))   # app
from import_bundle import import_bundle  # noqa: E402
from app.db import mongo  # noqa: E402
from app.db.mongo import SESSIONS  # noqa: E402
from app.services.usermodel.providers.offline_replay import OfflineReplayProvider  # noqa: E402

BUNDLE = Path("/scratch/prj/cllm/cue_sft/bundles") / os.environ.get("CUE_BUNDLE", "cue-param-biology-v1")


async def main() -> int:
    db = AsyncMongoMockClient()["cue_verify"]
    mongo.set_db(db)
    stats = await import_bundle(db, BUNDLE)
    print("imported:", json.dumps(stats))

    prov = OfflineReplayProvider()
    assert await prov._has_bundle(db, "biology_gcse"), "no bundle imported!"

    preds = [json.loads(l) for l in (BUNDLE / "predictions.jsonl").read_text().splitlines()]
    # sample real per-user rows across personas/snapshots
    sample = [r for r in preds if r["scope"] == "per_user"][:: max(1, len(preds) // 12)][:12]
    ok = 0
    for r in sample:
        await db[SESSIONS].update_one(
            {"user_id": r["learner_id"], "subject_id": r["subject_id"]},
            {"$set": {"user_id": r["learner_id"], "subject_id": r["subject_id"],
                      "parametric_snapshot": r["snapshot"]}}, upsert=True)
        q = {"_id": r["question_id"], "subject_id": r["subject_id"]}
        got = await prov.predict_correct(db, r["learner_id"], q)
        match = abs(got - r["p_correct"]) < 1e-6
        ok += match
        if not match:
            print(f"  MISMATCH {r['learner_id']} s{r['snapshot']} {r['question_id']}: "
                  f"got {got} want {r['p_correct']}")
    print(f"served {ok}/{len(sample)} real per-user rows correctly via offline_replay")

    # control: an unknown walk-up user falls back to the shared reference, never the stub.
    # Use a snapshot that actually exists in THIS bundle (snapshots vary by config).
    vsnap = sample[0]["snapshot"]
    await db[SESSIONS].update_one({"user_id": "walkup", "subject_id": "biology_gcse"},
                                  {"$set": {"user_id": "walkup", "subject_id": "biology_gcse",
                                            "parametric_snapshot": vsnap}}, upsert=True)
    q0 = sample[0]["question_id"]
    shared = next((r for r in preds if r["scope"] == "shared" and r["question_id"] == q0
                   and r["snapshot"] == vsnap), None)
    got = await prov.predict_correct(db, "walkup", {"_id": q0, "subject_id": "biology_gcse"})
    fb_ok = shared is not None and abs(got - shared["p_correct"]) < 1e-6
    print(f"walk-up -> shared-LoRA reference fallback: {'OK' if fb_ok else 'check'}")

    passed = ok == len(sample) and fb_ok
    print("\nHONESTY GATE:", "PASS — real bundle served via offline_replay" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
