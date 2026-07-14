"""Full SFT run: A1 (per-user LoRA) + A0 (shared) + A-null (base) -> prediction rows.

For each persona, the dual-rate LoRA is trained incrementally through its answer stream and the
held-out eval set is scored at each snapshot. A0 = one LoRA on the pooled streams; A-null = the
base model. Writes one predictions file per learner. Idempotent (skips finished learners) so it
resumes after any interruption. GPU-only (vllm_env on the B200); outputs to scratch.

  predictions/__base__.jsonl    A-null, scored once (also = per-user snapshot 0)
  predictions/__shared__.jsonl  A0
  predictions/<pid>.jsonl       A1 per persona, snapshots 5/10/20/40 (+0 = base)
  status.json                   live progress
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import sft_core as S

WS = Path("/scratch/prj/cllm/cue_sft")
PRED = WS / os.environ.get("CUE_PRED_DIR", "predictions")
PRED.mkdir(parents=True, exist_ok=True)
SNAPSHOTS = [int(x) for x in os.environ.get("CUE_SNAPSHOTS", "0,5,10,20,40").split(",")]
EPOCHS = int(os.environ.get("CUE_EPOCHS", "3"))


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def set_status(d: dict) -> None:
    (WS / "status.json").write_text(json.dumps(d, indent=2))


def load() -> dict:
    bank = {json.loads(l)["_id"]: json.loads(l)
            for l in (ROOT / "question_bank/biology_gcse.jsonl").read_text().splitlines()}
    splits = json.loads((ROOT / "substrate/splits.json").read_text())
    eval_qs = [bank[q] for q in splits["eval"] + splits["calib"]]  # score held-out: eval(report)+calib(fit)
    streams: dict[str, list] = {}
    for l in (ROOT / "substrate/streams.jsonl").read_text().splitlines():
        r = json.loads(l)
        streams.setdefault(r["persona"], []).append(r)
    for pid in streams:
        streams[pid] = [(bank[r["question_id"]]["stem"], r["answer_text"],
                         r["skill_id"].split("__")[-1])
                        for r in sorted(streams[pid], key=lambda x: x["round"])]
    personas = [p["pid"] for p in json.loads((ROOT / "substrate/persona_set.json").read_text())["personas"]]
    return {"bank": bank, "eval_qs": eval_qs, "streams": streams, "personas": personas}


def score_set(tok, m, eval_qs, learner, snapshot, scope) -> list[dict]:
    rows = []
    for q in eval_qs:
        sc = S.score_mcq(tok, m, q)
        rows.append({"learner_id": learner, "subject_id": q["subject_id"], "snapshot": snapshot,
                     "question_id": q["_id"], "format": "mcq", "scope": scope, **sc})
    return rows


def train_and_score(tok, m, init, stream, eval_qs, learner, scope, base_rows) -> list[dict]:
    """Reset adapters, train incrementally through `stream`, score eval at each snapshot."""
    S._reset_adapters(m, init)
    slow_p, fast_p = S._split_params(m)
    opt_s = torch.optim.AdamW(slow_p, lr=1e-5)   # slow MLP adapter
    opt_f = torch.optim.AdamW(fast_p, lr=2e-4)   # fast Attn adapter
    rows = [dict(r, learner_id=learner, snapshot=0, scope=scope) for r in base_rows]  # snap0 = base
    prev = 0
    for snap in [s for s in SNAPSHOTS if s > 0]:
        seg = stream[prev:snap]
        m.train()
        for _ in range(EPOCHS):
            for stem, ans, skill in seg:
                ids, labels = S._sample_ids(tok, stem, ans, skill)
                out = m(input_ids=torch.tensor([ids], device="cuda"),
                        labels=torch.tensor([labels], device="cuda"))
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(slow_p, 1.0)
                torch.nn.utils.clip_grad_norm_(fast_p, 1.0)
                opt_s.step(); opt_f.step(); opt_s.zero_grad(); opt_f.zero_grad()
        m.eval()
        rows += score_set(tok, m, eval_qs, learner, snap, scope)
        prev = snap
    return rows


def main() -> None:
    t0 = time.time()
    d = load()
    bank, eval_qs, streams, personas = d["bank"], d["eval_qs"], d["streams"], d["personas"]
    log(f"loaded: {len(personas)} personas, {len(eval_qs)} eval items, "
        f"{sum(len(s) for s in streams.values())} stream rounds")

    tok, base = S.load_base()
    base.eval()

    # A-null: base (also per-user snapshot 0). Score once.
    if not (PRED / "__base__.jsonl").exists():
        base_rows = score_set(tok, base, eval_qs, "__base__", 0, "base")
        (PRED / "__base__.jsonl").write_text("\n".join(json.dumps(r) for r in base_rows) + "\n")
        log("scored A-null (__base__)")
    else:
        base_rows = [json.loads(l) for l in (PRED / "__base__.jsonl").read_text().splitlines()]
        log("A-null cached")
    # base_rows for reuse as snap0: strip learner/snapshot/scope-specific fields
    base_pred = [{k: r[k] for k in ("subject_id", "question_id", "format",
                                    "option_logprobs", "p_correct_raw", "p_correct")}
                 | {"snapshot": 0} for r in base_rows]

    m = S.attach_dual_lora(base)
    init = S._init_adapter_state(m)

    # A0 shared: one LoRA on pooled streams
    if not (PRED / "__shared__.jsonl").exists():
        pooled = [s for pid in personas for s in streams[pid]]
        log(f"training A0 shared LoRA on {len(pooled)} pooled rounds")
        S._reset_adapters(m, init)
        slow_p, fast_p = S._split_params(m)
        opt_s = torch.optim.AdamW(slow_p, lr=1e-5); opt_f = torch.optim.AdamW(fast_p, lr=2e-4)
        m.train()
        for stem, ans, skill in pooled:  # 1 epoch over pooled
            ids, labels = S._sample_ids(tok, stem, ans, skill)
            out = m(input_ids=torch.tensor([ids], device="cuda"),
                    labels=torch.tensor([labels], device="cuda"))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(slow_p, 1.0); torch.nn.utils.clip_grad_norm_(fast_p, 1.0)
            opt_s.step(); opt_f.step(); opt_s.zero_grad(); opt_f.zero_grad()
        m.eval()
        shared_rows = []
        for snap in SNAPSHOTS:  # flat across snapshots (shared = pooled reference)
            shared_rows += score_set(tok, m, eval_qs, "__shared__", snap, "shared")
        (PRED / "__shared__.jsonl").write_text("\n".join(json.dumps(r) for r in shared_rows) + "\n")
        log("scored A0 (__shared__)")
    else:
        log("A0 cached")

    # A1 per persona
    done = sorted(p.stem for p in PRED.glob("persona_*.jsonl"))
    for i, pid in enumerate(personas):
        if pid in done:
            log(f"[{i+1}/{len(personas)}] {pid} cached")
            continue
        try:
            tp = time.time()
            base_for_p = [dict(r, learner_id=pid, scope="per_user") for r in base_pred]
            rows = train_and_score(tok, m, init, streams[pid], eval_qs, pid, "per_user", base_for_p)
            (PRED / f"{pid}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            log(f"[{i+1}/{len(personas)}] {pid} done in {time.time()-tp:.1f}s "
                f"(gpu {torch.cuda.max_memory_allocated()/1e9:.1f}GB)")
        except Exception as e:
            log(f"[{i+1}/{len(personas)}] {pid} FAILED: {e}\n{traceback.format_exc()}")
        set_status({"phase": "A1", "done": i + 1, "total": len(personas), "current": pid,
                    "elapsed_min": round((time.time() - t0) / 60, 1)})

    set_status({"phase": "complete", "personas": len(personas),
                "elapsed_min": round((time.time() - t0) / 60, 1)})
    log(f"ALL DONE in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
