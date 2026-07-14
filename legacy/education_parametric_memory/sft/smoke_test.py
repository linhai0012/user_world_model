"""GPU smoke test: train persona_00's dual-rate LoRA and confirm it shifts predictions toward
the persona's true (often wrong) behaviour vs the base model. Validates the whole SFT pipeline
before the full run. Run with vllm_env python on the B200."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import sft_core as S

PID = "persona_00"
bank = {json.loads(l)["_id"]: json.loads(l) for l in (ROOT / "question_bank/biology_gcse.jsonl").read_text().splitlines()}
srows = sorted((json.loads(l) for l in (ROOT / "substrate/streams.jsonl").read_text().splitlines()
               if json.loads(l)["persona"] == PID), key=lambda r: r["round"])
stream = [(bank[r["question_id"]]["stem"], r["answer_text"]) for r in srows]
truth10 = {r["question_id"]: r for r in (json.loads(l) for l in (ROOT / "substrate/eval_truth.jsonl").read_text().splitlines())
           if r["persona"] == PID and r["snapshot"] == 10}
eval_ids = list(truth10)[:5]
print(f"{PID}: stream={len(stream)} rounds, scoring {len(eval_ids)} held-out eval items")

tok, base = S.load_base()

# 1) base scores (true base, before any adapter)
base.eval()
base_p = {qid: S.score_mcq(tok, base, bank[qid])["p_correct"] for qid in eval_ids}

# 2) train dual LoRA snapshots, time it
m = S.attach_dual_lora(base)
init = S._init_adapter_state(m)
outdir = Path("/scratch/prj/cllm/cue_sft/runs/smoke_persona_00")
t = time.time()
saved = S.train_snapshots(tok, m, init, stream, [0, 5, 10], outdir, epochs_per_seg=3)
dt = time.time() - t
print(f"train [5,10] took {dt:.1f}s  (gpu mem {torch.cuda.max_memory_allocated()/1e9:.1f}GB)")

# 3) snap10 scores (m is at snap10 after training)
snap10_p = {qid: S.score_mcq(tok, m, bank[qid])["p_correct"] for qid in eval_ids}

print("\nqid_skill           base_p  snap10_p  persona_true_correct  moved_toward_truth")
moved = 0
for qid in eval_ids:
    sk = qid.split("__")[-1].split("#")[0]
    true_c = truth10[qid]["is_correct"]
    b, s = base_p[qid], snap10_p[qid]
    # "toward truth": if persona got it right, p should rise; if wrong, p should fall
    ok = (s > b) if true_c else (s < b)
    moved += ok
    print(f"  {sk:18s} {b:.3f}   {s:.3f}     {str(true_c):5s}                {'YES' if ok else 'no'}")
print(f"\nmoved toward truth on {moved}/{len(eval_ids)} items")
print(f"ESTIMATE: ~{dt:.0f}s for 2 segments(5,10) of 1 persona; full [5,10,20,40]x24 ~= {dt*2*24/60:.0f} min train")
print("SMOKE OK" if saved else "SMOKE FAILED")
