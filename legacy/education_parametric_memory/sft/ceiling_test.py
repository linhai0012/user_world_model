"""Ceiling/transfer test: train a few personas HARD, then score their TRAIN items vs HELD-OUT
eval items. If train-choice-accuracy is high but eval stays flat -> the LoRA memorises but does
not transfer per-skill mastery to new items (a data/recipe limit, not underfitting). Decisive
and cheap. vllm_env on the B200."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import sft_core as S

bank = {json.loads(l)["_id"]: json.loads(l) for l in (ROOT / "question_bank/biology_gcse.jsonl").read_text().splitlines()}
splits = json.loads((ROOT / "substrate/splits.json").read_text())
eval_ids = splits["eval"]
srows = [json.loads(l) for l in (ROOT / "substrate/streams.jsonl").read_text().splitlines()]
etruth = {(r["persona"], r["snapshot"], r["question_id"]): r
          for r in (json.loads(l) for l in (ROOT / "substrate/eval_truth.jsonl").read_text().splitlines())}
EPOCHS = 15
PIDS = ["persona_03", "persona_07", "persona_11"]

tok, base = S.load_base()
m = S.attach_dual_lora(base)
init = S._init_adapter_state(m)


def choice_acc(model, items, truth_choice):
    hit = 0
    for qid in items:
        sc = S.score_mcq(tok, model, bank[qid])
        top = max(sc["option_logprobs"], key=sc["option_logprobs"].get)
        hit += int(top == truth_choice[qid])
    return hit / len(items)


for pid in PIDS:
    pstream = sorted((r for r in srows if r["persona"] == pid), key=lambda x: x["round"])
    train_ids = [r["question_id"] for r in pstream]
    train_choice = {r["question_id"]: r["selected_option_id"] for r in pstream}
    eval_choice = {qid: etruth[(pid, 40, qid)]["selected_option_id"] for qid in eval_ids if (pid, 40, qid) in etruth}
    stream = [(bank[r["question_id"]]["stem"], r["answer_text"]) for r in pstream]

    # base choice-acc
    base_train = choice_acc(base, train_ids, train_choice)
    base_eval = choice_acc(base, list(eval_choice), eval_choice)

    # train HARD to 40 rounds
    S._reset_adapters(m, init)
    sp, fp = S._split_params(m)
    os_, of = torch.optim.AdamW(sp, lr=1e-5), torch.optim.AdamW(fp, lr=2e-4)
    m.train()
    for _ in range(EPOCHS):
        for stem, ans in stream:
            ids, labels = S._sample_ids(tok, stem, ans)
            out = m(input_ids=torch.tensor([ids], device="cuda"), labels=torch.tensor([labels], device="cuda"))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(sp, 1.0); torch.nn.utils.clip_grad_norm_(fp, 1.0)
            os_.step(); of.step(); os_.zero_grad(); of.zero_grad()
    m.eval()
    tr_train = choice_acc(m, train_ids, train_choice)
    tr_eval = choice_acc(m, list(eval_choice), eval_choice)
    print(f"{pid}: TRAIN choice-acc base={base_train:.3f} -> trained={tr_train:.3f} (memorise?) | "
          f"EVAL choice-acc base={base_eval:.3f} -> trained={tr_eval:.3f} (transfer?)", flush=True)
print("\nIf trained-TRAIN >> base-TRAIN but trained-EVAL ~= base-EVAL: memorises, no transfer.")
