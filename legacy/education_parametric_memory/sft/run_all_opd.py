"""Full OPD (oracle-teacher) run: A1 per-user + A0 shared + A-null base -> prediction rows.

Twin of run_all.py, but the per-user / shared LoRAs are trained by DISTILLATION against the
generator's exact option categorical g(theta,q) (the privileged teacher), not hard-label CE.
theta (mastery + held misconception) is given only to the teacher; the student LoRA sees stem-only
and must internalise "misconception -> mass on the consistent distractor" into the weights.

Loss per item = -sum_o p_teacher(o) log q_student(o)  (forward KL / Hinton soft-CE, mass-covering).
Output schema is IDENTICAL to run_all.py so package_bundle.py / analyze.py work unchanged with
CUE_PRED_DIR=predictions_opd. GPU-only (vllm_env). Resumable (skips finished personas).

Env: CUE_PRED_DIR(predictions_opd) CUE_SNAPSHOTS(0,13,26,52,104) CUE_EPOCHS(3)
     CUE_SUBSTRATE(substrate) CUE_TEACHER_TEMP(1.0)
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import sft_core as S
from personas import Persona, PersonaState, _WRONG_BIAS

WS = Path("/scratch/prj/cllm/cue_sft")
PRED = WS / os.environ.get("CUE_PRED_DIR", "predictions_opd")
PRED.mkdir(parents=True, exist_ok=True)
SUB = Path(os.environ.get("CUE_SUBSTRATE", str(ROOT / "substrate")))
SNAPSHOTS = [int(x) for x in os.environ.get("CUE_SNAPSHOTS", "0,13,26,52,104").split(",")]
EPOCHS = int(os.environ.get("CUE_EPOCHS", "3"))
TEMP = float(os.environ.get("CUE_TEACHER_TEMP", "1.0"))
# Which privileged TARGET to distill (set after the multi-arm picks the winner):
#   g            = oracle generating distribution (Run A)
#   hybrid       = 0.5*g + 0.5*one-hot(realized)   (blend the realized-error signal back in)
#   realized_opt = one-hot(realized) in option space
TARGET = os.environ.get("CUE_TARGET", "g")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def set_status(d: dict) -> None:
    (WS / "status_opd.json").write_text(json.dumps(d, indent=2))


# --------------------------------------------------------------- teacher = g(theta,q)
def load_persona(pj: dict) -> Persona:
    held = {s: (v["tag"], v["strength"]) for s, v in pj["held"].items()}
    return Persona(pid=pj["pid"], ability=pj["ability"], topic_offset=pj["topic_offset"],
                   skill_jitter=pj["skill_jitter"], held=held)


def teacher_dist(state: PersonaState, q: dict) -> dict:
    skill = q["skill_id"]
    m = state.mastery_of(skill)
    opts = q["options"]
    correct_id = next((o["id"] for o in opts if o.get("is_correct")), opts[0]["id"])
    wrong = [o for o in opts if not o.get("is_correct")]
    nw = len(wrong)
    held_tag = state.misconception_for(skill)
    fav = next((o for o in wrong if o.get("misconception_tag") == held_tag), None)
    wm = 1.0 - m
    P = {correct_id: m}
    for o in wrong:
        if fav is not None:
            P[o["id"]] = wm * (_WRONG_BIAS + (1 - _WRONG_BIAS) / nw) if o["id"] == fav["id"] \
                else wm * ((1 - _WRONG_BIAS) / nw)
        else:
            P[o["id"]] = wm * (1.0 / nw)
    return P


# --------------------------------------------------------------- load
def load() -> dict:
    bank = {json.loads(l)["_id"]: json.loads(l)
            for l in (ROOT / "question_bank/biology_gcse.jsonl").read_text().splitlines()}
    all_skills = sorted({q["skill_id"] for q in bank.values()})
    splits = json.loads((SUB / "splits.json").read_text())
    eval_qs = [bank[q] for q in splits["eval"] + splits["calib"]]
    srows = [json.loads(l) for l in (SUB / "streams.jsonl").read_text().splitlines()]
    personas_json = {p["pid"]: p for p in json.loads((SUB / "persona_set.json").read_text())["personas"]}
    personas = [p["pid"] for p in json.loads((SUB / "persona_set.json").read_text())["personas"]]

    # per-persona OPD training items (q, teacher_categorical) in round order, replaying state
    opd_items: dict[str, list] = {}
    for pid in personas:
        p = load_persona(personas_json[pid])
        state = PersonaState.initial(p, all_skills)
        pstream = sorted((r for r in srows if r["persona"] == pid), key=lambda x: x["round"])
        items = []
        for r in pstream:
            q = bank[r["question_id"]]
            items.append((q, teacher_dist(state, q), r["selected_option_id"]))
            state.practice(q["skill_id"])
        opd_items[pid] = items
    return {"bank": bank, "eval_qs": eval_qs, "opd_items": opd_items, "personas": personas}


# --------------------------------------------------------------- OPD loss
def _opt_logprob_grad(tok, m, stem: str, cont_text: str, skill):
    p = tok(S._prompt(stem, skill), add_special_tokens=False).input_ids
    c = tok(" " + cont_text, add_special_tokens=False).input_ids
    ids = torch.tensor([p + c], device="cuda")
    logits = m(input_ids=ids).logits[0]
    start = len(p) - 1
    lp = F.log_softmax(logits[start:start + len(c)].float(), dim=-1)
    tok_lp = lp[range(len(c)), torch.tensor(c, device="cuda")]
    return tok_lp.mean()


def _target_vec(q, td, realized_id):
    opts = q["options"]
    g = [td[o["id"]] for o in opts]
    if TARGET == "g":
        return g
    onehot = [1.0 if o["id"] == realized_id else 0.0 for o in opts]
    if TARGET == "realized_opt":
        return onehot
    if TARGET == "hybrid":
        return [0.5 * a + 0.5 * b for a, b in zip(g, onehot)]
    raise ValueError(f"unknown CUE_TARGET={TARGET}")


def _distill_step(tok, m, q, td, realized_id, sp, fp, opt_s, opt_f) -> None:
    skill_key = q.get("skill_id", "").split("__")[-1]
    opts = q["options"]
    s = torch.stack([_opt_logprob_grad(tok, m, q["stem"], o["text"], skill_key) for o in opts])
    logq = F.log_softmax(s / TEMP, dim=0)
    pvec = torch.tensor(_target_vec(q, td, realized_id), device="cuda", dtype=logq.dtype)
    loss = -(pvec * logq).sum()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(sp, 1.0)
    torch.nn.utils.clip_grad_norm_(fp, 1.0)
    opt_s.step(); opt_f.step(); opt_s.zero_grad(); opt_f.zero_grad()


def score_set(tok, m, eval_qs, learner, snapshot, scope) -> list[dict]:
    rows = []
    for q in eval_qs:
        sc = S.score_mcq(tok, m, q)
        rows.append({"learner_id": learner, "subject_id": q["subject_id"], "snapshot": snapshot,
                     "question_id": q["_id"], "format": "mcq", "scope": scope, **sc})
    return rows


def train_and_score_opd(tok, m, init, items, eval_qs, learner, scope, base_rows) -> list[dict]:
    S._reset_adapters(m, init)
    sp, fp = S._split_params(m)
    opt_s = torch.optim.AdamW(sp, lr=1e-5)
    opt_f = torch.optim.AdamW(fp, lr=2e-4)
    rows = [dict(r, learner_id=learner, snapshot=0, scope=scope) for r in base_rows]
    prev = 0
    for snap in [s for s in SNAPSHOTS if s > 0]:
        seg = items[prev:snap]
        m.train()
        for _ in range(EPOCHS):
            for q, td, rid in seg:
                _distill_step(tok, m, q, td, rid, sp, fp, opt_s, opt_f)
        m.eval()
        rows += score_set(tok, m, eval_qs, learner, snap, scope)
        prev = snap
    return rows


# --------------------------------------------------------------- main
def main() -> None:
    t0 = time.time()
    d = load()
    bank, eval_qs, opd_items, personas = d["bank"], d["eval_qs"], d["opd_items"], d["personas"]
    log(f"OPD run: target={TARGET} {len(personas)} personas, {len(eval_qs)} eval items, "
        f"temp={TEMP}, snapshots={SNAPSHOTS}, epochs={EPOCHS}")

    tok, base = S.load_base()
    base.eval()

    if not (PRED / "__base__.jsonl").exists():
        base_rows = score_set(tok, base, eval_qs, "__base__", 0, "base")
        (PRED / "__base__.jsonl").write_text("\n".join(json.dumps(r) for r in base_rows) + "\n")
        log("scored A-null (__base__)")
    else:
        base_rows = [json.loads(l) for l in (PRED / "__base__.jsonl").read_text().splitlines()]
        log("A-null cached")
    base_pred = [{k: r[k] for k in ("subject_id", "question_id", "format",
                                    "option_logprobs", "p_correct_raw", "p_correct")}
                 | {"snapshot": 0} for r in base_rows]

    m = S.attach_dual_lora(base)
    init = S._init_adapter_state(m)

    # A0 shared OPD: distill one LoRA on pooled (q, teacher) items, 1 epoch
    if not (PRED / "__shared__.jsonl").exists():
        pooled = [it for pid in personas for it in opd_items[pid]]
        log(f"distilling A0 shared LoRA on {len(pooled)} pooled items")
        S._reset_adapters(m, init)
        sp, fp = S._split_params(m)
        opt_s = torch.optim.AdamW(sp, lr=1e-5); opt_f = torch.optim.AdamW(fp, lr=2e-4)
        m.train()
        for q, td, rid in pooled:
            _distill_step(tok, m, q, td, rid, sp, fp, opt_s, opt_f)
        m.eval()
        shared_rows = []
        for snap in SNAPSHOTS:
            shared_rows += score_set(tok, m, eval_qs, "__shared__", snap, "shared")
        (PRED / "__shared__.jsonl").write_text("\n".join(json.dumps(r) for r in shared_rows) + "\n")
        log("scored A0 (__shared__)")
    else:
        log("A0 cached")

    done = sorted(p.stem for p in PRED.glob("persona_*.jsonl"))
    for i, pid in enumerate(personas):
        if pid in done:
            log(f"[{i+1}/{len(personas)}] {pid} cached")
            continue
        try:
            tp = time.time()
            base_for_p = [dict(r, learner_id=pid, scope="per_user") for r in base_pred]
            rows = train_and_score_opd(tok, m, init, opd_items[pid], eval_qs, pid, "per_user", base_for_p)
            (PRED / f"{pid}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            log(f"[{i+1}/{len(personas)}] {pid} done in {time.time()-tp:.1f}s "
                f"(gpu {torch.cuda.max_memory_allocated()/1e9:.1f}GB)")
        except Exception as e:
            log(f"[{i+1}/{len(personas)}] {pid} FAILED: {e}\n{traceback.format_exc()}")
        set_status({"phase": "A1-OPD", "done": i + 1, "total": len(personas), "current": pid,
                    "elapsed_min": round((time.time() - t0) / 60, 1)})

    set_status({"phase": "complete", "personas": len(personas),
                "elapsed_min": round((time.time() - t0) / 60, 1)})
    log(f"ALL DONE in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
