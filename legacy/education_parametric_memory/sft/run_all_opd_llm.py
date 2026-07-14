"""Full LLM-teacher hybrid-OPD run — the DEPLOYABLE version of the headline recipe.

run_all_opd.py's hybrid (0.5*g + 0.5*realized) won at scale (mastery_corr 0.18->0.49) but g uses the
authored θ (oracle privilege). This swaps g for a teacher WITHOUT oracle mastery: the frozen base
model conditioned on a privileged "predict this student's answer" prompt carrying the held
misconception + a coarse mastery bucket (low/med/high). Target = 0.5*llm_teacher + 0.5*one-hot(realized).
If this reproduces oracle-hybrid's mastery_corr at scale, the recipe is deployable (a real LLM teacher
reading a learner card, no oracle θ). Output schema = run_all.py so compare_opd.py / analyze.py reuse.

GPU-only (vllm_env). Env: CUE_PRED_DIR(predictions_opd_llm) CUE_SNAPSHOTS(0,13,26,52,104)
CUE_EPOCHS(3) CUE_SUBSTRATE(substrate) CUE_BLEND(0.5 = realized weight in the hybrid target).
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
from personas import Persona, PersonaState

WS = Path("/scratch/prj/cllm/cue_sft")
PRED = WS / os.environ.get("CUE_PRED_DIR", "predictions_opd_llm")
PRED.mkdir(parents=True, exist_ok=True)
SUB = Path(os.environ.get("CUE_SUBSTRATE", str(ROOT / "substrate")))
SNAPSHOTS = [int(x) for x in os.environ.get("CUE_SNAPSHOTS", "0,13,26,52,104").split(",")]
EPOCHS = int(os.environ.get("CUE_EPOCHS", "3"))
BLEND = float(os.environ.get("CUE_BLEND", "0.5"))   # weight on one-hot(realized); (1-BLEND) on llm teacher


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def set_status(d):
    (WS / "status_opd_llm.json").write_text(json.dumps(d, indent=2))


def load_persona(pj):
    held = {s: (v["tag"], v["strength"]) for s, v in pj["held"].items()}
    return Persona(pid=pj["pid"], ability=pj["ability"], topic_offset=pj["topic_offset"],
                   skill_jitter=pj["skill_jitter"], held=held)


# --------------------------------------------------------------- LLM teacher (frozen base + priv prompt)
def _bucket(m):
    return "low" if m < 0.4 else "medium" if m < 0.7 else "high"


def _teacher_prompt(stem, skill_name, misconception, m):
    pre = "You are predicting how one specific student will answer a multiple-choice question. "
    if misconception:
        pre += (f"This student holds a misconception about {skill_name}: \"{misconception}\". "
                "They tend to answer in a way that reflects this misconception. ")
    pre += f"Their mastery of {skill_name} is {_bucket(m)}.\n"
    return f"{pre}Question: {stem}\nThe student's answer:"


@torch.no_grad()
def _teacher_logprob(tok, base, t_prompt, cont_text):
    p = tok(t_prompt, add_special_tokens=False).input_ids
    c = tok(" " + cont_text, add_special_tokens=False).input_ids
    ids = torch.tensor([p + c], device="cuda")
    logits = base(input_ids=ids).logits[0]
    start = len(p) - 1
    lp = F.log_softmax(logits[start:start + len(c)].float(), dim=-1)
    return float(lp[range(len(c)), torch.tensor(c, device="cuda")].mean().item())


@torch.no_grad()
def teacher_dist_llm(tok, base, m, misc_text, q):
    skill_name = q["skill_id"].split("__")[-1].replace("_", " ")
    tp = _teacher_prompt(q["stem"], skill_name, misc_text, m)
    lps = {o["id"]: _teacher_logprob(tok, base, tp, o["text"]) for o in q["options"]}
    keys = list(lps)
    z = torch.logsumexp(torch.tensor([lps[k] for k in keys]), dim=0).item()
    return {k: math.exp(lps[k] - z) for k in keys}


# --------------------------------------------------------------- load raw (CPU) + build targets (GPU)
def load_raw():
    bank = {json.loads(l)["_id"]: json.loads(l)
            for l in (ROOT / "question_bank/biology_gcse.jsonl").read_text().splitlines()}
    all_skills = sorted({q["skill_id"] for q in bank.values()})
    splits = json.loads((SUB / "splits.json").read_text())
    eval_qs = [bank[q] for q in splits["eval"] + splits["calib"]]
    srows = [json.loads(l) for l in (SUB / "streams.jsonl").read_text().splitlines()]
    pj = {p["pid"]: p for p in json.loads((SUB / "persona_set.json").read_text())["personas"]}
    personas = [p["pid"] for p in json.loads((SUB / "persona_set.json").read_text())["personas"]]
    raw = {}  # pid -> [(q, realized_id, mastery, misc_text)]
    for pid in personas:
        p = load_persona(pj[pid])
        state = PersonaState.initial(p, all_skills)
        pstream = sorted((r for r in srows if r["persona"] == pid), key=lambda x: x["round"])
        items = []
        for r in pstream:
            q = bank[r["question_id"]]
            items.append((q, r["selected_option_id"], state.mastery_of(q["skill_id"]),
                          state.misconception_for(q["skill_id"])))
            state.practice(q["skill_id"])
        raw[pid] = items
    return {"bank": bank, "eval_qs": eval_qs, "raw": raw, "personas": personas}


def build_targets(tok, base, raw):
    """LLM-teacher hybrid target per item: (1-BLEND)*llm_dist + BLEND*one-hot(realized)."""
    opd = {}
    for pid, items in raw.items():
        out = []
        for q, realized, m, misc in items:
            ld = teacher_dist_llm(tok, base, m, misc, q)
            tgt = {o["id"]: (1 - BLEND) * ld[o["id"]] + BLEND * (1.0 if o["id"] == realized else 0.0)
                   for o in q["options"]}
            out.append((q, tgt))
        opd[pid] = out
    return opd


# --------------------------------------------------------------- distill + score
def _opt_logprob_grad(tok, m, stem, cont_text, skill):
    p = tok(S._prompt(stem, skill), add_special_tokens=False).input_ids
    c = tok(" " + cont_text, add_special_tokens=False).input_ids
    ids = torch.tensor([p + c], device="cuda")
    logits = m(input_ids=ids).logits[0]
    start = len(p) - 1
    lp = F.log_softmax(logits[start:start + len(c)].float(), dim=-1)
    return lp[range(len(c)), torch.tensor(c, device="cuda")].mean()


def _distill_step(tok, m, q, tgt, sp, fp, opt_s, opt_f):
    skill_key = q.get("skill_id", "").split("__")[-1]
    opts = q["options"]
    s = torch.stack([_opt_logprob_grad(tok, m, q["stem"], o["text"], skill_key) for o in opts])
    logq = F.log_softmax(s, dim=0)
    pvec = torch.tensor([tgt[o["id"]] for o in opts], device="cuda", dtype=logq.dtype)
    loss = -(pvec * logq).sum()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(sp, 1.0); torch.nn.utils.clip_grad_norm_(fp, 1.0)
    opt_s.step(); opt_f.step(); opt_s.zero_grad(); opt_f.zero_grad()


def score_set(tok, m, eval_qs, learner, snapshot, scope):
    rows = []
    for q in eval_qs:
        sc = S.score_mcq(tok, m, q)
        rows.append({"learner_id": learner, "subject_id": q["subject_id"], "snapshot": snapshot,
                     "question_id": q["_id"], "format": "mcq", "scope": scope, **sc})
    return rows


def train_and_score(tok, m, init, items, eval_qs, learner, scope, base_rows):
    S._reset_adapters(m, init)
    sp, fp = S._split_params(m)
    opt_s = torch.optim.AdamW(sp, lr=1e-5); opt_f = torch.optim.AdamW(fp, lr=2e-4)
    rows = [dict(r, learner_id=learner, snapshot=0, scope=scope) for r in base_rows]
    prev = 0
    for snap in [s for s in SNAPSHOTS if s > 0]:
        seg = items[prev:snap]
        m.train()
        for _ in range(EPOCHS):
            for q, tgt in seg:
                _distill_step(tok, m, q, tgt, sp, fp, opt_s, opt_f)
        m.eval()
        rows += score_set(tok, m, eval_qs, learner, snap, scope)
        prev = snap
    return rows


def main():
    t0 = time.time()
    d = load_raw()
    bank, eval_qs, raw, personas = d["bank"], d["eval_qs"], d["raw"], d["personas"]
    log(f"LLM-teacher hybrid OPD: {len(personas)} personas, blend(realized)={BLEND}, snaps={SNAPSHOTS}, ep={EPOCHS}")

    tok, base = S.load_base()
    base.eval()

    log("building LLM-teacher targets (frozen base + privileged prompt) ...")
    opd_items = build_targets(tok, base, raw)
    log(f"built targets for {len(opd_items)} personas ({sum(len(v) for v in opd_items.values())} items)")

    if not (PRED / "__base__.jsonl").exists():
        base_rows = score_set(tok, base, eval_qs, "__base__", 0, "base")
        (PRED / "__base__.jsonl").write_text("\n".join(json.dumps(r) for r in base_rows) + "\n")
        log("scored A-null (__base__)")
    else:
        base_rows = [json.loads(l) for l in (PRED / "__base__.jsonl").read_text().splitlines()]
    base_pred = [{k: r[k] for k in ("subject_id", "question_id", "format",
                                    "option_logprobs", "p_correct_raw", "p_correct")} | {"snapshot": 0}
                 for r in base_rows]

    m = S.attach_dual_lora(base)
    init = S._init_adapter_state(m)

    if not (PRED / "__shared__.jsonl").exists():
        pooled = [it for pid in personas for it in opd_items[pid]]
        log(f"distilling A0 shared on {len(pooled)} pooled items")
        S._reset_adapters(m, init)
        sp, fp = S._split_params(m)
        opt_s = torch.optim.AdamW(sp, lr=1e-5); opt_f = torch.optim.AdamW(fp, lr=2e-4)
        m.train()
        for q, tgt in pooled:
            _distill_step(tok, m, q, tgt, sp, fp, opt_s, opt_f)
        m.eval()
        shared_rows = []
        for snap in SNAPSHOTS:
            shared_rows += score_set(tok, m, eval_qs, "__shared__", snap, "shared")
        (PRED / "__shared__.jsonl").write_text("\n".join(json.dumps(r) for r in shared_rows) + "\n")
        log("scored A0 (__shared__)")

    done = sorted(p.stem for p in PRED.glob("persona_*.jsonl"))
    for i, pid in enumerate(personas):
        if pid in done:
            log(f"[{i+1}/{len(personas)}] {pid} cached"); continue
        try:
            tp = time.time()
            base_for_p = [dict(r, learner_id=pid, scope="per_user") for r in base_pred]
            rows = train_and_score(tok, m, init, opd_items[pid], eval_qs, pid, "per_user", base_for_p)
            (PRED / f"{pid}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            log(f"[{i+1}/{len(personas)}] {pid} done {time.time()-tp:.0f}s (gpu {torch.cuda.max_memory_allocated()/1e9:.1f}GB)")
        except Exception as e:
            log(f"[{i+1}/{len(personas)}] {pid} FAILED: {e}\n{traceback.format_exc()}")
        set_status({"phase": "A1-OPD-LLM", "done": i + 1, "total": len(personas),
                    "elapsed_min": round((time.time() - t0) / 60, 1)})
    set_status({"phase": "complete", "elapsed_min": round((time.time() - t0) / 60, 1)})
    log(f"ALL DONE in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
