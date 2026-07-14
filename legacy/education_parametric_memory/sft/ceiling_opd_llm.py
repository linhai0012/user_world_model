"""LLM-teacher OPD ceiling test (the DEPLOYABLE recipe, vs oracle-teacher and hard SFT).

Same transfer probe as ceiling_opd.py, but the privileged teacher is NOT the generator g — it is
the FROZEN base model conditioned on a privileged "predict this student's answer" prompt carrying
the held misconception + a coarse mastery bucket. The student LoRA (stem-only) distills toward this
LLM-teacher's option distribution. This is what transfers to NON-synthetic data, where g does not
exist and the teacher is an LLM reading a learner's misconception card.

Logic of the three-way read:
  hard SFT     : single chosen-option text (sparse hard label)            -> memorises, weak transfer
  oracle-OPD   : distill g(theta,q) exact categorical (upper bound)       -> is the signal distillable?
  LLM-teacher  : distill base+privileged-prompt categorical (realistic)   -> can a real teacher make it?

If LLM-teacher ~ oracle-OPD > hard, the deployable recipe works. If LLM-teacher ~ hard while
oracle-OPD > hard, the base model won't role-play the misconception from a card (teacher-quality
limit, a separate fixable problem). GPU-only (vllm_env).

Env: CUE_CEIL_EPOCHS(15) CUE_CEIL_PIDS CUE_CEIL_SNAP(104) CUE_TEACHER_TEMP(1.0) CUE_SUBSTRATE.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import sft_core as S
from personas import Persona, PersonaState

SUB = Path(os.environ.get("CUE_SUBSTRATE", str(ROOT / "substrate")))
SNAP = int(os.environ.get("CUE_CEIL_SNAP", "104"))
EPOCHS = int(os.environ.get("CUE_CEIL_EPOCHS", "15"))
PIDS = os.environ.get("CUE_CEIL_PIDS", "persona_03,persona_07,persona_11").split(",")
TEMP = float(os.environ.get("CUE_TEACHER_TEMP", "1.0"))


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


bank = {json.loads(l)["_id"]: json.loads(l)
        for l in (ROOT / "question_bank/biology_gcse.jsonl").read_text().splitlines()}
splits = json.loads((SUB / "splits.json").read_text())
eval_ids = splits["eval"]
all_skills = sorted({q["skill_id"] for q in bank.values()})
srows = [json.loads(l) for l in (SUB / "streams.jsonl").read_text().splitlines()]
etruth = {(r["persona"], r["snapshot"], r["question_id"]): r
          for r in (json.loads(l) for l in (SUB / "eval_truth.jsonl").read_text().splitlines())}
personas_json = {p["pid"]: p
                 for p in json.loads((SUB / "persona_set.json").read_text())["personas"]}


def load_persona(pj: dict) -> Persona:
    held = {s: (v["tag"], v["strength"]) for s, v in pj["held"].items()}
    return Persona(pid=pj["pid"], ability=pj["ability"], topic_offset=pj["topic_offset"],
                   skill_jitter=pj["skill_jitter"], held=held)


def _bucket(m: float) -> str:
    return "low" if m < 0.4 else "medium" if m < 0.7 else "high"


def _teacher_prompt(stem: str, skill_name: str, misconception, mastery: float) -> str:
    pre = "You are predicting how one specific student will answer a multiple-choice question. "
    if misconception:
        pre += (f"This student holds a misconception about {skill_name}: \"{misconception}\". "
                "They tend to answer in a way that reflects this misconception. ")
    pre += f"Their mastery of {skill_name} is {_bucket(mastery)}.\n"
    return f"{pre}Question: {stem}\nThe student's answer:"


@torch.no_grad()
def _teacher_logprob(tok, base, t_prompt: str, cont_text: str) -> float:
    p = tok(t_prompt, add_special_tokens=False).input_ids
    c = tok(" " + cont_text, add_special_tokens=False).input_ids
    ids = torch.tensor([p + c], device="cuda")
    logits = base(input_ids=ids).logits[0]
    start = len(p) - 1
    lp = F.log_softmax(logits[start:start + len(c)].float(), dim=-1)
    tok_lp = lp[range(len(c)), torch.tensor(c, device="cuda")]
    return float(tok_lp.mean().item())


def teacher_dist_llm(tok, base, state: PersonaState, q: dict) -> dict:
    """Frozen base + privileged misconception prompt -> normalized option categorical."""
    skill = q["skill_id"]
    skill_name = skill.split("__")[-1].replace("_", " ")
    misc = state.misconception_for(skill)
    held = state.persona.held.get(skill)
    misc_text = held[0] if (misc and held) else None
    tp = _teacher_prompt(q["stem"], skill_name, misc_text, state.mastery_of(skill))
    lps = {o["id"]: _teacher_logprob(tok, base, tp, o["text"]) for o in q["options"]}
    keys = list(lps)
    z = torch.logsumexp(torch.tensor([lps[k] for k in keys]), dim=0).item()
    return {k: math.exp(lps[k] - z) for k in keys}


def build_streams_llm(tok, base, pid: str):
    p = load_persona(personas_json[pid])
    state = PersonaState.initial(p, all_skills)
    pstream = sorted((r for r in srows if r["persona"] == pid), key=lambda x: x["round"])
    opd = []
    for r in pstream:
        q = bank[r["question_id"]]
        opd.append((q, teacher_dist_llm(tok, base, state, q)))   # state BEFORE practice
        state.practice(q["skill_id"])
    train_ids = [r["question_id"] for r in pstream]
    train_choice = {r["question_id"]: r["selected_option_id"] for r in pstream}
    return opd, train_ids, train_choice


# --------------------------------------------------------------- metrics (shared with ceiling_opd)
def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    return cov / math.sqrt(vx * vy) if vx > 0 and vy > 0 else float("nan")


def _opt_logprob_grad(m, stem: str, cont_text: str, skill):
    p = tok(S._prompt(stem, skill), add_special_tokens=False).input_ids
    c = tok(" " + cont_text, add_special_tokens=False).input_ids
    ids = torch.tensor([p + c], device="cuda")
    logits = m(input_ids=ids).logits[0]
    start = len(p) - 1
    lp = F.log_softmax(logits[start:start + len(c)].float(), dim=-1)
    tok_lp = lp[range(len(c)), torch.tensor(c, device="cuda")]
    return tok_lp.mean()


def choice_acc(model, items, truth_choice) -> float:
    hit = 0
    for qid in items:
        sc = S.score_mcq(tok, model, bank[qid])
        top = max(sc["option_logprobs"], key=sc["option_logprobs"].get)
        hit += int(top == truth_choice[qid])
    return hit / len(items)


def eval_metrics(model, pid: str) -> dict:
    ec = {qid: etruth[(pid, SNAP, qid)] for qid in eval_ids if (pid, SNAP, qid) in etruth}
    ps, ms = [], []
    choice_hit = misc_total = misc_hit = revert = 0
    for qid, t in ec.items():
        q = bank[qid]
        sc = S.score_mcq(tok, model, q)
        top = max(sc["option_logprobs"], key=sc["option_logprobs"].get)
        correct_id = next(o["id"] for o in q["options"] if o["is_correct"])
        choice_hit += int(top == t["selected_option_id"])
        ps.append(sc["p_correct"]); ms.append(t["mastery"])
        if not t["is_correct"]:
            misc_total += 1
            misc_hit += int(top == t["selected_option_id"])
            revert += int(top == correct_id)
    return {"eval_choice_acc": choice_hit / len(ec), "eval_mastery_corr": pearson(ps, ms),
            "misc_hit": (misc_hit / misc_total) if misc_total else float("nan"),
            "revert_correct": (revert / misc_total) if misc_total else float("nan"),
            "n_eval": len(ec), "n_wrong": misc_total}


def train_opd(opd) -> None:
    S._reset_adapters(m, init)
    sp, fp = S._split_params(m)
    opt_s, opt_f = torch.optim.AdamW(sp, lr=1e-5), torch.optim.AdamW(fp, lr=2e-4)
    m.train()
    for _ in range(EPOCHS):
        for q, td in opd:
            skill_key = q.get("skill_id", "").split("__")[-1]
            opts = q["options"]
            s = torch.stack([_opt_logprob_grad(m, q["stem"], o["text"], skill_key) for o in opts])
            logq = F.log_softmax(s / TEMP, dim=0)
            pvec = torch.tensor([td[o["id"]] for o in opts], device="cuda", dtype=logq.dtype)
            loss = -(pvec * logq).sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sp, 1.0); torch.nn.utils.clip_grad_norm_(fp, 1.0)
            opt_s.step(); opt_f.step(); opt_s.zero_grad(); opt_f.zero_grad()
    m.eval()


# --------------------------------------------------------------- run
log(f"LLM-teacher OPD: substrate={SUB.name} snap={SNAP} epochs={EPOCHS} temp={TEMP} pids={PIDS}")
tok, base = S.load_base()
base.eval()
m = S.attach_dual_lora(base)
init = S._init_adapter_state(m)

results = {}
for pid in PIDS:
    t0 = time.time()
    opd, train_ids, train_choice = build_streams_llm(tok, base, pid)
    log(f"{pid}: teacher targets built {time.time()-t0:.0f}s; base eval ...")
    base_m = eval_metrics(base, pid)
    base_tr = choice_acc(base, train_ids, train_choice)

    t0 = time.time()
    train_opd(opd)
    opd_tr = choice_acc(m, train_ids, train_choice)
    opd_m = eval_metrics(m, pid)
    log(f"{pid}: LLM-OPD done {time.time()-t0:.0f}s (train-acc {opd_tr:.3f}) "
        f"gpu {torch.cuda.max_memory_allocated()/1e9:.1f}GB")
    results[pid] = {"base": {**base_m, "train_acc": base_tr},
                    "llm_opd": {**opd_m, "train_acc": opd_tr}}

print("\n" + "=" * 92)
print("LLM-TEACHER OPD — ceiling/transfer test (held-out eval @ snap %d, %d epochs)" % (SNAP, EPOCHS))
print("=" * 92)
hdr = f"{'persona':>11} {'arm':>8} {'train_acc':>9} {'eval_acc':>8} {'mast_corr':>9} {'misc_hit':>8} {'revert→✓':>9}"
for pid in PIDS:
    print("\n" + hdr)
    for arm in ("base", "llm_opd"):
        r = results[pid][arm]
        print(f"{pid:>11} {arm:>8} {r['train_acc']:>9.3f} {r['eval_choice_acc']:>8.3f} "
              f"{r['eval_mastery_corr']:>9.3f} {r['misc_hit']:>8.3f} {r['revert_correct']:>9.3f}")
print("\n" + "=" * 92 + "\nPOOLED MEAN")
print(hdr)
for arm in ("base", "llm_opd"):
    def mean(k):
        vs = [results[p][arm][k] for p in PIDS if not math.isnan(results[p][arm][k])]
        return sum(vs) / len(vs) if vs else float("nan")
    print(f"{'ALL':>11} {arm:>8} {mean('train_acc'):>9.3f} {mean('eval_choice_acc'):>8.3f} "
          f"{mean('eval_mastery_corr'):>9.3f} {mean('misc_hit'):>8.3f} {mean('revert_correct'):>9.3f}")
print("\nCompare mast_corr/misc_hit to oracle-OPD (ceiling_opd.py) and hard SFT to see if a")
print("realistic LLM teacher recovers the privileged signal.")
