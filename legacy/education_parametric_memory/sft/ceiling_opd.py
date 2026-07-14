"""OPD oracle-teacher minimal ablation, run AS a ceiling test (transfer probe).

Head-to-head on the SAME personas / substrate / dual-rate LoRA / epochs as ceiling_test.py;
the ONLY change is the training loss:

  HARD (current lean SFT) : causal-LM CE on the learner's single chosen-option TEXT.
  OPD  (oracle-teacher)   : soft-label distillation of the student's 4-way option distribution
                            toward the GENERATOR's exact categorical g(theta,q) at that round's
                            state. theta (mastery + held misconception) is the PRIVILEGED signal
                            given only to the teacher; the student still sees stem-only and must
                            internalise "misconception -> mass shifts to the consistent distractor"
                            into the weights. Loss = -sum_o p_teacher(o) log q_student(o) (= forward
                            KL / Hinton soft-CE), mass-covering so the student must cover the
                            misconception mode, not collapse to the base's "correct answer" prior.

This answers: is the transfer ceiling a RECIPE limit (hard labels too sparse) or a STUDENT-capacity
limit? If oracle-OPD lifts held-out transfer / misconception-hit over hard SFT, the recipe is the
fix and the realistic LLM-teacher OPD is worth building. If not, the ceiling is student capacity.

Metrics per arm (held-out eval @ snapshot 104, vs hard SFT and the no-adapter base):
  TRAIN choice-acc  (memorise check)  · EVAL choice-acc (argmax transfer)
  EVAL mastery-corr (does p_correct track TRUE latent mastery — the metric that actually moved)
  EVAL misc-hit     (on items the learner got WRONG: does argmax pick the learner's OWN distractor)
  EVAL revert-correct (on WRONG items: does argmax fall back to the correct answer = base prior)

GPU-only (vllm_env). Env: CUE_CEIL_EPOCHS (15), CUE_CEIL_PIDS, CUE_TEACHER_TEMP (1.0).
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
from personas import Persona, PersonaState, _WRONG_BIAS

SUB = Path(os.environ.get("CUE_SUBSTRATE", str(ROOT / "substrate")))
SNAP = int(os.environ.get("CUE_CEIL_SNAP", "104"))
EPOCHS = int(os.environ.get("CUE_CEIL_EPOCHS", "15"))
PIDS = os.environ.get("CUE_CEIL_PIDS", "persona_03,persona_07,persona_11").split(",")
TEMP = float(os.environ.get("CUE_TEACHER_TEMP", "1.0"))


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------- data
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


def teacher_dist(state: PersonaState, q: dict) -> dict:
    """The generator g(theta,q)'s EXACT categorical over option ids (analytic, not sampled).
    Mirrors personas.answer_mcq: correct w.p. mastery; wrong mass biased to the held distractor."""
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


def build_streams(pid: str):
    """Replay the persona's state through its round_sequence; per round capture both the HARD
    target (its chosen-option text) and the OPD target (teacher categorical at that state)."""
    p = load_persona(personas_json[pid])
    state = PersonaState.initial(p, all_skills)
    pstream = sorted((r for r in srows if r["persona"] == pid), key=lambda x: x["round"])
    hard, opd = [], []
    for r in pstream:
        q = bank[r["question_id"]]
        skill_key = q.get("skill_id", "").split("__")[-1]
        hard.append((q["stem"], r["answer_text"], skill_key))
        opd.append((q, teacher_dist(state, q)))      # state BEFORE practice (matches simulate)
        state.practice(q["skill_id"])
    train_choice = {r["question_id"]: r["selected_option_id"] for r in pstream}
    train_ids = [r["question_id"] for r in pstream]
    return hard, opd, train_ids, train_choice


# --------------------------------------------------------------- scoring helpers
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
    """Grad-enabled twin of sft_core._mean_logprob (mean continuation-token logprob)."""
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


# --------------------------------------------------------------- training arms
def _fresh_opts():
    S._reset_adapters(m, init)
    sp, fp = S._split_params(m)
    return sp, fp, torch.optim.AdamW(sp, lr=1e-5), torch.optim.AdamW(fp, lr=2e-4)


def train_hard(hard) -> None:
    sp, fp, os_, of = _fresh_opts()
    m.train()
    for _ in range(EPOCHS):
        for stem, ans, skill in hard:
            ids, labels = S._sample_ids(tok, stem, ans, skill)
            out = m(input_ids=torch.tensor([ids], device="cuda"),
                    labels=torch.tensor([labels], device="cuda"))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(sp, 1.0); torch.nn.utils.clip_grad_norm_(fp, 1.0)
            os_.step(); of.step(); os_.zero_grad(); of.zero_grad()
    m.eval()


def train_opd(opd) -> None:
    sp, fp, os_, of = _fresh_opts()
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
            os_.step(); of.step(); os_.zero_grad(); of.zero_grad()
    m.eval()


# --------------------------------------------------------------- run
log(f"substrate={SUB.name} snap={SNAP} epochs={EPOCHS} temp={TEMP} pids={PIDS}")
tok, base = S.load_base()
base.eval()
m = S.attach_dual_lora(base)
init = S._init_adapter_state(m)

results = {}
for pid in PIDS:
    hard, opd, train_ids, train_choice = build_streams(pid)
    log(f"{pid}: {len(hard)} rounds; base eval ...")
    base_m = eval_metrics(base, pid)
    base_tr = choice_acc(base, train_ids, train_choice)

    t0 = time.time()
    train_hard(hard)
    hard_tr = choice_acc(m, train_ids, train_choice)
    hard_m = eval_metrics(m, pid)
    log(f"{pid}: HARD done {time.time()-t0:.0f}s (train-acc {hard_tr:.3f})")

    t0 = time.time()
    train_opd(opd)
    opd_tr = choice_acc(m, train_ids, train_choice)
    opd_m = eval_metrics(m, pid)
    log(f"{pid}: OPD  done {time.time()-t0:.0f}s (train-acc {opd_tr:.3f}) "
        f"gpu {torch.cuda.max_memory_allocated()/1e9:.1f}GB")

    results[pid] = {"base": {**base_m, "train_acc": base_tr},
                    "hard": {**hard_m, "train_acc": hard_tr},
                    "opd": {**opd_m, "train_acc": opd_tr}}

# --------------------------------------------------------------- report
print("\n" + "=" * 96)
print("ORACLE-TEACHER OPD vs HARD SFT — ceiling/transfer test (held-out eval @ snap %d, %d epochs)"
      % (SNAP, EPOCHS))
print("=" * 96)
hdr = f"{'persona':>11} {'arm':>5} {'train_acc':>9} {'eval_acc':>8} {'mast_corr':>9} {'misc_hit':>8} {'revert→✓':>9}"
for pid in PIDS:
    print("\n" + hdr)
    for arm in ("base", "hard", "opd"):
        r = results[pid][arm]
        print(f"{pid:>11} {arm:>5} {r['train_acc']:>9.3f} {r['eval_choice_acc']:>8.3f} "
              f"{r['eval_mastery_corr']:>9.3f} {r['misc_hit']:>8.3f} {r['revert_correct']:>9.3f}")

# pooled means
print("\n" + "=" * 96 + "\nPOOLED MEAN across personas")
print(hdr)
for arm in ("base", "hard", "opd"):
    def mean(k):
        vs = [results[p][arm][k] for p in PIDS if not math.isnan(results[p][arm][k])]
        return sum(vs) / len(vs) if vs else float("nan")
    print(f"{'ALL':>11} {arm:>5} {mean('train_acc'):>9.3f} {mean('eval_choice_acc'):>8.3f} "
          f"{mean('eval_mastery_corr'):>9.3f} {mean('misc_hit'):>8.3f} {mean('revert_correct'):>9.3f}")

print("\nReadout: OPD should LIFT mast_corr & misc_hit and DROP revert→✓ vs HARD if the privileged")
print("soft target fixes transfer. If OPD≈HARD on held-out, the ceiling is student capacity, not recipe.")
