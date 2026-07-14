"""Multi-arm recipe comparison (transfer probe) — what privileged-info TARGET fixes transfer?

Run A showed oracle-OPD (distilling g) RAISES overall correctness-prediction but LOWERS
misconception recovery (mast_corr/misc_hit down, revert-to-correct up): g is correct-dominated, so
distilling it teaches calibration, not the realized-error signal where the learner lives. This
probes whether an ERROR-FOCUSED target beats hard SFT, holding everything else fixed.

Four arms, identical dual-rate LoRA / epochs / substrate; only the per-item TARGET differs:
  hard         : token-CE on the realized chosen-option text (current recipe; mechanism = next-token)
  realized_opt : option-space soft-CE toward one-hot(realized choice)  [isolates mechanism vs target]
  opd          : option-space soft-CE toward g(theta,q)                [Run A's distribution distill]
  hybrid       : option-space soft-CE toward 0.5*g + 0.5*one-hot(realized)  [blend error into calib]

Readout: if hybrid/realized_opt lift mast_corr & misc_hit over opd (and >= hard), the realized-error
signal is what matters and the deployable recipe should carry it. If nothing beats hard, the ceiling
is student capacity / misconception-item sparsity (only ~3.4% of items carry an active distractor).
GPU-only (vllm_env). Env: CUE_CEIL_EPOCHS(6) CUE_CEIL_PIDS CUE_CEIL_SNAP(104) CUE_SUBSTRATE.
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
EPOCHS = int(os.environ.get("CUE_CEIL_EPOCHS", "6"))
PIDS = os.environ.get("CUE_CEIL_PIDS", "persona_03,persona_07,persona_11").split(",")
ARMS = os.environ.get("CUE_ARMS", "hard,realized_opt,opd,hybrid").split(",")

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


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_persona(pj):
    held = {s: (v["tag"], v["strength"]) for s, v in pj["held"].items()}
    return Persona(pid=pj["pid"], ability=pj["ability"], topic_offset=pj["topic_offset"],
                   skill_jitter=pj["skill_jitter"], held=held)


def teacher_dist(state, q):
    skill = q["skill_id"]; m = state.mastery_of(skill); opts = q["options"]
    cid = next((o["id"] for o in opts if o.get("is_correct")), opts[0]["id"])
    wrong = [o for o in opts if not o.get("is_correct")]; nw = len(wrong)
    ht = state.misconception_for(skill)
    fav = next((o for o in wrong if o.get("misconception_tag") == ht), None)
    wm = 1.0 - m; P = {cid: m}
    for o in wrong:
        P[o["id"]] = (wm * (_WRONG_BIAS + (1 - _WRONG_BIAS) / nw) if o["id"] == fav["id"]
                      else wm * ((1 - _WRONG_BIAS) / nw)) if fav else wm * (1.0 / nw)
    return P


def build_stream(pid):
    p = load_persona(personas_json[pid])
    state = PersonaState.initial(p, all_skills)
    pstream = sorted((r for r in srows if r["persona"] == pid), key=lambda x: x["round"])
    rows = []
    for r in pstream:
        q = bank[r["question_id"]]
        rows.append({"q": q, "ans_text": r["answer_text"], "realized": r["selected_option_id"],
                     "g": teacher_dist(state, q)})
        state.practice(q["skill_id"])
    train_ids = [r["question_id"] for r in pstream]
    train_choice = {r["question_id"]: r["selected_option_id"] for r in pstream}
    return rows, train_ids, train_choice


# --------------------------------------------------------------- targets (option-space)
def target_vec(arm, q, row):
    opts = q["options"]
    g = [row["g"][o["id"]] for o in opts]
    onehot = [1.0 if o["id"] == row["realized"] else 0.0 for o in opts]
    if arm == "opd":
        return g
    if arm == "realized_opt":
        return onehot
    if arm == "hybrid":
        return [0.5 * a + 0.5 * b for a, b in zip(g, onehot)]
    raise ValueError(arm)


# --------------------------------------------------------------- metrics
def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs); vy = sum((y - my) ** 2 for y in ys)
    return cov / math.sqrt(vx * vy) if vx > 0 and vy > 0 else float("nan")


def _opt_logprob_grad(m, stem, cont_text, skill):
    p = tok(S._prompt(stem, skill), add_special_tokens=False).input_ids
    c = tok(" " + cont_text, add_special_tokens=False).input_ids
    ids = torch.tensor([p + c], device="cuda")
    logits = m(input_ids=ids).logits[0]
    start = len(p) - 1
    lp = F.log_softmax(logits[start:start + len(c)].float(), dim=-1)
    return lp[range(len(c)), torch.tensor(c, device="cuda")].mean()


def choice_acc(model, items, truth_choice):
    hit = 0
    for qid in items:
        sc = S.score_mcq(tok, model, bank[qid])
        hit += int(max(sc["option_logprobs"], key=sc["option_logprobs"].get) == truth_choice[qid])
    return hit / len(items)


def eval_metrics(model, pid):
    ec = {qid: etruth[(pid, SNAP, qid)] for qid in eval_ids if (pid, SNAP, qid) in etruth}
    ps, ms = [], []
    choice_hit = misc_total = misc_hit = revert = 0
    for qid, t in ec.items():
        q = bank[qid]
        sc = S.score_mcq(tok, model, q)
        top = max(sc["option_logprobs"], key=sc["option_logprobs"].get)
        cid = next(o["id"] for o in q["options"] if o["is_correct"])
        choice_hit += int(top == t["selected_option_id"])
        ps.append(sc["p_correct"]); ms.append(t["mastery"])
        if not t["is_correct"]:
            misc_total += 1
            misc_hit += int(top == t["selected_option_id"])
            revert += int(top == cid)
    return {"eval_choice_acc": choice_hit / len(ec), "eval_mastery_corr": pearson(ps, ms),
            "misc_hit": (misc_hit / misc_total) if misc_total else float("nan"),
            "revert_correct": (revert / misc_total) if misc_total else float("nan")}


def _fresh():
    S._reset_adapters(m, init)
    sp, fp = S._split_params(m)
    return sp, fp, torch.optim.AdamW(sp, lr=1e-5), torch.optim.AdamW(fp, lr=2e-4)


def train_arm(arm, rows):
    sp, fp, os_, of = _fresh()
    m.train()
    for _ in range(EPOCHS):
        for row in rows:
            q = row["q"]; skill = q.get("skill_id", "").split("__")[-1]
            if arm == "hard":
                ids, labels = S._sample_ids(tok, q["stem"], row["ans_text"], skill)
                loss = m(input_ids=torch.tensor([ids], device="cuda"),
                         labels=torch.tensor([labels], device="cuda")).loss
            else:
                opts = q["options"]
                s = torch.stack([_opt_logprob_grad(m, q["stem"], o["text"], skill) for o in opts])
                logq = F.log_softmax(s, dim=0)
                pvec = torch.tensor(target_vec(arm, q, row), device="cuda", dtype=logq.dtype)
                loss = -(pvec * logq).sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sp, 1.0); torch.nn.utils.clip_grad_norm_(fp, 1.0)
            os_.step(); of.step(); os_.zero_grad(); of.zero_grad()
    m.eval()


# --------------------------------------------------------------- run
log(f"multi-arm: arms={ARMS} epochs={EPOCHS} snap={SNAP} pids={PIDS} substrate={SUB.name}")
tok, base = S.load_base()
base.eval()
m = S.attach_dual_lora(base)
init = S._init_adapter_state(m)

results = {}
for pid in PIDS:
    rows, train_ids, train_choice = build_stream(pid)
    results[pid] = {"base": {**eval_metrics(base, pid), "train_acc": choice_acc(base, train_ids, train_choice)}}
    for arm in ARMS:
        t0 = time.time()
        train_arm(arm, rows)
        results[pid][arm] = {**eval_metrics(m, pid), "train_acc": choice_acc(m, train_ids, train_choice)}
        log(f"{pid}/{arm}: {time.time()-t0:.0f}s train_acc={results[pid][arm]['train_acc']:.3f}")

print("\n" + "=" * 96)
print(f"MULTI-ARM RECIPE COMPARISON (held-out eval @ snap {SNAP}, {EPOCHS} epochs, {len(PIDS)} personas)")
print("=" * 96)
hdr = f"{'arm':>13} {'train_acc':>9} {'eval_acc':>8} {'mast_corr':>9} {'misc_hit':>8} {'revert→✓':>9}"
order = ["base"] + ARMS


def mean(arm, k):
    vs = [results[p][arm][k] for p in PIDS if not math.isnan(results[p][arm][k])]
    return sum(vs) / len(vs) if vs else float("nan")


print("\nPOOLED MEAN across personas")
print(hdr)
for arm in order:
    print(f"{arm:>13} {mean(arm,'train_acc'):>9.3f} {mean(arm,'eval_choice_acc'):>8.3f} "
          f"{mean(arm,'eval_mastery_corr'):>9.3f} {mean(arm,'misc_hit'):>8.3f} {mean(arm,'revert_correct'):>9.3f}")
print("\nper-persona:")
for pid in PIDS:
    print(f"\n  {pid}\n" + hdr)
    for arm in order:
        r = results[pid][arm]
        print(f"{arm:>13} {r['train_acc']:>9.3f} {r['eval_choice_acc']:>8.3f} "
              f"{r['eval_mastery_corr']:>9.3f} {r['misc_hit']:>8.3f} {r['revert_correct']:>9.3f}")

(Path("/scratch/prj/cllm/cue_sft") / "logs" / "ceiling_multi_results.json").write_text(json.dumps(results, indent=2))
