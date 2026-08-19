"""Print concrete good/bad health next-state predictions (qualitative diagnosis).

  python scripts/health/inspect_health_preds.py [--tag current] [--k 4]

Uses the best health model (shared LoRA + the given context cond) and, for each test record of the
12 trained pids, generates the predicted next-day state; contrasts with the frozen base (adapter
disabled) and persistence (=today's state). Prints full input/output for:
  - BEST   : lowest model per-record MAE
  - WORST  : highest model per-record MAE
  - WINS   : largest (persistence_MAE - model_MAE), i.e. genuine dynamic predictions the model
             got right where copying today would have been wrong.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch
from domains.health.data import (FIELDS, SYS, build_prompt, load_records, parse_state,
                                participant_baselines, render_state)
from domains.health.peruser_data import cond_tag

BASE = "Qwen/Qwen3-4B-Instruct-2507"
PIDS = ["p01", "p02", "p04", "p05", "p06", "p07", "p08", "p10", "p11", "p12", "p14", "p15"]


def rec_mae(pred, gold):
    return sum(abs(pred[f] - gold[f]) for f in FIELDS) / len(FIELDS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", default="+current")
    ap.add_argument("--k", type=int, default=4)
    args = ap.parse_args()
    import os
    tag = cond_tag(args.cond)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    train = load_records("train")
    test = [r for r in load_records("test") if r.pid in PIDS]
    baselines = participant_baselines(train)
    pop_mean = {f: round(sum(b[f] for b in baselines.values()) / len(baselines)) for f in FIELDS}

    tok = AutoTokenizer.from_pretrained(BASE)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    shared = Path(os.environ["UWM_MODELS"]) / "peruser" / f"health_shared_{tag}"
    pm = PeftModel.from_pretrained(model, str(shared), adapter_name="shared")

    prompts = [tok.apply_chat_template(
        [{"role": "system", "content": SYS},
         {"role": "user", "content": build_prompt(r, args.cond, baselines.get(r.pid, pop_mean))}],
        tokenize=False, add_generation_prompt=True) for r in test]

    @torch.no_grad()
    def gen():
        outs = []
        for i in range(0, len(prompts), 64):
            chunk = prompts[i:i + 64]
            enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            g = pm.generate(**enc, max_new_tokens=64, do_sample=False, pad_token_id=tok.pad_token_id)
            for j in range(len(chunk)):
                outs.append(tok.decode(g[j][enc["input_ids"].shape[1]:], skip_special_tokens=True))
        return outs

    pm.set_adapter("shared")
    model_txt = gen()
    with pm.disable_adapter():
        frozen_txt = gen()

    rows = []
    for i, r in enumerate(test):
        mp = parse_state(model_txt[i], r.state_n)
        fp = parse_state(frozen_txt[i], r.state_n)
        rows.append({
            "r": r, "model_pred": mp, "frozen_pred": fp,
            "model_mae": rec_mae(mp, r.state_n1),
            "frozen_mae": rec_mae(fp, r.state_n1),
            "persist_mae": rec_mae(r.state_n, r.state_n1),
            "model_raw": model_txt[i].strip().replace("\n", " ")[:120],
        })

    def show(title, items):
        print("\n" + "=" * 100 + f"\n{title}\n" + "=" * 100)
        for x in items:
            r = x["r"]
            print(f"\n--- pid={r.pid}  activity={r.activity} {r.duration_min:.0f}min"
                  f"   [model_MAE={x['model_mae']:.2f}  frozen_MAE={x['frozen_mae']:.2f}  "
                  f"persist_MAE={x['persist_mae']:.2f}] ---")
            if r.event_text:
                print(f"  event   : {r.event_text[:150]}")
            print(f"  TODAY   (input state_n) : {render_state(r.state_n)}")
            print(f"  GOLD    (true next n+1) : {render_state(r.state_n1)}")
            print(f"  MODEL   (shared+{tag})  : {render_state(x['model_pred'])}")
            print(f"  frozen base             : {render_state(x['frozen_pred'])}")
            print(f"  persistence (=today)    : {render_state(r.state_n)}")
            print(f"  model raw gen           : {x['model_raw']!r}")

    k = args.k
    show(f"BEST {k} — lowest model per-record MAE", sorted(rows, key=lambda x: x["model_mae"])[:k])
    show(f"WORST {k} — highest model per-record MAE", sorted(rows, key=lambda x: -x["model_mae"])[:k])
    show(f"BIGGEST WINS vs persistence {k} — model right where copying-today is wrong",
         sorted(rows, key=lambda x: (x["model_mae"] - x["persist_mae"]))[:k])
    show(f"BIGGEST LOSSES vs persistence {k} — model worse than just copying today",
         sorted(rows, key=lambda x: -(x["model_mae"] - x["persist_mae"]))[:k])

    mm = sum(x["model_mae"] for x in rows) / len(rows)
    pp = sum(x["persist_mae"] for x in rows) / len(rows)
    ff = sum(x["frozen_mae"] for x in rows) / len(rows)
    print(f"\n[overall] n={len(rows)}  model={mm:.3f}  frozen={ff:.3f}  persistence={pp:.3f}")


if __name__ == "__main__":
    main()
