"""Eval the HEALTH per-user LoRA arm vs frozen base, head-to-head on each participant's test set.

  source scripts/env.sh
  python scripts/eval_health_peruser.py                 # all pids with a trained adapter
  python scripts/eval_health_peruser.py --pids p14,p08

For each pid we predict next-day state (base context) two ways from the SAME model object:
  base     frozen Qwen3-4B with the adapter DISABLED
  peruser  the same model with that pid's LoRA adapter ENABLED
and report per-field + overall MAE for {persistence, pop-mean, base, peruser}, per-pid and pooled.
HF greedy decoding (temp 0) — base and peruser go through the identical path, so the Δ is clean.
Mirrors the general-domain per-user arm: the only change between base and peruser is the weights.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import statistics

import torch
from common.health_data import (FIELDS, SYS, build_prompt, load_records,
                                parse_state, participant_baselines)
from common.health_peruser_data import cond_tag

BASE = "Qwen/Qwen3-4B-Instruct-2507"


def mae(preds: list[dict], golds: list[dict]) -> dict:
    out = {f: round(sum(abs(p[f] - g[f]) for p, g in zip(preds, golds)) / len(golds), 3)
           for f in FIELDS}
    out["overall"] = round(sum(out[f] for f in FIELDS) / len(FIELDS), 3)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pids", default=None, help="comma list; default = all trained adapters")
    ap.add_argument("--cond", default="base")
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    models_root = Path(os.environ["UWM_MODELS"]) / "peruser"
    tag = cond_tag(args.cond)
    suffix = f"_{tag}"
    if args.pids:
        pids = [p.strip() for p in args.pids.split(",")]
    else:
        pids = sorted(d.name[len("health_"):-len(suffix)] for d in models_root.glob(f"health_*{suffix}")
                      if (d / "adapter_config.json").exists())
    if not pids:
        raise SystemExit(f"no trained adapters under {models_root}/health_*{suffix}")
    print(f"[peruser-eval] cond={args.cond} tag={tag} pids={pids}", flush=True)

    train = load_records("train")
    test = load_records("test")
    baselines = participant_baselines(train)
    pop_mean = {f: round(sum(b[f] for b in baselines.values()) / len(baselines)) for f in FIELDS}
    # per-user-mean control: each pid's own train next-state mean (the "lookup-table" bar)
    umean = {pid: {f: round(statistics.mean(r.state_n1[f] for r in train if r.pid == pid))
                   for f in FIELDS} for pid in pids}

    tok = AutoTokenizer.from_pretrained(BASE)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()

    pm = None
    for pid in pids:
        path = str(models_root / f"health_{pid}{suffix}")
        if pm is None:
            pm = PeftModel.from_pretrained(base_model, path, adapter_name=pid)
        else:
            pm.load_adapter(path, adapter_name=pid)

    @torch.no_grad()
    def gen(prompts: list[str]) -> list[str]:
        texts: list[str] = []
        for i in range(0, len(prompts), args.batch_size):
            chunk = prompts[i:i + args.batch_size]
            enc = tok(chunk, return_tensors="pt", padding=True,
                      add_special_tokens=False).to("cuda")
            out = pm.generate(**enc, max_new_tokens=args.max_new, do_sample=False,
                              pad_token_id=tok.pad_token_id)
            for j in range(len(chunk)):
                texts.append(tok.decode(out[j][enc["input_ids"].shape[1]:],
                                        skip_special_tokens=True))
        return texts

    per_pid = {}
    pooled = {k: ([], []) for k in ("persistence", "pop-mean", "umean", "base", "peruser")}
    for pid in pids:
        recs = [r for r in test if r.pid == pid]
        if not recs:
            print(f"  [skip] {pid}: no test records", flush=True)
            continue
        golds = [r.state_n1 for r in recs]
        prompts = [tok.apply_chat_template(
            [{"role": "system", "content": SYS},
             {"role": "user", "content": build_prompt(r, args.cond, baselines.get(pid, pop_mean))}],
            tokenize=False, add_generation_prompt=True) for r in recs]

        with pm.disable_adapter():
            base_pred = [parse_state(t, recs[i].state_n) for i, t in enumerate(gen(prompts))]
        pm.set_adapter(pid)
        pu_pred = [parse_state(t, recs[i].state_n) for i, t in enumerate(gen(prompts))]

        arms = {"persistence": [r.state_n for r in recs], "pop-mean": [pop_mean] * len(recs),
                "umean": [umean[pid]] * len(recs), "base": base_pred, "peruser": pu_pred}
        per_pid[pid] = {"n": len(recs), **{k: mae(v, golds)["overall"] for k, v in arms.items()}}
        d = per_pid[pid]
        print(f"  {pid:4s} n={d['n']:3d}  persist={d['persistence']:.3f}  umean={d['umean']:.3f}  "
              f"base={d['base']:.3f}  peruser={d['peruser']:.3f}  "
              f"Δ(pu-umean)={d['peruser']-d['umean']:+.3f}  "
              f"Δ(pu-persist)={d['peruser']-d['persistence']:+.3f}", flush=True)
        for k, v in arms.items():
            pooled[k][0].extend(v)
            pooled[k][1].extend(golds)

    overall = {k: mae(p, g)["overall"] for k, (p, g) in pooled.items() if p}
    summary = {"task": "health_stage1_peruser", "cond": args.cond, "tag": tag, "pids": pids,
               "n_test_pooled": len(pooled["base"][0]),
               "overall_mae": overall, "per_pid": per_pid}
    out = REPO / "experiments" / "results" / f"health_peruser_{tag}__mae.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n[pooled] " + "  ".join(f"{k}={v:.3f}" for k, v in overall.items()))
    print(f"[done] -> {out}")


if __name__ == "__main__":
    main()
