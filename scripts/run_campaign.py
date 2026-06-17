"""Sweep several baselines over a bench in ONE model load (CONVENTIONS.md §3, §5).

  source scripts/env.sh
  python scripts/run_campaign.py --bench pm32k --methods trivial,profile,oracle [--limit N]

Each method → run_id `bl__<method>__qwen3-4b__<bench>[__<variant>]`, scored by the shared
MCQ-PPL backend, with results → experiments/results/<run_id>__acc.json and preds/run_meta →
$UWM_RUNS/<run_id>/. The model is loaded once and reused across methods.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from common import scorer
from common.backends import make_backend
from common.data import PersonaMem
from common.runmeta import run_dir, write_run_meta

# method spec → (method_module_name, subdir, params, variant)
SPECS = {
    "trivial":        ("trivial", "",          {}, None),
    "profile":        ("profile", "",          {}, None),
    "oracle":         ("oracle",  "",          {"mode": "slice", "window": 2}, None),
    "oracle+profile": ("oracle",  "",          {"mode": "slice", "window": 2, "with_profile": True}, "wprofile"),
    "oracle-full":    ("oracle",  "",          {"mode": "full", "max_turns": None}, "full"),
    "naiverag":       ("naiverag", "tokenmem/", {"top_k": 5, "retriever": "bm25"}, None),
    "naiverag-dense": ("naiverag", "tokenmem/", {"top_k": 5, "retriever": "dense"}, "dense"),
}


def _load_method(module_name: str, subdir: str):
    path = REPO / "baselines" / subdir / module_name / "predict.py"
    spec = importlib.util.spec_from_file_location(f"m_{module_name}_{subdir.strip('/')}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_id(method_key: str, bench: str, eval_lens: str = "ppl") -> str:
    mod, _, _, variant = SPECS[method_key]
    rid = f"bl__{mod}__qwen3-4b__{bench}"
    if variant:
        rid = f"{rid}__{variant}"
    if eval_lens == "gen":           # ppl is the default lens (no suffix; matches legacy)
        rid = f"{rid}__gen"
    return rid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=["pm32k", "pm128k", "pm1m"])
    ap.add_argument("--methods", default="trivial,profile,oracle")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--max-model-len", type=int, default=None,
                    help="override; default auto from the methods (long-ctx needs more)")
    ap.add_argument("--eval", choices=["ppl", "gen"], default="ppl",
                    help="ppl = MCQ choice-perplexity (UserSim protocol); gen = reader answers")
    ap.add_argument("--persona", default=None, help="restrict to one persona_id (per-user arm)")
    ap.add_argument("--model-path", default=None,
                    help="override the served model (e.g. a merged per-user LoRA); run_id gets a tag")
    ap.add_argument("--run-tag", default=None, help="extra run_id suffix (e.g. 'peruser')")
    args = ap.parse_args()

    method_keys = [m.strip() for m in args.methods.split(",") if m.strip()]
    for k in method_keys:
        if k not in SPECS:
            raise SystemExit(f"unknown method {k!r}; known: {list(SPECS)}")

    data = PersonaMem(args.bench)
    mcqs = data.load_mcqs(limit=args.limit)
    if args.persona is not None:
        mcqs = [m for m in mcqs if m.persona_id == args.persona]
    print(f"[campaign] bench={args.bench} n_mcqs={len(mcqs)} methods={method_keys} "
          f"persona={args.persona} model={args.model_path or 'base'}", flush=True)

    # long-context methods need a big window; short ones don't — size to the max requested.
    long_ctx = any(SPECS[k][2].get("mode") == "full" for k in method_keys)
    bench_cap = {"pm32k": 33000, "pm128k": 40960, "pm1m": 40960}[args.bench]
    max_len = args.max_model_len or (bench_cap if long_ctx else 4096)
    bk_kw = {"max_model_len": max_len}
    if args.model_path:
        bk_kw["model"] = args.model_path
    backend = make_backend("qwen3-4b", **bk_kw)

    for k in method_keys:
        mod_name, subdir, params, _ = SPECS[k]
        run_id = _run_id(k, args.bench, args.eval)
        if args.persona is not None:
            run_id += f"__p{args.persona}"
        if args.run_tag:
            run_id += f"__{args.run_tag}"
        method = _load_method(mod_name, subdir)
        print(f"\n[campaign] === {run_id}  params={params} eval={args.eval} ===", flush=True)
        t0 = time.time()
        records, preds = [], []
        for i in range(0, len(mcqs), args.batch):
            chunk = mcqs[i:i + args.batch]
            items = [(method.build_context(m, data, params), m.query, m.options) for m in chunk]
            scored = (backend.pick_batch(items, max_ctx_tokens=max_len) if args.eval == "gen"
                      else backend.score_batch(items, max_ctx_tokens=max_len))
            for m, sc in zip(chunk, scored):
                records.append({"qtype": m.qtype, "gold": m.gold_letter, "pred": sc.pred_letter})
                preds.append({"qid": m.qid, "persona_id": m.persona_id, "qtype": m.qtype,
                              "gold": m.gold_letter, "pred": sc.pred_letter,
                              "nll": {kk: round(vv, 4) for kk, vv in sc.per_choice_nll.items()}})
            print(f"  {min(i + args.batch, len(mcqs))}/{len(mcqs)}", flush=True)

        summary = scorer.score(records)
        summary.update({"run_id": run_id, "method": k, "params": params, "eval": args.eval,
                        "bench": args.bench, "wall_s": round(time.time() - t0, 1)})
        scorer.write_results(run_id, REPO / "experiments" / "results", summary)
        rd = run_dir(run_id)
        (rd / "preds.jsonl").write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in preds))
        write_run_meta(run_id, {"run_id": run_id, "method": k, "params": params,
                                "bench": args.bench, "model": "qwen3-4b"},
                       extra={"summary": summary, "n_mcqs": len(mcqs)})
        print(f"[done] {run_id}  acc={summary['accuracy']}  ({summary['wall_s']}s)", flush=True)


if __name__ == "__main__":
    main()
