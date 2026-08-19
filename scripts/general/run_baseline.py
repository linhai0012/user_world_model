"""Run one baseline over a PersonaMem bench (CONVENTIONS.md §3, §5).

  source scripts/env.sh
  python scripts/general/run_baseline.py --run-id bl__trivial__qwen3-4b__pm32k [--limit 50]

Reads experiments/configs/<run_id>.yaml, loads the shared PersonaMem loader, asks the
method (domains/general/baselines/<method>/predict.py) for the context to inject per MCQ, scores every
option with the shared backend (MCQ-PPL), writes accuracy + per-qtype to
experiments/results/<run_id>__acc.json and full preds + run_meta to $UWM_RUNS/<run_id>/.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from domains.general import scorer
from common.backends import make_backend
from domains.general.data import PersonaMem
from common.runmeta import load_config, run_dir, write_run_meta


def _load_method(method: str):
    path = REPO / "domains" / "general" / "baselines" / method / "predict.py"
    if not path.exists():
        # token-memory methods live under domains/general/baselines/tokenmem/<method>/
        path = REPO / "domains" / "general" / "baselines" / "tokenmem" / method / "predict.py"
    if not path.exists():
        raise FileNotFoundError(f"no predict.py for method {method!r}")
    spec = importlib.util.spec_from_file_location(f"baselines_{method}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--limit", type=int, default=None, help="cap #MCQs (smoke testing)")
    ap.add_argument("--batch", type=int, default=256, help="MCQs per backend scoring call")
    args = ap.parse_args()

    cfg = load_config(args.run_id)
    method, model, bench = cfg["method"], cfg["model"], cfg["bench"]
    params = cfg.get("params") or {}
    print(f"[run] {args.run_id}  method={method} model={model} bench={bench} params={params}")

    data = PersonaMem(bench, data_dir=cfg.get("data_dir"))
    mcqs = data.load_mcqs(limit=args.limit)
    method_mod = _load_method(method)

    # backend sized to the bench (oracle/full needs long context; slice/trivial are short)
    max_len = {"pm32k": 8192, "pm128k": 32768, "pm1m": 32768}.get(bench, 8192)
    backend = make_backend(model, max_model_len=max_len)

    t0 = time.time()
    records, preds = [], []
    for i in range(0, len(mcqs), args.batch):
        chunk = mcqs[i:i + args.batch]
        items = [(method_mod.build_context(m, data, params), m.query, m.options) for m in chunk]
        scored = backend.score_batch(items)
        for m, sc in zip(chunk, scored):
            records.append({"qtype": m.qtype, "gold": m.gold_letter, "pred": sc.pred_letter})
            preds.append({"qid": m.qid, "persona_id": m.persona_id, "qtype": m.qtype,
                          "gold": m.gold_letter, "pred": sc.pred_letter,
                          "nll": {k: round(v, 4) for k, v in sc.per_choice_nll.items()}})
        print(f"  scored {min(i + args.batch, len(mcqs))}/{len(mcqs)}", flush=True)

    summary = scorer.score(records)
    summary["run_id"] = args.run_id
    summary["wall_s"] = round(time.time() - t0, 1)
    print(json.dumps({k: summary[k] for k in ("n", "accuracy", "wall_s")}, indent=2))

    res_path = scorer.write_results(args.run_id, REPO / "experiments" / "results" / "general", summary)
    rd = run_dir(args.run_id)
    (rd / "preds.jsonl").write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in preds))
    write_run_meta(args.run_id, cfg, extra={"summary": summary, "n_mcqs": len(mcqs)})
    print(f"[done] results → {res_path}\n        preds   → {rd/'preds.jsonl'}")


if __name__ == "__main__":
    main()
