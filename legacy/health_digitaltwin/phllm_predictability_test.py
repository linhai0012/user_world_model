"""Sanity check: is next-day RHR / HRV predictable on PH-LLM data?

Compares three predictors on a sliding-window task:
  Input window: days 1..k of metrics (RHR, HRV, RR, sleep, activity)
  Target:       day k+1 morning RHR and HRV

Baselines:
  P  Persistence: predict next = day k value
  M  7-day mean:  predict next = mean of last 7 days
  L  Qwen3-4B zero-shot LLM (optional, --use_llm)

Usage:
  python phllm_predictability_test.py                      # baselines only
  python phllm_predictability_test.py --use_llm            # add Qwen3-4B zero-shot
  python phllm_predictability_test.py --n_cases 10 --window 14
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")


DATA_FILE = Path("output/external/fitness_all.jsonl")
DATA_URL = (
    "https://raw.githubusercontent.com/Google-Health/consumer-health-research"
    "/main/phllm/data/fitness_case_studies.all.v2.jsonl"
)
OUT_DIR = Path("output/phllm_predictability")


def ensure_data():
    """Download PH-LLM fitness case studies if not already present."""
    if DATA_FILE.exists() and DATA_FILE.stat().st_size > 1_000_000:
        return
    import urllib.request
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    print(f"[data] Downloading {DATA_URL} -> {DATA_FILE}", flush=True)
    urllib.request.urlretrieve(DATA_URL, DATA_FILE)
    print(f"[data] {DATA_FILE.stat().st_size/1e6:.1f} MB", flush=True)


ORD_RE = re.compile(r"the (\d+)(?:st|nd|rd|th) day\b", re.IGNORECASE)


def _parse_day_idx(line: str) -> Optional[int]:
    m = ORD_RE.search(line)
    return int(m.group(1)) if m else None


def _parse_floats(line: str) -> list[float]:
    """Pull all numeric tokens (ints / floats) from a line, in order."""
    return [float(t) for t in re.findall(r"-?\d+\.?\d*", line)]


def _row_tail(line: str) -> str:
    """Return the part of the line AFTER 'the Nth day' (the data columns)."""
    m = ORD_RE.search(line)
    return line[m.end():] if m else ""


def parse_health_metrics(text: str) -> dict[int, dict]:
    """Parse the 30-row daily table from health_metrics_input.

    Columns: RHR (bpm), HRV RMSSD (ms), Respiratory Rate.
    Some days may be missing data — we skip those rows.
    """
    out: dict[int, dict] = {}
    for line in text.split("\n"):
        d = _parse_day_idx(line)
        if d is None or not (1 <= d <= 30):
            continue
        # Strip "the Nth day" then read trailing floats
        tail = _row_tail(line)
        nums = _parse_floats(tail)
        if len(nums) >= 3:
            rhr, hrv, rr = nums[0], nums[1], nums[2]
            out[d] = {"rhr": rhr, "hrv": hrv, "rr": rr}
    return out


def parse_training_load(text: str) -> dict[int, dict]:
    """Parse the 30-row daily activity table from training_load_input."""
    # The daily table comes BEFORE the per-workout details / aggregates.
    # We cut at "Today is" or "Here are some aggregate" to avoid noise.
    cut_markers = ["Today is", "Here are some aggregate", "These are exercise logs"]
    for marker in cut_markers:
        if marker in text:
            text = text.split(marker, 1)[0]
            break
    out: dict[int, dict] = {}
    for line in text.split("\n"):
        d = _parse_day_idx(line)
        if d is None or not (1 <= d <= 30):
            continue
        tail = _row_tail(line)
        nums = _parse_floats(tail)
        if len(nums) >= 5:
            fb, ca, pk, trimp, steps = nums[0], nums[1], nums[2], nums[3], nums[4]
            out[d] = {"fat_burn": fb, "cardio": ca, "peak": pk,
                      "trimp": trimp, "steps": steps}
    return out


def parse_sleep(text: str) -> dict[int, dict]:
    """Parse the 30-row daily sleep table.

    Columns: bedtime, wake, sleep_hours, awake_min, deep_min, REM_min, score.
    The first two are time strings (HH:MM); we keep them as strings but
    also expose sleep_hours and score numerically.
    """
    out: dict[int, dict] = {}
    for line in text.split("\n"):
        d = _parse_day_idx(line)
        if d is None or not (1 <= d <= 30):
            continue
        tail = _row_tail(line)
        # Find the floats — we expect 5 numerics: hours, awake, deep, REM, score
        nums = _parse_floats(tail)
        if len(nums) >= 5:
            sleep_hours, awake, deep, rem, score = nums[:5]
            out[d] = {"sleep_h": sleep_hours, "awake_min": awake,
                      "deep_min": deep, "rem_min": rem, "sleep_score": score}
    return out


def parse_case(rec: dict) -> dict:
    """Return {1..30: merged daily record}."""
    hm = parse_health_metrics(rec.get("health_metrics_input", ""))
    tl = parse_training_load(rec.get("training_load_input", ""))
    sl = parse_sleep(rec.get("sleep_input", ""))
    days: dict[int, dict] = {}
    for d in range(1, 31):
        merged = {}
        if d in hm:
            merged.update(hm[d])
        if d in tl:
            merged.update(tl[d])
        if d in sl:
            merged.update(sl[d])
        if merged:
            days[d] = merged
    return days


# ─── Baselines ──────────────────────────────────────────────────────────


def predict_persistence(window: list[dict], target_field: str) -> Optional[float]:
    """Predict tomorrow = today's value."""
    for d in reversed(window):
        if target_field in d:
            return float(d[target_field])
    return None


def predict_rolling_mean(window: list[dict], target_field: str, k: int = 7) -> Optional[float]:
    """Predict tomorrow = mean of last k available values."""
    vals = [d[target_field] for d in window[-k:] if target_field in d]
    if not vals:
        return None
    return float(np.mean(vals))


# ─── LLM zero-shot ──────────────────────────────────────────────────────


def format_window_for_llm(window: list[dict], start_day: int) -> str:
    lines = []
    for i, d in enumerate(window):
        day_num = start_day + i
        rhr = d.get("rhr", "?")
        hrv = d.get("hrv", "?")
        sl_h = d.get("sleep_h", "?")
        score = d.get("sleep_score", "?")
        fb = d.get("fat_burn", "?")
        ca = d.get("cardio", "?")
        pk = d.get("peak", "?")
        trimp = d.get("trimp", "?")
        lines.append(
            f"Day {day_num}: RHR={rhr}, HRV={hrv}ms, sleep={sl_h}h (score={score}), "
            f"exercise: fat_burn={fb}min, cardio={ca}min, peak={pk}min, TRIMP={trimp}"
        )
    return "\n".join(lines)


def build_llm_prompt(window: list[dict], start_day: int, target_day: int) -> str:
    return (
        "Below is a person's recent daily health and activity record from a wearable device.\n"
        "RHR = resting heart rate (bpm). HRV = overnight heart-rate variability RMSSD (ms).\n"
        "Sleep score is 0-100. Exercise zone times are how long the user spent in each HR zone.\n\n"
        f"{format_window_for_llm(window, start_day)}\n\n"
        f"Predict the user's morning RHR and HRV on Day {target_day} "
        "(measured before any activity that day).\n"
        'Reply ONLY a JSON object with the exact form: {"rhr": <number>, "hrv": <number>}\n'
        "Do not include any other text, explanation, or markdown."
    )


_LLM_STATE: dict = {}


def init_llm(model_name: str, dtype: str = "bfloat16"):
    """Lazy-load Qwen3-4B once."""
    if "tokenizer" in _LLM_STATE:
        return _LLM_STATE
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[llm] Loading {model_name} (dtype={dtype}) ...", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        dtype=getattr(torch, dtype),
        device_map="auto",
    )
    model.eval()
    print(f"[llm] Loaded in {time.time()-t0:.1f}s", flush=True)
    _LLM_STATE["tokenizer"] = tok
    _LLM_STATE["model"] = model
    return _LLM_STATE


def predict_llm(window: list[dict], start_day: int, target_day: int) -> tuple[Optional[float], Optional[float], str]:
    """Return (rhr_pred, hrv_pred, raw_text)."""
    import torch

    state = _LLM_STATE
    tok = state["tokenizer"]
    model = state["model"]

    prompt = build_llm_prompt(window, start_day, target_day)
    # Use Qwen3 chat template, with thinking disabled for direct numeric answer
    msgs = [{"role": "user", "content": prompt}]
    try:
        chat_text = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        chat_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    inputs = tok(chat_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tok.pad_token_id,
        )
    gen = out[0][inputs["input_ids"].shape[1]:]
    raw = tok.decode(gen, skip_special_tokens=True).strip()

    # Extract JSON
    rhr, hrv = None, None
    m = re.search(r"\{.*?\}", raw, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            rhr = float(obj.get("rhr")) if obj.get("rhr") is not None else None
            hrv = float(obj.get("hrv")) if obj.get("hrv") is not None else None
        except Exception:
            pass
    return rhr, hrv, raw


# ─── Pipeline ───────────────────────────────────────────────────────────


def _is_valid_rhr(v: float) -> bool:
    return 30.0 <= v <= 110.0


def _is_valid_hrv(v: float) -> bool:
    # HRV RMSSD in healthy adults: ~10-100 ms. Drop 0 (missing) and >200 (artifact).
    return 5.0 <= v <= 200.0


def build_pairs(days: dict[int, dict], window_size: int) -> list[dict]:
    """Sliding window: input = days [k-w+1 .. k], target = day k+1.

    Drop pairs whose target RHR or HRV is missing or out of physiological range.
    """
    pairs = []
    available = sorted(days.keys())
    for k in range(min(available) + window_size - 1, max(available)):
        if (k + 1) not in days:
            continue
        win_days = list(range(k - window_size + 1, k + 1))
        if not all(d in days for d in win_days):
            continue
        target = days[k + 1]
        if "rhr" not in target or "hrv" not in target:
            continue
        if not _is_valid_rhr(target["rhr"]) or not _is_valid_hrv(target["hrv"]):
            continue
        pairs.append({
            "start_day": win_days[0],
            "end_day": k,
            "target_day": k + 1,
            "window": [days[d] for d in win_days],
            "target_rhr": target["rhr"],
            "target_hrv": target["hrv"],
        })
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_cases", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--window", type=int, default=14)
    ap.add_argument("--use_llm", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max_pairs_for_llm", type=int, default=0,
                    help="Cap total LLM calls (0 = no cap). Useful when GPU limited.")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_tag = f"n{args.n_cases}_w{args.window}_seed{args.seed}"

    # ── 1. Load + sample ──
    ensure_data()
    with open(DATA_FILE, encoding="utf-8") as f:
        all_recs = [json.loads(l) for l in f if l.strip()]
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(all_recs), size=args.n_cases, replace=False)
    recs = [all_recs[i] for i in idx]
    print(f"Loaded {len(all_recs)} cases, sampled {len(recs)} (seed={args.seed})")

    # ── 2. Parse + build pairs ──
    all_pairs = []
    for rec in recs:
        days = parse_case(rec)
        pairs = build_pairs(days, args.window)
        for p in pairs:
            p["case_id"] = rec.get("case_study_id")
            p["user_id"] = rec.get("user_id")
        all_pairs.extend(pairs)
    print(f"Built {len(all_pairs)} (window, target) pairs from {len(recs)} cases")
    print(f"  per case: min={min(len(build_pairs(parse_case(r), args.window)) for r in recs)}, "
          f"max={max(len(build_pairs(parse_case(r), args.window)) for r in recs)}, "
          f"mean={len(all_pairs)/len(recs):.1f}")

    # ── 3. Quick variance summary ──
    rhr_targets = np.array([p["target_rhr"] for p in all_pairs])
    hrv_targets = np.array([p["target_hrv"] for p in all_pairs])
    print(f"\nTargets — RHR: mean={rhr_targets.mean():.1f}, std={rhr_targets.std():.2f}, "
          f"range={rhr_targets.min():.0f}-{rhr_targets.max():.0f}")
    print(f"Targets — HRV: mean={hrv_targets.mean():.1f}, std={hrv_targets.std():.2f}, "
          f"range={hrv_targets.min():.1f}-{hrv_targets.max():.1f}")

    # ── 4. Baselines ──
    rows = []
    for p in all_pairs:
        row = {"case_id": p["case_id"], "target_day": p["target_day"],
               "rhr_true": p["target_rhr"], "hrv_true": p["target_hrv"]}
        for field in ("rhr", "hrv"):
            row[f"{field}_persist"] = predict_persistence(p["window"], field)
            row[f"{field}_mean7"] = predict_rolling_mean(p["window"], field, 7)
        rows.append(row)

    def mae(rows, pred_key, true_key):
        diffs = [abs(r[pred_key] - r[true_key]) for r in rows
                 if r.get(pred_key) is not None]
        return float(np.mean(diffs)) if diffs else float("nan"), len(diffs)

    print("\n=== Baseline MAE ===")
    print(f"{'predictor':>20s}  {'RHR MAE':>10s}  {'HRV MAE':>10s}  {'n':>5s}")
    for label, key in [("Persistence", "persist"), ("7-day mean", "mean7")]:
        rhr_m, n_r = mae(rows, f"rhr_{key}", "rhr_true")
        hrv_m, n_h = mae(rows, f"hrv_{key}", "hrv_true")
        print(f"{label:>20s}  {rhr_m:>10.3f}  {hrv_m:>10.3f}  {n_r:>5d}")

    # ── 5. LLM (optional) ──
    if args.use_llm:
        init_llm(args.model, dtype=args.dtype)

        pairs_for_llm = all_pairs
        if args.max_pairs_for_llm > 0 and len(pairs_for_llm) > args.max_pairs_for_llm:
            pairs_for_llm = pairs_for_llm[: args.max_pairs_for_llm]
            print(f"\n[llm] Capped to first {len(pairs_for_llm)} pairs")
        else:
            print(f"\n[llm] Running {len(pairs_for_llm)} pairs through Qwen3-4B")

        t0 = time.time()
        for j, p in enumerate(pairs_for_llm):
            rhr_p, hrv_p, raw = predict_llm(p["window"], p["start_day"], p["target_day"])
            rows[j]["rhr_llm"] = rhr_p
            rows[j]["hrv_llm"] = hrv_p
            rows[j]["llm_raw"] = raw
            if (j + 1) % 10 == 0 or j + 1 == len(pairs_for_llm):
                elapsed = time.time() - t0
                eta = elapsed / (j + 1) * (len(pairs_for_llm) - j - 1)
                print(f"  [llm] {j+1}/{len(pairs_for_llm)}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s",
                      flush=True)

        rhr_m, n_r = mae(rows[: len(pairs_for_llm)], "rhr_llm", "rhr_true")
        hrv_m, n_h = mae(rows[: len(pairs_for_llm)], "hrv_llm", "hrv_true")
        print(f"\n=== Qwen3-4B zero-shot MAE ===")
        print(f"{'Qwen3-4B':>20s}  {rhr_m:>10.3f}  {hrv_m:>10.3f}  RHR_n={n_r}, HRV_n={n_h}")

    # ── 6. Save ──
    out_path = OUT_DIR / f"results_{out_tag}{'_llm' if args.use_llm else ''}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": vars(args),
            "n_cases": len(recs),
            "n_pairs": len(all_pairs),
            "rhr_target_stats": {"mean": float(rhr_targets.mean()),
                                  "std": float(rhr_targets.std())},
            "hrv_target_stats": {"mean": float(hrv_targets.mean()),
                                  "std": float(hrv_targets.std())},
            "rows": rows,
        }, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
