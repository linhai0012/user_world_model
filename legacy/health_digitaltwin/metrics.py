"""
Evaluation metrics for the Digital Twin model.

V3: Computes text quality + state prediction accuracy.
Also retains V1/V2 HR metrics for backward compatibility.
"""
from __future__ import annotations

import re
import numpy as np
from typing import Any

from config import (
    TOKEN_TS_START, TOKEN_TS_END,
    TOKEN_TEXT_START, TOKEN_TEXT_END,
    TOKEN_STATE_START, TOKEN_STATE_END,
    HR_TOKEN_PREFIX, HR_TOKEN_SUFFIX,
    WELLNESS_FIELDS,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Parsing
# ═══════════════════════════════════════════════════════════════════════════════

def parse_generated_output(text: str) -> dict[str, Any]:
    """
    Parse generated output into text feedback + state predictions.

    V3 format:
      <text_start> ... <text_end> <state_start> <fatigue_3> <mood_4> ... <state_end>

    Also handles V1/V2 HR format for backward compatibility.
    """
    result = {
        "user_feedback": "",
        "predicted_state": {},   # V3: {field: value}
        "hr_sequence": [],       # V1/V2: backward compat
        "parse_ok": False,
    }

    # Extract text feedback
    text_match = re.search(
        rf"{re.escape(TOKEN_TEXT_START)}\s*(.*?)\s*{re.escape(TOKEN_TEXT_END)}",
        text, re.DOTALL
    )
    if text_match:
        result["user_feedback"] = text_match.group(1).strip()

    # Try V3: extract state tokens
    state_match = re.search(
        rf"{re.escape(TOKEN_STATE_START)}\s*(.*?)\s*{re.escape(TOKEN_STATE_END)}",
        text, re.DOTALL
    )
    if state_match:
        state_block = state_match.group(1)
        for field in WELLNESS_FIELDS:
            m = re.search(rf"<{field}_(\d+)>", state_block)
            if m:
                result["predicted_state"][field] = int(m.group(1))
        result["parse_ok"] = bool(result["user_feedback"]) and len(result["predicted_state"]) > 0
    else:
        # Fallback: V1/V2 HR parsing
        ts_match = re.search(
            rf"{re.escape(TOKEN_TS_START)}\s*(.*?)\s*{re.escape(TOKEN_TS_END)}",
            text, re.DOTALL
        )
        if ts_match:
            hr_tokens = re.findall(
                rf"{re.escape(HR_TOKEN_PREFIX)}(\d+){re.escape(HR_TOKEN_SUFFIX)}",
                ts_match.group(1),
            )
            result["hr_sequence"] = [int(v) for v in hr_tokens]
            result["parse_ok"] = bool(result["user_feedback"]) and len(result["hr_sequence"]) > 0

    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  V3: State prediction metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_state_metrics(
    pred_state: dict[str, int],
    true_state: dict[str, int],
    day_n_state: dict[str, int] | None = None,
) -> dict[str, Any]:
    """
    Compute per-field state prediction metrics.

    If day_n_state is provided, also computes persistence baseline comparison.
    """
    result = {}
    for field in WELLNESS_FIELDS:
        pred_val = pred_state.get(field)
        true_val = true_state.get(field)

        if pred_val is None or true_val is None:
            result[f"state_{field}_error"] = float("nan")
            result[f"state_{field}_exact"] = False
            continue

        error = abs(pred_val - true_val)
        result[f"state_{field}_error"] = error
        result[f"state_{field}_exact"] = pred_val == true_val

        # Persistence baseline: how well does "predict tomorrow = today" do?
        if day_n_state and isinstance(day_n_state.get(field), (int, float)):
            persist_error = abs(int(day_n_state[field]) - true_val)
            result[f"state_{field}_persist_error"] = persist_error

    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  Text metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_text_metrics(pred_text: str, true_text: str) -> dict[str, float]:
    """Basic text overlap metrics."""
    if not pred_text or not true_text:
        return {"text_word_overlap": 0.0, "text_len_ratio": 0.0}

    pred_words = set(pred_text.lower().split())
    true_words = set(true_text.lower().split())

    if not true_words:
        return {"text_word_overlap": 0.0, "text_len_ratio": 0.0}

    overlap = len(pred_words & true_words) / len(true_words)
    len_ratio = len(pred_text.split()) / max(len(true_text.split()), 1)

    return {
        "text_word_overlap": round(overlap, 4),
        "text_len_ratio": round(len_ratio, 3),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Text-state consistency
# ═══════════════════════════════════════════════════════════════════════════════

def compute_consistency_v3(text: str, pred_state: dict[str, int]) -> dict[str, Any]:
    """
    Rule-based cross-modal consistency check for V3.
    Checks if generated text is consistent with predicted state values.
    """
    if not text or not pred_state:
        return {"consistency_score": float("nan"), "checks": {}}

    text_lower = text.lower()
    checks = {}
    n_pass = 0
    n_total = 0

    # Check 1: fatigue language ↔ fatigue score
    fatigue_words = ["tired", "exhausted", "fatigued", "drained", "worn out", "wiped"]
    energy_words = ["energetic", "energized", "refreshed", "fresh", "great energy"]
    fatigue_val = pred_state.get("fatigue")
    if fatigue_val is not None:
        mentions_tired = any(w in text_lower for w in fatigue_words)
        mentions_energy = any(w in text_lower for w in energy_words)
        if mentions_tired:
            n_total += 1
            ok = fatigue_val <= 2  # 1-2 = fatigued
            checks["fatigue_text_match"] = ok
            if ok: n_pass += 1
        elif mentions_energy:
            n_total += 1
            ok = fatigue_val >= 4  # 4-5 = energetic
            checks["energy_text_match"] = ok
            if ok: n_pass += 1

    # Check 2: soreness language ↔ soreness score
    sore_words = ["sore", "aching", "stiff", "tight", "tender"]
    soreness_val = pred_state.get("soreness")
    if soreness_val is not None:
        mentions_sore = any(w in text_lower for w in sore_words)
        if mentions_sore:
            n_total += 1
            ok = soreness_val <= 2  # 1-2 = very sore
            checks["soreness_text_match"] = ok
            if ok: n_pass += 1

    # Check 3: mood language ↔ mood score
    good_mood_words = ["happy", "great mood", "feeling good", "positive", "upbeat"]
    bad_mood_words = ["down", "low mood", "frustrated", "irritable"]
    mood_val = pred_state.get("mood")
    if mood_val is not None:
        if any(w in text_lower for w in good_mood_words):
            n_total += 1
            ok = mood_val >= 4
            checks["good_mood_match"] = ok
            if ok: n_pass += 1
        elif any(w in text_lower for w in bad_mood_words):
            n_total += 1
            ok = mood_val <= 2
            checks["bad_mood_match"] = ok
            if ok: n_pass += 1

    score = n_pass / n_total if n_total > 0 else float("nan")
    return {
        "consistency_score": round(score, 4),
        "checks_passed": n_pass,
        "checks_total": n_total,
        "checks": checks,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Batch evaluation (V3)
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_batch(
    predictions: list[str],
    references: list[dict[str, Any]],
) -> dict[str, float]:
    """
    Evaluate a batch of generated outputs against ground truth.
    Supports both V3 (state) and V1/V2 (HR) formats.
    """
    all_state_metrics = []
    all_text_metrics = []
    all_consistency = []
    parse_success = 0
    n = len(predictions)

    for pred_str, ref in zip(predictions, references):
        parsed = parse_generated_output(pred_str)
        if parsed["parse_ok"]:
            parse_success += 1

        # Text metrics
        ref_parsed = parse_generated_output(ref["output_text"])
        text_m = compute_text_metrics(parsed["user_feedback"], ref_parsed["user_feedback"])
        all_text_metrics.append(text_m)

        # V3: State metrics
        true_state = ref.get("wellness_day_n1", {})
        day_n_state = ref.get("wellness_day_n", {})
        if parsed["predicted_state"] and true_state:
            state_m = compute_state_metrics(parsed["predicted_state"], true_state, day_n_state)
            all_state_metrics.append(state_m)
            cons = compute_consistency_v3(parsed["user_feedback"], parsed["predicted_state"])
            all_consistency.append(cons)

    result = {"parse_success_rate": parse_success / n if n else 0}

    # Text metrics
    for key in ["text_word_overlap"]:
        vals = [m[key] for m in all_text_metrics]
        result[key] = round(np.mean(vals), 4) if vals else 0

    # State metrics (V3)
    if all_state_metrics:
        for field in WELLNESS_FIELDS:
            err_key = f"state_{field}_error"
            exact_key = f"state_{field}_exact"
            persist_key = f"state_{field}_persist_error"

            errors = [m[err_key] for m in all_state_metrics if not np.isnan(m.get(err_key, float("nan")))]
            exacts = [m[exact_key] for m in all_state_metrics if isinstance(m.get(exact_key), bool)]
            persists = [m[persist_key] for m in all_state_metrics if not np.isnan(m.get(persist_key, float("nan")))]

            if errors:
                result[f"{field}_mae"] = round(np.mean(errors), 4)
                result[f"{field}_exact_match"] = round(np.mean(exacts), 4) if exacts else 0
            if persists:
                result[f"{field}_persist_mae"] = round(np.mean(persists), 4)

    # Consistency
    cons_vals = [c["consistency_score"] for c in all_consistency
                 if not np.isnan(c["consistency_score"])]
    result["consistency_score"] = round(np.mean(cons_vals), 4) if cons_vals else float("nan")

    return result
