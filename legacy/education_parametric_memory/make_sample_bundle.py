"""Generate a tiny *sample* parametric artifact bundle (de-risk, NO GPU).

Purpose: lock the cross-team contract before the real GPU harness exists. This writes the
5-file bundle (docs/backend/parametric-bundle-schema.md) for 2-3 personas over the seeded
`biology_gcse` curriculum, with **plausible-but-synthetic** numbers. The live app's
`offline_replay` provider replays it; `import_bundle.py` loads it into Mongo. Once the real
GPU rig produces a real bundle in this exact shape, it drops into the same seam.

What is real here vs fake:
  - REAL: question_ids / skill_ids (pulled from app.content.taxonomy so they match the seeded
    DB exactly), the file layout, every schema field name.
  - FAKE: the p_correct / logprob / nll / brier / cost numbers (deterministic synthetic,
    seeded), and the personas' hidden mastery (we author it — it is the oracle theta).

Run (needs the backend importable, e.g. the repo venv):
    backend/.venv/bin/python parametric_offline/make_sample_bundle.py
    # -> parametric_offline/sample_bundle/{manifest,adapters,predictions,curves}.jsonl|json + cost.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from app.content import taxonomy
from personas import det_seed

# --------------------------------------------------------------------- config
SUBJECT = "biology_gcse"
SNAPSHOTS = [0, 5, 10, 20]
BASE_PRIOR = 0.62          # Qwen's correct-answer leak floor (the A-null story)
PERSONA_SET_ID = "cue-personas-v1-sample"
BUNDLE_ID = "cue-param-sample-2026-06-16a"

# Hidden ground-truth theta per persona: per-skill true mastery (slug -> m).
# Unlisted skills default to 0.5. This is the oracle we (the harness) author.
PERSONAS: dict[str, dict[str, float]] = {
    # a struggling-on-transport learner who holds the osmosis direction misconception
    "persona_07": {"cell_structure": 0.60, "diffusion": 0.30, "osmosis": 0.20,
                   "active_transport": 0.25, "enzymes": 0.50, "photosynthesis": 0.45},
    # a strong learner across the board
    "persona_12": {"cell_structure": 0.85, "diffusion": 0.80, "osmosis": 0.82,
                   "active_transport": 0.78, "enzymes": 0.80, "photosynthesis": 0.88},
    # a mixed learner
    "persona_19": {"cell_structure": 0.55, "diffusion": 0.65, "osmosis": 0.50,
                   "active_transport": 0.45, "enzymes": 0.70, "photosynthesis": 0.60},
}


# --------------------------------------------------------------------- helpers
def _skill_slug(skill_id: str) -> str:
    return skill_id.split("__")[-1]


def _true_mastery(persona: str, skill_id: str) -> float:
    return PERSONAS[persona].get(_skill_slug(skill_id), 0.5)


def _converge(snapshot: int) -> float:
    """How far the adapter's belief has moved from the base prior toward truth, by `snapshot`."""
    return snapshot / (snapshot + 6.0)  # 0 at s=0 -> ~0.77 at s=20


def _p_hat(true_m: float, snapshot: int) -> float:
    """Per-user adapter's predicted P(correct): starts at the base prior, converges to truth."""
    c = _converge(snapshot)
    return max(0.02, min(0.98, BASE_PRIOR + (true_m - BASE_PRIOR) * c))


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def _option_logprobs(q: dict, p_correct: float, rng: random.Random) -> dict[str, float]:
    """Synthesize per-option logprobs whose softmax = the persona's choice distribution.

    Correct option gets p_correct; the bulk of the error mass lands on ONE distractor (the
    persona's favoured misconception), the rest is spread. softmax(log p) == p by construction.
    """
    opts = q["options"]
    correct_id = next((o["id"] for o in opts if o["is_correct"]), opts[0]["id"])
    wrong_ids = [o["id"] for o in opts if not o["is_correct"]]
    probs: dict[str, float] = {correct_id: p_correct}
    if wrong_ids:
        # deterministic "favoured distractor" = first wrong option; 65% of error mass there.
        err = 1.0 - p_correct
        fav = wrong_ids[0]
        probs[fav] = err * 0.65
        rest = wrong_ids[1:]
        for w in rest:
            probs[w] = err * 0.35 / len(rest) if rest else 0.0
    # normalise (guard) and convert to logprobs
    z = sum(probs.values()) or 1.0
    return {k: round(_logit_safe_logprob(v / z), 4) for k, v in probs.items()}


def _logit_safe_logprob(p: float) -> float:
    return math.log(min(max(p, 1e-6), 1.0))


def _round_sequence(questions: list[dict], rng: random.Random) -> list[str]:
    """A per-persona ordering over the real seeded (non-probe) question_ids."""
    pool = [q["_id"] for q in questions if not q["is_probe"]]
    order = pool[:]
    rng.shuffle(order)
    return order


# --------------------------------------------------------------------- build
def build_bundle(created_offline: str) -> dict[str, object]:
    built = taxonomy.build_all()
    questions = [q for q in built["questions"] if q["subject_id"] == SUBJECT]
    eval_qs = questions  # predict on every seeded item (probes included, flagged is_probe)

    manifest = {
        "bundle_id": BUNDLE_ID,
        "base_model": "Qwen3-4B-Instruct-2507",
        "subject_ids": [SUBJECT],
        "created_offline": created_offline,
        "lora_recipe": {"slow_mlp_r": 32, "fast_attn_r": 16, "objective": "user_turn_response"},
        "raw_completion": True,
        "calibration": {"method": "isotonic", "fit_on": "held_out_calib_split"},
        "training": {"objective_detail": "direct SFT (no teacher) on observed answers",
                     "note": "SAMPLE bundle — synthetic numbers; real bundle from the GPU rig"},
        "persona_set": PERSONA_SET_ID,
        "round_sequence": {"seed": 42, "personas": {}},
        "snapshots": SNAPSHOTS,
    }

    predictions: list[dict] = []
    curves: list[dict] = []
    adapters: list[dict] = []

    # --- per-user (A1) personas ---
    for persona, _theta in PERSONAS.items():
        rng = random.Random(det_seed(persona, "seq"))
        manifest["round_sequence"]["personas"][persona] = {
            "subject_id": SUBJECT,
            "rounds": _round_sequence(questions, rng),
        }
        for snap in SNAPSHOTS:
            for q in eval_qs:
                _emit_prediction(predictions, persona, "per_user", q, snap)
            _emit_curve(curves, persona, "per_user", snap, eval_qs)
            _emit_adapter(adapters, persona, "per_user", snap)

    # --- A0 shared-LoRA control (pooled): predicts toward the global mean, not the persona ---
    for snap in SNAPSHOTS:
        for q in eval_qs:
            _emit_prediction(predictions, "__shared__", "shared", q, snap)
        _emit_curve(curves, "__shared__", "shared", snap, eval_qs)
        _emit_adapter(adapters, "__shared__", "shared", snap)

    # --- A-null base / no-input control: flat leak floor regardless of learner ---
    for snap in SNAPSHOTS:
        for q in eval_qs:
            _emit_prediction(predictions, "__base__", "base", q, snap)
        _emit_curve(curves, "__base__", "base", snap, eval_qs)
        _emit_adapter(adapters, "__base__", "base", snap)

    cost = {"train_s_per_snapshot": 95, "infer_ms_per_item": 120,
            "gpu": "1xA100-80GB", "total_train_s": 95 * len(SNAPSHOTS) * len(PERSONAS),
            "note": "SAMPLE — placeholder timings; real numbers measured on the rig"}

    return {"manifest": manifest, "predictions": predictions, "curves": curves,
            "adapters": adapters, "cost": cost}


def _global_mean_mastery(skill_id: str) -> float:
    vals = [PERSONAS[p].get(_skill_slug(skill_id), 0.5) for p in PERSONAS]
    return sum(vals) / len(vals)


def _emit_prediction(out: list, learner: str, scope: str, q: dict, snap: int) -> None:
    rng = random.Random(det_seed(learner, q["_id"], snap))
    if scope == "per_user":
        p_hat = _p_hat(_true_mastery(learner, q["skill_id"]), snap)
    elif scope == "shared":
        p_hat = _p_hat(_global_mean_mastery(q["skill_id"]), snap)
    else:  # base / no-input: flat leak floor, snapshot-independent
        p_hat = BASE_PRIOR
    row = {
        "learner_id": learner,
        "subject_id": SUBJECT,
        "snapshot": snap,
        "question_id": q["_id"],
        "format": q["format"],
        "p_correct_raw": round(_logit(p_hat) + rng.uniform(-0.05, 0.05), 4),  # NOT a probability
        "p_correct": round(p_hat, 4),                                          # CALIBRATED
        "scope": scope,
    }
    if q["format"] == "mcq":
        row["option_logprobs"] = _option_logprobs(q, p_hat, rng)
    out.append(row)


def _emit_curve(out: list, learner: str, scope: str, snap: int, eval_qs: list[dict]) -> None:
    # Synthesize NLL/Brier that improve with snapshot for per_user, flatter for the controls.
    if scope == "per_user":
        nll = 0.95 - 0.55 * _converge(snap)
        brier = 0.30 - 0.18 * _converge(snap)
    elif scope == "shared":
        nll = 0.85 - 0.20 * _converge(snap)
        brier = 0.27 - 0.07 * _converge(snap)
    else:
        nll, brier = 0.83, 0.26  # base: flat
    out.append({"learner_id": learner, "subject_id": SUBJECT, "snapshot": snap,
                "nll": round(nll, 4), "brier": round(brier, 4),
                "mean_mastery_stab": round(0.10 * (1 - _converge(snap)), 4), "scope": scope})


def _emit_adapter(out: list, learner: str, scope: str, snap: int) -> None:
    if scope == "per_user":
        nll, brier, probe = 0.95 - 0.55 * _converge(snap), 0.30 - 0.18 * _converge(snap), 0.50 + 0.30 * _converge(snap)
    elif scope == "shared":
        nll, brier, probe = 0.85 - 0.20 * _converge(snap), 0.27 - 0.07 * _converge(snap), 0.55 + 0.10 * _converge(snap)
    else:
        nll, brier, probe = 0.83, 0.26, 0.58
    out.append({"learner_id": learner, "subject_id": SUBJECT, "scope": scope,
                "version": f"v-s{snap}", "snapshot": snap, "n_rounds_trained": snap,
                "parent_version": None,
                "eval": {"nll": round(nll, 4), "brier": round(brier, 4), "probe_acc": round(probe, 4)},
                "best_step_caveat": False})


# --------------------------------------------------------------------- io
def write_bundle(bundle: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(bundle["manifest"], indent=2, ensure_ascii=False))
    (out_dir / "cost.json").write_text(json.dumps(bundle["cost"], indent=2, ensure_ascii=False))
    for name in ("predictions", "adapters", "curves"):
        with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for row in bundle[name]:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "sample_bundle")
    ap.add_argument("--created", type=str, default="2026-06-16",
                    help="created_offline stamp (fixed string; no Date.now in the app)")
    args = ap.parse_args()
    bundle = build_bundle(args.created)
    write_bundle(bundle, args.out)
    print(f"wrote sample bundle -> {args.out}")
    print(f"  personas: {list(PERSONAS)} + __shared__ (A0) + __base__ (A-null)")
    print(f"  snapshots: {SNAPSHOTS}")
    print(f"  predictions: {len(bundle['predictions'])} rows  "
          f"curves: {len(bundle['curves'])}  adapters: {len(bundle['adapters'])}")


if __name__ == "__main__":
    main()
