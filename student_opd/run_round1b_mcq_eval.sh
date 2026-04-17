#!/bin/bash
# Phase 2b Round 1b MCQ-PPL eval.
#
# Auto-discovers every ckpt-step-N/ under each persona's v2 output dir
# AND the final (root) state, then evaluates each that doesn't already
# have its JSON cached. No base_recent condition — R1b uses demo-only
# student input, so the only new condition per persona is student_dual_v2.
#
# base_demo and teacher_k3 are cached from Phase 2 / Round 1 runs and
# get reused automatically.
#
# Usage:
#   bash dynamic_usersim/student_opd/run_round1b_mcq_eval.sh
#   # override any subset:
#   TAG=dual_v2 OUT_ROOT=$SCRATCHDIR/P-OPSD/student_lora \
#     bash dynamic_usersim/student_opd/run_round1b_mcq_eval.sh
#
# Force-rerun: delete the specific mcqppl_*.json first.

set -e

R3="${R3:-$SCRATCHDIR/P-OPSD/teacher_sft_128k_k3/final}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
PERSONAS=(${PERSONAS:-0 4 12 14})
TAG="${TAG:-dual_v2}"
OUT_ROOT="${OUT_ROOT:-$SCRATCHDIR/P-OPSD/student_lora}"
EVAL_JSON_DIR=dynamic_usersim/outputs   # small files stay in $HOME
LOG_DIR=logs/round1b_eval_mcq
mkdir -p "$LOG_DIR" "$EVAL_JSON_DIR"

echo "=========================================================="
echo "Phase 2b Round 1b MCQ-PPL eval (auto-discover all ckpts)"
echo "  tag        : $TAG"
echo "  out_root   : $OUT_ROOT"
echo "  personas   : ${PERSONAS[*]}"
echo "  R3 teacher : $R3"
echo "=========================================================="

# run_cond cond_name base_path lora_path ctx_mode pid outjson
run_cond() {
    local name="$1" base="$2" lora="$3" ctx="$4" pid="$5" outjson="$6"
    if [[ -f "$outjson" ]]; then
        echo "  [cached]  pid=$pid  $name"
        return
    fi
    local lora_arg=()
    if [[ -n "$lora" ]]; then
        lora_arg=(--lora-path "$lora" --lora-mode dual)
    fi
    local log_file="$LOG_DIR/pid${pid}_${name}.log"
    echo "  [running] pid=$pid  $name  (ctx=$ctx, lora=${lora:-none})"
    torchrun --standalone --nnodes=1 --nproc_per_node=4 \
        dynamic_usersim/student_opd/eval_opd.py \
        --persona-id "$pid" \
        --base-model "$base" "${lora_arg[@]}" \
        --context-mode "$ctx" \
        --out-json "$outjson" \
        > "$log_file" 2>&1
    local acc=$(python3 -c "import json; print(f\"{json.load(open('$outjson'))['accuracy']:.3f}\")" 2>/dev/null || echo "?")
    echo "            -> acc=$acc"
}

for pid in "${PERSONAS[@]}"; do
    echo ""
    echo "===== persona $pid ====="

    # Baselines (typically cached from Phase 2 / Round 1)
    run_cond base_demo \
        "$BASE_MODEL" "" demo-only "$pid" \
        "$EVAL_JSON_DIR/mcqppl_128k_pid${pid}_base_demo.json"

    run_cond teacher_k3 \
        "$R3" "" last-n "$pid" \
        "$EVAL_JSON_DIR/mcqppl_128k_pid${pid}_teacher_k3.json"

    # R1b student: auto-discover every ckpt-step-N + final
    persona_dir="$OUT_ROOT/lora_pid${pid}_${TAG}_ep1_r3teacher"
    if [[ ! -d "$persona_dir" ]]; then
        echo "  [SKIP] persona_dir missing: $persona_dir"
        continue
    fi

    # Intermediate ckpts (sorted numerically)
    mapfile -t ckpt_paths < <(ls -d "$persona_dir"/ckpt-step-* 2>/dev/null | \
        sort -t- -k3 -n)

    for ckpt in "${ckpt_paths[@]}"; do
        step=$(basename "$ckpt" | sed 's/ckpt-step-//')
        if [[ ! -d "$ckpt/slow" ]] || [[ ! -d "$ckpt/fast" ]]; then
            echo "  [skip step=$step] missing slow/+fast/ in $ckpt"
            continue
        fi
        run_cond "student_${TAG}_demo_step${step}" \
            "$BASE_MODEL" "$ckpt" demo-only "$pid" \
            "$EVAL_JSON_DIR/mcqppl_128k_pid${pid}_student_${TAG}_demo_step${step}.json"
    done

    # Final (root of persona_dir) — training end-of-epoch state
    if [[ -d "$persona_dir/slow" ]] && [[ -d "$persona_dir/fast" ]]; then
        run_cond "student_${TAG}_demo_final" \
            "$BASE_MODEL" "$persona_dir" demo-only "$pid" \
            "$EVAL_JSON_DIR/mcqppl_128k_pid${pid}_student_${TAG}_demo_final.json"
    else
        echo "  [skip final] missing slow/+fast/ in $persona_dir"
    fi
done

# --------- Aggregate summary ---------
echo ""
echo "=========================================================="
echo "SUMMARY — Phase 2b Round 1b MCQ-PPL (128k)"
echo "  tag = $TAG"
echo "=========================================================="

python3 - "$EVAL_JSON_DIR" "$TAG" "${PERSONAS[@]}" <<'PYEOF'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
tag = sys.argv[2]
personas = [int(x) for x in sys.argv[3:]]

def load(name, pid):
    p = out / f"mcqppl_128k_pid{pid}_{name}.json"
    return json.loads(p.read_text())["accuracy"] if p.exists() else None

# Discover all steps present across all personas
steps = set()
for pid in personas:
    for p in out.glob(f"mcqppl_128k_pid{pid}_student_{tag}_demo_step*.json"):
        s = p.stem.split("_step")[-1]
        if s.isdigit():
            steps.add(int(s))
steps_sorted = sorted(steps)

cols = [("base_demo", "base_demo"), ("teacher_k3", "teacher_k3")]
for s in steps_sorted:
    cols.append((f"step{s}", f"student_{tag}_demo_step{s}"))
cols.append(("final", f"student_{tag}_demo_final"))

col_w = max(10, len(f"student_{tag}_demo_step{max(steps, default=0)}"))
print(f"{'pid':<5}" + " ".join(f"{c[0]:>{col_w}}" for c in cols))
print("-" * (5 + (col_w + 1) * len(cols)))

agg = {k: [] for k, _ in cols}
for pid in personas:
    row = [f"{pid:<5}"]
    for _, name in cols:
        a = load(name, pid)
        if a is None:
            row.append(f"{'-':>{col_w}}")
        else:
            agg[name if (name.startswith('student') or name in ('base_demo','teacher_k3')) else name].append(a)
            # Use col label as agg key consistently
            k = [c[0] for c in cols if c[1] == name][0]
            agg.setdefault(k, []).append(a) if k not in agg else None
            row.append(f"{a:>{col_w}.3f}")
    print(" ".join(row))

# Re-aggregate cleanly
agg = {c[0]: [] for c in cols}
for pid in personas:
    for label, name in cols:
        a = load(name, pid)
        if a is not None:
            agg[label].append(a)

print("-" * (5 + (col_w + 1) * len(cols)))
row = [f"{'AVG':<5}"]
for label, _ in cols:
    vals = agg[label]
    row.append(f"{sum(vals)/len(vals):>{col_w}.3f}" if vals else f"{'-':>{col_w}}")
print(" ".join(row))

# Closure curve
print()
print(f"--- closure (vs teacher_k3 − base_demo) per step ---")
if agg["base_demo"] and agg["teacher_k3"]:
    av = lambda k: sum(agg[k])/len(agg[k])
    b, t = av("base_demo"), av("teacher_k3")
    gap = t - b
    for label, _ in cols:
        if label in ("base_demo", "teacher_k3"):
            continue
        if agg[label]:
            closure = 100 * (av(label) - b) / gap
            print(f"  {label:<10}: {av(label):.3f}  ({closure:+6.1f}% closure)")

# Per-persona best vs final
print()
print(f"--- per-persona: best ckpt vs final ---")
for pid in personas:
    b = load("base_demo", pid)
    tch = load("teacher_k3", pid)
    if b is None or tch is None or tch == b:
        continue
    scores = {}
    for label, name in cols:
        if label in ("base_demo", "teacher_k3"):
            continue
        a = load(name, pid)
        if a is not None:
            scores[label] = a
    if not scores:
        continue
    best_label, best_acc = max(scores.items(), key=lambda kv: kv[1])
    final_acc = scores.get("final", float("nan"))
    best_cl = 100*(best_acc-b)/(tch-b)
    final_cl = 100*(final_acc-b)/(tch-b) if final_acc==final_acc else float("nan")
    print(f"  pid={pid}  base={b:.3f}  teacher={tch:.3f}  "
          f"best={best_acc:.3f}@{best_label} ({best_cl:+.1f}%)  "
          f"final={final_acc:.3f} ({final_cl:+.1f}%)")
PYEOF

echo ""
echo "Done. JSONs under $EVAL_JSON_DIR/mcqppl_128k_*"
echo "     Logs under $LOG_DIR/"
