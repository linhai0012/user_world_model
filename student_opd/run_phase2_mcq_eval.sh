#!/bin/bash
# Phase-2 MCQ-PPL aggregate evaluation.
#
# Runs 3 (or 5) conditions × 4 focal personas on 128k MCQs, each as a
# 4-GPU torchrun DP job, sequentially. Student checkpoint step is a
# parameter so we can compare step 200 vs 400 vs final.
#
# Caches non-student conditions (base_*, teacher_*) by output-file
# existence — so rerunning with a different --student-step only re-runs
# the student condition.
#
# Usage:
#   bash dynamic_usersim/student_opd/run_phase2_mcq_eval.sh              # default step=400, 3 conditions
#   bash dynamic_usersim/student_opd/run_phase2_mcq_eval.sh --student-step 200
#   bash dynamic_usersim/student_opd/run_phase2_mcq_eval.sh --student-step 400 --mode 5conds
#
# Force-rerun any condition: delete its output JSON first.

set -e

# --------- Defaults ---------
STUDENT_STEP=400
MODE=3conds   # or 5conds
R3_DEFAULT=${R3:-$SCRATCHDIR/P-OPSD/teacher_sft_128k_k3/final}
R3=$R3_DEFAULT
BASE_MODEL=Qwen/Qwen3-4B-Instruct-2507
PERSONAS=(0 4 12 14)

# --------- Args ---------
while [[ $# -gt 0 ]]; do
    case $1 in
        --student-step) STUDENT_STEP="$2"; shift 2 ;;
        --mode)         MODE="$2";         shift 2 ;;
        --r3)           R3="$2";           shift 2 ;;
        --base)         BASE_MODEL="$2";   shift 2 ;;
        -h|--help)
            sed -n '2,16p' "$0"
            exit 0
            ;;
        *) echo "unknown arg: $1"; exit 1 ;;
    esac
done

if [[ "$MODE" != "3conds" && "$MODE" != "5conds" ]]; then
    echo "ERROR: --mode must be 3conds or 5conds (got $MODE)"
    exit 1
fi

OUT_DIR=dynamic_usersim/outputs
LOG_DIR=logs/phase2_eval_mcq
mkdir -p "$LOG_DIR"

echo "=========================================================="
echo "Phase-2 MCQ-PPL eval"
echo "  student step  : $STUDENT_STEP"
echo "  mode          : $MODE"
echo "  R3 teacher    : $R3"
echo "  base model    : $BASE_MODEL"
echo "  personas      : ${PERSONAS[*]}"
echo "=========================================================="

# --------- Helper ---------
# Args: cond_name  base_or_teacher_path  lora_path_or_empty  context_mode  pid  out_json
run_cond() {
    local name="$1" base="$2" lora="$3" ctx="$4" pid="$5" outjson="$6"
    if [[ -f "$outjson" ]]; then
        echo "  [cached]  pid=$pid  $name  -> $outjson"
        return
    fi
    local lora_arg=""
    [[ -n "$lora" ]] && lora_arg="--lora-path $lora"
    local log_file="$LOG_DIR/pid${pid}_${name}.log"
    echo "  [running] pid=$pid  $name  (ctx=$ctx)"
    torchrun --standalone --nnodes=1 --nproc_per_node=4 \
        dynamic_usersim/student_opd/eval_opd.py \
        --persona-id "$pid" \
        --base-model "$base" $lora_arg \
        --context-mode "$ctx" \
        --out-json "$outjson" \
        > "$log_file" 2>&1
    local acc=$(python3 -c "import json; print(f\"{json.load(open('$outjson'))['accuracy']:.3f}\")" 2>/dev/null || echo "?")
    echo "            -> acc=$acc  (log: $log_file)"
}

# --------- Run loops ---------
for pid in "${PERSONAS[@]}"; do
    echo ""
    echo "===== persona $pid ====="

    run_cond base_demo \
        "$BASE_MODEL" "" demo-only "$pid" \
        "$OUT_DIR/mcqppl_128k_pid${pid}_base_demo.json"

    # Teacher: use last-n (K=3) instead of full. R3 was trained at K=3;
    # --context-mode full on 128k MCQs forces left-truncation to max_seq_len
    # via an O(n^2) re-tokenize loop in score_choice_ppl AND feeds R1-domain
    # RoPE positions R3 never saw. K=3 matches R3 training distribution and
    # avoids the tokenize-loop bottleneck.
    run_cond teacher_k3 \
        "$R3" "" last-n "$pid" \
        "$OUT_DIR/mcqppl_128k_pid${pid}_teacher_k3.json"

    LORA_PATH="$OUT_DIR/lora_pid${pid}_r32_ep1_r3teacher/ckpt-step-${STUDENT_STEP}"
    if [[ ! -d "$LORA_PATH" ]]; then
        echo "  [SKIP]    student_demo_step${STUDENT_STEP}: $LORA_PATH does not exist"
    else
        run_cond "student_demo_step${STUDENT_STEP}" \
            "$BASE_MODEL" "$LORA_PATH" demo-only "$pid" \
            "$OUT_DIR/mcqppl_128k_pid${pid}_student_demo_step${STUDENT_STEP}.json"
    fi

    if [[ "$MODE" == "5conds" ]]; then
        # base_full: Instruct-2507 is native 262k so "full" here is truly full —
        # no RoPE OOD for base, but still slow due to tokenize-loop
        run_cond base_full \
            "$BASE_MODEL" "" full "$pid" \
            "$OUT_DIR/mcqppl_128k_pid${pid}_base_full.json"

        run_cond teacher_demo \
            "$R3" "" demo-only "$pid" \
            "$OUT_DIR/mcqppl_128k_pid${pid}_teacher_demo.json"
    fi
done

# --------- Aggregate summary ---------
echo ""
echo "=========================================================="
echo "SUMMARY — Phase 2 MCQ-PPL (128k), student step = $STUDENT_STEP"
echo "=========================================================="

python3 - <<PYEOF
import json
from pathlib import Path

out = Path("$OUT_DIR")
step = $STUDENT_STEP
mode = "$MODE"
personas = [0, 4, 12, 14]

conds_base = [
    ("base_demo",     "mcqppl_128k_pid{pid}_base_demo.json"),
    ("teacher_k3",    "mcqppl_128k_pid{pid}_teacher_k3.json"),
    (f"student_step{step}", f"mcqppl_128k_pid{{pid}}_student_demo_step{step}.json"),
]
conds_extra = [
    ("base_full",     "mcqppl_128k_pid{pid}_base_full.json"),
    ("teacher_demo",  "mcqppl_128k_pid{pid}_teacher_demo.json"),
]
conds = conds_base + (conds_extra if mode == "5conds" else [])

# Per-persona accuracy table
col_width = 18
header = f"{'pid':<5} {'n':>5} " + " ".join(f"{c[0]:>{col_width}}" for c in conds)
print(header)
print("-" * len(header))

aggregates = {c[0]: [] for c in conds}
for pid in personas:
    row = [f"{pid:<5}"]
    n_val = "-"
    for name, pattern in conds:
        p = out / pattern.format(pid=pid)
        if not p.exists():
            row.append(f"{'MISSING':>{col_width}}")
            continue
        d = json.loads(p.read_text())
        acc = d.get("accuracy")
        n = d.get("n_total")
        n_val = n
        aggregates[name].append(acc)
        row.append(f"{acc:>{col_width-4}.3f}".rjust(col_width))
    row.insert(1, f"{n_val:>5}")
    print(" ".join(row))

print("-" * len(header))
# Macro-average across personas (simple mean)
row = [f"{'AVG':<5}", " " * 5]
for name, _ in conds:
    vals = aggregates[name]
    if vals:
        avg = sum(vals) / len(vals)
        row.append(f"{avg:>{col_width-4}.3f}".rjust(col_width))
    else:
        row.append(f"{'-':>{col_width}}")
print(" ".join(row))

# Phase-2 diagnostics
print()
print("--- Phase-2 diagnostics ---")
key_base = "base_demo"
key_teacher = "teacher_k3"
key_student = f"student_step{step}"
if all(aggregates[k] for k in (key_base, key_teacher, key_student)):
    base_avg = sum(aggregates[key_base]) / len(aggregates[key_base])
    teacher_avg = sum(aggregates[key_teacher]) / len(aggregates[key_teacher])
    student_avg = sum(aggregates[key_student]) / len(aggregates[key_student])
    gap_total = teacher_avg - base_avg
    gap_closed = student_avg - base_avg
    closure = gap_closed / gap_total if gap_total != 0 else float("nan")
    print(f"  base_demo avg       : {base_avg:.3f}")
    print(f"  student_step{step} avg : {student_avg:.3f}")
    print(f"  teacher_k3 avg      : {teacher_avg:.3f}")
    print(f"  student - base      : {gap_closed:+.3f}  ({100*closure:.1f}% of teacher_k3 - base_demo gap)")
    print(f"  student - teacher   : {student_avg - teacher_avg:+.3f}")
PYEOF

echo ""
echo "Done. Per-persona JSONs under $OUT_DIR/mcqppl_128k_*"
echo "     Per-run logs under $LOG_DIR/"
