#!/bin/bash
# Phase 2b Round 1 MCQ-PPL aggregate evaluation.
#
# Mirrors run_phase2_mcq_eval.sh but for the dual-LoRA layout:
#   - student conditions use --lora-mode dual + --context-mode recent-turns
#   - teacher_k3 stays at last-n=3 sessions (R3's training distribution)
#   - base conditions get a base_recent (same recent-turns context as student,
#     to ablate the 2-turn context contribution from the dual-LoRA contribution)
#
# Usage:
#   bash dynamic_usersim/student_opd/run_round1_mcq_eval.sh                     # default step=final
#   bash dynamic_usersim/student_opd/run_round1_mcq_eval.sh --student-step 200
#   bash dynamic_usersim/student_opd/run_round1_mcq_eval.sh --student-step 400 --recent-turns 2

set -e

# --------- Defaults ---------
STUDENT_STEP=final
RECENT_TURNS=2
R3_DEFAULT=${R3:-$SCRATCHDIR/P-OPSD/teacher_sft_128k_k3/final}
R3=$R3_DEFAULT
BASE_MODEL=Qwen/Qwen3-4B-Instruct-2507
PERSONAS=(0 4 12 14)
LORA_TAG=dual_s32f16   # matches train_opd_dual.py default output dir

# --------- Args ---------
while [[ $# -gt 0 ]]; do
    case $1 in
        --student-step) STUDENT_STEP="$2"; shift 2 ;;
        --recent-turns) RECENT_TURNS="$2"; shift 2 ;;
        --r3)           R3="$2";           shift 2 ;;
        --base)         BASE_MODEL="$2";   shift 2 ;;
        --lora-tag)     LORA_TAG="$2";     shift 2 ;;
        -h|--help)
            sed -n '2,16p' "$0"
            exit 0
            ;;
        *) echo "unknown arg: $1"; exit 1 ;;
    esac
done

OUT_DIR=dynamic_usersim/outputs
LOG_DIR=logs/round1_eval_mcq
mkdir -p "$LOG_DIR"

echo "=========================================================="
echo "Phase 2b Round 1 MCQ-PPL eval"
echo "  student step  : $STUDENT_STEP"
echo "  recent-turns  : $RECENT_TURNS"
echo "  R3 teacher    : $R3"
echo "  base model    : $BASE_MODEL"
echo "  lora tag      : $LORA_TAG"
echo "  personas      : ${PERSONAS[*]}"
echo "=========================================================="

# Args: cond  base  lora  ctx  pid  outjson  [extra_flags]
run_cond() {
    local name="$1" base="$2" lora="$3" ctx="$4" pid="$5" outjson="$6"
    shift 6
    local extra=("$@")
    if [[ -f "$outjson" ]]; then
        echo "  [cached]  pid=$pid  $name  -> $outjson"
        return
    fi
    local lora_arg=()
    if [[ -n "$lora" ]]; then
        lora_arg=(--lora-path "$lora" --lora-mode dual)
    fi
    local log_file="$LOG_DIR/pid${pid}_${name}.log"
    echo "  [running] pid=$pid  $name  (ctx=$ctx)"
    torchrun --standalone --nnodes=1 --nproc_per_node=4 \
        dynamic_usersim/student_opd/eval_opd.py \
        --persona-id "$pid" \
        --base-model "$base" "${lora_arg[@]}" \
        --context-mode "$ctx" \
        "${extra[@]}" \
        --out-json "$outjson" \
        > "$log_file" 2>&1
    local acc=$(python3 -c "import json; print(f\"{json.load(open('$outjson'))['accuracy']:.3f}\")" 2>/dev/null || echo "?")
    echo "            -> acc=$acc  (log: $log_file)"
}

for pid in "${PERSONAS[@]}"; do
    echo ""
    echo "===== persona $pid ====="

    # Phase-2 baseline base_demo (cached if Phase 2 already ran)
    run_cond base_demo \
        "$BASE_MODEL" "" demo-only "$pid" \
        "$OUT_DIR/mcqppl_128k_pid${pid}_base_demo.json"

    # New: base WITH recent-turns context, no LoRA — isolates the
    # contribution of the 2-turn context from the LoRA contribution
    run_cond "base_recent${RECENT_TURNS}" \
        "$BASE_MODEL" "" recent-turns "$pid" \
        "$OUT_DIR/mcqppl_128k_pid${pid}_base_recent${RECENT_TURNS}.json" \
        --recent-turns "$RECENT_TURNS"

    # Teacher_k3 (cached if Phase 2 already ran)
    run_cond teacher_k3 \
        "$R3" "" last-n "$pid" \
        "$OUT_DIR/mcqppl_128k_pid${pid}_teacher_k3.json"

    # Student dual-LoRA condition
    if [[ "$STUDENT_STEP" == "final" ]]; then
        LORA_PATH="$OUT_DIR/lora_pid${pid}_${LORA_TAG}_ep1_r3teacher"
        STUDENT_LABEL="student_${LORA_TAG}_recent${RECENT_TURNS}_final"
    else
        LORA_PATH="$OUT_DIR/lora_pid${pid}_${LORA_TAG}_ep1_r3teacher/ckpt-step-${STUDENT_STEP}"
        STUDENT_LABEL="student_${LORA_TAG}_recent${RECENT_TURNS}_step${STUDENT_STEP}"
    fi
    STUDENT_JSON="$OUT_DIR/mcqppl_128k_pid${pid}_${STUDENT_LABEL}.json"
    if [[ ! -d "$LORA_PATH/slow" ]] || [[ ! -d "$LORA_PATH/fast" ]]; then
        echo "  [SKIP]    ${STUDENT_LABEL}: dual-LoRA dir missing slow/+fast/ at $LORA_PATH"
    else
        run_cond "$STUDENT_LABEL" \
            "$BASE_MODEL" "$LORA_PATH" recent-turns "$pid" \
            "$STUDENT_JSON" \
            --recent-turns "$RECENT_TURNS"
    fi
done

# --------- Aggregate summary ---------
echo ""
echo "=========================================================="
echo "SUMMARY — Phase 2b Round 1 MCQ-PPL (128k)"
echo "  student step = $STUDENT_STEP, recent-turns = $RECENT_TURNS"
echo "=========================================================="

python3 - "$OUT_DIR" "$STUDENT_STEP" "$RECENT_TURNS" "$LORA_TAG" <<'PYEOF'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
step = sys.argv[2]
rt = sys.argv[3]
lora_tag = sys.argv[4]
personas = [0, 4, 12, 14]

if step == "final":
    student_label = f"student_{lora_tag}_recent{rt}_final"
else:
    student_label = f"student_{lora_tag}_recent{rt}_step{step}"

conds = [
    ("base_demo",           f"mcqppl_128k_pid{{pid}}_base_demo.json"),
    (f"base_recent{rt}",    f"mcqppl_128k_pid{{pid}}_base_recent{rt}.json"),
    ("teacher_k3",          f"mcqppl_128k_pid{{pid}}_teacher_k3.json"),
    (student_label,         f"mcqppl_128k_pid{{pid}}_{student_label}.json"),
]

col = 22
header = f"{'pid':<5} {'n':>5} " + " ".join(f"{c[0]:>{col}}" for c in conds)
print(header)
print("-" * len(header))

agg = {c[0]: [] for c in conds}
for pid in personas:
    row = [f"{pid:<5}"]
    n_val = "-"
    for name, pat in conds:
        p = out / pat.format(pid=pid)
        if not p.exists():
            row.append(f"{'MISSING':>{col}}")
            continue
        d = json.loads(p.read_text())
        acc = d.get("accuracy")
        n_val = d.get("n_total")
        agg[name].append(acc)
        row.append(f"{acc:>{col-4}.3f}".rjust(col))
    row.insert(1, f"{n_val:>5}")
    print(" ".join(row))

print("-" * len(header))
row = [f"{'AVG':<5}", " " * 5]
for name, _ in conds:
    vals = agg[name]
    row.append(f"{sum(vals)/len(vals):>{col-4}.3f}".rjust(col)
               if vals else f"{'-':>{col}}")
print(" ".join(row))

# Diagnostics
print()
print("--- diagnostics ---")
key_base = "base_demo"
key_base_recent = f"base_recent{rt}"
key_teacher = "teacher_k3"
key_student = student_label
if all(agg[k] for k in (key_base, key_base_recent, key_teacher, key_student)):
    a_base = sum(agg[key_base]) / len(agg[key_base])
    a_recent = sum(agg[key_base_recent]) / len(agg[key_base_recent])
    a_teacher = sum(agg[key_teacher]) / len(agg[key_teacher])
    a_student = sum(agg[key_student]) / len(agg[key_student])
    gap_total = a_teacher - a_base
    closure = (a_student - a_base) / gap_total if gap_total != 0 else float("nan")
    print(f"  base_demo                : {a_base:.3f}")
    print(f"  base_recent{rt}              : {a_recent:.3f}  "
          f"(2-turn context alone gives {a_recent - a_base:+.3f})")
    print(f"  teacher_k3               : {a_teacher:.3f}")
    print(f"  {key_student} : {a_student:.3f}")
    print(f"  student − base_demo      : {a_student - a_base:+.3f}  "
          f"({100*closure:.1f}% of teacher_k3 - base_demo gap)")
    print(f"  student − base_recent{rt}    : {a_student - a_recent:+.3f}  "
          f"(dual-LoRA gain on top of 2-turn context — Round 1 ablation)")
    print(f"  student − teacher_k3     : {a_student - a_teacher:+.3f}")
PYEOF

echo ""
echo "Done. Per-persona JSONs under $OUT_DIR/mcqppl_128k_*"
echo "     Per-run logs under $LOG_DIR/"
