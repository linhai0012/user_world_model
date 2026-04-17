#!/bin/bash
# Phase 2b Round 2c — joint gate (teacher confidence + top-1 disagreement).
#
# Replaces R2a's entropy gate after R2a failed its core hypothesis.
# qtype analysis showed:
#   - entropy gate dropped to ~7% by step 30 (Instruct-2507 base is more
#     peaked than R3 SFT teacher → H_s < H_t on most tokens → gate closed)
#   - track_evolution / recall_facts gains R1b had were ALL lost in R2a
#     (gate closed on tokens where teacher genuinely teaches well)
#   - Leilani acknowledge_latest dropped to 0.05 (R1b @200 = 0.20) —
#     gate concentration of gradient AMPLIFIED teacher's wrong direction
#
# Joint gate: gate = (H_teacher < tau) AND (argmax_t != argmax_s)
# Only train on tokens where (a) teacher is genuinely confident AND
# (b) student doesn't already agree with teacher's top choice.
#   - Decoupled from H_s → unaffected by Instruct-base overconfidence
#   - Skip when student & teacher already agree → save gradient for
#     genuine disagreement
#   - Doesn't address "teacher confidently wrong" (no GT signal) — see
#     plan §5.2 final paragraph
#
# Inherits ALL of R1b best settings (same as R2a):
#   - dual LoRA s32f16, slow_lr 5e-5, fast_lr 2e-4
#   - --student-ctx demo
#   - --save-every 200 --save-total-limit 0
#   - output to $SCRATCHDIR/P-OPSD/student_lora
# Adds:
#   - --gated-kl --gate-mode joint --gate-tau 1.0
#   - tag = dual_v3_joint_k3
#
# Usage:
#   bash dynamic_usersim/student_opd/run_round2c_train_interactive.sh
#   GATE_TAU=0.5 bash ...    # stricter teacher-confidence threshold

set -euo pipefail

PERSONAS_STR="0 4 12 14"
GPUS_STR=""
EPOCHS="${EPOCHS:-1}"
SAVE_EVERY="${SAVE_EVERY:-200}"
SAVE_TOTAL="${SAVE_TOTAL:-0}"
R3="${R3:-$SCRATCHDIR/P-OPSD/teacher_sft_128k_k3/final}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
SLOW_LR="${SLOW_LR:-5e-5}"
FAST_LR="${FAST_LR:-2e-4}"
STUDENT_CTX="${STUDENT_CTX:-demo}"
GATE_MODE="${GATE_MODE:-joint}"
GATE_TAU="${GATE_TAU:-1.0}"
OUT_ROOT="${OUT_ROOT:-$SCRATCHDIR/P-OPSD/student_lora}"
TAG="${TAG:-dual_v3_joint_k3}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --personas)    PERSONAS_STR="$2"; shift 2 ;;
        --gpus)        GPUS_STR="$2";     shift 2 ;;
        --epochs)      EPOCHS="$2";       shift 2 ;;
        --slow-lr)     SLOW_LR="$2";      shift 2 ;;
        --fast-lr)     FAST_LR="$2";      shift 2 ;;
        --student-ctx) STUDENT_CTX="$2";  shift 2 ;;
        --gate-mode)   GATE_MODE="$2";    shift 2 ;;
        --gate-tau)    GATE_TAU="$2";     shift 2 ;;
        --tag)         TAG="$2";          shift 2 ;;
        --out-root)    OUT_ROOT="$2";     shift 2 ;;
        --r3)          R3="$2";           shift 2 ;;
        --base)        BASE_MODEL="$2";   shift 2 ;;
        --no-resume)   NO_RESUME=1;       shift 1 ;;
        --extra)       EXTRA_ARGS+=($2);  shift 2 ;;
        -h|--help)     sed -n '2,40p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1"; exit 1 ;;
    esac
done

[[ -z "${NO_RESUME:-}" ]] && EXTRA_ARGS+=("--resume")

read -r -a PERSONAS <<< "$PERSONAS_STR"
if [[ -z "$GPUS_STR" ]]; then
    GPUS=()
    for ((i=0; i<${#PERSONAS[@]}; i++)); do GPUS+=("$i"); done
else
    read -r -a GPUS <<< "$GPUS_STR"
fi
[[ ${#PERSONAS[@]} -ne ${#GPUS[@]} ]] && \
    { echo "ERROR: ${#PERSONAS[@]} personas but ${#GPUS[@]} GPUs"; exit 1; }

REPO_DIR="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$REPO_DIR"
mkdir -p logs "$OUT_ROOT"

[[ ! -f "$R3/config.json" ]] && \
    { echo "ERROR: R3 not found at $R3/config.json"; exit 1; }

ts=$(date +%Y%m%d_%H%M%S)

echo "=========================================================="
echo "Phase 2b Round 2c — dual LoRA + JOINT-GATED reverse KL (K=3)"
echo "  personas:      ${PERSONAS[*]}"
echo "  gpus:          ${GPUS[*]}"
echo "  student_ctx:   $STUDENT_CTX"
echo "  slow_lr:       $SLOW_LR    fast_lr: $FAST_LR"
echo "  gate_mode:     $GATE_MODE   gate_tau: $GATE_TAU"
echo "  save_every:    $SAVE_EVERY    save_total_limit: $SAVE_TOTAL"
echo "  out_root:      $OUT_ROOT"
echo "  tag:           $TAG"
echo "  R3 teacher:    $R3"
echo "  ts:            $ts"
echo "=========================================================="

PIDS=()
LOGS=()
for i in "${!PERSONAS[@]}"; do
    pid="${PERSONAS[$i]}"
    gpu="${GPUS[$i]}"
    data="$REPO_DIR/dynamic_usersim/outputs/opd_128k_pid${pid}_k3.jsonl"
    out_dir="$OUT_ROOT/lora_pid${pid}_${TAG}_ep${EPOCHS}_r3teacher"
    log="logs/round2c_pid${pid}_${ts}.log"

    if [[ ! -f "$data" ]]; then
        echo "  WARN pid=$pid: data missing at $data — skipping"
        continue
    fi

    echo "  launching pid=$pid on gpu=$gpu  -> $log"
    python dynamic_usersim/student_opd/train_opd_dual.py \
        --persona-id "$pid" \
        --teacher-path "$R3" \
        --student-base "$BASE_MODEL" \
        --data-path "$data" \
        --output-dir "$out_dir" \
        --student-ctx "$STUDENT_CTX" \
        --slow-lr "$SLOW_LR" \
        --fast-lr "$FAST_LR" \
        --gated-kl \
        --gate-mode "$GATE_MODE" \
        --gate-tau "$GATE_TAU" \
        --gpu "$gpu" \
        --epochs "$EPOCHS" \
        --save-every "$SAVE_EVERY" \
        --save-total-limit "$SAVE_TOTAL" \
        "${EXTRA_ARGS[@]}" \
        > "$log" 2>&1 &
    PIDS+=($!)
    LOGS+=("$log")
done

[[ ${#PIDS[@]} -eq 0 ]] && { echo "no jobs launched"; exit 1; }

echo ""
echo "all ${#PIDS[@]} jobs launched. tail any log with:"
for L in "${LOGS[@]}"; do echo "  tail -f $L"; done
echo ""
echo "to kill all: kill ${PIDS[*]}"
echo ""
echo "waiting for completion..."

fail=0
for i in "${!PIDS[@]}"; do
    p="${PIDS[$i]}"
    if wait "$p"; then
        echo "  [done] pid=${PERSONAS[$i]}"
    else
        rc=$?
        echo "  [FAIL] pid=${PERSONAS[$i]} (exit=$rc) — see ${LOGS[$i]}"
        fail=1
    fi
done

[[ $fail -ne 0 ]] && { echo ""; echo "=== one or more personas failed ==="; exit 1; }

echo ""
echo "=== all personas done ==="
echo "outputs: $OUT_ROOT/lora_pid{0,4,12,14}_${TAG}_ep${EPOCHS}_r3teacher/"
echo "next:"
echo "  1) bash dynamic_usersim/student_opd/run_round2c_mcq_eval.sh"
echo "  2) python3 dynamic_usersim/student_opd/analyze_gate_ratio.py --tag $TAG"
