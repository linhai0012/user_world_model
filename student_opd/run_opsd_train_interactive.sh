#!/bin/bash
# OPSD (On-Policy Self Distillation, variant C: teacher sees GT as a prior
# user turn) training launcher. Mirrors run_round1b_train_interactive.sh
# but calls train_opsd_dual.py instead. All R1b hyperparameters are kept
# identical (dual LoRA s32f16, slow_lr 5e-5, demo-only student ctx, etc.)
# so the only thing changing vs R1b is teacher's GT-injection at scoring.
#
# Sanity check (§sanity_check_gt_injection.py) validated GT placement:
#   P(GT): A=0.57, B=0.89, C=0.94  (chose C = prior-turn; largest ΔP vs
#   mismatched-GT control = +0.384, confirming semantic use of GT).
#
# Usage:
#   bash dynamic_usersim/student_opd/run_opsd_train_interactive.sh
#   # or subset:
#   bash dynamic_usersim/student_opd/run_opsd_train_interactive.sh \
#       --personas "4" --gpus "0"

set -euo pipefail

# ---- Defaults (match R1b recipe) ----
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
OUT_ROOT="${OUT_ROOT:-$SCRATCHDIR/P-OPSD/student_lora}"
TAG="${TAG:-dual_s32f16_opsd}"   # distinct from R1b's 'dual_v2'
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --personas)    PERSONAS_STR="$2"; shift 2 ;;
        --gpus)        GPUS_STR="$2";     shift 2 ;;
        --epochs)      EPOCHS="$2";       shift 2 ;;
        --slow-lr)     SLOW_LR="$2";      shift 2 ;;
        --fast-lr)     FAST_LR="$2";      shift 2 ;;
        --student-ctx) STUDENT_CTX="$2";  shift 2 ;;
        --tag)         TAG="$2";          shift 2 ;;
        --out-root)    OUT_ROOT="$2";     shift 2 ;;
        --r3)          R3="$2";           shift 2 ;;
        --base)        BASE_MODEL="$2";   shift 2 ;;
        --no-resume)   NO_RESUME=1;       shift 1 ;;
        --extra)       EXTRA_ARGS+=($2);  shift 2 ;;
        -h|--help)     sed -n '2,30p' "$0"; exit 0 ;;
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
echo "OPSD-C dual-LoRA — teacher sees GT as prior user turn"
echo "  personas:     ${PERSONAS[*]}"
echo "  gpus:         ${GPUS[*]}"
echo "  student_ctx:  $STUDENT_CTX"
echo "  slow_lr:      $SLOW_LR    fast_lr: $FAST_LR"
echo "  save_every:   $SAVE_EVERY   save_total_limit: $SAVE_TOTAL"
echo "  out_root:     $OUT_ROOT"
echo "  tag:          $TAG"
echo "  R3:           $R3"
echo "  ts:           $ts"
echo "=========================================================="

PIDS=()
LOGS=()
for i in "${!PERSONAS[@]}"; do
    pid="${PERSONAS[$i]}"
    gpu="${GPUS[$i]}"
    data="$REPO_DIR/dynamic_usersim/outputs/opd_128k_pid${pid}_k3.jsonl"
    out_dir="$OUT_ROOT/lora_pid${pid}_${TAG}_ep${EPOCHS}_r3teacher"
    log="logs/opsd_pid${pid}_${ts}.log"

    if [[ ! -f "$data" ]]; then
        echo "  WARN pid=$pid: data missing at $data — skipping"
        continue
    fi

    echo "  launching pid=$pid on gpu=$gpu  -> $log"
    python dynamic_usersim/student_opd/train_opsd_dual.py \
        --persona-id "$pid" \
        --teacher-path "$R3" \
        --student-base "$BASE_MODEL" \
        --data-path "$data" \
        --output-dir "$out_dir" \
        --student-ctx "$STUDENT_CTX" \
        --slow-lr "$SLOW_LR" \
        --fast-lr "$FAST_LR" \
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
echo "outputs under: $OUT_ROOT/lora_pid{0,4,12,14}_${TAG}_ep${EPOCHS}_r3teacher/"
