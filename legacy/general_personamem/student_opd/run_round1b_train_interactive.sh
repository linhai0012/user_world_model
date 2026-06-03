#!/bin/bash
# Phase 2b Round 1b — combined ablation of Round 1's two failure modes.
#
# Diagnosis from Round 1 (see EXPERIMENTS.md / chat log): dual-LoRA avg
# best-step closure was 26% vs Phase 2 single-LoRA's 76%. Two culprits:
#   (1) 2-turn context is selectively poisonous — base_recent2 − base_demo
#       = −2.7pp for Lisa (the most important persona). Student LoRA
#       trained on poisoned input inherits the bad signal.
#   (2) slow_lr=1e-5 is 20x below fast_lr=2e-4 — MLP path barely trains.
#       Effectively reduces dual LoRA to "rank-16 attention only", less
#       capacity than Phase 2 single-LoRA (rank 32, all modules).
#
# Round 1b fixes both at once (resource-constrained single shot):
#   - --student-ctx demo    (drop 2-turn context; demo+chatbot_prev only)
#   - --slow-lr 5e-5        (4x diff vs fast, not 20x — MLP can actually move)
#   - --save-total-limit 0  (keep ALL ckpts; previous R1 lost 200/400)
#   - Output goes to $SCRATCHDIR (100 GiB $HOME quota per Isambard docs)
#
# Pre-reqs (same as R1):
#   data at $HOME/P-OPSD/dynamic_usersim/outputs/opd_128k_pid{N}_k3.jsonl
#
# Usage:
#   bash dynamic_usersim/student_opd/run_round1b_train_interactive.sh
#   # or subset:
#   bash dynamic_usersim/student_opd/run_round1b_train_interactive.sh \
#       --personas "14" --gpus "0"

set -euo pipefail

# --------- Defaults ---------
PERSONAS_STR="0 4 12 14"
GPUS_STR=""
EPOCHS="${EPOCHS:-1}"
SAVE_EVERY="${SAVE_EVERY:-200}"
SAVE_TOTAL="${SAVE_TOTAL:-0}"                       # 0 = keep ALL
R3="${R3:-$SCRATCHDIR/P-OPSD/teacher_sft_128k_k3/final}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
SLOW_LR="${SLOW_LR:-5e-5}"
FAST_LR="${FAST_LR:-2e-4}"
STUDENT_CTX="${STUDENT_CTX:-demo}"                  # R1b: drop 2-turn
OUT_ROOT="${OUT_ROOT:-$SCRATCHDIR/P-OPSD/student_lora}"
TAG="${TAG:-dual_v2}"                               # distinguishes R1 vs R1b
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --personas)   PERSONAS_STR="$2"; shift 2 ;;
        --gpus)       GPUS_STR="$2";     shift 2 ;;
        --epochs)     EPOCHS="$2";       shift 2 ;;
        --slow-lr)    SLOW_LR="$2";      shift 2 ;;
        --fast-lr)    FAST_LR="$2";      shift 2 ;;
        --student-ctx) STUDENT_CTX="$2"; shift 2 ;;
        --tag)        TAG="$2";          shift 2 ;;
        --out-root)   OUT_ROOT="$2";     shift 2 ;;
        --r3)         R3="$2";           shift 2 ;;
        --base)       BASE_MODEL="$2";   shift 2 ;;
        --no-resume)  NO_RESUME=1;       shift 1 ;;
        --extra)      EXTRA_ARGS+=($2);  shift 2 ;;
        -h|--help)    sed -n '2,40p' "$0"; exit 0 ;;
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
echo "Phase 2b Round 1b — dual-LoRA ablation (demo ctx + slow_lr 5e-5)"
echo "  personas:     ${PERSONAS[*]}"
echo "  gpus:         ${GPUS[*]}"
echo "  student_ctx:  $STUDENT_CTX"
echo "  slow_lr:      $SLOW_LR"
echo "  fast_lr:      $FAST_LR"
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
    log="logs/round1b_pid${pid}_${ts}.log"

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
echo "next: bash dynamic_usersim/student_opd/run_round1b_mcq_eval.sh"
