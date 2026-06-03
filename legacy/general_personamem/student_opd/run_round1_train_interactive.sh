#!/bin/bash
# Phase 2b Round 1 — interactive 4-GPU launcher (no Slurm).
#
# Launches train_opd_dual.py for 4 personas in parallel, one per GPU,
# as background subprocesses. Logs each persona to logs/round1_*.log.
# Resumable — drop a process and re-run; --resume picks up the latest
# ckpt-step-N/ in the persona's output dir.
#
# Pre-reqs (run once before this script):
#   1. Data with student_recent_messages field:
#        python dynamic_usersim/student_opd/build_opd_data.py \
#            --personas 0 4 12 14 --n-context-turns 2
#   2. R3 teacher available at $R3 (defaults to $SCRATCHDIR path).
#   3. (Optional) sanity check on 1 persona:
#        python dynamic_usersim/student_opd/train_opd_dual.py \
#            --persona-id 14 --teacher-path $R3 \
#            --data-path dynamic_usersim/outputs/opd_128k_pid14_k3.jsonl \
#            --max-samples 5 --gpu 0
#      (5 samples on GPU = 1-2 min; verifies real-teacher loss>0)
#
# Usage:
#   bash dynamic_usersim/student_opd/run_round1_train_interactive.sh
#   bash dynamic_usersim/student_opd/run_round1_train_interactive.sh \
#       --personas "0 4" --gpus "0 1"        # subset
#
# Env overrides: R3, BASE_MODEL, EPOCHS, SAVE_EVERY

set -euo pipefail

# --------- Defaults ---------
PERSONAS_STR="0 4 12 14"
GPUS_STR=""                                 # auto = 0..N-1 (one per persona)
EPOCHS="${EPOCHS:-1}"
SAVE_EVERY="${SAVE_EVERY:-200}"
SAVE_TOTAL="${SAVE_TOTAL:-2}"
R3="${R3:-$SCRATCHDIR/P-OPSD/teacher_sft_128k_k3/final}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
EXTRA_ARGS=()

# --------- Args ---------
while [[ $# -gt 0 ]]; do
    case $1 in
        --personas)   PERSONAS_STR="$2"; shift 2 ;;
        --gpus)       GPUS_STR="$2";     shift 2 ;;
        --epochs)     EPOCHS="$2";       shift 2 ;;
        --save-every) SAVE_EVERY="$2";   shift 2 ;;
        --r3)         R3="$2";           shift 2 ;;
        --base)       BASE_MODEL="$2";   shift 2 ;;
        --no-resume)  EXTRA_ARGS+=();    shift 1 ;;
        --extra)      EXTRA_ARGS+=($2);  shift 2 ;;
        -h|--help)    sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1"; exit 1 ;;
    esac
done

# Add --resume by default (override via --no-resume above)
if [[ ! " $* " =~ " --no-resume " ]]; then
    EXTRA_ARGS+=("--resume")
fi

read -r -a PERSONAS <<< "$PERSONAS_STR"
if [[ -z "$GPUS_STR" ]]; then
    GPUS=()
    for ((i=0; i<${#PERSONAS[@]}; i++)); do GPUS+=("$i"); done
else
    read -r -a GPUS <<< "$GPUS_STR"
fi

if [[ ${#PERSONAS[@]} -ne ${#GPUS[@]} ]]; then
    echo "ERROR: ${#PERSONAS[@]} personas but ${#GPUS[@]} GPUs"
    exit 1
fi

# --------- Resolve repo root + sanity ---------
REPO_DIR="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$REPO_DIR"
mkdir -p logs

if [[ ! -f "$R3/config.json" ]]; then
    echo "ERROR: R3 teacher not found at $R3 (no config.json)"
    echo "       set R3=/path/to/teacher_sft_ckpt"
    exit 1
fi

ts=$(date +%Y%m%d_%H%M%S)

echo "=========================================================="
echo "Phase 2b Round 1 dual-LoRA OPD — interactive launcher"
echo "  personas:    ${PERSONAS[*]}"
echo "  gpus:        ${GPUS[*]}"
echo "  epochs:      $EPOCHS"
echo "  save_every:  $SAVE_EVERY"
echo "  R3:          $R3"
echo "  base:        $BASE_MODEL"
echo "  extra args:  ${EXTRA_ARGS[*]}"
echo "  ts:          $ts"
echo "=========================================================="

PIDS=()
LOGS=()
for i in "${!PERSONAS[@]}"; do
    pid="${PERSONAS[$i]}"
    gpu="${GPUS[$i]}"
    data="dynamic_usersim/outputs/opd_128k_pid${pid}_k3.jsonl"
    out_dir="dynamic_usersim/outputs/lora_pid${pid}_dual_s32f16_ep${EPOCHS}_r3teacher"
    log="logs/round1_pid${pid}_${ts}.log"

    if [[ ! -f "$data" ]]; then
        echo "  WARN pid=$pid: data missing at $data — skipping"
        echo "       (run build_opd_data.py --personas $pid --n-context-turns 2)"
        continue
    fi

    echo "  launching pid=$pid on gpu=$gpu  -> $log"
    python dynamic_usersim/student_opd/train_opd_dual.py \
        --persona-id "$pid" \
        --teacher-path "$R3" \
        --student-base "$BASE_MODEL" \
        --data-path "$data" \
        --output-dir "$out_dir" \
        --gpu "$gpu" \
        --epochs "$EPOCHS" \
        --save-every "$SAVE_EVERY" \
        --save-total-limit "$SAVE_TOTAL" \
        "${EXTRA_ARGS[@]}" \
        > "$log" 2>&1 &
    PIDS+=($!)
    LOGS+=("$log")
done

if [[ ${#PIDS[@]} -eq 0 ]]; then
    echo "no jobs launched"
    exit 1
fi

echo ""
echo "all ${#PIDS[@]} jobs launched. tail any log with:"
for L in "${LOGS[@]}"; do
    echo "  tail -f $L"
done
echo ""
echo "to kill all: kill ${PIDS[*]}"
echo ""
echo "waiting for completion..."

fail=0
for i in "${!PIDS[@]}"; do
    p="${PIDS[$i]}"
    if wait "$p"; then
        echo "  [done] pid=${PERSONAS[$i]} (subprocess $p)"
    else
        rc=$?
        echo "  [FAIL] pid=${PERSONAS[$i]} (subprocess $p, exit=$rc) — see ${LOGS[$i]}"
        fail=1
    fi
done

if [[ $fail -ne 0 ]]; then
    echo ""
    echo "=== one or more personas failed ==="
    exit 1
fi

echo ""
echo "=== all personas done ==="
echo "outputs under: dynamic_usersim/outputs/lora_pid{0,4,12,14}_dual_s32f16_ep${EPOCHS}_r3teacher/"
echo "next:  bash dynamic_usersim/student_opd/run_round1_mcq_eval.sh --student-step final"
