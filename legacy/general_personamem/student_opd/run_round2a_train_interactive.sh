#!/bin/bash
# Phase 2b Round 2a — gated reverse KL on top of R1b's best dual-LoRA setup.
#
# Hypothesis: per-token entropy gating fixes the failure mode identified in
# the R1b qtype analysis — teacher pulls student WORSE than base on certain
# (persona, qtype) cells where teacher's own context is wrong/uncertain
# (e.g. Leilani `acknowledge_latest`: base 0.20, teacher_k3 0.10, R1b student
# converged to 0.075). With gated KL, those tokens have low teacher entropy
# vs student entropy → gate keeps closed → student's base-aligned behaviour
# is preserved.
#
# Inherits ALL of R1b's best settings:
#   - dual LoRA (slow MLP rank 32, fast Attn rank 16)
#   - slow_lr 5e-5  (4x diff vs fast 2e-4)
#   - --student-ctx demo
#   - --save-every 200 --save-total-limit 0  (keep all ckpts)
#   - output to $SCRATCHDIR/P-OPSD/student_lora
# New:
#   - --gated-kl  (entropy gate, hard binary, margin 0)
#   - tag = dual_v3_gated_k3  (so output dirs and JSONs don't clash with R1b)
#
# Pre-reqs: same data files as R1b (opd_128k_pid{N}_k3.jsonl already built).
#
# Usage:
#   bash dynamic_usersim/student_opd/run_round2a_train_interactive.sh
#   # subset / overrides:
#   bash dynamic_usersim/student_opd/run_round2a_train_interactive.sh \
#       --personas "14" --gpus "0"
#   GATE_MARGIN=0.1 bash ...   # add 0.1 nat entropy margin

set -euo pipefail

# --------- Defaults (mostly inherit from R1b launcher) ---------
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
GATE_MARGIN="${GATE_MARGIN:-0.0}"
OUT_ROOT="${OUT_ROOT:-$SCRATCHDIR/P-OPSD/student_lora}"
TAG="${TAG:-dual_v3_gated_k3}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --personas)    PERSONAS_STR="$2"; shift 2 ;;
        --gpus)        GPUS_STR="$2";     shift 2 ;;
        --epochs)      EPOCHS="$2";       shift 2 ;;
        --slow-lr)     SLOW_LR="$2";      shift 2 ;;
        --fast-lr)     FAST_LR="$2";      shift 2 ;;
        --student-ctx) STUDENT_CTX="$2";  shift 2 ;;
        --gate-margin) GATE_MARGIN="$2";  shift 2 ;;
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
echo "Phase 2b Round 2a — dual LoRA + gated reverse KL (K=3)"
echo "  personas:      ${PERSONAS[*]}"
echo "  gpus:          ${GPUS[*]}"
echo "  student_ctx:   $STUDENT_CTX"
echo "  slow_lr:       $SLOW_LR    fast_lr: $FAST_LR"
echo "  gated_kl:      ON   margin: $GATE_MARGIN"
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
    log="logs/round2a_pid${pid}_${ts}.log"

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
        --kl-gate-margin "$GATE_MARGIN" \
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
echo "  1) bash dynamic_usersim/student_opd/run_round2a_mcq_eval.sh"
echo "  2) python3 dynamic_usersim/student_opd/analyze_gate_ratio.py --tag $TAG"
