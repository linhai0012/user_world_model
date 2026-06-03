#!/bin/bash
# Phase 2b R1b — extend to all 20 personas (auto-skip already-trained).
#
# Wrapper around run_round1b_train_interactive.sh. For each target persona:
#   1. Build OPD data file if missing
#   2. Skip if a fully-trained LoRA already exists (root slow/ + fast/)
#   3. Otherwise queue for training
# Then run in batches of N_GPUS personas (default 4, parallel within batch,
# sequential across batches).
#
# All R1b hyperparameters inherited from run_round1b_train_interactive.sh
# defaults — no behavior change vs the 4-persona run we already validated:
#   dual s32f16, slow_lr=5e-5, fast_lr=2e-4, --student-ctx demo,
#   save_every=200, save_total_limit=0, output to $SCRATCHDIR, ungated KL.
#
# Re-runnable: skip-if-exists logic + --resume in inner launcher means you
# can ctrl-C, walk away, come back and re-run — it picks up where it left off.
#
# Usage:
#   bash dynamic_usersim/student_opd/run_round1b_extend.sh
#   # subset:
#   bash dynamic_usersim/student_opd/run_round1b_extend.sh \
#       --personas "1 2 3 5"
#   # change GPUs per batch (e.g. 8 if you ever get an 8-GPU node):
#   N_GPUS=8 bash dynamic_usersim/student_opd/run_round1b_extend.sh

set -euo pipefail

# --------- Defaults ---------
ALL_PERSONAS_STR="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19"
N_GPUS="${N_GPUS:-4}"
TAG="${TAG:-dual_v2}"
OUT_ROOT="${OUT_ROOT:-$SCRATCHDIR/P-OPSD/student_lora}"
EPOCHS="${EPOCHS:-1}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --personas) ALL_PERSONAS_STR="$2"; shift 2 ;;
        --n-gpus)   N_GPUS="$2";           shift 2 ;;
        --tag)      TAG="$2";              shift 2 ;;
        --out-root) OUT_ROOT="$2";         shift 2 ;;
        --epochs)   EPOCHS="$2";           shift 2 ;;
        -h|--help)  sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1"; exit 1 ;;
    esac
done

read -r -a ALL_PERSONAS <<< "$ALL_PERSONAS_STR"

REPO_DIR="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$REPO_DIR"
DATA_DIR="$REPO_DIR/dynamic_usersim/outputs"
mkdir -p logs

ts=$(date +%Y%m%d_%H%M%S)

echo "=========================================================="
echo "Phase 2b R1b — extend to multiple personas"
echo "  target personas: ${ALL_PERSONAS[*]} (${#ALL_PERSONAS[@]} total)"
echo "  N_GPUS per batch: $N_GPUS"
echo "  tag:              $TAG"
echo "  out_root:         $OUT_ROOT"
echo "  epochs:           $EPOCHS"
echo "  ts:               $ts"
echo "=========================================================="

# --------- Step 1: build missing OPD data files ---------
echo ""
echo "=== Step 1: build missing OPD data (128k) ==="
to_build=()
for pid in "${ALL_PERSONAS[@]}"; do
    [[ -f "$DATA_DIR/opd_128k_pid${pid}_k3.jsonl" ]] || to_build+=("$pid")
done
if [[ ${#to_build[@]} -gt 0 ]]; then
    echo "  building data for ${#to_build[@]} personas: ${to_build[*]}"
    python dynamic_usersim/student_opd/build_opd_data.py \
        --personas "${to_build[@]}" \
        --n-context-turns 0 \
        --version 128k 2>&1 | tee logs/r1b_extend_build_${ts}.log
else
    echo "  (all OPD data files already exist)"
fi

# --------- Step 2: identify training queue ---------
echo ""
echo "=== Step 2: identify training queue ==="
to_train=()
for pid in "${ALL_PERSONAS[@]}"; do
    out_dir="$OUT_ROOT/lora_pid${pid}_${TAG}_ep${EPOCHS}_r3teacher"
    if [[ -f "$out_dir/slow/adapter_config.json" ]] \
       && [[ -f "$out_dir/fast/adapter_config.json" ]]; then
        echo "  [SKIP]    pid=$pid  already trained ($out_dir)"
        continue
    fi
    to_train+=("$pid")
    if [[ -d "$out_dir" ]] && \
       compgen -G "$out_dir/ckpt-step-*" >/dev/null 2>&1; then
        latest_ckpt=$(ls -d "$out_dir"/ckpt-step-* 2>/dev/null \
            | sort -t- -k3 -n | tail -1)
        echo "  [resume]  pid=$pid  partial state: $(basename $latest_ckpt)"
    else
        echo "  [fresh]   pid=$pid  no prior state"
    fi
done

if [[ ${#to_train[@]} -eq 0 ]]; then
    echo ""
    echo "All ${#ALL_PERSONAS[@]} personas already trained. Done."
    exit 0
fi

n_total=${#to_train[@]}
n_batches=$(( (n_total + N_GPUS - 1) / N_GPUS ))
echo ""
echo "Personas to train: $n_total"
echo "Will run in $n_batches batches of up to $N_GPUS personas each."
echo "Estimated wall time: ~$((4 * n_batches))h (assuming ~4h per batch)."

# --------- Step 3: run in batches ---------
batch_idx=0
while [[ ${#to_train[@]} -gt 0 ]]; do
    batch_idx=$((batch_idx + 1))
    batch=("${to_train[@]:0:$N_GPUS}")
    to_train=("${to_train[@]:$N_GPUS}")

    # Assign GPUs 0..n-1 within batch
    gpus=()
    for ((i=0; i<${#batch[@]}; i++)); do gpus+=("$i"); done

    echo ""
    echo "=========================================================="
    echo "Batch $batch_idx / $n_batches"
    echo "  personas: ${batch[*]}"
    echo "  gpus:     ${gpus[*]}"
    echo "  remaining after batch: ${#to_train[@]} personas"
    echo "=========================================================="

    # Pass TAG / OUT_ROOT / SLOW_LR / etc as env so inner launcher's
    # defaults align with R1b. (Inner launcher reads these as env vars.)
    if ! TAG="$TAG" OUT_ROOT="$OUT_ROOT" EPOCHS="$EPOCHS" \
        bash dynamic_usersim/student_opd/run_round1b_train_interactive.sh \
        --personas "${batch[*]}" \
        --gpus "${gpus[*]}"; then
        echo ""
        echo "!!! Batch $batch_idx FAILED. Aborting remaining batches."
        echo "    Re-run this script to resume — completed personas auto-skip."
        exit 1
    fi

    echo ""
    echo "Batch $batch_idx done. (${#to_train[@]} personas remain)"
done

echo ""
echo "=========================================================="
echo "All training complete. Trained personas: ${ALL_PERSONAS[*]}"
echo "=========================================================="
echo ""
echo "Next:"
echo "  # Eval (auto-discovers all ckpts; reuses cached base/teacher JSONs)"
echo "  bash dynamic_usersim/student_opd/run_round1b_mcq_eval.sh"
echo "  # Big table over all trained personas"
echo "  python3 dynamic_usersim/student_opd/compare_rounds.py \\"
echo "      --personas ${ALL_PERSONAS[*]}"
