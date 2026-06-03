#!/bin/bash
# Parallel single-GPU MCQ-PPL eval — ~3-4× faster than the DDP launcher.
#
# Why: nvidia-smi during the standard launcher shows ~17% GPU util on
# all 4 cards because demo-only conditions have ~500-token inputs and
# per-MCQ overhead (tokenize/forward setup/NCCL barrier) dominates over
# compute. Splitting one condition across 4 GPUs barely helps.
#
# This launcher runs N independent single-GPU evals in parallel —
# each process owns one GPU, no DDP overhead. Each GPU runs near 100%
# util on its own condition.
#
# Per-condition wall time is similar to DDP version (model load
# dominates), but 4× parallelism → ~3-4× total speedup.
#
# Usage:
#   bash dynamic_usersim/student_opd/run_round1b_mcq_eval_parallel.sh
#   PERSONAS="0 4 12 14" bash ...
#   VERSION=1M bash ...
#   N_GPUS=4 bash ...   # default 4

set -euo pipefail

R3="${R3:-$SCRATCHDIR/P-OPSD/teacher_sft_128k_k3/final}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
PERSONAS=(${PERSONAS:-0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19})
TAG="${TAG:-dual_v2}"
OUT_ROOT="${OUT_ROOT:-$SCRATCHDIR/P-OPSD/student_lora}"
VERSION="${VERSION:-128k}"
N_GPUS="${N_GPUS:-4}"
EVAL_JSON_DIR=dynamic_usersim/outputs
LOG_DIR=logs/r1b_eval_parallel
mkdir -p "$LOG_DIR" "$EVAL_JSON_DIR"

if [[ "$VERSION" != "32k" && "$VERSION" != "128k" && "$VERSION" != "1M" ]]; then
    echo "ERROR: VERSION must be 32k / 128k / 1M (got $VERSION)"
    exit 1
fi

echo "=========================================================="
echo "Parallel single-GPU MCQ-PPL eval"
echo "  version  : $VERSION    tag: $TAG"
echo "  personas : ${PERSONAS[*]}"
echo "  N_GPUS   : $N_GPUS (parallel conditions)"
echo "  R3       : $R3"
echo "  out_root : $OUT_ROOT"
echo "=========================================================="

# Queue: each condition is 6 entries flat (name, base, lora, ctx, pid, outjson)
queue_name=()
queue_base=()
queue_lora=()
queue_ctx=()
queue_pid=()
queue_out=()

enqueue() {
    local name="$1" base="$2" lora="$3" ctx="$4" pid="$5" out="$6"
    if [[ -f "$out" ]]; then
        echo "  [cached] pid=$pid $name"
        return
    fi
    queue_name+=("$name"); queue_base+=("$base"); queue_lora+=("$lora")
    queue_ctx+=("$ctx"); queue_pid+=("$pid"); queue_out+=("$out")
}

# Build queue across all personas
echo ""
echo "=== Building queue ==="
for pid in "${PERSONAS[@]}"; do
    enqueue "base_demo" "$BASE_MODEL" "" "demo-only" "$pid" \
        "$EVAL_JSON_DIR/mcqppl_${VERSION}_pid${pid}_base_demo.json"
    enqueue "teacher_k3" "$R3" "" "last-n" "$pid" \
        "$EVAL_JSON_DIR/mcqppl_${VERSION}_pid${pid}_teacher_k3.json"

    persona_dir="$OUT_ROOT/lora_pid${pid}_${TAG}_ep1_r3teacher"
    [[ -d "$persona_dir" ]] || continue
    for ckpt in $(ls -d "$persona_dir"/ckpt-step-* 2>/dev/null | sort -t- -k3 -n); do
        [[ -d "$ckpt/slow" && -d "$ckpt/fast" ]] || continue
        step=$(basename "$ckpt" | sed 's/ckpt-step-//')
        enqueue "student_step${step}" "$BASE_MODEL" "$ckpt" "demo-only" "$pid" \
            "$EVAL_JSON_DIR/mcqppl_${VERSION}_pid${pid}_student_${TAG}_demo_step${step}.json"
    done
    if [[ -d "$persona_dir/slow" && -d "$persona_dir/fast" ]]; then
        enqueue "student_final" "$BASE_MODEL" "$persona_dir" "demo-only" "$pid" \
            "$EVAL_JSON_DIR/mcqppl_${VERSION}_pid${pid}_student_${TAG}_demo_final.json"
    fi
done

n_total=${#queue_name[@]}
if [[ $n_total -eq 0 ]]; then
    echo ""
    echo "Nothing to do — all conditions cached."
    exit 0
fi

n_batches=$(( (n_total + N_GPUS - 1) / N_GPUS ))
echo ""
echo "$n_total conditions queued, $n_batches batches of $N_GPUS"
echo "Estimated time: ~$((n_batches * 90 / 60)) minutes (assuming 90s per condition)"
echo ""

# Rolling worker pool: when ANY GPU finishes, immediately dequeue next.
# Prevents idle GPUs waiting for a slow condition (e.g. teacher_k3 with
# K=3 context can be 3-5× slower than demo-only conditions).
#
# Bash 4.4 compat: uses `wait -n` (added in 4.3) for "wait for any" but
# scans `kill -0 $pid` to identify which child died (bash 5.1's
# `wait -n -p VAR` would tell us directly, but Isambard runs 4.4).
ts=$(date +%Y%m%d_%H%M%S)
declare -A gpu_pid          # gpu_index -> background PID
declare -A pid_gpu          # PID -> gpu_index
declare -A pid_label        # PID -> "pid=X cond=Y" for logging
declare -A pid_start        # PID -> start timestamp (seconds since epoch)
n_done=0

launch_on_gpu() {
    local g="$1" idx="$2"
    local name="${queue_name[$idx]}"
    local base="${queue_base[$idx]}"
    local lora="${queue_lora[$idx]}"
    local ctx="${queue_ctx[$idx]}"
    local pid="${queue_pid[$idx]}"
    local out="${queue_out[$idx]}"
    local log="$LOG_DIR/pid${pid}_${name}_${ts}.log"
    local lora_arg=()
    [[ -n "$lora" ]] && lora_arg=(--lora-path "$lora" --lora-mode dual)

    CUDA_VISIBLE_DEVICES=$g python dynamic_usersim/student_opd/eval_opd.py \
        --persona-id "$pid" \
        --base-model "$base" "${lora_arg[@]}" \
        --context-mode "$ctx" \
        --mcq-version "$VERSION" \
        --out-json "$out" \
        > "$log" 2>&1 &
    local p=$!
    gpu_pid[$g]=$p
    pid_gpu[$p]=$g
    pid_label[$p]="pid=$pid $name"
    pid_start[$p]=$(date +%s)
    echo "  [GPU$g start  $(date +%H:%M:%S) pid=$pid $name]"
}

# Initial fill: launch first N_GPUS conditions
i=0
for ((g=0; g<N_GPUS && i<n_total; g++, i++)); do
    launch_on_gpu $g $i
done

# Rolling: each time any job finishes, free its GPU and launch next.
# Bash 4.4 doesn't have `wait -n -p`, so we wait for ANY child to finish
# (wait -n) then scan our pool for which PID is no longer alive.
while [[ ${#pid_gpu[@]} -gt 0 ]]; do
    # wait -n: blocks until ANY background child terminates; returns its
    # exit code in $? (we capture as `rc` for the most-recently-finished).
    wait -n 2>/dev/null
    rc=$?

    # Find which of our pool PIDs is no longer alive (= the one that just
    # finished). Loop because in rare race we may need to retry.
    done_pid=""
    for attempt in 1 2 3; do
        for p in "${!pid_gpu[@]}"; do
            if ! kill -0 "$p" 2>/dev/null; then
                done_pid="$p"
                break 2
            fi
        done
        sleep 0.2
    done

    if [[ -z "$done_pid" ]]; then
        # Shouldn't happen — wait -n returned but no pool PID is dead.
        # Defensive: full wait-all and exit loop to avoid spinning.
        echo "  WARN: wait -n returned (rc=$rc) but no pool PID died; "
        echo "        falling back to wait-all on remaining ${#pid_gpu[@]} jobs"
        for p in "${!pid_gpu[@]}"; do wait "$p" 2>/dev/null || true; done
        break
    fi

    g=${pid_gpu[$done_pid]}
    label=${pid_label[$done_pid]}
    start=${pid_start[$done_pid]}
    elapsed=$(( $(date +%s) - start ))
    n_done=$((n_done + 1))
    pct=$(( 100 * n_done / n_total ))

    status="ok"
    [[ $rc -ne 0 ]] && status="FAIL($rc)"
    echo "  [GPU$g done   $(date +%H:%M:%S) $label  ${elapsed}s  $status]  $n_done/$n_total ($pct%)"

    # Free GPU, drop PID
    unset 'gpu_pid[$g]'
    unset 'pid_gpu[$done_pid]'
    unset 'pid_label[$done_pid]'
    unset 'pid_start[$done_pid]'

    # Dequeue next if any
    if [[ $i -lt $n_total ]]; then
        launch_on_gpu $g $i
        i=$((i + 1))
    fi
done

echo ""
echo "=========================================================="
echo "All done. Run summary:"
echo "  python3 dynamic_usersim/student_opd/compare_rounds.py \\"
echo "      --personas ${PERSONAS[*]} --version $VERSION"
echo "=========================================================="
