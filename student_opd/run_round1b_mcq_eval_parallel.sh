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

# Run queue, N_GPUS at a time
ts=$(date +%Y%m%d_%H%M%S)
batch_idx=0
i=0
while [[ $i -lt $n_total ]]; do
    batch_idx=$((batch_idx + 1))
    batch_pids=()
    for ((g=0; g<N_GPUS && i<n_total; g++, i++)); do
        name="${queue_name[$i]}"
        base="${queue_base[$i]}"
        lora="${queue_lora[$i]}"
        ctx="${queue_ctx[$i]}"
        pid="${queue_pid[$i]}"
        out="${queue_out[$i]}"

        log="$LOG_DIR/pid${pid}_${name}_${ts}.log"
        lora_arg=()
        [[ -n "$lora" ]] && lora_arg=(--lora-path "$lora" --lora-mode dual)

        echo "  [GPU$g] pid=$pid $name -> $(basename $out)"
        CUDA_VISIBLE_DEVICES=$g python dynamic_usersim/student_opd/eval_opd.py \
            --persona-id "$pid" \
            --base-model "$base" "${lora_arg[@]}" \
            --context-mode "$ctx" \
            --mcq-version "$VERSION" \
            --out-json "$out" \
            > "$log" 2>&1 &
        batch_pids+=($!)
    done

    # Wait for batch to complete
    fail=0
    for p in "${batch_pids[@]}"; do
        wait "$p" || fail=1
    done

    n_done=$i
    pct=$(( 100 * n_done / n_total ))
    echo "  [batch $batch_idx/$n_batches done] $n_done / $n_total ($pct%)"

    if [[ $fail -ne 0 ]]; then
        echo "  WARN: at least one job in batch $batch_idx failed (see logs)"
        # continue — other batches independent
    fi
done

echo ""
echo "=========================================================="
echo "All done. Run summary:"
echo "  python3 dynamic_usersim/student_opd/compare_rounds.py \\"
echo "      --personas ${PERSONAS[*]} --version $VERSION"
echo "=========================================================="
