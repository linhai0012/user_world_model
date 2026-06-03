#!/bin/bash
# Phase 2b Round 2a MCQ-PPL eval — thin wrapper that retags run_round1b.
#
# Round 2a uses the same eval shape as R1b (auto-discover all ckpts under
# $OUT_ROOT/lora_pid{N}_${TAG}_..., demo-only context for student, reuse
# cached base_demo / teacher_k3 from Phase 2). Only the TAG differs.
#
# Usage:
#   bash dynamic_usersim/student_opd/run_round2a_mcq_eval.sh
#   # or override:
#   PERSONAS="14" bash dynamic_usersim/student_opd/run_round2a_mcq_eval.sh

set -e

TAG="${TAG:-dual_v3_gated_k3}" \
OUT_ROOT="${OUT_ROOT:-$SCRATCHDIR/P-OPSD/student_lora}" \
exec bash "$(dirname "$0")/run_round1b_mcq_eval.sh" "$@"
