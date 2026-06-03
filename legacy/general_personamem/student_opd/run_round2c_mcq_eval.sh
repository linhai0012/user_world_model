#!/bin/bash
# Phase 2b Round 2c MCQ-PPL eval — thin wrapper that retags run_round1b.
#
# Same auto-discovery + cache logic as R1b/R2a, just different tag
# (dual_v3_joint_k3) so JSONs don't clash with previous rounds.

set -e

TAG="${TAG:-dual_v3_joint_k3}" \
OUT_ROOT="${OUT_ROOT:-$SCRATCHDIR/P-OPSD/student_lora}" \
exec bash "$(dirname "$0")/run_round1b_mcq_eval.sh" "$@"
