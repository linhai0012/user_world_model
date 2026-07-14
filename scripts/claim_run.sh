#!/usr/bin/env bash
# Atomically claim a work area (a code dir or a run_id) before editing/running.
# Coordination is filesystem-based on shared CephFS. See CONVENTIONS.md §4.
#
# Usage:
#   scripts/claim_run.sh <area>             # claim (fails if already held)
#   scripts/claim_run.sh --release <area>   # release
#   scripts/claim_run.sh --list             # show current claims
set -euo pipefail
UWM_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCKDIR="$UWM_REPO/.uwm/locks"
mkdir -p "$LOCKDIR"

case "${1:-}" in
  --list)
    shopt -s nullglob
    held=0
    for d in "$LOCKDIR"/*/; do
      held=1
      printf "%-46s %s\n" "$(basename "$d")" "$(cat "$d/owner.txt" 2>/dev/null || echo '?')"
    done
    [ "$held" -eq 0 ] && echo "(no active claims)"
    exit 0 ;;
  --release)
    area="${2:?area required}"
    rm -f "$LOCKDIR/$area/owner.txt" 2>/dev/null || true
    if rmdir "$LOCKDIR/$area" 2>/dev/null; then echo "released: $area";
    else echo "not held / not empty: $area" >&2; exit 1; fi
    exit 0 ;;
  "" )
    echo "usage: claim_run.sh <area> | --release <area> | --list" >&2; exit 2 ;;
esac

area="$1"
owner="node=$(hostname) session=${UWM_SESSION:-${SLURM_JOB_ID:-shell}} pid=$$ time=$(date -u +%FT%TZ)"
if mkdir "$LOCKDIR/$area" 2>/dev/null; then
  printf '%s\n' "$owner" > "$LOCKDIR/$area/owner.txt"
  echo "CLAIMED  $area  ($owner)"
else
  echo "BUSY     $area  -> $(cat "$LOCKDIR/$area/owner.txt" 2>/dev/null || echo '?')" >&2
  exit 1
fi
