#!/usr/bin/env bash
# v591 sequential cycle queue.
#
# Launches multiple cycles ONE AT A TIME — waits for the prior cycle's
# python process to exit before starting the next. Avoids TRAPI
# rate-limit / retry-storm that killed cycle266+267 in parallel mode.
#
# Usage:
#   bash scripts/v591_sequential_queue.sh <suffix1> <suffix2> ... [--actions=N]
# Example:
#   bash scripts/v591_sequential_queue.sh 266 267 268 --actions=400

set -u

REPO=/home/v-seungplee/skilldiscovery/arc-agi-3-symbolica-research
QUEUE_LOG="${REPO}/simple_logs/ft09-9ab2447a/v591_queue_$(date +%s).log"
mkdir -p "$(dirname "$QUEUE_LOG")"

ACTIONS=400
WAIT_FOR=""
SUFFIXES=()
for arg in "$@"; do
  case "$arg" in
    --actions=*) ACTIONS="${arg#--actions=}" ;;
    --wait-for=*) WAIT_FOR="${arg#--wait-for=}" ;;
    *) SUFFIXES+=("$arg") ;;
  esac
done

if [ "${#SUFFIXES[@]}" -eq 0 ]; then
  echo "usage: $0 <suffix1> [<suffix2> ...] [--actions=N] [--wait-for=<launch.log>]" >&2
  exit 2
fi

echo "[queue] $(date '+%F %T %Z') start suffixes=${SUFFIXES[*]} actions=$ACTIONS wait_for=$WAIT_FOR" >> "$QUEUE_LOG"

# Run the queue itself in a detached session so the script survives
# PAM session close.
(
  setsid -f bash -c '
    exec </dev/null
    exec >>"'"$QUEUE_LOG"'" 2>&1
    cd "'"$REPO"'"
    SUFFIXES=('"${SUFFIXES[*]}"')
    ACTIONS="'"$ACTIONS"'"
    WAIT_FOR="'"$WAIT_FOR"'"
    if [ -n "$WAIT_FOR" ] && [ -f "$WAIT_FOR" ]; then
      echo "[queue] $(date) waiting for prior cycle to finish: $WAIT_FOR"
      while ! grep -qE "python exited rc=" "$WAIT_FOR" 2>/dev/null; do
        sleep 60
      done
      echo "[queue] $(date) prior cycle finished; starting queue"
      sleep 60
    fi
    for suffix in "${SUFFIXES[@]}"; do
      echo "[queue] $(date) launching cycle$suffix"
      LAUNCH_LOG=$(bash scripts/launch_v591_detached.sh "$suffix" "$ACTIONS" 2>&1 | tail -1)
      echo "[queue] $(date) cycle$suffix log=$LAUNCH_LOG"
      # Wait for the python process of this cycle to exit. Detect by
      # tailing the launch.log for "python exited rc=" sentinel emitted
      # by launch_v591_detached.sh.
      while ! grep -qE "python exited rc=" "$LAUNCH_LOG" 2>/dev/null; do
        sleep 30
      done
      echo "[queue] $(date) cycle$suffix exited; moving to next"
      # Brief gap to let TRAPI rate-limit window reset.
      sleep 60
    done
    echo "[queue] $(date) all cycles done"
  '
) &
disown $! 2>/dev/null || true

echo "${QUEUE_LOG}"
