#!/usr/bin/env bash
# v607 D1 detached launcher (5 ep × 30 turns LCL 0.005).
#
# Usage:
#   bash scripts/launch_v607_detached.sh <cycle_id_suffix> [max_actions]
#
# v607 specifics:
#   - -a lite (agentica_lite, NOT agentica_simple)
#   - ARC_LITE_SKILL_STATE_PATH per-cycle (no cross-contamination)
#   - ARC_NO_GOAL_LEAK=1 strict
#   - Per-cycle skill_state.json captures Reflector emissions + Beta posterior

set -u

REPO=/home/v-seungplee/skilldiscovery/arc-agi-3-symbolica-research
GAME=ft09-9ab2447a
SUFFIX="${1:-$(date +%s)}"
MAX_ACTIONS="${2:-150}"

NAMESPACE="cycle${1}_v607"
LOG_DIR="${REPO}/simple_logs/${GAME}/${NAMESPACE}"
LOG="${LOG_DIR}/launch.log"
HEARTBEAT="${LOG}.heartbeat"
SKILL_STATE="${LOG_DIR}/skill_state.json"

mkdir -p "$LOG_DIR"
echo "[launch] $(date '+%F %T %Z') starting ${NAMESPACE} max_actions=${MAX_ACTIONS}" >> "$LOG"
echo "[launch] skill_state_path=${SKILL_STATE}" >> "$LOG"

setsid -f bash -c '
  exec </dev/null
  exec >>"'"$LOG"'" 2>&1
  cd "'"$REPO"'"
  export PATH=/home/v-seungplee/miniconda3/bin:$PATH

  export ARC_NO_GOAL_LEAK=1
  export ARC_USE_LOCAL_ENV_ONLY=1
  export ARC_SMOKE_MAX_ACTIONS='"$MAX_ACTIONS"'
  export ARC_LOG_PATH="'"$LOG_DIR"'"/agent.log
  export ARC_LITE_SKILL_STATE_PATH="'"$SKILL_STATE"'"
  export ARC_AGENTICA_TRAPI_TIMEOUT_SEC=600

  # v607 P10: ARC_LITE_MODEL overrides proposer model (gpt-5.4-mini default).
  # Set to "gpt-5.5_2026-04-24" to reproduce cycle237-era model.
  # Set ARC_LITE_ANTI_LEAK_MODE to {strict|v_leak_only|off} to test leak hypothesis.
  [ -n "${ARC_LITE_MODEL:-}" ] && export ARC_LITE_MODEL
  [ -n "${ARC_LITE_ANTI_LEAK_MODE:-}" ] && export ARC_LITE_ANTI_LEAK_MODE

  ( while true; do
      touch "'"$HEARTBEAT"'" 2>/dev/null || break
      sleep 60
    done
  ) &
  HEARTBEAT_PID=$!

  echo "[detached] $(date) python launch starting (heartbeat pid=$HEARTBEAT_PID)"
  python -u main.py \
    -a lite \
    -g "'"$GAME"'"
  RC=$?
  echo "[detached] $(date) python exited rc=$RC"
  kill "$HEARTBEAT_PID" 2>/dev/null
  exit $RC
'

echo "${LOG}"
