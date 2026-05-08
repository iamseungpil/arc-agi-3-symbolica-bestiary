#!/usr/bin/env bash
# B19 v591 detached cycle launcher.
#
# Usage:
#   bash scripts/launch_v591_detached.sh <cycle_id_suffix> [max_actions]
#
# - Forks via `setsid -f` so python runs in its own session and
#   survives PAM session close (cycle263 root cause: at + systemd-run
#   --user both inherited PAM lifetime).
# - Heartbeat thread touches <log>.heartbeat every 60s for external
#   liveness check (plan §6 R-7).
# - Three model env vars set per plan §4.6:
#     M1 = ACTION  = gpt-5.4-pro
#     M3 = HYPOTHESIZE = gpt-5.4-pro
#     M4 = REFLEXION = gpt-5.5_2026-04-24

set -u

REPO=/home/v-seungplee/skilldiscovery/arc-agi-3-symbolica-research
GAME=ft09-9ab2447a
SUFFIX="${1:-$(date +%s)}"
MAX_ACTIONS="${2:-400}"

NAMESPACE="cycle${SUFFIX}_v591"
LOG_DIR="${REPO}/simple_logs/${GAME}/${NAMESPACE}"
LOG="${LOG_DIR}/launch.log"
HEARTBEAT="${LOG}.heartbeat"

mkdir -p "$LOG_DIR"
echo "[launch] $(date '+%F %T %Z') starting ${NAMESPACE} max_actions=${MAX_ACTIONS}" >> "$LOG"

setsid -f bash -c '
  exec </dev/null
  exec >>"'"$LOG"'" 2>&1
  cd "'"$REPO"'"
  # Use base miniconda python directly — conda activate fails inside
  # the setsid subshell (PATH not inherited).
  export PATH=/home/v-seungplee/miniconda3/bin:$PATH

  export ARC_NO_GOAL_LEAK=1
  export ARC_USE_LOCAL_ENV_ONLY=1
  export ARC_SMOKE_MAX_ACTIONS='"$MAX_ACTIONS"'
  export ARC_LOG_PATH="'"$LOG_DIR"'"/agent.log
  # Cycle263 root cause was NOT PAM session-close: it was openai client
  # timeout (default 120s). gpt-5.4-pro multimodal+14KB prompt routinely
  # exceeds 120s. Set to 600s (10min) to match openai SDK upper bound.
  export ARC_AGENTICA_TRAPI_TIMEOUT_SEC=600
  # round-7 model swap (2026-05-08): cycle266e ran 103 actions clean
  # with gpt-5.4 (gpt-5.4-pro normalized) but 0 L+ events. cycle237 era
  # (also our only L+2 success) used gpt-5.5. ft09 ~deterministic + same
  # prompt + same code → only difference left = model. Switch back to
  # gpt-5.5 for M1 + M3 to attempt cycle237 reproduction. M4 stays gpt-5.5.
  # User can override via V591_M1_MODEL etc env vars.
  export V591_M1_MODEL="${V591_M1_MODEL:-gpt-5.5}"
  export V591_M3_MODEL="${V591_M3_MODEL:-gpt-5.5}"
  export V591_M4_MODEL="${V591_M4_MODEL:-gpt-5.5_2026-04-24}"

  ( while true; do
      touch "'"$HEARTBEAT"'" 2>/dev/null || break
      sleep 60
    done
  ) &
  HEARTBEAT_PID=$!

  echo "[detached] $(date) python launch starting (heartbeat pid=$HEARTBEAT_PID)"
  python -u main.py \
    -a agentica_v57 \
    -g "'"$GAME"'"
  RC=$?
  echo "[detached] $(date) python exited rc=$RC"
  kill "$HEARTBEAT_PID" 2>/dev/null
  exit $RC
'

echo "${LOG}"
