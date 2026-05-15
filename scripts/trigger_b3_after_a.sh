#!/usr/bin/env bash
# trigger_b3_after_a.sh — watch paired_v7 runner; auto-launch paired_v8 on success.
#
# Usage: bash scripts/trigger_b3_after_a.sh <paired_v7_pid> [<paired_v8_timeout_s>]
#
# Verifies paired_v7 completed cleanly (DONE marker in runner.log) before
# launching paired_v8_skill (cold vs warm SkillLibrary ablation). If
# paired_v7 died early or crashed, this trigger refuses to auto-launch
# B3 to avoid wasting compute on a half-broken setup.
set -u

cd "$(dirname "$0")/.."

PAIRED_V7_PID="${1:?usage: $0 <paired_v7_pid> [<paired_v8_timeout_s>]}"
PAIRED_V8_TIMEOUT_S="${2:-3600}"
LOG="reports/trigger_b3_after_a.log"

mkdir -p reports

echo "[trigger] $(date) start; watching paired_v7 PID=$PAIRED_V7_PID; timeout for paired_v8=${PAIRED_V8_TIMEOUT_S}s" | tee -a "$LOG"

# Poll for paired_v7 completion (5min interval, paired_v7 takes hours)
while kill -0 "$PAIRED_V7_PID" 2>/dev/null; do
  sleep 300
done

echo "[trigger] $(date) paired_v7 PID $PAIRED_V7_PID exited" | tee -a "$LOG"

# Verify completion marker
if grep -q "^\[paired_v7\] DONE @" reports/paired_v7/runner.log 2>/dev/null; then
  echo "[trigger] paired_v7 DONE marker present, proceeding to B3" | tee -a "$LOG"
else
  echo "[trigger] paired_v7 DONE marker MISSING; refusing to auto-launch paired_v8. Manual review required." | tee -a "$LOG"
  tail -20 reports/paired_v7/runner.log 2>/dev/null | tee -a "$LOG"
  exit 1
fi

# Show paired_v7 summary
echo "[trigger] paired_v7 summary:" | tee -a "$LOG"
grep -E "control_s|treatment_s|rc=|DONE" reports/paired_v7/runner.log 2>/dev/null \
  | tail -20 | tee -a "$LOG"

# Brief cleanup + port wait
sleep 30
pkill -KILL -f "uvicorn scripts.trapi_openai_proxy" 2>/dev/null || true
pkill -KILL -f "agentica-server.*main.py" 2>/dev/null || true
sleep 10
for i in $(seq 1 12); do
  if ! ss -tan 2>/dev/null | grep -qE ':(9091|2345)\s'; then
    break
  fi
  sleep 5
done

echo "[trigger] $(date) launching paired_v8_skill (timeout ${PAIRED_V8_TIMEOUT_S}s, 6 seeds)" | tee -a "$LOG"

mkdir -p reports/paired_v8
PAIRED_V8_TIMEOUT_S="$PAIRED_V8_TIMEOUT_S" \
  bash scripts/paired_v8_skill.sh 42 43 44 45 46 47 \
  >> reports/paired_v8/runner.stdout.log 2>&1
RC=$?

echo "[trigger] $(date) paired_v8 exited rc=$RC" | tee -a "$LOG"
echo "[trigger] paired_v8 summary:" | tee -a "$LOG"
tail -20 reports/paired_v8/runner.log 2>/dev/null | tee -a "$LOG"
