#!/usr/bin/env bash
# paired_v8_skill.sh — V4 paired SkillLibrary ablation (cold vs warm).
#
# Intent: V4 autoresearch test for plan v651 SkillLibrary.
#   - cold arm:    S0_SKILL_ENABLED=1 + fresh per-episode tmp persistence (no skills loaded)
#   - warm arm:    S0_SKILL_ENABLED=1 + state/skills_ft09.json (cumulative N-1)
#   Both arms have S0_VG_ENABLED=1 to isolate the SkillLibrary effect from VG.
#
# Prerequisite (do not launch without these):
#   - V1a-V2a unit gates green:   pytest tests/test_skill_library.py -v
#   - V3a scope probe green:      python scripts/skill_scope_probe.py
#   - V6 frozen-boundary diff 0:  git diff -- agents/templates/agentica/ | wc -l == 0
#   - V7 V0a regression 15/15:    pytest tests/test_s0_trapi_proxy_contract.py -v
#
# Hypotheses (plan v651 §2):
#   H_B1: warm reaches first L>=1 in fewer actions than cold (same seed); median reduction >= 30%.
#   H_B2_strict: at least 5/6 paired runs satisfy warm <= cold + 0.2*cold (per-pair regression cap).
#   H_B3: no skill where falsify_n > confirm_n remains selectable (auto-quarantine fires).
#   H_B4: held-out non-ft09 frames -> graceful fail.
#   H_B5: per warm L>=1 clear, identify Skill recipe + evidence chain (trace required).
#   H_B6: per-episode recovery_cost <= 3 actions median across warm runs.
#   H_B7: zero skill_overrode_observation events with cost > 0.
#
# Safety (3-tier matches paired_v7):
#   tier 1: shell wall-clock PAIRED_V8_TIMEOUT_S (default 3600s).
#   tier 2: S0_MAX_ACTIONS=60 (game LL cap).
#   tier 3: aggressive port/process cleanup.
#
# Note on warm-arm bootstrapping: the first launch effectively bootstraps the
# library (consolidator-driven skills generated in cold episodes can be reused
# by later warm episodes). Post-hoc analysis distinguishes "true warm" (skills
# from prior session) vs "bootstrap warm" (skills from same-session prior cold).
set -u

cd "$(dirname "$0")/.."

OUT_DIR="reports/paired_v8"
mkdir -p "$OUT_DIR"

# Warm skill library lives at this path; cold uses a fresh per-episode tmp.
WARM_PERSIST="state/skills_ft09.json"

if [ $# -gt 0 ]; then
  SEEDS=("$@")
else
  SEEDS=(42 43 44 45 46 47)   # 6 seeds for paired sign-test (plan v651 §3 V4)
fi
TIMEOUT_S="${PAIRED_V8_TIMEOUT_S:-3600}"
echo "[paired_v8] seeds=${SEEDS[*]} max_actions=60 timeout=${TIMEOUT_S}s warm_persist=${WARM_PERSIST}" | tee "$OUT_DIR/runner.log"

ARMS=("cold:fresh" "warm:persist")

for SEED in "${SEEDS[@]}"; do
  for ARM_KV in "${ARMS[@]}"; do
    ARM_NAME="${ARM_KV%%:*}"
    ARM_MODE="${ARM_KV##*:}"
    TAG="${ARM_NAME}_s${SEED}"
    LOG_FILE="$OUT_DIR/${TAG}.log"
    PROXY_COPY="$OUT_DIR/${TAG}.proxy.jsonl"
    REPORT_COPY="$OUT_DIR/${TAG}.report.json"
    TELEMETRY_COPY="$OUT_DIR/${TAG}.skill_telemetry.json"

    if [ -s "$PROXY_COPY" ]; then
      echo "[paired_v8] SKIP ${TAG} (already has proxy snapshot $(wc -c <$PROXY_COPY) bytes)" \
        | tee -a "$OUT_DIR/runner.log"
      continue
    fi

    if [ "$ARM_MODE" = "fresh" ]; then
      EP_PERSIST="$OUT_DIR/${TAG}.skills_fresh.json"
      rm -f "$EP_PERSIST" "$EP_PERSIST.bak" "$EP_PERSIST.events.jsonl"
    else
      EP_PERSIST="$WARM_PERSIST"
    fi

    echo "" | tee -a "$OUT_DIR/runner.log"
    echo "[paired_v8] === ${TAG} (arm=${ARM_NAME} persist=${EP_PERSIST}) ===" | tee -a "$OUT_DIR/runner.log"
    date | tee -a "$OUT_DIR/runner.log"

    rm -f /tmp/s0_smoke_proxy.jsonl
    rm -f /tmp/s0_smoke_ft09_*.json
    rm -f /tmp/skill_telemetry_*.json

    pkill -KILL -f "uvicorn scripts.trapi_openai_proxy" 2>/dev/null || true
    pkill -KILL -f "agentica-server.*main.py" 2>/dev/null || true
    fuser -k -KILL 9091/tcp 2>/dev/null || true
    fuser -k -KILL 2345/tcp 2>/dev/null || true
    for i in $(seq 1 24); do
      if ! ss -tan 2>/dev/null | grep -qE ':(9091|2345)\s'; then
        break
      fi
      sleep 5
    done

    timeout "$TIMEOUT_S" env \
      S0_VG_ENABLED=1 \
      S0_SKILL_ENABLED=1 \
      S0_SKILL_PERSISTENCE_PATH="$EP_PERSIST" \
      S0_MAX_ACTIONS=60 \
      S0_MODEL_PRESET=gpt-5.5 \
      S0_SEED="$SEED" \
      TRAPI_PROXY_AAD_SCOPE="https://cognitiveservices.azure.com/.default" \
      TRAPI_BASE_URL="https://agl-dev.cognitiveservices.azure.com/openai" \
      TRAPI_PROXY_STRIP_DATE_SUFFIX=1 \
      TRAPI_PROXY_DEFAULT_MODEL=gpt-5.5 \
      TRAPI_API_VERSION="2025-04-01-preview" \
      TRAPI_PROXY_MAX_RETRIES=10 \
      TRAPI_PROXY_TRANSPARENT_RETRY_SLEEP_S=180 \
      TRAPI_PROXY_MAX_TRANSPARENT_RETRIES=10 \
      python scripts/s0_smoke_ft09.py >"$LOG_FILE" 2>&1
    RC=$?

    if [ -f /tmp/s0_smoke_proxy.jsonl ]; then
      cp /tmp/s0_smoke_proxy.jsonl "$PROXY_COPY"
    fi
    LATEST_REPORT=$(ls -t /tmp/s0_smoke_ft09_*.json 2>/dev/null | head -1 || true)
    if [ -n "${LATEST_REPORT:-}" ] && [ -f "$LATEST_REPORT" ]; then
      cp "$LATEST_REPORT" "$REPORT_COPY"
    fi
    LATEST_TEL=$(ls -t /tmp/skill_telemetry_*.json 2>/dev/null | head -1 || true)
    if [ -n "${LATEST_TEL:-}" ] && [ -f "$LATEST_TEL" ]; then
      cp "$LATEST_TEL" "$TELEMETRY_COPY"
    fi

    echo "[paired_v8] ${TAG} rc=$RC" | tee -a "$OUT_DIR/runner.log"
    grep -E "PASS|FAIL|levels_completed|action_count|wrote report|skill_telemetry" "$LOG_FILE" 2>/dev/null \
      | tail -10 | tee -a "$OUT_DIR/runner.log" || true
  done
done

echo "" | tee -a "$OUT_DIR/runner.log"
echo "[paired_v8] DONE @ $(date)" | tee -a "$OUT_DIR/runner.log"
