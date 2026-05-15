#!/usr/bin/env bash
# paired_v7_extended.sh — extended-budget paired VG ablation.
#
# Intent: paired_v6 was capped at S0_MAX_ACTIONS=10. We want to know whether
# stock Symbolica + VG can advance past L=1 when given the full game-side
# budget. We run paired same-seed control(VG_OFF) vs treatment(VG_ON) with:
#   - tier 1 safety: shell wall-clock `timeout 1800` per episode (avoids
#     RPC-decode runaway that wasted compute on control_s43 in paired_v6).
#   - tier 2 budget: S0_MAX_ACTIONS=60 (ft09 LL action cap — game ends
#     naturally without further intervention from us).
#   - tier 3 hygiene: aggressive port/process cleanup between episodes.
#
# Hypotheses (paired_v7):
#   H_A1: treatment(VG_ON) reaches L>=1 in >=50% of seeds (i.e. >=2 of 3).
#   H_A2: control(VG_OFF) stays at L=0 dominant (0-1 of 3 reach L>=1).
#   H_A3: same-seed action_count delta favors treatment — treatment either
#         reaches a higher L with comparable budget OR uses fewer actions
#         to reach the same L. McNemar on n=3 paired is exploratory only.
#
# Outputs:
#   reports/paired_v7/<arm>_s<seed>.log         — full stdout/stderr
#   reports/paired_v7/<arm>_s<seed>.proxy.jsonl — TRAPI proxy capture
#   reports/paired_v7/<arm>_s<seed>.report.json — s0_smoke_ft09 report
#   reports/paired_v7/runner.log                — runner summary
set -u

cd "$(dirname "$0")/.."

OUT_DIR="reports/paired_v7"
mkdir -p "$OUT_DIR"

if [ $# -gt 0 ]; then
  SEEDS=("$@")
else
  SEEDS=(42 43 44)
fi
TIMEOUT_S="${PAIRED_V7_TIMEOUT_S:-3600}"
echo "[paired_v7] seeds=${SEEDS[*]} max_actions=60 timeout=${TIMEOUT_S}s" | tee "$OUT_DIR/runner.log"

ARMS=("control:0" "treatment:1")

for SEED in "${SEEDS[@]}"; do
  for ARM_KV in "${ARMS[@]}"; do
    ARM_NAME="${ARM_KV%%:*}"
    VG_ENABLED="${ARM_KV##*:}"
    TAG="${ARM_NAME}_s${SEED}"
    LOG_FILE="$OUT_DIR/${TAG}.log"
    PROXY_COPY="$OUT_DIR/${TAG}.proxy.jsonl"
    REPORT_COPY="$OUT_DIR/${TAG}.report.json"

    # Skip already-completed episodes (proxy snapshot exists and non-empty)
    if [ -s "$PROXY_COPY" ]; then
      echo "[paired_v7] SKIP ${TAG} (already has proxy snapshot $(wc -c <$PROXY_COPY) bytes)" \
        | tee -a "$OUT_DIR/runner.log"
      continue
    fi

    echo "" | tee -a "$OUT_DIR/runner.log"
    echo "[paired_v7] === ${TAG} (S0_VG_ENABLED=${VG_ENABLED}) ===" | tee -a "$OUT_DIR/runner.log"
    date | tee -a "$OUT_DIR/runner.log"

    rm -f /tmp/s0_smoke_proxy.jsonl
    rm -f /tmp/s0_smoke_ft09_*.json

    # Aggressive cleanup between episodes (paired_v6 post-mortem)
    pkill -KILL -f "uvicorn scripts.trapi_openai_proxy" 2>/dev/null || true
    pkill -KILL -f "agentica-server/src/application/main.py" 2>/dev/null || true
    pkill -KILL -f "agentica-server.*main.py" 2>/dev/null || true
    fuser -k -KILL 9091/tcp 2>/dev/null || true
    fuser -k -KILL 2345/tcp 2>/dev/null || true
    for i in $(seq 1 24); do
      if ! ss -tan 2>/dev/null | grep -qE ':(9091|2345)\s'; then
        break
      fi
      sleep 5
    done

    # 2-tier safety net:
    #   tier 1: shell wall-clock TIMEOUT_S per episode (kills RPC-decode runaway)
    #   tier 2: S0_MAX_ACTIONS=60 game-side budget cap
    timeout "$TIMEOUT_S" env \
      S0_VG_ENABLED="$VG_ENABLED" \
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

    echo "[paired_v7] ${TAG} rc=$RC" | tee -a "$OUT_DIR/runner.log"
    grep -E "PASS|FAIL|levels_completed|action_count|wrote report" "$LOG_FILE" 2>/dev/null \
      | tail -8 | tee -a "$OUT_DIR/runner.log" || true
  done
done

echo "" | tee -a "$OUT_DIR/runner.log"
echo "[paired_v7] DONE @ $(date)" | tee -a "$OUT_DIR/runner.log"
