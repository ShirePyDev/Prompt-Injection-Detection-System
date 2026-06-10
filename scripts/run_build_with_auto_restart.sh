#!/usr/bin/env bash
# Run PIDS-Bench build with auto-restart on crash.
# Does NOT restart when crash is due to API rate limit / quota (429, rate limit, quota exceeded).

cd "$(dirname "$0")/.."
CONFIG="${1:-data_builder/config_v3.yaml}"
LOG_FILE="${2:-data_builder/audit/build_run.log}"

mkdir -p "$(dirname "$LOG_FILE")"

_is_rate_limit() {
    local out="$1"
    echo "$out" | grep -qiE "429|rate limit|quota exceeded|insufficient_quota|rate_limit" && return 0
    return 1
}

echo "=== Build started at $(date) ===" | tee -a "$LOG_FILE"
echo "Config: $CONFIG" | tee -a "$LOG_FILE"
echo "Log: $LOG_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

while true; do
    output=$(python -m data_builder.build_dataset --config "$CONFIG" 2>&1)
    exit_code=$?
    echo "$output" | tee -a "$LOG_FILE"

    if [ "$exit_code" -eq 0 ]; then
        echo "" | tee -a "$LOG_FILE"
        echo "=== Build completed successfully at $(date) ===" | tee -a "$LOG_FILE"
        exit 0
    fi

    if _is_rate_limit "$output"; then
        echo "" | tee -a "$LOG_FILE"
        echo "=== Stopped: API rate limit / quota. No restart. ===" | tee -a "$LOG_FILE"
        echo "Resume later with: python -m data_builder.build_dataset --config $CONFIG" | tee -a "$LOG_FILE"
        exit 1
    fi

    echo "" | tee -a "$LOG_FILE"
    echo "=== Crash (non-rate-limit). Restarting in 10s... ===" | tee -a "$LOG_FILE"
    sleep 10
done
