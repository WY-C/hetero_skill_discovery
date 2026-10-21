#!/usr/bin/env bash
set -euo pipefail

OLD_PID="${1:?Existing training PID is required}"
OLD_START_TICKS="${2:?Existing PID start ticks are required}"
OLD_LOG="${3:?Existing training log is required}"
OLD_CHECKPOINT="${4:?Existing final checkpoint directory is required}"
GPU="${5:-0}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p "$ROOT/logs" "$ROOT/pids"
exec 9>"$ROOT/pids/D_waiter.lock"
flock -n 9 || { echo "D waiter is already running" >&2; exit 10; }
printf '%s\n' "$$" > "$ROOT/pids/D_waiter.pid"

while kill -0 "$OLD_PID" 2>/dev/null; do
    CURRENT_START="$(awk '{print $22}' "/proc/$OLD_PID/stat" 2>/dev/null || true)"
    CURRENT_CMD="$(tr '\0' ' ' < "/proc/$OLD_PID/cmdline" 2>/dev/null || true)"
    if [[ "$CURRENT_START" != "$OLD_START_TICKS" \
          || "$CURRENT_CMD" != *"--config=hissd_capability_team_kl_noobs"* \
          || "$CURRENT_CMD" != *"--t_max=30000"* ]]; then
        echo "Existing PID identity changed; D will not start" >&2
        exit 11
    fi
    sleep 60
done

for _ in $(seq 1 10); do
    if [[ -s "$OLD_LOG" && -s "$OLD_CHECKPOINT/agent.th" \
          && -s "$OLD_CHECKPOINT/mixer.th" && -s "$OLD_CHECKPOINT/opt.th" \
          && -s "$OLD_CHECKPOINT/resume.th" ]] \
          && grep -Fq "Finished Training" "$OLD_LOG"; then
        break
    fi
    sleep 12
done
if [[ ! -s "$OLD_CHECKPOINT/agent.th" ]] \
        || ! grep -Fq "Finished Training" "$OLD_LOG" \
        || grep -Eq "Traceback|\[FAILED\]" "$OLD_LOG"; then
    echo "Existing training did not complete normally; D will not start" >&2
    exit 12
fi

GPU_UUID="$(nvidia-smi --id="$GPU" --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')"
while nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null \
        | sed 's/[[:space:]]//g' | grep -Fxq "$GPU_UUID"; do
    sleep 60
done

"$ROOT/scripts/start_strategy_background.sh" \
    joint_scratch "$GPU" D_joint_scratch.log -
