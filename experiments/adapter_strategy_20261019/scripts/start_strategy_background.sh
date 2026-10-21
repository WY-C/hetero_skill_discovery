#!/usr/bin/env bash
set -euo pipefail

STRATEGY="${1:?Strategy is required}"
GPU="${2:?GPU is required}"
LOG_NAME="${3:?Log name is required}"
CHECKPOINT="${4:--}"
RESUME_ROOT="${5:-}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p "$ROOT/logs" "$ROOT/pids"
LOG="$ROOT/logs/$LOG_NAME"
PIDFILE="$ROOT/pids/${LOG_NAME%.log}.pid"
if [[ -f "$PIDFILE" ]]; then
    OLD_PID="$(cat "$PIDFILE")"
    if [[ "$OLD_PID" =~ ^[0-9]+$ ]] && kill -0 "$OLD_PID" 2>/dev/null; then
        echo "$OLD_PID"
        exit 0
    fi
fi

UNIT="hissd-${LOG_NAME%.log}-20261019"
if command -v systemd-run >/dev/null 2>&1 \
        && systemctl --user is-system-running >/dev/null 2>&1; then
    EXISTING_PID="$(systemctl --user show "$UNIT.service" \
        -p MainPID --value 2>/dev/null || true)"
    if [[ "$EXISTING_PID" =~ ^[1-9][0-9]*$ ]] \
            && kill -0 "$EXISTING_PID" 2>/dev/null; then
        printf '%s\n' "$EXISTING_PID" > "$PIDFILE"
        echo "$EXISTING_PID"
        exit 0
    fi
    systemd-run --user --unit="$UNIT" --collect \
        --property="WorkingDirectory=$ROOT" \
        --property="StandardOutput=append:$LOG" \
        --property="StandardError=append:$LOG" \
        "$ROOT/scripts/run_strategy_experiment.sh" \
        "$STRATEGY" "$GPU" "$CHECKPOINT" "$RESUME_ROOT" >/dev/null
    sleep 8
    PID="$(systemctl --user show "$UNIT.service" -p MainPID --value)"
    if [[ ! "$PID" =~ ^[1-9][0-9]*$ ]] \
            || ! kill -0 "$PID" 2>/dev/null; then
        echo "Strategy failed during startup: $STRATEGY" >&2
        tail -80 "$LOG" >&2 || true
        exit 6
    fi
    printf '%s\n' "$PID" > "$PIDFILE"
    echo "$PID"
    exit 0
fi

nohup "$ROOT/scripts/run_strategy_experiment.sh" \
    "$STRATEGY" "$GPU" "$CHECKPOINT" "$RESUME_ROOT" \
    > "$LOG" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$PIDFILE"
sleep 8
if ! kill -0 "$PID" 2>/dev/null; then
    echo "Strategy failed during startup: $STRATEGY" >&2
    tail -80 "$LOG" >&2 || true
    exit 6
fi
echo "$PID"
