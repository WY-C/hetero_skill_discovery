#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-0}"
SEED="${2:-2}"
T_MAX="${3:-30000}"

[[ "$GPU" =~ ^[0-9]+$ ]] || { echo "GPU must be a non-negative integer" >&2; exit 2; }
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "SEED must be a non-negative integer" >&2; exit 2; }
[[ "$T_MAX" =~ ^[1-9][0-9]*$ ]] || { echo "T_MAX must be a positive integer" >&2; exit 2; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${HISSD_PYTHON:-}"
if [[ -z "$PY" ]]; then
    for CANDIDATE in \
        "$HOME/HiSSD_capability_eval/envs/hissd/bin/python" \
        "$HOME/anaconda3/envs/hissd/bin/python"; do
        if [[ -x "$CANDIDATE" ]]; then
            PY="$CANDIDATE"
            break
        fi
    done
fi
LOG="$ROOT/run_logs/noobs_optimized_seed${SEED}_${T_MAX}.log"
PIDFILE="$ROOT/run_logs/noobs_optimized_seed${SEED}_${T_MAX}.pid"

[[ -n "$PY" && -x "$PY" ]] || {
    echo "Missing HiSSD Python environment; set HISSD_PYTHON" >&2
    exit 3
}
[[ -e "$ROOT/dataset" ]] || { echo "Missing dataset link: $ROOT/dataset" >&2; exit 3; }
mkdir -p "$ROOT/run_logs"

if [[ -f "$PIDFILE" ]]; then
    OLD_PID="$(cat "$PIDFILE")"
    if [[ "$OLD_PID" =~ ^[0-9]+$ ]] && kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Already running: pid=$OLD_PID"
        exit 0
    fi
fi

GPU_UUID="$(nvidia-smi --id="$GPU" --query-gpu=uuid --format=csv,noheader 2>/dev/null | head -1 | tr -d '[:space:]')"
[[ -n "$GPU_UUID" ]] || { echo "GPU $GPU is unavailable" >&2; exit 4; }
if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null \
        | sed 's/[[:space:]]//g' | grep -Fxq "$GPU_UUID"; then
    echo "GPU $GPU is busy; no training was started" >&2
    nvidia-smi --query-compute-apps=gpu_uuid,pid,used_gpu_memory --format=csv,noheader >&2 || true
    exit 5
fi

cd "$ROOT"
export SC2PATH="${SC2PATH:-$HOME/StarCraftII}"
export PYTHONPATH="$ROOT/src:$ROOT/smac:${PYTHONPATH:-}"
export HISSD_EVAL_WORKERS=10

"$PY" -m py_compile \
    src/main.py src/mto.py \
    src/modules/agents/multi_task/hissd_capability_team_agent.py \
    src/learners/multi_task/hissd_capability_team_learner.py
"$PY" tests/test_v3_smoke.py

nohup env CUDA_VISIBLE_DEVICES="$GPU" HISSD_EVAL_WORKERS="$HISSD_EVAL_WORKERS" \
    "$PY" -u src/main.py --mto \
      --config=hissd_capability_team_noobs \
      --env-config=sc2_offline \
      --task-config=stalker-zealot-expert \
      --seed="$SEED" --t_max="$T_MAX" \
      --save_model=True --save_model_interval=10000 \
      --test_nepisode=100 --test_interval=1000 \
      --common_skill_adapter_enabled=True \
      --v3_log_interval_steps=100 \
      --progress_bar=False \
      --remark="optimized_seed${SEED}_${T_MAX}" \
      > "$LOG" 2>&1 < /dev/null &

PID=$!
echo "$PID" > "$PIDFILE"
sleep 3
if ! kill -0 "$PID" 2>/dev/null; then
    echo "Training stopped during startup" >&2
    tail -80 "$LOG" >&2 || true
    exit 6
fi

echo "Started pid=$PID GPU=$GPU seed=$SEED t_max=$T_MAX"
echo "Log: $LOG"
