#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-0}"
CHECKPOINT="${2:?Usage: eval_v3_noobs_optimized.sh GPU CHECKPOINT_DIR [STEP]}"
STEP="${3:-30000}"

[[ "$GPU" =~ ^[0-9]+$ ]] || { echo "GPU must be a non-negative integer" >&2; exit 2; }
[[ "$STEP" =~ ^[1-9][0-9]*$ ]] || { echo "STEP must be a positive integer" >&2; exit 2; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT="$(realpath "$CHECKPOINT")"
[[ -f "$CHECKPOINT/$STEP/agent.th" ]] || {
    echo "Missing checkpoint: $CHECKPOINT/$STEP/agent.th" >&2
    exit 3
}

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
[[ -n "$PY" && -x "$PY" ]] || {
    echo "Missing HiSSD Python environment; set HISSD_PYTHON" >&2
    exit 3
}

mkdir -p "$ROOT/run_logs"
LOG="$ROOT/run_logs/noobs_eval_step${STEP}.log"

cd "$ROOT"
export SC2PATH="${SC2PATH:-$HOME/StarCraftII}"
export PYTHONPATH="$ROOT/src:$ROOT/smac:${PYTHONPATH:-}"
export HISSD_EVAL_WORKERS=10

CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u src/main.py --mto \
    --config=hissd_capability_team_noobs \
    --env-config=sc2_offline \
    --task-config=stalker-zealot-expert \
    --seed=2026 --evaluate=True \
    --checkpoint_path="$CHECKPOINT" --load_step="$STEP" \
    --test_nepisode=100 --batch_size_run=1 \
    --save_model=False --save_replay=False \
    --common_skill_adapter_enabled=True \
    --progress_bar=False \
    2>&1 | tee "$LOG"
