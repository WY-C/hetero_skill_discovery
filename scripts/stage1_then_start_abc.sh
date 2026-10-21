#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
<<<<<<< HEAD
REMOTE_ROOT="/home/noward/Desktop/HiSSD_adapter_strategy_experiments_20261019"
REMOTE2_ROOT="/home/noward2/Desktop/HiSSD_adapter_strategy_experiments_20261019"
=======
REMOTE_HOST="${HISSD_REMOTE_HOST:-noward}"
REMOTE2_HOST="${HISSD_REMOTE2_HOST:-noward2}"
REMOTE_ROOT="${HISSD_REMOTE_ROOT:-HiSSD_adapter_strategy_experiments}"
REMOTE2_ROOT="${HISSD_REMOTE2_ROOT:-HiSSD_adapter_strategy_experiments}"
>>>>>>> origin/main
ARTIFACT_ROOT="$ROOT/artifacts/official_hissd_seed2_30k"
LOG="${STAGE1_LOG_PATH:-$ROOT/logs/stage1_official_hissd.log}"
STAGE1_RESUME_ROOT="${STAGE1_RESUME_ROOT:-}"

PY="${HISSD_PYTHON:-}"
if [[ -z "$PY" ]]; then
    for CANDIDATE in \
        "$HOME/HiSSD_capability_eval/envs/hissd/bin/python" \
        "$HOME/anaconda3/envs/hissd/bin/python"; do
        if [[ -x "$CANDIDATE" ]]; then PY="$CANDIDATE"; break; fi
    done
fi
[[ -n "$PY" && -x "$PY" ]] || { echo "Missing HiSSD Python" >&2; exit 3; }

mkdir -p "$ROOT/logs" "$ROOT/pids" "$ROOT/artifacts"
exec 9>"$ROOT/pids/stage1_then_start_abc.lock"
flock -n 9 || { echo "Stage 1 supervisor is already running" >&2; exit 10; }
printf '%s\n' "$$" > "$ROOT/pids/stage1_supervisor.pid"

if [[ ! -s "$ARTIFACT_ROOT/30000/agent.th" ]]; then
    mapfile -t CHECKPOINTS < <(
        find "$ROOT/results" -type f -path '*/mto_stage1_official_seed2/*/models/30000/agent.th' \
            -printf '%h\n'
    )
    if [[ "${#CHECKPOINTS[@]}" -eq 0 ]]; then
        "$ROOT/scripts/run_official_stage1.sh" \
            0 "$STAGE1_RESUME_ROOT" > "$LOG" 2>&1 &
        STAGE1_PID=$!
        printf '%s\n' "$STAGE1_PID" > "$ROOT/pids/stage1_official.pid"
        if ! wait "$STAGE1_PID"; then
            echo "Official HiSSD Stage 1 failed; A/B/C will not start" >&2
            exit 11
        fi
        mapfile -t CHECKPOINTS < <(
            find "$ROOT/results" -type f -path '*/mto_stage1_official_seed2/*/models/30000/agent.th' \
                -printf '%h\n'
        )
    fi
    [[ "${#CHECKPOINTS[@]}" -eq 1 ]] || {
        echo "Expected one Stage 1 checkpoint, found ${#CHECKPOINTS[@]}" >&2
        exit 12
    }
    CHECKPOINT="${CHECKPOINTS[0]}"
    RUN_DIR="$(dirname "$(dirname "$CHECKPOINT")")"
    PYTHONPATH="$ROOT/src:$ROOT/smac:${PYTHONPATH:-}" \
        "$PY" "$ROOT/scripts/verify_official_checkpoint.py" \
        "$CHECKPOINT" "$RUN_DIR/config.json" > "$ROOT/logs/stage1_checkpoint_sha256.txt"
    mkdir -p "$ARTIFACT_ROOT"
    cp -a "$CHECKPOINT" "$ARTIFACT_ROOT/"
    cp "$ROOT/logs/stage1_checkpoint_sha256.txt" "$ARTIFACT_ROOT/30000/SHA256SUMS"
fi

cd "$ARTIFACT_ROOT/30000"
sha256sum -c SHA256SUMS
<<<<<<< HEAD
ssh -o BatchMode=yes noward \
    "mkdir -p '$REMOTE_ROOT/artifacts/official_hissd_seed2_30k'"
ssh -o BatchMode=yes noward2 \
    "mkdir -p '$REMOTE2_ROOT/artifacts/official_hissd_seed2_30k'"
rsync -a "$ARTIFACT_ROOT/" "noward:$REMOTE_ROOT/artifacts/official_hissd_seed2_30k/"
rsync -a "$ARTIFACT_ROOT/" "noward2:$REMOTE2_ROOT/artifacts/official_hissd_seed2_30k/"
ssh -o BatchMode=yes noward \
    "cd '$REMOTE_ROOT/artifacts/official_hissd_seed2_30k/30000' && sha256sum -c SHA256SUMS"
ssh -o BatchMode=yes noward2 \
=======
ssh -o BatchMode=yes "$REMOTE_HOST" \
    "mkdir -p '$REMOTE_ROOT/artifacts/official_hissd_seed2_30k'"
ssh -o BatchMode=yes "$REMOTE2_HOST" \
    "mkdir -p '$REMOTE2_ROOT/artifacts/official_hissd_seed2_30k'"
rsync -a "$ARTIFACT_ROOT/" "$REMOTE_HOST:$REMOTE_ROOT/artifacts/official_hissd_seed2_30k/"
rsync -a "$ARTIFACT_ROOT/" "$REMOTE2_HOST:$REMOTE2_ROOT/artifacts/official_hissd_seed2_30k/"
ssh -o BatchMode=yes "$REMOTE_HOST" \
    "cd '$REMOTE_ROOT/artifacts/official_hissd_seed2_30k/30000' && sha256sum -c SHA256SUMS"
ssh -o BatchMode=yes "$REMOTE2_HOST" \
>>>>>>> origin/main
    "cd '$REMOTE2_ROOT/artifacts/official_hissd_seed2_30k/30000' && sha256sum -c SHA256SUMS"

"$ROOT/scripts/start_strategy_background.sh" \
    encoder_frozen 0 A_encoder_frozen.log "$ARTIFACT_ROOT/30000"
<<<<<<< HEAD
ssh -o BatchMode=yes noward \
    "'$REMOTE_ROOT/scripts/start_strategy_background.sh' full_finetune 0 B_full_finetune.log '$REMOTE_ROOT/artifacts/official_hissd_seed2_30k/30000'"
ssh -o BatchMode=yes noward \
=======
ssh -o BatchMode=yes "$REMOTE_HOST" \
    "'$REMOTE_ROOT/scripts/start_strategy_background.sh' full_finetune 0 B_full_finetune.log '$REMOTE_ROOT/artifacts/official_hissd_seed2_30k/30000'"
ssh -o BatchMode=yes "$REMOTE_HOST" \
>>>>>>> origin/main
    "'$REMOTE_ROOT/scripts/start_strategy_background.sh' adapter_only 1 C_adapter_only.log '$REMOTE_ROOT/artifacts/official_hissd_seed2_30k/30000'"
