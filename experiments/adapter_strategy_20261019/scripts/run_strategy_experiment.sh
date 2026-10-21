#!/usr/bin/env bash
set -euo pipefail

STRATEGY="${1:?Usage: run_strategy_experiment.sh STRATEGY GPU CHECKPOINT_OR_DASH}"
GPU="${2:?GPU index is required}"
CHECKPOINT="${3:--}"
RESUME_ROOT="${4:-}"
EXTRA_ARGS=()

case "$STRATEGY" in
    encoder_frozen|full_finetune|adapter_only|joint_scratch) ;;
    *) echo "Unknown strategy: $STRATEGY" >&2; exit 2 ;;
esac
[[ "$GPU" =~ ^[0-9]+$ ]] || { echo "Invalid GPU: $GPU" >&2; exit 2; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$STRATEGY" == "joint_scratch" ]]; then
    [[ "$CHECKPOINT" == "-" ]] || {
        echo "joint_scratch must not receive a checkpoint" >&2
        exit 2
    }
    CHECKPOINT=""
    if [[ -n "$RESUME_ROOT" ]]; then
        RESUME_ROOT="$(realpath "$RESUME_ROOT")"
        [[ -d "$RESUME_ROOT" ]] || {
            echo "Missing joint-scratch resume root: $RESUME_ROOT" >&2
            exit 3
        }
        EXTRA_ARGS+=(--checkpoint_path="$RESUME_ROOT" --load_step=0)
    fi
else
    [[ -z "$RESUME_ROOT" ]] || {
        echo "$STRATEGY does not accept a training resume root" >&2
        exit 2
    }
    CHECKPOINT="$(realpath "$CHECKPOINT")"
    for FILE in agent.th mixer.th; do
        [[ -s "$CHECKPOINT/$FILE" ]] || {
            echo "Missing official checkpoint file: $CHECKPOINT/$FILE" >&2
            exit 3
        }
    done
fi

PY="${HISSD_PYTHON:-}"
if [[ -z "$PY" ]]; then
    for CANDIDATE in \
        "$HOME/HiSSD_capability_eval/envs/hissd/bin/python" \
        "$HOME/anaconda3/envs/hissd/bin/python"; do
        if [[ -x "$CANDIDATE" ]]; then PY="$CANDIDATE"; break; fi
    done
fi
[[ -n "$PY" && -x "$PY" ]] || { echo "Missing HiSSD Python" >&2; exit 3; }
[[ -e "$ROOT/dataset" ]] || { echo "Missing dataset: $ROOT/dataset" >&2; exit 3; }

SC2_ROOT="${HISSD_SC2PATH:-}"
if [[ -z "$SC2_ROOT" ]]; then
    for CANDIDATE in "$HOME/StarCraftII" "$ROOT/../../3rdparty/StarCraftII"; do
        if [[ -d "$CANDIDATE" ]]; then SC2_ROOT="$CANDIDATE"; break; fi
    done
fi
[[ -n "$SC2_ROOT" && -d "$SC2_ROOT" ]] || { echo "Missing StarCraftII" >&2; exit 3; }

GPU_UUID="$(nvidia-smi --id="$GPU" --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')"
[[ -n "$GPU_UUID" ]] || { echo "GPU $GPU not found" >&2; exit 4; }
while IFS=',' read -r UUID PID; do
    UUID="${UUID//[[:space:]]/}"
    PID="${PID//[[:space:]]/}"
    [[ "$UUID" == "$GPU_UUID" && "$PID" =~ ^[0-9]+$ ]] || continue
    CMD="$(ps -p "$PID" -o args= 2>/dev/null || true)"
    [[ "$CMD" == *"/usr/share/code/code --type=gpu-process"* ]] && continue
    echo "GPU $GPU is busy: pid=$PID cmd=$CMD" >&2
    exit 5
done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null)

cd "$ROOT"
export SC2PATH="$SC2_ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/smac:${PYTHONPATH:-}"
export HISSD_EVAL_WORKERS=10

exec env CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u src/main.py --mto \
    --config=hissd_capability_team_noobs \
    --env-config=sc2_offline \
    --task-config=stalker-zealot-expert \
    --seed=2 --t_max=30000 --pretrain=False \
    --optim_type=AdamW --lr=0.0003 --critic_lr=0.0001 --weight_decay=0.0001 \
    --save_model=True --save_model_interval=10000 \
    --test_nepisode=100 --test_interval=1000 \
    --common_skill_adapter_enabled=True \
    --adapter_training_strategy="$STRATEGY" \
    --stage2_init_checkpoint="$CHECKPOINT" \
    --fused_adamw=True --cuda_matmul_precision=high \
    --v3_log_interval_steps=100 --progress_bar=True \
    --remark="$STRATEGY" \
    "${EXTRA_ARGS[@]}"
