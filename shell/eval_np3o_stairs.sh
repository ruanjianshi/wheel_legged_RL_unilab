#!/bin/bash
# ============================================================
# XqRobotV2 楼梯专项策略验证 (NP3O)
# 用法:
#   bash shell/eval_np3o_stairs.sh                          # 最新模型, 策略回放
#   bash shell/eval_np3o_stairs.sh --keyboard                # 最新模型, 键盘遥控
#   bash shell/eval_np3o_stairs.sh <run_id>                  # 指定 run, 策略回放
#   bash shell/eval_np3o_stairs.sh <run_id> --keyboard       # 指定 run, 键盘遥控
# ============================================================
set -e

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

LOAD_RUN=""
ACTION_MODE="policy"
KEYBOARD=""
HYDRA_OVERRIDES=()

for arg in "$@"; do
    case "$arg" in
        --keyboard)
            KEYBOARD=true
            ;;
        policy|zero)
            ACTION_MODE="$arg"
            ;;
        --*)
            echo "Unknown option: $arg"
            exit 1
            ;;
        *=*)
            HYDRA_OVERRIDES+=("$arg")
            ;;
        *)
            if [[ "$arg" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2} ]]; then
                LOAD_RUN="$arg"
            else
                ACTION_MODE="$arg"
            fi
            ;;
    esac
done

CMD="uv run scripts/play/play_interactive.py"
CMD="$CMD --algo np3o --task xqrobotV2_stairs --sim mujoco"
CMD="$CMD interactive.action_mode=${ACTION_MODE}"
if [ -n "$KEYBOARD" ]; then
    CMD="$CMD interactive.keyboard=true"
fi
if [ -n "$LOAD_RUN" ]; then
    CMD="$CMD algo.load_run=${LOAD_RUN}"
fi
for ov in "${HYDRA_OVERRIDES[@]}"; do
    CMD="$CMD $ov"
done

echo "Running: $CMD"
eval "$CMD"
