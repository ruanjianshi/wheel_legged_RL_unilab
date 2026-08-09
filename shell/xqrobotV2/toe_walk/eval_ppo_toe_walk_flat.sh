#!/bin/bash
# ============================================================
# XqRobotV2 点足行走策略验证 (平坦地形 + 键盘控制)
# 用法:
#   bash shell/xqrobotV2/toe_walk/eval_ppo_toe_walk_flat.sh                          # 最新模型
#   bash shell/xqrobotV2/toe_walk/eval_ppo_toe_walk_flat.sh --keyboard                # 键盘遥控
#   bash shell/xqrobotV2/toe_walk/eval_ppo_toe_walk_flat.sh <run_id> --keyboard
# ============================================================
set -e

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT_DIR"

LOAD_RUN=""
ACTION_MODE="policy"
KEYBOARD=""

for arg in "$@"; do
    case "$arg" in
        --keyboard)
            KEYBOARD=true
            ;;
        policy|zero)
            ACTION_MODE="$arg"
            ;;
        -*)
            echo "Unknown option: $arg"
            exit 1
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
CMD="$CMD --algo ppo --task xqrobotV2_toe_walk_flat --sim mujoco"
CMD="$CMD interactive.action_mode=${ACTION_MODE}"
if [ -n "$KEYBOARD" ]; then
    CMD="$CMD interactive.keyboard=true"
fi
if [ -n "$LOAD_RUN" ]; then
    CMD="$CMD algo.load_run=${LOAD_RUN}"
fi

echo "Running: $CMD"
eval "$CMD"
