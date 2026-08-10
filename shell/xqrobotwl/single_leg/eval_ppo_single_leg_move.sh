#!/bin/bash
# ============================================================
# XqRobotWL 单腿移动策略验证 (MuJoCo 原生窗口 + 键盘控制)
# 用法:
#   bash shell/xqrobotwl/single_leg/eval_ppo_single_leg_move.sh            # 最新 run
#   bash shell/xqrobotwl/single_leg/eval_ppo_single_leg_move.sh --keyboard
#   bash shell/xqrobotwl/single_leg/eval_ppo_single_leg_move.sh 2000       # 指定 checkpoint
#   bash shell/xqrobotwl/single_leg/eval_ppo_single_leg_move.sh <run_id>
# ============================================================
ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT_DIR"
set -e

# Platform detection: use mjpython on macOS, uv run on Linux
if [[ "$(uname)" == "Darwin" ]]; then
    PYTHON="uv run mjpython"
else
    PYTHON="uv run"
fi

LOAD_RUN=""
ACTION_MODE="policy"
KEYBOARD=""
CKPT=""
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
            elif [[ "$arg" =~ ^[0-9]+$ ]]; then
                CKPT="$arg"
            else
                ACTION_MODE="$arg"
            fi
            ;;
    esac
done

CMD="$PYTHON scripts/play/play_interactive.py"
CMD="$CMD --algo ppo --task xqrobotwl_single_leg_move --sim mujoco"
CMD="$CMD interactive.action_mode=${ACTION_MODE}"
if [ -n "$KEYBOARD" ]; then
    # 命令 5D, 跳过速度命令 obs 检查 (与其它 5D 命令任务一致)
    CMD="$CMD interactive.keyboard=true +interactive.require_keyboard_command_obs=false"
fi
if [ -n "$LOAD_RUN" ]; then
    CMD="$CMD algo.load_run=${LOAD_RUN}"
fi
if [ -n "$CKPT" ]; then
    CMD="$CMD algo.checkpoint=${CKPT}"
fi
for ov in "${HYDRA_OVERRIDES[@]}"; do
    CMD="$CMD $ov"
done

echo "Running: $CMD"
eval "$CMD"
