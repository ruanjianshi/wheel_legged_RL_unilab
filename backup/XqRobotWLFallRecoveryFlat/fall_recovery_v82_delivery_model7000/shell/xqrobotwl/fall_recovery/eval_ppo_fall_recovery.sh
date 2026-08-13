#!/bin/bash
# ============================================================
# XqRobotWL 跌倒恢复策略验证 (CPO 训练, MuJoCo 原生窗口 + 键盘控制)
# 用法:
#   bash shell/xqrobotwl/fall_recovery/eval_ppo_fall_recovery.sh          # 最新模型, 策略回放
#   bash shell/xqrobotwl/fall_recovery/eval_ppo_fall_recovery.sh --keyboard
#   bash shell/xqrobotwl/fall_recovery/eval_ppo_fall_recovery.sh <run_id> --keyboard
# ============================================================
set -e

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
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

# Platform detection: use mjpython on macOS, uv run on Linux
if [[ "$(uname)" == "Darwin" ]]; then
    PYTHON="uv run mjpython"
else
    PYTHON="uv run"
fi

CMD="$PYTHON scripts/play/play_interactive.py"
CMD="$CMD --algo ppo --task xqrobotwl_fall_recovery_flat --sim mujoco"
CMD="$CMD interactive.action_mode=${ACTION_MODE}"
if [ -n "$KEYBOARD" ]; then
    # 命令 5D [vx,vy,vyaw,tsk,h], 跳过速度命令 obs 检查 (与其它 5D 命令任务一致)
    CMD="$CMD interactive.keyboard=true +interactive.require_keyboard_command_obs=false"
fi
if [ -n "$LOAD_RUN" ]; then
    CMD="$CMD algo.load_run=${LOAD_RUN}"
fi
for ov in "${HYDRA_OVERRIDES[@]}"; do
    CMD="$CMD $ov"
done

echo "Running: $CMD"
eval "$CMD"
