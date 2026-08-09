#!/bin/bash
# ============================================================
# XqRobotWL 后空翻策略验证 (MuJoCo 原生窗口 + 键盘控制)
# 用法:
#   bash shell/xqrobotwl/eval_ppo_backflip.sh              # 最新模型, 策略回放
#   bash shell/xqrobotwl/eval_ppo_backflip.sh --keyboard    # 最新模型, 键盘遥控
#   bash shell/xqrobotwl/eval_ppo_backflip.sh <run_id> --keyboard
#   # 指定 checkpoint (推荐: detflip 交付模型是 model_1000, 最新 model_1999 已发散不会翻):
#   bash shell/xqrobotwl/eval_ppo_backflip.sh 2026-08-05_18-35-40_mujoco__detflip --keyboard algo.checkpoint=1000
# ============================================================
set -e

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
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
CMD="$CMD --algo ppo --task xqrobotwl_backflip_flat --sim mujoco"
CMD="$CMD interactive.action_mode=${ACTION_MODE}"
if [ -n "$KEYBOARD" ]; then
    # backflip 命令 vx/vy/vyaw=0 (原地翻转, 第5维才是触发器), 跳过速度命令 obs 检查
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
