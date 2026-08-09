#!/bin/bash
# ============================================================
# XqRobotWL 改进结构单轮平衡策略验证 (MuJoCo 原生窗口 + 键盘控制)
# 用法:
#   bash shell/xqrobotwl/eval_ppo_single_leg_unicycle.sh               # 最新 run, model_4000 (最佳)
#   bash shell/xqrobotwl/eval_ppo_single_leg_unicycle.sh --keyboard    # 键盘遥控
#   bash shell/xqrobotwl/eval_ppo_single_leg_unicycle.sh 2000          # 指定 checkpoint
#   bash shell/xqrobotwl/eval_ppo_single_leg_unicycle.sh <run_id>      # 指定 run
#   # 指定 checkpoint (推荐: 最佳 model_3000/4000 满 8s, model_4999 已退化):
#   bash shell/xqrobotwl/eval_ppo_single_leg_unicycle.sh <run_id> --keyboard algo.checkpoint=3000
# ============================================================
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
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
CKPT="4000"
HYDRA_OVERRIDES=()

for arg in "$@"; do
    case "$arg" in
        --keyboard)
            KEYBOARD=true
            ;;
        --latest)
            CKPT=""  # 用最新 checkpoint (注意: model_4999 已退化)
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
CMD="$CMD --algo ppo --task xqrobotwl_single_leg_unicycle --sim mujoco"
CMD="$CMD interactive.action_mode=${ACTION_MODE}"
if [ -n "$KEYBOARD" ]; then
    # 命令 5D [vx,vy,vyaw,tsk,h], play_interactive 的 3D 探测命令会破坏 obs 帧,
    # 跳过速度命令 obs 检查 (与 eval_ppo_backflip.sh 同)
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
