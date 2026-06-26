#!/bin/bash
# ============================================================
# XqRobotV2 跳跃策略验证 (MuJoCo 原生窗口)
# 用法:
#   bash shell/eval_ppo_jump.sh                          # 最新模型, 策略回放
#   bash shell/eval_ppo_jump.sh <run_id>                  # 指定 run
# ============================================================
set -e

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

LOAD_RUN=""
ACTION_MODE="policy"

for arg in "$@"; do
    case "$arg" in
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
CMD="$CMD --algo ppo --task xqrobotV2_jump_flat --sim mujoco"
CMD="$CMD interactive.action_mode=${ACTION_MODE}"
if [ -n "$LOAD_RUN" ]; then
    CMD="$CMD algo.load_run=${LOAD_RUN}"
fi

echo "Running: $CMD"
eval "$CMD"
