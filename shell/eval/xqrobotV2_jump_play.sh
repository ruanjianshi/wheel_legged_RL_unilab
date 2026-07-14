#!/bin/bash
# ============================================================
# XqRobotV2 Wheeled-SRL 跳跃策略验证脚本
# 用法:
#   bash shell/eval/xqrobotV2_jump_play.sh                # 最新模型
#   bash shell/eval/xqrobotV2_jump_play.sh <run_id>        # 指定 run
# ============================================================
set -e

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

LOAD_RUN=""

for arg in "$@"; do
    if [[ "$arg" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2} ]]; then
        LOAD_RUN="$arg"
    fi
done

CMD="uv run scripts/play/play_interactive.py"
CMD="$CMD --algo ppo --task xqrobotV2_jump_flat --sim mujoco"
CMD="$CMD interactive.action_mode=policy"
if [ -n "$LOAD_RUN" ]; then
    CMD="$CMD algo.load_run=${LOAD_RUN}"
fi

echo "Running: $CMD"
eval "$CMD"
