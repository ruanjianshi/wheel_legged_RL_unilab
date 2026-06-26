#!/bin/bash
# ============================================================
# XqRobotV2 策略验证脚本
# macOS: play_viser (浏览器可视化, keyboard 模式无遥控)
# Linux: play_interactive (原生窗口键盘控制)
# 用法:
#   bash shell/eval/xqrobotV2_play.sh                    # 最新模型
#   bash shell/eval/xqrobotV2_play.sh <run_id>
#   bash shell/eval/xqrobotV2_play.sh <run_id> policy
# ============================================================
set -e

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

LOAD_RUN=""
ACTION_MODE="policy"

if [[ "$1" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2} ]]; then
    LOAD_RUN="$1"
    ACTION_MODE="${2:-policy}"
else
    ACTION_MODE="${1:-policy}"
fi

if [[ "$(uname)" == "Darwin" ]]; then
    # macOS: viser 浏览器 (mujoco-uni 无原生窗口)
    CMD="uv run scripts/play/play_viser.py"
    CMD="$CMD task=xqrobotV2_walk_flat/mujoco algo.num_envs=1"
    CMD="$CMD interactive.action_mode=${ACTION_MODE}"
    if [ -n "$LOAD_RUN" ]; then
        CMD="$CMD algo.load_run=${LOAD_RUN}"
    fi
    echo "http://localhost:8080"
else
    CMD="uv run scripts/play/play_interactive.py"
    CMD="$CMD --algo ppo --task xqrobotV2_walk_flat --sim mujoco"
    CMD="$CMD interactive.action_mode=${ACTION_MODE}"
    if [ -n "$LOAD_RUN" ]; then
        CMD="$CMD algo.load_run=${LOAD_RUN}"
    fi
fi

echo "Running: $CMD"
eval "$CMD"
