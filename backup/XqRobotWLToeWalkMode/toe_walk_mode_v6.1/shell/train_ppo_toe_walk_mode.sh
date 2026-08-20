#!/bin/bash
# ============================================================
# XqRobotWL 双模式点足行走 PPO 训练脚本 (站立 ⇄ 点足抬腿, mode 命令切换)
# 用法: bash shell/xqrobotwl/toe_walk_mode/train_ppo_toe_walk_mode.sh
# ============================================================
set -e

# Platform detection: use mjpython on macOS, uv run on Linux
if [[ "$(uname)" == "Darwin" ]]; then
    PYTHON="uv run mjpython"
else
    PYTHON="uv run"
fi

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT_DIR"

TASK="xqrobotwl_toe_walk_mode/mujoco"

$PYTHON scripts/training/train_rsl_rl.py \
    task="${TASK}" \
    training.task_name=XqRobotWLToeWalkMode \
    "$@"