#!/bin/bash
# ============================================================
# XqRobotWL 点足行走 PPO 训练脚本
# 用法: bash shell/xqrobotwl/toe_walk/train_ppo_toe_walk.sh
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

TASK="xqrobotwl_toe_walk_flat/mujoco"

$PYTHON scripts/training/train_rsl_rl.py \
    task="${TASK}" \
    training.task_name=XqRobotWLToeWalkFlat \
    "$@"
