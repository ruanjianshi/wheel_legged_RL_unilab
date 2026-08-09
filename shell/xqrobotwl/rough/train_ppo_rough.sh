#!/bin/bash
# ============================================================
# XqRobotWL PPO 粗糙地形训练脚本
# 用法: bash shell/xqrobotwl/rough/train_ppo_rough.sh
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

TASK="xqrobotwl_walk_rough/mujoco"

$PYTHON scripts/training/train_rsl_rl.py \
    task="${TASK}" \
    training.task_name=XqRobotWLWalkRough \
    "$@"
