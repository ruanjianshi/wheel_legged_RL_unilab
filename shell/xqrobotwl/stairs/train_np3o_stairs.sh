#!/bin/bash
# ============================================================
# XqRobotWL NP3O 楼梯专项训练脚本
# 用法: bash shell/xqrobotwl/stairs/train_np3o_stairs.sh
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

TASK="xqrobotwl_stairs/mujoco"

$PYTHON scripts/training/train_np3o.py \
    task="${TASK}" \
    training.task_name=XqRobotWLStairs \
    "$@"
