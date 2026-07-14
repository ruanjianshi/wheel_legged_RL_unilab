#!/bin/bash
# ============================================================
# XqRobotV2 NP3O 楼梯专项训练脚本
# 用法: bash shell/train_np3o_stairs.sh
# ============================================================
set -e

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

TASK="xqrobotV2_stairs/mujoco"

uv run scripts/training/train_np3o.py \
    task="${TASK}" \
    training.task_name=XqRobotV2Stairs \
    "$@"
