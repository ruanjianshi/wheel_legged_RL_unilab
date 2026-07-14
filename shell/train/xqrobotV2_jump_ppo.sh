#!/bin/bash
# ============================================================
# XqRobotV2 Wheeled-SRL 跳跃训练脚本
# ============================================================
set -e

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

# ---- 训练配置 ----
# 使用跳跃专用配置: conf/ppo/task/xqrobotV2_jump_flat/mujoco.yaml
TASK="xqrobotV2_jump_flat/mujoco"
# --------------------

uv run scripts/training/train_rsl_rl.py \
    task="${TASK}" \
    training.task_name=XqRobotV2JumpFlat \
    "$@"
