#!/bin/bash
# ============================================================
# XqRobotV2 两轮腿机器人 PPO 训练脚本
# 用法: bash shell/train/xqrobotV2_ppo.sh
# ============================================================
set -e

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

# ---- 训练配置 ----
TASK="xqrobotV2_walk_flat/mujoco"
# --------------------

uv run scripts/training/train_rsl_rl.py \
    task="${TASK}" \
    training.task_name=XqRobotV2WalkFlat \
    "$@"
