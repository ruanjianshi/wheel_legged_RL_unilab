#!/bin/bash
# ============================================================
# XqRobotV2 PPO 跳跃训练脚本
# 用法: bash shell/train_ppo_jump.sh
# ============================================================
set -e

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

TASK="xqrobotV2_jump_flat/mujoco"

uv run scripts/training/train_rsl_rl.py \
    task="${TASK}" \
    training.task_name=XqRobotV2JumpFlat \
    "$@"
