#!/bin/bash
# ============================================================
# XqRobotV2 PPO 点足行走训练脚本 (平坦地形)
# 用法: bash shell/train_ppo_toe_walk_flat.sh
# ============================================================
set -e

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

TASK="xqrobotV2_toe_walk_flat/mujoco"

uv run scripts/training/train_rsl_rl.py \
    task="${TASK}" \
    training.task_name=XqRobotV2ToeWalkFlat \
    "$@"
