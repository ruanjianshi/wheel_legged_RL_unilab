#!/bin/bash
# ============================================================
# XqRobotWL Wheeled-SRL 跳跃 PPO 训练
# 用法: bash shell/train/xqrobotwl_jump_ppo.sh
# ============================================================
set -e
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"
TASK="xqrobotwl_jump_flat/mujoco"
uv run scripts/training/train_rsl_rl.py \
    task="${TASK}" \
    training.task_name=XqRobotWLJumpFlat \
    "$@"
