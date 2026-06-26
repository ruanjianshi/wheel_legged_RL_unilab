#!/bin/bash
# ============================================================
# APPO (异步 PPO) 训练启动脚本
# 用法: bash shell/train/appo_mujoco.sh
# ============================================================
set -e

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

# ---- 训练配置 ----
TASK="go2w_joystick_flat/mujoco"
NUM_ENVS=4096
MAX_ITERATIONS=5000
# --------------------

uv run scripts/training/train_appo.py \
    task="${TASK}" \
    algo.num_envs="${NUM_ENVS}" \
    algo.max_iterations="${MAX_ITERATIONS}" \
    "$@"
