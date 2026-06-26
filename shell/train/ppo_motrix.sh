#!/bin/bash
# ============================================================
# PPO 训练启动脚本 (Motrix 后端, CPU 大规模并行)
# 用法: bash shell/train/ppo_motrix.sh
# ============================================================
set -e

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

# ---- 训练配置 ----
TASK="go2w_joystick_flat/motrix"
NUM_ENVS=16384
MAX_ITERATIONS=5000
# --------------------

uv run scripts/training/train_rsl_rl.py \
    task="${TASK}" \
    algo.num_envs="${NUM_ENVS}" \
    algo.max_iterations="${MAX_ITERATIONS}" \
    training.sim_backend="motrix" \
    "$@"
