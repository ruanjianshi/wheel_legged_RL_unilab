#!/bin/bash
# ============================================================
# PPO 训练启动脚本 (MuJoCo 后端)
# 用法: bash shell/train/ppo_mujoco.sh
# ============================================================
set -e

# 项目根目录
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

# ---- 训练配置 (按需修改) ----
TASK="go2w_joystick_flat/mujoco"   # 替换为你的 task
NUM_ENVS=4096
MAX_ITERATIONS=5000
# ------------------------------

uv run scripts/training/train_rsl_rl.py \
    task="${TASK}" \
    algo.num_envs="${NUM_ENVS}" \
    algo.max_iterations="${MAX_ITERATIONS}" \
    training.sim_backend="mujoco" \
    "$@"
