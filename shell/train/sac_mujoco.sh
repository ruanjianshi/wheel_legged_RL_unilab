#!/bin/bash
# ============================================================
# SAC 训练启动脚本 (off-policy)
# 用法: bash shell/train/sac_mujoco.sh
# ============================================================
set -e

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

# ---- 训练配置 ----
TASK="sac/g1_walk_flat/mujoco"
NUM_ENVS=256
MAX_ITERATIONS=10000
# --------------------

uv run scripts/training/train_offpolicy.py \
    algo=sac \
    task="${TASK}" \
    algo.num_envs="${NUM_ENVS}" \
    algo.max_iterations="${MAX_ITERATIONS}" \
    "$@"
