#!/bin/bash
# ============================================================
# XqRobotWL PPO 后空翻训练脚本
# 用法: bash shell/xqrobotwl/train_ppo_backflip.sh
# ============================================================
set -e

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

TASK="xqrobotwl_backflip_flat/mujoco"

uv run scripts/training/train_rsl_rl.py \
    task="${TASK}" \
    training.task_name=XqRobotWLBackflipFlat \
    "$@"
