#!/bin/bash
# ============================================================
# XqRobotWL Wheeled-SRL PPO 跳跃训练脚本
# 用法:
#   bash shell/xqrobotwl/train_ppo_jump_srl.sh                   # 完整版
#   bash shell/xqrobotwl/train_ppo_jump_srl.sh ablation=no_fsm    # 消融1
#   bash shell/xqrobotwl/train_ppo_jump_srl.sh ablation=no_wheel_match
# ============================================================
set -e

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

TASK="xqrobotwl_jump_srl_flat/mujoco"

uv run scripts/training/train_rsl_rl.py \
    task="${TASK}" \
    training.task_name=XqRobotWLJumpSRLFlat \
    "$@"
