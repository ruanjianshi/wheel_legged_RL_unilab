#!/bin/bash
# ============================================================
# XqRobotWL Wheeled-SRL PPO 跳跃训练脚本
# 用法:
#   bash shell/xqrobotwl/jump/train_ppo_jump_srl.sh                   # 完整版
#   bash shell/xqrobotwl/jump/train_ppo_jump_srl.sh ablation=no_fsm    # 消融1
#   bash shell/xqrobotwl/jump/train_ppo_jump_srl.sh ablation=no_wheel_match
# ============================================================
set -e

# Platform detection: use mjpython on macOS, uv run on Linux
if [[ "$(uname)" == "Darwin" ]]; then
    PYTHON="uv run mjpython"
else
    PYTHON="uv run"
fi

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

TASK="xqrobotwl_jump_srl_flat/mujoco"

$PYTHON scripts/training/train_rsl_rl.py \
    task="${TASK}" \
    training.task_name=XqRobotWLJumpSRLFlat \
    "$@"
