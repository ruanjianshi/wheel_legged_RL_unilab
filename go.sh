#!/bin/bash
# ============================================================
# XqRobotV2 一键启动: 激活环境 + 键盘评估
# 用法: bash go.sh
# ============================================================
set -e

cd "$(dirname "$0")"

# 激活 conda 环境
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate unilab

echo "环境: $(conda info --envs | grep '*' | awk '{print $1}')"
echo ""

bash shell/eval_ppo_flat.sh --keyboard
