#!/bin/bash
# MPC×SAC 融合控制训练 (独立任务轨)
# 用法: bash shell/xqrobotwl/fusion_control/mpc_sac/train_mpc_sac.sh [walk_flat|walk_rough] [max_iterations]
ROOT_DIR="$(cd "$(dirname "$0")/../../../../.." && pwd)"
cd "$ROOT_DIR"; set -e
PYTHON="uv run"; [[ "$(uname)" == "Darwin" ]] && PYTHON="uv run mjpython"
$PYTHON scripts/fusion_control/mpc_sac/train_mpc_sac.py --task "${1:-walk_flat}" --max_iterations "${2:-3000}"
