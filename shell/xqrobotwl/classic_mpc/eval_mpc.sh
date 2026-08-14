#!/bin/bash
# MPC 批量评估 (独立任务轨)
# 用法: bash shell/xqrobotwl/classic_mpc/eval_mpc.sh 1     # P1
#       bash shell/xqrobotwl/classic_mpc/eval_mpc.sh all   # 全阶段
ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT_DIR"; set -e
PYTHON="uv run"; [[ "$(uname)" == "Darwin" ]] && PYTHON="uv run mjpython"
if [ "$1" = "all" ]; then
  $PYTHON scripts/classic_control/mpc/eval_mpc.py --phases 1 2 3 4 --episodes 5 --out logs/classic/report_mpc.md
else
  $PYTHON scripts/classic_control/mpc/balance_mpc.py --phase "${1:-1}"
fi
