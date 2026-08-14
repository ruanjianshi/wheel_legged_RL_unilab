#!/bin/bash
# LQR 批量评估 (独立任务轨)
# 用法: bash shell/xqrobotwl/classic_lqr/eval_lqr.sh 1     # P1
#       bash shell/xqrobotwl/classic_lqr/eval_lqr.sh all   # 全阶段
ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT_DIR"; set -e
PYTHON="uv run"; [[ "$(uname)" == "Darwin" ]] && PYTHON="uv run mjpython"
if [ "$1" = "all" ]; then
  $PYTHON scripts/classic_control/lqr/eval_lqr.py --phases 1 2 3 4 --episodes 5 --out logs/classic/report_lqr.md
else
  $PYTHON scripts/classic_control/lqr/balance_lqr.py --phase "${1:-1}"
fi
