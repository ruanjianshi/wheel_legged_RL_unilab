#!/bin/bash
# LQR 平衡交互控制 (独立任务轨)
# 用法: bash shell/xqrobotwl/classic_lqr/play_lqr.sh [flat|rough] [speed]
ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT_DIR"; set -e
PYTHON="uv run"; [[ "$(uname)" == "Darwin" ]] && PYTHON="uv run mjpython"
$PYTHON scripts/classic_control/lqr/play_lqr.py --terrain "${1:-flat}" --speed "${2:-1.0}"
