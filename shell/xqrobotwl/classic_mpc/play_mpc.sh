#!/bin/bash
# MPC 平衡交互控制 (独立任务轨)
# 用法: bash shell/xqrobotwl/classic_mpc/play_mpc.sh [flat|rough] [speed]
ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT_DIR"; set -e
PYTHON="uv run"; [[ "$(uname)" == "Darwin" ]] && PYTHON="uv run mjpython"
$PYTHON scripts/classic_control/mpc/play_mpc.py --terrain "${1:-flat}" --speed "${2:-1.0}"
