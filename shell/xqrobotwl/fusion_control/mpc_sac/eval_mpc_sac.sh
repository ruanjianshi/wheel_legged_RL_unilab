#!/bin/bash
# MPC×SAC 融合控制评估 (独立任务轨)
# 用法: bash shell/xqrobotwl/fusion_control/mpc_sac/eval_mpc_sac.sh walk_flat <ckpt>
ROOT_DIR="$(cd "$(dirname "$0")/../../../../.." && pwd)"
cd "$ROOT_DIR"; set -e
PYTHON="uv run"; [[ "$(uname)" == "Darwin" ]] && PYTHON="uv run mjpython"
$PYTHON scripts/fusion_control/mpc_sac/eval_mpc_sac.py --task "${1:-walk_flat}" --checkpoint "${2:?需要 ckpt 路径}" --episodes 5 --out "logs/fusion_control/mpc_sac/report_${1:-walk_flat}.md"
