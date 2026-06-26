#!/bin/bash
# ============================================================
# 模型导出脚本 (导出 ONNX 用于部署)
# 用法: bash shell/eval/export_onnx.sh
# ============================================================
set -e

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

# ---- 配置 ----
TASK="go2w_joystick_flat/mujoco"
LOAD_RUN="${1:-}"
# --------------

# 设置环境变量触发导出
EXPORT_POLICY=True

if [ -n "$LOAD_RUN" ]; then
    uv run scripts/training/train_rsl_rl.py \
        task="${TASK}" \
        algo.load_run="${LOAD_RUN}" \
        training.no_play=false \
        "$@"
else
    uv run scripts/training/train_rsl_rl.py \
        task="${TASK}" \
        training.no_play=false \
        "$@"
fi
