#!/bin/bash
# ============================================================
# 策略验证/回放脚本 (MuJoCo 可视化)
# 用法: bash shell/eval/play_policy.sh [load_run]
# 示例: bash shell/eval/play_policy.sh 2026-06-24_12-00-00
# ============================================================
set -e

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

# ---- 配置 ----
TASK="go2w_joystick_flat/mujoco"
LOAD_RUN="${1:-}"          # 可选: 指定 run 时间戳
# --------------

if [ -n "$LOAD_RUN" ]; then
    uv run scripts/play/play_interactive.py \
        task="${TASK}" \
        algo.load_run="${LOAD_RUN}" \
        interactive.action_mode=policy \
        "$@"
else
    uv run scripts/play/play_interactive.py \
        task="${TASK}" \
        interactive.action_mode=policy \
        "$@"
fi
