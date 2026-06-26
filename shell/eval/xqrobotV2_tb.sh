#!/bin/bash
# ============================================================
# 查看 XqRobotV2 训练数据 (TensorBoard)
# 用法:
#   bash shell/eval/xqrobotV2_tb.sh              # 默认端口 6006
#   bash shell/eval/xqrobotV2_tb.sh 8080          # 指定端口
# ============================================================
set -e

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

PORT="${1:-6006}"
LOG_DIR="logs/rsl_rl_ppo/XqRobotV2WalkFlat"

if [ ! -d "$LOG_DIR" ]; then
    echo "日志目录不存在: $LOG_DIR"
    echo "先运行训练: bash shell/train/xqrobotV2_ppo.sh"
    exit 1
fi

echo "TensorBoard 已启动 → http://localhost:${PORT}"
echo "按 Ctrl+C 停止"
uv run tensorboard --logdir "${LOG_DIR}" --port "${PORT}" --bind_all
