#!/bin/bash
# 自循环训练结束后的评估 + 邮件报告一键发送
# 用法:
#   bash tools/email/send.sh <task> <algo> <run> <ckpt> [<to>]
# 示例:
#   SMTP_USER=x15347348975 SMTP_PASS=x15347348975 \
#     bash tools/email/send.sh flat_walk ppo 2026-07-01_13-55-35_mujoco 20000

set -euo pipefail

TASK="${1:-flat_walk}"
ALGO="${2:-ppo}"
RUN="${3:?请提供 run ID}"
CKPT="${4:?请提供 checkpoint iter}"
TO="${5:-qfantastic@2925.com}"

echo "=== Assess → 邮件报告 ==="
echo "  task=$TASK  algo=$ALGO  run=$RUN  ckpt=$CKPT  to=$TO"
echo ""

cd "$(dirname "$0")/../.."

# Step 1: 评估 (如果还没跑过)
echo "[1/2] 运行评估..."
uv run assess/runner.py -t "$TASK" -a "$ALGO" -r "$RUN" -c "$CKPT" -s full

# Step 2: 生成报告并发送
echo ""
echo "[2/2] 发送邮件报告..."
uv run tools/email/report.py \
    -t "$TASK" -a "$ALGO" -r "$RUN" -c "$CKPT" \
    --to "$TO"

echo ""
echo "完成。"