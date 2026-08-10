#!/usr/bin/env bash
# ============================================================
# 训练监控辅助 (CLAUDE.md §1.2: 每 1000 iter 自动提醒 + 异常捕获)
#
# 用法:
#   bash shell/xqrobotwl/tools/monitor_training.sh <训练日志路径>
#
# 示例 (训练在后台跑, 日志重定向到文件, monitor 前台挂载):
#   bash shell/xqrobotwl/fall_recovery/train_ppo_fall_recovery.sh \
#       > /tmp/xq_train_logs/fall_recovery.log 2>&1 &
#   bash shell/xqrobotwl/tools/monitor_training.sh /tmp/xq_train_logs/fall_recovery.log
#
# 行为:
#   - 每 1000 iter 打印提醒 (提示跑无辅助确定性评估: eval_*.py / dump_pose_data.py)
#   - 捕获异常/崩溃关键字 (Traceback / Error / OOM / Killed / NaN)
#   - tail -F 跟随文件轮转, Ctrl+C 退出
# ============================================================
set -u

LOG="${1:?用法: monitor_training.sh <训练日志路径>}"
if [ ! -f "$LOG" ]; then
    echo "[monitor] 找不到日志: $LOG" >&2
    exit 1
fi

echo "[monitor] 挂载 $LOG ... (每 1000 iter 提醒, Ctrl+C 退出)"
tail -F "$LOG" | while IFS= read -r line; do
    # 每 1000 iter 里程碑提醒
    if [[ "$line" =~ Learning[[:space:]]+iteration[[:space:]]+([0-9]+) ]]; then
        iter="${BASH_REMATCH[1]}"
        if (( iter > 0 && iter % 1000 == 0 )); then
            echo "[monitor] ★ ${iter} iter 达成 — 跑无辅助确定性评估:"
            echo "[monitor]     uv run tools/xqrobotwl/eval_*.py --run <run_dir> [--pose ...]"
            echo "[monitor]     uv run tools/xqrobotwl/dump_pose_data.py --run <run_dir> --ckpt model_${iter}.pt"
        fi
    fi
    # 异常/崩溃捕获
    if [[ "$line" =~ (Traceback|AssertionError|CUDA[[:space:]]out[[:space:]]of[[:space:]]memory|Killed|RuntimeError|NaN|nan) ]]; then
        echo "[monitor] ⚠ 检测到: $line"
    fi
done
