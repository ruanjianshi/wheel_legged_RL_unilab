#!/bin/bash
# ============================================================
# XqRobotV2 TensorBoard 训练曲线查看
# 用法:
#   bash shell/tensorboard.sh                  # 查看所有已配置的日志
#   bash shell/tensorboard.sh flat             # 只查看平坦地形
#   bash shell/tensorboard.sh rough            # 只查看粗糙地形
#   bash shell/tensorboard.sh flat 8080        # 指定端口
# ============================================================
set -e

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

# ============================================================
# ★ 日志路径: 将需要查看的任务 log 路径填入下方
#   新任务训练后, 取消注释或追加即可
# ============================================================
FLAT_LOG="logs/rsl_rl_ppo/XqRobotV2WalkFlat"
ROUGH_LOG="logs/rsl_rl_ppo/XqRobotV2WalkRough"
JUMP_FLAT_LOG="logs/rsl_rl_ppo/XqRobotV2JumpFlat"
TOE_WALK_FLAT_LOG="logs/rsl_rl_ppo/XqRobotV2ToeWalkFlat"
# TOE_WALK_ROUGH_LOG="logs/rsl_rl_ppo/XqRobotV2ToeWalkRough"  # 后续
# ============================================================

MODE="${1:-all}"
PORT="${2:-6006}"

declare -a LOG_DIRS=()
case "$MODE" in
    flat)   LOG_DIRS+=("$FLAT_LOG") ;;
    rough)  LOG_DIRS+=("$ROUGH_LOG") ;;
    jump)   LOG_DIRS+=("$JUMP_FLAT_LOG") ;;
    toe)    LOG_DIRS+=("$TOE_WALK_FLAT_LOG") ;;
    all)
        [ -n "$FLAT_LOG" ]  && LOG_DIRS+=("$FLAT_LOG")
        [ -n "$ROUGH_LOG" ] && LOG_DIRS+=("$ROUGH_LOG")
        [ -n "$JUMP_FLAT_LOG" ]  && LOG_DIRS+=("$JUMP_FLAT_LOG")
        [ -n "$TOE_WALK_FLAT_LOG" ] && LOG_DIRS+=("$TOE_WALK_FLAT_LOG")
        ;;
    *)
        echo "用法: bash shell/tensorboard.sh [flat|rough|jump|toe|all] [port]"
        exit 1
        ;;
esac

for dir in "${LOG_DIRS[@]}"; do
    if [ ! -d "$dir" ]; then
        echo "日志目录不存在: $dir"
    fi
done

LOG_DIR_ARG=$(IFS=,; echo "${LOG_DIRS[*]}")

echo "TensorBoard 已启动 → http://localhost:${PORT}"
echo "模式: ${MODE}"
echo "按 Ctrl+C 停止"
uv run tensorboard --logdir_spec "${LOG_DIR_ARG}" --port "${PORT}" --bind_all
