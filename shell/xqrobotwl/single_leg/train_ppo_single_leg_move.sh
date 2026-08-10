#!/bin/bash
# ============================================================
# XqRobotWL 可移动单轮平衡 PPO (独立 task, 从零训练)
# 用法:
#   bash shell/xqrobotwl/train_ppo_single_leg_move.sh           # 全量 10000 iter
#   bash shell/xqrobotwl/train_ppo_single_leg_move.sh quick      # 快速验证 500 iter
#   bash shell/xqrobotwl/train_ppo_single_leg_move.sh 1000       # 指定迭代数
#   bash shell/xqrobotwl/train_ppo_single_leg_move.sh resume     # 续训最新
# ============================================================
set -e

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT_DIR"

MODE="${1:-full}"
LOAD_RUN=""
case "$MODE" in
    quick)
        MAX_ITERS="algo.max_iterations=500"
        ;;
    full)
        MAX_ITERS=""
        ;;
    resume)
        LATEST=$(ls -dt logs/rsl_rl_ppo/XqRobotWLSingleLegMove/*/ 2>/dev/null | head -1)
        if [ -z "$LATEST" ]; then
            echo "无历史训练可续"
            exit 1
        fi
        LOAD_RUN="algo.load_run=$(basename "$LATEST")"
        MAX_ITERS=""
        ;;
    *)
        if [[ "$MODE" =~ ^[0-9]+$ ]]; then
            MAX_ITERS="algo.max_iterations=$MODE"
        else
            echo "Unknown option: $MODE (支持: quick / full / resume / 迭代数)"
            exit 1
        fi
        ;;
esac

TASK="xqrobotwl_single_leg_move/mujoco"

echo "=============================================="
echo "启动可移动单轮平衡训练 (从零, 前台): mode=$MODE ${LOAD_RUN:+($LOAD_RUN)}"
echo "TensorBoard: uv run tensorboard --logdir logs/rsl_rl_ppo/XqRobotWLSingleLegMove/"
echo "=============================================="

exec uv run scripts/training/train_rsl_rl.py \
    task="${TASK}" \
    training.task_name=XqRobotWLSingleLegMove \
    ${LOAD_RUN} \
    ${MAX_ITERS}
