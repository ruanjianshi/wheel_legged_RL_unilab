#!/bin/bash
# ============================================================
# XqRobotWL PPO 单腿平衡 (前台运行, 输出实时可见)
# 用法:
#   bash shell/xqrobotwl/launch_ppo_single_leg.sh              # 全量 10000 iter
#   bash shell/xqrobotwl/launch_ppo_single_leg.sh quick         # 快速验证 500 iter
#   bash shell/xqrobotwl/launch_ppo_single_leg.sh 1000          # 指定迭代数
#   bash shell/xqrobotwl/launch_ppo_single_leg.sh warmstart     # 热启动 walk 平衡
#   bash shell/xqrobotwl/launch_ppo_single_leg.sh resume        # 续训最新
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
    warmstart)
        # 热启动: 从 walk_flat 平衡模型继承两轮平衡俯仰能力, 只学横滚
        LOAD_RUN="algo.load_run=warmstart_from_walk"
        MAX_ITERS=""
        ;;
    resume)
        LATEST=$(ls -dt logs/rsl_rl_ppo/XqRobotWLSingleLegFlat/*/ 2>/dev/null | head -1)
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
            echo "Unknown option: $MODE (支持: quick / full / warmstart / resume / 迭代数)"
            exit 1
        fi
        ;;
esac

TASK="xqrobotwl_single_leg_flat/mujoco"

echo "=============================================="
echo "启动单腿平衡训练 (前台): mode=$MODE ${LOAD_RUN:+($LOAD_RUN)}"
echo "Ctrl-C 停止后, checkpoint 会自动保存可续训"
echo "TensorBoard: uv run tensorboard --logdir logs/rsl_rl_ppo/XqRobotWLSingleLegFlat/"
echo "=============================================="

exec uv run scripts/training/train_rsl_rl.py \
    task="${TASK}" \
    training.task_name=XqRobotWLSingleLegFlat \
    ${LOAD_RUN} \
    ${MAX_ITERS}
