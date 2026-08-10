#!/bin/bash
# ============================================================
# XqRobotWL PPO 后空翻训练 (前台运行, 输出实时可见)
# 用法:
#   bash shell/xqrobotwl/launch_ppo_backflip.sh              # 全量 10000 iter
#   bash shell/xqrobotwl/launch_ppo_backflip.sh quick         # 快速验证 200 iter
#   bash shell/xqrobotwl/launch_ppo_backflip.sh 1000          # 指定迭代数
# ============================================================
set -e

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

MODE="${1:-full}"
LOAD_RUN=""
case "$MODE" in
    quick)
        MAX_ITERS="algo.max_iterations=200"
        ;;
    full)
        MAX_ITERS=""
        ;;
    warmstart)
        # 热启动: 从 walk_flat 平衡模型继承能力, 只学翻转
        LOAD_RUN="algo.load_run=warmstart_from_walk"
        MAX_ITERS=""
        ;;
    resume)
        # 续训: 从最新 checkpoint 继续 (可配合奖励调整)
        LATEST=$(ls -dt logs/rsl_rl_ppo/XqRobotWLBackflipFlat/*/ 2>/dev/null | head -1)
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

TASK="xqrobotwl_backflip_flat/mujoco"

echo "=============================================="
echo "启动后空翻训练 (前台): mode=$MODE ${LOAD_RUN:+($LOAD_RUN)}"
echo "Ctrl-C 停止后, checkpoint 会自动保存可续训"
echo "TensorBoard: uv run tensorboard --logdir logs/rsl_rl_ppo/XqRobotWLBackflipFlat/"
echo "=============================================="

exec uv run scripts/training/train_rsl_rl.py \
    task="${TASK}" \
    training.task_name=XqRobotWLBackflipFlat \
    ${LOAD_RUN} \
    ${MAX_ITERS}
