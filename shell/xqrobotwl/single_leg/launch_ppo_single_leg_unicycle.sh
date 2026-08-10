#!/bin/bash
# ============================================================
# XqRobotWL 改进结构单轮平衡 PPO (独立 task, 从零训练)
# 结构 (devlog 20/21): 横躺 up=[0,-1,0], 左腿近直支撑, 右腿伸直配重,
#   腿 kp=300 + 轮扭矩源 (env init 运行时设置). RL 控 [配重roll, 轮pitch].
# 用法:
#   bash shell/xqrobotwl/launch_ppo_single_leg_unicycle.sh           # 全量 20000 iter
#   bash shell/xqrobotwl/launch_ppo_single_leg_unicycle.sh quick      # 快速验证 500 iter
#   bash shell/xqrobotwl/launch_ppo_single_leg_unicycle.sh 1000       # 指定迭代数
#   bash shell/xqrobotwl/launch_ppo_single_leg_unicycle.sh resume     # 续训最新
# 注意: PPO 后期会回退 (model_3000/4000 达 8s, model_4999 退化) —
#   部署用 eval 验证选 checkpoint, 不要用最后一个.
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
        LATEST=$(ls -dt logs/rsl_rl_ppo/XqRobotWLSingleLegUnicycle/*/ 2>/dev/null | head -1)
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

TASK="xqrobotwl_single_leg_unicycle/mujoco"

echo "=============================================="
echo "启动改进结构单轮平衡训练 (从零, 前台): mode=$MODE ${LOAD_RUN:+($LOAD_RUN)}"
echo "TensorBoard: uv run tensorboard --logdir logs/rsl_rl_ppo/XqRobotWLSingleLegUnicycle/"
echo "=============================================="

exec uv run scripts/training/train_rsl_rl.py \
    task="${TASK}" \
    training.task_name=XqRobotWLSingleLegUnicycle \
    ${LOAD_RUN} \
    ${MAX_ITERS}
