#!/bin/bash
# ============================================================
# 批量评估脚本
# 评估所有模型 × 所有场景, 输出 JSON 到 results/
#
# 用法: bash evaluate/eval_batch.sh
# ============================================================
set -e
cd "$(dirname "$0")/.."
PROJ=".."

SCENARIOS="fix_01m fix_02m fix_03m random platform"
NUM_EPS=50  # 每场景评估回合数

echo "=== 跳跃模型批量评估 ==="
echo ""

for model_type in ppo_only srl_full srl_no_fsm srl_no_wheel_match srl_no_flight_mod srl_no_vel_track; do
    echo "--- 评估: $model_type ---"
    OUTDIR="results/$model_type"
    mkdir -p "$OUTDIR"

    # 查找模型 checkpoint (按 run_suffix 匹配)
    case "$model_type" in
        ppo_only)   LOG_DIR="$PROJ/logs/rsl_rl_ppo/XqRobotWLJumpFlat";     SUFFIX="_mujoco" ;;
        srl_full)   LOG_DIR="$PROJ/logs/rsl_rl_ppo/XqRobotWLJumpSRLFlat";  SUFFIX="_mujoco" ;;
        srl_no_fsm) LOG_DIR="$PROJ/logs/rsl_rl_ppo/XqRobotWLJumpSRLFlat";  SUFFIX="_mujoco_no_fsm" ;;
        srl_no_wheel_match) LOG_DIR="$PROJ/logs/rsl_rl_ppo/XqRobotWLJumpSRLFlat"; SUFFIX="_mujoco_no_wheel_match" ;;
        srl_no_flight_mod)  LOG_DIR="$PROJ/logs/rsl_rl_ppo/XqRobotWLJumpSRLFlat"; SUFFIX="_mujoco_no_flight_mod" ;;
        srl_no_vel_track)   LOG_DIR="$PROJ/logs/rsl_rl_ppo/XqRobotWLJumpSRLFlat"; SUFFIX="_mujoco_no_vel_track" ;;
    esac

    # 找到匹配的 subdir
    SUBDIR=$(ls -td "$LOG_DIR"/*"$SUFFIX"/ 2>/dev/null | head -1)
    if [ -z "$SUBDIR" ]; then
        echo "  跳过 (无模型): $LOG_DIR"
        continue
    fi

    # 取最后 checkpoint
    CKPT=$(ls "$SUBDIR"/model_*.pt 2>/dev/null | sort -t_ -k2 -n | tail -1)
    if [ -z "$CKPT" ]; then
        echo "  跳过 (无checkpoint): $SUBDIR"
        continue
    fi

    echo "  Checkpoint: $CKPT"

    for scenario in $SCENARIOS; do
        OUT_JSON="$OUTDIR/${scenario}.json"
        echo "  场景: $scenario -> $OUT_JSON"

        # 调用评估 Python 脚本
        cd "$PROJ"
        uv run python -c "
from jump_management.evaluate.runner import eval_one_scenario
eval_one_scenario(
    model_path='$CKPT',
    task_name='$model_type',
    scenario='$scenario',
    num_episodes=$NUM_EPS,
    out_path='$OUT_JSON',
)
"
    done

    echo ""
done

echo "=== 评估完成 ==="
echo "结果: results/*/"
