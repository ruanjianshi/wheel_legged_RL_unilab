#!/bin/bash
# ============================================================
# 策略验证/回放脚本 (Viser Web 可视化)
# 支持多 env 并行查看, 远程浏览器访问
# 用法: bash shell/eval/play_viser.sh
# ============================================================
set -e

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

# ---- 配置 ----
TASK="go2w_joystick_flat/mujoco"
NUM_ENVS=4
VISER_PORT=8080
LOAD_RUN="${1:-}"
# --------------

CMD="uv run scripts/play/play_viser.py \
    task=\"${TASK}\" \
    algo.num_envs=${NUM_ENVS} \
    viser.port=${VISER_PORT} \
    interactive.action_mode=policy"

if [ -n "$LOAD_RUN" ]; then
    CMD="${CMD} algo.load_run=\"${LOAD_RUN}\""
fi

eval "$CMD" "$@"
