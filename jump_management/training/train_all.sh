#!/bin/bash
# ============================================================
# 一键启动全部 6 个训练 (PPO-only + SRL full + 4 消融)
# 消融共享 SRL base config, 通过 Hydra overrides 修改特定字段.
# 分配到 2 张 GPU, 每张 GPU 串行.
# ============================================================
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=== 启动训练 (2 GPU) ==="
echo "GPU 0: PPO-only -> no_fsm -> no_wheel_match (串行)"
echo "GPU 1: SRL full -> no_flight_mod -> no_vel_track (串行)"
echo ""

# GPU 0
CUDA_VISIBLE_DEVICES=0 setsid bash -c "
  cd '$PROJ' && \
  bash shell/xqrobotwl/train_ppo_jump_flat.sh && \
  bash shell/xqrobotwl/train_ppo_jump_srl.sh \
    training.run_suffix=no_fsm reward.feedback_gain=0.0 && \
  bash shell/xqrobotwl/train_ppo_jump_srl.sh \
    training.run_suffix=no_wheel_match reward.scales.wheel_ground_matching=0.0
" &>/tmp/jump_gpu0.log &

sleep 2

# GPU 1
CUDA_VISIBLE_DEVICES=1 setsid bash -c "
  cd '$PROJ' && \
  bash shell/xqrobotwl/train_ppo_jump_srl.sh && \
  bash shell/xqrobotwl/train_ppo_jump_srl.sh \
    training.run_suffix=no_flight_mod && \
  bash shell/xqrobotwl/train_ppo_jump_srl.sh \
    training.run_suffix=no_vel_track \
    reward.scales.tracking_lin_vel=0.0 reward.scales.tracking_ang_vel=0.0
" &>/tmp/jump_gpu1.log &

echo "训练已后台启动"
echo "  GPU 0 log: tail -f /tmp/jump_gpu0.log"
echo "  GPU 1 log: tail -f /tmp/jump_gpu1.log"
