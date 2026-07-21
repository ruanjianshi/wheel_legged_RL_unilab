#!/bin/bash
# ============================================================
# 一键启动全部6个训练 (PPO-only + SRL full + 4消融)
# 分配到2张GPU, 消融串行
# ============================================================
set -e
cd "$(dirname "$0")/.."

PROJ=".."

echo "=== 启动训练 (2 GPU) ==="
echo "GPU 0: PPO-only + no_fsm + no_wheel_match (串行)"
echo "GPU 1: SRL full + no_flight_mod + no_vel_track (串行)"
echo ""

# GPU 0: PPO-only -> no_fsm -> no_wheel_match
CUDA_VISIBLE_DEVICES=0 setsid bash -c "
  cd '$PROJ' && \
  bash shell/xqrobotwl/train_ppo_jump_flat.sh && \
  bash shell/xqrobotwl/train_ppo_jump_srl.sh \
    env.ablation_mode=no_fsm training.run_suffix=no_fsm reward.feedback_gain=1.0 && \
  bash shell/xqrobotwl/train_ppo_jump_srl.sh \
    env.ablation_mode=no_wheel_match training.run_suffix=no_wheel_match reward.scales.wheel_ground_matching=0.0
" &>/tmp/jump_gpu0.log &

sleep 2

# GPU 1: SRL full -> no_flight_mod -> no_vel_track
CUDA_VISIBLE_DEVICES=1 setsid bash -c "
  cd '$PROJ' && \
  bash shell/xqrobotwl/train_ppo_jump_srl.sh && \
  bash shell/xqrobotwl/train_ppo_jump_srl.sh \
    env.ablation_mode=no_flight_mod training.run_suffix=no_flight_mod && \
  bash shell/xqrobotwl/train_ppo_jump_srl.sh \
    env.ablation_mode=no_vel_track training.run_suffix=no_vel_track reward.scales.tracking_lin_vel=0.0
" &>/tmp/jump_gpu1.log &

echo "训练已后台启动"
echo "  GPU 0: tail -f /tmp/jump_gpu0.log"
echo "  GPU 1: tail -f /tmp/jump_gpu1.log"
