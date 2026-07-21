#!/bin/bash
# ============================================================
# SRL 消融实验单启动
# 用法: bash training/train_ablation.sh <mode>
#   mode: no_fsm | no_wheel_match | no_flight_mod | no_vel_track
# ============================================================
set -e
cd "$(dirname "$0")/.."

MODE="${1:-no_fsm}"
PROJ=".."

case "$MODE" in
  no_fsm)           OVERRIDES="env.ablation_mode=no_fsm training.run_suffix=no_fsm reward.scales.wheel_ground_matching=20.0 reward.feedback_gain=1.0" ;;
  no_wheel_match)   OVERRIDES="env.ablation_mode=no_wheel_match training.run_suffix=no_wheel_match reward.scales.wheel_ground_matching=0.0" ;;
  no_flight_mod)    OVERRIDES="env.ablation_mode=no_flight_mod training.run_suffix=no_flight_mod" ;;
  no_vel_track)     OVERRIDES="env.ablation_mode=no_vel_track training.run_suffix=no_vel_track reward.scales.tracking_lin_vel=0.0" ;;
  *) echo "Unknown mode: $MODE"; exit 1 ;;
esac

echo "=== Training SRL ablation: $MODE ==="
cd "$PROJ"
bash shell/xqrobotwl/train_ppo_jump_srl.sh $OVERRIDES

