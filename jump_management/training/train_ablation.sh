#!/bin/bash
# ============================================================
# SRL 消融实验训练脚本
# 用法: bash training/train_ablation.sh <mode>
#   mode: no_fsm | no_wheel_match | no_flight_mod | no_vel_track
#
# 消融配置参考: jump_management/config/ablate_*.yaml
# ============================================================
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODE="${1:-no_fsm}"

echo "=== Training SRL ablation: $MODE ==="
cd "$PROJ"

# Each ablation modifies specific reward/config fields via Hydra overrides.
# The training.run_suffix appends to the log directory name for separation.
case "$MODE" in
  no_fsm)
    bash shell/xqrobotwl/train_ppo_jump_srl.sh \
      training.run_suffix=no_fsm \
      reward.feedback_gain=0.0
    ;;
  no_wheel_match)
    bash shell/xqrobotwl/train_ppo_jump_srl.sh \
      training.run_suffix=no_wheel_match \
      reward.scales.wheel_ground_matching=0.0
    ;;
  no_flight_mod)
    bash shell/xqrobotwl/train_ppo_jump_srl.sh \
      training.run_suffix=no_flight_mod
    ;;
  no_vel_track)
    bash shell/xqrobotwl/train_ppo_jump_srl.sh \
      training.run_suffix=no_vel_track \
      reward.scales.tracking_lin_vel=0.0 \
      reward.scales.tracking_ang_vel=0.0
    ;;
  *)
    echo "Unknown mode: $MODE"
    echo "Valid modes: no_fsm | no_wheel_match | no_flight_mod | no_vel_track"
    exit 1
    ;;
esac
