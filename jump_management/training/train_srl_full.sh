#!/bin/bash
# ============================================================
# Wheeled-SRL 完整版训练
# ============================================================
set -e
cd "$(dirname "$0")/.."

PROJ=".."
cd "$PROJ"
bash shell/xqrobotwl/jump/train_ppo_jump_srl.sh
