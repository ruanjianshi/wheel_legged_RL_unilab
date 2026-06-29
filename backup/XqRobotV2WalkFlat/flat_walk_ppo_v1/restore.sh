#!/bin/bash
# ============================================================
# 恢复脚本: 将备份文件覆盖到 UniLab 项目目录
# 用法: bash backup/XqRobotV2WalkFlat/flat_walk_ppo_v1/restore.sh
# ============================================================
set -e

BACKUP_DIR="$(cd "$(dirname "$0")" && pwd)"
# 项目根目录: backup/XqRobotV2WalkFlat/flat_walk_ppo_v1/ -> ../../../../
PROJECT_DIR="$(cd "$BACKUP_DIR/../../../../" && pwd)"

echo "备份: $BACKUP_DIR"
echo "目标: $PROJECT_DIR"
echo ""

# conf
echo "[1/5] 恢复配置..."
cp -v "$BACKUP_DIR/conf/ppo/task/xqrobotV2_walk_flat/mujoco.yaml" \
      "$PROJECT_DIR/conf/ppo/task/xqrobotV2_walk_flat/mujoco.yaml"

# src
echo "[2/5] 恢复环境代码..."
cp -v "$BACKUP_DIR/src/unilab/envs/locomotion/xqrobotV2/__init__.py" \
      "$PROJECT_DIR/src/unilab/envs/locomotion/xqrobotV2/__init__.py"
cp -v "$BACKUP_DIR/src/unilab/envs/locomotion/xqrobotV2/base.py" \
      "$PROJECT_DIR/src/unilab/envs/locomotion/xqrobotV2/base.py"
cp -v "$BACKUP_DIR/src/unilab/envs/locomotion/xqrobotV2/joystick.py" \
      "$PROJECT_DIR/src/unilab/envs/locomotion/xqrobotV2/joystick.py"

# scene
echo "[3/5] 恢复场景文件..."
cp -v "$BACKUP_DIR/scene_flat.xml" \
      "$PROJECT_DIR/src/unilab/assets/robots/xqrobotV2/scene_flat.xml"

# shell
echo "[4/5] 恢复脚本..."
cp -v "$BACKUP_DIR/shell/train_ppo_flat.sh" \
      "$PROJECT_DIR/shell/train_ppo_flat.sh"
cp -v "$BACKUP_DIR/shell/eval_ppo_flat.sh" \
      "$PROJECT_DIR/shell/eval_ppo_flat.sh"
cp -v "$BACKUP_DIR/shell/tensorboard.sh" \
      "$PROJECT_DIR/shell/tensorboard.sh"

echo "[5/5] 完成."
echo ""
echo "启动训练: cd $PROJECT_DIR && bash shell/train_ppo_flat.sh"
