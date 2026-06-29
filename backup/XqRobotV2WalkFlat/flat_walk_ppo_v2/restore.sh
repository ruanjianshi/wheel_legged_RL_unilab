#!/bin/bash
set -e
BACKUP_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$BACKUP_DIR/../../../../" && pwd)"
echo "Restoring flat_walk_ppo_v2 to $PROJECT_DIR"
cp -v "$BACKUP_DIR"/conf/ppo/task/xqrobotV2_walk_flat/mujoco.yaml "$PROJECT_DIR"/conf/ppo/task/xqrobotV2_walk_flat/
cp -v "$BACKUP_DIR"/src/unilab/envs/locomotion/xqrobotV2/*.py "$PROJECT_DIR"/src/unilab/envs/locomotion/xqrobotV2/
cp -v "$BACKUP_DIR"/shell/*.sh "$PROJECT_DIR"/shell/
cp -v "$BACKUP_DIR"/scene_flat.xml "$PROJECT_DIR"/src/unilab/assets/robots/xqrobotV2/
cp -v "$BACKUP_DIR"/xqrobotV2.xml "$PROJECT_DIR"/src/unilab/assets/robots/xqrobotV2/
echo "Done. Run: cd $PROJECT_DIR && bash shell/train_ppo_flat.sh"
