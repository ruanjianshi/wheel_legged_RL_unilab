---
name: xqrobotwl-dev
description: Use ONLY when editing xqrobotwl robot configuration in wheel_legged_RL_unilab project. Handles RL training, simulation, and asset management.
---

# xqrobotwl Development Skill

## Commands

`train-rl` - Train reinforcement learning model:
```bash
python -m unilab.base.backend.mujoco.train --robot xqrobotwl --task locomotion
```

`simulate` - Run Gazebo simulation:
```bash
python -m unilab.base.backend.mujoco.simulate --robot xqrobotwl --scene assets/robots/go2/scene_flat.xml
```

`view-urdf` - Visualize robot model:
```bash
python -m unilab.assets.robots.go2.view
```

## Configuration Files
- Robot definition: `src/unilab/assets/robots/go2/locomotion_task.xml`
- Training config: `src/unilab/base/backend/mujoco/train.py`

> After creating this skill, restart opencode for changes to take effect.