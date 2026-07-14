# 02 xqrobotwl flat_walk 训练配置

**日期**: 2026-07-12
**关联**: [01_create_robot](../../../2026-07-12/01_create_robot.md)

---

## 配置

| 项 | 值 |
|------|-----|
| 任务名 | `XqRobotWLWalkFlat` |
| Hydra key | `xqrobotwl_walk_flat/mujoco` |
| 算法 | PPO |
| envs | 1024 × 24 steps |
| iters | 10000 |
| 地形 | 平地 (plane) |

## 修改文件

| 文件 | 改动 |
|------|------|
| `conf/ppo/task/xqrobotwl_walk_flat/mujoco.yaml` | 从 xqrobotV2 复制 + 类名替换 |
| `src/unilab/envs/locomotion/xqrobotwl/joystick.py` | 含 sign_flip, 含 collision 关节极限 |

## 启动

```bash
bash shell/xqrobotwl/train_ppo_flat.sh
```

---

*记录人: AI (opencode)*
