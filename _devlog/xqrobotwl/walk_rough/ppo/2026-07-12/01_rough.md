# 03 xqrobotwl rough_walk 训练配置

**日期**: 2026-07-12
**关联**: [01_create_robot](../../../2026-07-12/01_create_robot.md)

---

## 配置

| 项 | 值 |
|------|-----|
| 任务名 | `XqRobotWLWalkRough` |
| Hydra key | `xqrobotwl_walk_rough/mujoco` |
| 算法 | PPO |
| envs | 1024 × 25 steps |
| iters | 10000 |
| 地形 | 平地20% + 粗糙35% + 波浪35% + 斜坡10% |

## 修改文件

| 文件 | 改动 |
|------|------|
| `conf/ppo/task/xqrobotwl_walk_rough/mujoco.yaml` | 从 xqrobotV2 复制，Vy=0, tracking=4.0+sigma=0.15, hip_roll=-4.0 |
| `src/unilab/envs/locomotion/xqrobotwl/rough.py` | 继承 xqrobotwl joystick |

## 启动

```bash
bash shell/xqrobotwl/train_ppo_rough.sh
```

---

*记录人: AI (opencode)*
