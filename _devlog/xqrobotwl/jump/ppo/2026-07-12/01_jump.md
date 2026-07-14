# 05 xqrobotwl jump 训练配置

**日期**: 2026-07-12
**关联**: [01_create_robot](../../../2026-07-12/01_create_robot.md)

---

## 配置

| 项 | 值 |
|------|-----|
| 任务名 | `XqRobotWLJumpFlat` |
| Hydra key | `xqrobotwl_jump_flat/mujoco` |
| 算法 | PPO |
| envs | 1024 × 24 steps |
| iters | 10000 |
| 地形 | 平地 |

## 修改文件

| 文件 | 改动 |
|------|------|
| `conf/ppo/task/xqrobotwl_jump_flat/mujoco.yaml` | 从 xqrobotV2 复制，tracking=2.0, Vy=0, crouch_prep=2.0, wheel_air_time=2.0 |
| `src/unilab/envs/locomotion/xqrobotwl/jump.py` | wheel_contact 阈值 10N |

## 启动

```bash
bash shell/xqrobotwl/train_ppo_jump_flat.sh
```

---

*记录人: AI (opencode)*
