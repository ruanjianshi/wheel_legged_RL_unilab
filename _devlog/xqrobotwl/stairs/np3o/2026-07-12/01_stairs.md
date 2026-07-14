# 04 xqrobotwl stairs 训练配置

**日期**: 2026-07-12
**关联**: [01_create_robot](../../../2026-07-12/01_create_robot.md)

---

## 配置

| 项 | 值 |
|------|-----|
| 任务名 | `XqRobotWLStairs` |
| Hydra key | `xqrobotwl_stairs/mujoco` |
| 算法 | NP3O (cost critic + viol_loss) |
| envs | 1024 × 25 steps |
| iters | 10000 |
| 地形 | 100% 楼梯 (上下各50%) |

## 修改文件

| 文件 | 改动 |
|------|------|
| `conf/np3o/task/xqrobotwl_stairs/mujoco.yaml` | 从 xqrobotV2 复制，Vy=0, tracking=4.0+sigma=0.15, cost_viol=0.01 |
| `src/unilab/envs/locomotion/xqrobotwl/stairs.py` | 继承 xqrobotwl rough |

## 启动

```bash
bash shell/xqrobotwl/train_np3o_stairs.sh
```

---

*记录人: AI (opencode)*
