# 03 启动粗糙地形 PPO 训练

**日期**: 2026-07-01  
**来源**: 开发推进  
**关联**: [01 地形工具](01_terrain_setup.md) [02 丰富地形](02_enrich_terrain.md)

---

## 训练配置

| 参数 | 值 |
|------|-----|
| 任务 | XqRobotV2WalkRough |
| 算法 | PPO |
| GPU | GPU 1 |
| Envs | 2048 |
| Steps/env | 24 |
| **Max iter** | **20000** |
| LR | 1e-4 |
| Entropy coef | 0.002 |
| Init noise std | 0.3 |
| Desired KL | 0.005 |
| ETA | ~21h |

## 地形配置（已更新）

| 类型 | 比例 | 参数 |
|------|------|------|
| 上楼梯 | 15% | step 0.02-0.08m |
| 下楼梯 | 15% | step 0.02-0.08m |
| 随机粗糙 | 30% | noise 0.01-0.06m |
| 波浪 | 30% | amp 0-0.10m |
| 上坡 | 5% | slope 0-0.15 |
| 下坡 | 5% | slope 0-0.15 |

## 奖励配置（与 flat_walk 对齐）

与 `xqrobotV2_walk_flat/mujoco.yaml` 完全一致，增加 `hip_roll=-2.0`, `wheel_symmetry=-0.5`, `feet_distance=-1.0`, `similar_calf=-1.0`。

## 修改记录

| 文件 | 改动 |
|------|------|
| `conf/ppo/task/xqrobotV2_walk_rough/mujoco.yaml` | max_iter 3000→20000, entropy 0.01→0.002, noise 1.0→0.3, kl 0.01→0.005, 对齐 reward scales |
| `src/unilab/envs/locomotion/xqrobotV2/joystick.py:244-247` | 修复 `_LEG_GEOM_NAMES` 缺失导致的 AttributeError |
| `src/unilab/envs/locomotion/xqrobotV2/rough.py:73-113` | 添加 stairs，调整比例 |

## 训练命令

```bash
CUDA_VISIBLE_DEVICES=1 nohup uv run scripts/training/train_rsl_rl.py \
  task=xqrobotV2_walk_rough/mujoco &>/tmp/rough_train.log &
```

## 后续计划

- [ ] iter=5000 首次评估
- [ ] iter=10000 评估
- [ ] iter=15000 评估
- [ ] iter=20000 全量评估

---

*记录人: AI (opencode)*
