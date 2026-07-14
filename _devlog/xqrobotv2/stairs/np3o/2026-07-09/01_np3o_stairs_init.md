# 01 初始化楼梯专项训练 (NP3O + 100% stairs)

**日期**: 2026-07-09
**来源**: 移植 Tita RL NP3O 算法 + 100% 楼梯地形，从 rough_walk 中独立
**关联**: [参考文档](../../../docs/references/LearningHumanoidWalking.md), [Tita RL 参考](../../../docs/references/TitaRL.md), [BoltLocomotion 参考](../../../docs/references/BoltLocomotion.md)

---

## 问题描述

rough_walk PPO 训练需要同时学习多类型地形（粗糙/波浪/斜坡），楼梯训练效果被稀释。需要独立的楼梯专项训练任务。

## 解决方案

### 1. 算法移植: NP3O

从 Tita RL 参考项目移植 NP3O (Neural Proximal Policy Optimization with constraints)：

```
loss = surrogate_loss + value_loss
     + cost_value_loss_coef * cost_value_loss     # cost critic 价值估计
     + cost_viol_loss_coef * viol_loss            # 违规损失
     - entropy_coef * entropy

cost_returns = GAE(costs, cost_values, γ, λ)
cost_adv     = normalize(cost_returns - cost_values) per-dim
cost_viol    = normalize((1-γ)*(cost_returns - d)) per-dim
cost_surr    = max(cost_adv*ratio, cost_adv*clip(ratio)).mean(0) per-cost
viol_loss    = Σ k * ReLU(cost_surr + cost_viol.mean())
k            = min(1, k_init * k_growth^iter)
```

### 2. 地形: 100% 楼梯

```
StairsOnlyTerrainCfg:
  8 行 × 4 列 = 32 个子地形
  50% pyramid_stairs (上楼)
  50% pyramid_stairs_inv (下楼)
  step_height: 0.03-0.11m
  step_width: 0.55m
  地形课程: 走远升级/走近降级
```

### 3. 奖励设计 (对齐 Tita)

仅保留 tracking + penalties，无显式抬腿/爬楼奖励：

```
tracking_lin_vel: 2.0    (速度跟踪)
tracking_ang_vel: 1.0
collision: -1.0          (力传感器触地惩罚)
orientation: -8.0
base_height: -5.0
... (机器人对称奖励)
alive: 0.0               (无存活奖励，与 Tita 一致)
```

### 4. 训练超参 (对齐 Tita PPOCfg)

| 参数 | 值 |
|------|-----|
| lr | 1e-3 |
| entropy | 0.01 |
| desired_kl | 0.01 |
| max_grad_norm | 0.01 |
| init_noise_std | 1.0 |
| max_iterations | 10000 |
| num_envs | 512 |

### 5. NP3O 参数

| 参数 | 值 |
|------|-----|
| num_costs | 6 |
| cost_critic_dims | [512, 256, 128] |
| cost_value_loss_coef | 0.1 |
| cost_viol_loss_coef | 0.1 |
| k_init | 0.3 |
| k_growth | 1.0004 |
| k_max | 1.0 |

## 修改文件

| 文件 | 说明 |
|------|------|
| `src/unilab/algos/torch/np3o.py` | NP3O 算法实现 |
| `src/unilab/structured_configs.py` | NP3OConfig + NP3OAlgorithmConfig |
| `conf/np3o/config.yaml` | NP3O 基础配置 |
| `conf/np3o/task/xqrobotV2_stairs/mujoco.yaml` | 楼梯任务配置 |
| `src/unilab/envs/locomotion/xqrobotV2/stairs.py` | 楼梯环境 |
| `scripts/training/train_np3o.py` | NP3O 训练入口 |
| `scripts/play/play_interactive.py` | 注册 np3o algo |
| `shell/train_np3o_stairs.sh` / `eval_np3o_stairs.sh` | 训练/评估脚本 |
| `tools/mujoco/show_stairs.py` | 地形可视化 |

## 验证方法

```bash
# 训练
CUDA_VISIBLE_DEVICES=0 bash shell/train_np3o_stairs.sh

# 评估
bash shell/eval_np3o_stairs.sh

# 可视化
uv run tools/mujoco/show_stairs.py
```

训练启动成功 (iter 89/10000, 0.78s/iter, ETA ~2.5h)。

## 后续计划

- [ ] 训练至 10000 iter
- [ ] 评估 Vx/Vy 跟踪 + 楼梯攀爬能力
- [ ] 与 rough_walk PPO 对比

---

*记录人: AI (opencode)*
