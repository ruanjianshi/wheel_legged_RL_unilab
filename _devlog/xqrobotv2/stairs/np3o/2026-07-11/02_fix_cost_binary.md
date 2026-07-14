# 02 NP3O cost 0/1 二元化 + 物理数据源

**日期**: 2026-07-11
**来源**: stairs iter=4756 评估发现转圈 + cost_value_loss=8.7M 爆炸
**关联**: [01_np3o_stairs_init](2026-07-09/01_np3o_stairs_init.md), [BoltLocomotion 参考](../../../docs/references/BoltLocomotion.md)

---

## 问题描述

stairs NP3O iter=4756: cost_value_loss = 8,763,934，viol_loss = 0.58，机器人转圈。cost critic 完全发散。

## 根因分析

`_extract_costs` 返回连续值（关节速度 0~10 rad/s），GAE 累积回报 ~3000，cost critic 预测 ~0：

```
MSE = (3000 - 0.1)² ≈ 9,000,000 → cost_value_loss = 8.7M
```

梯度爆炸 → viol_loss 污染 PPO 梯度 → 策略退化。

Tita RL 的成本是 0/1 二元阈值检测，每步 ≤1，GAE 回报 ≤100。

## 解决方案

`_extract_costs` 全部改为 0/1 二元，从 critic observation（真实物理数据）中提取：

| cost | 检测方式 | 阈值 |
|------|------|------|
| orientation | `||gravity_xy||` | >0.3 rad |
| joint_vel | `mean|leg_vel|` | >5 rad/s |
| joint_acc | `|Δvel|/dt` | >800 rad/s² |
| torque proxy | `mean|leg_pos_err|` | >0.04 rad |
| force proxy | `|Δlinvel|/dt` | >30 m/s² |
| stumble | `|linvel_z|` | >0.2 m/s |

## 修改文件

| 文件 | 修改内容 |
|------|---------|
| `src/unilab/algos/torch/np3o.py:L115-161` | `_extract_costs` 全部阈值 0/1 |

## 验证方法

v6 重训，成本值正常（cost_value_loss < 100）。

## 后续计划

- [x] v6 重训启动 (iter ~900，运行中)
- [ ] 训练完成评估

---

*记录人: AI (opencode)*
