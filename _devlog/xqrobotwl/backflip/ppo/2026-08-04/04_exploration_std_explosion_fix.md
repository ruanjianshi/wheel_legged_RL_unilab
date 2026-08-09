# 04 — 修复策略探索 std 暴涨 (init_noise_std 0.3→0.05)

## 日期

2026-08-04

## 来源

预热 bug 修复后重跑 quick 200 iter: 翻转奖励仍为 0, 但 ep_len 从 392 暴跌到 103, reward 中途 22.8→1.87 (翻转触发但失败)。

## 问题描述

加载 model_199 检查: 策略探索标准差 **exp(std_param)=3.9** (log_std 从 log(0.3)=-1.2 涨到 0.31)。训练策略确定性输出只走到蹲姿就反复失败重置, 翻转从未展开。

## 根因分析

1. **init_noise_std=0.3 太高** + 预热期纯站立: 高 std 采样动作被 ff 和位置控制器抵消, 站立奖励不塌 → 熵奖励让 std 自由上涨到 3.9。
2. std=3.9 时采样动作 ±5+ 远超 ff(±1.7), 翻转被淹没 → flip 学习无梯度 → 恶性循环。
3. 反向验证: 纯 ff + 随机噪声 std=0.3 反而达 102% 翻转 — 说明**固定小噪声不破坏翻转, 是训练中 std 失控**。

## 解决方案

`conf/ppo/task/xqrobotwl_backflip_flat/mujoco.yaml`: `init_noise_std: 0.3 → 0.05` (低探索引导, ff 主导, 让早期策略贴近已验证翻转)。

## 修改文件

| 文件 | 内容 |
|------|------|
| `conf/ppo/task/xqrobotwl_backflip_flat/mujoco.yaml` | init_noise_std 0.3→0.05 |

## 验证方法

重跑 quick 200 iter (init_noise_std=0.05):
- ✅ 探索 std: 0.05→0.06 (受控, 不再暴涨)
- ✅ 确定性策略: 能翻转至 292° (fsm→2飞行→3展开)
- ✅ flip_progress 奖励: 最后迭代出现非零 (iter 188-195: 0.48/0.78/0.29/0.39), 但振荡(不稳定, 正常起步阶段)
- ✅ ep_len ~400 (翻转尝试不摔, 可恢复)
- 结论: quick 200 iter 只有最后 ~50 iter 在学翻转(前 100 预热), 翻转刚起步, 需全量训练

## 后续计划

- 开全量 10000 iter 训练, 监控 flip_progress/flip_complete 收敛
- 若 flip 可靠性长期不升: 调翻转奖励权重或加成功率课程

## 关联日志

- [03_warmup_trigger_bugfix](2026-08-04/03_warmup_trigger_bugfix.md) — 预热修复
- [02_env_development](2026-08-04/02_env_development.md) — 环境开发
