# [23] 修复 verify_jump 配置漂移 bug: 验证 env 与训练配置不一致

**日期**: 2026-08-07
**来源**: 训练监控
**关联**: [[21_vmc_kd_l0_fix]], [[22_pure_ppo_v5_retrain]]

---

## 问题描述

监控 VMC 变体重训时发现严重矛盾:

- **训练 TensorBoard** 显示 `reward/jump_height` = 0.9 (VMC+SRL), 意味着策略在腾空且 base_z 接近目标
- **verify_jump.py** 实测旧 best checkpoint (model_9999) 只有 `air_frac=0.03~0.04`, `jump_height=0.034~0.047`
- 新 checkpoint (model_3000/4000) 也只有 air 4~8%, 低于目标 10%

与之前记录的 "VMC+SRL model_9999 air 10%" 严重矛盾。

## 根因分析

**verify_jump.py 用硬编码的默认配置构建 env, 与训练时的 Hydra 配置完全错位**:

| 参数 | verify 默认值 | VMC+SRL 训练值 | PPO+VMC 训练值 |
|------|--------------|---------------|---------------|
| `kd_l0` | 5.0 (vmc.py 默认) | **20.0** (旧run) / 5.0 (新run) | 5.0 |
| `feedback_gain` | 0.15 (jump_srl.py 默认) | **0.5** | - |
| thrust 增益 | 3.0/3.0 (默认) | 3.0/3.0 | - |

关键缺陷:
1. **旧 run (19-54-14) 用 kd_l0=20 训练**, verify 用默认 5.0 → 完全不同的动力学, 旧模型验证失真
2. **SRL+VMC 训练用 feedback_gain=0.5**, verify 用默认 0.15 → 策略残差被压到 1/3, 参考主导导致跳不起来
3. 之前记录 "air 10%" 是另一组配置下的偶然结果, 不可复现

## 解决方案

让 verify_jump.py 从 `run_config.json` (训练启动时保存的完整 Hydra 配置快照) 重建 env:

- 新增 `trained_env_overrides(checkpoint_path)`: 读取 checkpoint 同目录 `run_config.json`
- 提取 `config['env']` (vmc/control_config/commands/curriculum) + `config['reward']` (含 feedback_gain)
- `registry.make(..., env_cfg_override=trained_ov)` 深合并到默认 cfg
- 无 run_config 时回退旧逻辑 (纯 PPO / SRL)
- diag_jump_trajectory.py 同步复用

修复后实测:

| checkpoint | 修复前 air | 修复后 air | 修复后 jump_height |
|-----------|-----------|-----------|-------------------|
| 旧 VMC+SRL model_9999 (kd_l0=20) | 3% | 6% | 0.034 |
| 新 PPO+VMC model_4000 | 8% | 9% | 0.081 |
| 新 VMC+SRL model_3000 | 4% | 9% | 0.068 |

diag 轨迹确认新模型是**真实跳跃**: 下蹲 0.28 → 蹬伸 0.712 (standing ~0.5), 双轮离地, air 9%。

## 修改文件

| 文件 | 修改内容 |
|------|---------|
| `scripts/verify_jump.py` | 新增 `trained_env_overrides()`, env 构建优先用训练配置快照 |
| `scripts/diag_jump_trajectory.py` | 同步复用 `trained_env_overrides` |

## 验证方法

- 修复后重新验证旧/新 checkpoint, air 与训练 reward 趋势一致 (VMC+SRL 0.9 → air 9%)
- diag 轨迹显示完整下蹲→蹬伸→腾空→落地循环
- 存活率 600/600

## 后续计划

- 训练进行中 (PPO+VMC 4000+/10000, VMC+SRL 3700+/10000)
- 训练完成后用修复后的 verify 做最终四算法对比
- 历史 four_algo_comparison.json 数据已污染, 需用修复后 verify 重新评估
