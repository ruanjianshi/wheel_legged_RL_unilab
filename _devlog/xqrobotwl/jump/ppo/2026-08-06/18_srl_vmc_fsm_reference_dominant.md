# 18 修复 SRL+VMC: FSM 参考主导 + 策略残差(策略不再抵消参考)

**日期**: 2026-08-06
**来源**: 四算法监控 — SRL+VMC 训练模型只跳 0.033m,比开环 FSM(0.13m)还差
**关联**: [17_vmc_jump_param_fix](2026-08-06/17_vmc_jump_param_fix.md)

---

## 问题描述

SRL+VMC 训练到 iter 3000-4000,训练 jump_height 奖励 0.25-0.49(在升),但 verify_jump 实测只跳 **0.033m**,甚至低于开环 FSM 的 0.13m。

## 根因分析

**策略学会了抵消 FSM 参考**(参考项目 Wheel-Legged-Lab 明确警告过的问题):

原混合公式 `final_L0 = FSM_ref × gain + 策略_L0`(gain=0.15):
- FSM 只贡献 15%,策略(权重 1.0)主导
- 策略发现"抵消 FSM 参考 + 只平衡"更易存活 → 训练后 L0 完全不跟随参考

开环测试证实:即使 FSM 参考满幅(gain=1.0),开环也只能跳 0.17m;而训练后的策略把开环的 0.13m 反而压到 0.033m。

## 解决方案

翻转混合公式为**参考项目模式**(FSM 参考主导,策略小残差):

```python
# 旧: actions[:,2] = target_action * gain + actions[:,2]   (策略主导)
# 新: actions[:,2] = target_action + residual_scale * actions[:,2]  (参考主导)
```

`feedback_gain`(YAML 0.15)现在作为 `residual_scale` 使用。

## 修改文件

| 文件 | 修改内容 |
|------|---------|
| `src/unilab/envs/locomotion/xqrobotwl/jump_srl_vmc.py` | `step()` 混合公式翻转:FSM 参考主导 + 残差×策略 |

## 验证方法

- 开环 FSM(零策略): 起跳 **0.17m**(策略无法再抵消参考)✅
- 60 步 RL 冒烟: 无 NaN, FSM 正常 ✅
- 已重启 SRL+VMC 训练(ETA ~6h)

## 后续计划

- [ ] 监控重启后 SRL+VMC 是否在 iter 1000-2000 显著跳高(应 ≥ 0.17m 基线)
- [ ] 纯 VMC(无 FSM)仍在观察: iter 4000+ 只跳 0.03m,若到 6000 仍不跳,作为弱基线呈现(支持"SRL 指导帮助 VMC"论点)
- [ ] 训练完成后四算法评估对比

## 关联日志

- [17_vmc_jump_param_fix](2026-08-06/17_vmc_jump_param_fix.md)
- [16_ppo_vmc_srl_vmc_algorithms](2026-08-06/16_ppo_vmc_srl_vmc_algorithms.md)
