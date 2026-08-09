# 20 方案A: PPO+VMC 加入 SLIP-FSM 参考(全动作) + 纯PPO 落地存活修复

**日期**: 2026-08-06
**来源**: 用户要求 — PPO+VMC 需跳起(方案A,与 VMC+SRL 区分), 纯PPO 落地存活率不能为 0, SRL+VMC 效果最好
**关联**: [19_four_algo_training_compare](2026-08-06/19_four_algo_training_compare.md)

---

## 问题描述

四算法对比后:
1. **PPO+VMC(纯VMC无参考)不跳**(air 0.5%)——策略在虚拟腿空间无法自行发现跳跃时序
2. **纯PPO 跳 0.5m 但落地必摔**(存活 0%)——落地冲击导致终止

## 解决方案

### 1. 方案A: PPO+VMC 加入 SLIP-FSM 参考(与 VMC+SRL 区分)
- **重构**: 把 FSM 状态机 + 阶段腿长参考 + 阶段相关增益从 jump_srl_vmc 移入 `XqRobotWLJumpVMCFlatEnv`(jump_vmc.py)
- **PPO+VMC 用"参考 + 策略全动作"**: `final_L0 = FSM参考 + 策略全动作`(策略有完整权限叠加)
- **VMC+SRL 用"参考 + 策略残差"**: `final_L0 = FSM参考 + 0.15×策略`(残差模式,保持原样)
- 两者 obs 均含 FSM 特征(387/486)
- 开环验证: PPO+VMC 零策略跳 0.17m(FSM 参考生效)

### 2. 纯PPO 落地存活修复
| 参数 | 旧 | 新 | 理由 |
|------|-----|-----|------|
| `entropy_coef` | 0.005 | **0.002** | 降探索噪声(原 std 饱和到 1.0 致落地乱) |
| `landing_soft` | 15 | **30** | 更强软着陆奖励 |
| `landing_recovery` | 无 | **5.0** | 新增: 落地后双轮着地+直立+高度恢复奖励 |
| `jump_height` | 12 | 10 | 略降, 缓解落地冲击 |
| `vertical_thrust` | 30 | 25 | 略降, 降低冲击 |

`landing_recovery` 新奖励函数加入 jump.py: 双轮接触 + 直立(tilt 小) + 高度接近目标 → 直接鼓励落地后恢复。

## 修改文件

| 文件 | 修改内容 |
|------|---------|
| `src/unilab/envs/locomotion/xqrobotwl/vmc.py` | XqRobotWLVMCConfig 增加 SLIP 参考参数(crouch_length 等) |
| `src/unilab/envs/locomotion/xqrobotwl/jump_vmc.py` | 加入 FSM+参考+阶段增益, step 用"参考+全动作" |
| `src/unilab/envs/locomotion/xqrobotwl/jump_srl_vmc.py` | 精简, 只保留残差 step + SRL 奖励集 |
| `src/unilab/envs/locomotion/xqrobotwl/jump.py` | 新增 `_reward_landing_recovery` |
| `conf/ppo/task/xqrobotwl_jump_flat/mujoco.yaml` | entropy/landing 调优 + landing_recovery |
| `tests/.../test_xqrobotwl_jump_vmc.py` | obs 369→387 断言更新 |

## 验证方法

- 10/10 测试通过
- PPO+VMC 开环零策略跳 0.17m(参考生效)
- 两训练已重启(PPO 落地修复版、VMC 参考全动作版), SRL/VMC+SRL 保留原模型
- 训练完成后跑四算法最终对比

## ⚠️ 事故记录

清理冒烟 run 目录时, `find -newermt` 误匹配父目录, `rm -rf` 误删了 **旧 PPO 和旧 VMC 的完整训练目录**(含 model_9999)。SRL/VMC+SRL 完好, 对比指标已存 `four_algo_comparison.json`。由于这两个模型本来就要重训(新配置), 影响有限: 旧模型指标保留在 JSON 作为基线, 重训后更新。

## 后续计划

- [ ] 监控两个重训(纯PPO 落地存活、PPO+VMC 参考跳跃)
- [ ] 完成后跑四算法最终对比, SRL+VMC 应最好
- [ ] 更新 four_algo_comparison.json

## 关联日志

- [19_four_algo_training_compare](2026-08-06/19_four_algo_training_compare.md)
- [18_srl_vmc_fsm_reference_dominant](2026-08-06/18_srl_vmc_fsm_reference_dominant.md)
