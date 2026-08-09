# 19 四算法跳跃训练完成 + 对比评估

**日期**: 2026-08-06
**来源**: 论文四算法对比训练全部完成(PPO/SRL/VMC/SRLVMC,各 10000 iter)
**关联**: [16_ppo_vmc_srl_vmc_algorithms](2026-08-06/16_ppo_vmc_srl_vmc_algorithms.md), [17_vmc_jump_param_fix](2026-08-06/17_vmc_jump_param_fix.md), [18_srl_vmc_fsm_reference_dominant](2026-08-06/18_srl_vmc_fsm_reference_dominant.md)

---

## 训练配置(四者统一公平)
- 1024 env × 10000 iter,`jump_curriculum_end: 0`(无热身)
- 四者共用 phase-gated 跳跃奖励与命令系统
- VMC 版本修复后参数: feedforward_force=110, action_scale_l0=0.12; SRL+VMC 用 FSM 参考主导 + 策略残差

## 最终四算法对比(4速度 × 4回合)

| 算法 | 跳高(m) | air(真跳) | 存活率 | 跳跃成功率 |
|------|------|------|------|------|
| **SRL** | 0.423 | **21.6%** | **100%** | **100%** |
| **纯PPO** | 0.496 | 20.3% | **0%** | 100% |
| **SRL+VMC** | 0.056 | 8.5% | 69% | 100% |
| **纯VMC** | 0.032 | 0.5% | 62% | 0% |

## 核心结论(论文支撑)

1. **SRL(SLIP-FSM 前馈)明显最优**:跳高与纯 PPO 相当(0.42 vs 0.50m)、air 相当(21.6% vs 20.3%),但**存活率 100% vs 0%**——SLIP-FSM 前馈的价值在于跳跃后的稳定与落地恢复,而非跳得更高。
2. **纯 PPO 跳得最高但不稳**:0.50m 高、20% air,但每次落地都摔(存活 0%)。无热身课程下纯 PPO 过度探索、落地协调差。
3. **SRL+VMC 学会真跳但幅度有限**:air 8.5%、69% 存活——FSM 参考让 VMC 能腾空,但力控的腾空幅度低于关节位置控制。FSM 参考主导(参考项目模式)是 VMC 学跳的关键(修复前 2% → 修复后 8.5%)。
4. **纯 VMC 未实现腾空**:air 0.5%。力控难以打破轮地接触("先有鸡还是先有蛋":FSM 需确认腾空才进飞行收腿,但收腿才是拉轮离地的关键)。这是 VMC 力控在轮腿平台的已知难题。

## 修改文件

| 文件 | 修改内容 |
|------|---------|
| `scripts/compare_jump.py` | 新建,四算法对比评估(多速度×多回合,air 按实际存活步数计) |
| `scripts/verify_jump.py` | 修复测量(air_frac 为主指标;确认 standing 参考一致性) |
| `jump_management/results/four_algo_comparison.json` | 新建,对比结果存档 |

## 训练期间的关键修复(前序日志)
- [17] feedforward 80→110、action_scale_l0 0.05→0.12(否则 VMC 物理上无法起跳)
- [18] SRL+VMC 混合翻转:FSM 参考主导 + 策略残差(否则策略抵消参考,air 2% → 8.5%)

## 后续计划

- [ ] 生成论文图表(训练曲线对比、跳跃轨迹、指标表)
- [ ] 纯 VMC 腾空问题深调(推力爆发/轮控/飞行时序)——可作后续工作
- [ ] 评估 runner 扩展两个 VMC 任务到 jump_management/evaluate

## 关联日志

- [18_srl_vmc_fsm_reference_dominant](2026-08-06/18_srl_vmc_fsm_reference_dominant.md)
- [17_vmc_jump_param_fix](2026-08-06/17_vmc_jump_param_fix.md)
- [16_ppo_vmc_srl_vmc_algorithms](2026-08-06/16_ppo_vmc_srl_vmc_algorithms.md)
