# [02] 移植 CPO 算法 (惩罚函数法) — 约束代价由 env 提供

**日期**: 2026-08-07
**来源**: FTSR 方案确定 (01), 论文用 CPO 把辅助力/力矩作为可优化约束
**关联**: [01_fsr_design](01_fsr_design.md) / [03_fsr_env](03_fsr_env.md)

---

## 问题描述

论文用 CPO (Constrained Policy Optimization) 把外部辅助力 F 和力矩 T 表述为
可优化约束 (C1=F, C2=T, d→0), 引导策略逐步降低依赖。项目是 rsl_rl PPO, 无 CPO。
需移植。

## 根因分析

项目已有 **NP3O** (带 cost critic + viol_loss 的约束 PPO), 结构上就是惩罚函数法
CPO。差异: ① NP3O 从物理观测提取二进制 cost violation; ② 论文约束代价是 env
施加的辅助力/力矩幅值 (连续值)。→ 新增 CPO 类: 约束代价由 **env 提供**
(extras), β 罚因子代替 NP3O 的 K 退火。

## 解决方案

1. **`src/unilab/algos/torch/cpo.py`** (新建, 继承 FinalObservationAwarePPO):
   - 2 约束 (F, T), 约束 cost critic (512/256/128)
   - `process_env_step` 读 `extras["constraint_costs"]` (env 施加的 F/T 幅值)
   - `compute_returns` 约束 GAE → cost_adv/cost_viol (归一化)
   - `update`: PPO surrogate + β·ReLU(cost_surr + (1-γ)(J_Ci - d_i)) + cost value loss
   - β 温和增长退火 (β_init 0.001, 论文基线)
2. **`src/unilab/training/rsl_rl.py`**: wrapper `step` 转发
   `state.info["constraint_costs"]` → `infos["constraint_costs"]` (float32, 修 MPS dtype)
3. **配置**: `conf/ppo/task/xqrobotwl_fall_recovery_flat/mujoco.yaml`
   `algo.algorithm.class_name: unilab.algos.torch.cpo:CPO` + num_constraints=2,
   beta_init=0.001, d_values=[0,0]
4. 训练入口复用 `scripts/training/train_rsl_rl.py` (class_name 选算法)

## 修改文件

| 文件 | 修改内容 |
|------|---------|
| `src/unilab/algos/torch/cpo.py` | 新建: 惩罚函数法 CPO |
| `src/unilab/training/rsl_rl.py` | wrapper 转发 constraint_costs 到 extras |

## 验证方法

`uv run mjpython scripts/training/train_rsl_rl.py task=xqrobotwl_fall_recovery_flat/mujoco training.task_name=XqRobotWLFallRecoveryFlat algo.num_envs=16 algo.max_iterations=40`

## 评估结果

- CPO 被选中 (run_config.json: `cpo:CPO`) ✓
- 训练日志出现 `constraint_value loss` + `constraint_viol loss` (cost critic 在学习, 约束代价流动正常) ✓
- 无 NaN; episode 415 步 (终止门控修复后)

## 后续计划

- [ ] β 自适应策略调优 (当前温和增长)
- [ ] 约束代价归一化尺度验证 (F 幅值 ~0-60N vs T ~0-8Nm)

## 关联日志

- [01_fsr_design](01_fsr_design.md) / [03_fsr_env](03_fsr_env.md)
- NP3O (约束 PPO 模板): `src/unilab/algos/torch/np3o.py`
