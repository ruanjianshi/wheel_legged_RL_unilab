# [04] CPO 独立 conf/cpo 目录 + 约束 value loss 数值修复

**日期**: 2026-08-07
**来源**: 用户要求 CPO 像 np3o/appo 一样有独立 `conf/cpo` 目录; 训练冒烟发现 constraint_value loss 爆炸
**关联**: [01_fsr_design](01_fsr_design.md) / [02_cpo_port](02_cpo_port.md) / [03_fsr_env_and_verify](03_fsr_env_and_verify.md)

---

## 问题描述

1. CPO 配置最初放在 `conf/ppo/task/` 下 (复用 train_rsl_rl.py), 用户要求独立
   `conf/cpo` 目录 + 独立入口, 与其他算法 (ppo/np3o/appo/offpolicy) 一致。
2. CPO 训练冒烟: `constraint_value loss` 爆炸 (11M → 665M), cost critic 发散。

## 根因分析

1. 目录: 算法应有自己的 conf 目录和入口脚本 (np3o 模式)。
2. value loss 爆炸: 约束代价是原始 F/T 幅值 (0-60N), cost critic 预测大数值;
   且 cost return 是 γ=0.99 的折扣累计, 有效尺度 ~1/(1-γ)=100 → 平方误差巨大。

## 解决方案

1. **独立 CPO 目录**:
   - `conf/cpo/config.yaml` (基于 ppo config: `algo.algo=cpo`, `algo_log_name=rsl_rl_cpo`,
     `algorithm.class_name=unilab.algos.torch.cpo:CPO`, 默认 task=fall_recovery)
   - `conf/cpo/task/xqrobotwl_fall_recovery_flat/mujoco.yaml` (从 conf/ppo 移入)
   - `scripts/training/train_cpo.py` (train_rsl_rl.py 副本, config_path=conf/cpo)
   - 启动脚本 → train_cpo.py, 日志 `logs/rsl_rl_cpo/`
2. **数值修复**:
   - env: 约束代价归一化到 [0,1] (`F/Fmax`, `T/Tmax`)
   - CPO: cost value loss 目标乘 `(1-γ)` (论文 cost_viol 同理), 缩到 ~1 尺度
   - `constraint_value_loss_coef` 1.0 → 0.1 (对齐 NP3O)

## 修改文件

| 文件 | 修改内容 |
|------|---------|
| `conf/cpo/config.yaml` | 新建: CPO 独立算法配置 |
| `conf/cpo/task/xqrobotwl_fall_recovery_flat/mujoco.yaml` | 从 conf/ppo 移入, coef 0.1 |
| `scripts/training/train_cpo.py` | 新建: config_path=conf/cpo |
| `shell/xqrobotwl/launch_ppo_fall_recovery.sh` | → train_cpo.py, 日志 rsl_rl_cpo |
| `src/unilab/envs/locomotion/xqrobotwl/fall_recovery.py` | 约束代价归一化 [0,1] |
| `src/unilab/algos/torch/cpo.py` | cost value loss 目标 × (1-γ) |
| `scripts/xqrobotwl/smoke_fall_recovery.py` / `warmstart*.py` | 路径 → conf/cpo / rsl_rl_cpo |
| 删除 `conf/ppo/task/xqrobotwl_fall_recovery_flat/` | 已移入 conf/cpo |

## 验证方法

`uv run mjpython scripts/training/train_cpo.py task=xqrobotwl_fall_recovery_flat/mujoco training.task_name=XqRobotWLFallRecoveryFlat algo.num_envs=16 algo.max_iterations=60`

## 评估结果

- constraint_value loss: 爆炸 665M → **稳定 ~250** (RMSE ~16, 可接受)
- constraint_viol loss: 稳定 ~0.004 (约束罚项正常)
- reward 10.93 (↑), ep_len **653 步** (↑↑, 机器人在恢复+保持平衡)
- 无 NaN; `run_config.json` 确认 `cpo:CPO`

## 后续计划

- [ ] 完整训练 (warmstart 自 walk) + 确定性评估
- [ ] cost critic 精度提升 (mini-batch 对齐检查, NP3O 同款 `[:bs]` 切片的潜在错位)
- [ ] Fmax/μ/force_end_iters 力参数调优

## 关联日志

- [01_fsr_design](01_fsr_design.md) / [02_cpo_port](02_cpo_port.md) / [03_fsr_env_and_verify](03_fsr_env_and_verify.md)
