# [03] FTSR 环境实现细节 + 冒烟验证 + 终止门控修复

**日期**: 2026-08-07
**来源**: FTSR 方案 (01) + CPO (02) 落地, 冒烟验证
**关联**: [01_fsr_design](01_fsr_design.md) / [02_cpo_port](02_cpo_port.md)

---

## 问题描述

FTSR env 重写后冒烟遇到两个问题: ① MPS float64 dtype 崩溃; ② 所有 episode
1 步就终止。

## 根因分析

1. **float64 崩溃**: `_constraint_costs` 初始化硬编码 `np.float64`, MPS 不支持
   float64 → 改 `get_global_dtype()` (float32)。
2. **1 步终止**: 初始倒地 tilt≈90° > max_tilt(55°), tilt 终止条件立即触发。
   倒地是**合法起始态**, 不能按倾覆终止。

## 解决方案

1. dtype: 统一 float32。
2. **has_recovered 门控**: base_z 达 h_cmd1 (0.35) 锁存 `_has_recovered=True`;
   **恢复前** 不按倾覆/塌缩终止 (倒地是起始态), 只按贴地超时 (idle_ground_time);
   **恢复后** 再倾覆/塌缩才终止。idle_ground_time 10s→6s (在 max_episode 10s 前
   终止卡死 episode, 加快节奏)。
3. 力引导细节: F=(1-e^{-μ(h_cmd-h)})·sat(1-steps/force_end)·Fmax 向上;
   T 用旋转向量 (up→[0,0,1] 轴×角, 裁剪 1rad) 对齐直立; 经 `apply_body_wrench`
   施加到 base_link; 幅值 [|F|,|T|] 进 `info["constraint_costs"]` → CPO。

## 修改文件

| 文件 | 修改内容 |
|------|---------|
| `src/unilab/envs/locomotion/xqrobotwl/fall_recovery.py` | _constraint_costs float32; has_recovered 终止门控; idle 6s |
| `conf/ppo/task/xqrobotwl_fall_recovery_flat/mujoco.yaml` | idle_ground_time 6.0 |
| `scripts/xqrobotwl/smoke_fall_recovery.py` | 重写: 多姿态/力引导/分阶段验证 |

## 验证方法

1. `uv run mjpython scripts/xqrobotwl/smoke_fall_recovery.py`
2. CPO 训练冒烟 (40 iter × 16 env)

## 评估结果

- Smoke: obs 297/324 ✓; 多姿态复位 (up 分布 x/y, 4 姿态覆盖) ✓; 力引导施加
  (F≈38N/T≈5Nm, 399/400 步) ✓; reward 有限 ✓
- CPO 训练: 40 iter, episode **1→415 步**, reward 0.02→8.79, 无 NaN
- 力幅值: Fmax=60N 时实际 F≈38N (height_term 0.63), 远低于自重 183N — 是辅助非
  吊车, 腿需承担主要举升 (符合设计)

## 后续计划

- [ ] 完整训练 (warmstart 自 walk, 数万 iter) + 确定性评估
- [ ] Fmax/μ/force_end_iters 调优; 力衰减后 (t_coeff→0) 恢复成功率验证

## 关联日志

- [01_fsr_design](01_fsr_design.md) / [02_cpo_port](02_cpo_port.md)
