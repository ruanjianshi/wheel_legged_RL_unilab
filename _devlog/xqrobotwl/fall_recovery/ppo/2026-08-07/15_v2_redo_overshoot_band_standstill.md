# [15] v1 过度约束致恢复率崩 → v2: 宽松超高 band + 独立 stand_still 项

**日期**: 2026-08-07
**来源**: [14](14_fix_pose_overshoot_drift.md) 的 v1 修复训练后恢复率持续下滑
**关联**: [14_fix_pose_overshoot_drift](14_fix_pose_overshoot_drift.md)

---

## 问题描述 (v1 训练结果, run 2026-08-07_19-56-29)

v1 三个改动 (band 0.08 + settle 水平乘法 + dof_pos -0.5) 训练后**恢复率一路崩**:

| iter | 恢复率 | 保持 | 漂移 | 高度 |
|---|---|---|---|---|
| 3000 | 80% | 0.33s | 0.89m | 0.61 |
| 4000 | 85% | 0.32s | 0.82m | 0.60 |
| 5000 | 60% | 0.39s | 0.90m | 0.57 |
| 6000 | **55%** | 0.32s | 0.85m | 0.54 |

训练日志 recover_complete: 34.9 → 8.3 → 3.3 (**一路降**)。对比 v0 fromscratch 是
"先升 100% 再发散" — v1 是"恢复率根本学不起来", 属**奖励设计问题不是甜点位问题**。
(且 5000 按姿态测 4 姿态都能恢复 70-100%, 说明不是某姿态缺失。)

## 根因分析

v1 三个改动**全部在跟恢复动作打架**:

1. **超高惩罚 band 0.08 太紧**: 起身自然冲高到 0.60-0.61 就被罚到 0.25 分,
   `rise * (1-over)` 把"推过 0.55"这段梯度杀掉了 → 学不起来。
2. **settle 水平乘法**: `still_h` 乘进 settle, 但恢复期 base 水平运动本来就大
   (推腿/起身), settle 被乘到≈0 → "站稳"信号整个消失。
3. **dof_pos -0.5**: 恢复期腿部大摆动 (0→0.5 rad), -0.5 惩罚太重。

**★ 关键新证据 (自然站姿高度)**: walk 模型真实站立 median = **0.522m**
(实测 L_knee=-0.111, R_knee=+0.046, 接近伸直)。DEFAULT_LEG_ANGLES (弯膝 ±0.15)
站高 0.474m。→ **h_cmd2=0.55 本身偏高**, 要到 0.55 必须"升直腿"。用户要的是
walk 自然站姿 ≈0.52。

## 解决方案 (v2)

1. **h_cmd2: 0.55 → 0.52** — 对齐 walk 自然站立 median 0.522。
2. **超高惩罚 band: 0.08 → 0.12** — 起身冲高 0.60 仍得 0.33 (不杀起身),
   但 0.596 过伸站姿得 0.37、0.67+ 归零。
3. **settle 还原为 v12 版 (仅垂直静止)** — 水平乘法取消。
4. **新增独立 `stand_still` 奖励** (scale 10, 加法项): 直立 × 接近站立高度 ×
   水平速度 < 0.5 m/s。防"一直后退"但不乘死 settle。
5. **dof_pos -0.5 → -0.1** — 还原, 不惩罚恢复期腿摆动。

## 修改文件

| 文件 | 修改内容 |
|------|---------|
| `src/unilab/envs/locomotion/xqrobotwl/fall_recovery.py` | base_height band 0.08→0.12; settle 还原垂直静止; 新增 `_reward_stand_still` 并注册 |
| `conf/cpo/task/xqrobotwl_fall_recovery_flat/mujoco.yaml` | h_cmd2 0.55→0.52; vel_limit 0.55→0.52; 加 stand_still:10.0 (ru/rs); dof_pos rs -0.5→-0.1 |

## 验证方法

- 数值单测: base_height @ 0.52/0.60/0.596 = 1.0/0.33/0.37 ✓; settle 恒 1 (不再被水平乘死) ✓;
  stand_still @ v=0/后退0.3/前冲0.6 = 1.0/0.4/0.0 ✓
- env 创建 OK, h_cmd2=0.52 ✓
- 训练 v2 (从零, 1024 envs, 8000 iter) run `fix_pose_v2`
- 达标: 恢复率 > 80% + 站立高度 ≈0.52 + 漂移 < 0.5m + 保持 > 0.5s

## 后续计划

- [ ] v2 每 1000 iter 检查趋势 (重点: 恢复率不再崩 + 高度贴近 0.52)
- [ ] 达标后按姿态评估 + 渲染 4 姿态视频
- [ ] 交付甜点位

## 关联日志

- [14_fix_pose_overshoot_drift](14_fix_pose_overshoot_drift.md) — v1 (过度约束, 恢复率崩)
- [13_delivery_100pct_recovery_stable](13_delivery_100pct_recovery_stable.md) — v0 (能恢复但姿态差)
