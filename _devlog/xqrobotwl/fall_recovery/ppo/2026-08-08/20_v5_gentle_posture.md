# [20] v4 姿态约束过强致保持崩 → v5 温和约束 (v3b 平衡 + 轻姿态惩罚)

**日期**: 2026-08-08
**来源**: v4 训练完成, 保持时间从 v3b 的 3.4-6.5s 跌到 0.2-0.4s
**关联**: [19_v4_natural_stance_no_yaw](19_v4_natural_stance_no_yaw.md)

---

## 问题描述 (v4 训练结果)

v4 加了 dof_pos -2.0 / leg_bias -0.5 (自然站姿目标) + no_yaw 8.0,
训练后**恢复率 100% 但保持时间崩了**:

| checkpoint | 恢复率 | 最长连续站立 |
|---|---|---|
| 3000 | 100% | 0.37s |
| 4000 | 100% | 0.18s |
| 5000 | 100% | 0.37s |
| 6000 | 100% | 0.29s |
| 7000 | 100% | 0.35s |
| 7999 | 100% | 0.38s |

v3b 同期是 3.4-6.5s。**v4 强约束把平衡保持压坏了**。

## 根因分析

训练奖励: settle=1.14, stand_still=0.86, no_yaw=1.07 都很高,
但 **recover_complete=0.20 几乎为 0** — 策略"会站着"(settle/no_yaw 高) 但
**没学会"起身后保持站住"**。诊断 model_5000: L_knee=-0.756 (自然 -0.079),
**起身蓄力用深蹲 (屈膝 -0.7), dof_pos -2.0 强惩罚把起身压住** → 恢复后站不稳。

**本质矛盾**: 恢复起身必须大幅屈膝 (L_knee 0→-0.7), 但 dof_pos 拉向自然站姿
(屈膝浅) — 强惩罚 (-2.0) 在起身期就压制腿部动作, 破坏恢复动力学。

## 解决方案 (v5)

**v3b 的平衡结构保留 (它让保持达 3.4-6.5s) + 温和姿态约束**:

1. **dof_pos -2.0 → -0.5**: 目标保留 standing_angles (正确自然站姿), 但强度温和,
   不压起身。
2. **leg_bias -0.5 → -0.1**: 温和罚左右不对称。
3. **保留**: settle 指数 gyro 门 (20), stand_still (10), no_yaw (8),
   recover_complete (100) — v3b 平衡结构原样。

## 修改文件

| 文件 | 修改内容 |
|------|---------|
| `conf/cpo/task/xqrobotwl_fall_recovery_flat/mujoco.yaml` | rs dof_pos -2.0→-0.5; leg_bias -0.5→-0.1 |

(代码 standing_angles 目标 + no_yaw 奖励已在 v4 加好, v5 只调权重。)

## 验证方法

- 训练 v5 (从零, 1024 envs, 8000 iter) run `fix_pose_v5`
- 达标: 保持时间 ≥ 1s + 腿角≈自然站姿 (|偏差|<0.3) + yaw 累计 < 30° + 4 姿态都行

## 后续计划

- [ ] v5 每 1000 iter 检查 (重点: 保持时间 ≥ 1s 且姿态对称)
- [ ] 达标后按姿态评估 + 渲染
- [ ] 交付

## 关联日志

- [19_v4_natural_stance_no_yaw](19_v4_natural_stance_no_yaw.md) — v4 (姿态约束过强, 保持崩)
- [18_delivery_v3b_standing_time](18_delivery_v3b_standing_time.md) — v3b (平衡好但姿态丑)
