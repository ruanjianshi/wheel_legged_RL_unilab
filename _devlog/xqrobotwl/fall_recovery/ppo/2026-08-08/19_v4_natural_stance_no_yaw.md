# [19] 用户反馈: 恢复后腿一前一后 + 转圈 + 轮点地 → 自然站姿 dof_pos + no_yaw

**日期**: 2026-08-08
**来源**: 用户看 v3b model_4000 视频后反馈 "恢复后姿态很大问题: 腿变一前一后,
大致以右腿为圆心转圈, 左腿在前右腿在后, 轮子一直在点地"
**关联**: [18_delivery_v3b_standing_time](18_delivery_v3b_standing_time.md)

---

## 问题描述

v3b 站立时间解决了 (3.4-6.5s), 但恢复后**站立姿态严重错误**:
1. **腿一前一后**: 站立时 L_hip_pitch=-0.75, R_hip_pitch=-0.40, 左右差 0.35 rad (应≈0)
2. **以右腿为圆心转圈**: 站立期间 yaw 累计旋转 **260°** (几乎一整圈)
3. **轮子点地**: 不对称站姿 + 前倾导致轮子间歇接触

## 根因分析 (诊断数据)

| 指标 | 实测 | 名义 | 结论 |
|---|---|---|---|
| L_hip_pitch | -0.75 | +0.15 | 左腿大幅后摆 |
| R_hip_pitch | -0.40 | -0.15 | 右腿大幅后摆 |
| 左右错位 | -0.35 rad | 0 | 一前一后 |
| yaw 累计旋转 | 260° | 0 | 以右腿为轴转圈 |
| L_knee | -0.66 | — | 姿态变形 |

**★ 关键发现 — dof_pos 目标错误**: `DEFAULT_LEG_ANGLES` 膝盖符号与自然站姿**相反**。
walk 模型实测自然站立腿角 = `[+0.102, +0.083, -0.079, +0.013, -0.108, +0.019]`
→ base_z≈0.518m。而 DEFAULT_LEG_ANGLES (L_knee=+0.15) → 0.474m。
`dof_pos` 一直把腿往 DEFAULT 拉 (rs -0.1), 与自然站姿冲突, 压不住错误姿态。

**惩罚太弱**: rs dof_pos -0.1 (当前站姿误差 1.586 → 只罚 0.16), leg_bias -0.02 (近零)。
**无朝向约束**: 没有 yaw 静止奖励 → 转圈无梯度。

## 解决方案 (v4)

1. **dof_pos 目标改 `standing_angles`** (新增配置, walk 实测自然站姿 → 0.518m),
   与 base_height 目标 0.52 兼容 (DEFAULT→0.474 是冲突根源)。
2. **rs dof_pos -0.1 → -2.0**: 误差 1.586 → 罚 3.17 (有效拉回自然站姿)。
3. **rs leg_bias -0.02 → -0.5**: 罚左右不对称。
4. **新增 `no_yaw` 奖励** (scale 8): `exp(-|gyro_z|/0.8)` 罚 yaw 角速度, 防转圈。

## 修改文件

| 文件 | 修改内容 |
|------|---------|
| `src/unilab/envs/locomotion/xqrobotwl/fall_recovery.py` | dataclass 加 `standing_angles`; `_reward_dof_pos` 目标改 standing_angles; 新增 `_reward_no_yaw`; update_state 注入 info["standing_angles"] |
| `conf/cpo/task/xqrobotwl_fall_recovery_flat/mujoco.yaml` | 加 `standing_angles`; rs dof_pos -0.1→-2.0; rs leg_bias -0.02→-0.5; ru/rs 加 no_yaw: 8.0 |

## 验证方法

- no_yaw @ gyro_z=0/0.5/2/4/6 = 1.0/0.54/0.08/0.01/0.001 ✓ (转越快分越低)
- dof_pos 自然站姿→0, 错误姿态→1.6 ✓
- env 创建 OK, standing_angles 注入 info ✓
- 训练 v4 (从零, 1024 envs, 8000 iter) run `fix_pose_v4`
- 达标: 站立腿角≈自然站姿 (|差|<0.2), yaw 累计旋转 < 30°, 保持 > 1s, 4 姿态都行

## 后续计划

- [ ] v4 每 1000 iter 检查趋势
- [ ] 达标后按姿态评估 (重点: 腿角对称 + yaw 不转) + 渲染视频
- [ ] 交付甜点位

## 关联日志

- [18_delivery_v3b_standing_time](18_delivery_v3b_standing_time.md) — v3b 交付 (站立时间解决, 姿态仍错)
- [17_v3_gyro_stillness](17_v3_gyro_stillness.md) — v3 修复 (角速度静止)
