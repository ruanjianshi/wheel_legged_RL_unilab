# [22] v6 残留转圈 → v7: 收紧 wheel_symmetry + 门控站姿奖励 (治蹲姿)

**日期**: 2026-08-09
**来源**: v6 model_6000 轮子空转已治 (差速 474→3.9), 但转圈残留 (yaw 299°) + 腿仍蹲姿
**关联**: [21_v6_wheel_symmetry](21_v6_wheel_symmetry.md)

---

## 问题描述 (v6 model_6000 诊断)

v6 甜点位 model_6000 (保持 2.05s, |gyro|=0.66 摇摆治好), 但:

| 指标 | 实测 | 问题 |
|---|---|---|
| L_wheel | +1.21 rad/s | 左轮持续慢正转 |
| R_wheel | -0.19 rad/s | 右轮微反转 |
| 差速 (方向恒定) | +1.40 rad/s | **小差速但方向恒定 → 累积转圈 yaw 299°** |
| L_pitch | -0.83 (自然 +0.08) | 蹲姿后摆 |
| R_knee | +0.91 (自然 +0.02) | 蹲姿深屈 |
| \|gyro\| | 0.66 rad/s | ✅ 摇摆治好 |

**两个残留问题**:
1. **转圈**: 不是狂转 (v5), 而是**左轮持续 +1.2 rad/s 的慢差速**, 方向恒定累积。
   v6 的 wheel_symmetry 阈值 /20 太宽 — 差速 1.4 时得 0.93 (几乎满分), 没抓到。
2. **蹲姿**: L_pitch -0.83, R_knee +0.91 — 双腿对称深蹲, 产生持续 yaw 力矩 → 转圈。
   dof_pos -0.5 太弱压不住 (强了压坏起身 v4 教训)。

## 解决方案 (v7)

1. **wheel_symmetry 阈值 /20 → /5**: 差速 1.4 时得 0.76 (有梯度), 抓"小而恒定"差速。
2. **新增门控站姿奖励 `stand_pose`** (scale 15): `height_ok × up_ok × exp(-||dof-standing_angles||²/σ²)`
   — **正奖励 + 门控**, 只在站立期激活 (起身期≈0 不影响), 站对自然站姿给分, 把蹲姿拉直。
   σ²=2*(0.25)²。解决 dof_pos 的"强压坏起身/弱压不住蹲姿"两难。

## 修改文件

| 文件 | 修改内容 |
|------|---------|
| `src/unilab/envs/locomotion/xqrobotwl/fall_recovery.py` | wheel_symmetry /20→/5; 新增 `_reward_stand_pose` (门控正奖励) 并注册 |
| `conf/cpo/task/xqrobotwl_fall_recovery_flat/mujoco.yaml` | ru/rs 加 `stand_pose: 15.0` |

## 验证方法

- wheel_symmetry @ 差速 0/1.4/5/20 = 1.0/0.76/0.37/0.02 ✓ (抓小差速)
- stand_pose: 自然站姿→1.0, 蹲姿→0.0 ✓
- env 创建 OK ✓
- 训练 v7 (从零, 1024 envs, 8000 iter) run `fix_pose_v7`
- 达标: 站立轮速差 < 5 + yaw 累计 < 30° + 腿角≈自然站姿 (|偏差|<0.3) + 保持 > 1s

## 后续计划

- [ ] v7 每 1000 iter 检查 (重点: stand_pose 奖励生效 → 蹲姿拉直, yaw 不转)
- [ ] 达标后按姿态评估 + 渲染 4 姿态
- [ ] 交付

## 关联日志

- [21_v6_wheel_symmetry](21_v6_wheel_symmetry.md) — v6 (狂转治好, 残留小差速+蹲姿)
- [20_v5_gentle_posture](20_v5_gentle_posture.md) — v5
