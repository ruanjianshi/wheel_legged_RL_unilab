# [21] 用户反馈"转圈更明显" → 轮速约束 wheel_symmetry + wheel_speed (v6)

**日期**: 2026-08-09
**来源**: 用户看 v5 model_7000 视频后反馈 "效果不好, 转圈现象更加明显了"
**关联**: [20_v5_gentle_posture](20_v5_gentle_posture.md)

---

## 问题描述

v5 model_7000 保持 3.37s 但**转圈更明显**。用户明确反馈。

## ★ 决定性根因 (轮速诊断)

对 v5 model_7000 站立时测量**左右轮实际速度**:

| 指标 | 实测 | 判断 |
|---|---|---|
| L_wheel 速度 | -91 rad/s | 左轮慢转 |
| R_wheel 速度 | **-565 rad/s** | 右轮狂转 (≈90 转/秒!) |
| 差速 (L-R) | **+474 rad/s** | **巨大差速 → 原地转圈** |
| gyro_z (yaw) | -33°/s | 持续旋转 |
| base_y 漂移 | 1.97m | 转圈横向漂移 |

**根因**: 右轮以 -565 rad/s 疯狂空转, 与左轮差速 474 rad/s → 机器人以右轮为轴
原地打转。轮子是速度控制 (kv=1 velocity actuator), **fall_recovery 完全没有轮速约束**。

**对比 walk 模型 (能稳定站立)**: walk 配置有
`wheel_action_rate: -0.005` (罚轮动作跳变) + `wheel_symmetry: -0.5` (罚左右轮速不等)。
fall_recovery 遗漏了这两个约束 → 策略自由输出巨大轮速空转。

## 解决方案 (v6)

1. **`wheel_symmetry` 奖励** (scale 5): `exp(-|wL-wR|/20)` — 轮速差小则 1, 差大则 0。
   直接用实际轮速差 (walk 用动作差, 恢复任务用轮速更直接)。
2. **`wheel_speed` 奖励** (scale 3): `exp(-(|wL|+|wR|)/100)` — 站立应几乎静止。
3. update_state 把 `dof_vel[:, 6:8]` 存进 `info["wheel_vel"]` 供奖励读取。

## 修改文件

| 文件 | 修改内容 |
|------|---------|
| `src/unilab/envs/locomotion/xqrobotwl/fall_recovery.py` | update_state 存 wheel_vel; 新增 `_reward_wheel_symmetry` + `_reward_wheel_speed` (指数衰减); 注册 |
| `conf/cpo/task/xqrobotwl_fall_recovery_flat/mujoco.yaml` | ru/rs 加 `wheel_symmetry: 5.0`, `wheel_speed: 3.0` |

## 验证方法

- 数值单测: wheel_symmetry @ 差速 0/20/150/565 = 1.0/0.37/0.001/0.0 ✓
  wheel_speed @ 总速 0/20/150/565 = 1.0/0.82/0.22/0.004 ✓
- env 创建 OK, wheel_vel 注入 info ✓
- 训练 v6 (从零, 1024 envs, 8000 iter) run `fix_pose_v6`
- 达标: 站立时轮速 < 50 rad/s + 差速 < 20 + yaw 累计 < 30° + 保持 > 1s

## 后续计划

- [ ] v6 每 1000 iter 检查 (重点: wheel_symmetry/wheel_speed 奖励生效)
- [ ] 达标后按姿态评估 + 渲染
- [ ] 交付

## 关联日志

- [20_v5_gentle_posture](20_v5_gentle_posture.md) — v5 (保持 3.37s 但转圈)
- [19_v4_natural_stance_no_yaw](19_v4_natural_stance_no_yaw.md) — v4 (姿态约束)
