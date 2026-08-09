# [17] 用户反馈: 站立时间太短 → settle 加角速度静止项 (罚摇摆)

**日期**: 2026-08-08
**来源**: 用户看 v2 model_4000 视频后反馈 "还需要改进, 跌倒恢复的站立时间太短"
**关联**: [16_delivery_4pose_posefix](16_delivery_4pose_posefix.md)

---

## 问题描述

v2 model_4000 恢复率 80-95%, 高度/漂移已修复, 但**站立保持时间 0.33-0.40s**,
远低于 0.5s 达标线, 用户要求更长站立。

## 根因分析 (诊断数据)

对 model_4000 逐帧诊断站立中断:

| 指标 | 值 | 结论 |
|---|---|---|
| 站立时角速度 \|gyro\| | **mean 6.07, p90 9.02 rad/s** | 剧烈摇摆! |
| 中断原因 | 倾斜 5 / 塌陷 16 / 轮离地 10 | 摆动累积成倾覆/塌缩 |

**根因**: `_reward_settle` 只罚垂直速度 (`still_v`), **完全不罚角速度 (gyro)** →
机器人站立时以 6-9 rad/s 疯狂摇摆不受任何惩罚, 几帧后就摆倒。
(v1 曾把水平静止乘进 settle 杀掉恢复; 但**角速度**是另一回事 — 平衡的本质是角速度小,
且 settle 已被 height_ok+up_ok 门控只在站立期激活, 加 gyro 项不会影响起身。)

## 解决方案 (v3, 两次迭代)

**v3a (硬门, 失败)**: `still_g = 1 - clip(|gyro|/0.5, 0, 1)` + settle 15→20。
训练到 iter 3000 评估: 保持 0.27s (比 v2 的 0.48s 还差!)。**硬门是"全有或全无"** —
gyro 6→3→1 rad/s 全程都是 0 分, 中间无梯度, 策略从 6 要"一步跳到"0.5 以下才得分,
学不到。v1 band 0.08 教训重演。

**v3b (指数衰减, 正确)**: `still_g = exp(-|gyro|/2.0)` —
gyro=0→1.0, 0.5→0.78, 2→0.37, 4→0.14, 6→0.05 — **全程连续梯度**, 最优在 0,
策略持续压低摇摆。settle scale 保持 20。

## 修改文件

| 文件 | 修改内容 |
|------|---------|
| `src/unilab/envs/locomotion/xqrobotwl/fall_recovery.py` | `_reward_settle` 加 `still_g = exp(-|gyro|/2.0)` (指数衰减角速度项) |
| `conf/cpo/task/xqrobotwl_fall_recovery_flat/mujoco.yaml` | settle scale 15→20 (ru/rs) |

## 验证方法

- 数值单测: settle @ gyro=0/0.5/2/4/6 = 1.0/0.78/0.37/0.14/0.05 ✓ (全程梯度)
- 训练 v3b (从零, 1024 envs, 8000 iter) run `fix_pose_v3b`
- 达标: 保持时间 ≥ 0.5s (目标 > 1s), 恢复率 ≥ 80%, 4 姿态都行

## 后续计划

- [ ] v3b 每 1000 iter 检查趋势 (重点: 保持时间上升)
- [ ] 达标后按姿态评估 + 渲染 4 姿态视频
- [ ] 交付甜点位

## 关联日志

- [16_delivery_4pose_posefix](16_delivery_4pose_posefix.md) — v2 交付 (保持 0.33-0.40s 太短)
- [15_v2_redo_overshoot_band_standstill](15_v2_redo_overshoot_band_standstill.md) — v2 修复
