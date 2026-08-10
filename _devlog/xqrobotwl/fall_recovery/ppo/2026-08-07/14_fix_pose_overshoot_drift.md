# [14] 用户反馈: 恢复后姿态差 (升直腿过高 + 一直后退) → base_height 超高惩罚 + settle 水平静止

**日期**: 2026-08-07
**来源**: 用户看 model_5000 跟踪视频后反馈 "恢复后姿态太差, 莫名升直腿, 高度太高 (0.67m 非站姿高度), 还一直后退, 需要姿态稳定 + 高度正常站立"
**关联**: [13_delivery_100pct_recovery_stable](13_delivery_100pct_recovery_stable.md)

---

## 问题描述

model_5000 虽然 100% 能恢复, 但恢复后姿态错误:

1. **升直腿, 高度过高**: 站立时 base_z **median 0.596m** (目标 0.55), 腿角严重偏离自然站姿
   (hip_pitch L=-0.49/R=+0.68 vs 名义 ±0.15; knee ±0.5 vs ±0.15)。
2. **一直后退**: 恢复后水平漂移 **2.48-4.62m** (8s episode 全程漂移)。

## 根因分析

**Bug 1 — 升直腿/过高**: `_reward_base_height` 是线性进度
`clip((h-_IDLE_Z)/(h_cmd-_IDLE_Z), 0, 1)`, **h ≥ h_cmd (0.55) 后饱和为 1.0, 无超高惩罚**。
直腿站 0.60-0.67m 与 0.55m 得分相同 → 策略学会"升直腿蹭高度"局部最优
(0.596 median 就是证据)。FK 验证: 直腿自然站 0.515m, 弯膝 0.474m — 0.596 明显过伸。

**Bug 2 — 一直后退**: `_reward_settle` 的静止项 `still = 1-clip(|avz|/1.0)` **只罚垂直速度**,
水平速度 (linvel) 完全没进奖励 → 向后漂移无梯度, 策略可一边站一边退。

**旁证 (用户疑问 "为什么只有前倒能恢复")**: 按姿态评估 4 种倒地 (论文 FTSR 多姿态),
恢复率 **仰卧83% / 俯卧100% / 左躺100% / 右躺100%** — 4 种姿态都能恢复,
之前视频只挑了"第一个恢复的姿态" (俯卧), 让用户误以为只有前倒能恢复。

## 解决方案

1. **`_reward_base_height` 加超高惩罚**: 线性升 (保留起身梯度) × `(1 - clip((h-h_cmd)/0.08))`
   — h_cmd 以上 0.08m 内归零 (0.63m 以上 = 0)。直腿 0.60/0.67 从 1.0 → 0.375/0.0。
2. **`_reward_settle` 加水平静止项**: `still_h = 1-clip(|linvel_xy|/0.5)` —
   恢复后水平速度 <0.5 m/s 才给满分, 直接罚"一直后退"。
3. **rs 阶段 dof_pos 加强**: -0.1 → **-0.5** — 恢复后拉回自然弯膝站姿, 防过伸。

## 修改文件

| 文件 | 修改内容 |
|------|---------|
| `src/unilab/envs/locomotion/xqrobotwl/fall_recovery.py` | `_reward_base_height` 超高惩罚; `_reward_settle` 水平静止 (linvel[:2]) |
| `conf/cpo/task/xqrobotwl_fall_recovery_flat/mujoco.yaml` | scales_rs `dof_pos: -0.1 → -0.5` |

## 验证方法

- 数值单测: base_height @ 0.55/0.60/0.67 = 1.0/0.375/0.0 ✓; settle @ v=0/后退0.3/前0.6 = 1.0/0.4/0.0 ✓
- 训练 (从零, 1024 envs, 8000 iter) 已启动 run `fix_pose`
- 无辅助确定性评估: 恢复率 + 站立 base_z median ≈ 0.55 + 漂移 < 0.5m + 4 姿态分别测

## 后续计划

- [ ] 训练完成后按姿态评估 (4 种倒地恢复率 + 高度 + 漂移)
- [ ] 渲染多姿态恢复视频 (每姿态一个) 交付 — 解答用户"为什么只有前倒"疑问
- [ ] 交付 checkpoint (甜点位)

## 关联日志

- [13_delivery_100pct_recovery_stable](13_delivery_100pct_recovery_stable.md) — 上版交付 (能恢复但姿态差)
- [12_settle_balance_1024envs_fromscratch](12_settle_balance_1024envs_fromscratch.md) — settle 引入 (当时只罚垂直速度)
