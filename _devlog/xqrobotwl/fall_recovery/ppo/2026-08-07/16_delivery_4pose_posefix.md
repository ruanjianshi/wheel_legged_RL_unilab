# [16] 交付 — v2 姿态修复 + 4 姿态恢复视频 (model_4000)

**日期**: 2026-08-08
**来源**: v2 训练完成, 确认甜点位 + 渲染 4 姿态视频
**关联**: [15_v2_redo_overshoot_band_standstill](15_v2_redo_overshoot_band_standstill.md)

---

## 交付 checkpoint

**model_4000** (run `2026-08-07_22-17-51_mujoco`, v2 从零训练, 1024 envs)

## 评估结果 (无辅助确定性, 每姿态 20 episodes)

| 姿态 | 恢复率 | 平均最大高度 | 平均最大水平漂移 | 最长连续站立 |
|---|---|---|---|---|
| 仰卧 (supine) | **95%** | 0.59m | 0.70m | 0.38s |
| 俯卧 (前倒) | **90%** | 0.60m | 0.69m | 0.34s |
| 左躺 | **80%** | 0.56m | 0.77m | 0.33s |
| 右躺 | **90%** | 0.56m | 0.55m | 0.40s |

随机 4 姿态混合: 恢复率 95%, 高度 0.57m, 漂移 0.52m

## 对比 v0 (原交付 model_5000)

| 指标 | v0 (model_5000) | v2 (model_4000) |
|---|---|---|
| 站立 median 高度 | 0.596m (过高) | **0.50-0.54m** (对齐 0.52) |
| 水平漂移 | 2.48-4.62m | **0.52-0.77m** (改善 5-8x) |
| 恢复率 (混合) | 100% | 95% |
| 4 姿态都能恢复 | ✓ | ✓ (80-95%) |

## 关键修复 (v1→v2)

1. **h_cmd2 0.55→0.52** (对齐 walk 自然站立 median 0.522) — 高度问题修复
2. **超高惩罚 band 0.08→0.12** — 不杀起身冲高, 仍罚 0.596 过伸
3. **settle 还原仅垂直静止** + **新增独立 stand_still 项** — 防后退但不乘死 settle
4. **dof_pos -0.5→-0.1** — 恢复期腿摆动不被过罚

## 交付物

| 文件 | 说明 |
|------|------|
| `video/fall_recovery/recovery_model_4000.pt_仰卧_supine.mp4` | 仰卧恢复 |
| `video/fall_recovery/recovery_model_4000.pt_俯卧_前倒_prone.mp4` | 俯卧/前倒恢复 |
| `video/fall_recovery/recovery_model_4000.pt_左躺_left.mp4` | 左躺恢复 |
| `video/fall_recovery/recovery_model_4000.pt_右躺_right.mp4` | 右躺恢复 |
| checkpoint `model_4000.pt` | 无辅助确定性 4 姿态 80-95% 恢复率 |

## 剩余问题

- **保持时间 0.33-0.40s < 0.5s 达标线** — 站起来后平衡保持还不够长
- **漂移 0.55-0.77m 仍略超 0.5m** — stand_still 速度阈值 (0.5 m/s) 对缓慢持续后退偏松
- 过训练发散: model_4000 后漂移持续恶化 (5000: 1.03, 6000: 1.78, 7000: 1.11, 7999: 0.86) — 甜点位在 4000

## 后续计划

- [x] 4 姿态恢复验证 + 渲染交付
- [ ] (可选 v3) 位置惩罚项 (罚离开站立点距离, 直击缓慢持续后退) + 提高 settle/保持
- [ ] (可选) 师生蒸馏 / 行走阶段

## 关联日志

- [15_v2_redo_overshoot_band_standstill](15_v2_redo_overshoot_band_standstill.md) — v2 修复
- [14_fix_pose_overshoot_drift](14_fix_pose_overshoot_drift.md) — v1 (过度约束失败)
- [13_delivery_100pct_recovery_stable](13_delivery_100pct_recovery_stable.md) — v0 (姿态差)
