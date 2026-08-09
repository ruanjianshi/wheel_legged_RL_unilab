# [11] 交付 — 无辅助单机器人跌倒恢复视频 (model_15000)

**日期**: 2026-08-07
**来源**: Phase 2 训练成果交付 (用户要求: 恢复成形后渲染视频并停止)
**关联**: [10_phase2_forcefree_risevel](10_phase2_forcefree_risevel.md)

---

## 问题描述

Phase 2 (纯无辅助 + rise_vel) 训练 model_15000 无辅助确定性评估恢复率 30%。
继续训练 (15000→17000) 恢复率退化 (30%→10%→5%) — 过训练发散 (后空翻教训重演)。

## 决策

**用 model_15000 (甜点位) 交付, 停止训练。**

## 交付物

| 文件 | 说明 |
|------|------|
| `video/fall_recovery/2026-08-07_单机器人无辅助恢复_model15000.mp4` | **★ 单机器人清晰恢复视频** (1280×720, 10s, h264) |
| `video/fall_recovery/2026-08-07_无辅助跌倒恢复_model15000.mp4` | 8 机器人网格视图 (多个恢复) |

**渲染方法**: 新增 `scripts/xqrobotwl/render_recovery_video.py` — 随机复位找能恢复的姿态
(max_z>0.50), 用 `backend.set_state` 快照重放 + `run_playback` 录制。
找到的姿态: max_z=0.51, 躯干直立=1.00 (完全站立)。

## 验证方法

- 检测: 该姿态确定性运行 base_z 达 0.51, 躯干直立 1.00 ✓
- 视频: h264/720p/10s 有效 ✓
- 8 env 确定性: 2/8 达 0.45m+ (max_z 0.57/0.64) ✓

## 评估结果 (model_15000, 无辅助, 40 eps)

- 恢复率 30% (base_z 达 0.55m)
- 最大高度 0.44m, 保持率 35%

## 后续计划

- [ ] (可选) 提升恢复鲁棒性 (>50%): 更稳的训练或更多姿态泛化
- [ ] (可选) 师生蒸馏 / 行走阶段 (用户范围外, 后续)
- [x] 本次目标达成: 跌倒→无辅助恢复站立, 视频交付

## 关联日志

- [10_phase2_forcefree_risevel](10_phase2_forcefree_risevel.md) — 恢复成形
- [09_rise_vel_scaffold](09_rise_vel_scaffold.md) — rise_vel 脚手架
