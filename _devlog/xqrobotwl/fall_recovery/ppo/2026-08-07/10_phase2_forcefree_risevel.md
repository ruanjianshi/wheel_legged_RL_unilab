# [10] Phase 2 — 纯无辅助 + rise_vel, 恢复技能首次成形 (30% 无辅助恢复率)

**日期**: 2026-08-07
**来源**: Plan B 力衰减到 0 仍塌缩 (力锁定) → 放弃力, 纯无辅助 + rise_vel 脚手架
**关联**: [09_rise_vel_scaffold](09_rise_vel_scaffold.md) / [08_milestone_ladder_planB](08_milestone_ladder_planB.md)

---

## 问题描述

Plan B (阶梯 + 慢衰减) 力归零后恢复率仍 0% — 策略"力锁定", 缺失推腿自举技能。
放弃力引导, 改纯无辅助训练 + rise_vel 密集奖励。

## 解决方案

**Phase 2 (run `2026-08-07_14-13-41_mujoco`)**:
- `reward.force_assist_enabled=false` — 从头无辅助
- `rise_vel: 15.0` (双轮着地门控的向上速度奖励) — 逐帧奖"推"动作
- 从 model_11000 (半撑 0.30m) 续训 5000 iter
- 奖励阶梯保留: base_height (8/12) + rise (0.35m→30) + recover_complete (0.45m→100)

## 评估结果 (无辅助确定性, 40 episodes)

| checkpoint | 恢复率 | max_z | 保持率 |
|---|---|---|---|
| model_12000 | 0% | 0.33 | - |
| model_14000 | **25%** | 0.43 | 10% |
| model_15000 | **30%** | 0.44 | 35% |
| model_15999 (终) | 0%* | 0.33 | 60% |

*20-episode 采样 (小样本偏差); 14000/15000 用 40 episodes。

**结论**: rise_vel 脚手架打破力锁定 — 机器人学会无辅助推腿站起。但恢复技能**脆弱不稳**
(14000 25% → 15000 30% → 15999 采样 0%), 训练末段 recover_complete 指标回落到 0 —
恢复出现但未收敛。**过训练发散风险** (后空翻教训): 不取最终, 选 15000 甜点位。

## 修改文件

本次无代码修改 — 纯训练配置干预 (force_assist_enabled=false + rise_vel 已在 09 加入)。

## 后续计划

- [ ] 从 model_15000 (30% 恢复率) 续训 +4000 iter (run `2026-08-07_15-37-35_mujoco`)
- [ ] 每 1000 iter 无辅助评估, 选最佳 checkpoint (不只看最终)
- [ ] 恢复率稳定 >50% → 渲染视频 → 停止

## 关联日志

- [09_rise_vel_scaffold](09_rise_vel_scaffold.md) — rise_vel 脚手架 (本阶段的功臣)
