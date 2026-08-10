# [08] Plan B — 里程碑阶梯 + 慢衰减重启 (桥接半撑→站立梯度死区)

**日期**: 2026-08-07
**来源**: 力衰减修复后重训 (11-41-29 run) 力归零后策略仍卡半撑, recover_complete 连续 3000+ iter 为 0
**关联**: [07_force_decay_unit_bug](07_force_decay_unit_bug.md) / [06_force_hcmd2_deadlock_fix](06_force_hcmd2_deadlock_fix.md)

---

## 问题描述

力衰减单位 bug 修复后重启, 力在 iter 9000 精确归零, 但:
- 力 50% 以下策略塌缩到 0.28-0.30m 半撑 (轮着地, 躯干 0.7 直立)
- 力归零后的 2000 iter 力独立期 recover_complete 全程 0, 无学习迹象
- 约束值 loss →0.01 证实力衰减机制正确, 但策略是"力依赖型"无法转独立

## 根因分析

**半撑 (0.28-0.30m) → 站立 (0.45m) 之间奖励梯度死区**:
- base_height 奖励 scale 只有 4-6, 0.30→0.45 边际增量 ~1.5-3 (弱)
- recover_complete (100) 是够不到的悬崖 — 需 base_z>0.45, 而机器人卡 0.30
- 策略理性选择留在半撑混 alive/upriight/wheel_force 分, 无动力冒险推腿
- 力衰减 (3000 iter) 过快: 力 100%→0 只给 3000 iter 适应, 策略在力 33% 处就崩

## 解决方案 (Plan B)

1. **中间里程碑 `rise`** (新): base_z>0.35 + 直立>0.80 + 双轮着地, 保持 0.3s 锁存 → 30 分
   — 桥接 0.30(半撑)→0.45(站立) 的梯度死区, 与 recover_complete (100) 形成阶梯
2. **base_height 奖励增强**: ru 4→8, rs 6→12 — 连续高度梯度变强
3. **慢衰减**: force_end_iters 3000→5000, 从 model_6000 (最佳力辅助 checkpoint) 重启
   — 5000 iter 力衰减 + 2000 iter 力独立, 每个力水平适应时间 +67%
4. `recover_height`/`rise_height`/`rise_hold` 改为可配置字段 (原 0.45 硬编码)

## 修改文件

| 文件 | 修改内容 |
|------|---------|
| `src/unilab/envs/locomotion/xqrobotwl/fall_recovery.py` | 新增 `_reward_rise`; 配置字段 `recover_height`/`rise_height`/`rise_hold`; `_rise_hold`/`_rise_completed` 状态 + reset; update_state 计算 rise 锁存, standing 用 `recover_height` |
| `conf/cpo/task/xqrobotwl_fall_recovery_flat/mujoco.yaml` | scales_ru/rs 加 `rise: 30.0`, base_height 4→8 / 6→12; 新增阶梯字段 |

## 验证方法

冒烟测试 (model_6000 策略 + 满力):
- max_z 0.56m, **rise 达成 100%, recover 达成 100%** — 阶梯机制正常触发 ✓

## 后续计划

- [ ] 从 model_6000 重启 (12-47-03 run), 5000 iter 力衰减 + 2000 力独立
- [ ] model_11000 (力归零) / model_12000 / model_12999 无辅助评估
- [ ] 恢复成形 → 渲染视频 → 停止

## 关联日志

- [07_force_decay_unit_bug](07_force_decay_unit_bug.md) — 力衰减单位 bug (前置修复)
- [06_force_hcmd2_deadlock_fix](06_force_hcmd2_deadlock_fix.md) — 力目标解耦
