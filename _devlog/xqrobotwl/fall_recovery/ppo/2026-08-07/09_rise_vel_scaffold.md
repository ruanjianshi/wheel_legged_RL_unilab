# [09] rise_vel 奖励 — 直接搭"推腿自举"脚手架 (干预 C 准备)

**日期**: 2026-08-07
**来源**: Plan B run 力 20% 时阶梯也塌缩 — 缺失的技能是"推腿自举", 力辅助时代策略从不主动举升
**关联**: [08_milestone_ladder_planB](08_milestone_ladder_planB.md)

---

## 问题描述

Plan B (里程碑阶梯 + 慢衰减) 中:
- model_8000 (力 60%): rise 10-21%, recover 22% — 阶梯生效
- model_10000 (力 20%): rise 0, recover 0 — 又塌缩
- 与上轮相同模式: 力 <40% 后策略撑不住, 无法无辅助站立

## 根因分析

**策略是"力锁定"的 — 缺失"推腿自举"技能**:
- 力辅助 (Fmax=160≈自重87%) 承担了几乎全部举升, 策略只需在力抬升时平衡
- 力撤走时, 策略从未学过"主动伸腿把躯干推高"的动作
- base_height 奖励是**端点式** (只看当前高度), 从 0.30→0.31 只给 ~0.3 分 —
  逐帧梯度太稀, 策略没动力发现推力

**关键洞察**: 0.30m 半撑 + 躯干 0.81 直立 + 轮着地 — 姿势已接近站立, 只差最后推腿。

## 解决方案

**新增 `rise_vel` 密集奖励 (脚手架)**: 奖向上 base_z 速度
```python
dbz = base_z - prev_base_z
rise_vel = clip(dbz/ctrl_dt, 0, 1.0) * wheel_on_mask  # 双轮着地才生效 (防跳起刷分)
```
- 逐帧即时反馈 — 任何向上的推力立刻有奖, 不需等到高度明显变化
- 双轮着地门控 → 不能靠跳/弹刷分 (配合 no_fly -2.0)
- scale 15.0 (ru/rs 都加), 与 base_height (8/12) + 阶梯 (rise 30 / recover 100) 互补

**计划用途**: 纯无辅助 Phase 2 训练 (force_assist_enabled=false), 从当前 run 终点
(无辅助适应后的半撑策略) 出发, rise_vel 给最小推力一个即时梯度 → 学推 → 达 rise →
达 recover。

## 修改文件

| 文件 | 修改内容 |
|------|---------|
| `src/unilab/envs/locomotion/xqrobotwl/fall_recovery.py` | 新增 `_reward_rise_vel`; `_prev_base_z` 状态 + `_sync_prev_base_z` (reset 同步); update_state 计算 `info["rise_vel"]` (双轮着地门控) |
| `conf/cpo/task/xqrobotwl_fall_recovery_flat/mujoco.yaml` | scales_ru/rs 加 `rise_vel: 15.0` |

## 验证方法

冒烟测试 (力关): base_z 上升时 rise_vel>0 (0.26-1.0), 下降/平速时 =0 ✓
奖励 dispatch 无崩溃 ✓

## 后续计划

- [ ] 当前 Plan B run 跑完 (model_12999) → 评估
- [ ] 若失败 → Phase 2: `force_assist_enabled=false` + rise_vel, 从 model_12999 重启 ~5000 iter
- [ ] 恢复成形 → 渲染视频 → 停止

## 关联日志

- [08_milestone_ladder_planB](08_milestone_ladder_planB.md) — 阶梯 (rise_vel 的补充)
