# 04 — flip_complete 判据改物理事件锁存 (根因: 旋转积分阈值不可达)

**日期**: 2026-08-05
**来源**: 用户反馈"RL 后空翻效果不好, 想继续开发" → 深度排查 17 篇 devlog + 训练曲线 + 端到端验证
**关联**: [02_flip_progress_measurement_fix](2026-08-05/02_flip_progress_measurement_fix.md) — 测量修复; [03_ff_tracking_reference](2026-08-05/03_ff_tracking_reference.md) — 参考跟踪

---

## 问题描述

- 所有训练 run 的 `flip_complete` 奖励**恒为 0** (仅 08-05 00:51 run 靠旧版坏掉的机身系测量虚增到 2707 rad 才"骗"出过, 对应乱转视频)。
- 机器人**物理上确实完成后空翻**: 实测纯 ff(策略=0) 下绕世界 Y 累计旋转 ~386° (up-vector +1→负→+1, 确凿翻过一次)。
- 但 env 的 `flip_progress` 峰值只有 **~5.2 rad (299°)**, 永远到不了锁存阈值 `FLIP_TARGET-0.3 = 5.98 rad`。

## 根因分析

1. **旋转积分测量低估**: `flip_progress = -∫ω_y_world dt` 只累计绕世界 Y 轴分量, 机器人翻转带横滚/偏航时低估实际旋转 (物理 360° 只测出 ~299°)。devlog 02 声称"世界系 122%"是单一干净模型, 本 env 的翻转含 roll 污染, 普遍低估。
2. **站立态重置截断**: `flip_progress` 在 FSM 回到站立态(-1)时重置为 0, 而翻转收尾 (~300°→360°) 恰好发生在落地阶段, 被重置截断 → 锁存阈值永远不可达。
3. **双重前馈污染**: backflip env 继承 `jump_srl`, 其 `step()` 会在 backflip 前馈之上再叠 `jump_ff*0.15` (crouch 相位膝目标被改 ~20%), 使翻转偏离 P1 干净轨迹。
4. **缺轮子急加速**: P1 开环用 45 rad/s 轮速"翘头"启动翻转, env 的 `_FLIP_FF[0]/[1]` 轮子分量=0, 少初始后仰力矩 (补上后 290°→299°)。

## 解决方案

1. **flip_complete 锁存改物理事件判据** (核心修复): 飞行态(1/2/3)中 up-vector 翻过身(`up_z < -0.3`, >107°) 且后翻方向有实际进度(`flip_progress > 1.5`, 排除前翻/侧滚) → 锁存 `flip_completed`。之后站起来(`up_z > 0.6, z > 0.25`) 即触发 `flip_complete` 奖励。**不再依赖旋转积分阈值**。
2. **补轮子急加速**: `_FLIP_FF[0]/[1]` 轮子 ff 0→4.5 (复现 P1 的 45 rad/s)。
3. **去掉 jump_srl 双重前馈**: `step()` 改直接调 `XqRobotWLWalkFlatEnv.step(self, fused)`。
4. **站立态 fp 重置保留**: 经分析, 若改成 ep 级累计, 站立期(命令重采样 4s)后向晃动会把 fp 抬近 FLIP_TARGET, 使飞行期 `flip_progress` 奖励被 `not_over` 提前封顶。新锁存用物理事件, 不依赖 fp 绝对值, 故保留重置最安全。

## 修改文件

| 文件 | 修改内容 |
|------|---------|
| `src/unilab/envs/locomotion/xqrobotwl/backflip.py:412-415` | 锁存 `flip_progress>=5.98` → 飞行态翻过身物理事件 (`airborne & up_z<-0.3 & fp>1.5`) |
| `src/unilab/envs/locomotion/xqrobotwl/backflip.py:53-64` | `_FLIP_FF[0]/[1]` 轮子 ff 0→4.5 (P1 轮急加速) |
| `src/unilab/envs/locomotion/xqrobotwl/backflip.py:377-379` | `step()` 直接调 `XqRobotWLWalkFlatEnv.step` 绕过 jump_srl 双重前馈 |
| `src/unilab/envs/locomotion/xqrobotwl/backflip.py:426-432` | 保留站立态 fp 重置, 补充注释说明为什么不能去掉 |

## 验证方法

smoke test (16 env, 纯 ff, flip_trigger=1):
- `flip_completed` 触发: 修复前 0, 修复后翻转期频繁触发, 16/16 env 至少完成一次"翻转+站立"
- `flip_complete` 奖励手动计算: 触发 60 次 (修复前恒 0)
- `flip_progress` 稠密奖励: 累计 488 (正常)
- 无 NaN, obs 维度不变 (324/351)
- `ruff check` 通过 (backflip.py 及相关 env 文件)

## 评估结果

> 2026-08-05 补充 (1000 iter 快速训练验证, 修复后代码+旧权重):

| 指标 | iter 0-500 | iter ~650 | iter 999 |
|------|-----------|-----------|----------|
| `flip_progress` | 0→非0 | 非0 | 非0 |
| `flip_complete` | 0 | **0.45-0.73** | 0.49-0.75 |
| `mean_reward` | 2.7→ | → | 10-12.7 |

- 锁存修复在**真实训练中有效**: 策略训练到 ~650 iter 学会完成后空翻+落地直立, flip_complete 从恒 0 变稳定非 0
- 早期 (0-500 iter) 的 spin-hacking (飞行期偏离 ff 刷 flip_progress) 是**暂态**, 非收敛终点
- 基线 model_999 数值评估: up_z 最低 -1.0 (完全翻转), 最终存活 env 100% 直立, flip_complete 触发 128 次/19200 env-step

## 后续计划

- [ ] 500-1000 iter 快速训练: 确认 `flip_complete` > 0, 存活率回升 — **✅ 已验证**
- [ ] 渲染验证: 翻转干净 + 落地直立 (不甩腿/不麻花) — 数值评估通过 (up_z=-1.0, 100% 直立)
- [ ] 训练稳定后收紧终止 (分阶段方案C 第二阶段)
- [ ] `make test-all` 门禁: 预存在 `tools/pinocchio_traj/visualize_toe_walk.py` 16 个 F841 错误 (与本次无关, 未修复)

---

*记录人: AI | 审核: xiaoq*

## 后续计划

- [ ] 500-1000 iter 快速训练: 确认 `flip_complete` > 0, 存活率回升
- [ ] 渲染验证: 翻转干净 + 落地直立 (不甩腿/不麻花)
- [ ] 训练稳定后收紧终止 (分阶段方案C 第二阶段)
- [ ] `make test-all` 门禁: 预存在 `tools/pinocchio_traj/visualize_toe_walk.py` 16 个 F841 错误 (与本次无关, 未修复)

---

*记录人: AI | 审核: xiaoq*
