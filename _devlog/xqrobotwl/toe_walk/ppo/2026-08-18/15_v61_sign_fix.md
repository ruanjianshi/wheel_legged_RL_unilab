# 15 — v6.1: 修复 window_penalty 符号 bug (惩罚被实现成正奖励, 导致策略被奖励"不抬腿")

## 日期
2026-08-18

## 来源
v6 训练 (2026-08-18_13-56-30) 奖励分解审查 (iter 4000):
- **reward/window_penalty: +12.7360 (正号!)** — 惩罚项应为负
- reward 192~205 / ep_len 满 → 策略被"每窗未离地 +500"奖励, 选择站姿白拿 (swing_lift 0, knee_lift 0)

## 根因
`_reward_mode_window_penalty` 返回 `info["lift_win_penalty"]` (值为 1.0 正), dispatch 乘 scale 500 → **+500 奖励未离地**。
修补只需一个负号: 函数返回 `-lift_win_penalty` → -1.0 × 500 = -500 ✓

## 修改了什么
| 文件 | 改动 |
|------|------|
| `toe_walk_mode.py` | `_reward_mode_window_penalty` 返回 `-lift_win_penalty` (负值) |

验证: 编译通过; 语义 — 不抬腿每窗 -500 (每周期 -1000), 单侧抬 -250~+250 之间, 双侧完美 +1500(逐时 swing_lift 30)。

## 预期效果
1. "不抬腿"从 +12.7/步 变为 -14/步 → 策略必须学离地
2. 窗级罚重新成为真正的惩罚 (window_penalty 训练曲线应为负且随学习减小)

## 验证方法
- 训练 300 iter 内: reward/window_penalty <0 且 phase_swing_lift 开始上升 (对比之前版本同阶段为 0)
- Stage2 末: window_penalty 绝对值下行 + phase_swing_lift >0.2
- 训练后 verify 全流程

## 教训
连续多轮的"策略不抬腿"根因链条: v2 终止误杀 → v3 单侧经济性 → v4 弯膝刷分 → v5 稀疏反馈 → v6 **符号 bug 把罚变奖**。每一轮都通过训练中奖励分解日志定位 (reward log 是每个诊断的决定性工具) — 应更早全量打印 reward 分解而非只看均值。

## 后续计划
- v6.1 收尾 (~1.2h) → 全量评估 → 渲染视频 → 备份

## 关联日志
- [14_v6_dual_layer](2026-08-18/14_v6_dual_layer.md) — v6 双层设计 (含本 bug 的源头)