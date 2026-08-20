# 23 — v10.2: 回归 v1 温和配方 + 增强 (窗级重罚对单模式从零学习致命)

## 日期
2026-08-19

## 来源
v10/v10.1 单模式训练崩盘实证。

## 失败链与分析

| 版本 | 2100-2200 iter | 判定 |
|------|----------------|------|
| v10 (window_penalty 500 固定) | swing 0.03, ep_len **14.6** | ❌ 强罚压死探索 |
| v10.1 (window_penalty 60→500 课程) | swing 0.037, ep_len **17.6** | ❌ 课程化也没用 |

**根因**: 窗级重罚 (未离地窗 -500) 对**无平衡预热的从零学习**致命。v6.1 双模式成功靠先 1800 iter 纯站立预热 (平衡先学会); 单模式无预热, 策略早期(500-2200 iter)动作探索即摔倒 (ep_len 崩), 大罚使梯度混乱越学越差。v1 (2026-07-25) 历史能收敛 (ep_len 922/swing 0.48) 靠**温和接触罚 swing_contact 5** 而非窗级大罚。

## 修改了什么 (v10.2)
| 文件 | 改动 |
|------|------|
| `mujoco.yaml` | `window_penalty` 500→**0** (关闭窗级重罚, 机制保留); 恢复 `swing_contact_penalty` **5** (v1 温和); 保留 phase_swing_lift 30 / knee_lift 15 (窗内门控) / lift_symmetry -20 / tracking 10&5 (指令门控) / curriculum_steps 4000 / 终止放宽 (thigh ±0.9 + 5帧延迟) |

## v10.2 配方 (本质 = v1(07-25) + 已验证无害增强)
```
phase_swing_lift 30 + swing_contact_penalty 5 + knee_lift 15(门控) + knee_stance 5
+ stance_penalty 5 + feet_regulation 2
+ lift_symmetry -20 (v3-v6 验证, 温和)
+ tracking_lin/ang 10/5 指令门控 (v7 验证, 修复"零指令白拿")
+ 终止放宽 thigh ±0.9 / 接触 8N / 异常帧延迟 5 (v3 验证, 防单帧误杀)
+ 课程 curriculum_steps 4000: 先抬腿后追踪
```

## 预期效果
- 温和起步 (swing_contact 5) → 策略敢探索抬腿 (v1 同路径)
- 对称EMA + 追踪门控 + 终止放宽 提供 v1 没有的质量增益

## 验证方法
- 2200-3500 iter: swing_lift 应像 v1 一样回升, ep_len 不崩 (<50)
- 训练后: verify 交替/对称 + 追踪实测 + 视频

## 关联日志
- [22_v10_single_mode](2026-08-19/22_v10_single_mode.md) — v10 起始 (窗级重罚引入)