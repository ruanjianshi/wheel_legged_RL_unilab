# 07 — "一翻就倒"根因: 控制率 100Hz 太粗, 改 200Hz (ctrl_dt=0.005)

**日期**: 2026-08-05
**来源**: 用户实测 model_999 "一翻就倒" → 深度排查 env 物理 vs P1 开环
**关联**: [04_flip_complete_event_latch](2026-08-05/04_flip_complete_event_latch.md), [06_overfit_action_explosion_v2](2026-08-05/06_overfit_action_explosion_v2.md)

---

## 问题描述

model_999 (1000 iter, 评估 flip_complete 224 次, up_z=-1.0) 被用户实测发现"一翻就倒"——翻转后站不住。

## 根因分析

**env 的翻转在物理上就是地面打滚, 不是 P1 的空中后空翻**:

| 测量 | 结果 |
|------|------|
| env 纯 ff 飞行态轮子离地 | 0-2/60 步 (贴地打滚) |
| env 翻转 maxfp | ~3.5 rad (只转 ~180° 就落地倒挂) |
| P1 精确 ctrl 重放 env backend (200Hz) | **最终 up=0.99 (干净翻转!)** |
| 同一 ctrl 但 100Hz | 最终 up=0.17 (打滚!) |

**结论**: env 默认 `ctrl_dt=0.01` (100Hz) 对爆发蹬地(膝 0.07s 猛伸)太粗糙 → 翻转不足 → 地面打滚 → 落地倒挂 → "一翻就倒"。**200Hz (ctrl_dt=0.005) 下 env 物理完全能做出 P1 的翻转。**

## 解决方案

1. **`conf/.../xqrobotwl_backflip_flat/mujoco.yaml` 加 `env.ctrl_dt: 0.005`** (200Hz, 对齐 P1)
2. **`backflip.py` 的 `_compute_feedforward` 重写为 P1 开环序列直接复现** (devlog 04 的瞬时跳变改 P1 式目标渐变 blend):
   - 各相位内 `r = timer/ramp_time` 渐变 (crouch 0.08s, launch 0.07s, flight 0.15s, deploy 0.12s)
   - launch 髋先探 +0.10 再后仰 -0.60 (P1 的后旋角动量来源)
   - deploy 轮子 [5,5] rad/s (落地滚到轮上)
   - 轮子 W=0 (devlog 04 加的轮子急加速是**错的**, P1 获胜参数 W=0, 轮子加速=前翻)

## 修改文件

| 文件 | 内容 |
|------|------|
| `conf/ppo/task/xqrobotwl_backflip_flat/mujoco.yaml` | `env.ctrl_dt: 0.005` (+注释) |
| `src/unilab/envs/locomotion/xqrobotwl/backflip.py` | `_compute_feedforward` 重写: P1 渐变 blend + launch 髋先探后仰 + deploy 轮 [5,5] + W=0 |

## 验证方法

200Hz 纯 ff (策略=0), 键盘场景 (站立→按 H→松开), 16 env:
- 翻过去 (up<-0.5): **16/16**
- 翻转完成+站起 (flip_complete): **16/16**
- 最终直立 (up>0.6): **16/16** (up_z 全 0.89)
- 100Hz 对照: 打滚 (up=0.17)

## 评估结果

> env 层修复完成。下一步 200Hz 重训, 确认 RL 模型翻转后站稳。

## 后续计划

- [ ] 200Hz 重训 (fresh run), 验证 flip_complete + 落地站稳
- [ ] 渲染 + 键盘实机验证
- [ ] 收紧终止 (分阶段方案C 第二阶段)

---

*记录人: AI | 审核: xiaoq*
