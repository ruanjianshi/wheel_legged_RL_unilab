# 08 — 确定性翻转: 翻转期策略屏蔽, 纯 ff 驱动 (用户决定)

**日期**: 2026-08-05
**来源**: 200Hz 下 4 次奖励配置 (ff_tracking 1/10/3 × flip_progress 80/30/10) 全部失败 → 用户选择确定性翻转
**关联**: [07_control_rate_200hz_fix](2026-08-05/07_control_rate_200hz_fix.md), [06_overfit_action_explosion_v2](2026-08-05/06_overfit_action_explosion_v2.md)

---

## 问题描述

200Hz (ctrl_dt=0.005) + P1 复现 ff 下, **纯 ff 可靠翻转** (16/16 env 翻转+落地直立), 初始模型 (近零策略) 也 8/8。但所有奖励配置训练都**漂移**:

| 配置 | 结果 |
|------|------|
| flip_progress=80, ff_tracking=1 | iter~1400 发散 (ff_tracking -1.6, ep_len 120) |
| flip_progress=30, ff_tracking=10 | 探索被罚死 (mean_reward~0, ff_tracking -2.3) |
| flip_progress=10, ff_tracking=3 | iter 800 仍 flip_complete=0, ff_tracking -1.8 |
| **初始模型 (近零策略, 未训练)** | **8/8 翻转+直立** ✅ |

## 根因分析

**奖励结构把策略从"好的初始解"（近零动作=ff 翻转）拉走**:
- `flip_progress` (稠密) 诱导策略刷旋转 → 飞行期偏离 ff → 翻转/落地搞坏
- `flip_complete` (稀疏 100) 在翻转被搞坏后永远不触发 → 无法纠正
- ff_tracking 太弱 (1) 挡不住刷分, 太强 (10) 罚死探索, 无中间甜点

## 解决方案

**确定性翻转 (用户决定)**: env `step` 加 `stand_mask` — 翻转期 (FSM 0-5) 策略动作强制为 0, 纯 ff 驱动; RL 只在站立态 (-1) 生效, 专注两轮足平衡。

```python
stand_mask = (self._fsm_state == -1).astype(np.float64)[:, None]
actions_masked = actions * stand_mask
fused = ff * self._ff_gain + actions_masked
```

**验证**: 随机策略 (std 0.05) + 掩码 → **8/8 翻转完成 + 8/8 最终直立** (策略无法搞坏翻转, 保证成功)。

## 修改文件

| 文件 | 内容 |
|------|------|
| `src/unilab/envs/locomotion/xqrobotwl/backflip.py` | step 加 stand_mask: 翻转期屏蔽策略动作 |
| `conf/.../xqrobotwl_backflip_flat/mujoco.yaml` | 注释更新为确定性翻转设计 |

## 评估结果

> 2026-08-05 补充 (2000 iter 重训完成):

- 训练日志: flip_complete 在 iter 400+ 稳定非 0 (0.05-0.34), 首次在 200Hz 下成功
- **交付模型 = model_1000 (iter 1000)**:
  - 翻过去 (up<-0.5): **32/32**
  - 翻转完成+站起 (flip_complete): **32/32**
  - 最终直立: **32/32** (up_z 全 1.0)
  - max|a| = 0.55 (动作干净)
  - 视频: `video/backflip/2026-08-05_确定性翻转成功_model1000.mp4`
- **过训练教训再次验证**: model_1999 发散 (max|a|=2.29, 0/32 翻转) — 训练中途 checkpoint 才是甜点位, 用 model_1000

## 后续计划

- [x] 重训验证: flip_complete > 0, 落地站稳 — **✅ model_1000 32/32**
- [ ] 渲染 + 键盘实机验证 (后空翻后不倒)
- [ ] 若需策略参与翻转 (抗扰动), 后续可尝试 ff_gain 调度 (翻转期 ff_gain 0.5, 策略轻微修正)

---

*记录人: AI | 审核: xiaoq*
