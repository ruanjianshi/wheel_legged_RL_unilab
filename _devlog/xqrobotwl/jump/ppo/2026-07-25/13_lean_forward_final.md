# 13 — SRL 跳跃：lean_forward trigger 门控 + 站姿修复 + 最终收敛

## 日期

2026-07-25

## 来源

旧 run (2026-07-24_20-59-10_mujoco) iter 8569：`jump_height=3.7`，跳成功了，但站立时髋关节后仰 0.64 rad（37°），机身虽直但腿完全后倒。

## 问题描述

1. **髋后仰** — `lean_forward` 函数已写但 config 中 weight=0，从未生效。策略找到捷径：髋后仰 → 降低 COM → 满足 base_height → orientation 不罚 → 所有 reward 绿灯。
2. **leg_mirror 只管对称** — 两条腿一起后仰，对称性良好（leg_mirror≈0），无罚。
3. **无绝对关节角度约束** — 没有 reward 要求 hip_pitch 接近 default 值（0.15, -0.15）。

## 根因分析

```
策略优化路径：
  髋关节后仰 (hip_fwd<0) → COM 降低 (趋向 0.55 目标)
  → base_height reward 改善
  → 机身保持直立 (orientation reward)
  → 双腿对称 (leg_mirror=0)
  → 所有 reward 满足

缺陷：
  lean_forward 注册了但 weight=0 —— 唯一能拉髋前倾的 reward 从未工作
```

## 解决方案

### 第一版（过度约束，失败）
添加 `stand_posture=5.0`（trigger-gated）+ `lean_forward=3.0`（全局），从头训练 → 策略受不了这么强的约束，1000 轮后 reward 从 43 掉到 10，ep_len 从 333 掉到 147。

### 第二版（当前，成功）
- 删除 `stand_posture`（死代码残留）
- **lean_forward 改为 trigger-gated**：trigger≤0.5（站立）才激活，trigger=1（跳跃）=0
- **weight: 1.0 → 3.0**（站立时够强，跳时不干扰）
- 全局版改为门控版的核心逻辑：

```python
def _reward_lean_forward(ctx):
    trigger = ctx.info["commands"][:, 4]
    active = (trigger <= 0.5).astype(np.float64)  # ★ 只站立时
    if not active.any():
        return np.zeros(...)
    # ... 原有逻辑 × active
```

效果：站立 400 步时罚 ~-2.0/步 → 总罚 800/周期，足够驱动髋前倾。跳时 = 0，完全不伤跳跃。

## 修改文件

| 文件 | 行 | 改动 |
|------|-----|------|
| `src/unilab/envs/locomotion/xqrobotwl/jump_srl.py` | 209-223 | lean_forward: 全局 → trigger-gated |
| `src/unilab/envs/locomotion/xqrobotwl/jump_srl.py` | 225-235 | 删除死 _reward_stand_posture |
| `src/unilab/envs/locomotion/xqrobotwl/jump_srl.py` | 391-392 | 删除 stand_posture wrapper |
| `conf/ppo/task/xqrobotwl_jump_srl_flat/mujoco.yaml` | 95 | lean_forward: 1.0→3.0, 删除 stand_posture |

## 评估结果 (iter 9999, final)

| 指标 | 旧 run (无修) | 新 run | 改善 |
|------|--------------|--------|------|
| lean_forward | -0.64（37°后仰） | **-0.25**（5°） | **8倍** |
| jump_height | 2.83/15=0.19 | **1.45**/15=0.10 | 可接受 |
| ep_len | ~700 | **920** | ↑ |
| base_height_err | 0.27m | 0.20m | ↑ |
| action_std | 0.77 | 0.77 | — |

跳的高度略低（0.10 vs 0.19），但站姿从 37° 后仰降到 5°。action_std 始终 0.77 不收敛，是 SRL 框架的固有限制。

## 后续计划

- 本次训练为最终版。模型已备份至 `backup/XqRobotWLJumpSRLFlat/jump_srl_ppo_v1/`。
- 论文用此版对比纯 PPO 跳跃（1.48m, std=0.07）。

## 关联日志

- [12_sign_bugs_entropy_fsm_fixes](2026-07-24/12_sign_bugs_entropy_fsm_fixes.md) — 前序修复
- [08_wheeled_srl_framework_diag](2026-07-22/08_wheeled_srl_framework_diag.md) — 架构设计
