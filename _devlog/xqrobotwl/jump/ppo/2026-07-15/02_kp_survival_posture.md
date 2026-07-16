# 02 跳跃训练迭代：Kp 适配、存活率、后仰修正

**日期**: 2026-07-15
**来源**: 多轮训练 episode_length 低 + 蹲姿错误
**关联**: [01_phase_gated_rewards](2026-07-14/01_phase_gated_rewards.md)

---

## 问题描述

### 问题 1: episode_length=47（Kp=60 运行）

warm-start 从 flat(Kp=30) 到 jump(Kp=60) → 力矩 2× → 每步剧烈过度运动 → 0.5 秒内坍塌。

### 问题 2: episode_length=243（Kp=30, action_scale=1.2）

存活有所改善（2.4 秒），但终止门槛太窄，跳越大动作触发 `thigh_collapsed` 和 `calf_extreme`。

### 问题 3: 后仰冒充蹲

```
正确蹲姿: hip 前倾 (+0.5+), knee 深弯 (+0.5+)
当前蹲姿: hip=-0.88 后仰, knee=-0.10 伸直 → 靠后仰+伸腿降 base
```

蹲姿奖励只看高度不看姿态，策略发现后仰也能降 base → 拿到 crouch 奖励 → 不用学真正蓄力。

## 根因分析

| 问题 | 根因 | 影响 |
|------|------|------|
| 存活低 | Kp=60 × action_scale 过大 | warm-start 不兼容 |
| 存活低 | 终止门槛与 action_scale 不匹配 | 正常动作触发终止 |
| 后仰 | crouch_prep 只检查 base_z | 后仰也能降高度 |

## 解决方案

### 1. Kp 回退 30 + action_scale=0.8

```
flat:  Kp=30 × action_scale=0.5  = 有效 15
jump:  Kp=30 × action_scale=0.8  = 有效 24  (1.6× flat, warm-start 兼容)
```

### 2. 放宽终止门槛

| 检查 | 旧 | 新 | 理由 |
|------|------|------|------|
| `thigh_collapsed` L_hip | < 0.02 | **< -1.0** | 允许后向延伸 |
| `thigh_collapsed` R_hip | > -0.02 | **> 1.0** | 允许前向延伸 |
| `calf_extreme` | abs > 1.2 | **abs > 1.8** | 允许深蹲 |

### 3. 蹲姿增加姿态检查

```python
# crouch_prep 新增 4 个约束:
hip_fwd_L > 0.1  &  hip_fwd_R > 0.1    ← 必须前倾
knee_bend_L > 0.1 &  knee_bend_R > 0.1  ← 必须弯膝
# 后仰+伸腿 → 拿到 0 分 → 必须学真正的蹲
```

### 4. 反蹲罚 (anti_loiter)

```python
# phase > 30 后仍蹲着 (base_z < 0.55) → 扣分
penalty = clip((0.55 - base_z) / 0.3, 0, 2.0) × 5.0
# 蹲到 0.40 → 罚 -2.5/步 → 逼站起或跳起
```

## 修改文件

| 文件 | 改动 |
|------|------|
| `src/unilab/assets/robots/xqrobotwl/xqrobotwl.xml` | 腿关节 Kp 60→30 (回退) |
| `src/unilab/envs/locomotion/xqrobotwl/jump.py` | crouch_prep 姿态检查, anti_loiter, 终止放宽 |
| `conf/ppo/task/xqrobotwl_jump_flat/mujoco.yaml` | action_scale 0.5→0.8, anti_loiter=5.0 |

## 训练迭代历史

| Run | iter | 主要改动 | ep_len | 问题 |
|------|------|------|------|------|
| 2026-07-15_16-30-26 | 9999 | Kp=30, scale=0.8, 终止放宽, anti_loiter | **943** ✅ | 蹲→向后仰 |
| 2026-07-14_23-52-47 | 9999 | Kp=60, scale=0.5 | 47 ❌ | 坍塌 |
| 2026-07-14_17-31-01 | 9999 | phase-gated, crouch↓, thrust↑ | 50 ❌ | phase 窗口锁死 |
| 2026-07-14_10-53-03 | 9999 | 奖励×3, 噪声↓ | 989 | 蹲住不跳 |

## 后续计划

- [ ] 重训 (当前改动已就绪)
- [ ] 验证蹲姿正确 (前倾+弯膝)
- [ ] 验证 anti_loiter 生效 (phase>30 后 base 回升)
- [ ] 如需: 增加 crouch_posture 显式奖励

---

*记录人: AI (opencode)*
