# SRL v5: height_progress 修复 + 中间评估 (iter-2000)

**日期**: 2026-08-16
**算法**: SRL (`xqrobotwl_jump_srl_flat`)
**配置**: v5 = v4 重平衡 + height_progress 时序 bug 修复
**训练**: GPU1, 1024 env, 4000 iter (进程 PID 2517484)
**模型**: `logs/rsl_rl_ppo/XqRobotWLJumpSRLFlat/2026-08-16_13-04-11_mujoco/model_2000.pt`

---

## 1. 背景

SRL v4 在"落地恢复稳定性"与"跳高"之间做了取舍: landing_recovery 8 + anti_drift -4 + wgm 8
把恢复 |gyro| 从 7.3 压到 0.8, 但跳高从 0.57m 跌到 0.23m。

`_reward_height_progress` 存在时序 bug: `episode_max_height` 在奖励计算**之前**被更新,
导致 `base_z - max_z` 恒为 0, height_progress 奖励从未触发 (训练日志恒 0.0000)。

**v5 修复**: `update_state` 先存 `state.info["episode_prev_max_height"]` 再更新
`_episode_max_height`, `_reward_height_progress` 改用 `episode_prev_max_height`。

## 2. 训练奖励趋势 (iter ~2100)

```
MR=119.77  EL=878
jump_height=10.11   (强, 跳高激励恢复)
height_progress=0.115  ✅ (已触发, bug 修复生效)
orientation=-0.15     (直立惩罚轻微)
ang_vel_xy=-0.52      (晃动惩罚仍在 — 站姿仍有晃动)
landing_recovery=0.38 (已发放但较弱)
base_height=-1.74
```

## 3. 详细三段评估 (model_2000)

工具: `eval_jump_detail.py` (settle=100, pulse=160, tail=200)

### 段1 跳跃前站立姿态
```
base_z 0.527 ✅ (目标 0.52)   up_z 0.942 ✅ 直立
|gyro| 1.316 ❌ 晃动 (阈值 1.0)  对称 ❌ 左右腿一前一后
姿态: 正常站立 + 左右腿一前一后 + 摇摆
```

### 段2 键盘触发跳跃过程
```
跳高 0.347m ✅ (>0.20)   腾空 103 步   空中 up_z 0.936 ✅
下蹲最低点  z=0.297 (过深, <0.25 临界"倒地")  膝屈 +0.89/-0.74  前倾 pitch +0.31
起跳前      z=0.671 伸腿过高 + 前倾 +0.41  左右高低腿
腾空峰值    z=0.867 伸腿/过高 + 左右高低腿 + 左右倾斜
落地        z=0.397 下蹲 + 前倾 +0.56 + 摇摆
```

### 段3 落地后姿态
```
恢复站立 @step 286 (落地后 26 步) ✅
恢复期 |gyro| 均值 1.794 max 5.47 ❌ 晃动明显
恢复期 base_z 均值 0.428 波动 0.107
最终姿态 step459: z=0.330 下蹲 + 左右高低腿 + 摇摆 ❌ 未恢复正常站立
```

## 4. 结论

| 指标 | v4 | v5 (iter-2000) | v5 (final 3999) | 判定 |
|---|---|---|---|---|
| 跳高 | 0.23m | 0.347m | **0.277m** | ✅ 超 0.20 阈值 |
| height_progress | 恒 0 | 0.115 触发 | 0.05~0.15 | ✅ bug 修复 |
| 站姿 |gyro| | 0.8 | 1.316 | 1.595 | ❌ 晃动 (收敛后未改善) |
| 落地恢复 | 稳定 | 26 步恢复 | **58 步恢复, 最终 z=0.295 未站立** | ❌ 变差 |
| 深蹲 | 正常 | 过深 (0.297) | 过深 (0.252, 临界"倒地") | ⚠️ |

**核心矛盾**: height_progress 修复把跳高推到 0.277m, 但策略偏向"高跳+前冲"换取高度,
**收敛后站姿晃动加剧 (1.595), 落地恢复变慢, 最终姿态仍下蹲 (z=0.295) 未回归站立**。

## 5. 最终判定 (model_3999)

**SRL v5 跳高达标 (0.277m), 但姿态不达标**:
- 站姿 |gyro| 1.595 > 1.0 阈值 ❌
- 恢复期 |gyro| 1.74 max 4.79 ❌
- 最终 z=0.295 下蹲 + 左右腿一前一后 + 高低腿 ❌ (恢复后未站立)

**→ 需要 SRL v6**: 保持跳高 margin 前提下加强姿态项:
- `ang_vel_xy` -0.05 → **-0.15** (压站姿/恢复晃动)
- `landing_recovery` 4 → **8** (驱动落地后恢复站立)
- `anti_drift` -2 → **-3** (略收紧防恢复期漂移)
- 保持 jump_height 25 / height_progress 20 (跳高主力不动)
- 注意避免 v4 覆辙 (过强恢复项压垮跳高: 0.57→0.23)
