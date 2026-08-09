# [28] 纯PPO 奖励对抗三连 (v10-v12): 从白嫖到真实跳跃

**日期**: 2026-08-09
**来源**: 用户要求"继续改进跳跃, 尤其是纯PPO"
**关联**: [[27_landing_soft_exploit_fix]]

---

## 问题描述

纯PPO 6 轮 (v5-v9) 从未真正腾空 (air 恒 0%)。修复 landing_soft 白嫖 (#27) 后,
逐轮探测发现策略不断发明新的"偷懒盆地":

| 轮次 | 盆地 | 证据 | 修复 |
|------|------|------|------|
| v10/v5 | **推而不蹲** | crouch_prep/depth≈0, thrust 17/步, jump_height~0.02 | base_height 窗口内归零 + thrust 门控需先蹲 + crouch_depth 8 |
| v11 | **推而不起 (stutter)** | thrust 17/步持续, 反复短推 vz 尖峰但不伸腿离地 | **vz² → launch_rise (实际升高)** |
| v12 | **突破!** | verify: **jump_height 0.203m, air 11%** | — |

## 根因分析

每一层偷懒都源于奖励奖励了"过程的代理量"而非"跳跃本身":
1. landing_soft (phase>=30) 奖励"站着不动" (#27 已修)
2. base_height 锚定惩罚下蹲 + thrust 不要求蹲 → 策略不蹲直接推 (v10)
3. vz² thrust 奖励"推的动作" → 策略短推刷 vz 但不完全伸腿 (v11)

## 解决方案 (v12 核心)

**`vertical_thrust` (vz²) → `launch_rise` (base_z − 窗口最深下蹲)**:

```python
rise = np.clip(base_z - floor, 0.0, 1.0)   # floor = window_min_z
active = (phase >= 1.0) & on_ground & window_crouched
reward = rise * scale(40) * active
```

- **推而不起**: 身体回到下蹲点 → rise=0 → 奖励归零
- **缓慢站起**: rise ~0.1 → 小额
- **真实蹬到离地**: rise 最高 → 重奖; 空中由 jump_height + wheel_air_time 接棒

`window_min_z` 追踪 (仅接地下蹲, 排除 reset 下坠): 不蹲就永远没有升高空间,
把"下蹲"硬编码为一切的前置。

## 修改文件

| 文件 | 内容 |
|------|------|
| `jump.py` | `_reward_vertical_thrust` → `_reward_launch_rise`; 新增 `_window_min_z` 追踪 (grounded-eligible); reward dict 替换; 窗口内 base_height=0; crouch 门控 (window_crouched) |
| `jump_vmc.py` | update_state 传 base_z 给 air progress |
| `xqrobotwl_jump_flat/mujoco.yaml` | vertical_thrust 30 → launch_rise 40; crouch_depth 4→8; landing_soft 40→20; landing_window 10 |
| `xqrobotwl_jump_vmc_flat/mujoco.yaml` | 同上 (crouch_depth 8, launch_rise 40) |

## 验证方法

- **探测 (2500 iter, model_1000)**: verify_jump → `jump_height=0.203m, air_frac=0.11`,
  max_base_z=0.746, standing 0.543。纯PPO **首次真实腾空**!
  - 对比 v5: 0.184m 假高度, air 0.000
- 10 项 pytest 通过, ruff 通过

## 全量训练

| 算法 | run | GPU | ETA |
|------|-----|-----|-----|
| 纯PPO v12 | `2026-08-09_01-21-11_mujoco` | 1 | ~4h |
| PPO+VMC v12 | `2026-08-09_01-21-12_mujoco` | 0 | ~5.8h |

## 论文叙事影响

若纯PPO 全量收敛出稳定跳跃 (即使 modest): 叙事从"纯PPO 不腾空 (air 0%)"
升级为 **"无参考纯学习能学会跳跃但显著弱于 SLIP 参考 (高度/腾空/成功率均低)"** ——
更强的对照, 支撑"参考轨迹至关重要"的核心论点。

## 最终结果 (2026-08-09, 全量 10000 iter 完成)

纯PPO 全量训练发散 (后段滑向 handstand 病理行为, 最优是**早期 model_1000**),
PPO+VMC 收敛稳定。最终四算法对比 (compare_jump, 4 速度 × 5 集):

| 算法 | 跳高(m) | air | 存活率 | 成功率 | checkpoint |
|------|--------|-----|--------|--------|-----------|
| SRL | **0.540** | **0.217** | **1.00** | 1.00 | 08-06_01-16-20/model_9999 |
| PPO+VMC v12 | 0.179 | 0.174 | **1.00** | 1.00 | 08-09_01-21-12/model_9999 |
| VMC+SRL | 0.354 | 0.091 | 0.90 | 1.00 | 08-08_01-05-51/model_9999 |
| 纯PPO v12 | 0.278 | 0.085 | 0.50 | 1.00 | 08-09_01-21-11/**model_1000** |

### 关键结论

1. **纯PPO 从 air 0% → 真实跳跃** (0.278m/air 8.5%/成功 100%): 6 轮奖励工程后
   纯PPO 终于能离开地面并完成跳跃循环。**但存活率仅 50%**, 且训练中后期发散
   (model_2000+ 跳得更高但失控, 最终滑向竖直平衡病理) —— 无参考时跳跃**不可控**。
2. **VMC 控制层提供稳定性**: PPO+VMC v12 存活 100% (旧 v4 仅 80%), air 17.4%
   (旧 4.6%), 高度 0.179m 较低 (无参考高度受限)。
3. **SLIP 参考提供控制 + 高度**: SRL 0.540m + 100% 存活, 全面最优。

论文叙事: **"参考轨迹 (SLIP-FSM) 提供跳跃的控制力与高度; 纯 PPO 无参考只能学到
不可控的小跳 (存活 50%); VMC 控制层可弥补稳定性但无法弥补高度"**。

## 修改文件 (本日志)

| 文件 | 内容 |
|------|------|
| `jump.py` | `_reward_vertical_thrust` → `_reward_launch_rise`; `_window_min_z` 追踪; 窗口内 base_height=0; crouch 门控 |
| `jump_vmc.py` | update_state 传 base_z |
| `xqrobotwl_jump_flat/mujoco.yaml` | launch_rise 40 / crouch_depth 8 / landing_soft 20 / landing_window 10 |
| `xqrobotwl_jump_vmc_flat/mujoco.yaml` | 同上 |
| `make_paper_figures.py` | run 目录 → v12; tag vertical_thrust → jump_height/landing_soft |
| `four_algo_comparison.json` | 最终四算法数据 |

## 后续计划

- [x] 完成两个全量训练 + 四算法对比
- [x] 更新 JSON + 重出论文图
- [ ] 论文图微调 (配色/标注) 按需
- [ ] SRL / VMC+SRL 保持不动 (奖励未改, checkpoint 仍有效)
