# SRL+VMC v8e: 根因修复 (3 项)

**日期**: 2026-08-16
**来源**: v8d 收敛后跳高塌缩 (0.12m) + 起跳仍外展, 负责人质疑代码。深度诊断发现 3 个根因。

## 诊断纠正

- **纠正之前错误结论**: "VMC 力控 0.156m 天花板"是测试缺陷 — 纯参考(零动作)机器人站不稳
  (初始 0.65m 坠落, FSM 反复 idle→crouch→终止→重置), 0.156m 是坠落峰值非真跳高。
- **真根因**: ①蹲下相支撑前馈(110N)对抗压缩 ②奖励梯度下小跳比大跳划算
  (0.12m 跳 jump_height~32/步 vs 0.27m 跳 ~40/步, VMC 软力控大跳更费力 → 策略选小跳)
  ③起跳瞬间 roll 脉冲借力, 按步罚压不住瞬时外展。

## 修改了什么 (v8e, 3 项修复)

1. **`jump_srl_vmc.py` `get_l0_control_parameters`**: 蹲下相 (phase 0) 支撑前馈归零
   — 腿自由压缩, 更深下蹲 → 更大蹬伸行程。
2. **`_reward_jump_height`**: 加最低跳高门槛 (base_z > base_height_target + 0.15 才算
   真跳) — 消灭"小跳刷分" (v8d 塌缩根因)。height_progress/vertical_thrust 仍给信号。
3. **`_reward_lateral_posture`**: 增加 roll 速度惩罚 (×0.5) — 压起跳瞬间的瞬时外展脉冲。

## 验证方法

- 冒烟 + VMC pytest 10 passed
- 重训 4000 iter (`logs/train/jump_srl_vmc_v8e.log`, GPU0)

## 训练后效果

### v8e iter-1000 — 门槛太死, 卡在 0.145m
| 指标 | 值 | 判定 |
|---|---|---|
| 跳高 | 0.145m (未过 0.70 门槛) | ❌ jump_height 几乎恒 0, 无梯度 |
| 髋外展 | 0.154 | ✅ 压住了 |
| 站立 \|gyro\| | 0.72 | ✅ |

硬门槛 (0.70) 消灭了刷分但**也消灭了梯度** — 策略卡在 0.145m 学不过去。

### v8e2 (真跳事件激励) 重训中
- 新增 `jump_event` 一次性奖励 (base_z 首次>0.72, 每窗口一次, scale 100) — 明确"做真跳"信号
- jump_height 门槛降到 0.65 (保留梯度)
- 保留 v8e 的蹲下相前馈归零 + roll 速度惩罚

## 后续计划

- 监控 jump_event 是否触发 (>0) — 触发说明策略学会跨过 0.72
- 最终四算法对比

**关联**: [[NN_jump_srl_vmc_v8d_best_tuning]], [[assess: 2026-08-16_four_algo_visual_problems_diag]]
