# SRL+VMC v8d: 实用版调优 (跳高最高 + 姿态最好)

**日期**: 2026-08-16
**来源**: 2×2 消融证实 VMC 靠髋外展借力起跳 (髋外展 0.627, 锁roll 后仅 0.087m)。负责人选 B: 给 SRL+VMC 加回姿态塑形, 调出最高+最干净的跳跃。

## 根因回顾 (为什么 VMC 会外展)

- VMC 的 L0 力控单独无法干净起跳: 纯 FSM 参考 (零策略) 仅 0.156m; 锁 roll 后 v8b 仅 0.087m。
- 髋外展像"摆臂助跳": 额外贡献 +0.05m (0.156→0.204), 是真实生物力学助跳, 且干净消融里无任何惩罚。
- SRL (关节 PD) 的 FSM 关节参考干净下蹲→蹬伸就 0.328m, 不需要外展。

## 修改了什么 (v8d)

1. **`jump_srl_vmc.py`**: 加回 `_reward_lateral_posture` (跳跃期髋外展惩罚 + 左右对称), 注册进奖励集
2. **配置**:
   - `lateral_posture: 15.0` (加回, 压外展)
   - `thrust_kp_scale: 2.5→3.0`, `thrust_ff_scale: 2.5→3.5`, `thrust_kd_scale: 0.5→0.3` (加强蹬伸, 补回外展失去的起跳力)
   - `crouch_length: 0.28→0.24` (更深下蹲 → 更大蹬伸行程)
   - `fsm_crouch_time: 0.45→0.50` (更长时间压缩)

## 验证方法

- 冒烟: obs 315D, lateral_posture 15, thrust_ff 3.5, crouch 0.24 生效
- 重训 4000 iter (`logs/train/jump_srl_vmc_v8d.log`, GPU0)
- 训练后 diag 复检: 跳高/髋外展/站立/膝, 与 SRL (0.328m) 对比

## 训练后效果

### iter-1000 中间评估 — 外展压住 + 跳高上涨
| 指标 | v8b 消融 | v8d iter-1000 |
|---|---|---|
| 跳高 | 0.197m | **0.273m** ↑ |
| 髋外展 | 0.627 ❌ | **0.296** ✅ (<0.5) |
| 站立 \|gyro\| | 0.70 | **0.61** (end 0.15) ✅ |
| 腾空 | 58 | 69 |

lateral_posture + 强蹬伸 + 深蹲组合有效。待 4000 收敛复测 (防 v7c 式塌缩)。

## 后续计划

- 若跳高仍 < SRL: 继续调 (jump_height/height_progress 权重, crouch 深度, 蹬伸)
- 最终四算法对比 + 论文结论

**关联**: [[NN_jump_srl_vmc_v8_clean_ablation]], [[NN_ppo_vmc_v9_no_reference_ablation]], [[assess: 2026-08-16_four_algo_visual_problems_diag]]
