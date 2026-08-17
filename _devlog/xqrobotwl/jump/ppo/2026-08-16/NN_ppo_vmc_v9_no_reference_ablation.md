# PPO+VMC v9: 无参考消融臂 (只输出层与纯PPO 不同)

**日期**: 2026-08-16
**来源**: 负责人思路 — "PPO+VMC 应该和纯PPO 上也只有输出层面的区别", 四算法构成完整 2×2 消融。

## 2×2 消融设计 (每对各自奖励)

| | 无 SLIP 参考 | 有 SLIP 参考 |
|---|---|---|
| 关节空间 | 纯PPO (297D, 纯PPO奖励) | SRL (315D, SRL奖励) |
| 虚拟腿 VMC | **PPO+VMC v9** (297D, 纯PPO奖励) | SRL+VMC v8b (315D, SRL奖励) |

- 纯PPO vs PPO+VMC: 只差输出层 (关节PD vs 虚拟腿VMC), 同奖励/同观测
- SRL vs SRL+VMC: 只差输出层, 同奖励/同观测

## 修改了什么

1. **`src/unilab/envs/locomotion/xqrobotwl/jump_vmc.py`**
   - `step()`: 去掉 SLIP-FSM 参考混合 → 直接透传 (策略直接输出虚拟腿参考)
   - `get_l0_control_parameters()`: 恒定默认增益 (去分阶段缩放)
   - `_init_reward_functions()`: 只继承 jump.py (去 anti_early_extend)
   - `_compute_obs()`: 用 walk 基类关节观测 (297D, 去虚拟腿/FSM 特征)
   - `obs_groups_spec`: 297/324; `__init__`: 观测帧 41→33
   - 删除 `_reward_launch_rise` (FSM 门控版) / `_reward_anti_early_extend` / `_leg_L0`
2. **`conf/ppo/task/xqrobotwl_jump_vmc_flat/mujoco.yaml`**: scales 与纯PPO 完全一致 (含 anti_lazy 60, 去 anti_early_extend; landing_soft 30/launch_rise 40/crouch_depth 8 等), window_crouch_threshold 0.45
3. **`tests/.../test_xqrobotwl_jump_vmc.py`**: PPO+VMC obs 387→297
4. **附带修复 SRL+VMC 参考双重混合**: 原 SRL+VMC.step 混一次参考后调 jump_vmc.step 又混一次 → 参考 L0 目标 ×2。现在 jump_vmc.step 透传, SRL+VMC 单次混合 (v8b 重启)

## 验证方法

- 冒烟: PPO+VMC obs=297D, scales 含 anti_lazy/无 anti_early_extend; SRL+VMC obs=315D 正常
- VMC pytest 10 passed
- 重训 PPO+VMC v9 (`logs/train/jump_vmc_v9.log`, GPU1) + SRL+VMC v8b (`logs/train/jump_srl_vmc_v8b.log`, GPU0)

## 训练后效果 (待回填)

## 后续计划

- 训练完成后四算法 2×2 消融对比, 检验 SRL+VMC 是否最优

**关联**: [[NN_jump_srl_vmc_v8_clean_ablation]], [[assess: 2026-08-16_four_algo_visual_problems_diag]]
