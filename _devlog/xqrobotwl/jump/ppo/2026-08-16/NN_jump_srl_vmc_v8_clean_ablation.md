# SRL+VMC v8: 干净消融 — 只输出层与 SRL 不同

**日期**: 2026-08-16
**来源**: 负责人提出思路 — "SRL+VMC 应该只在输出与 SRL 有区别,用于做对比", 并请求评估。

## 思路评估 (负责人思路正确)

干净消融实验应隔离单一变量。当前 SRL vs SRL+VMC 在**奖励、观测、参考融合、控制层**四处都不同 → 无法归因"VMC 控制层"的效果。v8 统一为:
- **奖励**: 与 SRL 完全相同 (去掉 v7 的 VMC 专属塑形 jump_upright/lateral_posture/standing_still/下蹲门控, 补 anti_drift/action_magnitude)
- **观测**: 与 SRL 相同 (315D 关节空间 + FSM, 去掉虚拟腿运动学)
- **控制层 (唯一不同)**: 关节 PD (SRL) vs 虚拟腿 VMC (SRL+VMC)

预期: 若 SRL+VMC 在相同奖励下重现外展/站立振荡 → **归因于 VMC 控制层本身**, 是合法且重要的发现。

## 修改了什么

1. **`src/unilab/envs/locomotion/xqrobotwl/jump_srl_vmc.py`**
   - `_init_reward_functions`: 奖励集换成 SRL 的 (去 jump_upright/lateral_posture/standing_still, 补 anti_drift/action_magnitude)
   - 所有奖励 wrapper 指向 `_srl._reward_*` (含 landing_soft/landing_recovery/wheel_ground_matching 用 SRL 版, 去掉下蹲门控)
   - `__init__`: 观测帧 41D→33D, 重分配历史缓冲; `obs_groups_spec` → 315/342
   - 新增 `_compute_obs`: 跳过 jump_vmc 虚拟腿观测, 用 walk 基类关节观测 + FSM features (315D)
2. **`conf/ppo/task/xqrobotwl_jump_srl_vmc_flat/mujoco.yaml`**
   - scales 与 SRL 完全一致: ang_vel_xy -0.5, jump_height 25, landing_soft 8, landing_recovery 8, wheel_ground_matching 8, height_progress 20, anti_drift -3 (新增); 去掉 jump_upright/lateral_posture/standing_still
3. **`tests/.../test_xqrobotwl_jump_vmc.py`**: `test_srl_vmc_obs_dims_match_spec` 387→315

## 验证方法

- 冒烟: SRL+VMC env 构建, obs=315D, scales 与 SRL 一致
- VMC pytest 10 passed
- 重训 4000 iter (`logs/train/jump_srl_vmc_v8.log`, GPU0)
- 训练后 diag 复检: 跳高/髋外展/站立振荡, 与 SRL 直接对比

## 训练后效果 (待回填)

## 后续计划

- 若 v8 重现外展/站立振荡 → 如实记录为 "VMC 控制层需额外塑形" 的发现 (论文对比结论)
- 四算法最终对比报告

**关联**: [[NN_jump_srl_vmc_v7_abduction_fix]], [[assess: 2026-08-16_four_algo_visual_problems_diag]]
