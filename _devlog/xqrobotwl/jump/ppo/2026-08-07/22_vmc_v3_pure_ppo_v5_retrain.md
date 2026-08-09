# [22] VMC 变体 + 纯PPO v5 重训 (最终对比用)

**日期**: 2026-08-07
**来源**: 用户要求跑满 10000 轮, 确保四个算法都能跳且落地不倒
**关联**: [[21_vmc_kd_l0_fix]], [[23_verify_jump_config_drift]]

---

## 问题描述

之前四算法对比时, PPO+VMC / VMC+SRL / 纯PPO 各有不足:

- PPO+VMC: 起跳弱, 需提升
- VMC+SRL: 训练中 policy 一度抵消参考
- 纯PPO: 落地存活率 0%, 需加 landing_recovery

用户要求: 训练继续跑 10000 轮, 保持对比变量一致, 确保全部可跳跃且落地不倒。

## 解决方案

启动 3 个重训 (从 model_0 开始, 目标 10000):

| 算法 | task_name | 配置要点 | GPU |
|------|-----------|---------|-----|
| PPO+VMC | XqRobotWLJumpVMC | kd_l0=5 (低阻尼爆发) | 0 |
| VMC+SRL | XqRobotWLJumpSRLVMC | kd_l0=5 + thrust_kp_scale=3.0, thrust_ff_scale=3.0, feedback_gain=0.5 | 1 |
| 纯PPO v5 | XqRobotWLJumpFlat | landing_recovery=4.0, landing_soft=40 | 1 |

SRL (XqRobotWLJumpSRLFlat) 使用已有 model_9999, 不重训。

## 修改文件

| 文件 | 修改内容 |
|------|---------|
| `conf/ppo/task/xqrobotwl_jump_flat/mujoco.yaml` | landing_recovery 4.0, landing_soft 40, jump_height 15 |
| `conf/ppo/task/xqrobotwl_jump_vmc_flat/mujoco.yaml` | kd_l0=5 |
| `conf/ppo/task/xqrobotwl_jump_srl_vmc_flat/mujoco.yaml` | kd_l0=5, thrust_kp_scale 3.0, thrust_ff_scale 3.0, feedback_gain 0.5 |
| `src/unilab/envs/locomotion/xqrobotwl/vmc.py` | kd_l0 20→5 默认 |

## 验证方法

- 训练 TensorBoard 监控: reward / episode_length / jump_height 上升
- verify_jump 验证 (修复 [[23_verify_jump_config_drift]] 后):
  - PPO+VMC model_4000: air 9%, jump 0.081m, 存活 100%
  - VMC+SRL model_3000: air 9%, jump 0.068m, 存活 100%
- diag 轨迹确认真实下蹲→蹬伸→腾空→落地循环

## 后续计划

- 等 3 个训练跑满 10000
- 用修复后的 verify 做最终四算法对比 (SRL 用已有模型)
- 重生成 2x2 训练图 + 2x1 验证图
