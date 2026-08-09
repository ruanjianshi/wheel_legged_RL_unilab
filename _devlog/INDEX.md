# 全局开发索引

按机器人、任务、算法、时间线组织。

```
_devlog/<robot>/<task>/<algo>/<date>/<序号>_<slug>.md
```

## 机器人

| 机器人 | 说明 | 索引 |
|--------|------|------|
| xqrobotV2 | 轮腿双足 V2（mesh 碰撞体） | [→](xqrobotv2/INDEX.md) |
| xqrobotwl | 轮腿双足 V2 简化碰撞体 | [→](xqrobotwl/INDEX.md) |

## 时间线 (最新在前)

### 2026-08-07

| 机器人 | 任务 | 算法 | 标题 |
|--------|------|------|------|
| xqrobotwl | fall_recovery | ppo | [CPO 独立 conf/cpo 目录 + 约束 value loss 数值修复](xqrobotwl/fall_recovery/ppo/2026-08-07/04_cpo_conf_dir_and_value_scale.md) |
| xqrobotwl | fall_recovery | ppo | [FTSR 环境实现 + 冒烟验证 + 终止门控修复](xqrobotwl/fall_recovery/ppo/2026-08-07/03_fsr_env_and_verify.md) |
| xqrobotwl | fall_recovery | ppo | [移植 CPO 算法 (惩罚函数法)](xqrobotwl/fall_recovery/ppo/2026-08-07/02_cpo_port.md) |
| xqrobotwl | fall_recovery | ppo | [方案改向 FTSR (力引导+分阶段+CPO)](xqrobotwl/fall_recovery/ppo/2026-08-07/01_fsr_design.md) |

### 2026-08-06

| 机器人 | 任务 | 算法 | 标题 |
|--------|------|------|------|
| xqrobotwl | fall_recovery | ppo | [RL 环境 + 配置 + 启动脚本 (FSM 前馈翻身 + start_in_balance 课程)](xqrobotwl/fall_recovery/ppo/2026-08-06/02_env_and_config.md) |
| xqrobotwl | fall_recovery | ppo | [P1 可行性 — 贴地后空翻起身机制](xqrobotwl/fall_recovery/ppo/2026-08-06/01_p1_feasibility.md) |

### 2026-08-05

| 机器人 | 任务 | 算法 | 标题 |
|--------|------|------|------|
| xqrobotwl | backflip | ppo | [确定性翻转: 翻转期策略屏蔽, 纯 ff 驱动 (用户决定)](xqrobotwl/backflip/ppo/2026-08-05/08_deterministic_flip.md) |
| xqrobotwl | backflip | ppo | ["一翻就倒"根因: 控制率 100Hz 太粗, 改 200Hz (ctrl_dt=0.005)](xqrobotwl/backflip/ppo/2026-08-05/07_control_rate_200hz_fix.md) |
| xqrobotwl | backflip | ppo | [V2 续训发散 (动作爆炸) — 模型选择教训](xqrobotwl/backflip/ppo/2026-08-05/06_overfit_action_explosion_v2.md) |
| xqrobotwl | backflip | ppo | [奖励重平衡: ff_tracking 10 / flip_progress 30 (防刷旋转)](xqrobotwl/backflip/ppo/2026-08-05/05_rebalance_ff_tracking_vs_flip_progress.md) |
| xqrobotwl | backflip | ppo | [flip_complete 判据改物理事件锁存 (根因: 旋转积分阈值不可达)](xqrobotwl/backflip/ppo/2026-08-05/04_flip_complete_event_latch.md) |

### 2026-08-04

| 机器人 | 任务 | 算法 | 标题 |
|--------|------|------|------|
| xqrobotwl | backflip | ppo | [修复预热斜坡累乘 bug + 站立态 flip_progress 重置](xqrobotwl/backflip/ppo/2026-08-04/03_warmup_trigger_bugfix.md) |
| xqrobotwl | backflip | ppo | [环境开发: backflip.py + FSM前馈 + 相位门控奖励 + 训练管线](xqrobotwl/backflip/ppo/2026-08-04/02_env_development.md) |
| xqrobotwl | backflip | ppo | [开环脚本物理可行性验证 (360° 后空翻 + 落地 3.8°)](xqrobotwl/backflip/ppo/2026-08-04/01_physics_feasibility.md) |

### 2026-07-18

| 机器人 | 任务 | 算法 | 标题 |
|--------|------|------|------|
| xqrobotwl | jump | ppo | [跳跃姿态优化 + 站姿根因](xqrobotwl/jump/ppo/2026-07-18/06_posture_knee_wheel.md) |

### 2026-07-17

| 机器人 | 任务 | 算法 | 标题 |
|--------|------|------|------|
| xqrobotwl | jump | ppo | [站立姿态 + 先蹲后跳 + 按键优化](xqrobotwl/jump/ppo/2026-07-17/05_stand_still_crouch_first.md) |

### 2026-07-16

| 机器人 | 任务 | 算法 | 标题 |
|--------|------|------|------|
| xqrobotwl | jump | ppo | [姿态修正 + 腾空优化](xqrobotwl/jump/ppo/2026-07-16/03_posture_airtime.md) |

### 2026-07-15

| 机器人 | 任务 | 算法 | 标题 |
|--------|------|------|------|
| xqrobotwl | jump | ppo | [Kp 适配 + 存活率 + 后仰修正](xqrobotwl/jump/ppo/2026-07-15/02_kp_survival_posture.md) |

### 2026-07-14

| 机器人 | 任务 | 算法 | 标题 |
|--------|------|------|------|
| xqrobotwl | jump | ppo | [Phase-gated 跳跃奖励设计](xqrobotwl/jump/ppo/2026-07-14/01_phase_gated_rewards.md) |

### 2026-07-13

| 机器人 | 任务 | 算法 | 标题 |
|--------|------|------|------|
| xqrobotwl | — | — | [修复 keyframe 默认角度 + 终止条件](xqrobotwl/2026-07-13/01_fix_keyframe_and_termination.md) |

### 2026-07-12

| 机器人 | 任务 | 算法 | 标题 |
|--------|------|------|------|
| xqrobotwl | — | — | [创建 xqrobotwl](xqrobotwl/2026-07-12/01_create_robot.md) |
| xqrobotwl | flat | ppo | [flat_walk 训练配置](xqrobotwl/walk_flat/ppo/2026-07-12/01_flat.md) |
| xqrobotwl | rough | ppo | [rough_walk 训练配置](xqrobotwl/walk_rough/ppo/2026-07-12/01_rough.md) |
| xqrobotwl | stairs | np3o | [stairs NP3O 配置](xqrobotwl/stairs/np3o/2026-07-12/01_stairs.md) |
| xqrobotwl | jump | ppo | [jump 训练配置](xqrobotwl/jump/ppo/2026-07-12/01_jump.md) |

### 2026-07-11

| 机器人 | 任务 | 算法 | 标题 |
|--------|------|------|------|
| xqrobotV2 | rough | ppo | [全量评估三个任务 + 优化](xqrobotv2/rough_walk/ppo/2026-07-11/10_full_assess_optimize.md) |
| xqrobotV2 | rough | ppo | [三任务 Vy 归零](xqrobotv2/rough_walk/ppo/2026-07-11/09_vy_zero_all.md) |
| xqrobotV2 | stairs | np3o | [cost 0/1 二元化](xqrobotv2/stairs/np3o/2026-07-11/02_fix_cost_binary.md) |
| xqrobotV2 | jump | ppo | [修复跳跃传感器 + J 键持续化](xqrobotv2/jump/ppo/2026-07-11/01_fix_sensor_jump_key.md) |

### 2026-07-10

| 机器人 | 任务 | 算法 | 标题 |
|--------|------|------|------|
| xqrobotV2 | rough | ppo | [修复转圈退化：碰撞回退 + 命令解耦](xqrobotv2/rough_walk/ppo/2026-07-10/08_fix_spin_decouple.md) |

### 2026-07-09

| 机器人 | 任务 | 算法 | 标题 |
|--------|------|------|------|
| xqrobotV2 | rough | ppo | [rough 去台阶 + stairs 独立](xqrobotv2/rough_walk/ppo/2026-07-09/07_stairs_split_collision_force.md) |
| xqrobotV2 | stairs | np3o | [初始化 NP3O stairs](xqrobotv2/stairs/np3o/2026-07-09/01_np3o_stairs_init.md) |

### 2026-07-08

| 机器人 | 任务 | 算法 | 标题 |
|--------|------|------|------|
| xqrobotV2 | rough | ppo | [Tita RL 移植：动作平滑 + 地形课程](xqrobotv2/rough_walk/ppo/2026-07-08/06_tita_action_smooth_terrain_curriculum.md) |

### 2026-07-06

| 机器人 | 任务 | 算法 | 标题 |
|--------|------|------|------|
| xqrobotV2 | rough | ppo | [启用域随机化 + 腿长自适应](xqrobotv2/rough_walk/ppo/2026-07-06/05_enable_dr_leg_adaptive.md) |

### 2026-07-04

| 机器人 | 任务 | 算法 | 标题 |
|--------|------|------|------|
| xqrobotV2 | rough | ppo | [修复前向速度跟踪失败](xqrobotv2/rough_walk/ppo/2026-07-04/04_fix_forward_tracking.md) |

### 2026-07-01

| 机器人 | 任务 | 算法 | 标题 |
|--------|------|------|------|
| xqrobotV2 | flat | ppo | [修复髋关节对称性](xqrobotv2/flat_walk/ppo/2026-07-01/01_fix_hip_and_assess.md) |
| xqrobotV2 | flat | ppo | [iter=16200 全量评估](xqrobotv2/flat_walk/ppo/2026-07-01/02_full_eval_16200.md) |
| xqrobotV2 | rough | ppo | [地形可视化 + 配置](xqrobotv2/rough_walk/ppo/2026-07-01/01_terrain_setup.md) |
| xqrobotV2 | rough | ppo | [丰富地形配置](xqrobotv2/rough_walk/ppo/2026-07-01/02_enrich_terrain.md) |
| xqrobotV2 | rough | ppo | [启动 PPO 训练](xqrobotv2/rough_walk/ppo/2026-07-01/03_start_training.md) |
