# [25] 诊断三个算法跳跃弱点 + 针对性改进 (±50 力矩 / 长 FSM 时序 / 破局奖励)

**日期**: 2026-08-08
**来源**: 用户要求改进另外三个算法的姿态和高度
**关联**: [[24_final_four_algo_compare_figs]], [[23_verify_jump_config_drift]]

---

## 问题描述

最终四算法对比 (修复后配置) 显示三个算法偏弱:

| 算法 | 跳跃高度 | air | 与 SRL 差距 |
|------|---------|-----|------------|
| SRL | 0.552 | 0.221 | 基准 |
| VMC+SRL | 0.251 | 0.107 | 高度仅 SRL 的 45% |
| PPO+VMC | 0.271 | 0.051 | 高度 49% |
| 纯PPO | 0.190 | 0.000 | 高度 34%, 不腾空 |

用户要求改进这三个。

## 根因分析

### 1. VMC 变体 (VMC+SRL / PPO+VMC): 关节力矩饱和 ±30 N·m

逐关节追踪蹬伸发现:
- 蹬伸需要膝盖扭矩 **-60~-134 N·m** (深蹲越深需要越多)
- ctrlrange 只有 ±30 → 实际只能输出 -30 → 蹬伸无力
- 裸 `compute_torques` (不 clip) 命令 L0=0.50 时膝盖正确伸展到 -64.6°、L0 到 0.578 → **物理方向正确, 单纯是饱和限制**
- FSM 时序也短: crouch 0.25s + thrust 0.20s, VMC 力控制下蹲/蹬伸来不及 (SRL 位置控制能瞬间到位)

验证: 原模型 + ±50 + 长时序 (0.35/0.30) 立即跳高 0.385m (原 0.248), 证明改进有效。

### 2. 纯PPO: 奖励稀疏死锁

- `jump_height` 奖励需要 `air_factor>0` (轮离地) 才给 → 死锁: 不离地→没奖励→不学跳
- `vertical_thrust` 门槛 `phase>=25` (触发 25 步后) → 触发早期无蹬伸反馈
- 策略学到"触发时轻微下蹲再站起"的偷懒解 (避免 base_height -60 惩罚), mean_reward 反而最高 (147.9)

## 解决方案

### VMC 变体: 扩大力矩范围 + 加长 FSM 时序

| 修改 | 内容 |
|------|------|
| `xqrobotwl_vmc.xml` | 腿 ctrlrange ±30 → **±50 N·m** (蹬伸需 -60~-134) |
| `vmc.py` 默认 | `fsm_crouch_time 0.25→0.35`, `fsm_thrust_time 0.20→0.30` |
| `srl_vmc YAML` | 显式配 `fsm_crouch_time=0.35`, `fsm_thrust_time=0.30` |

物理依据: ±50 在深蹲 L0=0.20 时提供 ~152N/腿 (超体重 91N 有加速余量), ±30 只有 91N (刚好支撑)。

### 纯PPO: 破局奖励

| 修改 | 内容 |
|------|------|
| `jump.py` `_reward_vertical_thrust` | phase 门槛 `25 → 5` (触发 5 步即奖励蹬伸尝试) |
| `jump.py` 新增 `_reward_anti_lazy` | 触发窗口 (phase 1-35) 内膝盖幅值 <0.5 rad 重罚, 逼真实蹲-蹬 (破"缓慢站起"局部最优) |
| `jump_flat YAML` | `anti_lazy: 60.0` |

v6 (仅 vertical_thrust 门槛降) 失败: 策略仍学"缓慢站起" (膝盖 0.15rad/100步, base_z 缓慢升), mean_reward 177 但 jump_height 0.009。→ 停止 v6, 用 anti_lazy 重启 v7。

v7 (anti_lazy v1 = 膝盖幅值惩罚) 有进展但退化: step 478 学会完整蹲-蹬-腾 (膝盖 -47°→+8°, 跳 0.275m), 之后退化到"膝盖动够但不离地" (anti_lazy 只要求膝盖幅度≥0.5rad, 策略满足后停止进步)。→ 停止 v7。

v8 (anti_lazy v2 = **触发窗口内 base_z 必须上升 ≥0.12m**): 直接从"身体是否真的升高"惩罚, 策略无法用"小幅动膝盖"逃避。启动于 13:36, GPU1。

**根本矛盾发现** (v9 之前): base_height 惩罚 = (base_z-0.55)²×60。跳跃时 base_z 0.55→0.65, 惩罚从 0→36, 而 jump_height 奖励上限只有 15。**跳得越高惩罚越重 → 策略理性选择不跳**。这就是纯PPO 多轮学不会跳跃的根本原因。

v9 (关键破局): 
1. **跳跃窗口内 base_height 锚定到触发时高度** (`_reward_base_height_jump`): 上升不再受 -60 惩罚, 只惩罚下蹲不够 (min(base_z, start_z))
2. anti_lazy 简化回膝盖幅值 (v1, 门槛 0.8rad): base_height 障碍移除后不再需要复杂乘积
启动于 14:06, GPU1。

### 新训练 (3 个, 10000 iter)

| 算法 | run 前缀 | GPU | 关键变化 |
|------|---------|-----|---------|
| PPO+VMC v4 | 2026-08-08_01-05-51 | 0 | ±50 + 长时序 |
| VMC+SRL v4 | 2026-08-08_01-05-51 | 1 | ±50 + 长时序 |
| 纯PPO v6 | 2026-08-08_01-08-12 | 1 | vertical_thrust 门槛 25→5 |

## 修改文件

| 文件 | 修改内容 |
|------|---------|
| `src/unilab/assets/robots/xqrobotwl/xqrobotwl_vmc.xml` | 腿 ctrlrange ±30→±50 |
| `src/unilab/envs/locomotion/xqrobotwl/vmc.py` | fsm_crouch_time 0.35 / fsm_thrust_time 0.30 默认 |
| `src/unilab/envs/locomotion/xqrobotwl/jump_srl.py` | `_update_fsm_state` 增加 crouch_time/thrust_time 参数 |
| `src/unilab/envs/locomotion/xqrobotwl/jump_vmc.py` | FSM 调用传入 vmc_cfg 的时序 |
| `conf/ppo/task/xqrobotwl_jump_srl_vmc_flat/mujoco.yaml` | fsm_crouch_time/thrust_time 显式 |
| `src/unilab/envs/locomotion/xqrobotwl/jump.py` | vertical_thrust phase 门槛 25→5 |

## 验证方法

- 冒烟测试: 纯PPO env 构建/step 正常
- 原模型 + 新配置: VMC+SRL jump 0.248→0.385, PPO+VMC 0.271→0.340
- 三个新训练进行中, 完成后 verify 验证

## 后续计划

- 等 3 个训练完成 (10000 iter)
- 用修复后 verify 重新四算法对比
- 重出论文图
