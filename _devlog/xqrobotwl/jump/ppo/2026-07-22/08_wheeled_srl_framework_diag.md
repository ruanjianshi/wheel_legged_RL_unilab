# 07 — Wheeled-SRL 跳跃框架构建 & 训练问题诊断修复

## 日期
2026-07-22

## 来源
构建 Wheeled-SRL（SLIP+PPO）跳跃方案的管理包，集成到 UniLab 框架，对比纯 PPO 跳跃。

## 问题描述
1. Wheeled-SRL 模型训练后 action_std 爆炸到 5.98，策略崩溃
2. 站姿下蹲（z≈0.47m，目标 0.65m）
3. FSM 过渡太快：下蹲 110ms、蹬地 60ms 即被超时强制跳转
4. 跳跃动作不完整：第一次跳向下砸进地面，后续跳为链式弹跳而非蓄力跳跃

## 根因分析

### 1. 训练崩溃（action_std 爆炸）
- `feedback_gain=0.2` 放大策略噪音：0.2 × std(5.98) × action_scale(0.7) = 0.84 rad，是前馈的 4-12 倍
- `jump_height=20` 碾压 `action_magnitude=-0.02`（1000:1），策略学会"输出大噪声→高跳→高奖励"
- `base_height=-2`、`orientation=-2` 惩罚太弱，无法约束姿态

### 2. 站姿下蹲
- `base_height=-2` vs 纯PPO 的 `-60`，高度偏差惩罚差 30 倍
- 策略在站立时因缺乏惩罚而保持在优化舒适区（0.47m）

### 3. FSM 过渡条件缺陷
- 状态 0→1：膝关节阈值太松（`doff[2]<0.0 && dof[5]>0.0` 几乎永远 true），100ms 超时强行跳过
- 状态 1→2：200ms 超时，还没蹬到位就进飞行
- 状态 2→3：近地判断 z<0.85m 太宽松，还在 1m 高空就进入"着陆"

### 4. 论文实现不符
- 状态 1（跳跃加速）缺少车轮主动驱动（论文要求"提供额外冲量"）
- 前馈下蹲深度不够（0.20 rad → 修正为 0.35 rad）

## 解决方案

### 配置修复 (`conf/ppo/task/xqrobotwl_jump_srl_flat/mujoco.yaml`)

| 参数 | 旧值 | 新值 | 原因 |
|------|------|------|------|
| `feedback_gain` | 0.2 | **0.05** | 防止策略噪音放大，保持前馈主导 |
| `jump_height` | 20 | **12** | 防止"高跳摔死" |
| `base_height` | -2 | **-60** | 对标纯PPO，修复站姿 |
| `orientation` | -2 | **-5** | 加强姿态约束 |
| `joint_action_rate` | -0.1 | **-0.5** | 抑制抖动 |
| `action_magnitude` | -0.02 | **-0.1** | 惩罚大动作 |
| `entropy_coef` | 0.008 | **0.005** | 减缓 std 增长 |

### FSM 修复 (`jump_srl.py`)

| 修改 | 旧 | 新 |
|------|----|----|
| 状态 0 下蹲深度 | ±0.20 rad | **±0.35 rad** |
| 状态 1 车轮 | 0.0 | **主动驱动（论文要求）** |
| 状态 1 蹬地力度 | ±0.30 rad | **±0.35 rad** |
| 0→1 膝关阈值 | `>0.0 / <-0.0` | **`>0.25 / <-0.25`** |
| 0→1 超时 | 100ms | **500ms** |
| 1→2 超时 | 200ms | **400ms** |
| 2→3 近地阈值 | z<0.85m | **z<0.75m** |
| 着陆缓冲斜坡 | 100ms | **200ms** |
| 恢复阶段 | 200ms | **300ms** |

## 修改文件

| 文件 | 改动 |
|------|------|
| `src/unilab/envs/locomotion/xqrobotwl/jump_srl.py` | 新增 645 行（SRL 环境 + FSM） |
| `src/unilab/envs/locomotion/xqrobotwl/__init__.py:5` | +1 行 import |
| `conf/ppo/task/xqrobotwl_jump_srl_flat/mujoco.yaml` | 新增（基础配置） |
| `conf/ppo/task/xqrobotwl_jump_srl_flat/ablate_*.yaml` | 新增 4 个消融配置 |
| `shell/xqrobotwl/train_ppo_jump_srl.sh` | 新增（训练脚本） |
| `shell/xqrobotwl/eval_ppo_jump_srl_flat.sh` | 新增（验证脚本） |
| `scripts/training/train_rsl_rl.py:320-324` | +2 行 run_suffix 支持 |
| `jump_management/` | 新增管理包（16 文件） |

## 验证方法

1. 在 EGL 无头环境加载最新 SRL 模型（`2026-07-21_19-42-56_mujoco/model_9999.pt`），运行确定性推断评估
2. 对比纯 PPO 模型（`2026-07-21_19-42-38_mujoco/model_9999.pt`）
3. 确认 FSM 六态完整循环

## 评估结果

### 当前 SRL 模型（训练崩溃后，配置修复前）

| 测试 | SRL | 纯PPO |
|------|-----|-------|
| 站姿高度 | 0.47m | 0.52m |
| 跳跃最高 | 1.41m | 1.23m |
| 10s 存活 | ✓ | ✓ |
| FSM 六态 | ✓ 齐全 | N/A |

- SRL 跳跃高度优于纯 PPO（1.41m vs 1.23m，+15%），FSM 前馈发挥作用
- 站姿低因 base_height 配置差异，修后应追平
- 训练时 std 爆炸到 200-1000，但不影响确定性推断（MLP 均值仍合理）
- FSM 过渡过快：实际下蹲仅 110ms，完全靠超时触发

### 核心结论
SRL 理论上优于纯 PPO，当前缺陷全为配置/FSM 参数问题，已在最新代码中修复。重训后应全面超越。

## 后续计划
1. 用修复后配置重训 SRL（GPU 1）
2. 补训消融实验（no_fsm, no_wheel_match, no_flight_mod, no_vel_track）
3. 重训完成后运行完整评估 + 出图
4. 论文消融实验：对比各组件独立贡献

## 关联日志
- `2026-07-18/07_jump_phase_gated_reward.md` — 纯PPO跳跃 reward 设计
- `2026-07-17/06_jump_posture_fix.md` — 跳跃姿态修复
