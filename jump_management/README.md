# 跳跃管理包 — Wheeled-SRL 论文实验全集

## 两种方案

| 方案 | Task名 | 算法 | 特点 |
|------|--------|------|------|
| **纯PPO** | `XqRobotWLJumpFlat` | 单PPO | Phase-gated奖励, crouch->thrust->flight->landing |
| **Wheeled-SRL** | `XqRobotWLJumpSRLFlat` | SLIP前馈+PPO反馈 | 6状态FSM, 前馈-反馈融合, 轮地匹配 |

## 目录

```
training/       训练入口脚本
evaluate/       评估场景 & 批量评估
analyze/        论文图表 & 表格生成
config/         独立配置副本 (对照用)
results/        评估输出 (JSON)
paper/          论文出图脚本
```

## 快速开始

```bash
# 1. 纯PPO (GPU 0)
CUDA_VISIBLE_DEVICES=0 bash ../wheel_legged_RL_unilab/shell/xqrobotwl/jump/train_ppo_jump_flat.sh

# 2. Wheeled-SRL 完整版 (GPU 1)
CUDA_VISIBLE_DEVICES=1 bash training/train_srl_full.sh

# 3. 消融实验 (GPU 0/1 分配)
bash training/train_all.sh

# 4. 评估 (训练完成后)
bash evaluate/eval_batch.sh

# 5. 出图
uv run python analyze/plot_training_curves.py
uv run python analyze/plot_trajectory.py
uv run python analyze/plot_ablation.py
uv run python analyze/tab_baseline.py
```

## 评估场景

| 场景 | 说明 | 指标 |
|------|------|------|
| fix_01m | 固定vx, 周期跳 | 实际跳距, 最大高度, 着陆轮滑 |
| fix_02m | 同上, 更大vx | 同上 |
| fix_03m | 同上 | 同上 |
| random | vx~U[0,1.0] | 距离误差分布 |
| platform | 跳上0.15m平台 | 成功率 |

## 消融实验

| 模式 | 说明 | 配置 |
|------|------|------|
| full | 完整Wheeled-SRL | `mujoco.yaml` |
| no_fsm | 移除FSM前馈 | `ablate_no_fsm.yaml` |
| no_wheel_match | 移除轮地匹配奖励 | `ablate_no_wheel_match.yaml` |
| no_flight_mod | 飞行阶段轮速不斜升 | `ablate_no_flight_mod.yaml` |
| no_vel_track | 移除速度跟踪奖励 | `ablate_no_vel_track.yaml` |
