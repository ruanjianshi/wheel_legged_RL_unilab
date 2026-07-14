# XqRobotV2 策略评估框架 (Assess)

面向 XqRobotV2 轮腿机器人强化学习策略的标准化评估系统。支持**多任务 × 多算法 × 多版本**的定量对比。

## 快速开始

```bash
# 列出任务和算法
uv run assess/runner.py --list-tasks

# 平地步态 PPO 评估
uv run assess/runner.py -t flat_walk -a ppo -r <run> -c 16200

# 全量：评估 + 绘图 + CSV + 报告 + 轨迹
uv run assess/runner.py -t flat_walk -a ppo -r <run> -c <ckpt> \
    -s full --plot --csv --report --record

# 跨版本趋势
uv run assess/runner.py -t flat_walk -a ppo -r <run> \
    --trend --ckpts 5000,10000,15000,20000

# PPO vs SAC 对比（训练完成后）
uv run assess/runner.py --cmp \
    results/flat_walk/ppo/<session>/metrics.json \
    results/flat_walk/sac/<session>/metrics.json --plot
```

## 目录结构

```
assess/
├── README.md
├── tasks.py               # 任务 + 算法注册表
├── runner.py              # CLI 入口
├── metrics.py             # 22 项论文标准指标
├── scenarios.py           # 测试场景定义
├── recorder.py            # 全轨迹录制
├── plotter.py             # 6 种图表
├── exporter.py            # CSV / JSON 数据库
├── reporter.py            # Markdown 分析报告
│
├── results/               # <task>/<algo>/<session>/
├── plots/                 # <task>/<algo>/<session>/
├── reports/               # <task>/<algo>/<session>/
├── database/              # 累积历史库
└── configs/               # 自定义场景 YAML
```

## 已注册任务与算法

### 任务

| 任务 ID | 名称 | 机器人 | 默认场景 |
|---------|------|--------|---------|
| `flat_walk` | 平坦地面行走 | XqRobotV2 | full |
| `toe_walk` | 点足行走 | XqRobotV2 | toe_walk |

### 算法

| 算法 ID | 名称 | 类型 | 说明 |
|---------|------|------|------|
| `ppo` | PPO | On-policy | Proximal Policy Optimization (RSL-RL) |
| `sac` | SAC | Off-policy | Soft Actor-Critic |
| `appo` | APPO | On-policy | Asynchronous PPO |
| `td3` | TD3 | Off-policy | Twin Delayed DDPG |

### 已训练的组合

| 组合 | 日志路径 |
|------|---------|
| `flat_walk/ppo` | `logs/rsl_rl_ppo/XqRobotV2WalkFlat` |
| `toe_walk/ppo` | `logs/rsl_rl_ppo/XqRobotV2ToeWalkFlat` |

添加新组合：在 `tasks.py` 中调用 `register("task", "algo")`。

## 命令行参数

| 名称 | 场景数 | 说明 |
|------|--------|------|
| `decoupling` | 6 | Vx/Vy 解耦快速测试 |
| `full` | 16 | 全速度扫频（vx/vy/vyaw/后退/组合）|
| `standing` | 1 | 静止站立稳定性 |

## CLI 参数

| 参数 | 说明 |
|------|------|
| `-t, --task` | 任务名（默认 `flat_walk`）|
| `-a, --algo` | 算法名（默认 `ppo`）|
| `-r, --run` | 训练运行名 |
| `-c, --ckpt` | checkpoint 迭代数（不填取最新）|
| `-s, --suite` | 场景集（不填用任务默认）|
| `--plot` | 生成图表 PNG |
| `--csv` | 导出 CSV |
| `--report` | 生成 Markdown 分析报告 |
| `--record` | 录制全轨迹时间序列 |
| `--trend` | 跨 checkpoint 趋势分析 |
| `--ckpts` | 趋势分析的 checkpoint 列表（逗号分隔）|
| `--cmp` | 对比多个评估结果 JSON |
| `--list-tasks` | 列出已注册任务+算法组合 |

## 评估指标

参考论文：RMA (2021)、ANYmal (2019)、Cassie (2020)、IsaacGym (2022)、DreamWaQ (2023)。

### 指令跟踪 (Command Tracking)

| 指标 | 说明 | 参考 |
|------|------|------|
| `vx_tracking_rmse` | 前向速度 RMSE | RMA, ANYmal |
| `vy_tracking_rmse` | 横向速度 RMSE | RMA |
| `vyaw_tracking_rmse` | 偏航角速度 RMSE | ANYmal |
| `avg_vx` / `avg_vy` | 实际平均速度 | — |
| `vel_tracking_ratio` | 速度跟踪比（实际/指令） | — |
| `vel_coupling` | Vx/Vy 串扰量 | XqRobotV2 |

### 稳定性 (Stability)

| 指标 | 说明 |
|------|------|
| `base_height_mean` | 基座平均高度 |
| `base_height_std` | 高度波动（越小越稳） |
| `orientation_rmse_deg` | 姿态角 RMSE（度） |
| `yaw_stability` | 偏航角速度标准差 |

### 运动质量 (Motion Quality)

| 指标 | 说明 |
|------|------|
| `action_smoothness` | 相邻帧动作变化量 |
| `joint_velocity_mean` | 平均关节速度 |
| `gait_symmetry` | 左右腿运动对称性 |

### 能效 (Energy Efficiency)

| 指标 | 说明 |
|------|------|
| `mean_torque` | 平均关节力矩 |
| `cost_of_transport` | 运输能耗 CoT = P/(mgv) |

### 步态特征 (Gait Characteristics)

| 指标 | 说明 |
|------|------|
| `step_frequency` | 步频（FFT 主频） |
| `leg_workspace_utilization` | 关节运动范围 |

## 输出示例

### 运行摘要

```
Task:  Flat Ground Walking (XqRobotV2, PPO)
Model: 2026-07-01_13-55-35_mujoco @ iter 16200
Suite: decoupling (6 scenarios)
Output: assess/results/flat_walk/<session>/

=====================================================================================
RESULTS — 2026-07-01_13-55-35_mujoco iter=16200  [flat_walk]
=====================================================================================
Scenario                    vx      vy  vx_rmse vy_xtalk  base_h
-------------------------------------------------------------------------------------
fwd_vx=0.6               0.119   0.007    0.525    0.007   0.525
fwd_vx=0.3               0.331   0.016    0.040    0.016   0.537
fwd_vx=-0.3             -0.032  -0.009    0.269    0.009   0.510
lat_vy=+0.3              0.170   0.019    0.170    0.170   0.524
lat_vy=-0.3              0.174  -0.018    0.177    0.174   0.532
fwd+lat                  0.332   0.005    0.042    0.000   0.533
```

### 生成文件

```
JSON:   results/flat_walk/<session>/metrics.json
Traj:   results/flat_walk/<session>/trajectory.json
CSV:    results/flat_walk/<session>/metrics.csv
Plots:  plots/flat_walk/<session>/{velocity,stability,gait,metric_bars}.png
Report: reports/flat_walk/<session>/analysis.md
```

## 自定义场景

在 `configs/<task>/` 下创建 YAML：

```yaml
name: my_suite
description: "Custom evaluation"
ctrl_dt: 0.01
scenarios:
  - name: "fast"
    cmd: [0.8, 0.0, 0.0, 0.0, 0.65]
    duration: 5.0
    warmup: 2.0
    description: "High speed"
```

```bash
uv run assess/runner.py -t flat_walk -r <run> -c <ckpt> --suite-file configs/flat_walk/my_suite.yaml
```

## 工作流

```
训练 PPO  → 定时评估 checkpoint → 趋势图
训练 SAC  → 定时评估 checkpoint → 趋势图
         ↓
PPO vs SAC → 同场景对比报告 → 雷达图 + 柱状图
```

添加新算法时，只需在 `tasks.py` 中注册组合：

```python
# 例如：SAC 训练完成后
register("flat_walk", "sac", log_subdir="offpolicy/XqRobotV2WalkFlat")
```

## 参考文献

1. Kumar, A., et al. "RMA: Rapid Motor Adaptation for Legged Robots." RSS, 2021.
2. Hwangbo, J., et al. "Learning Agile and Dynamic Motor Skills for Legged Robots." Science Robotics, 2019.
3. Xie, Z., et al. "Learning Locomotion Skills for Cassie." ICRA, 2020.
4. Rudin, N., et al. "Learning to Walk in Minutes Using Massively Parallel Deep RL." CoRL, 2022.
5. Nahrendra, I., et al. "DreamWaQ: Learning Robust Quadrupedal Locomotion with Implicit Terrain Imagination." RAL, 2023.
6. Miki, T., et al. "Learning Robust Perceptive Locomotion for Quadrupedal Robots in the Wild." Science Robotics, 2022.


