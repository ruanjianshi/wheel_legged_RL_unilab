# UniLab — Universal Lab for Robot Learning

> 高性能、模块化、多后端 RL 训练基础设施。支持双足/四足/人形/轮腿/灵巧手等多类机器人，
> 集成 PPO / SAC / TD3 / APPO / HIM-PPO / FlashSAC 等主流强化学习算法，
> 提供 MuJoCo / Motrix 双物理后端。
>
> 更新：2026-07-02

---

## 一、项目架构

```
wheel_legged_RL_unilab/
│
├── src/unilab/                   # ★ 核心库
│   ├── algos/                    # RL 算法实现 (torch + mlx)
│   │   ├── torch/                #   PPO(rsl-rl), APPO, SAC, TD3, FlashSAC, HORA, HIM-PPO
│   │   ├── mlx/                  #   MLX PPO (Apple Silicon)
│   │   └── common/               #   网络骨架 (MLP, LayerNormMLP, RunningMeanStd)
│   │
│   ├── envs/                     # 环境实现
│   │   ├── locomotion/           #   移动任务: go1, go2, go2w, go2_arm, g1, xqrobotV2, smallHumanoidRobot
│   │   ├── manipulation/         #   操作任务: allegro, sharpa, stewart
│   │   ├── motion_tracking/      #   动捕模仿: g1, x2
│   │   └── common/               #   共享函数 (rewards, commands, terrain_spawn, height_scan)
│   │
│   ├── terrains/                 # 地形生成系统
│   │   ├── terrain_generator.py  #   主生成器 (网格 + 子地形)
│   │   ├── heightfield_terrains.py # 7 种子地形: 平地/台阶/楼梯/斜坡/波浪/粗糙
│   │   └── config.py             #   预设配置
│   │
│   ├── assets/                   # 机器人模型 + 场景 XML
│   │   ├── robots/               #   12 种机器人 (xqrobotV2, go1, go2, go2w, g1, ...)
│   │   └── motions/              #   动捕数据 (g1, x2)
│   │
│   ├── base/                     # 核心抽象
│   │   ├── np_env.py             #   NpEnv 契约 (reset/step/obs 规范)
│   │   ├── backend/base.py       #   SimBackend 抽象接口 (700+ lines)
│   │   ├── backend/mujoco/       #   MuJoCo 后端实现
│   │   ├── backend/motrix/       #   Motrix GPU 后端实现
│   │   ├── registry.py           #   环境注册工厂 (envcfg / env 装饰器)
│   │   ├── scene.py              #   场景配置 (模型 + 地形 + fragment)
│   │   ├── curriculum.py         #   课程学习 (PenaltyCurriculum)
│   │   └── observations.py       #   观测分组规范
│   │
│   ├── dr/                       # 域随机化
│   │   ├── manager.py            #   DR 管理器 (init/interval/reset 三阶段)
│   │   ├── provider.py           #   DR 提供者基类
│   │   └── types.py              #   随机化载荷数据结构
│   │
│   ├── training/                 # 训练基础设施
│   │   ├── experiment.py         #   ExperimentTracker (wandb/tensorboard)
│   │   ├── run.py                #   checkpoint 解析 + 加载
│   │   ├── rsl_rl.py             #   RSL-RL 桥接
│   │   ├── sim2sim.py            #   跨后端契约验证
│   │   └── reward.py             #   奖励配置提取
│   │
│   ├── ipc/                      # 进程间通信 (异步训练)
│   │   ├── async_runner.py       #   异步 runner
│   │   ├── replay_buffer.py      #   重放缓冲区
│   │   └── weight_sync.py        #   权重同步
│   │
│   ├── visualization/            # 可视化 & 回放
│   ├── tools/                    # CLI 工具 (export, import, render, viz)
│   └── cli.py                    # 命令行入口 (train / eval / demo)
│
├── conf/                         # ★ Hydra 配置体系
│   ├── ppo/                      #   基础 + 任务配置
│   │   ├── config.yaml
│   │   └── task/
│   ├── appo/                     #   APPO 配置
│   ├── offpolicy/                #   SAC/TD3/FlashSAC
│   │   ├── config.yaml
│   │   ├── algo/
│   │   └── task/
│   ├── ppo_him/                  #   HIM-PPO
│   └── hora_distill/             #   HORA 蒸馏
│
├── scripts/training/             # 训练入口脚本 (train_rsl_rl, train_appo, train_offpolicy, ...)
├── shell/                        # 便利脚本 (train / eval / tensorboard)
├── assess/                       # ★ 策略评估框架 (独立于 Hydra)
│   ├── runner.py                 #   CLI 入口
│   ├── metrics.py                #   22 项评估指标 (5 类)
│   ├── scenarios.py              #   4 套预设场景
│   ├── plotter.py / reporter.py  #   图表 + 报告生成
│   └── results/ plans/ reports/  #   评估输出
├── _devlog/                      # 开发日志 (AI 自记录)
├── docs/                         # 文档 (Sphinx)
├── tools/                        # 外部工具 (mujoco 可视化等)
├── tests/                        # 测试
├── backup/                       # 模型备份
├── notebook/                     # Jupyter 笔记本
├── AGENTS.md                     # AI 智能体开发规范
├── pyproject.toml                # 项目配置 (uv 包管理)
└── go.sh                         # 一键启动键盘控制评估
```

---

## 二、支持的机器人

### 移动机器人 (Locomotion)

| 机器人 | 目录 | 类型 | 任务 |
|--------|------|------|------|
| **XqRobotV2** | `xqrobotV2/` | 轮腿双足 | flat walk, rough walk, jump flat, toe walk |
| **Go1** | `go1/` | 四足 | joystick flat, joystick rough |
| **Go2** | `go2/` | 四足 | joystick flat, joystick rough, footstand |
| **Go2W** | `go2w/` | 轮式四足 | joystick flat, joystick rough |
| **Go2-Arm** | `go2_arm/` | 四足+手臂 | manip-loco |
| **G1** | `g1/` | 人形 | walk flat/rough, motion tracking, flip, climb, box, wall flip |
| **smallHumanoidRobot** | `smallHumanoidRobot/` | 小人形 | walk flat |
| **X2** | `x2/` | 人形 | wall flip tracking |

### 操作机器人 (Manipulation)

| 机器人 | 目录 | 类型 | 任务 |
|--------|------|------|------|
| **Allegro Hand** | `allegro_hand/` | 四指灵巧手 | inhand manipulation, grasp |
| **Sharpa Wave** | `sharpa_wave/` | 软体手 | inhand manipulation, grasp |
| **Stewart** | `stewart/` | 六自由度平台 | balance |

---

## 三、RL 算法

| 算法 | 类型 | 配置目录 | 特点 |
|------|------|----------|------|
| **PPO** | On-policy | `conf/ppo/` | RSL-RL 实现, 主力算法 |
| **APPO** | Async on-policy | `conf/appo/` | 异步 PPO + V-trace, 多进程 |
| **SAC** | Off-policy | `conf/offpolicy/` | 分布式 SAC, 多 GPU, distributional critic |
| **FlashSAC** | Off-policy | `conf/offpolicy/` | 高性能 SAC (flash attention) |
| **TD3** | Off-policy | `conf/offpolicy/` | Twin Delayed DDPG |
| **HIM-PPO** | On-policy | `conf/ppo_him/` | History-conditioned PPO, 部分可观测 |
| **HORA** | Multi | `conf/hora_distill/` | 多算法框架 (PPO/SAC/APPO) + 蒸馏 |
| **MLX PPO** | On-policy | `conf/ppo/config_mlx.yaml` | Apple Silicon 专用 |
| **SAC (MuJoCo)** | Off-policy | `conf/offpolicy/algo/sac.yaml` | Distributional SAC (101 atoms) |

### 算法超参数默认值

| 参数 | PPO | SAC | TD3 | APPO |
|------|-----|-----|-----|------|
| num_envs | 4096 | 4096 | 4096 | 4096 |
| num_steps_per_env | 24 | - | - | 24 |
| learning_rate | 1e-3 | 3e-4 | 3e-4 | 1e-3 |
| gamma | 0.99 | 0.97 | 0.99 | 0.99 |
| actor_hidden | [512,256,128] | [512] | [256] | [512,256,128] |
| critic_hidden | [512,256,128] | [768] | [512] | [512,256,128] |
| activation | elu | elu | elu | elu |
| entropy_coef / alpha | 0.01 | auto (0.01) | - | 0.01 |

---

## 四、地形系统

### 7 种子地形类型

| 地形 | 实现 | 参数 |
|------|------|------|
| **flat** | `HfFlatTerrainCfg` | 高度 = 0 |
| **pyramid_stairs** | `HfPyramidStairsTerrainCfg` | 台阶高度递增/递减, platform_width=2.0, border_width=0.2 |
| **pyramid_stairs_inv** | `HfInvertedPyramidStairsTerrainCfg` | 反向台阶 |
| **hf_pyramid_slope** | `HfPyramidSlopedTerrainCfg` | 斜坡 (slope_range 0.0-0.15) |
| **hf_pyramid_slope_inv** | | 反向斜坡 |
| **random_rough** | `HfRandomUniformTerrainCfg` | 随机高度场 (max_height=0.05) |
| **wave_terrain** | `HfWaveTerrainCfg` | 正弦波地形 (amplitude=0.05, period=2.0) |

### 地形配置

```python
# 粗糙地形网格 (训练用)
ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    num_rows=10, num_cols=20,        # 10×20 网格
    size=8.0, border_width=20.0,     # 每格 8m
    horizontal_scale=0.1,            # 分辨率 10cm
    sub_terrains={
        "flat": 0.20,
        "pyramid_stairs": 0.20,      # 每个子地形在不同网格随机生成
        "pyramid_stairs_inv": 0.20,
        "hf_pyramid_slope": 0.10,
        "hf_pyramid_slope_inv": 0.10,
        "random_rough": 0.10,
        "wave_terrain": 0.10,
    }
)

# 台阶课程 (难度递增)
STAIRS_TERRAINS_CFG = TerrainGeneratorCfg(
    num_rows=10, num_cols=4,
    curriculum=True,                  # 课程模式: 按行难度递增
)
```

### 场景配置 (`SceneCfg`)

```python
SceneCfg(
    model_file="robot.xml",            # 机器人 UVo 模型
    fragment_files=["locomotion_task.xml"],  # 任务 keyframe
    terrain=TerrainSceneCfg(           # 地形 (optional)
        generator=TerrainGeneratorCfg(),
        hfield_name="terrain_hfield",
        geom_name="floor",
    ),
)
```

- `model_file`: 机器人本体 XML (body/joint/actuator/sensor)
- `fragment_files`: 任务场景 XML (keyframe/灯光/相机), **禁止放 keyframe 到 robot.xml**
- `terrain`: 非平坦任务的地形配置，Geom 挂载到 heightfield

---

## 五、域随机化 (Domain Randomization)

### 支持的维度

| 维度 | 说明 |
|------|------|
| `randomize_base_mass` | 基座质量 (添加 ±5kg 噪声) |
| `randomize_ground_friction` | 地面摩擦系数 (0.5-2.0) |
| `randomize_kp / kd` | PD 增益 (±20%) |
| `random_com` | 质心偏移 (±2cm) |
| `randomize_leg_length` | 腿几何缩放 (95%-105%) |
| `push_robots` | 间隔扰动 (外力脉冲) |
| `randomize_init_yaw` | 初始偏航角 |

### 三阶段随机化

| 阶段 | 时机 | 内容 |
|------|------|------|
| **Init** | 环境创建时 | 模型变体 (geometry overrides) |
| **Reset** | 每次 reset | 质量/摩擦/COM/Kp/Kd |
| **Interval** | 训练中 | 外力推动 |

### `DomainRandomizationManager`

```python
manager = DomainRandomizationManager(cfg.dr, backend)
manager.apply_init_randomization()       # 冷路径
manager.apply_reset_randomization(env_ids) # 每个 episode
manager.apply_interval_randomization()    # 训练中
```

---

## 六、课程学习 (Curriculum Learning)

### PenaltyCurriculum

动态缩放惩罚性奖励权重：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `initial_scale` | 0.5 | 初始惩罚权重缩放 |
| `min_scale` | 0.5 | 最小惩罚权重 |
| `max_scale` | 1.0 | 最大惩罚权重 |
| `level_up_threshold` | 750.0 | episode_length 超过此值→增加惩罚 |
| `level_down_threshold` | 150.0 | episode_length 低于此值→减少惩罚 |

### 速度命令课程

```yaml
# conf/ppo/config.yaml
curriculum:
  enabled: true
  vel_step: 0.002              # 每次扩展的线速度步长
  ang_vel_step: 0.004          # 角速度步长
  init_vx_fraction: 0.3        # 初始速度 = 配置上限的 30%
  update_interval: 25          # 每 25 步评估一次
  error_threshold: 0.35        # 追踪误差阈值
```

课程从配置值的 30% 开始，当追踪误差满足阈值时对称扩展速度范围。

### 地形课程

`STAIRS_TERRAINS_CFG` 设置 `curriculum=True`，按网格行排序地形难度（easy→moderate→challenging）。

---

## 七、奖励函数系统

### 架构

```
Hydra YAML (reward.scales) → resolve_reward_dict() → run_reward_dispatch()
```

`run_reward_dispatch()`:
1. 构建 `RewardContext`（传感器快照）
2. 遍历 `scales`，调用对应奖励函数
3. 加权求和 × `ctrl_dt`
4. 写入 `info["log"]` 用于 TensorBoard

### 共享奖励函数 (25+)

| 函数 | 类型 | 说明 |
|------|------|------|
| `tracking_lin_vel` | exp 跟踪 | `exp(-v_err²/σ²)`, σ=0.3 |
| `tracking_ang_vel` | exp 跟踪 | `exp(-ω_err²/σ²)` |
| `lin_vel_z` | 惩罚 | 垂直速度² |
| `ang_vel_xy` | 惩罚 | roll/pitch 角速度² |
| `base_height` | 惩罚 | `(h - h_target)²` |
| `orientation` | 惩罚 | `gravity_xy²` — 倾斜 |
| `action_rate` | 惩罚 | 动作变化率² — 平滑 |
| `torques` | 惩罚 | 关节力矩² — 节能 |
| `alive` | 奖励 | 存活常数 |
| `feet_air_time` | 奖励 | 脚离地时间 |
| `joint_power` | 惩罚 | 关节功率² |
| `stand_still` | 惩罚 | 零指令时的不动 |
| `joint_pos_limits` | 惩罚 | 关节极限惩罚 |
| `forward_progress` | 奖励 | 前向位移 |
| `upright` | 奖励 | 直立姿态 |

### XqRobotV2 奖励配置

| 奖励项 | Scale | 说明 |
|--------|-------|------|
| `tracking_lin_vel` | +1.5 | 线速度追踪 |
| `tracking_ang_vel` | +1.5 | 角速度追踪 |
| `lin_vel_z` | -0.2 | 垂直速度 |
| `ang_vel_xy` | -0.05 | roll/pitch 角速度 |
| `base_height` | -5.0 | 高度目标 0.65m |
| `orientation` | **-10.0** | 倾斜惩罚 (最重要) |
| `joint_action_rate` | -0.1 | 腿动作平滑 |
| `wheel_action_rate` | -0.005 | 轮动作平滑 |
| `similar_calf` | -1.0 | 腿对称 |
| `hip_roll` | -2.0 | 髋关节内收 |
| `wheel_symmetry` | -0.5 | 轮对称 |
| `tsk` | -2.0 | 髋差动跟踪 |
| `feet_distance` | -1.0 | 轮距 [0.3, 0.6] |
| `alive` | +1.0 | 存活 |

---

## 八、后端

| 后端 | 安装 | 平台 | 特性 |
|------|------|------|------|
| **MuJoCo** | `pip install mujoco-uni==3.8.0` | Linux, macOS | 主力后端, 可视化完善, 域随机化全面 |
| **Motrix** | `pip install motrixsim-core==0.8.2` | Linux (Pascal+), macOS | GPU 加速, 大批量训练 |

### 后端切换

```bash
# PPO + MuJoCo
uv run train --algo ppo --task xqrobotV2_walk_flat --sim mujoco

# PPO + Motrix
uv run train --algo ppo --task xqrobotV2_walk_flat --sim motrix
```

配置路径自动映射: `conf/ppo/task/xqrobotV2_walk_flat/{mujoco,motrix}.yaml`

### Sim2Sim 契约

跨后端评估时，以下字段必须一致（不一致会报 `CrossBackendIncompatibleError`）:
- `algo.obs_groups` — 观测分组
- `env.control_config.action_scale` — 动作缩放
- `algo.policy.{actor,critic}_hidden_dims` — 网络结构
- `algo.empirical_normalization / obs_normalization` — 归一化

契约验证: `src/unilab/training/sim2sim.py`
契约审计: `scripts/audit_sim2sim_contracts.py`

---

## 九、环境契约

### NpEnv 规范

```python
class NpEnv:
    @property
    def obs_groups_spec(self) -> dict[str, int]:
        """返回观测分组维度: {"obs": 297, "critic": 324}"""
    
    def reset(self) -> tuple[dict[str, np.ndarray], dict]:
        """返回 (obs_dict, info_dict)"""
    
    def step(self, action: np.ndarray) -> NpEnvState:
        """返回 NpEnvState(obs, reward, terminated, truncated, info)"""
```

### 注册系统

```python
# 环境配置
@envcfg("XqRobotV2WalkFlat")
@dataclass
class XqRobotV2WalkFlatCfg(XqRobotBaseCfg):
    ...

# 环境类 (可注册多后端)
@env("XqRobotV2WalkFlat", sim_backend="mujoco")
class XqRobotV2WalkFlatEnv(XqRobotBaseEnv):
    ...

@env("XqRobotV2WalkFlat", sim_backend="motrix")
class XqRobotV2WalkFlatMotrixEnv(XqRobotBaseEnv):
    ...
```

工厂创建:
```python
from unilab.base.registry import make
env = make("XqRobotV2WalkFlat", "mujoco", num_envs=4096)
```

---

## 十、策略评估框架 (`assess/`)

独立于训练和 Hydra 的策略评估系统。

### 结构

```
assess/
├── tasks.py                     # 任务+算法注册
├── runner.py                    # CLI (评估/趋势/比较/列表)
├── metrics.py                   # 22 项指标 (5 类)
├── scenarios.py                 # 4 套场景
├── recorder.py / exporter.py    # 轨迹录制 + CSV 导出
├── plotter.py / reporter.py     # 图表 + Markdown 报告
├── results/<task>/<algo>/<session>/
├── plots/<task>/<algo>/<session>/
├── reports/<task>/<algo>/<session>/
└── database/
```

### 评估场景

| 场景 | 场景数 | 说明 |
|------|--------|------|
| `decoupling` | 6 | Vx/Vy 方向解耦测试 (前/后/侧/对角) |
| `full` | 16 | 全量扫描 (速度 0.1-0.6, 侧向 0.1-0.3, 偏航 0.5-1.0, 后退) |
| `standing` | 1 | 零指令稳定性测试 |
| `toe_walk` | (待定) | 脚趾行走 |

### 常用命令

```bash
# 单次评估 (默认 flat_walk/ppo)
uv run assess/runner.py -t flat_walk -a ppo -r <run> -c <ckpt>

# 全量评估 + 绘图 + CSV + 报告
uv run assess/runner.py -t flat_walk -a ppo -r <run> -c <ckpt> \
    -s full --plot --csv --report --record

# 跨 checkpoint 趋势
uv run assess/runner.py -t flat_walk -a ppo -r <run> \
    --trend --ckpts 5000,10000,15000,20000

# 跨模型比较
uv run assess/runner.py --cmp \
    results/flat_walk/ppo/<s>/metrics.json \
    results/rough_walk/ppo/<s>/metrics.json --plot

# 列出已注册任务
uv run assess/runner.py --list-tasks
```

### 22 项评估指标

| 类别 | 指标 | 说明 |
|------|------|------|
| **跟踪** | vx/vy/vyaw RMSE, avg_velocity, tracking_ratio | 命令跟踪精度 |
| **稳定** | base_height_rmse, roll/pitch_std, max_tilt, survival_rate | 姿态稳定性 |
| **质量** | jerk_xy, jerk_z | 运动平滑度 |
| **能效** | mean_torque, mean_power, COT | 运输成本 |
| **步态** | stance_duty_factor, swing_symmetry, step_frequency | 步态特征 |

---

## 十一、XqRobotV2 专项

### 机器人说明

轮腿双足机器人: 6 个腿关节 + 2 个轮关节 = 8 执行器

### 关节定义

| 索引 | 关节 | 轴 | 范围 | 控制 |
|------|------|-----|------|------|
| 0 | L_hip | x (roll) | [-π, π] | 位置, kp=30 |
| 1 | L_thigh | y (pitch) | [0, 2.09] | 位置, kp=30 |
| 2 | L_calf | y (pitch) | [-0.87, 0.87] | 位置, kp=30 |
| 3 | L_wheel | y | 无限 | **速度, kv=1** |
| 4 | R_hip | x (roll) | [-π, π] | 位置, kp=30 |
| 5 | R_thigh | y (pitch) | [0, 2.09] | 位置, kp=30 |
| 6 | R_calf | y (pitch) | [-0.87, 0.87] | 位置, kp=30 |
| 7 | R_wheel | y | 无限 | **速度, kv=1** |

> **髋标定**: L_hip (+→内缩, -→外展), R_hip (+→外展, -→内缩)
> 对称外展站位: `DEFAULT_LEG_ANGLES = [-0.1, 0.1, -0.1, 0.1, 0.1, -0.1]` (L/R 髋均外展)

### 动作空间 (8 维)

| 索引 | 关节 | 缩放 | 目标 |
|------|------|------|------|
| 0-5 | 腿 | `action × 0.25 + default` | 位置 (rad) |
| 6-7 | 轮 | `action × 10.0` | 速度 (rad/s) |

### 观测空间

9 帧历史堆叠:

| 观测组 | 维度 | 内容 |
|--------|------|------|
| **Actor** | 33×9 = **297** | gyro(3), gravity(3), leg_diff(6), leg_vel(6), wheel_vel(2), last_actions(8), commands(5) |
| **Critic** | 36×9 = **324** | Actor(33) + base_linvel(3) — 特权观测 |

### 5D 命令

| 维度 | 范围 (flat) | 范围 (rough) | 说明 |
|------|-------------|--------------|------|
| vx | [-0.6, 0.6] | [-1.0, 1.0] | 前向速度 (m/s) |
| vy | [-0.3, 0.3] | [-0.5, 0.5] | 侧向速度 |
| vyaw | [-1.0, 1.0] | [-1.5, 1.5] | 角速度 (rad/s) |
| tsk | [-0.1, 0.1] | [-0.1, 0.1] | 髋差动 |
| height | [0.45, 0.85] | [0.40, 0.90] | 目标高度 (m) |

### 终止条件

| 条件 | 值 |
|------|-----|
| 倾斜角 | > 60° |
| 底盘高度 | < 0.20m |
| 大腿塌陷 | < 0.02 rad |
| 小腿过伸 | \|angle\| > 0.85 rad |

### 训练任务

| 任务 | 配置 | 地形 | 状态 |
|------|------|------|------|
| `xqrobotV2_walk_flat` | `conf/ppo/task/xqrobotV2_walk_flat/mujoco.yaml` | 平面 | ✅ 完成 (20000 iter) |
| `xqrobotV2_walk_rough` | `conf/ppo/task/xqrobotV2_walk_rough/mujoco.yaml` | 粗糙地形 | 🔄 训练中 |
| `xqrobotV2_jump_flat` | `conf/ppo/task/xqrobotV2_jump_flat/mujoco.yaml` | 平面 | 待训练 |
| `xqrobotV2_toe_walk_flat` | `conf/ppo/task/xqrobotV2_toe_walk_flat/mujoco.yaml` | 平面 | 待训练 |

---

## 十二、快速开始

### 环境

```bash
cd /home/robot/xiaoq/wheel_legged_RL_unilab
source .venv/bin/activate  # Python 3.13, uv 管理
```

### XqRobotV2 训练

```bash
# Flat Walk (GPU 0)
CUDA_VISIBLE_DEVICES=0 bash shell/train_ppo_flat.sh

# Rough Walk (GPU 1)
CUDA_VISIBLE_DEVICES=1 bash shell/train_ppo_rough.sh
```

### 通用训练命令

```bash
# PPO
uv run train --algo ppo --task <task> --sim mujoco

# SAC (offpolicy)
uv run train --algo sac --task <task> --sim mujoco

# APPO (async)
uv run train --algo appo --task <task> --sim mujoco
```

### 评估

```bash
# 键盘控制
bash shell/eval_ppo_flat.sh --keyboard

# 策略评估
uv run assess/runner.py -t flat_walk -a ppo -r <run> -c <ckpt>
```

### TensorBoard

```bash
uv run tensorboard --logdir logs/rsl_rl_ppo/XqRobotV2WalkFlat
```

---

## 十三、训练流程

```
CLI (train --algo ppo --task xqrobotV2_walk_flat --sim mujoco)
  │
  ├─ Hydra 组合配置: conf/ppo/config.yaml + conf/ppo/task/xqrobotV2_walk_flat/mujoco.yaml
  │
  ├─ ExperimentTracker.start() → run_config.json (含 contract_snapshot)
  │
  ├─ registry.make("XqRobotV2WalkFlat", "mujoco", num_envs=4096)
  │     └─ SceneCfg → MuJoCo backend → terrain generator → env
  │
  ├─ DomainRandomizationManager (init/reset/interval)
  │
  ├─ OnPolicyRunner(env, ActorCritic, ppo_cfg)
  │     ├─ for iter in 1..max_iterations:
  │     │   ├─ rollout: env.step × num_steps_per_env
  │     │   ├─ PPO update: 5 epochs × 4 mini-batches
  │     │   ├─ Curriculum update
  │     │   └─ Logging → TensorBoard/W&B
  │     └─ save checkpoint → model_{iter}.pt
  │
  └─ Optional: play after training
```

### Checkpoint 格式

```
logs/<algo_log>/<task>/<timestamp>/model_<iter>.pt
例: logs/rsl_rl_ppo/XqRobotV2WalkFlat/2026-06-30_22-49-32_mujoco/model_20000.pt
```

Checkpoint 包含: `model_state_dict`, `optimizer_state_dict`, `iter`, `obs_rms_state`, `reward_rms_state`, `actor_state_dict`

---

## 十四、开发规范

### 核心原则

1. **Contract First**: NpEnv 契约不可破坏，backed 实现必须继承 `SimBackend`
2. **Fix at Owner Layer**: `scripts/` 只组装，`src/` 承载业务逻辑
3. **Backend Isolation**: MuJoCo/Motrix 差异留在 backend 适配层
4. **Keyframe 规则**: `<keyframe>` 放 task fragment, **禁止**放 robot.xml
5. **热路径禁解析**: `step/reset` 等热路径不解析 asset/XML 元数据
6. **Sim2Sim 契约**: 跨后端策略 I/O 字段必须一致

### AI 开发日志

每次代码/超参/架构修改后，AI 必须写入 `_devlog/`:
```
_devlog/<task>/<algo>/<YYYY-MM-DD>/<序号>_<描述>.md
```

### 代码质量

```bash
make format       # ruff format
make type         # mypy
make test         # pytest
make test-all     # format + type + test
```

### 详见

- 架构标准: `docs/sphinx/source/zh_CN/4-developer_guide/0-index.md`
- 协作流程: `docs/sphinx/source/zh_CN/4-developer_guide/5-contributing_workflow.md`
- 智能体规范: `AGENTS.md`
- 贡献指南: `CONTRIBUTING.md`
- 评估框架: `assess/README.md`
- 开发日志: `_devlog/README.md`
