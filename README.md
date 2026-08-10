# UniLab — Universal Lab for Robot Learning

> 高性能、模块化、多后端 RL 训练基础设施。支持双足/四足/人形/轮腿/灵巧手等多类机器人，
> 集成 PPO / SAC / TD3 / APPO / HIM-PPO / FlashSAC 等主流强化学习算法，
> 提供 MuJoCo / Motrix 双物理后端。
>
> 更新：2026-08-10

---

## 一、项目架构

```
wheel_legged_RL_unilab/
│
├── src/unilab/                   # ★ 核心库
│   ├── algos/                    # RL 算法实现 (torch + mlx)
│   │   ├── torch/                #   PPO(rsl-rl), CPO, NP3O, APPO, SAC, TD3, FlashSAC, HORA, HIM-PPO
│   │   ├── mlx/                  #   MLX PPO (Apple Silicon)
│   │   └── common/               #   网络骨架 (MLP, LayerNormMLP, RunningMeanStd)
│   │
│   ├── envs/                     # 环境实现
│   │   ├── locomotion/xqrobotwl/ #   ★ 轮腿机器人 xqrobotwl: 8 大任务环境 (每任务一文件)
│   │   ├── locomotion/xqrobotV2/ #   轮腿机器人 xqrobotV2 (旧版)
│   │   └── common/               #   共享函数 (rewards, commands, terrain_spawn, height_scan)
│   │
│   ├── terrains/                 # 地形生成系统
│   │   ├── terrain_generator.py  #   主生成器 (网格 + 子地形)
│   │   ├── heightfield_terrains.py # 7 种子地形: 平地/台阶/楼梯/斜坡/波浪/粗糙
│   │   └── config.py             #   预设配置
│   │
│   ├── assets/                   # 机器人模型 + 场景 XML (xqrobotwl / xqrobotV2)
│   ├── base/                     # 核心抽象 (np_env / backend / registry / scene / curriculum / observations)
│   ├── dr/                       # 域随机化 (init/reset/interval 三阶段)
│   ├── training/                 # 训练基础设施 (experiment / run / rsl_rl / sim2sim / reward)
│   ├── ipc/                      # 进程间通信 (异步训练)
│   ├── visualization/            # 可视化 & 回放
│   ├── tools/                    # CLI 工具 (export, import, render, viz)
│   └── cli.py                    # 命令行入口 (train / eval / demo)
│
├── conf/<algo>/task/<task>/      # ★ Hydra 任务配置 (按 algo × 任务隔离)
│   ├── ppo/task/                 #   xqrobotwl_{walk_flat, walk_rough, jump×4, single_leg×3, toe_walk, backflip}
│   │                             #   + xqrobotV2×4
│   ├── np3o/task/                #   xqrobotwl_stairs, xqrobotV2_stairs
│   └── cpo/task/                 #   xqrobotwl_fall_recovery_flat
│
├── tools/xqrobotwl/              # ★ 任务脚本 (eval_* / render_* / *_feasibility / warmstart_* / dump_pose_data / infer_pose_from_csv)
├── tools/                        # 全仓库工具 (analyze_offpolicy / audit_sim2sim / generate_support_matrix / … + email / mujoco / pinocchio_traj / xqrobotV2)
├── scripts/training/             # 训练入口 (train_rsl_rl / train_cpo / train_np3o / train_appo / …)
├── scripts/play/                 # 交互/回放入口 (play_interactive 施力回灌)
├── shell/xqrobotwl/<task>/       # ★ 启动/评估脚本 (每任务 train_<algo>_<task>.sh + eval_<algo>_<task>.sh)
├── logs/                         # 训练产物 (git 忽略) — run 目录 + pose_data CSV
├── video/<task>/                 # 结果视频 (8 任务目录)
├── _devlog/<robot>/<task>/<algo>/<date>/   # 开发日志 (AI 自记录)
├── docs/                         # 文档 (references / timeline / sphinx / project_tree)
├── thesis/                       # 论文开发指导中心 (框架图 / 专家文档 / 调度 / 整合)
├── backup/<Robot>/<task>_v<N>/   # 版本备份 (开箱即跑)
├── tests/  benchmark/  picture/  Sim2real/   # 测试 / 平台 / 素材
├── CLAUDE.md                     # AI 智能体开发规范 (企业级闭环)
└── pyproject.toml                # 项目配置 (uv 包管理)
```

---

## 二、支持的机器人

### 移动机器人 (Locomotion)

| 机器人 | 目录 | 类型 | 任务 |
|--------|------|------|------|
| **XqRobotWL** | `xqrobotwl/` | 轮腿双足 | **八大任务** (见下): 平地行走 / 点足行走 / 粗糙地形 / 跳跃 / 后空翻 / 单腿平衡 / 跌倒恢复 / 抬腿上台阶 |
| **XqRobotV2** | `xqrobotV2/` | 轮腿双足 | walk_flat / walk_rough / jump / toe_walk / stairs (旧版, 已由 XqRobotWL 取代) |

### XqRobotWL 八大任务 (CLAUDE.md §7)

每个任务独立: env 文件 + conf 目录 + shell 启动/评估脚本 + devlog 目录 + video 目录 + 任务脚本,
**互不共享可变状态**, 支持并行开发 / 训练 / 评估 (详见 CLAUDE.md §3)。

| # | 任务 | env | conf | shell | devlog | video |
|---|------|-----|------|-------|--------|-------|
| 1 | 平地滚动行走 | `joystick.py` | `conf/ppo/task/xqrobotwl_walk_flat/` | `shell/xqrobotwl/flat/` | `_devlog/xqrobotwl/walk_flat/ppo/` | `video/walk/` |
| 2 | 点足平地行走 | `toe_walk.py` | `conf/ppo/task/xqrobotwl_toe_walk_flat/` | `shell/xqrobotwl/toe_walk/` | `_devlog/xqrobotwl/toe_walk/ppo/` | `video/toe_walk/` |
| 3 | 不平坦地形行走 | `rough.py` | `conf/ppo/task/xqrobotwl_walk_rough/` | `shell/xqrobotwl/rough/` | `_devlog/xqrobotwl/{walk_rough,rough}/ppo/` | `video/rough/` |
| 4 | 平地跳跃 | `jump*.py` (5 变体) | `conf/ppo/task/xqrobotwl_jump*_flat/` | `shell/xqrobotwl/jump/` | `_devlog/xqrobotwl/jump/ppo/` | `video/jump/` |
| 5 | 平地后空翻 | `backflip.py` | `conf/ppo/task/xqrobotwl_backflip_flat/` | `shell/xqrobotwl/backflip/` | `_devlog/xqrobotwl/backflip/ppo/` | `video/backflip/` |
| 6 | 单腿平衡 (三态) | `single_leg*.py` (3) | `conf/ppo/task/xqrobotwl_single_leg*/` | `shell/xqrobotwl/single_leg/` | `_devlog/xqrobotwl/single_leg/ppo/` | `video/single_leg/` |
| 7 | 跌倒恢复 | `fall_recovery.py` | `conf/cpo/task/xqrobotwl_fall_recovery_flat/` | `shell/xqrobotwl/fall_recovery/` | `_devlog/xqrobotwl/fall_recovery/ppo/` | `video/fall_recovery/` |
| 8 | 抬腿上台阶 | `stairs.py` | `conf/np3o/task/xqrobotwl_stairs/` | `shell/xqrobotwl/stairs/` | `_devlog/xqrobotwl/stairs/np3o/` | `video/stairs/` |

> **并行 agent 开发**: 8 个 agent 可各自认领一个任务 (独立 env/conf/shell/devlog/video), 并行训练互不干扰
> (每个 run 独立时间戳目录 `logs/rsl_rl_<algo>/<Task>/<timestamp>/`)。共享只读基座
> `joystick.py` / `base.py` / 机器人 XML 不得被单个 agent 独占修改 (CLAUDE.md §3.2)。

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
契约审计: `tools/audit_sim2sim_contracts.py`

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

## 十、评估与数据闭环 (CLAUDE.md §1.3/§1.5/§2.4)

按任务独立评估 (无辅助确定性 rollout)。**数据优先**: 姿态数据 CSV 才是评估依据,
视频/图只是给负责人看的展示。

### 工具链

| 阶段 | 工具 | 说明 |
|------|------|------|
| 确定性评估 | `tools/xqrobotwl/eval_*.py` | 每任务评估脚本 (eval_fall_recovery / eval_single_leg_move / eval_single_leg_unicycle …) |
| 姿态数据导出 | `tools/xqrobotwl/dump_pose_data.py` | 每步姿态 CSV (26 列, 数值保留两位小数) → `logs/pose_data/` |
| 姿态反推统计 | `tools/xqrobotwl/infer_pose_from_csv.py` | 按 §1.3 反推表逐行判姿态 + 各姿态时长/占比 |
| 渲染视频 | `tools/xqrobotwl/render_*.py` | 相机跟踪 (机器人始终在视角内) → `video/<task>/` |

### 常用命令

```bash
# 确定性评估 (跌倒恢复: --pose 0-3 逐姿态, 每姿态 ≥20 ep)
uv run mjpython tools/xqrobotwl/eval_fall_recovery.py \
    --run <run_dir> --ckpt model_4000.pt --num_envs 20

# 导出每步姿态数据 CSV (两位小数)
uv run mjpython tools/xqrobotwl/dump_pose_data.py \
    --run <run_dir> --ckpt model_4000.pt --pose 0

# 从 CSV 反推姿态 + 统计时长/占比
uv run tools/xqrobotwl/infer_pose_from_csv.py logs/pose_data/xxx.csv

# 渲染视频 (相机跟踪, 不出视角)
uv run mjpython tools/xqrobotwl/render_recovery_video.py \
    --run <run_dir> --ckpt model_4000.pt --pose 0
```

### 达标指标 (详见 CLAUDE.md 附录 A + §7.0)

恢复率 ≥80% · 站立高度 ≈0.52m · 站立 |gyro| <1 rad/s · 轮子离地率 0%
**长时评估**: 站立保持 ≥10s / 行走 ≥30s / 动作类 ≥10 次 / 跌倒恢复每姿态 ≥20 episodes

---

## 十一、XqRobotV2 专项

> 旧版机器人, 已由 XqRobotWL (八大任务) 取代; 以下技术参数保留供参考。

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

### 训练 (XqRobotWL 八大任务)

每个任务有独立启动脚本 `shell/xqrobotwl/<task>/train_<algo>_<task>.sh` (如 `flat/`→平地滚动行走):

```bash
# 平地滚动行走 (GPU 0)
CUDA_VISIBLE_DEVICES=0 bash shell/xqrobotwl/flat/train_ppo_flat.sh

# 粗糙地形行走 (GPU 1)
CUDA_VISIBLE_DEVICES=1 bash shell/xqrobotwl/rough/train_ppo_rough.sh

# 8 个任务可并行训练 (不同 GPU / 不同 run 目录, 互不干扰)
CUDA_VISIBLE_DEVICES=2 bash shell/xqrobotwl/fall_recovery/train_ppo_fall_recovery.sh
```

训练日志: `logs/rsl_rl_<algo>/<Task>/<timestamp>/model_<iter>.pt` (每个 run 独立时间戳目录)。

### 通用训练命令

```bash
# PPO
uv run train --algo ppo --task xqrobotwl_walk_flat --sim mujoco

# CPO (约束策略优化, 跌倒恢复)
uv run train --algo cpo --task xqrobotwl_fall_recovery_flat --sim mujoco

# NP3O (台阶)
uv run train --algo np3o --task xqrobotwl_stairs --sim mujoco
```

### 评估

```bash
# 键盘控制 (交互回放)
bash shell/xqrobotwl/flat/eval_ppo_flat.sh --keyboard

# 确定性评估 (无辅助)
uv run mjpython tools/xqrobotwl/eval_fall_recovery.py \
    --run <run_dir> --ckpt model_4000.pt --pose 0

# 姿态数据 → 反推统计
uv run tools/xqrobotwl/infer_pose_from_csv.py logs/pose_data/xxx.csv
```

### TensorBoard

```bash
uv run tensorboard --logdir logs/rsl_rl_ppo/XqRobotWLWalkFlat
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
- 智能体规范: `CLAUDE.md`
- 评估与数据闭环: 见本 README 第十章 (tools/xqrobotwl 工具链)
- 项目结构: `docs/project_tree.md`
- 开发进展时间线: `docs/timeline/`
- 开发日志: `_devlog/README.md`
