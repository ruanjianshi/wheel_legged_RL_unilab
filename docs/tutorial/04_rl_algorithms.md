# 04. RL 算法与训练

## 目录

- [Hydra 配置体系](#hydra-配置体系)
- [PPO 训练配置详解](#ppo-训练配置详解)
- [启动训练](#启动训练)
- [训练流水线内部](#训练流水线内部)
- [其他算法 (SAC/TD3/APPO)](#其他算法-sactd3appo)
- [Checkpoint 与恢复](#checkpoint-与恢复)

---

## Hydra 配置体系

```
conf/
├── ppo/
│   ├── config.yaml              # ★ PPO 默认超参 (所有任务共用)
│   └── task/
│       └── xqrobotV2_walk_flat/
│           └── mujoco.yaml      # ★ 任务专属配置 (覆盖默认值)
├── appo/
│   ├── config.yaml
│   └── task/
├── offpolicy/                   # SAC / TD3 / FlashSAC
│   ├── config.yaml
│   ├── algo/sac.yaml            # SAC 算法默认值
│   └── task/
└── ppo_him/                     # HIM-PPO
```

### Hydra 继承覆盖

训练启动时，Hydra 从下到上合并配置：
```
CLI 参数 (优先级最高)
  ↓ 覆盖
task/mujoco.yaml (任务专属)
  ↓ 覆盖
config.yaml (算法默认)
  ↓ 覆盖
Hydra 内置默认值
```

示例：
```bash
uv run train --algo ppo \
    --task xqrobotV2_walk_flat \
    --sim mujoco \
    algo.algorithm.entropy_coef=0.01  # CLI 覆盖
```

---

## PPO 训练配置详解

### `conf/ppo/config.yaml` — 全局默认 (省略辅助字段)

```yaml
algo:
  algo: ppo
  algo_log_name: rsl_rl_ppo
  seed: 1
  num_envs: 4096              # ★ 被任务配置覆盖为 1024
  num_steps_per_env: 24       # ★ 被覆盖为 25
  max_iterations: 101          # ★ 被覆盖为 5000
  save_interval: 100
  empirical_normalization: false

  obs_groups:
    default:
      - policy                 # 哪些观测组送给策略

  policy:
    init_noise_std: 1.0        # ★ 被覆盖为 0.3
    actor_hidden_dims: [512, 256, 128]   # ★ 被覆盖为 [512,512,256,128]
    critic_hidden_dims: [512, 256, 128]  # ★ 同上
    activation: elu
    class_name: ActorCritic

  algorithm:
    class_name: unilab.algos.torch.rsl_rl_ppo:FinalObservationAwarePPO
    value_loss_coef: 1.0
    use_clipped_value_loss: true
    clip_param: 0.2
    entropy_coef: 0.01         # ★ 被覆盖为 0.002
    num_learning_epochs: 5           # PPO 每批更新 5 轮
    num_mini_batches: 4              # 分 4 个小批
    learning_rate: 1.0e-3      # ★ 被覆盖为 1e-4
    schedule: adaptive               # 自适应学习率
    gamma: 0.99                # ★ 保持不变
    lam: 0.95                  # ★ 保持不变
    desired_kl: 0.01           # ★ 被覆盖为 0.005
    max_grad_norm: 1.0
```

### `conf/ppo/task/xqrobotV2_walk_flat/mujoco.yaml` — 任务专属

```yaml
training:
  task_name: XqRobotV2WalkFlat    # ★ 注册表查表键
  sim_backend: mujoco

algo:
  num_envs: 1024                  # 1024 并行环境
  num_steps_per_env: 25           # 每环境 25 步 = 25600 样本/批
  max_iterations: 5000            # 训练 5000 轮
  empirical_normalization: false  # 不用观测归一化
  obs_groups:
    default:
      - actor                     # 策略只看 actor 观测
  policy:
    activation: elu
    actor_hidden_dims: [512, 512, 256, 128]    # 4 层 MLP
    critic_hidden_dims: [512, 512, 256, 128]
    init_noise_std: 0.3           # 初始探索噪声
  algorithm:
    learning_rate: 1.0e-4         # 缓慢学习
    entropy_coef: 0.002           # 低 entropy → 策略精确
    desired_kl: 0.005             # 严格控制策略更新幅度

env:
  control_config:
    action_scale: 0.5             # 腿动作缩放
    wheel_action_scale: 10.0      # 轮动作缩放
    clip_actions: 100.0           # 几乎不裁剪
  commands:
    vel_limit:
      - [-0.6, -0.3, -1.0, -0.1, 0.45]  # [vx, vy, vyaw, tsk, height] 下限
      - [ 0.6,  0.3,  1.0,  0.1, 0.85]  # 上限
    resampling_time: 3.0          # 命令每 3 秒重采样
  curriculum:
    enabled: true
  domain_rand:
    randomize_base_mass: false    # ★ 初版关闭 DR，收敛后再开
    randomize_ground_friction: false
    randomize_kp: false
    randomize_kd: false
    random_com: false
    randomize_leg_length: false

reward:
  only_positive_rewards: false
  scales:
    tracking_lin_vel: 1.5         # 最重要: 速度跟踪
    tracking_ang_vel: 1.5
    orientation: -10.0            # 最重要: 保持直立
    base_height: -5.0
    similar_calf: -1.0            # 腿对称
    hip_roll: -2.0                # 髋关节内收
    wheel_symmetry: -0.5          # 轮子对称
    tsk: -2.0                     # 髋差动跟踪
    feet_distance: -1.0           # 轮距
    alive: 1.0                    # 存活奖励
    lin_vel_z: -0.2               # 垂直速度惩罚
    ang_vel_xy: -0.02             # 横滚/俯仰惩罚
    joint_action_rate: -0.1       # 腿动作平滑
    wheel_action_rate: -0.005     # 轮动作平滑
  tracking_sigma: 0.3       # exp(-err²/0.3) 的 σ
  base_height_target: 0.65  # 目标高度
  max_tilt_deg: 60.0        # 超过则终止
  min_base_height: 0.20     # 低于则终止
```

### 关键超参速查

| 参数 | 值 | 说明 |
|------|-----|------|
| `num_envs` | 1024 | 并行环境数 |
| `num_steps_per_env` | 25 | 每次更新采集步数 |
| `batch_size` | 1024 × 25 = 25600 | 每次更新的样本数 |
| `mini_batch_size` | 25600 / 4 = 6400 | 每个小批的样本数 |
| `max_iterations` | 5000 | 训练总轮数 |
| `total_steps` | 1024 × 25 × 5000 = 1.28 亿 | 总交互步数 |
| `lr` | 1e-4 | 学习率 |
| `entropy_coef` | 0.002 | 探索系数 |
| `desired_kl` | 0.005 | 目标 KL 散度 |
| `gamma` | 0.99 | 折扣因子 |
| `lam` | 0.95 | GAE λ |
| `hidden_dims` | [512, 512, 256, 128] | 网络结构 |

---

## 启动训练

### 基础命令

```bash
# 使用 uv run + Hydra CLI
uv run train --algo ppo --task xqrobotV2_walk_flat --sim mujoco

# GPU 指定 + 后台运行
setsid bash -c 'CUDA_VISIBLE_DEVICES=0 uv run train \
    --algo ppo --task xqrobotV2_walk_flat --sim mujoco' \
    &>/tmp/flat_train.log & disown

# 或使用便利脚本
CUDA_VISIBLE_DEVICES=0 bash shell/train_ppo_flat.sh
```

### 运行时覆盖超参

```bash
# 覆盖任意 Hydra 路径
uv run train --algo ppo --task xqrobotV2_walk_flat --sim mujoco \
    algo.algorithm.entropy_coef=0.005 \
    algo.algorithm.learning_rate=5e-5

# 覆盖命令范围
uv run train ... \
    'env.commands.vel_limit=[[-1,-0.5,-1.5,-0.1,0.4],[1,0.5,1.5,0.1,0.9]]'
```

---

## 训练流水线内部

### 完整生命周期

```python
# scripts/train_rsl_rl.py (简化版)

# 1. Hydra 组合配置
cfg = compose("config.yaml", "task/xqrobotV2_walk_flat/mujoco.yaml")

# 2. 创建实验追踪器 (wandb / tensorboard)
tracker = ExperimentTracker(cfg)

# 3. 写入 run_config.json + contract_snapshot
tracker.start()

# 4. 创建环境
env = registry.make("XqRobotV2WalkFlat", "mujoco", num_envs=1024)

# 5. 域随机化管理器
dr_manager = DomainRandomizationManager(cfg, env)

# 6. 创建 PPO Runner
runner = OnPolicyRunner(
    env=env,
    train_cfg=cfg.algo,
    log_dir=tracker.log_dir,
    device="cuda",
)

# 7. 训练循环
for iteration in range(5000):
    # 7a. 采集 rollout (25 步 × 1024 envs)
    runner.observe(25)

    # 7b. PPO 更新 (5 epochs, 4 mini-batches)
    runner.update()

    # 7c. 日志 & checkpoint
    if iteration % 100 == 0:
        runner.save(f"model_{iteration}.pt")
```

### PPO 更新内部

```
For each iteration:
  collect 1024 envs × 25 steps = 25600 transitions
  compute GAE advantages (γ=0.99, λ=0.95)
  For epoch in 1..5:
    For minibatch in shuffled 4-way split:
      compute policy_loss = -ratio * advantage + clip_loss
      compute value_loss = (value - return)²
      total_loss = policy_loss + 1.0 * value_loss - 0.002 * entropy
      Adam step (lr=1e-4, max_grad_norm=1.0)
    compute KL_divergence(old_pi || new_pi)
    if KL > 0.005: scale down lr for next epoch
    if KL < 0.005: scale up lr for next epoch
```

### FinalObservationAwarePPO

当 episode 因超时 `truncated`（非 `terminated`）而结束，标准 PPO 会用 `value=0` 作为未来回报的估计。但 `FinalObservationAwarePPO` 用 **truncated 时的最后一帧观测重新计算价值**，得到更准确的 bootstrap 值：

```python
# 伪代码
if truncated and not terminated:
    final_value = critic(final_obs)  # 用截断时观测重新估值
    returns[-1] = reward[-1] + γ * final_value
else:
    returns[-1] = reward[-1]
```

---

## 其他算法 (SAC/TD3/APPO)

### SAC 启动

```bash
uv run train --algo sac --task xqrobotV2_walk_flat --sim mujoco
```

配置: `conf/offpolicy/config.yaml` + `conf/offpolicy/algo/sac.yaml` + `conf/offpolicy/task/sac/xqrobotV2_walk_flat/mujoco.yaml`

关键 SAC 默认值:
```yaml
algo:
  num_envs: 4096
  updates_per_step: 4
  actor_lr: 3e-4
  critic_lr: 3e-4
  gamma: 0.97
  tau: 0.125          # 软更新系数
  actor_hidden_dim: 512
  critic_hidden_dim: 768
  num_atoms: 101       # 分布式 critic (C51)
  obs_normalization: true
  use_layer_norm: true
```

### TD3 启动

```bash
uv run train --algo td3 --task xqrobotV2_walk_flat --sim mujoco
```

### APPO (异步 PPO)

```bash
uv run train --algo appo --task xqrobotV2_walk_flat --sim mujoco
```

APPO 将 learner 和 collector 分离为独立进程，支持 V-trace 偏差修正。

---

## Checkpoint 与恢复

### 目录结构

```
logs/rsl_rl_ppo/XqRobotV2WalkFlat/
└── 2026-06-30_22-49-32_mujoco/   # Run timestamp
    ├── run_config.json            # 完整 Hydra 配置快照
    ├── model_100.pt               # iter=100 checkpoint
    ├── model_5000.pt              # iter=5000 checkpoint
    └── ...
```

### Checkpoint 内容

```python
ckpt = {
    "model_state_dict": {...},       # ActorCritic 权重
    "optimizer_state_dict": {...},   # Adam 状态
    "iter": 5000,                    # 当前迭代数
    "obs_rms": {...},                # 观测归一化统计量
    "actor_state_dict": {...},       # 仅 actor 权重 (用于评估)
}
```

### 恢复训练

```bash
uv run train --algo ppo --task xqrobotV2_walk_flat --sim mujoco \
    training.load_run=2026-06-30_22-49-32_mujoco \
    training.checkpoint=model_2500.pt
```

### 评估已训练模型

```bash
# 键盘操控验证
bash shell/eval_ppo_flat.sh --keyboard --run 2026-06-30_22-49-32_mujoco --ckpt 5000

# 自动化评估
uv run assess/runner.py -t flat_walk -a ppo \
    -r 2026-06-30_22-49-32_mujoco -c 5000 -s full
```

---

## 关键要点

1. **配置覆盖顺序**: CLI > task YAML > config YAML
2. **批量大小**: `num_envs × num_steps_per_env = 25600` 样本/迭代
3. **entropy_coef = 0.002**—低 entropy 意味着策略更确定（适合已收敛的任务）
4. **desired_kl = 0.005**—严格控制每次更新步长
5. **FinalObservationAwarePPO** 处理 episode 截断的 bootstrap 问题
6. **Checkpoint = 模型权重 + 优化器状态 + 归一化统计量**

---

> 上一章：[03. 机器人建模](./03_robot_modeling.md)
> 下一章：[05. URDF 机器人移植](./05_urdf_import.md)
