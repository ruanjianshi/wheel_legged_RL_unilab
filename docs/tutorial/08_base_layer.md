# src/unilab/base — 核心抽象层代码解读

> 文件路径: `src/unilab/base/`  
> 行数统计: ~1400 行 (不含 backend 子目录)  
> 更新: 2026-07-02

---

## 目录

1. [概览: 分层架构](#概览-分层架构)
2. [base.py — ABEnv 抽象 + EnvCfg 配置基类](#basepy--abenv-抽象--envcfg-配置基类)
3. [np_env.py — NpEnv 向量化环境核心](#np_envpy--npenv-向量化环境核心)
4. [registry.py — 环境注册与工厂](#registrypy--环境注册与工厂)
5. [scene.py — 场景配置](#scenepy--场景配置)
6. [curriculum.py — 课程学习](#curriculumpy--课程学习)
7. [final_observation.py — 截断观测 Bootstrap](#final_observationpy--截断观测-bootstrap)
8. [observations.py — 观测工具函数](#observationspy--观测工具函数)
9. [augmentation.py — 对称增强](#augmentationpy--对称增强)
10. [backend/ — 仿真后端抽象](#backend--仿真后端抽象)

---

## 概览: 分层架构

```
src/unilab/base/
├── base.py               # ABEnv 抽象基类 + EnvCfg 配置基类 (229 行)
├── np_env.py             # NpEnv: 向量化 numpy 环境实现 (471 行)
├── registry.py            # @envcfg / @env 注册系统 + make() 工厂 (324 行)
├── scene.py               # SceneCfg / TerrainSceneCfg (29 行)
├── curriculum.py          # 惩罚课程 + episode 长度追踪 (90 行)
├── final_observation.py   # 截断/终止观测处理 (160 行)
├── observations.py        # obs 扁平化/分割工具 (37 行)
├── augmentation.py        # 对称增强 (G1 专用)
└── backend/
    ├── base.py            # SimBackend 抽象接口 (~700 行)
    ├── mujoco/            # MuJoCo 后端实现
    └── motrix/            # Motrix GPU 后端实现
```

| 文件 | 职责 | 依赖关系 |
|------|------|----------|
| `base.py` | 定义 ABEnv/EnvCfg 契约 | 被所有 env 继承 |
| `np_env.py` | 实现向量化 step/reset 循环 | 继承 ABEnv, 持有 SimBackend |
| `registry.py` | 环境注册表，工厂创建 | 操作 ABEnv 子类 |
| `scene.py` | 声明场景源配置 | 被 EnvCfg 引用 |
| `curriculum.py` | 自适应惩罚缩放 | 操作 reward_config.scales |
| `final_observation.py` | PPO Bootstrap 补丁 | 在训练器层使用 |
| `observations.py` | obs 维度提取/拼接 | 被训练器/评估使用 |
| `backend/` | 物理引擎抽象 + 实现 | `np_env.py` 通过接口调用 |

---

## base.py — ABEnv 抽象 + EnvCfg 配置基类

**文件**: `src/unilab/base/base.py` (229行)

### EnvCfg — 环境配置基类 (L24-62)

```python
@dataclass
class EnvCfg:
    scene: SceneCfg | None = None          # 场景源配置
    sim_dt: float = 0.01                   # 物理仿真步长 (100Hz)
    max_episode_seconds: Optional[float] = None  # episode 最大秒数
    ctrl_dt: float = 0.01                  # 控制步长 (100Hz)
    render_spacing: float = 1.0            # 多 env 渲染间距
    render_offset_mode: str = "grid"       # 渲染布局模式
    motrix_max_iterations: Optional[int] = None
    post_step_forward_sensor: bool = False
```

**关键派生属性**:

```python
@property
def max_episode_steps(self) -> Optional[int]:
    """episode 最大步数 = 秒数 / 控制间隔"""
    return int(self.max_episode_seconds / self.ctrl_dt)

@property
def sim_substeps(self) -> int:
    """每控制步的仿真實步数 = ctrl_dt / sim_dt"""
    return int(round(self.ctrl_dt / self.sim_dt))
```

**核心概念**: `sim_dt` 和 `ctrl_dt` 的关系决定了**减采样比**。XqRobotV2 用 `sim_dt=0.005` + `ctrl_dt=0.01` → `sim_substeps=2` (每控制步仿真 2 步)。

### ABEnv — 环境抽象基类 (L64-229)

```python
class ABEnv(abc.ABC):
    @property
    @abc.abstractmethod
    def num_envs(self) -> int: ...
    @property
    @abc.abstractmethod
    def observation_space(self) -> gym.Space: ...
    @property
    @abc.abstractmethod
    def action_space(self) -> gym.Space: ...
    @property
    @abc.abstractmethod
    def obs_groups_spec(self) -> dict[str, int]: ...
    @abc.abstractmethod
    def step(self, actions: np.ndarray) -> Any: ...
```
**职责**: 定义所有环境必须实现的**最小契约**—`step()`, `obs_groups_spec`, `observation_space` 等。`NpEnv` 在此基础上增加向量化和 auto-reset 逻辑。

### EnvPlayCapabilities (L15-21)

```python
@dataclass(frozen=True)
class EnvPlayCapabilities:
    supports_native_interactive_renderer: bool = False
    supports_physics_state_playback: bool = False
    supports_native_video_capture: bool = False
```

供训练入口查询后端渲染能力，决定是否可以键盘操控/录像。

---

## np_env.py — NpEnv 向量化环境核心

**文件**: `src/unilab/base/np_env.py` (471行)  
**这是整个项目最重要的文件**—所有移动机器人环境都继承它。

### NpEnvState — 环境状态容器 (L25-35)

```python
@dataclass
class NpEnvState:
    obs: dict[str, np.ndarray]              # {"obs": (N,297), "critic": (N,324)}
    reward: np.ndarray                       # (N,) float
    terminated: np.ndarray                   # (N,) bool — 自然终止 (摔倒等)
    truncated: np.ndarray                    # (N,) bool — 超时截断
    info: dict[str, Any]                     # 任意元信息 (commands, steps, timing)
    final_observation: dict | None = None    # 终止时的观测快照 (用于 bootstrap)
```

**关键字段说明**:

- `terminated` vs `truncated`: Gymnasium 规范 — `terminated` = 环境逻辑终止 (摔倒); `truncated` = 达到时间上限。两者都触发 reset，但 PPO 的 bootstrap 处理不同。
- `info["steps"]`: `(N,) uint32` — 每环境独立的步数计数器，用于命令重采样和日志门控。
- `info["commands"]`: `(N, 5)` — 当前速度命令 `[vx, vy, vyaw, tsk, height]`
- `final_observation`: 当 env 终止时，保存终止前最后一帧观测。PPO 训练器用它来给 `truncated` 环境做值函数 bootstrap。

### NpEnv.__init__ — 初始化 (L41-52)

```python
class NpEnv(ABEnv):
    def __init__(self, cfg: EnvCfg, backend: SimBackend, num_envs: int):
        self._cfg = cfg
        self._backend: SimBackend = backend
        self._num_envs = num_envs
        self._state: Optional[NpEnvState] = None
        self._truncated_scratch = np.zeros((num_envs,), dtype=bool)
        self.step_counter = 0
        self._dr_manager: DomainRandomizationManager | None = None
        self._autoreset = True              # 默认自动 reset
```

### NpEnv.step() — 核心步进循环 (L104-177)

这是单步仿真的完整流程：

```python
def step(self, actions: np.ndarray) -> NpEnvState:
    # ① apply_action: 策略输出 → 物理控制量
    ctrl = self.apply_action(actions, self._state)
    #    XqRobotV2: action*scale + DEFAULT_ANGLES, 重排执行器顺序

    # ② 间隔域随机化 (if due)
    self._dr_manager.apply_interval_randomization_if_due(self.step_counter)

    # ③ backend.step: 执行仿真实步
    backend_result = self._backend.step(ctrl, self._cfg.sim_substeps)

    # ④ update_state: 计算观测/奖励/终止
    self._state = self.update_state(self._state)
    #    子类重写: 读传感器 → 算 obs → 算 reward → 算 terminated

    # ⑤ 步数 + 截断
    self._state.info["steps"] += 1
    self.step_counter += 1
    truncated = self._compute_truncated(self._state)
    np.logical_or(self._state.truncated, truncated, out=self._state.truncated)

    # ⑥ auto-reset: 终止环境自动重开
    done = self._state.terminated | self._state.truncated
    if self._autoreset and np.any(done):
        self._reset_done_envs()           # ★ 保存 final_observation 到 info

    # ⑦ NaN 检查 → 归零 → 返回
    np.nan_to_num(self._state.reward, copy=False, nan=0.0)
    return self._state
```

**时序图**:
```
Policy → actions (N,8)
  → apply_action → ctrl (N,8) [1]
  → backend.step(ctrl, substeps) [2]
  → update_state → obs, reward, terminated [3]
  → _compute_truncated [4]
  → _reset_done_envs [5] (if any done)
  → return NpEnvState
```

### NpEnv._reset_done_envs() — 自动 Reset (L179-214)

```python
def _reset_done_envs(self) -> None:
    done = self._state.terminated | self._state.truncated
    env_indices = np.flatnonzero(done)
    self._state.info["steps"][env_indices] = 0

    # ★ 保存 final_observation (给 PPO bootstrap 用)
    final_obs = {...}
    for key in self._state.obs:
        final_obs[key][env_indices] = self._state.obs[key][env_indices]
    self._state.final_observation = final_obs

    # 调用 domain randomization 的 reset
    new_obs, info1 = self.reset(env_indices)  # → dr_manager.reset()

    # 覆盖 done envs 的 obs
    for key in self._state.obs:
        self._state.obs[key][env_indices] = new_obs[key]
```

**关键设计**: `final_observation` 保存的是**终止时刻**的观测快照，而不是 reset 后的。这让 PPO 可以用终止帧再做一次价值估计来 bootstrap。

### 子类必须实现的抽象方法

```python
@abc.abstractmethod
def apply_action(self, actions, state) -> np.ndarray:
    """子类实现动作→控制量转换"""
@abc.abstractmethod
def update_state(self, state) -> NpEnvState:
    """子类计算观测/奖励/终止"""
```

这两个方法是环境开发的**核心接口**—所有行为差异都在这两个方法中。

### 播放/渲染支持 (L310-452)

- `init_play_renderer()` — 初始化后端渲染器
- `run_playback()` — 回放录制
- `capture_play_video_frame()` — 视频帧采集
- `get_physics_state_snapshot()` — 物理状态快照

---

## registry.py — 环境注册与工厂

**文件**: `src/unilab/base/registry.py` (324行)

### 注册系统

```python
_envs: Dict[str, EnvMeta] = {}          # 全局注册表

@dataclass
class EnvMeta:
    env_cfg_cls: Type[EnvCfg]            # 配置类
    env_cls_dict: Dict[str, Type[ABEnv]]  # backend → env 类映射
```

### 注册流程

```python
# Step 1: 注册配置 (环境定义时)
@envcfg("XqRobotV2WalkFlat")
@dataclass
class XqRobotV2WalkFlatCfg(XqRobotBaseCfg):
    scene: SceneCfg = ...
# → register_env_config("XqRobotV2WalkFlat", XqRobotV2WalkFlatCfg)

# Step 2: 注册环境类 (每个 backend 单独注册)
@env("XqRobotV2WalkFlat", sim_backend="mujoco")
class XqRobotV2WalkFlatEnv(XqRobotV2WalkFlatMotrixEnv):
    pass
# → register_env("XqRobotV2WalkFlat", XqRobotV2WalkFlatEnv, "mujoco")
```

### make() 工厂 (L200-245)

```python
def make(name, sim_backend=None, env_cfg_override=None, num_envs=1) -> ABEnv:
    meta = _envs[name]                           # 查表
    env_cfg = meta.env_cfg_cls()                 # 实例化配置
    if env_cfg_override:
        apply_cfg_overrides(env_cfg, env_cfg_override)  # 应用覆写
    sim_backend = sim_backend or meta.available_sim_backend()  # 回退到第一个可用
    env_cls = meta.env_cls_dict[sim_backend]     # 取后端专属类
    return env_cls(env_cfg, num_envs=num_envs, backend_type=sim_backend)
```

### apply_cfg_overrides() — 递归配置覆写 (L163-197)

```python
def apply_cfg_overrides(target_obj, overrides):
    for key, value in overrides.items():
        existing = getattr(target_obj, key)
        if isinstance(value, dict):
            if dataclasses.is_dataclass(existing):
                apply_cfg_overrides(existing, value)  # ★ 递归 deep merge
                continue
        setattr(target_obj, key, value)
```

支持嵌套字典覆盖，如 `env.scene.terrain.generator.num_rows=4` → 只改 `num_rows`，保留 `sub_terrains` 等字段。

### ensure_registries() — 包自动导入 (L259-324)

```python
_DEFAULT_REGISTRY_PACKAGES = (
    "unilab.envs.locomotion",
)

def ensure_registries(packages=None):
    for package in packages:
        pkg = importlib.import_module(package)
        modules = pkg.__unilab_registry_modules__  # 每个包声明的模块列表
        for module_name in modules:
            importlib.import_module(module_name)    # 导入即触发 @envcfg/@env
```

**启动时自动执行**: 训练脚本导入 `ensure_registries()` → 导入所有注册模块 → 装饰器自动填入 `_envs` 字典。

---

## scene.py — 场景配置

**文件**: `src/unilab/base/scene.py` (29行)

```python
@dataclass
class TerrainSceneCfg:
    generator: TerrainGeneratorCfg | None = None   # 非空 → 生成 heightfield 地形
    hfield_name: str = "terrain_hfield"            # MuJoCo heightfield 资源名
    geom_name: str | None = None                   # 碰撞几何体名

@dataclass
class SceneCfg:
    model_file: str                                # ★ 必填: 机器人 XML 路径
    fragment_files: list[str] = field(default_factory=list)  # Keyframe/灯光 XML
    terrain: TerrainSceneCfg | None = None         # None = 平地, 非 None = 生成地形
    visual_model_file: str | None = None           # 可选高精渲染模型
```

**核心逻辑**:
- `terrain.generator is None` → 后端生成平面 ground
- `terrain.generator is not None` → 后端调用 `TerrainGenerator.generate()` → 写 PNG → 挂 hfield geom
- `fragment_files` → 在模型加载前合并到主 XML（用于注入 Keyframe）

---

## curriculum.py — 课程学习

**文件**: `src/unilab/base/curriculum.py` (90行)

### EpisodeLengthTracker (L10-24)

```python
class EpisodeLengthTracker:
    def __init__(self, num_envs, window_size=1000):
        self.window_size = max(1, int(window_size * num_envs / 4096))  # 按 env 数缩放
    def update(self, episode_lengths):
        avg = np.mean(episode_lengths)
        weight = min(len(episode_lengths) / self.window_size, 1.0)
        self.average_length = self.average_length * (1-weight) + avg * weight  # EMA
```

### PenaltyCurriculum (L27-90)

```python
class PenaltyCurriculum:
    def __init__(self, env, enabled=True, initial_scale=0.5,
                 min_scale=0.5, max_scale=1.0,
                 level_down_threshold=150.0, level_up_threshold=750.0, degree=0.001):
        # 1. 识别所有负权重奖励 → 标记为 penalty
        # 2. 初始时 penalty scale = 0.5 (惩罚减半)
        # 3. 随着 episode 长度增长，逐步恢复惩罚到 1.0
```

**工作原理**:
```
episode_length < 150  → 降低惩罚 (让机器人多探索，不容易死)
episode_length > 750  → 增加惩罚 (机器人学会了，要求更严格)
每个 update 步进 degree=0.001 (缓慢变化)
```

---

## final_observation.py — 截断观测 Bootstrap

**文件**: `src/unilab/base/final_observation.py` (160行)

### 核心问题

PPO 需要 `next_value` 来计算 GAE advantage。当 episode 因为 `truncated` 结束时，如果直接用 reset 后的 `obs` 算 `next_value`，会得到错误的值（因为环境已经重置了）。

### 解决方案

`FinalObservationAwarePPO` 在更新前调用 `patch_transition_next_obs()`:

```python
def patch_transition_next_obs(next_obs, final_observation, done, info):
    terminal_mask = resolve_terminal_mask(done, info)  # 哪些 env 终止了
    transition_next_obs = next_obs.copy()
    # ★ 用终止时的观测替换 reset 后的观测
    transition_next_obs[terminal_mask] = final_observation["obs"][terminal_mask]
    return transition_next_obs, ...
```

**效果**: PPO 用终止瞬时的观测来计算 value bootstrap，得到准确的 `V(s_terminal)`。

### 两个 Contract 类

```python
@dataclass(frozen=True)
class TransitionBootstrapContract:
    actor_next_obs       # 给 actor 用的观测 (不修改)
    transition_next_obs  # 给 value bootstrap 用的观测 (终止 env 已替换)
    terminal_mask        # 哪些 env 终止了
    timeout_terminal_mask# 哪些 env 是 truncated (区别于真实 terminated)

@dataclass(frozen=True)
class TerminalObservationContract:
    terminal_obs         # 终止时的观测
    terminal_mask        # 终止 mask
    timeout_terminal_mask# truncated mask
```

---

## observations.py — 观测工具函数

**文件**: `src/unilab/base/observations.py` (37行)

```python
def flatten_obs_dict(obs: dict) -> np.ndarray:
    return np.concatenate(list(obs.values()), axis=1)

def split_obs_dict(obs: dict) -> tuple[np.ndarray, np.ndarray]:
    actor = obs["obs"]
    return actor, obs.get("critic", actor)  # 无 critic 时回退到 actor

def get_obs_dims(obs_groups_spec: dict) -> tuple[int, int]:
    obs_dim = obs_groups_spec.get("obs", 0)
    return obs_dim, obs_groups_spec.get("critic", obs_dim)
```

纯函数工具，被训练器和评估框架使用。

---

## augmentation.py — 对称增强

G1 人形机器人专用，利用肢体对称性做数据增强，XqRobotV2 不使用。

---

## backend/ — 仿真后端抽象

**文件**: `src/unilab/base/backend/base.py` (~700行)

### SimBackend 接口

```python
class SimBackend(abc.ABC):
    # ── 模型查询 ──
    def get_actuator_ctrl_range(self) -> np.ndarray: ...
    def get_keyframe_qpos(self) -> np.ndarray: ...
    def get_joint_range(self) -> np.ndarray: ...

    # ── 仿真核 ──
    def set_state(self, env_indices, qpos, qvel, randomization=None) -> None: ...
    def step(self, ctrl: np.ndarray, nsteps: int = 1) -> dict | None: ...
    def materialize(self) -> None: ...

    # ── 传感器 (零拷贝视图) ──
    def get_base_pos(self) -> np.ndarray: ...
    def get_base_quat(self) -> np.ndarray: ...
    def get_base_lin_vel(self) -> np.ndarray: ...
    def get_dof_pos(self) -> np.ndarray: ...
    def get_dof_vel(self) -> np.ndarray: ...
    def get_sensor_data(self, name: str) -> np.ndarray: ...

    # ── 域随机化 ──
    def get_dr_capabilities(self): ...
    def apply_init_randomization(self): ...
    def apply_interval_randomization(self): ...
```

### MuJoCo 后端实现要点

**文件**: `src/unilab/base/backend/mujoco/backend.py` (~1200行)

**状态布局**:
```python
self._idx_qpos = 1                          # qpos 在 physics_state 中的偏移
self._idx_qvel = 1 + self.nq                # qvel 偏移
self._root_qpos_dim = 7 if freejoint else 0 # floating base = 7 qpos
self._dof_pos_view = self._physics_state[
    :, self._idx_qpos + 7 : self._idx_qpos + self.nq
]  # 零拷贝 DOF 视图
```

**step 实现**:
```python
def step(self, ctrl, nsteps=1):
    control_traj = np.broadcast_to(ctrl[:, None, :], (N, nsteps, A))
    state_np, sensor_np = self._pool.step(
        self._physics_state, nstep=nsteps, control=control_traj, ...
    )
    self._physics_state[:] = state_np
    self._sensor_data[:] = sensor_np
```

**set_state 实现**:
```python
def set_state(self, env_indices, qpos, qvel, randomization=None):
    state_np = zeros((num_reset, nstate))
    state_np[:, qpos_slice] = qpos
    state_np[:, qvel_slice] = qvel
    state_out, sensor = self._pool.reset(env_indices, state_np, randomization)
    self._physics_state[env_indices] = state_out
    self._sensor_data[env_indices] = sensor
```

---

## 关键路径总结

### 训练时调用链

```
train_rsl_rl.py
  → ensure_registries()          # registry.py — 导入所有环境模块
  → make("XqRobotV2WalkFlat")    # registry.py — 查表创建 env
      ├─ XqRobotV2WalkFlatCfg()  # 实例化配置
      ├─ SceneCfg → MuJoCoBackend.__init__()
      │    ├─ _build_mujoco_scene_context()  # 加载 XML + 生成地形
      │    ├─ _load_base_model()             # 编译 MJCF
      │    └─ 预分配 _physics_state          # 向量化状态数组
      └─ XqRobotV2WalkFlatEnv(cfg, backend)
           └─ NpEnv.__init__()
  → env._init_domain_randomization()   # np_env.py — 编译 model variants
  → OnPolicyRunner(env, ...)
      → for iter in 1..5000:
          → env.step(action)
              ├─ apply_action()         # env 层 — 动作→控制
              ├─ backend.step(ctrl)     # backend 层 — 物理仿真
              └─ update_state()         # env 层 — obs/reward/terminated
          → PPO update
              ├─ patch_transition_next_obs()  # final_observation.py — bootstrap
              └─ actor_critic.update()        # 网络更新
```

### 评估时调用链

```
assess/runner.py
  → build_env(task)              # 直接创建 env (绕过 Hydra)
      └─ XqRobotV2WalkFlatCfg() + XqRobotV2WalkFlatEnv()
  → policy(obs) → action
  → env.step(action)
  → 记录 metrics
```

---

## 相关文件

| 文件 | 行数 | 说明 |
|------|------|------|
| `base/base.py` | 229 | ABEnv + EnvCfg |
| `base/np_env.py` | 471 | NpEnv 核心 (step/reset 循环) |
| `base/registry.py` | 324 | 环境注册 + make 工厂 |
| `base/scene.py` | 29 | 场景配置 |
| `base/curriculum.py` | 90 | 惩罚课程 |
| `base/final_observation.py` | 160 | Bootstrap 补丁 |
| `base/observations.py` | 37 | 观测工具 |
| `base/backend/base.py` | ~700 | SimBackend 接口 |
| `base/backend/mujoco/backend.py` | ~1200 | MuJoCo 实现 |
