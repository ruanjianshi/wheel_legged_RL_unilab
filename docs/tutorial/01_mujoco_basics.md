# 01. MuJoCo 基础与后端集成

## 目录

- [MuJoCo MJCF 语法](#mujoco-mjcf-语法)
- [仿真后端抽象 (SimBackend)](#仿真后端抽象-simbackend)
- [场景配置 (SceneCfg)](#场景配置-scenecfg)
- [MuJoCo 后端实现](#mujoco-后端实现)
- [环境契约 (NpEnv)](#环境契约-npenv)
- [全流程串联](#全流程串联)

---

## MuJoCo MJCF 语法

MuJoCo 使用 XML 格式描述机器人模型。以下是一个最小的完整模型：

```xml
<mujoco model="simple">
  <!-- 编译选项 -->
  <compiler angle="radian" />              <!-- 角度单位：弧度 -->

  <!-- 默认配置（可被继承） -->
  <default>
    <joint damping="0.5" armature="0.002"/>  <!-- 关节默认阻尼 -->
  </default>

  <!-- 世界坐标系下的物体 -->
  <worldbody>
    <!-- 地板 -->
    <geom name="floor" type="plane" size="0 0 0.05"/>

    <!-- 基座（根刚体） -->
    <body name="base" pos="0 0 0.6">
      <freejoint/>                           <!-- 自由浮动关节 = 6 DOF -->
      <inertial pos="0 0 0" mass="1.0" diaginertia="0.01 0.01 0.01"/>

      <!-- 大腿关节 -->
      <body name="thigh" pos="0 0 -0.3">
        <joint name="knee" type="hinge" range="-1.5 1.5" axis="0 1 0"/>
        <!-- ... -->
      </body>
    </body>
  </worldbody>

  <!-- 执行器定义 -->
  <actuator>
    <position name="knee_act" joint="knee" kp="30"/>
  </actuator>

  <!-- 传感器定义 -->
  <sensor>
    <jointpos name="knee_pos" joint="knee"/>
  </sensor>

  <!-- Keyframe：初始姿态 -->
  <keyframe>
    <key name="home"
      qpos="0 0 0.6  1 0 0 0   0.5"
      ctrl="0"/>
  </keyframe>
</mujoco>
```

### 核心元素说明

| 元素 | 说明 | 关键属性 |
|------|------|----------|
| `<mujoco>` | 根元素 | `model="name"` |
| `<compiler>` | 编译选项 | `angle` (radian/degree), `meshdir` |
| `<default>` | 可继承默认值 | `class="..."` 给子元素继承 |
| `<worldbody>` | 世界坐标下的物体 | — |
| `<body>` | 刚体 | `name`, `pos`, `childclass` |
| `<freejoint>` | 6-DOF 自由浮动 | 通常挂根刚体上 |
| `<joint>` | 运动副 | `type` (hinge/slide/ball), `range`, `axis` |
| `<geom>` | 碰撞/视觉几何 | `type` (mesh/box/sphere/plane), `group`, `contype`, `conaffinity` |
| `<inertial>` | 惯性参数 | `mass`, `diaginertia`, `pos` |
| `<site>` | 锚点 | 用于传感器、外力施加点 |
| `<actuator>` | 驱动器 | `<position kp>` 位置伺服; `<velocity kv>` 速度伺服 |
| `<sensor>` | 传感器 | `framelinvel`, `gyro`, `jointpos`, `jointvel`, `force`, `torque` |
| `<keyframe>` | 初始状态 | `qpos` (关节位置), `ctrl` (执行器目标) |

### qpos 和 ctrl 的排列

`qpos` 包含 **所有** 自由度（含 floating base）的初始值：
```
[base_x, base_y, base_z,  base_qw, base_qx, base_qy, base_qz,  joint_0, joint_1, ...]
```

`ctrl` 包含 **所有执行器** 的初始目标值，按定义顺序排列。

---

## 仿真后端抽象 (SimBackend)

**文件**: `src/unilab/base/backend/base.py`

UniLab 通过 `SimBackend` 抽象接口隔离物理引擎。所有环境只调用接口方法，不直接依赖 MuJoCo API。

### 核心方法

```python
class SimBackend(abc.ABC):
    """仿真后端抽象基类"""

    # ── 模型查询 ──
    def get_actuator_ctrl_range(self) -> np.ndarray: ...
    def get_keyframe_qpos(self) -> np.ndarray: ...
    def get_joint_range(self) -> np.ndarray: ...

    # ── 状态操作 ──
    def set_state(self, env_indices, qpos, qvel, randomization=None) -> None: ...
    def step(self, ctrl: np.ndarray, nsteps: int = 1) -> dict | None: ...

    # ── 传感器读取 ──
    def get_base_pos(self) -> np.ndarray: ...       # (N, 3)
    def get_base_quat(self) -> np.ndarray: ...       # (N, 4)
    def get_base_lin_vel(self) -> np.ndarray: ...    # (N, 3)
    def get_base_ang_vel(self) -> np.ndarray: ...    # (N, 3)
    def get_dof_pos(self) -> np.ndarray: ...         # (N, num_dof)
    def get_dof_vel(self) -> np.ndarray: ...         # (N, num_dof)
    def get_sensor_data(self, name) -> np.ndarray: ...

    # ── 生命周期 ──
    def materialize(self) -> None: ...
```

### 关键设计

1. **批量并行**：所有方法接受 `(num_envs, ...)` 形状的数组，一次操作所有环境
2. **零拷贝**：`get_dof_pos()` 返回对内部状态数组的 view，不分配新内存
3. **域随机化**：`set_state()` 的可选 `randomization` 参数在 reset 时应用随机化

---

## 场景配置 (SceneCfg)

**文件**: `src/unilab/base/scene.py`

```python
@dataclass
class TerrainSceneCfg:
    """地形场景槽位"""
    generator: TerrainGeneratorCfg | None = None   # 非空时生成地形
    hfield_name: str = "terrain_hfield"            # 高度场名称
    geom_name: str | None = None                   # 地形碰撞体名

@dataclass
class SceneCfg:
    """场景源配置"""
    model_file: str                                # 机器人 XML 路径 ★ 必填
    fragment_files: list[str] = field(default_factory=list)  # Keyframe/场景 XML
    terrain: TerrainSceneCfg | None = None         # 地形配置（可选）
    visual_model_file: str | None = None           # 可选高精视觉模型
```

### 场景组装流程

```python
SceneCfg(model_file="xqrobotV2.xml",
         fragment_files=["locomotion_task.xml"],      # Keyframe 在这里
         terrain=TerrainSceneCfg(generator=...))       # 非空则生成地形
```

```
_build_mujoco_scene_context()
   ├─ 如果无地形: 直接 merge robot.xml + fragment .xml
   └─ 如果有地形:
        ├─ TerrainGenerator.generate() → 高度场
        ├─ 将高度场写入 PNG 纹理
        └─ materialize_mujoco_hfield_attached_scene() 合并所有 XML
          │
          ▼
MuJoCoBackend 加载编译好的 MJCF 模型
```

---

## MuJoCo 后端实现

**文件**: `src/unilab/base/backend/mujoco/backend.py`

### 状态布局初始化

```python
class MuJoCoBackend(SimBackend):
    def __init__(self, scene, num_envs, sim_dt, ...):
        self._model = self._load_base_model()        # 加载编译后的 MJCF

        # 计算状态数组中各段的偏移
        self._idx_qpos = 1                           # qpos 起始位置
        self._idx_qvel = 1 + self.nq                 # qvel 起始位置
        self._root_qpos_dim, self._root_qvel_dim = _root_state_dims(self._model)

        # 预分配批量状态数组 (N, nstate)
        nstate = mujoco.mj_stateSize(...)
        self._physics_state = np.zeros((num_envs, nstate))
        self._sensor_data = np.zeros((num_envs, nsensordata))

        # 零拷贝 DOF 视图（排除 floating base 的 7 个 qpos）
        self._dof_pos_view = self._physics_state[
            :, self._idx_qpos + 7 : self._idx_qpos + self.nq
        ]
        # 同理底座位置、姿态、速度视图...
```

### Step 实现

```python
def step(self, ctrl: np.ndarray, nsteps: int = 1) -> dict | None:
    # ctrl shape: (num_envs, num_actuators)
    # 广播为 (num_envs, nsteps, num_actuators) 的轨迹
    control_traj = np.broadcast_to(
        ctrl[:, None, :], (num_envs, nsteps, ctrl.shape[-1])
    )

    # 批量仿真: 所有环境并行步进 nsteps 步
    state_np, sensor_np = self._pool.step(
        self._physics_state,
        nstep=nsteps,
        control=control_traj,
        control_spec=mujoco.mjtState.mjSTATE_CTRL,
        return_sensor=True,
    )

    # 写回缓存
    self._physics_state[:] = state_np
    self._sensor_data[:] = sensor_np
```

### Set State (Reset) 实现

```python
def set_state(self, env_indices, qpos, qvel, randomization=None):
    # 拼接 qpos + qvel → 完整物理状态
    state_np = np.zeros((num_reset, self._physics_state.shape[1]))
    state_np[:, self._idx_qpos : self._idx_qpos + self.nq] = qpos
    state_np[:, self._idx_qvel : self._idx_qvel + self.nv] = qvel

    # Pool.reset() 应用域随机化并 forward 到新状态
    state_out, sensor_np = self._pool.reset(
        env_ids=env_indices,
        initial_state=state_np,
        randomization=self._translate_reset_randomization(randomization, num_reset),
    )

    # 写回
    self._physics_state[env_indices] = state_out
    self._sensor_data[env_indices] = sensor_np
```

---

## 环境契约 (NpEnv)

**文件**: `src/unilab/base/np_env.py`

```python
@dataclass
class NpEnvState:
    obs: dict[str, np.ndarray]          # {"obs": (N, 297), "critic": (N, 324)}
    reward: np.ndarray                   # (N,)
    terminated: np.ndarray               # (N,) bool — 自然终止
    truncated: np.ndarray                # (N,) bool — 超时截断
    info: dict[str, Any]                 # 任意额外信息
    final_observation: dict | None = None # 终止时的最后一帧观测

class NpEnv(ABEnv):
    def step(self, actions: np.ndarray) -> NpEnvState:
        ctrl = self.apply_action(actions, self._state)   # ① 动作→控制
        self._backend.step(ctrl, sim_substeps)            # ② 物理仿真
        self._state = self.update_state(self._state)     # ③ 计算 obs/reward/terminated
        return self._state
```

### 子类必须实现

| 方法 | 职责 | 输入 → 输出 |
|------|------|-------------|
| `apply_action(actions, state)` | 动作转换 | `(N, 8) → (N, 8)` 控制量 |
| `update_state(state)` | 观测/奖励/终止 | `NpEnvState → NpEnvState` |
| `obs_groups_spec` | 观测维度 | `property → {"obs": 297, "critic": 324}` |

---

## 全流程串联

以 XqRobotV2 为例，一个完整的仿真步进流程：

```
Policy 输出 actions (N, 8)
  │
  ▼
NpEnv.step(actions)
  │
  ├─ apply_action():
  │     leg[0:3] = action[0:3] * 0.5 + DEFAULT_ANGLES[0:3]  → L_leg 位置目标
  │     leg[3:6] = action[3:6] * 0.5 + DEFAULT_ANGLES[3:6]  → R_leg 位置目标
  │     wheel[0] = action[6] * 10.0   → L_wheel 速度目标
  │     wheel[1] = action[7] * 10.0   → R_wheel 速度目标
  │     重排为 MuJoCo 执行器顺序:
  │       [L_hip, L_thigh, L_calf, L_wheel, R_hip, R_thigh, R_calf, R_wheel]
  │
  ├─ backend.step(ctrl, nsteps=2):     # ctrl_dt=0.01, sim_dt=0.005 → 2 substeps
  │     control_traj = broadcast(ctrl, (N, 2, 8))
  │     pool.step(state, 2, control_traj) → new_state, sensors
  │
  └─ update_state():
        ├─ 读传感器: linvel(3), gyro(3), gravity(3), dof_pos(8), dof_vel(8)
        ├─ 计算观测: 33 维单帧 → 堆叠 9 帧历史 → 297 维
        ├─ 计算奖励: 调用 13 个奖励函数 → 加权求和 × 0.01
        ├─ 检查终止: tilt > 60° | height < 0.2 | thigh 塌陷 | calf 过伸
        └─ 返回 NpEnvState(obs, reward, terminated, truncated, info)
```

---

## 关键要点

1. **`<freejoint>` 必须挂根刚体上**，提供 6-DOF 浮动底座
2. **Keyframe 放 fragment XML**，不放进 robot.xml
3. **执行器顺序与关节顺序**：在 `apply_action()` 中确保控制量排列与 MJCF 中 `<actuator>` 定义顺序一致
4. **步长关系**：`ctrl_dt = sim_dt × sim_substeps`，XqRobotV2 用 `0.01 = 0.005 × 2`

---

> 下一章：[02. 地形建模](./02_terrain_modeling.md)
