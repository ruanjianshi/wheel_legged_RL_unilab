# 03. 机器人建模

## 目录

- [MuJoCo 机器人 XML 结构](#mujoco-机器人-xml-结构)
- [XqRobotV2 逐行拆解](#xqrobotv2-逐行拆解)
- [关节与执行器](#关节与执行器)
- [传感器配置](#传感器配置)
- [Keyframe 与场景 XML](#keyframe-与场景-xml)
- [mesh 网格文件组织](#mesh-网格文件组织)

---

## MuJoCo 机器人 XML 结构

一个完整的机器人 MJCF 文件包含以下模块：

```xml
<mujoco model="robot_name">
  <!-- 1. 默认类 (可继承配置) -->
  <default>
    <default class="robot">
      <joint damping="..." armature="..."/>
      <default class="visual">
        <geom contype="0" conaffinity="0" group="2"/>
      </default>
      <default class="collision">
        <geom contype="0" conaffinity="1" group="3"/>
      </default>
    </default>
  </default>

  <!-- 2. 编译选项 -->
  <compiler angle="radian" meshdir="assets" balanceinertia="true"/>

  <!-- 3. 运动学树 (worldbody) -->
  <worldbody>
    <body name="base_link" childclass="robot">
      <freejoint/>  <!-- 浮动底座 -->
      <inertial .../>
      <!-- 子树: 腿/臂/头 ... -->
    </body>
  </worldbody>

  <!-- 4. 资源 (网格/纹理/材质) -->
  <asset>
    <mesh name="link_collision" file="link.STL"/>
  </asset>

  <!-- 5. 执行器 -->
  <actuator>
    <position name="joint_act" joint="joint_name" kp="30"/>
    <velocity name="wheel_act"  joint="wheel_name" kv="1"/>
  </actuator>

  <!-- 6. 传感器 -->
  <sensor>
    <framelinvel .../>  <!-- 底部线速度 -->
    <gyro .../>          <!-- 角速度 -->
    <jointpos .../>      <!-- 关节位置 -->
    <jointvel .../>      <!-- 关节速度 -->
    <force .../>         <!-- 力传感器 -->
    <torque .../>        <!-- 力矩传感器 -->
  </sensor>
</mujoco>
```

---

## XqRobotV2 逐行拆解

**文件**: `src/unilab/assets/robots/xqrobotV2/xqrobotV2.xml` (140行)

### 默认类与编译

```xml
<mujoco model="xqrobotV2">
  <default>
    <default class="robot">
      <joint damping="0.5" armature="0.002" frictionloss="0.01" />
      <default class="visual">
        <geom contype="0" conaffinity="0" group="2" />
      </default>
      <default class="collision">
        <geom contype="0" conaffinity="1" group="3" />
      </default>
    </default>
  </default>
  <compiler angle="radian" meshdir="assets" balanceinertia="true" />
```

**说明**:
- `childclass="robot"` → 所有子 body 自动继承关节阻尼、几何体分组
- `group="2"` 放视觉几何体，`group="3"` 放碰撞几何体（MuJoCo 渲染组）
- `contype=0` → 自身不产生接触；`conaffinity=1` → 可被其他刚体碰到
- `balanceinertia` → 自动平衡惯性矩阵（确保正定）

### 底座 (Line 15-19)

```xml
  <worldbody>
    <body name="base_link" childclass="robot" pos="0 0 0.5022">
      <inertial pos="0 0 0" mass="5.0" diaginertia="0.1 0.1 0.1" />
      <site name="imu_in_base" size="0.01" />
      <geom name="base_link_collision" type="mesh" mesh="base_link_collision_base_link" class="collision" />
      <geom name="base_link_visual" type="mesh" mesh="base_link_base_link" class="visual" />
```

**说明**:
- `mass="5.0"` — 底座质量 5kg
- `imu_in_base` site — 传感器挂载点
- 每个刚体都需要 **两个 `<geom>`**：`class="collision"` 用于物理，`class="visual"` 用于渲染

### 左腿链 (Line 20-47)

```xml
      <!-- 髋 Roll 关节 (左) -->
      <body name="left_link_1" pos="0.069 -0.124 -0.001">
        <inertial pos="-0.033 -0.011 0.000" mass="0.174" diaginertia="9.8e-5 1.8e-4 1.3e-4" />
        <joint name="left_joint_1" type="hinge" range="-3.14 3.14" axis="1 0 0" />
        <!-- axis="1 0 0" = x 轴 = roll 转动 -->
        <geom name="left_link_1_collision" type="mesh" mesh="left_link_1_collision_left_link_1" class="collision" />
        <site name="left_hip_site" size="0.005" />

        <!-- 大腿 Pitch 关节 (左) -->
        <body name="left_link_2" pos="-0.070 -0.019 0.000">
          <inertial ... mass="0.673" />
          <joint name="left_joint_2" type="hinge" range="0 2.094" axis="0 1 0" />
          <!-- axis="0 1 0" = y 轴 = pitch 转动 -->
          <site name="left_thigh_site" size="0.005" />

          <!-- 小腿 Pitch 关节 (左) -->
          <body name="left_link_3" pos="-0.224 0.015 -0.200">
            <inertial ... mass="0.420" />
            <joint name="left_joint_3" type="hinge" range="-0.873 0.873" axis="0 1 0" />
            <site name="left_calf_site" size="0.005" />

            <!-- 轮子 (左) — 无关节限制 -->
            <body name="left_link_wheel" pos="0.224 -0.037 -0.199">
              <inertial ... mass="2.323" />
              <joint name="left_joint_wheel" type="hinge" axis="0 1 0" />
              <!-- 注意: 没有 range 属性 = 无限旋转 -->
            </body>
          </body>
        </body>
      </body>
```

**说明**: 每个 body 的 `pos` 是**相对于父 body** 的偏移。从 base_link → hip → thigh → calf → wheel 形成 5 级嵌套树。

### 右腿链 (Line 48-75)

右腿是左腿的 **镜像**，注意 y 坐标符号相反：

```xml
      <body name="right_link_1" pos="0.069 0.124 -0.001">    <!-- y=+0.124 vs 左=-0.124 -->
        <joint name="right_joint_1" type="hinge" range="-3.14 3.14" axis="1 0 0" />
        <body name="right_link_2" pos="-0.070 0.019 0.000">   <!-- y=+0.019 vs 左=-0.019 -->
          ...
```

### 浮动关节 (Line 76)

```xml
      <freejoint name="floating_base" />
    </body>
  </worldbody>
```

`<freejoint>` 必须挂在**根 body 上**，提供 6-DOF 自由运动（3 平移 + 3 旋转）。

### 网格资源 (Line 79-98)

```xml
  <asset>
    <mesh name="base_link_collision_base_link" file="base_link.STL" />
    <mesh name="base_link_base_link" file="base_link.STL" />
    <mesh name="left_link_1_collision_left_link_1" file="left_link_1.STL" />
    <!-- ... 共 18 个 mesh 引用 (9 collision + 9 visual) ... -->
  </asset>
```

**命名规范**: `{body_name}_{class}_{mesh_name}` — body_name 对应 `<body name="...">`，class 对应 `collision/visual`。

### 执行器 (Line 99-108)

```xml
  <actuator>
    <position name="left_joint_1"  joint="left_joint_1"  kp="30" />
    <position name="left_joint_2"  joint="left_joint_2"  kp="30" />
    <position name="left_joint_3"  joint="left_joint_3"  kp="30" />
    <velocity name="left_joint_wheel" joint="left_joint_wheel" kv="1" />
    <position name="right_joint_1" joint="right_joint_1" kp="30" />
    <position name="right_joint_2" joint="right_joint_2" kp="30" />
    <position name="right_joint_3" joint="right_joint_3" kp="30" />
    <velocity name="right_joint_wheel" joint="right_joint_wheel" kv="1" />
  </actuator>
```

**两种执行器类型**:

| 类型 | 属性 | 控制模型 | 用途 |
|------|------|----------|------|
| `<position kp="N">` | 位置伺服 | `τ = kp × (target_pos - actual_pos)` | 腿关节角度 |
| `<velocity kv="N">` | 速度伺服 | `τ = kv × (target_vel - actual_vel)` | 轮子转速 |

**执行器顺序**: 6 个位置执行器 + 2 个速度执行器 = 8 个执行器（交替排列：L_hip, L_thigh, L_calf, L_wheel, R_hip, R_thigh, R_calf, R_wheel）

### 传感器 (Line 109-139)

```xml
  <sensor>
    <!-- 底座 IMU -->
    <framelinvel name="local_linvel" objtype="body" objname="base_link" />
    <gyro name="gyro" site="imu_in_base" />
    <framezaxis name="upvector" objtype="body" objname="base_link" />

    <!-- 8 个关节位置传感器 (以 left_joint_1 为例) -->
    <jointpos name="left_joint_1_pos" joint="left_joint_1" />
    <jointvel name="left_joint_1_vel" joint="left_joint_1" />
    <!-- ... 共 16 个 jointpos/jointvel ... -->

    <!-- 力/力矩传感器 (10 个) -->
    <force  name="left_wheel_force"  site="left_wheel_site" />
    <torque name="left_hip_torque"   site="left_hip_site" />
    <!-- ... -->
  </sensor>
```

**传感器命名规范**: `{joint_prefix}_pos/vel`，与 `base.py` 的 `stack_joint_sensors()` 函数匹配。

---

## 关节与执行器

### 位置控制 vs 速度控制

```
┌───────────────────────────────────────────────────────┐
│  pos actuator (kp=30)                                 │
│                                                       │
│  target_pos ← action[i] * action_scale + default_angle│
│                        │                              │
│                        ▼                              │
│  MuJoCo:  τ = kp × (target_pos - actual_pos)         │
│           PD 伺服器将关节驱动到目标位置                  │
└───────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────┐
│  velocity actuator (kv=1)                             │
│                                                       │
│  target_vel ← action[i] * wheel_action_scale          │
│                        │                              │
│                        ▼                              │
│  MuJoCo:  τ = kv × (target_vel - actual_vel)         │
│           直接控制转速，不跟踪位置                       │
└───────────────────────────────────────────────────────┘
```

### 为什么轮子用速度控制？

1. 轮子需要**连续旋转**，Position control 需要处理角度环绕
2. 速度控制更自然地对应"移动速度"的概念
3. RL 策略直接学习"轮子该转多快才能达到目标速度"

---

## 传感器配置

### 传感器类型速查

| 传感器 | 输出 | 用途 |
|--------|------|------|
| `framelinvel` | (N,3) 局部线速度 | Rewards: tracking_lin_vel |
| `gyro` | (N,3) 角速度 | Obs: 角速度分量 |
| `framezaxis` | (N,3) 重力投影 | Obs: "上"方向; Rewards: orientation |
| `jointpos` | (N,) 关节角度 | Obs: 关节偏差; Rewards: hip roll, calf symmetry |
| `jointvel` | (N,) 关节速度 | Obs: 速度分量 |
| `force` | (N,3) site 受力 | 高级 reward（本配置未启用） |
| `torque` | (N,3) site 扭矩 | 力矩估算（本配置未启用） |

### 环境如何读取传感器

```python
# base.py - batch sensor reading
def stack_joint_sensors(backend, *, dtype):
    """读取所有 8 个关节位置"""
    names = tuple(f"{p}_pos" for p in JOINT_PREFIXES)
    # → ("left_joint_1_pos", "left_joint_2_pos", ..., "right_joint_wheel_pos")
    values = backend.get_sensor_data_batch(names)
    return values.reshape(N, -1)[:, :8]
```

---

## Keyframe 与场景 XML

### 核心规则

> **`<keyframe>` 必须放在 task-level XML 中，禁止放进 `robot.xml`。**
>
> `robot.xml` 是**纯机器人描述**（body/joint/actuator/sensor）。
> `keyframe` 是任务起始姿态，属于场景或任务资源。

### 场景 XML 示例

**文件**: `scene_flat.xml`

```xml
<mujoco model="xqrobotV2 scene">
  <include file="xqrobotV2.xml"/>  <!-- 引入机器人定义 -->

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <global azimuth="-130" elevation="-20"/>
  </visual>

  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
  </worldbody>

  <!-- ★ Keyframe: 初始姿态 -->
  <keyframe>
    <key name="home"
      qpos="
        0 0 0.65   1 0 0 0      ← base pos(xyz=0,0,0.65) + quat(w=1,xyz=0)
        -0.1 0.1 -0.1 0           ← L_hip, L_thigh, L_calf, L_wheel
         0.1 0.1 -0.1 0"          ← R_hip, R_thigh, R_calf, R_wheel
      ctrl="0 0 0 0 0 0 0 0"/>
  </keyframe>
</mujoco>
```

### Keyframe qpos 排列

```
base     base     base      base quaternion    左腿 4 关节    右腿 4 关节
x        y        z         w   x   y   z      hip thigh calf wheel  hip thigh calf wheel
0        0        0.65      1   0   0   0      -0.1 0.1  -0.1 0      0.1  0.1  -0.1 0
```

### 任务 Fragment

**文件**: `locomotion_task.xml` — 纯 Keyframe，用于训练：

```xml
<mujoco model="xqrobotV2 locomotion task">
  <keyframe>
    <key name="home" qpos="..." ctrl="..."/>
  </keyframe>
</mujoco>
```

**组装方式**：通过 `SceneCfg.fragment_files` 引用：
```python
SceneCfg(
    model_file="xqrobotV2.xml",
    fragment_files=["locomotion_task.xml"],  # 合并此 XML 的 keyframe
    terrain=...,
)
```

---

## mesh 网格文件组织

```
src/unilab/assets/robots/xqrobotV2/
├── xqrobotV2.xml          # 机器人定义 (引用 "left_link_1.STL" 等)
├── assets/                # mesh 文件目录
│   ├── base_link.STL
│   ├── left_link_1.STL
│   ├── left_link_2.STL
│   ├── left_link_3.STL
│   ├── left_link_wheel.STL
│   ├── right_link_1.STL
│   ├── right_link_2.STL
│   ├── right_link_3.STL
│   └── right_link_wheel.STL
├── scene_flat.xml         # 场景 + Keyframe
└── locomotion_task.xml    # 纯 Keyframe (训练用)
```

**关键**: `<compiler meshdir="assets"/>` 让 MuJoCo 在 `assets/` 子目录查找网格文件。

---

## 关键要点

1. **`<freejoint>`** 提供浮动底座，必须挂根 body 上
2. **几何体双重定义**：`class="collision"` 用于物理，`class="visual"` 用于渲染
3. **执行器类型**：腿用 `<position kp>`（角度伺服），轮用 `<velocity kv>`（速度伺服）
4. **传感器命名**：必须与 `base.py` 的 prefix 列表匹配
5. **Keyframe 分离**：`robot.xml` = 物理结构，`scene.xml` / `locomotion_task.xml` = Keyframe
6. **qpos 维度**：`7 (base) + num_joints`，ctrl 维度 = `num_actuators`

---

> 上一章：[02. 地形建模](./02_terrain_modeling.md)
> 下一章：[04. RL 算法与训练](./04_rl_algorithms.md)
