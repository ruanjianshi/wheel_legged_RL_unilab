# 05. URDF 机器人移植

## 目录

- [移植流程总览](#移植流程总览)
- [Step 1: URDF→MJCF 转换](#step-1-urdfmjcf-转换)
- [Step 2: 网格文件重组](#step-2-网格文件重组)
- [Step 3: XML 后处理](#step-3-xml-后处理)
- [Step 4: 生成场景 XML](#step-4-生成场景-xml)
- [Step 5: 交互式 Keyframe 调试](#step-5-交互式-keyframe-调试)
- [Step 6: 编写环境代码](#step-6-编写环境代码)
- [Step 7: 注册 & 配置](#step-7-注册--配置)

---

## 移植流程总览

```
URDF 文件 (your_robot.urdf)
  │
  └─ uv run unilab-import-robot your_robot.urdf
       │
       ├─ Step 1: urdf-to-mjcf  →  assets/robots/<name>/<name>.xml
       ├─ Step 2: 重组 mesh 文件 →  assets/
       ├─ Step 3: XML 后处理      →  修正路径, 转换执行器
       ├─ Step 4: 生成 scene.xml  →  初始 Keyframe
       └─ Step 5: 交互调试       →  MuJoCo viewer 调整姿态
              │
              ▼
          完成后的目录:
          src/unilab/assets/robots/<name>/
          ├── <name>.xml           # 机器人 MJCF
          ├── scene.xml            # 场景 + Keyframe
          └── assets/              # STL/OBJ 网格文件
```

**工具入口**: `src/unilab/tools/import_robot.py` (666行)

---

## Step 1: URDF→MJCF 转换

```bash
uv run unilab-import-robot /path/to/your_robot.urdf --name my_robot
```

内部调用 `urdf-to-mjcf` 开源工具：

```python
# import_robot.py:66-73
def _convert_urdf(urdf: Path, output_xml: Path) -> None:
    output_xml.parent.mkdir(parents=True, exist_ok=True)
    _run([
        "uv", "run", "--with", "urdf-to-mjcf",
        "urdf-to-mjcf", str(urdf), "-o", str(output_xml),
    ])
```

**URDF→MJCF 映射**:
| URDF | MuJoCo |
|------|--------|
| `<joint type="revolute">` | `<joint type="hinge" range="...">` |
| `<joint type="continuous">` | `<joint type="hinge">` (无 range) |
| `<joint type="prismatic">` | `<joint type="slide">` |
| `<joint type="fixed">` | 合并/固定 |
| `<inertial>` + `<origin>` | `<inertial pos="...">` |
| `<visual>/<collision>` geometry | `<geom type="mesh">` |

---

## Step 2: 网格文件重组

```python
# import_robot.py:82-95
def _move_mesh_assets(robot_dir: Path) -> None:
    generated_mesh_dir = robot_dir / "meshes" / "meshes"
    target_assets_dir = robot_dir / "assets"
    generated_mesh_dir.rename(target_assets_dir)  # 展平到 assets/
    meshes_parent = robot_dir / "meshes"
    meshes_parent.rmdir()
```

输出目录:
```
assets/robots/my_robot/
├── my_robot.xml
└── assets/              # ★ 所有 STL 文件
    ├── base_link.STL
    ├── leg_upper.STL
    └── ...
```

---

## Step 3: XML 后处理

```python
# import_robot.py:301-310
def _postprocess_xml(xml_path: Path) -> None:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    _set_mesh_paths(root)           # ① 修正 mesh 路径
    _convert_motor_actuators(root)  # ② <motor> → <position>
    _ensure_robot_default_joint(root)  # ③ 默认关节阻尼
    _remove_default_material(root)  # ④ 删除自动材质
    _strip_generated_scene_bits(root)  # ⑤ 删除场景元素
    tree.write(xml_path, encoding="unicode")
```

### ① mesh 路径修正
```xml
<!-- 转换前 -->
<compiler meshdir="meshes"/>
<mesh file="meshes/link.STL"/>
<!-- 转换后 -->
<compiler meshdir="assets"/>
<mesh file="link.STL"/>
```

### ② 执行器转换
```xml
<!-- 转换前 -->
<motor name="joint_1" joint="joint_1" gear="1"/>
<!-- 转换后 -->
<position name="joint_1" joint="joint_1" kp="30"/>
```

### ③④⑤ 清理
删除：地板 geom、灯光、纹理材质、specular/shininess 属性。这些属于场景 XML。

---

## Step 4: 生成场景 XML

```python
# import_robot.py:287-298
def _write_scene_xml(robot_xml, scene_xml, robot_name):
    qpos, ctrl = _scene_keyframe_values(robot_xml)
    # 创建 <mujoco><keyframe><key qpos="..." ctrl="..."/></keyframe></mujoco>
```

生成 `scene.xml`:
```xml
<mujoco model="my_robot scene">
  <keyframe>
    <key name="home"
      qpos="0 0 0.6  1 0 0 0  0 0 0 ..."
      ctrl="0 0 0 0 ..."/>
  </keyframe>
</mujoco>
```

---

## Step 5: 交互式 Keyframe 调试

```python
# import_robot.py:632-636
def _tune_scene_keyframe(robot_xml, scene_xml):
    model = _compile_tuning_scene(robot_xml, scene_xml)
    data = _load_keyframe(model)
    _open_tuning_viewer(model, data)      # ★ 打开 MuJoCo 交互界面
    _write_tuned_scene_keyframe(scene_xml, model, data)  # 保存调整
```

MuJoCo 查看器提供 **4 个滑块** (X/Y/Z 平移 + 旋转) + **高度微调执行器** (kp=1000)。拖滑块调好姿态后关闭窗口，`scene.xml` 自动保存。

### 调参建议
1. 调 Z 让机器人接地（脚/轮触地）
2. 确保底座水平（roll/pitch ≈ 0）
3. 调腿关节使起始姿态自然
4. 关闭窗口 → keyframe 自动写入

---

## Step 6: 编写环境代码

### 最少文件

```
src/unilab/envs/locomotion/my_robot/
├── __init__.py       # 注册声明
├── base.py           # 关节定义, 配置类
└── joystick.py       # env 实现
```

### `__init__.py`
```python
__unilab_registry_modules__ = (
    "unilab.envs.locomotion.my_robot.joystick",
)
```

### `base.py` (核心字段)
```python
JOINT_PREFIXES = ("left_hip", "left_thigh", ...)  # ★ 必须与 XML 一致
NUM_ACTIONS = len(JOINT_PREFIXES)
DEFAULT_ANGLES = np.array([0.0, 0.3, -0.6, ...])  # 站立姿态

@dataclass
class MyRobotControlConfig:
    action_scale: float = 0.25
    clip_actions: float = 1.0
    simulate_action_latency: bool = False

@dataclass
class MyRobotWalkCfg(LocomotionBaseCfg):
    control_config = MyRobotControlConfig()
    noise_config = MyRobotNoiseConfig()
    sim_dt: float = 0.005
    ctrl_dt: float = 0.01
```

### `joystick.py` (必须实现的 3 个方法)

```python
class MyRobotWalkEnv(LocomotionBaseEnv):
    def apply_action(self, actions, state):
        """动作→控制量: action*scale + default"""
        leg_targets = actions[:, :6] * cfg.action_scale + DEFAULT_ANGLES
        return leg_targets

    def _compute_obs(self, info, linvel, gyro, gravity, dof_pos, dof_vel):
        """构建观测: [gyro, -gravity, leg_diff, leg_vel, commands]"""
        ...

    def _compute_terminated(self, gravity, dof_pos):
        """终止条件: tilt>1.0 | height<0.2"""
        tilt = np.arccos(gravity[:, 2])
        height = self._backend.get_base_pos()[:, 2]
        return (tilt > 1.0) | (height < 0.2)
```

完整参考: `src/unilab/envs/locomotion/xqrobotV2/joystick.py` (484行)

---

## Step 7: 注册 & 配置

### 环境注册
```python
@envcfg("MyRobotWalkFlat")
@dataclass
class MyRobotWalkFlatCfg(MyRobotWalkCfg):
    scene: SceneCfg = field(default_factory=lambda: SceneCfg(
        model_file=str(ASSETS_PATH / "robots/my_robot/my_robot.xml"),
        fragment_files=[str(ASSETS_PATH / "robots/my_robot/scene.xml")],
    ))

@env("MyRobotWalkFlat", sim_backend="mujoco")
class MyRobotWalkFlatEnv(MyRobotWalkEnv):
    pass
```

### 训练配置
`conf/ppo/task/my_robot_walk_flat/mujoco.yaml`:
```yaml
training:
  task_name: MyRobotWalkFlat    # ★ = @envcfg 注册名
  sim_backend: mujoco
algo:
  num_envs: 1024
  num_steps_per_env: 25
  max_iterations: 5000
  # ...
env:
  control_config:
    action_scale: 0.5
  commands:
    vel_limit:
      - [-0.6, -0.3, -1.0]
      - [ 0.6,  0.3,  1.0]
reward:
  scales:
    tracking_lin_vel: 1.5
    orientation: -10.0
    alive: 1.0
```

### 启动训练
```bash
uv run train --algo ppo --task my_robot_walk_flat --sim mujoco
```

---

## 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| 机器人飘走 | Keyframe 底座高度不对 | Step 5 调 Z 到接地 |
| 关节不动 | 执行器未转换 | 检查 `<motor>→<position>` |
| Mesh 找不到 | 路径错误 | `meshdir="assets"` + `file="link.STL"` |
| 传感器为空 | 命名不匹配 | `JOINT_PREFIXES` 对齐 sensor name |
| qpos 维度错 | 关节数统计 | `<freejoint>`=7 qpos, +N joints |

---

> 上一章: [04. RL 算法与训练](./04_rl_algorithms.md)
> 下一章: [06. 奖励函数设计](./06_reward_function.md)
