# XqRobotV2 项目架构与实现细节

> README 速览，本文档补充内部实现和开发历史。
> 更新：2026-06-27

---

## 一、关键实现逻辑

### 1.1 默认站姿

```
DEFAULT_LEG_ANGLES = [0.1, 0.1, -0.1, 0.1, 0.1, -0.1]
```

双髋设为 0.1 而非 0.0 的原因：XqRobotV2 结构 COM 高，髋=0.0 时双腿完全直立，横向支撑面过窄，初始探索即大量摔倒 → 策略崩溃。0.1 提供外展支撑。keyframe XML 右髋=0（历史遗留），控制器自动从 0 拉到 0.1。

### 1.2 髋关节镜像

两个髋关节轴都是 `"1 0 0"`（同向 convention）。正确镜像为 `left + right ≈ 0`。`similar_calf` 奖励中 `(left_hip + right_hip)²` 项确保这一点。

### 1.3 轮子速度控制

```
XML:  <velocity joint="left_joint_wheel" kv="1"/>
代码: wheel_target = action * 10.0  (不给 DEFAULT 偏移)
```

从位置控制 (kp=3) 改为速度控制 (kv=1) 后，轮子直接响应目标转速而非缓慢爬位置。ctrl_dt 从 0.02 降到 0.01 匹配 100Hz。

### 1.4 5D 命令与观测维度同步

改动命令维度时，`obs_frame_dim` 和 `critic_frame_dim` 必须同步：

| 命令维度 | obs_frame_dim | critic_frame_dim |
|----------|---------------|------------------|
| 4D | 32 | 35 |
| 5D | 33 | 36 |

不改会导致 history buffer 维度不匹配，观测乱码 → 策略崩溃。

### 1.5 课程对称扩展

```python
vx_range = max(abs(low[0]), abs(high[0]))
low[0] = max(low[0] - step, -vx_range)
high[0] = min(high[0] + step, vx_range)
```

旧代码 `full_low_x = 0.0` 只扩正向，对称 vel_limit 会被破坏。

### 1.6 feet_distance 轮距约束

```python
over = max(0, dist - 0.6)  # 过宽
under = max(0, 0.3 - dist)  # 过窄
return (over + under) * 0.3 * scale(-1.0)
```

XqRobotV2 默认轮距约 0.33m，接近下限。权重 -100.0 会压倒其他奖励，已降至 -1.0。

---

## 二、文件依赖关系

```
mujoco.yaml (config)
  └→ XqRobotV2WalkFlatCfg (dataclass)
       ├→ scene_flat.xml (MuJoCo 模型 + keyframe)
       └→ XqRobotV2WalkFlatEnv.__init__
            ├→ backend = create_backend(xqrobotV2.xml)
            ├→ _init_reward_functions()  → self._reward_fns
            ├→ _init_domain_randomization() → XqRobotDRProvider
            └→ apply_action()            → 动作→ctrl 映射

training loop:
  update_state() → _update_commands() → _update_curriculum()
               → _compute_terminated()
               → _compute_reward()    → run_reward_dispatch(scales, fns)
               → _compute_obs()       → 9 帧 history stacking
```

---

## 三、CJ-003 对比

| 参数 | CJ-003 (Genesis) | XqRobotV2 (UniLab) |
|------|-----------------|-------------------|
| 框架 | Genesis | UniLab (MuJoCo) |
| 轮控制 | 速度控制, kv=5 | 速度控制, kv=1 |
| ctrl_dt | 0.01s | 0.01s |
| 腿 kp | 30 | 30 |
| 默认髋角 | 0.0 | 0.1 |
| tracking_sigma | 0.3 | 0.3 |
| orientation 权重 | -14.0 | -10.0 |
| 命令维度 | 5D | 5D |
| vel_limit | [-2.0, 2.0] | [-0.6, 0.6] |

---

## 四、调试技巧

```bash
# 验证关节定义
uv run python -c "
import mujoco
m = mujoco.MjModel.from_xml_path('src/unilab/assets/robots/xqrobotV2/xqrobotV2.xml')
for i in range(m.njnt):
    name = mujoco.mj_id2name(m, 3, i)
    if name and 'joint' in name:
        print(f'{name:25s} axis={m.jnt_axis[i]} range={m.jnt_range[i]}')
"

# 验证执行器
uv run python -c "
import mujoco
m = mujoco.MjModel.from_xml_path('src/unilab/assets/robots/xqrobotV2/xqrobotV2.xml')
print(f'actuators: {m.nu}, sensors: {m.nsensor}')
for i in range(m.nu):
    name = mujoco.mj_id2name(m, 5, i)
    print(f'  {name} type={m.actuator_trntype[i]}')
"

# 验证 Hydra 配置
uv run python -c "
from hydra import compose, initialize_config_dir
from pathlib import Path
with initialize_config_dir(version_base='1.3', config_dir=str(Path.cwd()/'conf'/'ppo')):
    cfg = compose(config_name='config', overrides=['task=xqrobotV2_walk_flat/mujoco'])
    print('task_name:', cfg.training.task_name)
    print('vel_limit:', cfg.env.commands.vel_limit)
    print('num_envs:', cfg.algo.num_envs)
"

# 物理验证：正轮速 → 前进
uv run python -c "
import mujoco, numpy as np
m = mujoco.MjModel.from_xml_path('src/unilab/assets/robots/xqrobotV2/scene_flat.xml')
d = mujoco.MjData(m)
d.ctrl[6:] = [10, 10]  # 正轮速
for _ in range(200): mujoco.mj_step(m, d)
print('vx:', d.sensor('local_linvel').data[0], '(应为正)')
"
```

---

## 五、依赖注意

```bash
# uv 包管理，不用 pip 直接装
uv sync --extra mujoco

# mujoco-uni==3.8.0 和 mujoco 不能共存
```

---

## 六、问题记录

详细开发问题记录见 **[docs/PROBLEMS.md](PROBLEMS.md)**
