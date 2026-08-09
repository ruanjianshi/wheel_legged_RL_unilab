# XqRobotV2 项目架构与实现细节

> README 速览，本文档补充内部实现和开发历史。
> 更新：2026-06-29

---

## 一、关键实现逻辑

### 1.1 默认站姿

```
DEFAULT_LEG_ANGLES = [0.1, 0.1, -0.1, 0.1, 0.1, -0.1]
```

双髋 0.1：XqRobotV2 结构需要外展提供横向支撑。改为 0.0 会导致训练崩溃（ep=1）。

### 1.2 actuator 顺序（关键）

MuJoCo XML 中 actuator 顺序：`[L_hip, L_thigh, L_calf, L_wheel, R_hip, R_thigh, R_calf, R_wheel]`。轮子夹在中间。`apply_action` 必须精确匹配这个顺序，否则右腿关节全部错位。

```python
# apply_action 正确的拼接：
np.concatenate([
    leg_targets[:, :3],       # L hip, thigh, calf
    wheel_targets[:, :1],     # L wheel
    leg_targets[:, 3:],       # R hip, thigh, calf
    wheel_targets[:, 1:],     # R wheel
], axis=1)
```

### 1.3 轮子速度控制

```
XML:  <velocity joint="left_joint_wheel" kv="1"/>
代码: wheel_target = action × 10.0  (不给 DEFAULT 偏移)
```

从位置控制 (kp=3) 改为速度控制 (kv=1)。ctrl_dt=0.01 (100Hz)。

### 1.4 轮子接触力检测

```xml
<site name="left_wheel_site" size="0.005"/>
<force name="left_wheel_force" site="left_wheel_site"/>
```

通过 `get_sensor_data("left_wheel_site")` 读取 3D 力向量，|F| > 1.5N 判断着地。用于 `swing_lift` 奖励。

### 1.5 5D 命令

| 命令维度 | obs_frame | critic_frame |
|----------|-----------|--------------|
| 5D `[vx,vy,vyaw,tsk,height]` | 33 | 36 |

不改会导致 history buffer 维度不匹配。

### 1.6 课程对称扩展

```python
vx_range = max(abs(low[0]), abs(high[0]))
low[0] = max(low[0] - step, -vx_range)
high[0] = min(high[0] + step, vx_range)
```

旧代码 `full_low_x = 0.0` 只扩正向。

---

## 二、文件依赖关系

```
mujoco.yaml
  → XqRobotV2WalkFlatCfg
    → scene_flat.xml (模型 + keyframe)
    → XqRobotV2WalkFlatEnv
      → backend = create_backend(xqrobotV2.xml)
      → apply_action()     → ctrl 拼接 (见 §1.2)
      → update_state()
        → _compute_ref_dof_pos()  → 正弦参考轨迹 (重心转移 + 快抬)
        → _update_feet_distance() → 轮距约束
        → _compute_terminated()   → 触地终止
        → _compute_reward()
        → _compute_obs()          → 9 帧 history stacking

toe_walk 专用：
  → _compute_ref_dof_pos()   → 重心转移 + 压缩摆动相
  → _update_wheel_contact()  → 力传感器检测离地
  → _reward_swing_lift()     → 离地才奖励
```

---

## 三、点足行走参考轨迹

### 3.1 步态序列 (0.5s 周期)

```
T0:        重心右倾 → 左腿即将抬
T0-0.09:   左膝快收 → 左轮离地
T0.09-0.18: 左大腿前摆 → 迈步
T0.18-0.25: 左轮着地 + 重心回正
T0.25:     重心左倾 → 右腿即将抬
...交替
```

### 3.2 关键参数

- 摆动时间 ~0.09s（原 0.25s，压缩 3 倍）
- 重心转移幅值 scale×1.5（髋侧倾）
- 双支撑占比 ~50%（sin 阈值 0.4）

---

## 四、调试技巧

```bash
# 验证 actuator 类型
uv run python -c "
import mujoco
m = mujoco.MjModel.from_xml_path('src/unilab/assets/robots/xqrobotV2/xqrobotV2.xml')
for i in range(m.nu):
    name = mujoco.mj_id2name(m, 5, i)  # 5 = mjOBJ_ACTUATOR
    t = m.actuator_trntype[i]
    print(f'  [{i}] {name}: type={t}')
"

# 物理验证：正轮速 → 前进
uv run python -c "
import mujoco, numpy as np
m = mujoco.MjModel.from_xml_path('src/unilab/assets/robots/xqrobotV2/scene_flat.xml')
d = mujoco.MjData(m)
d.ctrl[3] = d.ctrl[7] = 10   # 速度控制: ctrl[3]=L_wheel, ctrl[7]=R_wheel
for _ in range(200): mujoco.mj_step(m, d)
print('vx:', d.sensor('local_linvel').data[0], '(应为正)')
"

# 验证 Hydra 配置
uv run python -c "
from hydra import compose, initialize_config_dir
from pathlib import Path
with initialize_config_dir(version_base='1.3', config_dir=str(Path.cwd()/'conf'/'ppo')):
    cfg = compose(config_name='config', overrides=['task=xqrobotV2_walk_flat/mujoco'])
    print('vel_limit:', cfg.env.commands.vel_limit)
    print('num_envs:', cfg.algo.num_envs)
    print('ctrl_dt:', cfg.env.control_config)
"
```

---

## 五、参考项目对比

| 参数 | CJ-003 | HumanoidSW2 | PAI | XqRobotV2 |
|------|--------|-------------|-----|-----------|
| 框架 | Genesis | Isaac Gym | Isaac Gym | UniLab (MuJoCo) |
| DOF | 8 (轮腿) | 12 (双足) | 12 (纯下肢) | 8 (轮腿) |
| 轮控制 | 速度, kv=5 | — | — | 速度, kv=1 |
| ctrl_dt | 0.01 | 0.02 | 0.02 | 0.01 |
| 默认髋 | 0.0 | — | 0.0 (内置蹲) | 0.1 |
| 步态 | — | 正弦跟踪 | 正弦跟踪 | 正弦 + 重心转移 |
| 能量约束 | — | — | — | — |

详见 `docs/references/`

---

## 六、依赖

```bash
uv sync --extra mujoco
# mujoco-uni==3.8.0
```

## 七、问题记录

详见 **[docs/PROBLEMS.md](PROBLEMS.md)** (15 个问题)
