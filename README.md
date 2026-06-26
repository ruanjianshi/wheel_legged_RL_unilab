# XqRobotV2 两轮腿机器人 RL 训练项目文档

> 基于 UniLab 框架。
> 最后更新：2026-06-26，iter 3212/5000

---

## 一、项目结构

```
UniLab/
├── conf/ppo/task/xqrobotV2_walk_flat/
│   ├── mujoco.yaml          # ★ 主训练配置
│   └── motrix.yaml          # Motrix 后端配置
├── src/unilab/envs/locomotion/xqrobotV2/
│   ├── base.py              # 机器人基础定义（关节、默认角度）
│   └── joystick.py          # ★ 环境实现（奖励、观测、命令、课程）
├── src/unilab/assets/robots/xqrobotV2/
│   ├── xqrobotV2.xml        # ★ MuJoCo 模型（关节轴、范围、执行器、传感器）
│   └── scene_flat.xml       # 场景 + keyframe
├── shell/
│   ├── train/xqrobotV2_ppo.sh   # 训练启动
│   └── eval/xqrobotV2_play.sh   # 验证（macOS→viser, Linux→原生窗口）
└── docs/
    └── PROJECT_ARCHITECTURE.md  # 本文件
```

---

## 二、机器人物理参数

### 2.1 尺寸规格

| 部件 | 长度 |
|------|------|
| 大腿 (left_joint_2 / right_joint_2) | 300mm |
| 小腿 (left_joint_3 / right_joint_3) | 300mm |
| 轮子直径 | 200mm |

### 2.2 关节定义 (xqrobotV2.xml)

| 关节 | 类型 | 轴 | 范围 | 默认角度 | kp |
|------|------|-----|------|---------|-----|
| left_joint_1 (左髋) | hinge | `1 0 0` | `[-π, π]` | 0.1 | 30 |
| left_joint_2 (左大腿) | hinge | `0 1 0` | `[0, 2.09]` | 0.1 | 30 |
| left_joint_3 (左小腿) | hinge | `0 1 0` | `[-0.87, 0.87]` | -0.1 | 30 |
| left_joint_wheel (左轮) | hinge | `0 1 0` | 无限 | 0 | 3 |
| right_joint_1 (右髋) | hinge | `1 0 0` | `[-π, π]` | 0.1 | 30 |
| right_joint_2 (右大腿) | hinge | `0 1 0` | `[0, 2.09]` | 0.1 | 30 |
| right_joint_3 (右小腿) | hinge | `0 1 0` | `[-0.87, 0.87]` | -0.1 | 30 |
| right_joint_wheel (右轮) | hinge | `0 1 0` | 无限 | 0 | 3 |

**关键设计决策**：
- 所有关节轴左/右**完全一致**（参考 CJ-003 设计），不做镜像
- 大腿范围 `min=0`（参考设计），防止后弯塌陷
- 不再使用髋关节镜像代码（早期尝试过，自相矛盾，已移除）
- 轮子 kp=3（kp=5 导致跺脚，已回退）
- 无辅助轮（已删除 4 个 miniwheel）

### 2.3 PD 参数

| 参数 | 腿关节 | 轮子 |
|------|--------|------|
| kp | 30 | 3 |
| kv | —（已从 XML 移除） | — |
| joint damping | 0.5 | 0.5 |
| armature | 0.002 | 0.002 |
| frictionloss | 0.01 | 0.01 |

**注意**：`kv` 属性在 mujoco-uni 3.8.0 中会导致执行器丢失，已从所有 `<position>` 中移除。MuJoCo 后端假设全部为 position actuator，不支持速度控制。

---

## 三、动作空间

8 维连续动作 `[-1, 1]`，映射到关节目标：

| 索引 | 关节 | 缩放 | 目标公式 |
|------|------|------|---------|
| 0 | 左髋 | ×0.5 | `action * 0.5 + 0.1` |
| 1 | 左大腿 | ×0.5 | `action * 0.5 + 0.1` |
| 2 | 左小腿 | ×0.5 | `action * 0.5 - 0.1` |
| 3 | 右髋 | ×0.5 | `action * 0.5 + 0.1` |
| 4 | 右大腿 | ×0.5 | `action * 0.5 + 0.1` |
| 5 | 右小腿 | ×0.5 | `action * 0.5 - 0.1` |
| 6 | 左轮 | ×10.0 | `action * 10.0 + 0` |
| 7 | 右轮 | ×10.0 | `action * 10.0 + 0` |

---

## 四、观测空间

### 4.1 单帧观测（32 维）

| 特征 | 维度 | 说明 |
|------|------|------|
| gyro | 3 | 角速度 |
| -gravity | 3 | 重力方向（取反） |
| leg_diff | 6 | 关节角度 - 默认角度 |
| leg_vel | 6 | 关节速度 |
| wheel_vel | 2 | 轮子速度 |
| last_actions | 8 | 上一步动作 |
| commands | 4 | [vx, vy, ω, tsk] |

### 4.2 历史堆叠

- **9 帧历史**（参考 CJ-003 设计）
- obs 总维度：32 × 9 = **288**
- critic 总维度：35 × 9 = **315**（多 3 维 linvel）
- 实现：滚动缓冲区，重置时填充所有帧

---

## 五、奖励函数

| 奖励 | 权重 | 类型 | 说明 |
|------|------|------|------|
| tracking_lin_vel | +1.5 | exp | 线速度跟踪 `exp(-error²/0.3)` |
| tracking_ang_vel | +1.5 | exp | 角速度跟踪 `exp(-error²/0.3)` |
| lin_vel_z | -0.2 | error² | z 轴速度惩罚 |
| ang_vel_xy | -0.02 | error² | roll/pitch 角速度惩罚 |
| base_height | **-5.0** | error² | 高度偏差惩罚 `(z-0.65)²` |
| orientation | **-5.0** | error² | 倾斜惩罚 `gravity_xy²` |
| joint_action_rate | -0.1 | error² | 腿动作平滑 |
| wheel_action_rate | -0.015 | error² | 轮动作平滑 |
| similar_calf | -0.3 | error² | 小腿对称 |
| tsk | -2.0 | error² | 髋差动命令跟踪 `(hip_L - hip_R - cmd)²` |
| alive | +1.0 | 常量 | 存活奖励 |

**关键迭代**：
- `orientation: -14.0` 和 `base_height: -10.0` 过重导致"不动不出错"→ 降至 -5.0
- `only_positive_rewards` 目前为 false；参考用 true 裁剪负奖励

---

## 六、命令系统（4D）

| 索引 | 含义 | 初始范围 | 全范围 |
|------|------|---------|--------|
| 0 | 前向速度 vx | [0, 0.6] | [0, 2.0] |
| 1 | 侧向速度 vy | [0, 0] | [0, 0] |
| 2 | 角速度 ω | [0, 1.0] | [0, 12.0] |
| 3 | tsk 髋差动 | [-0.1, 0.1] | [-0.1, 0.1] |

**逆比约束**（参考 CJ-003 high_speed 模式）：
```python
angv_max = 2.0 / |lin_vel_x|  # 线速度越低，角速度上限越小
```

---

## 七、课程学习

| 参数 | 值 | 说明 |
|------|-----|------|
| vel_step | 0.001 | 线速度扩展步长（比例） |
| ang_vel_step | 0.002 | 角速度扩展步长 |
| min_vel_range_frac | 0.3 | 初始 30% 全范围 |
| min_ang_range_frac | 0.05 | 初始 5% 全范围 |
| update_interval | 25 | 每 25 步评估 |
| err_threshold | 0.35 | 跟踪误差阈值 |

逻辑：平均跟踪误差 < 阈值 → 扩展速度范围；tracking_ang_vel 优秀 → 角速度扩得快 → **线速度被落下**（已知问题）。

---

## 八、触地终止

| 条件 | 阈值 |
|------|------|
| 倾斜 > 60° | `tilt > max_tilt` |
| 底座高度 < 0.20m | `base_height < min_base_height` |
| 大腿过直（塌陷） | `thigh < 0.02 rad` |
| 小腿极限角 | `abs(calf) > 0.85 rad` |

---

## 九、域随机化

代码已备（`XqRobotDomainRandConfig`），**当前关闭**，稳定后开启。支持：质量偏移、COM 偏移、地面摩擦、KP/KD 随机。

---

## 十、PPO 训练超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| num_envs | 4096 | 参考 8192 |
| num_steps_per_env | 25 | 参考 25 |
| max_iterations | 10000 | 当前训练 5000 |
| learning_rate | 1e-4 | 参考 1e-4 |
| entropy_coef | 0.01 | 参考 0.01 |
| gamma | 0.99 | |
| lam | 0.95 | |
| clip_param | 0.2 | |
| desired_kl | 0.01 | |
| init_noise_std | 0.5 | 从 1.5 降低 |
| network | [512,512,256,128] + elu | |
| ctrl_dt | 0.02 (50Hz) | 参考 0.01 (100Hz) |
| episode_length | 20s | |

---

## 十一、训练指标趋势（iter 3212）

| 指标 | 值 | 状态 |
|------|-----|------|
| tracking_ang_vel | 1.29/1.5 | ✅ 优秀 |
| tracking_lin_vel | 0.51/1.5 | ⚠️ 课程扩速快于学习 |
| base_height | -0.30 | 🔄 改善中 |
| orientation | -0.07 | ✅ |
| lin_vel_z | -0.01 | ✅ |
| ang_vel_xy | -0.03 | ✅ |
| episode length | 848/1000 | ✅ |

---

## 十二、已知问题 & 修复历史

| 问题 | 根因 | 修复 |
|------|------|------|
| 机器人自转 | 角速度命令独立采样，快转命令多 | 加逆比约束 angv = 2.0/linv |
| 线速度起不来 | orientation/base_height 惩罚过重 | -14→-5, -10→-5 |
| 髋关节不对称 | 左右轴相同但镜像代码自相矛盾 | 移除全部髋镜像，参考设计 |
| 小腿碰地/站不起来 | 默认 squat 太深 (thigh=0.4) | 改为微曲站立 (thigh=0.1) |
| 轮子跺脚 | kp=5 力矩尖峰 | kp→3 |
| 右腿 actuator 丢失 | XML kv 属性不兼容 | 移除 kv，恢复 actuator |
| 辅助轮干扰自平衡 | 4 个 miniwheel 提供额外支撑 | 删除辅助轮 |
| value loss 爆炸 | base_height 函数改为 error² | 权重从正改负 |
| konular 扩速过快 | ang_vel 跟踪好→课程扩到全范围 | 降低 ang_vel 初始范围 |
| macOS 无原生 viewer | mujoco-uni 不含 .app bundle | 用 viser，无键盘控制 |

---

## 十三、调试技巧

```bash
# 验证关节轴和范围对称性
uv run python -c "
import mujoco
m = mujoco.MjModel.from_xml_path('src/unilab/assets/robots/xqrobotV2/xqrobotV2.xml')
for i in range(m.njnt):
    name = mujoco.mj_id2name(m, 3, i)
    if name and 'joint' in name:
        print(f'{name:25s} axis={m.jnt_axis[i]} range={m.jnt_range[i]}')
"

# 验证执行器数量
uv run python -c "
import mujoco
m = mujoco.MjModel.from_xml_path('src/unilab/assets/robots/xqrobotV2/xqrobotV2.xml')
print('actuators:', m.nu)
"

# 验证 Hydra 配置
uv run python -c "
from hydra import compose, initialize_config_dir
from pathlib import Path
with initialize_config_dir(version_base='1.3', config_dir=str(Path.cwd()/'conf'/'ppo')):
    cfg = compose(config_name='config', overrides=['task=xqrobotV2_walk_flat/mujoco'])
    print('task_name:', cfg.training.task_name)
    print('obs_groups:', cfg.algo.obs_groups)
    print('vel_limit:', cfg.env.commands.vel_limit)
"
```

---

## 十四、依赖注意事项

```bash
# macOS: 必须同时安装这两个 extra
uv sync --extra mujoco --extra viser

# mujoco-uni 和 mujoco 不能共存（会冲突）
# 当前使用 mujoco-uni==3.8.0
```

---

## 十五、训练命令

```bash
# 训练
bash shell/train/xqrobotV2_ppo.sh

# 查看训练曲线
bash shell/eval/xqrobotV2_tb.sh

# 验证（macOS→viser, Linux→原生窗口）
bash shell/eval/xqrobotV2_play.sh <run_id>
```

---

---

## 十六、XqRobotV2 功能总结 (2026-06-26)

### 四个训练任务

| 任务 | Shell 脚本 | 命令 | 地形 | 说明 |
|------|-----------|------|------|------|
| **Flat Walk** | `train_ppo_flat.sh` | 5D | 平面 | 基础轮腿行走 |
| **Rough Walk** | `train_ppo_rough.sh` | 5D | 程序化 6×6 网格 | random_rough 40% + wave 40% + slope 各 10% |
| **Jump Flat** | `train_ppo_jump_flat.sh` | 5D | 平面 | 下蹲→爆发→腾空→着陆 |
| **Toe Walk Flat** | `train_ppo_toe_walk_flat.sh` | 4D | 平面 | 轮子当脚，交替抬腿 |

### 5D 命令向量

```
[vx, vy, vyaw, tsk, height_target]
```

### 键盘控制 (play_interactive.py)

| 按键 | 功能 |
|------|------|
| ↑/↓ | 前进/后退 (vx) |
| ←/→ | 左转/右转 (vyaw) |
| A/D | 左移/右移 (vy) |
| Q/E | 降低/升高 机身高度 |
| Enter | 刹停 |
| Backspace | 重置 |
| 空格 | 暂停/恢复 |

### 已修复的关键问题

| 问题 | 根因 | 修复 |
|------|------|------|
| **左右腿一前一后** | `similar_calf` 只约束小腿，髋/大腿无对称 | `similar_calf` 改为覆盖髋+大腿+小腿：`(hip_L+hip_R)² + (thigh_L-thigh_R)² + (calf_L-calf_R)²` |
| **机身向一侧倾斜** | 双髋同向 roll，无镜像约束 | 髋关节 `axis="1 0 0"` 同向 → 正确镜像为 `left+right≈0` |
| **keyframe-Python 不一致** | XML 右髋=0，Python 右髋=0.1 | Python 改为 0.0，与 keyframe 对齐 |
| **课程只向正方向扩展** | `full_low_x=0.0` 硬编码 | 对称扩展 `low←max(low-step, -limit)`, `high←min(high+step, limit)` |
| **vel_limit 4D→5D 维度** | 新增 height_target 第五维 | `obs_frame_dim=33, critic=36`，config 和代码同步 |
| **KeyboardCommander 崩溃** | 要求 shape (2,3)，XqRobotV2 是 (2,4) | 改为接受 (2,K) K≥3，只写前 3 列 |
| **feet_distance 轮距约束** | 无约束，策略可能过度内收/外展 | 跟踪左右轮 Y 坐标差，惩罚偏离 [0.3, 0.6]m |

### 文件结构

```
src/unilab/envs/locomotion/xqrobotV2/
├── base.py       # DEFAULT_LEG_ANGLES, 工具函数
├── joystick.py   # Flat Walk env
├── rough.py      # Rough Walk env
├── jump.py       # Jump Flat env
├── toe_walk.py   # Toe Walk Flat env
└── __init__.py

conf/ppo/task/
├── xqrobotV2_walk_flat/mujoco.yaml
├── xqrobotV2_walk_rough/mujoco.yaml
├── xqrobotV2_jump_flat/mujoco.yaml
└── xqrobotV2_toe_walk_flat/mujoco.yaml

shell/
├── train_ppo_flat.sh / train_ppo_rough.sh
├── train_ppo_jump_flat.sh / train_ppo_toe_walk_flat.sh
├── eval_ppo_flat.sh / eval_ppo_rough.sh
├── eval_ppo_jump_flat.sh / eval_ppo_toe_walk_flat.sh
└── tensorboard.sh

assets/robots/xqrobotV2/
└── locomotion_task.xml   # keyframe fragment (地形模式)
```

### 训练命令

```bash
# 单 GPU
bash shell/train_ppo_flat.sh
bash shell/train_ppo_rough.sh
bash shell/train_ppo_jump_flat.sh
bash shell/train_ppo_toe_walk_flat.sh

# 多 GPU (2×RTX4090)
CUDA_VISIBLE_DEVICES=0 bash shell/train_ppo_flat.sh &
CUDA_VISIBLE_DEVICES=1 bash shell/train_ppo_rough.sh &

# TensorBoard
bash shell/tensorboard.sh          # all
bash shell/tensorboard.sh flat 8080

# 键盘验证
bash shell/eval_ppo_flat.sh --keyboard
bash shell/eval_ppo_rough.sh --keyboard
```
