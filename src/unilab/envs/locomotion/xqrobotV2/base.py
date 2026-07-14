"""xqrobotV2 环境基类 — 轮腿双足机器人 (6 腿关节 + 2 轮关节)

定义机器人的关节命名约定、默认姿态、控制参数和观测传感器。
所有任务变体 (walk/jump/rough/toe) 均从此基类派生。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np

from unilab.envs.locomotion.common.base import (
    BaseNoiseConfig,
    LocomotionBaseCfg,
    LocomotionBaseEnv,
)

# ═══ 关节命名约定 ═══
# 与 xqrobotV2.xml 中的 actuator name 严格一致
# 顺序: [左腿3关节, 右腿3关节, 左轮, 右轮]
JOINT_PREFIXES: tuple[str, ...] = (
    "left_joint_1",    # 左髋 (hip roll)  — axis=(1,0,0)
    "left_joint_2",    # 左大腿 (thigh pitch) — axis=(0,1,0)
    "left_joint_3",    # 左小腿 (calf pitch) — axis=(0,1,0)
    "right_joint_1",   # 右髋 (hip roll)
    "right_joint_2",   # 右大腿 (thigh pitch)
    "right_joint_3",   # 右小腿 (calf pitch)
    "left_joint_wheel",  # 左轮 (速度控制)
    "right_joint_wheel", # 右轮 (速度控制)
)
# 仅腿部 6 关节 (位置控制, kp=30, 不含轮子)
LEG_PREFIXES: tuple[str, ...] = JOINT_PREFIXES[:6]
NUM_LEG_ACTIONS = len(LEG_PREFIXES)     # 6
NUM_WHEEL_ACTIONS = 2                    # 2 (速度控制, kv=1)
NUM_ACTIONS = len(JOINT_PREFIXES)        # 8 = 6 腿 + 2 轮

# ═══ 默认站立姿态 ═══
# 双髋对称外展: L_hip=-0.1(左腿向外) R_hip=+0.1(右腿向外) → 形成对称支撑面
# 臀/膝微屈: thigh=+0.1(前屈) calf=-0.1(后伸) → 轻微 crouch 站姿
# hip 必须非零, 否则 XqRobotV2 无法获得横向稳定性 (hip=0 直接塌陷)
# 轮子默认角 = 0 (静止)
DEFAULT_LEG_ANGLES = np.array([-0.1, 0.1, -0.1, 0.1, 0.1, -0.1], dtype=np.float64)
DEFAULT_WHEEL_ANGLES = np.zeros(NUM_WHEEL_ACTIONS, dtype=np.float64)
DEFAULT_ANGLES = np.concatenate([DEFAULT_LEG_ANGLES, DEFAULT_WHEEL_ANGLES])


# ═══ 配置数据类 ═══


@dataclass
class XqRobotNoiseConfig(BaseNoiseConfig):
    """观测噪声配置 — 继承通用噪声参数, 追加轮子速度噪声"""
    scale_wheel_vel: float = 0.5        # 轮子速度观测噪声标准差


@dataclass
class XqRobotControlConfig:
    """动作控制参数

    action_scale: 腿部动作缩放因子 (策略输出 × scale + default → 关节目标角)
    wheel_action_scale: 轮子动作缩放因子 (策略输出 × scale → 轮子目标速度)
    clip_actions: 策略输出裁剪阈值 [-clip, +clip], 防止野值
    simulate_action_latency: 是否模拟 1 步动作延迟 (训练稳定性 vs 现实精度权衡)
    """
    action_scale: float = 0.25
    wheel_action_scale: float = 10.0    # 轮子用速度控制, 需要大 scale (m/s 量级)
    clip_actions: float = 1.0
    simulate_action_latency: bool = False
    action_smoothing: float = 0.0


@dataclass
class XqRobotSensor:
    """IMU 传感器名称 — 对应 xqrobotV2.xml 中 <sensor> 标签的 name 属性"""
    local_linvel: str = "local_linvel"   # 机体坐标系线速度 (framelinvel)
    gyro: str = "gyro"                   # 角速度 (gyro, 安装在 imu site)
    upvector: str = "upvector"           # 重力方向 (framezaxis = body z 轴在世界系投影)


@dataclass
class XqRobotAsset:
    """机器人资源标识 — 用于后端定位 base_link 等关键 body"""
    base_name: str = "base_link"


@dataclass
class XqRobotBaseCfg(LocomotionBaseCfg):
    """xqrobotV2 基类配置 — 继承通用 LocomotionBaseCfg

    关键物理参数:
    - sim_dt=0.005: 物理仿真步长 200Hz (MuJoCo 内部子步)
    - ctrl_dt=0.01:  控制频率 100Hz (每 2 个物理步发一次动作)
    
    子类通过 override scene/model_file 切换场景, 
    通过 override commands 切换任务命令格式。
    """
    noise_config: XqRobotNoiseConfig = field(default_factory=XqRobotNoiseConfig)  # type: ignore[assignment]
    control_config: XqRobotControlConfig = field(default_factory=XqRobotControlConfig)  # type: ignore[assignment]
    sensor: XqRobotSensor = field(default_factory=XqRobotSensor)  # type: ignore[assignment]
    asset: XqRobotAsset = field(default_factory=XqRobotAsset)
    sim_dt: float = 0.005          # 200Hz 物理步长
    ctrl_dt: float = 0.01          # 100Hz 控制频率
    num_observations: int = 8      # 被 obs_groups_spec 覆盖, 历史遗留字段

    # 以下字段供 go2-like 地形环境使用, 平地环境保持默认即可
    env_spacing: float = 0.0       # 多 env 间距 (平地不适用)
    terrain: dict = field(default_factory=dict)       # 地形配置 (rough env 覆盖)
    measure_heights: bool = False  # 高度扫描 (rough env 覆盖)


# ═══ 传感器读取辅助函数 ═══


def stack_joint_sensors(backend, *, dtype: np.dtype | type) -> np.ndarray:
    """从后端读取所有 8 个关节位置传感器, 返回 (num_envs, 8) 数组

    传感器命名: {joint_name}_pos, 例如 left_joint_1_pos → 左髋当前位置
    """
    names = tuple(f"{p}_pos" for p in JOINT_PREFIXES)
    values = backend.get_sensor_data_batch(names)
    return np.asarray(values.reshape(values.shape[0], -1)[:, :NUM_ACTIONS], dtype=dtype)


def stack_joint_vel_sensors(backend, *, dtype: np.dtype | type) -> np.ndarray:
    """从后端读取所有 8 个关节速度传感器, 返回 (num_envs, 8) 数组

    传感器命名: {joint_name}_vel, 例如 left_joint_wheel_vel → 左轮当前角速度
    """
    names = tuple(f"{p}_vel" for p in JOINT_PREFIXES)
    values = backend.get_sensor_data_batch(names)
    return np.asarray(values.reshape(values.shape[0], -1)[:, :NUM_ACTIONS], dtype=dtype)


# ═══ 环境基类 ═══


class XqRobotBaseEnv(LocomotionBaseEnv):
    """xqrobotV2 所有任务的抽象基类

    职责:
    - 定义动作空间: Box(-1, 1) 8 维
    - 提供 dof_pos/dof_vel 统一接口 (封装后端传感器读取)
    
    子类必须实现: apply_action, update_state, _compute_obs, _compute_reward
    """
    _cfg: XqRobotBaseCfg

    def _init_action_space(self) -> None:
        """动作空间: 8 维连续, 每维 ∈ [-1, 1]

        - 索引 0-5: 腿关节 (position control, 策略输出 × scale + default)
        - 索引 6-7: 轮关节 (velocity control, 策略输出 × wheel_scale)
        """
        self._action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(NUM_ACTIONS,),
            dtype=np.float32,
        )

    def get_dof_pos(self) -> np.ndarray:
        """读取所有 8 关节当前位置 → (num_envs, 8)"""
        return stack_joint_sensors(self._backend, dtype=self.default_angles.dtype)

    def get_dof_vel(self) -> np.ndarray:
        """读取所有 8 关节当前速度 → (num_envs, 8)"""
        return stack_joint_vel_sensors(self._backend, dtype=self.default_angles.dtype)
