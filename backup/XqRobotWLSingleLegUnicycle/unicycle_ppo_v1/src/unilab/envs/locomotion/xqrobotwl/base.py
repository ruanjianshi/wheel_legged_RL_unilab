"""xqrobotwl base env: wheel-legged bipedal robot (6 leg + 2 wheel joints)."""

from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np

from unilab.envs.locomotion.common.base import (
    BaseNoiseConfig,
    LocomotionBaseCfg,
    LocomotionBaseEnv,
)

JOINT_PREFIXES: tuple[str, ...] = (
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "left_wheel",
    "right_wheel",
)
LEG_PREFIXES: tuple[str, ...] = JOINT_PREFIXES[:6]
NUM_LEG_ACTIONS = len(LEG_PREFIXES)
NUM_WHEEL_ACTIONS = 2
NUM_ACTIONS = len(JOINT_PREFIXES)

# 物理对称站立: 左右镜像轴需不对称默认角度
# L_hip_roll=外展(+0.1), L_hip_pitch=前倾(+0.15,轴+Y), L_knee=微弯(+0.15,轴-Y)
# R_hip_roll=外展(-0.1), R_hip_pitch=前倾(-0.15,轴-Y), R_knee=微弯(-0.15,轴+Y)
# sign_flip [1,1,-1,1,-1,1,1,-1] 在策略输出层保证对称动作→对称运动
DEFAULT_LEG_ANGLES = np.array([0.1, 0.15, 0.15, -0.1, -0.15, -0.15], dtype=np.float64)
DEFAULT_WHEEL_ANGLES = np.zeros(NUM_WHEEL_ACTIONS, dtype=np.float64)
DEFAULT_ANGLES = np.concatenate([DEFAULT_LEG_ANGLES, DEFAULT_WHEEL_ANGLES])


@dataclass
class XqRobotWLNoiseConfig(BaseNoiseConfig):
    scale_wheel_vel: float = 0.5


@dataclass
class XqRobotWLControlConfig:
    action_scale: float = 0.25
    wheel_action_scale: float = 10.0
    clip_actions: float = 1.0
    simulate_action_latency: bool = False
    action_smoothing: float = 0.0


@dataclass
class XqRobotWLSensor:
    local_linvel: str = "local_linvel"
    gyro: str = "gyro"
    upvector: str = "upvector"


@dataclass
class XqRobotWLAsset:
    base_name: str = "base_link"


@dataclass
class XqRobotWLBaseCfg(LocomotionBaseCfg):
    noise_config: XqRobotWLNoiseConfig = field(default_factory=XqRobotWLNoiseConfig)  # type: ignore[assignment]
    control_config: XqRobotWLControlConfig = field(default_factory=XqRobotWLControlConfig)  # type: ignore[assignment]
    sensor: XqRobotWLSensor = field(default_factory=XqRobotWLSensor)  # type: ignore[assignment]
    asset: XqRobotWLAsset = field(default_factory=XqRobotWLAsset)
    sim_dt: float = 0.005
    ctrl_dt: float = 0.01
    num_observations: int = 8

    # These are only for go2-like envs using terrain; keep defaults for flat env.
    env_spacing: float = 0.0
    terrain: dict = field(default_factory=dict)
    measure_heights: bool = False


def stack_joint_sensors(backend, *, dtype: np.dtype | type) -> np.ndarray:
    names = tuple(f"{p}_pos" for p in JOINT_PREFIXES)
    values = backend.get_sensor_data_batch(names)
    return np.asarray(values.reshape(values.shape[0], -1)[:, :NUM_ACTIONS], dtype=dtype)


def stack_joint_vel_sensors(backend, *, dtype: np.dtype | type) -> np.ndarray:
    names = tuple(f"{p}_vel" for p in JOINT_PREFIXES)
    values = backend.get_sensor_data_batch(names)
    return np.asarray(values.reshape(values.shape[0], -1)[:, :NUM_ACTIONS], dtype=dtype)


class XqRobotWLBaseEnv(LocomotionBaseEnv):
    _cfg: XqRobotWLBaseCfg

    def _init_action_space(self) -> None:
        self._action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(NUM_ACTIONS,),
            dtype=np.float32,
        )

    def get_dof_pos(self) -> np.ndarray:
        return stack_joint_sensors(self._backend, dtype=self.default_angles.dtype)

    def get_dof_vel(self) -> np.ndarray:
        return stack_joint_vel_sensors(self._backend, dtype=self.default_angles.dtype)
