"""xqrobotV2 base env: wheel-legged bipedal robot (6 leg + 2 wheel joints)."""

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
    "left_joint_1",
    "left_joint_2",
    "left_joint_3",
    "right_joint_1",
    "right_joint_2",
    "right_joint_3",
    "left_joint_wheel",
    "right_joint_wheel",
)
LEG_PREFIXES: tuple[str, ...] = JOINT_PREFIXES[:6]
NUM_LEG_ACTIONS = len(LEG_PREFIXES)
NUM_WHEEL_ACTIONS = 2
NUM_ACTIONS = len(JOINT_PREFIXES)

# Widened squat stance: [L_hip, L_thigh, L_knee, R_hip, R_thigh, R_knee]
# 更伸展的默认姿态: 接近站立, 方便策略学会站直
DEFAULT_LEG_ANGLES = np.array([0.1, 0.1, -0.1, 0.1, 0.1, -0.1], dtype=np.float64)
DEFAULT_WHEEL_ANGLES = np.zeros(NUM_WHEEL_ACTIONS, dtype=np.float64)
DEFAULT_ANGLES = np.concatenate([DEFAULT_LEG_ANGLES, DEFAULT_WHEEL_ANGLES])


@dataclass
class XqRobotNoiseConfig(BaseNoiseConfig):
    scale_wheel_vel: float = 0.5


@dataclass
class XqRobotControlConfig:
    action_scale: float = 0.25
    wheel_action_scale: float = 10.0
    clip_actions: float = 1.0
    simulate_action_latency: bool = False


@dataclass
class XqRobotSensor:
    local_linvel: str = "local_linvel"
    gyro: str = "gyro"
    upvector: str = "upvector"


@dataclass
class XqRobotAsset:
    base_name: str = "base_link"


@dataclass
class XqRobotBaseCfg(LocomotionBaseCfg):
    noise_config: XqRobotNoiseConfig = field(default_factory=XqRobotNoiseConfig)  # type: ignore[assignment]
    control_config: XqRobotControlConfig = field(default_factory=XqRobotControlConfig)  # type: ignore[assignment]
    sensor: XqRobotSensor = field(default_factory=XqRobotSensor)  # type: ignore[assignment]
    asset: XqRobotAsset = field(default_factory=XqRobotAsset)
    sim_dt: float = 0.005
    ctrl_dt: float = 0.02
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


class XqRobotBaseEnv(LocomotionBaseEnv):
    _cfg: XqRobotBaseCfg

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
