"""smallHumanoidRobot base env: humanoid robot with 10 leg + 10 arm joints (20 total)."""

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
    "left_leg_joint_1",
    "left_leg_joint_2",
    "left_leg_joint_3",
    "left_leg_joint_4",
    "left_leg_joint_5",
    "right_leg_joint_1",
    "right_leg_joint_2",
    "right_leg_joint_3",
    "right_leg_joint_4",
    "right_leg_joint_5",
    "left_arm_joint_1",
    "left_arm_joint_2",
    "left_arm_joint_3",
    "left_arm_joint_4",
    "left_arm_joint_5",
    "right_arm_joint_1",
    "right_arm_joint_2",
    "right_arm_joint_3",
    "right_arm_joint_4",
    "right_arm_joint_5",
)
NUM_LEG_ACTIONS = 10
NUM_ARM_ACTIONS = 10
NUM_ACTIONS = len(JOINT_PREFIXES)

# Squat stance + arms hanging at side
# Legs: [L_j1 L_j2 L_j3 L_j4 L_j5  R_j1 R_j2 R_j3 R_j4 R_j5]
# Arms: [L_a1 L_a2 L_a3 L_a4 L_a5  R_a1 R_a2 R_a3 R_a4 R_a5]
DEFAULT_ANGLES = np.array([
    0.15, 0.0, -0.4, 0.6, -0.2,      # left leg
    -0.15, 0.0, -0.4, 0.6, -0.2,     # right leg
    0.0, 0.0, 0.0, 0.0, 0.0,         # left arm (hanging down)
    0.0, 0.0, 0.0, 0.0, 0.0,         # right arm (hanging down)
], dtype=np.float64)


@dataclass
class SmallHumanoidNoiseConfig(BaseNoiseConfig):
    scale_joint_angle: float = 0.01
    scale_joint_vel: float = 1.5


@dataclass
class SmallHumanoidControlConfig:
    action_scale: float = 0.25
    clip_actions: float = 1.0
    simulate_action_latency: bool = False


@dataclass
class SmallHumanoidSensor:
    local_linvel: str = "local_linvel"
    gyro: str = "gyro"
    upvector: str = "upvector"


@dataclass
class SmallHumanoidAsset:
    base_name: str = "base_link"


@dataclass
class SmallHumanoidBaseCfg(LocomotionBaseCfg):
    noise_config: SmallHumanoidNoiseConfig = field(default_factory=SmallHumanoidNoiseConfig)  # type: ignore[assignment]
    control_config: SmallHumanoidControlConfig = field(default_factory=SmallHumanoidControlConfig)  # type: ignore[assignment]
    sensor: SmallHumanoidSensor = field(default_factory=SmallHumanoidSensor)  # type: ignore[assignment]
    asset: SmallHumanoidAsset = field(default_factory=SmallHumanoidAsset)
    sim_dt: float = 0.005
    ctrl_dt: float = 0.02
    num_observations: int = 20
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


class SmallHumanoidBaseEnv(LocomotionBaseEnv):
    _cfg: SmallHumanoidBaseCfg

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
