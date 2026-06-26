"""xqrobotV2 toe-walk env: point-foot walking on flat terrain.

Differs from joystick walking by encouraging an alternating leg-lifting gait
where wheels act as feet (intermittent ground contact), not as continuous rollers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from unilab.base import registry
from unilab.base.np_env import NpEnvState
from unilab.dtype_config import get_global_dtype
from unilab.envs.locomotion.common import rewards
from unilab.envs.locomotion.common.commands import Commands
from unilab.envs.locomotion.common.rewards import RewardContext

from .joystick import (
    XqRobotCurriculumConfig,
    XqRobotDRProvider,
    XqRobotV2WalkFlatCfg,
    XqRobotV2WalkFlatEnv,
)
from .base import NUM_LEG_ACTIONS, NUM_WHEEL_ACTIONS


@dataclass
class XqRobotToeWalkCommands(Commands):
    vel_limit: list[list[float]] = field(
        default_factory=lambda: [[-0.5, -0.2, -0.8, -0.1], [0.5, 0.2, 0.8, 0.1]]
    )
    resampling_time: float = 6.0


@dataclass
class XqRobotToeWalkRewardConfig:
    scales: dict[str, float]
    tracking_sigma: float = 0.25
    base_height_target: float = 0.65
    only_positive_rewards: bool = False
    max_tilt_deg: float = 45.0
    min_base_height: float = 0.15


@dataclass
class XqRobotToeWalkCurriculumConfig(XqRobotCurriculumConfig):
    enabled: bool = True
    vel_step: float = 0.0005
    ang_vel_step: float = 0.001
    err_threshold: float = 0.4


def _reward_step_pattern(ctx: RewardContext) -> np.ndarray:
    wheel_contact = ctx.info.get("wheel_contact", np.zeros((ctx.num_envs, 2)))
    left = wheel_contact[:, 0]
    right = wheel_contact[:, 1]
    alternating = left * (1.0 - right) + right * (1.0 - left)
    return alternating * 0.5


def _reward_wheel_lift(ctx: RewardContext) -> np.ndarray:
    wheel_contact = ctx.info.get("wheel_contact", np.ones((ctx.num_envs, 2)))
    air = 1.0 - np.mean(wheel_contact, axis=1)
    return air * 0.3


def _reward_static_wheel(ctx: RewardContext) -> np.ndarray:
    wheel_vel = ctx.dof_vel[:, -NUM_WHEEL_ACTIONS:]
    speed = np.sqrt(np.sum(np.square(wheel_vel), axis=1))
    stationary = 1.0 / (1.0 + speed)
    return stationary * 0.2


@registry.envcfg("XqRobotV2ToeWalkFlat")
@dataclass
class XqRobotV2ToeWalkFlatCfg(XqRobotV2WalkFlatCfg):
    commands: XqRobotToeWalkCommands = field(default_factory=XqRobotToeWalkCommands)
    reward_config: XqRobotToeWalkRewardConfig | None = None
    curriculum: XqRobotToeWalkCurriculumConfig = field(
        default_factory=XqRobotToeWalkCurriculumConfig
    )
    max_episode_seconds: float = 15.0


class XqRobotToeWalkDRProvider(XqRobotDRProvider):
    def _sample_commands(self, env: Any, num_reset: int) -> np.ndarray:
        low = np.asarray(env._cfg.commands.vel_limit[0], dtype=get_global_dtype())
        high = np.asarray(env._cfg.commands.vel_limit[1], dtype=get_global_dtype())
        cmds = np.asarray(
            np.random.uniform(low=low, high=high, size=(num_reset, low.shape[0])),
            dtype=get_global_dtype(),
        )
        safe_linv = np.maximum(np.abs(cmds[:, 0]), 1e-4)
        angv_limit = 2.0 / safe_linv
        cmds[:, 2] = np.clip(cmds[:, 2], -angv_limit, angv_limit)
        return cmds


@registry.env("XqRobotV2ToeWalkFlat", sim_backend="mujoco")
class XqRobotV2ToeWalkFlatEnv(XqRobotV2WalkFlatEnv):
    _cfg: XqRobotV2ToeWalkFlatCfg

    def __init__(self, cfg: XqRobotV2ToeWalkFlatCfg, num_envs=1, backend_type="mujoco"):
        self._toe_cfg = cfg.reward_config
        super().__init__(cfg, num_envs=num_envs, backend_type=backend_type)
        self._dr_manager._provider = XqRobotToeWalkDRProvider()

    def _init_reward_functions(self) -> None:
        self._reward_fns: dict[str, Any] = {
            "tracking_lin_vel": rewards.tracking_lin_vel,
            "tracking_ang_vel": rewards.tracking_ang_vel,
            "lin_vel_z": rewards.lin_vel_z,
            "ang_vel_xy": rewards.ang_vel_xy,
            "base_height": rewards.base_height,
            "orientation": rewards.orientation,
            "joint_action_rate": self._reward_joint_action_rate,
            "wheel_action_rate": self._reward_wheel_action_rate,
            "leg_mirror": self._reward_leg_mirror,
            "tsk": self._reward_tsk,
            "alive": rewards.alive,
            "step_pattern": _reward_step_pattern,
            "wheel_lift": _reward_wheel_lift,
            "static_wheel": _reward_static_wheel,
        }

    def _reward_joint_action_rate(self, ctx: RewardContext) -> np.ndarray:
        current = ctx.info["current_actions"][:, :NUM_LEG_ACTIONS]
        last = ctx.info["last_actions"][:, :NUM_LEG_ACTIONS]
        return np.sum(np.square(current - last), axis=1)

    def _reward_wheel_action_rate(self, ctx: RewardContext) -> np.ndarray:
        current = ctx.info["current_actions"][:, NUM_LEG_ACTIONS:]
        last = ctx.info["last_actions"][:, NUM_LEG_ACTIONS:]
        return np.sum(np.square(current - last), axis=1)

    def _reward_leg_mirror(self, ctx: RewardContext) -> np.ndarray:
        hip_error = ctx.dof_pos[:, 0] + ctx.dof_pos[:, 3]
        pitch_error = ctx.dof_pos[:, 1:3] - ctx.dof_pos[:, 4:6]
        return np.square(hip_error) + np.sum(np.square(pitch_error), axis=1)

    def _reward_tsk(self, ctx: RewardContext) -> np.ndarray:
        tsk_cmd = ctx.info["commands"][:, 3]
        hip_diff = ctx.dof_pos[:, 0] + ctx.dof_pos[:, 3]
        return np.square(hip_diff - tsk_cmd)

    def update_state(self, state: NpEnvState) -> NpEnvState:
        self._update_commands(state.info)
        linvel = self.get_local_linvel()
        gyro = self.get_gyro()
        gravity = self._backend.get_sensor_data(self._cfg.sensor.upvector)
        dof_pos = self.get_dof_pos()
        dof_vel = self.get_dof_vel()
        self._update_wheel_contact(state.info)
        terminated = self._compute_terminated(gravity, dof_pos)
        reward = self._compute_reward(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        obs = self._compute_obs(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        self._update_curriculum(state.info)
        return state.replace(obs=obs, reward=reward, terminated=terminated)

    def _update_wheel_contact(self, info: dict) -> None:
        try:
            left = self._backend.get_sensor_data("left_link_wheel_force")
            right = self._backend.get_sensor_data("right_link_wheel_force")
            left_f = np.asarray(left, dtype=get_global_dtype())
            right_f = np.asarray(right, dtype=get_global_dtype())
            if left_f.ndim == 1:
                left_f = left_f.reshape(-1, 3)
            if right_f.ndim == 1:
                right_f = right_f.reshape(-1, 3)
            left_contact = (np.linalg.norm(left_f, axis=1) > 0.1).astype(np.float64)[: self._num_envs]
            right_contact = (np.linalg.norm(right_f, axis=1) > 0.1).astype(np.float64)[: self._num_envs]
            info["wheel_contact"] = np.stack([left_contact, right_contact], axis=1)
        except (KeyError, AttributeError):
            info["wheel_contact"] = np.zeros((self._num_envs, 2), dtype=np.float64)

    def _compute_terminated(self, gravity: np.ndarray, dof_pos: np.ndarray) -> np.ndarray:
        tilt = np.arccos(np.clip(gravity[:, 2], -1, 1))
        max_tilt = np.deg2rad(self._toe_cfg.max_tilt_deg)
        terminated = tilt > max_tilt
        base_height = np.asarray(self._backend.get_base_pos(), dtype=get_global_dtype())[:, 2]
        terminated |= base_height < self._toe_cfg.min_base_height
        thigh_collapsed = (dof_pos[:, 1] < 0.02) | (dof_pos[:, 4] < 0.02)
        calf_extreme = (np.abs(dof_pos[:, 2]) > 0.85) | (np.abs(dof_pos[:, 5]) > 0.85)
        terminated |= thigh_collapsed
        terminated |= calf_extreme
        return terminated

    def _compute_reward(self, info: dict, linvel, gyro, gravity, dof_pos, dof_vel) -> np.ndarray:
        dtype = get_global_dtype()
        num_obs = linvel.shape[0]
        ctx = RewardContext(
            info=info,
            linvel=linvel,
            gyro=gyro,
            dof_pos=dof_pos[:, :NUM_LEG_ACTIONS],
            dof_vel=dof_vel,
            num_envs=num_obs,
            default_angles=self.default_angles[:NUM_LEG_ACTIONS].astype(dtype),
            tracking_sigma=self._toe_cfg.tracking_sigma,
            base_height_target=self._toe_cfg.base_height_target,
            base_height=self._base_height_values(num_obs),
            gravity=gravity,
            joint_range=None,
        )
        return rewards.run_reward_dispatch(
            scales=self._toe_cfg.scales,
            fns=self._reward_fns,
            ctx=ctx,
            info=info,
            enable_log=self._enable_reward_log,
            ctrl_dt=self._cfg.ctrl_dt,
            only_positive=self._toe_cfg.only_positive_rewards,
        )

    def _base_height_values(self, num_obs: int) -> np.ndarray:
        base_pos = np.asarray(self._backend.get_base_pos(), dtype=get_global_dtype())
        if base_pos.shape[0] != num_obs:
            return np.zeros((num_obs,), dtype=get_global_dtype())
        return np.asarray(base_pos[:, 2], dtype=get_global_dtype())
