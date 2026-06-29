"""xqrobotV2 toe-walk env: sinusoidal trajectory tracking for bipedal stepping.

Uses a phase clock to generate reference joint trajectories, and the RL policy
learns to track them + add corrections. This is adapted from the HumanoidSW2
approach (livelybot_pi_rl_baseline).
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
    _reward_feet_distance,
    _reward_wheel_symmetry,
    _reward_hip_roll,
)
from .base import DEFAULT_LEG_ANGLES, NUM_LEG_ACTIONS, NUM_WHEEL_ACTIONS

_HISTORY_LEN = 9


@dataclass
class XqRobotToeWalkCommands(Commands):
    vel_limit: list[list[float]] = field(
        default_factory=lambda: [[-0.3, -0.1, -0.3, -0.1], [0.3, 0.1, 0.3, 0.1]]
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
    cycle_time: float = 0.5
    ref_scale: float = 0.15


def _reward_ref_tracking(ctx: RewardContext) -> np.ndarray:
    ref = ctx.info.get("ref_dof_pos")
    if ref is None:
        return np.zeros((ctx.num_envs,), dtype=np.float64)
    err = np.sum(np.square(ctx.dof_pos[:, :NUM_LEG_ACTIONS] - ref), axis=1)
    return np.exp(-2.0 * err) * 1.2 - 0.2 * err


def _reward_static_wheel(ctx: RewardContext) -> np.ndarray:
    wheel_vel = ctx.dof_vel[:, -NUM_WHEEL_ACTIONS:]
    speed = np.sqrt(np.sum(np.square(wheel_vel), axis=1))
    return 1.0 / (1.0 + speed)


@registry.envcfg("XqRobotV2ToeWalkFlat")
@dataclass
class XqRobotV2ToeWalkFlatCfg(XqRobotV2WalkFlatCfg):
    commands: XqRobotToeWalkCommands = field(default_factory=XqRobotToeWalkCommands)
    reward_config: XqRobotToeWalkRewardConfig | None = None
    curriculum: XqRobotCurriculumConfig = field(default_factory=lambda: XqRobotCurriculumConfig(enabled=False))
    max_episode_seconds: float = 12.0


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
        # Phase clock: random offset so half envs start with left leading, half with right
        self._phase_offset = np.random.uniform(0, 2 * np.pi, (num_envs,)).astype(np.float64)
        self._ref_dof_pos = np.zeros((num_envs, NUM_LEG_ACTIONS), dtype=np.float64)
        # Override obs dims: add sin/cos phase (2 dims), 4D commands
        self._obs_frame_dim = 34  # 32(base 4D) + 2 = 34
        self._critic_frame_dim = 37  # 35(base 4D) + 2 = 37
        self._obs_history = np.zeros((num_envs, self._hist_len, self._obs_frame_dim), dtype=self._np_dtype)
        self._critic_history = np.zeros((num_envs, self._hist_len, self._critic_frame_dim), dtype=self._np_dtype)

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
            "ref_tracking": _reward_ref_tracking,
            "static_wheel": _reward_static_wheel,
            "feet_distance": _reward_feet_distance,
            "wheel_symmetry": _reward_wheel_symmetry,
            "hip_roll": _reward_hip_roll,
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
        hip = ctx.dof_pos[:, 0] + ctx.dof_pos[:, 3]
        thigh = ctx.dof_pos[:, 1] - ctx.dof_pos[:, 4]
        calf = ctx.dof_pos[:, 2] - ctx.dof_pos[:, 5]
        return np.square(hip) + np.square(thigh) + np.square(calf)

    def _reward_tsk(self, ctx: RewardContext) -> np.ndarray:
        tsk_cmd = ctx.info["commands"][:, 3]
        hip_diff = ctx.dof_pos[:, 0] - ctx.dof_pos[:, 3]
        return np.square(hip_diff - tsk_cmd)

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        clipped = np.clip(actions, -self._cfg.control_config.clip_actions, self._cfg.control_config.clip_actions)
        state.info["last_actions"] = state.info.get("current_actions", np.zeros_like(clipped))
        state.info["current_actions"] = clipped
        exec_actions = state.info["last_actions"] if self._cfg.control_config.simulate_action_latency else clipped
        # Leg: correction on top of reference trajectory
        leg_corr = exec_actions[:, :NUM_LEG_ACTIONS] * self._cfg.control_config.action_scale
        leg_targets = self._ref_dof_pos[: leg_corr.shape[0]] + leg_corr
        # Wheel: velocity control, no offset (stays near 0 for toe-walk)
        wheel_targets = exec_actions[:, NUM_LEG_ACTIONS:] * self._cfg.control_config.wheel_action_scale * 0.2
        return np.concatenate([leg_targets, wheel_targets], axis=1, dtype=self._np_dtype)

    def _compute_ref_dof_pos(self, info: dict) -> None:
        cycle_time = self._toe_cfg.cycle_time
        steps = info.get("steps", np.zeros((self._num_envs,), dtype=np.float64))
        dt = self._cfg.ctrl_dt
        phase = (steps * dt / cycle_time) + self._phase_offset / (2 * np.pi)
        sin_pos = np.sin(2 * np.pi * phase)[:, None]  # (num_envs, 1)

        scale = self._toe_cfg.ref_scale
        # Left leg active when sin < 0
        left_active = np.clip(-sin_pos, 0, None)
        # Right leg active when sin >= 0
        right_active = np.clip(sin_pos, 0, None)

        ref = np.zeros((self._num_envs, NUM_LEG_ACTIONS), dtype=np.float64)
        # Hip stays at default
        ref[:, 0] = DEFAULT_LEG_ANGLES[0]
        ref[:, 3] = DEFAULT_LEG_ANGLES[3]
        # Thigh: swing forward (positive from default 0.1)
        ref[:, 1] = DEFAULT_LEG_ANGLES[1] + left_active[:, 0] * scale * 2
        ref[:, 4] = DEFAULT_LEG_ANGLES[4] + right_active[:, 0] * scale * 2
        # Calf: flex during swing (more negative from default -0.1)
        ref[:, 2] = DEFAULT_LEG_ANGLES[2] - left_active[:, 0] * scale * 3
        ref[:, 5] = DEFAULT_LEG_ANGLES[5] - right_active[:, 0] * scale * 3

        self._ref_dof_pos = ref

    def update_state(self, state: NpEnvState) -> NpEnvState:
        self._compute_ref_dof_pos(state.info)
        state.info["ref_dof_pos"] = self._ref_dof_pos
        return super().update_state(state)

    def _compute_obs(self, info: dict, linvel, gyro, gravity, dof_pos, dof_vel) -> dict[str, np.ndarray]:
        noise_cfg = self._cfg.noise_config
        leg_diff = dof_pos[:, :NUM_LEG_ACTIONS] - DEFAULT_LEG_ANGLES[:NUM_LEG_ACTIONS]
        leg_vel = dof_vel[:, :NUM_LEG_ACTIONS]
        wheel_vel = dof_vel[:, NUM_LEG_ACTIONS:]
        noisy_gyro = self._obs_noise(gyro, noise_cfg.scale_gyro)
        noisy_gravity = self._obs_noise(-gravity, noise_cfg.scale_gravity)
        noisy_leg_diff = self._obs_noise(leg_diff, noise_cfg.scale_joint_angle)
        noisy_leg_vel = self._obs_noise(leg_vel, noise_cfg.scale_joint_vel)
        noisy_wheel_vel = self._obs_noise(wheel_vel, noise_cfg.scale_wheel_vel)
        last_actions = info.get("current_actions", np.zeros((linvel.shape[0], NUM_LEG_ACTIONS + NUM_WHEEL_ACTIONS)))

        obs_frame = np.concatenate([
            noisy_gyro, noisy_gravity,
            noisy_leg_diff, noisy_leg_vel, noisy_wheel_vel,
            last_actions, info["commands"],
        ], axis=1, dtype=get_global_dtype())

        critic_frame = np.concatenate([
            gyro, -gravity,
            leg_diff, leg_vel, wheel_vel,
            last_actions, info["commands"], linvel,
        ], axis=1, dtype=get_global_dtype())

        batch_size = obs_frame.shape[0]

        # Phase encoding
        steps_p = info.get("steps", np.zeros((batch_size,), dtype=np.float64))
        phase = (steps_p[:batch_size] * self._cfg.ctrl_dt / self._toe_cfg.cycle_time) + self._phase_offset[:batch_size] / (2 * np.pi)
        sin_phase = np.sin(2 * np.pi * phase)[:, None]
        cos_phase = np.cos(2 * np.pi * phase)[:, None]

        obs_frame = np.concatenate([obs_frame, sin_phase, cos_phase], axis=1, dtype=get_global_dtype())
        critic_frame = np.concatenate([critic_frame, sin_phase, cos_phase], axis=1, dtype=get_global_dtype())

        steps_val = int(info.get("steps", np.zeros(1, dtype=np.uint32))[0])

        if steps_val <= 1:
            for i in range(self._hist_len):
                self._obs_history[:batch_size, i, :] = obs_frame
                self._critic_history[:batch_size, i, :] = critic_frame
        else:
            self._obs_history[:batch_size, :-1, :] = self._obs_history[:batch_size, 1:, :]
            self._obs_history[:batch_size, -1, :] = obs_frame
            self._critic_history[:batch_size, :-1, :] = self._critic_history[:batch_size, 1:, :]
            self._critic_history[:batch_size, -1, :] = critic_frame

        obs = self._obs_history[:batch_size].reshape(batch_size, -1)
        critic = self._critic_history[:batch_size].reshape(batch_size, -1)
        return {"obs": obs, "critic": critic}

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
            default_angles=DEFAULT_LEG_ANGLES[:NUM_LEG_ACTIONS].astype(dtype),
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
