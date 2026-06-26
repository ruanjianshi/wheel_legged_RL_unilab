"""smallHumanoidRobot joystick (flat terrain) env: walk with velocity commands."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.backend import create_backend
from unilab.base.np_env import NpEnvState
from unilab.base.scene import SceneCfg
from unilab.dr import DomainRandomizationCapabilities, ResetPlan, ResetRandomizationPayload
from unilab.dr.dr_utils import (
    build_interval_push_plan,
    validate_interval_push_support,
    zero_actions,
)
from unilab.dtype_config import get_global_dtype
from unilab.envs.common.rotation import np_quat_mul, np_yaw_to_quat
from unilab.envs.locomotion.common import rewards
from unilab.envs.locomotion.common.commands import Commands
from unilab.envs.locomotion.common.domain_rand import DomainRandConfig
from unilab.envs.locomotion.common.dr_provider import LocomotionDRProvider
from unilab.envs.locomotion.common.rewards import RewardContext
from unilab.envs.locomotion.smallHumanoidRobot.base import (
    DEFAULT_ANGLES,
    NUM_ACTIONS,
    NUM_LEG_ACTIONS,
    SmallHumanoidBaseCfg,
    SmallHumanoidBaseEnv,
    stack_joint_sensors,
    stack_joint_vel_sensors,
)


@dataclass
class SmallHumanoidDomainRandConfig(DomainRandConfig):
    randomize_init_yaw: bool = True
    init_yaw_range: list[float] = field(default_factory=lambda: [-math.pi, math.pi])
    randomize_kp: bool = False
    kp_multiplier_range: list[float] = field(default_factory=lambda: [0.9, 1.1])
    randomize_kd: bool = False
    kd_multiplier_range: list[float] = field(default_factory=lambda: [0.9, 1.1])


@dataclass
class SmallHumanoidRewardConfig:
    scales: dict[str, float]
    tracking_sigma: float = 0.25
    base_height_target: float = 0.52
    only_positive_rewards: bool = False
    max_tilt_deg: float = 20.0
    min_base_height: float = 0.22
    pose_weights: list[float] | None = None


@registry.envcfg("SmallHumanoidWalkFlat")
@dataclass
class SmallHumanoidWalkFlatCfg(SmallHumanoidBaseCfg):
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "smallHumanoidRobot" / "scene_flat.xml")
        )
    )
    max_episode_seconds: float = 20.0
    commands: Commands = field(default_factory=Commands)
    reward_config: SmallHumanoidRewardConfig | None = None
    domain_rand: SmallHumanoidDomainRandConfig = field(default_factory=SmallHumanoidDomainRandConfig)


class SmallHumanoidDRProvider(LocomotionDRProvider):
    def build_reset_plan(self, env: Any, env_ids: np.ndarray) -> ResetPlan:
        num_reset = len(env_ids)
        qpos = np.tile(env._init_qpos, (num_reset, 1))
        qvel = np.tile(env._init_qvel, (num_reset, 1))
        qpos[:, 0:2] += np.random.uniform(-0.5, 0.5, (num_reset, 2))
        qpos[:, 0:3] += env._spawn.origins_for(env_ids)
        yaw = self._sample_reset_yaw(env, num_reset)
        qpos[:, 3:7] = np_quat_mul(qpos[:, 3:7], np_yaw_to_quat(yaw))
        commands = self._sample_commands(env, num_reset)
        info_updates: dict[str, Any] = {
            "commands": commands,
            "current_actions": zero_actions(num_reset, env._num_action),
            "last_actions": zero_actions(num_reset, env._num_action),
        }
        return ResetPlan(
            env_ids=env_ids,
            qpos=qpos,
            qvel=qvel,
            info_updates=info_updates,
            randomization=None,
        )

    def validate(self, env: Any, capabilities: DomainRandomizationCapabilities) -> None:
        validate_interval_push_support(env, capabilities)

    def build_interval_randomization_plan(self, env: Any, step_counter: int):
        return build_interval_push_plan(env, step_counter)

    def _compute_reset_obs(self, env, env_ids, info_updates, linvel, gyro, gravity, dof_pos, dof_vel):
        del env_ids
        return env._compute_obs(info_updates, linvel, gyro, gravity, dof_pos, dof_vel)

    def _sample_reset_yaw(self, env: Any, num_reset: int) -> np.ndarray:
        domain_rand = env._cfg.domain_rand
        if not domain_rand.randomize_init_yaw:
            return np.zeros((num_reset,), dtype=get_global_dtype())
        low, high = float(min(domain_rand.init_yaw_range)), float(max(domain_rand.init_yaw_range))
        return np.asarray(np.random.uniform(low, high, size=(num_reset,)), dtype=get_global_dtype())

    def _sample_commands(self, env: Any, num_reset: int) -> np.ndarray:
        low = np.asarray(env._cfg.commands.vel_limit[0], dtype=get_global_dtype())
        high = np.asarray(env._cfg.commands.vel_limit[1], dtype=get_global_dtype())
        return np.asarray(np.random.uniform(low=low, high=high, size=(num_reset, 3)), dtype=get_global_dtype())


@registry.env("SmallHumanoidWalkFlat", sim_backend="mujoco")
class SmallHumanoidWalkFlatEnv(SmallHumanoidBaseEnv):
    _cfg: SmallHumanoidWalkFlatCfg

    def __init__(self, cfg: SmallHumanoidWalkFlatCfg, num_envs=1, backend_type="mujoco"):
        if cfg.reward_config is None:
            raise ValueError("reward_config must be provided via Hydra configuration")
        backend = create_backend(
            backend_type,
            cfg.scene,
            num_envs,
            cfg.sim_dt,
            base_name=cfg.asset.base_name,
            push_body_name=getattr(cfg.domain_rand, "push_body_name", None),
            motrix_max_iterations=cfg.motrix_max_iterations,
            post_step_forward_sensor=cfg.post_step_forward_sensor,
        )
        super().__init__(cfg, backend, num_envs)
        self._np_dtype = get_global_dtype()
        self._reward_cfg = cfg.reward_config
        self._enable_reward_log = True
        self._init_reward_functions()
        self._init_domain_randomization(SmallHumanoidDRProvider())

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        # gyro(3) + gravity(3) + joint_diff(20) + dof_vel(20) + last_actions(20) + cmd(3) = 69
        # critic = obs(69) + linvel(3) = 72
        return {"obs": 69, "critic": 72}

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        clipped_actions = np.asarray(
            np.clip(actions, -self._cfg.control_config.clip_actions, self._cfg.control_config.clip_actions),
            dtype=self._np_dtype,
        )
        state.info["last_actions"] = state.info.get("current_actions", np.zeros_like(clipped_actions))
        state.info["current_actions"] = clipped_actions
        exec_actions = (
            state.info["last_actions"] if self._cfg.control_config.simulate_action_latency else clipped_actions
        )
        return np.asarray(exec_actions * self._cfg.control_config.action_scale + self.default_angles, dtype=self._np_dtype)

    def update_state(self, state: NpEnvState) -> NpEnvState:
        self._update_commands(state.info)
        linvel = self.get_local_linvel()
        gyro = self.get_gyro()
        gravity = self._backend.get_sensor_data(self._cfg.sensor.upvector)
        dof_pos = self.get_dof_pos()
        dof_vel = self.get_dof_vel()
        terminated = self._compute_terminated(gravity)
        reward = self._compute_reward(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        obs = self._compute_obs(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        return state.replace(obs=obs, reward=reward, terminated=terminated)

    def _compute_terminated(self, gravity: np.ndarray) -> np.ndarray:
        base_height = np.asarray(self._backend.get_base_pos(), dtype=get_global_dtype())[:, 2]
        tilt = np.arccos(np.clip(gravity[:, 2], -1, 1))
        max_tilt = np.deg2rad(self._reward_cfg.max_tilt_deg)
        return np.logical_or(tilt > max_tilt, base_height < self._reward_cfg.min_base_height)

    def _init_reward_functions(self) -> None:
        self._reward_fns: dict[str, Any] = {
            "tracking_lin_vel": rewards.tracking_lin_vel,
            "tracking_ang_vel": rewards.tracking_ang_vel,
            "lin_vel_z": rewards.lin_vel_z,
            "ang_vel_xy": rewards.ang_vel_xy,
            "base_height": rewards.base_height,
            "orientation": rewards.orientation,
            "action_rate": rewards.action_rate,
            "weighted_pose": rewards.weighted_pose,
            "alive": rewards.alive,
        }

    def _compute_reward(self, info: dict, linvel, gyro, gravity, dof_pos, dof_vel) -> np.ndarray:
        dtype = get_global_dtype()
        num_obs = linvel.shape[0]
        pose_weights = None
        if self._reward_cfg.pose_weights is not None:
            pose_weights = np.asarray(self._reward_cfg.pose_weights, dtype=dtype)
        ctx = RewardContext(
            info=info,
            linvel=linvel,
            gyro=gyro,
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            num_envs=num_obs,
            default_angles=self.default_angles.astype(dtype),
            tracking_sigma=self._reward_cfg.tracking_sigma,
            base_height_target=self._reward_cfg.base_height_target,
            base_height=self._base_height_values(num_obs),
            gravity=gravity,
            joint_range=None,
            pose_weights=pose_weights,
        )
        return rewards.run_reward_dispatch(
            scales=self._reward_cfg.scales,
            fns=self._reward_fns,
            ctx=ctx,
            info=info,
            enable_log=self._enable_reward_log,
            ctrl_dt=self._cfg.ctrl_dt,
            only_positive=self._reward_cfg.only_positive_rewards,
        )

    def _update_commands(self, info: dict) -> None:
        commands = info.get("commands")
        if commands is None:
            return
        commands_arr = np.asarray(commands, dtype=get_global_dtype())
        resampling_time = float(getattr(self._cfg.commands, "resampling_time", 0.0))
        if resampling_time > 0.0:
            interval_steps = max(int(round(resampling_time / self._cfg.ctrl_dt)), 1)
            steps = np.asarray(info.get("steps", np.zeros((self._num_envs,), dtype=np.uint32)))
            resample_mask = (steps > 0) & ((steps % interval_steps) == 0)
            if np.any(resample_mask):
                num_resample = int(np.count_nonzero(resample_mask))
                low = np.asarray(self._cfg.commands.vel_limit[0], dtype=get_global_dtype())
                high = np.asarray(self._cfg.commands.vel_limit[1], dtype=get_global_dtype())
                sampled = np.random.uniform(low=low, high=high, size=(num_resample, 3)).astype(get_global_dtype())
                commands_arr[resample_mask] = sampled
        info["commands"] = commands_arr

    def _compute_obs(self, info: dict, linvel, gyro, gravity, dof_pos, dof_vel) -> dict[str, np.ndarray]:
        noise_cfg = self._cfg.noise_config
        joint_diff = dof_pos - self.default_angles
        noisy_gyro = self._obs_noise(gyro, noise_cfg.scale_gyro)
        noisy_gravity = self._obs_noise(gravity, noise_cfg.scale_gravity)
        noisy_joint_diff = self._obs_noise(joint_diff, noise_cfg.scale_joint_angle)
        noisy_joint_vel = self._obs_noise(dof_vel, noise_cfg.scale_joint_vel)
        last_actions = info.get("current_actions", np.zeros((linvel.shape[0], NUM_ACTIONS)))

        obs = np.concatenate([
            noisy_gyro,
            -noisy_gravity,
            noisy_joint_diff,
            noisy_joint_vel,
            last_actions,
            info["commands"],
        ], axis=1, dtype=get_global_dtype())

        critic = np.concatenate([
            gyro,
            -gravity,
            joint_diff,
            dof_vel,
            last_actions,
            info["commands"],
            linvel,
        ], axis=1, dtype=get_global_dtype())
        return {"obs": obs, "critic": critic}

    def _base_height_values(self, num_obs: int) -> np.ndarray:
        base_pos = np.asarray(self._backend.get_base_pos(), dtype=get_global_dtype())
        if base_pos.shape[0] != num_obs:
            return np.zeros((num_obs,), dtype=get_global_dtype())
        return np.asarray(base_pos[:, 2], dtype=get_global_dtype())


registry.register_env("SmallHumanoidWalkFlat", SmallHumanoidWalkFlatEnv, sim_backend="motrix")
