"""xqrobotV2 joystick (flat terrain) env: walk with velocity commands.

Features: contact termination, history stacking, curriculum, domain randomization.
"""

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
from unilab.dr import DomainRandomizationCapabilities, ResetPlan
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
from unilab.envs.locomotion.xqrobotV2.base import (
    DEFAULT_ANGLES,
    NUM_ACTIONS,
    NUM_LEG_ACTIONS,
    NUM_WHEEL_ACTIONS,
    XqRobotBaseCfg,
    XqRobotBaseEnv,
    stack_joint_sensors,
    stack_joint_vel_sensors,
)

# History config
_HISTORY_LEN = 9  # 参考 9 帧历史堆叠


@dataclass
class XqRobotDomainRandConfig(DomainRandConfig):
    randomize_init_yaw: bool = True
    init_yaw_range: list[float] = field(default_factory=lambda: [-math.pi, math.pi])
    # 域随机化
    randomize_base_mass: bool = True
    added_mass_range: list[float] = field(default_factory=lambda: [-1.5, 1.5])
    randomize_ground_friction: bool = True
    ground_friction_multiplier_range: list[float] = field(default_factory=lambda: [0.2, 1.6])
    randomize_kp: bool = True
    kp_multiplier_range: list[float] = field(default_factory=lambda: [0.8, 1.2])
    randomize_kd: bool = True
    kd_multiplier_range: list[float] = field(default_factory=lambda: [0.8, 1.2])
    random_com: bool = True
    com_offset_x: float = 0.03
    com_offset_y: float = 0.03
    randomize_gravity: bool = False
    push_robots: bool = False
    randomize_leg_length: bool = False
    leg_length_scale_range: list[float] = field(default_factory=lambda: [0.8, 1.2])


@dataclass
class XqRobotCurriculumConfig:
    enabled: bool = True
    vel_step: float = 0.001           # 线速度课程步长
    ang_vel_step: float = 0.002
    min_vel_range_frac: float = 0.3
    min_ang_range_frac: float = 0.05
    update_interval: int = 25
    err_threshold: float = 0.35


@dataclass
class XqRobotRewardConfig:
    scales: dict[str, float]
    tracking_sigma: float = 0.25
    base_height_target: float = 0.65
    only_positive_rewards: bool = True
    max_tilt_deg: float = 60.0
    min_base_height: float = 0.20


# ── 自定义奖励函数 ─────────────────────────────────────────────────


def _reward_joint_action_rate(ctx: RewardContext) -> np.ndarray:
    current = ctx.info["current_actions"][:, :NUM_LEG_ACTIONS]
    last = ctx.info["last_actions"][:, :NUM_LEG_ACTIONS]
    return np.sum(np.square(current - last), axis=1)


def _reward_wheel_action_rate(ctx: RewardContext) -> np.ndarray:
    current = ctx.info["current_actions"][:, NUM_LEG_ACTIONS:]
    last = ctx.info["last_actions"][:, NUM_LEG_ACTIONS:]
    return np.sum(np.square(current - last), axis=1)


def _reward_similar_calf(ctx: RewardContext) -> np.ndarray:
    # 髋: 镜像 left+right≈0;  大腿/小腿: 平行 left-right≈0
    hip = ctx.dof_pos[:, 0] + ctx.dof_pos[:, 3]
    thigh = ctx.dof_pos[:, 1] - ctx.dof_pos[:, 4]
    calf = ctx.dof_pos[:, 2] - ctx.dof_pos[:, 5]
    return np.square(hip) + np.square(thigh) + np.square(calf)


def _reward_wheel_symmetry(ctx: RewardContext) -> np.ndarray:
    # 直线行走时左右轮速应相等
    commands = ctx.info["commands"]
    turning = np.abs(commands[:, 2]) > 0.1  # vyaw > 0.1 = 转弯, 不惩罚
    wheel_actions = ctx.info["current_actions"][:, -2:]
    diff = np.square(wheel_actions[:, 0] - wheel_actions[:, 1])
    return diff * (1.0 - turning.astype(np.float64)) * 0.5


def _reward_tsk(ctx: RewardContext) -> np.ndarray:
    tsk_cmd = ctx.info["commands"][:, 3]
    hip_diff = ctx.dof_pos[:, 0] - ctx.dof_pos[:, 3]
    return np.square(hip_diff - tsk_cmd)


def _reward_feet_distance(ctx: RewardContext) -> np.ndarray:
    feet_dist = ctx.info.get("feet_distance")
    if feet_dist is None:
        return np.zeros((ctx.num_envs,), dtype=np.float64)
    dist = np.asarray(feet_dist, dtype=np.float64).reshape(-1)
    over = np.maximum(0.0, dist - 0.6)
    under = np.maximum(0.0, 0.3 - dist)
    return (over + under) * 0.3


@registry.envcfg("XqRobotV2WalkFlat")
@dataclass
class XqRobotV2WalkFlatCfg(XqRobotBaseCfg):
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "xqrobotV2" / "scene_flat.xml")
        )
    )
    max_episode_seconds: float = 20.0
    commands: Commands = field(default_factory=Commands)
    reward_config: XqRobotRewardConfig | None = None
    domain_rand: XqRobotDomainRandConfig = field(default_factory=XqRobotDomainRandConfig)
    curriculum: XqRobotCurriculumConfig = field(default_factory=XqRobotCurriculumConfig)

    # 触地终止: 检测这些 body 是否触地
    contact_body_names: list[str] = field(default_factory=lambda: [
        "left_link_2", "left_link_3", "right_link_2", "right_link_3",
    ])


class XqRobotDRProvider(LocomotionDRProvider):
    def build_reset_plan(self, env: Any, env_ids: np.ndarray) -> ResetPlan:
        num_reset = len(env_ids)
        qpos = np.tile(env._init_qpos, (num_reset, 1))
        qvel = np.tile(env._init_qvel, (num_reset, 1))
        qpos[:, 0:2] += np.random.uniform(-0.5, 0.5, (num_reset, 2))
        qpos[:, 0:3] += env._spawn.origins_for(env_ids)
        yaw = self._sample_reset_yaw(env, num_reset)
        qpos[:, 3:7] = np_quat_mul(qpos[:, 3:7], np_yaw_to_quat(yaw))

        # 域随机化: 暂无, 后续可通过 backend payload 开启
        randomization = None

        commands = self._sample_commands(env, num_reset)
        info_updates: dict[str, Any] = {
            "commands": commands,
            "current_actions": zero_actions(num_reset, env._num_action),
            "last_actions": zero_actions(num_reset, env._num_action),
        }
        return ResetPlan(
            env_ids=env_ids, qpos=qpos, qvel=qvel,
            info_updates=info_updates, randomization=randomization,
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
        cmds = np.asarray(np.random.uniform(low=low, high=high, size=(num_reset, low.shape[0])), dtype=get_global_dtype())
        # 逆比: 线速度低时限制角速度, 防止原地打转
        safe_linv = np.maximum(np.abs(cmds[:, 0]), 1e-4)
        angv_limit = 2.0 / safe_linv  # inverse_linx_angv
        cmds[:, 2] = np.clip(cmds[:, 2], -angv_limit, angv_limit)
        return cmds


@registry.env("XqRobotV2WalkFlat", sim_backend="mujoco")
class XqRobotV2WalkFlatEnv(XqRobotBaseEnv):
    _cfg: XqRobotV2WalkFlatCfg

    def __init__(self, cfg: XqRobotV2WalkFlatCfg, num_envs=1, backend_type="mujoco"):
        if cfg.reward_config is None:
            raise ValueError("reward_config must be provided via Hydra configuration")
        backend = create_backend(
            backend_type, cfg.scene, num_envs, cfg.sim_dt,
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
        self._init_domain_randomization(XqRobotDRProvider())

        if cfg.domain_rand.randomize_leg_length:
            self._base_leg_geom_sizes = {
                name: backend.get_geom_size(name) for name in XqRobotDRProvider._LEG_GEOM_NAMES
            }

        import mujoco as _mj
        if hasattr(backend, "_model"):  # type: ignore[union-attr]
            self._left_wheel_bid = _mj.mj_name2id(backend._model, _mj.mjtObj.mjOBJ_BODY, "left_link_wheel")  # type: ignore[union-attr]
            self._right_wheel_bid = _mj.mj_name2id(backend._model, _mj.mjtObj.mjOBJ_BODY, "right_link_wheel")  # type: ignore[union-attr]
        else:
            self._left_wheel_bid = -1
            self._right_wheel_bid = -1

        # ── 历史堆叠 ──
        self._hist_len = _HISTORY_LEN
        self._obs_frame_dim = 33   # 5D cmd: gyro(3)+grav(3)+diff(6)+vel(6)+wheel(2)+act(8)+cmd(5)
        self._critic_frame_dim = 36  # 5D cmd + linvel(3)
        self._obs_history = np.zeros((num_envs, self._hist_len, self._obs_frame_dim), dtype=self._np_dtype)
        self._critic_history = np.zeros((num_envs, self._hist_len, self._critic_frame_dim), dtype=self._np_dtype)

        # ── 课程学习 ──
        self._curriculum_step_count = 0
        self._tracking_err_buf = np.zeros((num_envs,), dtype=self._np_dtype)

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {"obs": self._obs_frame_dim * self._hist_len, "critic": self._critic_frame_dim * self._hist_len}

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
        leg_targets = (
            exec_actions[:, :NUM_LEG_ACTIONS] * self._cfg.control_config.action_scale + DEFAULT_ANGLES[:NUM_LEG_ACTIONS]
        )
        wheel_targets = exec_actions[:, NUM_LEG_ACTIONS:] * self._cfg.control_config.wheel_action_scale  # velocity ctrl
        return np.concatenate([leg_targets, wheel_targets], axis=1, dtype=self._np_dtype)

    def update_state(self, state: NpEnvState) -> NpEnvState:
        self._update_commands(state.info)
        linvel = self.get_local_linvel()
        gyro = self.get_gyro()
        gravity = self._backend.get_sensor_data(self._cfg.sensor.upvector)
        dof_pos = self.get_dof_pos()
        dof_vel = self.get_dof_vel()
        self._update_feet_distance(state.info)
        terminated = self._compute_terminated(gravity, dof_pos)
        reward = self._compute_reward(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        obs = self._compute_obs(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        # 课程学习
        self._update_curriculum(state.info)
        return state.replace(obs=obs, reward=reward, terminated=terminated)

    def _compute_terminated(self, gravity: np.ndarray, dof_pos: np.ndarray) -> np.ndarray:
        base_height = np.asarray(self._backend.get_base_pos(), dtype=get_global_dtype())[:, 2]
        tilt = np.arccos(np.clip(gravity[:, 2], -1, 1))
        max_tilt = np.deg2rad(self._reward_cfg.max_tilt_deg)
        terminated = np.logical_or(tilt > max_tilt, base_height < self._reward_cfg.min_base_height)
        # 触地终止: 大腿后弯(过直)或小腿过度弯曲 = 塌陷
        thigh_collapsed = (dof_pos[:, 1] < 0.02) | (dof_pos[:, 4] < 0.02)
        calf_extreme = (np.abs(dof_pos[:, 2]) > 0.85) | (np.abs(dof_pos[:, 5]) > 0.85)
        terminated |= thigh_collapsed
        terminated |= calf_extreme
        return terminated

    def _update_feet_distance(self, info: dict) -> None:
        try:
            body_ids = np.array([self._left_wheel_bid, self._right_wheel_bid], dtype=np.int32)
            pos = self._backend.get_body_pos_w(body_ids)
            info["feet_distance"] = np.abs(pos[1, :, 1] - pos[0, :, 1]).astype(np.float64)
        except Exception:
            info["feet_distance"] = np.full((self._num_envs,), 0.45, dtype=np.float64)

    def _init_reward_functions(self) -> None:
        self._reward_fns: dict[str, Any] = {
            "tracking_lin_vel": rewards.tracking_lin_vel,
            "tracking_ang_vel": rewards.tracking_ang_vel,
            "lin_vel_z": rewards.lin_vel_z,
            "ang_vel_xy": rewards.ang_vel_xy,
            "base_height": rewards.base_height,
            "orientation": rewards.orientation,
            "joint_action_rate": _reward_joint_action_rate,
            "wheel_action_rate": _reward_wheel_action_rate,
            "similar_calf": _reward_similar_calf,
            "tsk": _reward_tsk,
            "wheel_symmetry": _reward_wheel_symmetry,
            "feet_distance": _reward_feet_distance,
            "alive": rewards.alive,
        }

    def _compute_reward(self, info: dict, linvel, gyro, gravity, dof_pos, dof_vel) -> np.ndarray:
        dtype = get_global_dtype()
        num_obs = linvel.shape[0]
        ctx = RewardContext(
            info=info, linvel=linvel, gyro=gyro,
            dof_pos=dof_pos[:, :NUM_LEG_ACTIONS],
            dof_vel=dof_vel[:, :NUM_LEG_ACTIONS],
            num_envs=num_obs,
            default_angles=DEFAULT_ANGLES[:NUM_LEG_ACTIONS].astype(dtype),
            tracking_sigma=self._reward_cfg.tracking_sigma,
            base_height_target=self._reward_cfg.base_height_target,
            base_height=self._base_height_values(num_obs),
            gravity=gravity, joint_range=None,
        )
        return rewards.run_reward_dispatch(
            scales=self._reward_cfg.scales, fns=self._reward_fns, ctx=ctx, info=info,
            enable_log=self._enable_reward_log, ctrl_dt=self._cfg.ctrl_dt,
            only_positive=self._reward_cfg.only_positive_rewards,
        )

    # ── 课程学习 ──────────────────────────────────────────────────

    def _update_curriculum(self, info: dict) -> None:
        cc = self._cfg.curriculum
        if not cc.enabled:
            return
        commands = info.get("commands")
        linvel = self.get_local_linvel()
        if commands is not None and linvel is not None:
            self._tracking_err_buf += np.abs(commands[:, 0] - linvel[:, 0])
        self._curriculum_step_count += 1
        if self._curriculum_step_count < cc.update_interval:
            return
        self._curriculum_step_count = 0
        ep_steps = info.get("steps", np.zeros((self._num_envs,)))
        mean_ep_len = ep_steps[ep_steps > 0].mean() if np.any(ep_steps > 0) else 0.0
        max_steps = int(self._cfg.max_episode_seconds / self._cfg.ctrl_dt)
        survive_ratio = mean_ep_len / max_steps if max_steps > 0 else 0.0
        if survive_ratio < 0.5:
            return
        active = ep_steps[ep_steps > 0]
        mean_err = self._tracking_err_buf[ep_steps > 0].mean() / active.mean() if len(active) > 0 else 999.0
        self._tracking_err_buf[:] = 0.0
        low = np.array(self._cfg.commands.vel_limit[0], dtype=self._np_dtype)
        high = np.array(self._cfg.commands.vel_limit[1], dtype=self._np_dtype)
        vx_range = max(abs(low[0]), abs(high[0]))
        vyaw_range = max(abs(low[2]), abs(high[2]))
        if mean_err < cc.err_threshold:
            low[0] = max(low[0] - cc.vel_step, -vx_range)
            high[0] = min(high[0] + cc.vel_step, vx_range)
            low[2] = max(low[2] - cc.ang_vel_step, -vyaw_range)
            high[2] = min(high[2] + cc.ang_vel_step, vyaw_range)
        self._cfg.commands.vel_limit[0] = low.tolist()
        self._cfg.commands.vel_limit[1] = high.tolist()

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
                sampled = np.random.uniform(low=low, high=high, size=(num_resample, low.shape[0])).astype(get_global_dtype())
                safe_linv = np.maximum(np.abs(sampled[:, 0]), 1e-4)
                angv_limit = 2.0 / safe_linv
                sampled[:, 2] = np.clip(sampled[:, 2], -angv_limit, angv_limit)
                commands_arr[resample_mask] = sampled
        info["commands"] = commands_arr

    # ── 观测 + 历史堆叠 ──────────────────────────────────────────

    def _compute_obs(self, info: dict, linvel, gyro, gravity, dof_pos, dof_vel) -> dict[str, np.ndarray]:
        noise_cfg = self._cfg.noise_config
        leg_diff = dof_pos[:, :NUM_LEG_ACTIONS] - DEFAULT_ANGLES[:NUM_LEG_ACTIONS]
        leg_vel = dof_vel[:, :NUM_LEG_ACTIONS]
        wheel_vel = dof_vel[:, NUM_LEG_ACTIONS:]
        noisy_gyro = self._obs_noise(gyro, noise_cfg.scale_gyro)
        noisy_gravity = self._obs_noise(gravity, noise_cfg.scale_gravity)
        noisy_leg_diff = self._obs_noise(leg_diff, noise_cfg.scale_joint_angle)
        noisy_leg_vel = self._obs_noise(leg_vel, noise_cfg.scale_joint_vel)
        noisy_wheel_vel = self._obs_noise(wheel_vel, noise_cfg.scale_wheel_vel)
        last_actions = info.get("current_actions", np.zeros((linvel.shape[0], NUM_ACTIONS)))

        obs_frame = np.concatenate([
            noisy_gyro, -noisy_gravity,
            noisy_leg_diff, noisy_leg_vel, noisy_wheel_vel,
            last_actions, info["commands"],
        ], axis=1, dtype=get_global_dtype())

        critic_frame = np.concatenate([
            gyro, -gravity,
            leg_diff, leg_vel, wheel_vel,
            last_actions, info["commands"], linvel,
        ], axis=1, dtype=get_global_dtype())

        batch_size = obs_frame.shape[0]
        steps_val = int(info.get("steps", np.zeros(1, dtype=np.uint32))[0])

        if steps_val <= 1:
            # reset: 填满所有历史帧 (仅对当前 batch)
            for i in range(self._hist_len):
                self._obs_history[:batch_size, i, :] = obs_frame
                self._critic_history[:batch_size, i, :] = critic_frame
        else:
            # step: 滚动缓冲区
            self._obs_history[:batch_size, :-1, :] = self._obs_history[:batch_size, 1:, :]
            self._obs_history[:batch_size, -1, :] = obs_frame
            self._critic_history[:batch_size, :-1, :] = self._critic_history[:batch_size, 1:, :]
            self._critic_history[:batch_size, -1, :] = critic_frame

        obs = self._obs_history[:batch_size].reshape(batch_size, -1)
        critic = self._critic_history[:batch_size].reshape(batch_size, -1)
        return {"obs": obs, "critic": critic}

    def _base_height_values(self, num_obs: int) -> np.ndarray:
        base_pos = np.asarray(self._backend.get_base_pos(), dtype=get_global_dtype())
        if base_pos.shape[0] != num_obs:
            return np.zeros((num_obs,), dtype=get_global_dtype())
        return np.asarray(base_pos[:, 2], dtype=get_global_dtype())


registry.register_env("XqRobotV2WalkFlat", XqRobotV2WalkFlatEnv, sim_backend="motrix")
