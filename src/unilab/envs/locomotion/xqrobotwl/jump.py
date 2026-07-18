"""xqrobotwl jump env: learn to jump via periodic height-triggered commands."""

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
from unilab.envs.locomotion.xqrobotwl.base import (
    NUM_LEG_ACTIONS,
    NUM_WHEEL_ACTIONS,
)

from .joystick import (
    XqRobotWLCurriculumConfig,
    XqRobotWLDRProvider,
    XqRobotWLWalkFlatCfg,
    XqRobotWLWalkFlatEnv,
)

_NUM_JUMP_CMD_DIM = 5


@dataclass
class XqRobotWLJumpCommands(Commands):
    vel_limit: list[list[float]] = field(
        default_factory=lambda: [[-0.3, -0.1, -0.5, -0.1, 0], [0.3, 0.1, 0.5, 0.1, 1]]
    )
    resampling_time: float = 4.0


@dataclass
class XqRobotWLJumpRewardConfig:
    scales: dict[str, float]
    tracking_sigma: float = 0.25
    base_height_target: float = 0.65
    only_positive_rewards: bool = False
    max_tilt_deg: float = 60.0
    min_base_height: float = 0.20
    jump_height_target: float = 1.0
    crouch_height_target: float = 0.40
    # 跳跃课程: [start, end] step 范围, 线性插值 jump_trigger + 跳跃奖励
    jump_curriculum_start: int = 0
    jump_curriculum_end: int = 100_000


@dataclass
class XqRobotWLJumpCurriculumConfig(XqRobotWLCurriculumConfig):
    enabled: bool = False


# ── 跳跃奖励 (phase-gated: 蹲→蹬→飞→落) ────────────────────────────
# jump_phase = consecutive steps with trigger > 0.5
# Phase  [1, 30]: crouch (下蹲蓄力, 30步后强制停)
# Phase  anytime: thrust (蹬地, 不锁窗, 蹲够了随时弹)
# Phase     >=1: flight/height/air (腾空奖励, 全程开)
# Phase    >=30: landing (落地缓冲)


def _reward_crouch_prep(ctx: RewardContext, jump_cfg: XqRobotWLJumpRewardConfig) -> np.ndarray:
    base_z = ctx.base_height
    phase = ctx.info.get("jump_phase", np.zeros(ctx.num_envs, dtype=np.float64))
    active = ((phase >= 1.0) & (phase <= 40.0)).astype(np.float64)
    crouching = base_z < jump_cfg.base_height_target
    target = jump_cfg.crouch_height_target
    height_ok = (base_z > jump_cfg.min_base_height) & (base_z < target + 0.15)
    # Posture: 前倾 + 弯膝 + 髋不外展 (禁止后仰 + 伸腿 + 叉腿)
    hip_fwd_L = ctx.dof_pos[:, 1]  # +Y axis: positive=forward
    hip_fwd_R = -ctx.dof_pos[:, 4]  # -Y axis: negative=forward, so -value=forward
    knee_bend_L = ctx.dof_pos[:, 2]  # -Y axis: positive=bent
    knee_bend_R = -ctx.dof_pos[:, 5]  # +Y axis: negative=bent, so -value=bent
    roll_ok = (np.abs(ctx.dof_pos[:, 0] - 0.1) < 0.12) & (np.abs(ctx.dof_pos[:, 3] + 0.1) < 0.12)
    posture_ok = ((hip_fwd_L > 0.1) & (hip_fwd_R > 0.1) & (knee_bend_L > 0.1) & (knee_bend_R > 0.1)
                  & roll_ok)
    weight = ctx.info.get("jump_curriculum", 1.0)
    return height_ok.astype(np.float64) * crouching.astype(np.float64) * posture_ok.astype(np.float64) * active * 0.5 * weight


def _reward_crouch_depth(ctx: RewardContext, jump_cfg: XqRobotWLJumpRewardConfig) -> np.ndarray:
    base_z = ctx.base_height
    phase = ctx.info.get("jump_phase", np.zeros(ctx.num_envs, dtype=np.float64))
    active = ((phase >= 1.0) & (phase <= 25.0)).astype(np.float64)
    crouching = base_z < jump_cfg.base_height_target
    depth = np.clip((jump_cfg.base_height_target - base_z) / 0.25, 0.0, 1.0)
    # Same posture check as crouch_prep: no backward lean + bent knees
    hip_fwd_L = ctx.dof_pos[:, 1]
    hip_fwd_R = -ctx.dof_pos[:, 4]
    knee_bend_L = ctx.dof_pos[:, 2]
    knee_bend_R = -ctx.dof_pos[:, 5]
    posture_ok = (hip_fwd_L > 0.1) & (hip_fwd_R > 0.1) & (knee_bend_L > 0.1) & (knee_bend_R > 0.1)
    weight = ctx.info.get("jump_curriculum", 1.0)
    return depth * crouching.astype(np.float64) * posture_ok.astype(np.float64) * active * 0.5 * weight


def _reward_stand_posture(ctx: RewardContext) -> np.ndarray:
    # When no trigger: reward staying at default posture + standing height
    trigger = ctx.info["commands"][:, 4]
    standing = (trigger <= 0.5).astype(np.float64)
    height_err = np.square(ctx.base_height - 0.65)
    dof_err = np.sum(np.square(ctx.dof_pos[:, :6] - ctx.default_angles[:6]), axis=1)
    act_mag = np.sum(np.square(ctx.info["current_actions"][:, :6]), axis=1)
    return -(height_err * 15.0 + dof_err * 3.0 + act_mag * 1.0) * standing


def _reward_lean_forward(ctx: RewardContext) -> np.ndarray:
    # 全局惩罚后仰: 时刻保持髋关节前倾
    hip_fwd_L = ctx.dof_pos[:, 1]
    hip_fwd_R = -ctx.dof_pos[:, 4]
    penalty_L = np.clip(-hip_fwd_L, 0, 1)
    penalty_R = np.clip(-hip_fwd_R, 0, 1)
    knee_bend_L = ctx.dof_pos[:, 2]
    knee_bend_R = -ctx.dof_pos[:, 5]
    p_knee_L = np.clip(-knee_bend_L, 0, 1)
    p_knee_R = np.clip(-knee_bend_R, 0, 1)
    # Penalize hip_roll abduction beyond ±0.2 from default (±0.1)
    roll_dev_L = np.clip(np.abs(ctx.dof_pos[:, 0] - 0.1) - 0.10, 0, 1)
    roll_dev_R = np.clip(np.abs(ctx.dof_pos[:, 3] + 0.1) - 0.10, 0, 1)
    return -(penalty_L + penalty_R + p_knee_L + p_knee_R + roll_dev_L + roll_dev_R) * 0.5


def _reward_vertical_thrust(ctx: RewardContext, jump_cfg: XqRobotWLJumpRewardConfig) -> np.ndarray:
    vz = ctx.linvel[:, 2]
    phase = ctx.info.get("jump_phase", np.zeros(ctx.num_envs, dtype=np.float64))
    wheel_contact = ctx.info.get("wheel_contact", np.ones((ctx.num_envs, 2)))
    on_ground = (np.max(wheel_contact, axis=1) > 0.5).astype(np.float64)
    active = ((phase >= 25.0) & (vz > 0.0) & (on_ground > 0)).astype(np.float64)
    vz_capped = np.clip(vz, 0, 2.0)
    weight = ctx.info.get("jump_curriculum", 1.0)
    return vz_capped * active * weight


def _reward_jump_height(ctx: RewardContext, jump_cfg: XqRobotWLJumpRewardConfig) -> np.ndarray:
    base_z = ctx.base_height
    phase = ctx.info.get("jump_phase", np.ones(ctx.num_envs, dtype=np.float64) * 1e9)
    active = (phase >= 1.0).astype(np.float64)
    target = jump_cfg.jump_height_target
    clamped = np.clip(base_z / target, 0.0, 1.0)
    wheel_contact = ctx.info.get("wheel_contact", np.ones((ctx.num_envs, 2)))
    air_factor = 1.0 - np.mean(wheel_contact, axis=1)
    weight = ctx.info.get("jump_curriculum", 1.0)
    return clamped * active * air_factor * 2.0 * weight


def _reward_wheel_air_time(ctx: RewardContext) -> np.ndarray:
    wheel_contact = ctx.info.get("wheel_contact", np.ones((ctx.num_envs, 2)))
    air = 1.0 - np.mean(wheel_contact, axis=1)
    phase = ctx.info.get("jump_phase", np.zeros(ctx.num_envs, dtype=np.float64))
    active = (phase >= 1.0).astype(np.float64)
    weight = ctx.info.get("jump_curriculum", 1.0)
    return air * 0.5 * weight * active


def _reward_landing_soft(ctx: RewardContext) -> np.ndarray:
    base_linvel_z = ctx.linvel[:, 2]
    vz_mag = np.abs(base_linvel_z)
    soft = np.exp(-vz_mag / 0.5)
    phase = ctx.info.get("jump_phase", np.zeros(ctx.num_envs, dtype=np.float64))
    active = (phase >= 30.0).astype(np.float64)
    weight = ctx.info.get("jump_curriculum", 1.0)
    return soft * 0.3 * weight * active


def _reward_anti_loiter(ctx: RewardContext) -> np.ndarray:
    # Penalize staying crouched after the crouch window closes (phase > 30)
    phase = ctx.info.get("jump_phase", np.zeros(ctx.num_envs, dtype=np.float64))
    base_z = ctx.base_height
    loitering = ((phase > 30.0) & (base_z < 0.55)).astype(np.float64)
    # Deeper crouch = bigger penalty → forces extension or jump
    penalty = np.clip((0.55 - base_z) / 0.3, 0.0, 2.0)
    weight = ctx.info.get("jump_curriculum", 1.0)
    return -penalty * loitering * weight


@registry.envcfg("XqRobotWLJumpFlat")
@dataclass
class XqRobotWLJumpFlatCfg(XqRobotWLWalkFlatCfg):
    commands: XqRobotWLJumpCommands = field(default_factory=XqRobotWLJumpCommands)
    reward_config: XqRobotWLJumpRewardConfig | None = None
    curriculum: XqRobotWLJumpCurriculumConfig = field(default_factory=XqRobotWLJumpCurriculumConfig)
    max_episode_seconds: float = 10.0


class XqRobotWLJumpDRProvider(XqRobotWLDRProvider):
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


@registry.env("XqRobotWLJumpFlat", sim_backend="mujoco")
class XqRobotWLJumpFlatEnv(XqRobotWLWalkFlatEnv):
    _cfg: XqRobotWLJumpFlatCfg

    def __init__(self, cfg: XqRobotWLJumpFlatCfg, num_envs=1, backend_type="mujoco"):
        self._jump_cfg = cfg.reward_config
        self._total_env_steps = 0
        self._jump_curriculum_start = getattr(cfg.reward_config, "jump_curriculum_start", 0)
        self._jump_curriculum_end = getattr(cfg.reward_config, "jump_curriculum_end", 100_000)
        range_span = self._jump_curriculum_end - self._jump_curriculum_start
        self._jump_curriculum_step = float(range_span) if range_span > 0 else 1.0
        super().__init__(cfg, num_envs=num_envs, backend_type=backend_type)
        self._dr_manager._provider = XqRobotWLJumpDRProvider()
        self._obs_frame_dim = 33
        self._critic_frame_dim = 36
        self._obs_history = np.zeros(
            (num_envs, self._hist_len, self._obs_frame_dim), dtype=self._np_dtype
        )
        self._critic_history = np.zeros(
            (num_envs, self._hist_len, self._critic_frame_dim), dtype=self._np_dtype
        )

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {
            "obs": self._obs_frame_dim * self._hist_len,
            "critic": self._critic_frame_dim * self._hist_len,
        }

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
            "jump_height": self._reward_jump_height,
            "crouch_prep": self._reward_crouch_prep,
            "landing_soft": self._reward_landing_soft,
            "wheel_air_time": self._reward_wheel_air_time,
            "vertical_thrust": self._reward_vertical_thrust,
            "crouch_depth": self._reward_crouch_depth,
            "anti_loiter": self._reward_anti_loiter,
            "lean_forward": self._reward_lean_forward,
            "lean_forward": self._reward_lean_forward,
        }

    def _reward_jump_height(self, ctx: RewardContext) -> np.ndarray:
        return _reward_jump_height(ctx, self._jump_cfg)

    def _reward_crouch_prep(self, ctx: RewardContext) -> np.ndarray:
        return _reward_crouch_prep(ctx, self._jump_cfg)

    def _reward_landing_soft(self, ctx: RewardContext) -> np.ndarray:
        return _reward_landing_soft(ctx)

    def _reward_wheel_air_time(self, ctx: RewardContext) -> np.ndarray:
        return _reward_wheel_air_time(ctx)

    def _reward_vertical_thrust(self, ctx: RewardContext) -> np.ndarray:
        return _reward_vertical_thrust(ctx, self._jump_cfg)

    def _reward_crouch_depth(self, ctx: RewardContext) -> np.ndarray:
        return _reward_crouch_depth(ctx, self._jump_cfg)

    def _reward_anti_loiter(self, ctx: RewardContext) -> np.ndarray:
        return _reward_anti_loiter(ctx)

    def _reward_lean_forward(self, ctx: RewardContext) -> np.ndarray:
        return _reward_lean_forward(ctx)

    def _reward_stand_posture(self, ctx: RewardContext) -> np.ndarray:
        return _reward_stand_posture(ctx)

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
        pitch_error = ctx.dof_pos[:, 1:3] + ctx.dof_pos[:, 4:6]
        return np.square(hip_error) + np.sum(np.square(pitch_error), axis=1)

    def _reward_tsk(self, ctx: RewardContext) -> np.ndarray:
        tsk_cmd = ctx.info["commands"][:, 3]
        hip_diff = ctx.dof_pos[:, 0] + ctx.dof_pos[:, 3]
        return np.square(hip_diff - tsk_cmd)

    def update_state(self, state: NpEnvState) -> NpEnvState:
        self._update_commands(state.info)
        # Jump curriculum: linear ramp jump_trigger + jump rewards
        self._total_env_steps += self._num_envs
        progress = np.clip(
            (self._total_env_steps - self._jump_curriculum_start) / self._jump_curriculum_step,
            0.0, 1.0,
        )
        state.info["commands"][:, 4] *= np.float64(progress)
        state.info["jump_curriculum"] = np.float64(progress)
        # Jump phase: consecutive steps with trigger > 0.5
        trigger_active = state.info["commands"][:, 4] > 0.5
        prev_phase = state.info.get("jump_phase", np.zeros(self._num_envs, dtype=np.float64))
        new_phase = np.where(trigger_active, prev_phase + 1, np.float64(0))
        state.info["jump_phase"] = new_phase
        linvel = self.get_local_linvel()
        gyro = self.get_gyro()
        gravity = self._backend.get_sensor_data(self._cfg.sensor.upvector)
        dof_pos = self.get_dof_pos()
        dof_vel = self.get_dof_vel()
        self._update_wheel_contact(state.info)
        terminated = self._compute_terminated(gravity, dof_pos)
        reward = self._compute_reward(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        obs = self._compute_obs(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        return state.replace(obs=obs, reward=reward, terminated=terminated)

    def _update_wheel_contact(self, info: dict) -> None:
        try:
            left = self._backend.get_sensor_data("left_wheel_force")
            right = self._backend.get_sensor_data("right_wheel_force")
            left_f = np.asarray(left, dtype=get_global_dtype())
            right_f = np.asarray(right, dtype=get_global_dtype())
            if left_f.ndim == 1:
                left_f = left_f.reshape(-1, 3)
            if right_f.ndim == 1:
                right_f = right_f.reshape(-1, 3)
            left_contact = (np.linalg.norm(left_f, axis=1) > 10.0).astype(np.float64)[
                : self._num_envs
            ]
            right_contact = (np.linalg.norm(right_f, axis=1) > 10.0).astype(np.float64)[
                : self._num_envs
            ]
            info["wheel_contact"] = np.stack([left_contact, right_contact], axis=1)
        except (KeyError, AttributeError):
            info["wheel_contact"] = np.zeros((self._num_envs, 2), dtype=np.float64)

    def _compute_terminated(self, gravity: np.ndarray, dof_pos: np.ndarray) -> np.ndarray:
        tilt = np.arccos(np.clip(gravity[:, 2], -1, 1))
        max_tilt = np.deg2rad(self._jump_cfg.max_tilt_deg)
        terminated = tilt > max_tilt
        thigh_collapsed = (dof_pos[:, 1] < -1.0) | (dof_pos[:, 4] > 1.0)
        calf_extreme = (np.abs(dof_pos[:, 2]) > 1.8) | (np.abs(dof_pos[:, 5]) > 1.8)
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
            dof_vel=dof_vel[:, :NUM_LEG_ACTIONS],
            num_envs=num_obs,
            default_angles=self.default_angles[:NUM_LEG_ACTIONS].astype(dtype),
            tracking_sigma=self._jump_cfg.tracking_sigma,
            base_height_target=self._jump_cfg.base_height_target,
            base_height=self._base_height_values(num_obs),
            gravity=gravity,
            joint_range=None,
        )
        return rewards.run_reward_dispatch(
            scales=self._jump_cfg.scales,
            fns=self._reward_fns,
            ctx=ctx,
            info=info,
            enable_log=self._enable_reward_log,
            ctrl_dt=self._cfg.ctrl_dt,
            only_positive=self._jump_cfg.only_positive_rewards,
        )

    def _base_height_values(self, num_obs: int) -> np.ndarray:
        base_pos = np.asarray(self._backend.get_base_pos(), dtype=get_global_dtype())
        if base_pos.shape[0] != num_obs:
            return np.zeros((num_obs,), dtype=get_global_dtype())
        return np.asarray(base_pos[:, 2], dtype=get_global_dtype())
