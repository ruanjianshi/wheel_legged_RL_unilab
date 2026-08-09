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
    # 真实腾空落地后, landing_soft 奖励的持续步数 (防"站着不动白拿落地奖励")
    landing_window: int = 10
    # 腾空计入"真实跳跃"前, 必须先稳定落地这么多步 (排除 reset 初始自由落体)
    min_grounded_steps: int = 5
    # 蹬伸奖励需要本窗口先深蹲过 (base_z < window_crouch_threshold), 逼出
    # "先蹲后蹬"序列, 否则策略直接推地 (不蹲) 拿 thrust 但永远离不了地。
    # 且下蹲必须发生在"稳定接地 min_grounded_steps 步之后", 排除 reset 初始
    # 自由落体 (其本身就掉到 ~0.45, 会免费解锁门控)
    thrust_requires_crouch: bool = True
    window_crouch_threshold: float = 0.42


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
    posture_ok = (
        (hip_fwd_L > 0.1) & (hip_fwd_R > 0.1) & (knee_bend_L > 0.1) & (knee_bend_R > 0.1) & roll_ok
    )
    weight = ctx.info.get("jump_curriculum", 1.0)
    return (
        height_ok.astype(np.float64)
        * crouching.astype(np.float64)
        * posture_ok.astype(np.float64)
        * active
        * 0.5
        * weight
    )


def _reward_crouch_depth(ctx: RewardContext, jump_cfg: XqRobotWLJumpRewardConfig) -> np.ndarray:
    base_z = ctx.base_height
    phase = ctx.info.get("jump_phase", np.zeros(ctx.num_envs, dtype=np.float64))
    active = ((phase >= 1.0) & (phase <= 25.0)).astype(np.float64)
    crouching = base_z < jump_cfg.base_height_target
    depth = np.clip((jump_cfg.base_height_target - base_z) / 0.25, 0.0, 1.0)
    # Same posture check as crouch_prep: no backward lean + bent knees + no hip spread
    hip_fwd_L = ctx.dof_pos[:, 1]
    hip_fwd_R = -ctx.dof_pos[:, 4]
    knee_bend_L = ctx.dof_pos[:, 2]
    knee_bend_R = -ctx.dof_pos[:, 5]
    roll_ok = (np.abs(ctx.dof_pos[:, 0] - 0.1) < 0.12) & (np.abs(ctx.dof_pos[:, 3] + 0.1) < 0.12)
    posture_ok = (
        (hip_fwd_L > 0.1) & (hip_fwd_R > 0.1) & (knee_bend_L > 0.1) & (knee_bend_R > 0.1) & roll_ok
    )
    weight = ctx.info.get("jump_curriculum", 1.0)
    return (
        depth * crouching.astype(np.float64) * posture_ok.astype(np.float64) * active * 0.5 * weight
    )


def _reward_stand_posture(ctx: RewardContext) -> np.ndarray:
    # When no trigger: reward staying at default posture + standing height
    trigger = ctx.info["commands"][:, 4]
    standing = (trigger <= 0.5).astype(np.float64)
    height_err = np.square(ctx.base_height - 0.65)
    dof_err = np.sum(np.square(ctx.dof_pos[:, :6] - ctx.default_angles[:6]), axis=1)
    act_mag = np.sum(np.square(ctx.info["current_actions"][:, :6]), axis=1)
    return -(height_err * 15.0 + dof_err * 3.0 + act_mag * 1.0) * standing


def _reward_lean_forward(ctx: RewardContext) -> np.ndarray:
    # 站立时罚髋偏离默认角, 跳时不罚 — trigger≤0.5 激活
    trigger = ctx.info["commands"][:, 4]
    active = (trigger <= 0.5).astype(np.float64)
    if not active.any():
        return np.zeros(ctx.num_envs, dtype=np.float64)
    hip_fwd_L = ctx.dof_pos[:, 1]  # default=+0.15
    hip_fwd_R = -ctx.dof_pos[:, 4]  # default=+0.15 (after flip)
    # Bidirectional: penalize deviation from default ± tolerance
    dev_L = np.clip(np.abs(hip_fwd_L - 0.15) - 0.05, 0, 1)
    dev_R = np.clip(np.abs(hip_fwd_R - 0.15) - 0.05, 0, 1)
    knee_bend_L = ctx.dof_pos[:, 2]
    knee_bend_R = -ctx.dof_pos[:, 5]
    p_knee_L = np.clip(-knee_bend_L, 0, 1) ** 2
    p_knee_R = np.clip(-knee_bend_R, 0, 1) ** 2
    roll_dev_L = np.clip(np.abs(ctx.dof_pos[:, 0] - 0.1) - 0.10, 0, 1)
    roll_dev_R = np.clip(np.abs(ctx.dof_pos[:, 3] + 0.1) - 0.10, 0, 1)
    return -(dev_L + dev_R + p_knee_L + p_knee_R + roll_dev_L + roll_dev_R) * 1.5 * active


def _reward_launch_rise(ctx: RewardContext, jump_cfg: XqRobotWLJumpRewardConfig) -> np.ndarray:
    """Reward how far the body actually rises above this window's deepest crouch.

    Replaces the old ``vertical_thrust`` (quadratic vz). Rewarding vz>0 while on
    the ground let the pure-PPO policy farm a "push but never lift" stutter
    (repeated short pushes spike vz without committing to full leg extension ->
    no air). Measuring the RISE above the window floor instead gives a push
    that returns to the crouch exactly zero, a slow stand a little, and a real
    launch that keeps extending the most. The rise floor (``window_min_z``)
    only drops on GROUNDED crouches, so the airborne reset drop is excluded.
    """
    base_z = ctx.base_height
    phase = ctx.info.get("jump_phase", np.zeros(ctx.num_envs, dtype=np.float64))
    wheel_contact = ctx.info.get("wheel_contact", np.ones((ctx.num_envs, 2)))
    on_ground = (np.max(wheel_contact, axis=1) > 0.5).astype(np.float64)
    floor = ctx.info.get(
        "window_min_z", np.full(ctx.num_envs, jump_cfg.base_height_target, dtype=np.float64)
    )
    rise = np.clip(base_z - floor, 0.0, 1.0)
    active = ((phase >= 1.0) & (on_ground > 0)).astype(np.float64)
    # Must have genuinely crouched first (a standing reset drop below the
    # threshold would otherwise unlock rise for free).
    if getattr(jump_cfg, "thrust_requires_crouch", True):
        window_crouched = ctx.info.get("window_crouched", np.ones(ctx.num_envs, dtype=np.float64))
        active = active * (window_crouched > 0)
    weight = ctx.info.get("jump_curriculum", 1.0)
    return rise * active * weight


def _reward_jump_height(ctx: RewardContext, jump_cfg: XqRobotWLJumpRewardConfig) -> np.ndarray:
    base_z = ctx.base_height
    phase = ctx.info.get("jump_phase", np.ones(ctx.num_envs, dtype=np.float64) * 1e9)
    active = (phase >= 1.0).astype(np.float64)
    target = jump_cfg.jump_height_target
    clamped = np.clip(base_z / target, 0.0, 1.0)
    wheel_contact = ctx.info.get("wheel_contact", np.ones((ctx.num_envs, 2)))
    air_factor = 1.0 - np.mean(wheel_contact, axis=1)
    # Don't reward height with locked knees
    knee_ok = ((np.abs(ctx.dof_pos[:, 2]) < 0.8) & (np.abs(ctx.dof_pos[:, 5]) < 0.8)).astype(
        np.float64
    )
    weight = ctx.info.get("jump_curriculum", 1.0)
    return clamped * active * air_factor * knee_ok * 2.0 * weight


def _reward_wheel_air_time(ctx: RewardContext) -> np.ndarray:
    wheel_contact = ctx.info.get("wheel_contact", np.ones((ctx.num_envs, 2)))
    air = 1.0 - np.mean(wheel_contact, axis=1)
    phase = ctx.info.get("jump_phase", np.zeros(ctx.num_envs, dtype=np.float64))
    active = (phase >= 1.0).astype(np.float64)
    # Penalize wheel spinning in the air
    wheel_vel = ctx.info["current_actions"][:, -2:]
    wheel_spin = np.sum(np.abs(wheel_vel), axis=1) * air
    weight = ctx.info.get("jump_curriculum", 1.0)
    return (air * 0.5 - wheel_spin * 0.1) * weight * active


def _reward_landing_soft(ctx: RewardContext) -> np.ndarray:
    """Reward a gentle, recovered landing AFTER a real jump.

    Previously this fired for any ``jump_phase >= 30`` (any trigger window held
    long enough), so standing perfectly still while the trigger was on collected
    the full scale every step without ever leaving the ground -- the exact
    "fake jump" the pure-PPO policy settled into (v9 landing_soft 30/step vs
    jump_height ~0). Now it is gated on ``landing_timer``: the wheels must have
    left the ground this window (had_air) and come back down (airborne ->
    grounded transition) before the reward arms for ``landing_window`` steps.
    """
    base_linvel_z = ctx.linvel[:, 2]
    vz_mag = np.abs(base_linvel_z)
    soft = np.exp(-vz_mag / 0.5)
    timer = ctx.info.get("landing_timer", np.zeros(ctx.num_envs, dtype=np.float64))
    active = (timer > 0.0).astype(np.float64)
    weight = ctx.info.get("jump_curriculum", 1.0)
    return soft * 1.0 * weight * active


def _reward_landing_recovery(ctx: RewardContext) -> np.ndarray:
    """Reward upright, stable wheel-contact posture (landing recovery).

    Targets the pure-PPO failure mode of jumping high but crashing on landing:
    the policy earns credit for being back on both wheels, upright and near the
    commanded base height, which encourages recovering after a jump instead of
    settling into a tilted/collapsed landing.
    """
    wheel_contact = ctx.info.get("wheel_contact", np.ones((ctx.num_envs, 2)))
    both_contact = (np.min(wheel_contact, axis=1) > 0.5).astype(np.float64)
    tilt = np.arccos(np.clip(-ctx.gravity[:, 2], -1, 1))
    upright = np.exp(-np.square(tilt) / 0.15)
    height_ok = np.exp(-np.square(ctx.base_height - ctx.base_height_target) / 0.05)
    weight = ctx.info.get("jump_curriculum", 1.0)
    return both_contact * upright * height_ok * weight


def _reward_anti_loiter(ctx: RewardContext) -> np.ndarray:
    # Penalize staying crouched after the crouch window closes (phase > 30)
    phase = ctx.info.get("jump_phase", np.zeros(ctx.num_envs, dtype=np.float64))
    base_z = ctx.base_height
    loitering = ((phase > 30.0) & (base_z < 0.55)).astype(np.float64)
    # Deeper crouch = bigger penalty → forces extension or jump
    penalty = np.clip((0.55 - base_z) / 0.3, 0.0, 2.0)
    weight = ctx.info.get("jump_curriculum", 1.0)
    return -penalty * loitering * weight


def _reward_anti_lazy(ctx: RewardContext) -> np.ndarray:
    """Penalise a trigger window that does not move the legs enough.

    With the jump-window base_height penalty removed (see
    ``_reward_base_height_jump``), the main reason the pure-PPO policy stayed
    still is gone. This term is the remaining gentle push: during the trigger
    window the knees must actually move through a crouch -> thrust sequence
    (excursion ~0.8 rad). It stops a policy from just sitting at a fixed
    posture while the trigger is held.
    """
    phase = ctx.info.get("jump_phase", np.zeros(ctx.num_envs, dtype=np.float64))
    active = ((phase >= 1.0) & (phase <= 40.0)).astype(np.float64)
    knee_L = ctx.dof_pos[:, 2]
    knee_R = -ctx.dof_pos[:, 5]
    excursion = np.abs(knee_L - 0.15) + np.abs(knee_R - 0.15)
    lazy = np.clip(0.8 - excursion, 0.0, None)
    weight = ctx.info.get("jump_curriculum", 1.0)
    return -lazy * active * weight


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
        self._jump_start_z = np.zeros(num_envs, dtype=np.float64)
        # 真实腾空追踪: had_air (本窗口内轮是否离地) + landing_timer (落地后
        # 奖励窗口) — 防止纯PPO 用 "站着不动白拿 landing_soft" 的假跳高解
        self._landing_window = int(getattr(cfg.reward_config, "landing_window", 10))
        self._min_grounded_steps = float(getattr(cfg.reward_config, "min_grounded_steps", 5))
        self._thrust_requires_crouch = bool(
            getattr(cfg.reward_config, "thrust_requires_crouch", True)
        )
        self._window_crouch_threshold = float(
            getattr(cfg.reward_config, "window_crouch_threshold", 0.45)
        )
        self._had_air = np.zeros(num_envs, dtype=np.float64)
        self._landing_timer = np.zeros(num_envs, dtype=np.float64)
        self._prev_airborne = np.zeros(num_envs, dtype=np.float64)
        self._grounded_steps = np.zeros(num_envs, dtype=np.float64)
        self._window_crouched = np.zeros(num_envs, dtype=np.float64)
        self._window_min_z = np.full(num_envs, 0.55, dtype=np.float64)
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
            "base_height": self._reward_base_height_jump,
            "orientation": rewards.orientation,
            "joint_action_rate": self._reward_joint_action_rate,
            "wheel_action_rate": self._reward_wheel_action_rate,
            "leg_mirror": self._reward_leg_mirror,
            "tsk": self._reward_tsk,
            "alive": rewards.alive,
            "jump_height": self._reward_jump_height,
            "crouch_prep": self._reward_crouch_prep,
            "landing_soft": self._reward_landing_soft,
            "landing_recovery": self._reward_landing_recovery,
            "wheel_air_time": self._reward_wheel_air_time,
            "launch_rise": self._reward_launch_rise,
            "crouch_depth": self._reward_crouch_depth,
            "anti_loiter": self._reward_anti_loiter,
            "lean_forward": self._reward_lean_forward,
            "anti_lazy": self._reward_anti_lazy,
        }

    def _reward_jump_height(self, ctx: RewardContext) -> np.ndarray:
        return _reward_jump_height(ctx, self._jump_cfg)

    def _reward_base_height_jump(self, ctx: RewardContext) -> np.ndarray:
        """Base-height penalty that does NOT fight a real jump.

        The stock ``rewards.base_height`` punishes deviation from the standing
        height (scale -60), so the higher the robot jumps the harder it is
        punished -- which is why the pure-PPO policy chose not to jump at all.
        During the jump window we instead anchor the penalty to the height at
        trigger start: crouching (down) is mildly penalised, but rising above
        trigger height is free, so a genuine jump no longer carries a -60
        counterweight against the jump_height reward. Outside the window the
        normal penalty applies.
        """
        phase = ctx.info.get("jump_phase", np.zeros(ctx.num_envs, dtype=np.float64))
        active = (phase >= 1.0).astype(np.float64)
        target = ctx.base_height_target
        # During the jump window the height penalty is ZERO: crouching must be
        # free (the anti_loiter / anti_lazy / crouch_depth terms shape the
        # crouch). Any penalty on dropping below the trigger height cancelled
        # the crouch reward, so the pure-PPO policy skipped the crouch and just
        # pushed from the tall stance (which can never launch). Outside the
        # window the usual standing-height penalty applies.
        err = np.where(active > 0, np.float64(0), np.square(ctx.base_height - target))
        return err  # type: ignore[no-any-return]

    def _reward_crouch_prep(self, ctx: RewardContext) -> np.ndarray:
        return _reward_crouch_prep(ctx, self._jump_cfg)

    def _reward_landing_soft(self, ctx: RewardContext) -> np.ndarray:
        return _reward_landing_soft(ctx)

    def _reward_landing_recovery(self, ctx: RewardContext) -> np.ndarray:
        return _reward_landing_recovery(ctx)

    def _reward_wheel_air_time(self, ctx: RewardContext) -> np.ndarray:
        return _reward_wheel_air_time(ctx)

    def _reward_launch_rise(self, ctx: RewardContext) -> np.ndarray:
        return _reward_launch_rise(ctx, self._jump_cfg)

    def _reward_crouch_depth(self, ctx: RewardContext) -> np.ndarray:
        return _reward_crouch_depth(ctx, self._jump_cfg)

    def _reward_anti_loiter(self, ctx: RewardContext) -> np.ndarray:
        return _reward_anti_loiter(ctx)

    def _reward_anti_lazy(self, ctx: RewardContext) -> np.ndarray:
        return _reward_anti_lazy(ctx)

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
        hip_error = np.abs(ctx.dof_pos[:, 0] + ctx.dof_pos[:, 3])
        pitch_error = ctx.dof_pos[:, 1:3] + ctx.dof_pos[:, 4:6]
        asym = hip_error + np.sum(np.abs(pitch_error), axis=1)
        return np.clip(asym - 0.15, 0, 2.0)

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
            0.0,
            1.0,
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
        # Record base_z at the moment a jump trigger starts (phase 0 -> 1), so
        # anti_lazy can measure "how far did the body actually rise" this window.
        base_z_now = np.asarray(self._backend.get_base_pos(), dtype=get_global_dtype())[:, 2]
        just_started = (prev_phase == 0) & (new_phase == 1)
        if just_started.any():
            self._jump_start_z[just_started] = base_z_now[just_started]
        state.info["jump_start_z"] = self._jump_start_z.copy()
        self._update_wheel_contact(state.info)
        self._update_jump_air_progress(state.info, base_z_now)
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

    def _update_jump_air_progress(self, info: dict, base_z: np.ndarray) -> None:
        """Track real jumps so passive standing cannot farm ``landing_soft``.

        Arms ``info["landing_timer"]`` only when the wheels have truly left the
        ground during the current trigger window (``had_air``) and have come
        back down (airborne -> grounded transition). A policy that holds the
        trigger and stands still never gets credit.

        A reset drops the robot from a hovering keyframe (wheels ~0.15m off the
        ground), so the very first "air" of an episode is just the initial
        free-fall. Air only counts as a real jump once the robot has been
        firmly grounded for ``min_grounded_steps`` consecutive steps.

        Also records ``info["window_crouched"]``: whether the base dropped below
        ``window_crouch_threshold`` at any point during the current window.
        ``vertical_thrust`` is gated on it so the policy must crouch before it
        can earn thrust reward (a crouch is what gives the legs the travel to
        actually launch).
        """
        phase = info.get("jump_phase", np.zeros(self._num_envs, dtype=np.float64))
        wheel_contact = info.get("wheel_contact", np.ones((self._num_envs, 2)))
        air = 1.0 - np.mean(wheel_contact, axis=1)
        airborne = (air > 0.5).astype(np.float64)
        window_active = (phase >= 1.0).astype(np.float64)
        idle = (window_active <= 0.0).astype(np.float64)
        # Freshly reset envs carry stale grounded/had_air state into the new
        # episode's free-fall; zero them so the reset drop is never credited.
        fresh = np.asarray(info.get("steps", np.zeros(self._num_envs)))[: self._num_envs] <= 1
        if fresh.any():
            self._grounded_steps[fresh] = 0.0
            self._had_air[fresh] = 0.0
            self._landing_timer[fresh] = 0.0
            self._prev_airborne[fresh] = 0.0
            self._window_crouched[fresh] = 0.0
            self._window_min_z[fresh] = 0.55
        # Consecutive grounded steps (checked on the PREVIOUS step, so the
        # step that lifts off still sees its standing history).
        prev_grounded = self._grounded_steps
        self._grounded_steps = np.where(airborne > 0, np.float64(0), prev_grounded + 1.0)
        real_air = np.where(
            (window_active > 0) & (prev_grounded >= self._min_grounded_steps),
            airborne,
            np.float64(0),
        )
        # had_air: any real-air step inside an active trigger window
        self._had_air = np.where(idle > 0, np.float64(0), np.maximum(self._had_air, real_air))
        just_landed = ((self._prev_airborne > 0) & (airborne <= 0) & (self._had_air > 0)).astype(
            np.float64
        )
        self._landing_timer = np.where(
            just_landed > 0, np.float64(self._landing_window), self._landing_timer
        )
        self._landing_timer = np.maximum(self._landing_timer - 1.0, 0.0)
        # Still airborne, or outside any trigger window -> no landing credit
        self._landing_timer = np.where(airborne > 0, np.float64(0), self._landing_timer)
        self._landing_timer = np.where(idle > 0, np.float64(0), self._landing_timer)
        self._prev_airborne = airborne
        # window_crouched: base dropped below the crouch threshold while the
        # robot was FIRMLY grounded (prev_grounded >= min_grounded_steps). The
        # reset free-fall also reaches ~0.45 but happens while airborne, so it
        # must not unlock the thrust gate.
        base_z_arr = np.asarray(base_z, dtype=np.float64)[: self._num_envs]
        crouch_eligible = (prev_grounded >= self._min_grounded_steps) & (window_active > 0)
        crouched_now = (base_z_arr < self._window_crouch_threshold) & (crouch_eligible > 0)
        self._window_crouched = np.where(
            idle > 0,
            np.float64(0),
            np.maximum(self._window_crouched, crouched_now.astype(np.float64)),
        )
        # window_min_z: deepest GROUNDED base_z this window (the airborne reset
        # drop is excluded, so it is a real crouch). Resets to the current
        # height when the window ends. ``launch_rise`` measures the body's rise
        # above this floor, so a "push but never lift" stutter earns nothing.
        grounded = (airborne <= 0).astype(np.float64)
        self._window_min_z = np.where(
            idle > 0,
            base_z_arr,
            np.where(
                (window_active > 0) & (grounded > 0),
                np.minimum(self._window_min_z, base_z_arr),
                self._window_min_z,
            ),
        )
        info["had_air"] = self._had_air.copy()
        info["landing_timer"] = self._landing_timer.copy()
        info["window_crouched"] = self._window_crouched.copy()
        info["window_min_z"] = self._window_min_z.copy()

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
