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
    # v10: landing_soft 是否要求本窗口先深蹲 (window_crouched)。
    # PPO+VMC (fsm 强制下蹲) 开; 纯PPO (无参考, 深蹲难发现) 关 — v8/v9 门控后
    # 纯PPO 退化回站立+开车, 其最优是 v7 小跳。
    landing_soft_requires_crouch: bool = False


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
    # Phase 2 (anti-hover): 只奖上升段 (vz>0)。旧公式奖绝对高度, 纯PPO 策略收敛到
    # "悬空不落地" (v3 iter1000 实测 air_frac 0.76, 悬空刷 jump_height+wheel_air_time)。
    # 门控 vz>0 后, 悬停在顶点 (vz≈0) 不再得分, 必须落地恢复 (landing_recovery) 才有持续收益。
    ascending = (ctx.linvel[:, 2] > 0.0).astype(np.float64)
    # Phase 3 (anti-rocket): 还须本窗口先深蹲过 (window_crouched)。v4 iter1000 实测
    # crouch_prep≈0 (策略不深蹲就原地起跳弹跳, air 0.81), 升空段仍被刷分。
    # 与 launch_rise 同一门控: 必须先蹲到 <0.42, 升空才有跳高奖励 → 逼出"先蹲后蹬"。
    window_crouched = ctx.info.get(
        "window_crouched", np.ones(ctx.num_envs, dtype=np.float64)
    )
    weight = ctx.info.get("jump_curriculum", 1.0)
    return (
        clamped
        * active
        * air_factor
        * knee_ok
        * ascending
        * (window_crouched > 0).astype(np.float64)
        * 2.0
        * weight
    )


def _reward_wheel_air_time(ctx: RewardContext) -> np.ndarray:
    wheel_contact = ctx.info.get("wheel_contact", np.ones((ctx.num_envs, 2)))
    air = 1.0 - np.mean(wheel_contact, axis=1)
    phase = ctx.info.get("jump_phase", np.zeros(ctx.num_envs, dtype=np.float64))
    active = (phase >= 1.0).astype(np.float64)
    # Phase 2 (anti-hover): 去掉正 air 项 (air*0.5 奖励"待在空中的时间", 被悬空策略刷),
    # 只罚空中转轮。腾空收益由 jump_height (vz>0 门控) 负责。
    wheel_vel = ctx.info["current_actions"][:, -2:]
    wheel_spin = np.sum(np.abs(wheel_vel), axis=1) * air
    weight = ctx.info.get("jump_curriculum", 1.0)
    return (-wheel_spin * 0.1) * weight * active


def _reward_landing_soft(
    ctx: RewardContext, jump_cfg: XqRobotWLJumpRewardConfig
) -> np.ndarray:
    """Reward a gentle, recovered landing AFTER a real jump.

    Previously this fired for any ``jump_phase >= 30`` (any trigger window held
    long enough), so standing perfectly still while the trigger was on collected
    the full scale every step without ever leaving the ground -- the exact
    "fake jump" the pure-PPO policy settled into (v9 landing_soft 30/step vs
    jump_height ~0). Now it is gated on ``landing_timer``: the wheels must have
    left the ground this window (had_air) and come back down (airborne ->
    grounded transition) before the reward arms for ``landing_window`` steps.

    v10: ``landing_soft_requires_crouch`` 可配置深蹲门控。PPO+VMC (fsm 强制下蹲)
    可开; 纯PPO (无参考, 深蹲难发现) 必须关 — v8/v9 门控后纯PPO 退化回站立+开车,
    其最优是 v7 小跳。
    """
    base_linvel_z = ctx.linvel[:, 2]
    vz_mag = np.abs(base_linvel_z)
    soft = np.exp(-vz_mag / 0.5)
    timer = ctx.info.get("landing_timer", np.zeros(ctx.num_envs, dtype=np.float64))
    active = (timer > 0.0).astype(np.float64)
    if getattr(jump_cfg, "landing_soft_requires_crouch", False):
        window_crouched = ctx.info.get("window_crouched", np.ones(ctx.num_envs, dtype=np.float64))
        active = active * (window_crouched > 0).astype(np.float64)
    weight = ctx.info.get("jump_curriculum", 1.0)
    return soft * 1.0 * weight * active


def _reward_landing_recovery(ctx: RewardContext) -> np.ndarray:
    """Reward upright, stable wheel-contact posture (landing recovery).

    Targets the pure-PPO failure mode of jumping high but crashing on landing:
    the policy earns credit for being back on both wheels, upright and near the
    commanded base height, which encourages recovering after a jump instead of
    settling into a tilted/collapsed landing.
    """
    assert ctx.gravity is not None
    wheel_contact = ctx.info.get("wheel_contact", np.ones((ctx.num_envs, 2)))
    both_contact = (np.min(wheel_contact, axis=1) > 0.5).astype(np.float64)
    # upvector 传感器直立时 gravity[:,2]≈+1; 用正值算 tilt (负号会让 upright 恒 0)。
    # 与 jump_srl._reward_landing_recovery / _compute_terminated 一致。
    tilt = np.arccos(np.clip(ctx.gravity[:, 2], -1, 1))
    upright = np.exp(-np.square(tilt) / 0.15)
    height_ok = np.exp(-np.square(ctx.base_height - ctx.base_height_target) / 0.05)
    # Phase 3: 门控到 trigger-off (命令关闭) — ON 时段站着不奖, 逼出跳跃;
    # OFF 时段站立恢复才有收益 → 策略学"ON 跳 / OFF 站稳" (对齐 §7.5 评估)。
    trigger = ctx.info["commands"][:, 4]
    idle = (trigger <= 0.5).astype(np.float64)
    weight = ctx.info.get("jump_curriculum", 1.0)
    return both_contact * upright * height_ok * idle * weight


def _reward_jump_land(ctx: RewardContext, jump_cfg: XqRobotWLJumpRewardConfig) -> np.ndarray:
    """Phase 4: 每次真实跳跃落地时, 按本窗口腾空峰值高度一次性奖励。

    ``jump_height`` 是逐步的绝对高度奖励, 被"悬空/火箭/小跳"利用过。本奖励按
    ``window_max_z - window_min_z`` (腾空峰值 - 下蹲最低点) 在真实腾空→落地
    (just_landed) 时发放: 悬空不落地拿不到, 火箭落地前终止拿不到, 小跳峰值低
    奖励小, 跳得高才拿大额 → 直接编码 §7.5 "跳高 + 落地恢复"。
    """
    peak = ctx.info.get(
        "window_max_z", np.full(ctx.num_envs, jump_cfg.base_height_target, dtype=np.float64)
    )
    floor = ctx.info.get(
        "window_min_z", np.full(ctx.num_envs, jump_cfg.base_height_target, dtype=np.float64)
    )
    rise = np.clip(peak - floor, 0.0, 1.5)
    just_landed = ctx.info.get("just_landed", np.zeros(ctx.num_envs, dtype=np.float64))
    target = jump_cfg.jump_height_target
    height_score = np.clip(rise / max(target, 1e-6), 0.0, 1.5)
    weight = ctx.info.get("jump_curriculum", 1.0)
    return height_score * just_landed * weight


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
    reward_config: XqRobotWLJumpRewardConfig | None = None  # type: ignore[assignment]
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
    _jump_cfg: XqRobotWLJumpRewardConfig  # type: ignore[assignment]  # 收窄基类奖励配置类型

    def __init__(self, cfg: XqRobotWLJumpFlatCfg, num_envs=1, backend_type="mujoco"):
        if cfg.reward_config is None:
            raise ValueError("reward_config must be provided via Hydra configuration")
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
        # jump_land (Phase4): 本窗口真实腾空期间的最高 base_z, 落地时按峰值一次性奖励。
        # 悬空/火箭 (不落地) 刷不到, 小跳 (峰值低) 奖励小, 跳高才能拿大额 → 逼高跳跃。
        self._window_max_z = np.full(num_envs, 0.55, dtype=np.float64)
        super().__init__(cfg, num_envs=num_envs, backend_type=backend_type)
        self._dr_manager._provider = XqRobotWLJumpDRProvider()  # type: ignore[union-attr]
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
            "jump_land": self._reward_jump_land,
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
        return _reward_landing_soft(ctx, self._jump_cfg)

    def _reward_landing_recovery(self, ctx: RewardContext) -> np.ndarray:
        return _reward_landing_recovery(ctx)

    def _reward_jump_land(self, ctx: RewardContext) -> np.ndarray:
        return _reward_jump_land(ctx, self._jump_cfg)

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
        self._update_wheel_contact_geom(state.info)
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

    def _update_wheel_contact_geom(self, info: dict) -> None:
        """Robust wheel-ground contact from the wheel body height (isolated).

        The force-magnitude heuristic (``norm(wheel_force) > 10``) reports
        contact even in mid-air: while the legs are moving the wheel is
        accelerated relative to free-fall, so the wheel-site force exceeds
        10 N (the wheel alone weighs ~23 N). A pure-PPO policy that flails its
        legs in the air therefore reads as permanently grounded, the air-gated
        rewards (jump_height / wheel_air_time / landing_soft) never activate,
        and ``verify_jump`` reports air=0 while the robot is actually flying.

        Instead we detect contact geometrically: the wheel centre is on the
        ground iff its world z is within the wheel radius (+ penetration
        margin) of the floor plane (z=0). Robust to leg flailing and to the
        hover-keyframe reset (wheels ~0.035 m up -> no contact).

        Only the *left* wheel framepos sensor is defined in the robot XML, so
        the symmetric right-wheel contact is inferred from the left. Isolated to
        ``XqRobotWLJumpFlatEnv.update_state`` (VMC / VMC+SRL re-implement
        ``update_state`` and keep the force heuristic).
        """
        try:
            left = np.asarray(
                self._backend.get_sensor_data("left_wheel_world_pos"),
                dtype=get_global_dtype(),
            ).reshape(-1, 3)[: self._num_envs]
            wheel_radius = 0.11
            contact = (left[:, 2] < wheel_radius + 0.02).astype(np.float64)
            info["wheel_contact"] = np.stack([contact, contact], axis=1)
        except (KeyError, AttributeError):
            self._update_wheel_contact(info)

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
            self._window_max_z[fresh] = 0.55
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
        # window_max_z: peak base_z during REAL-AIR flight this window (jump_land
        # 奖励用). 仅真实腾空 (排除 reset 自由落体) 计峰, 窗口结束重置。
        self._window_max_z = np.where(
            idle > 0,
            base_z_arr,
            np.maximum(
                self._window_max_z,
                np.where(real_air > 0, base_z_arr, self._window_max_z),
            ),
        )
        info["had_air"] = self._had_air.copy()
        info["landing_timer"] = self._landing_timer.copy()
        info["window_crouched"] = self._window_crouched.copy()
        info["window_min_z"] = self._window_min_z.copy()
        info["window_max_z"] = self._window_max_z.copy()
        info["just_landed"] = just_landed.copy()

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
