"""xqrobotwl PPO-SRL 跳跃环境 — PPO骨架 + SLIP前馈参考 + FSM观测

核心设计:
  1. 奖励 = PPO成熟版 (phase-gated, base_height=-60, jump=12, thrust=30)
  2. 动作 = SLIP前馈(ff) + 策略输出 (gain=1.0, 策略全权)
  3. 观测 = 297D(原PPO) + 18D(FSM状态+计时器) = 315D
  4. FSM提供结构化时序引导, PPO决定具体动作

Joint order: [L_hip_roll, L_hip_pitch, L_knee, R_hip_roll, R_hip_pitch, R_knee, L_wheel, R_wheel]
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

# ── SLIP FSM 前馈 (gain=0.15 — 仅提供方向性建议) ──
WHEEL_R = 0.065
SLIP_FF_GAIN = 0.15

_FSM_FEEDFORWARD = {
    -1: [0.1, 0.15, 0.15, -0.1, -0.15, -0.15, 0.0, 0.0],
    0: [0.1, 0.10, 0.50, -0.1, -0.10, -0.50, 0.0, 0.0],  # v4: 膝 0.70→0.50 防下蹲过深 (实测 0.24"倒地")
    1: [0.1, 0.10, -0.87, -0.1, -0.10, 0.87, 0.0, 0.0],
    2: [0.1, 0.10, 0.00, -0.1, -0.10, 0.00, 0.0, 0.0],
    3: [0.1, 0.15, -0.30, -0.1, -0.15, 0.30, 0.0, 0.0],
    4: [0.1, 0.15, 0.15, -0.1, -0.15, -0.15, 0.0, 0.0],
}


def _compute_feedforward(fsm_state, default_angles, dof_pos):
    ff = np.zeros((fsm_state.shape[0], 8), dtype=np.float64)
    for s, vals in _FSM_FEEDFORWARD.items():
        m = fsm_state == s
        if not m.any():
            continue
        if s == 4:
            r = 0.5
            cur = dof_pos[m, :6]
            df = default_angles[:6]
            ff[m, :6] = df + (cur - df) * (1 - r)
        else:
            ff[m] = vals
    return ff


def _update_fsm_state(
    fsm_state,
    fsm_timer,
    base_height,
    base_linvel,
    dof_pos,
    jt,
    dh,
    dt,
    crouch_time: float = 0.25,
    thrust_time: float = 0.20,
):
    vz = base_linvel[:, 2]
    fsm_timer += dt
    for s in range(-1, 5):
        m = fsm_state == s
        if not m.any():
            continue
        if s == -1:
            t = (jt[m] > 0.5) & (fsm_timer[m] > 0.1)
            nxt = np.zeros_like(fsm_state, dtype=bool)
            nxt[m] = t
            fsm_state[nxt] = 0
            fsm_timer[nxt] = 0
        elif s == 0:
            t = fsm_timer[m] > crouch_time
            nxt = np.zeros_like(fsm_state, dtype=bool)
            nxt[m] = t
            fsm_state[nxt] = 1
            fsm_timer[nxt] = 0
        elif s == 1:
            t = fsm_timer[m] > thrust_time
            nxt = np.zeros_like(fsm_state, dtype=bool)
            nxt[m] = t
            fsm_state[nxt] = 2
            fsm_timer[nxt] = 0
        elif s == 2:
            t = (vz[m] < 0) & (base_height[m] < dh + 0.20)
            t |= fsm_timer[m] > 0.6
            nxt = np.zeros_like(fsm_state, dtype=bool)
            nxt[m] = t
            fsm_state[nxt] = 3
            fsm_timer[nxt] = 0
        elif s == 3:
            t = (base_height[m] < dh + 0.04) & (fsm_timer[m] > 0.05)
            nxt = np.zeros_like(fsm_state, dtype=bool)
            nxt[m] = t
            fsm_state[nxt] = 4
            fsm_timer[nxt] = 0
        elif s == 4:
            t = fsm_timer[m] > 0.15
            nxt = np.zeros_like(fsm_state, dtype=bool)
            nxt[m] = t
            fsm_state[nxt] = -1
            fsm_timer[nxt] = 0
    return fsm_state, fsm_timer


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
    # 落地恢复: 真实腾空→落地后 landing_timer 持续步数 (landing_soft/landing_recovery 门控)
    landing_window: int = 30
    min_grounded_steps: int = 5
    # 消融实验
    feedback_gain: float = 0.15  # SLIP FSM 前馈混合比 (no_fsm=0.0)
    ablation_mode: str = "full"


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
    active = ((phase >= 1.0) & (phase <= 18.0)).astype(np.float64)
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
        (hip_fwd_L >= 0.05)
        & (hip_fwd_R >= 0.05)
        & (knee_bend_L >= 0.05)
        & (knee_bend_R >= 0.05)
        & roll_ok
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
    active = ((phase >= 1.0) & (phase <= 14.0)).astype(np.float64)
    crouching = base_z < jump_cfg.base_height_target
    depth = np.clip((jump_cfg.base_height_target - base_z) / 0.25, 0.0, 1.0)
    # Same posture check as crouch_prep: no backward lean + bent knees + no hip spread
    hip_fwd_L = ctx.dof_pos[:, 1]
    hip_fwd_R = -ctx.dof_pos[:, 4]
    knee_bend_L = ctx.dof_pos[:, 2]
    knee_bend_R = -ctx.dof_pos[:, 5]
    roll_ok = (np.abs(ctx.dof_pos[:, 0] - 0.1) < 0.12) & (np.abs(ctx.dof_pos[:, 3] + 0.1) < 0.12)
    posture_ok = (
        (hip_fwd_L >= 0.05)
        & (hip_fwd_R >= 0.05)
        & (knee_bend_L >= 0.05)
        & (knee_bend_R >= 0.05)
        & roll_ok
    )
    weight = ctx.info.get("jump_curriculum", 1.0)
    return (
        depth * crouching.astype(np.float64) * posture_ok.astype(np.float64) * active * 0.5 * weight
    )


def _reward_lean_forward(ctx: RewardContext) -> np.ndarray:
    """站立时罚髋后仰, 跳时不罚 — trigger≤0.5 激活"""
    trigger = ctx.info["commands"][:, 4]
    active = (trigger <= 0.5).astype(np.float64)
    if not active.any():
        return np.zeros(ctx.num_envs, dtype=np.float64)
    hip_fwd_L = ctx.dof_pos[:, 1]
    hip_fwd_R = -ctx.dof_pos[:, 4]
    penalty_L = np.clip(-hip_fwd_L, 0, 1)
    penalty_R = np.clip(-hip_fwd_R, 0, 1)
    knee_bend_L = ctx.dof_pos[:, 2]
    knee_bend_R = -ctx.dof_pos[:, 5]
    p_knee_L = np.clip(-knee_bend_L, 0, 1) ** 2
    p_knee_R = np.clip(-knee_bend_R, 0, 1) ** 2
    roll_dev_L = np.clip(np.abs(ctx.dof_pos[:, 0] - 0.1) - 0.10, 0, 1)
    roll_dev_R = np.clip(np.abs(ctx.dof_pos[:, 3] + 0.1) - 0.10, 0, 1)
    return -(penalty_L + penalty_R + p_knee_L + p_knee_R + roll_dev_L + roll_dev_R) * 0.5 * active


def _reward_vertical_thrust(ctx: RewardContext, jump_cfg: XqRobotWLJumpRewardConfig) -> np.ndarray:
    vz = ctx.linvel[:, 2]
    phase = ctx.info.get("jump_phase", np.zeros(ctx.num_envs, dtype=np.float64))
    wheel_contact = ctx.info.get("wheel_contact", np.ones((ctx.num_envs, 2)))
    on_ground = (np.max(wheel_contact, axis=1) > 0.5).astype(np.float64)
    active = ((phase >= 12.0) & (vz > 0.0) & (on_ground > 0)).astype(np.float64)
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
    # Don't reward height with locked knees
    knee_ok = ((np.abs(ctx.dof_pos[:, 2]) < 1.2) & (np.abs(ctx.dof_pos[:, 5]) < 1.2)).astype(
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
    """落地缓冲: 真实腾空落地后的窗口内, 奖低垂直冲击 (vz→0).

    门控 ``landing_timer`` (仅真实腾空→落地后 >0), 防"触发时站着不动白拿
    phase>=35 奖励"。SRL+VMC (jump_srl_vmc) 不填充 landing_timer, 回退到
    旧的 phase>=35 行为, 保持兼容。
    """
    base_linvel_z = ctx.linvel[:, 2]
    vz_mag = np.abs(base_linvel_z)
    soft = np.exp(-vz_mag / 0.5)
    if "landing_timer" in ctx.info:
        timer = ctx.info.get("landing_timer", np.zeros(ctx.num_envs, dtype=np.float64))
        active = (timer > 0.0).astype(np.float64)
    else:
        phase = ctx.info.get("jump_phase", np.zeros(ctx.num_envs, dtype=np.float64))
        active = (phase >= 35.0).astype(np.float64)
    weight = ctx.info.get("jump_curriculum", 1.0)
    return soft * 0.3 * weight * active


def _reward_height_progress(ctx: RewardContext) -> np.ndarray:
    base_z = ctx.base_height
    max_z = ctx.info.get("episode_max_height", base_z)
    new_max = np.clip(base_z - max_z, 0, 1.0)
    phase = ctx.info.get("jump_phase", np.zeros(ctx.num_envs, dtype=np.float64))
    active = (phase >= 1.0).astype(np.float64)
    weight = ctx.info.get("jump_curriculum", 1.0)
    return new_max * active * 5.0 * weight


def _reward_landing_recovery(ctx: RewardContext) -> np.ndarray:
    """落地后恢复稳定站立: 双轮着地 + 直立 + 高度接近目标.

    门控 ``landing_timer`` (真实腾空→落地后窗口), 只奖"跳完重新站稳",
    不奖触发时原地站着。直接针对 §7.5 成功率: 每次落地必须恢复站立。
    """
    assert ctx.gravity is not None
    wheel_contact = ctx.info.get("wheel_contact", np.ones((ctx.num_envs, 2)))
    both_contact = (np.min(wheel_contact, axis=1) > 0.5).astype(np.float64)
    # upvector 传感器直立时 gravity[:,2]≈+1, 用正值算 tilt (负号会让 upright 恒 0)
    tilt = np.arccos(np.clip(ctx.gravity[:, 2], -1, 1))
    upright = np.exp(-np.square(tilt) / 0.15)
    height_ok = np.exp(-np.square(ctx.base_height - ctx.base_height_target) / 0.05)
    timer = ctx.info.get("landing_timer", np.zeros(ctx.num_envs, dtype=np.float64))
    active = (timer > 0.0).astype(np.float64)
    weight = ctx.info.get("jump_curriculum", 1.0)
    return both_contact * upright * height_ok * active * weight


def _reward_anti_drift(ctx: RewardContext) -> np.ndarray:
    """站立期微动平衡: 触发关闭时惩罚水平速度超出指令的残余漂移.

    站立 (trigger≤0.5) 时机器人应保持位置; 残余水平速度 (超出 vx 指令) 被惩罚,
    落地后吸收前冲动量、停止滑动, 解决 §7.5 落地后漂移。
    """
    trigger = ctx.info["commands"][:, 4]
    active = (trigger <= 0.5).astype(np.float64)
    v_xy = np.hypot(ctx.linvel[:, 0], ctx.linvel[:, 1])
    vx_cmd = np.abs(ctx.info["commands"][:, 0])
    residual = np.clip(v_xy - vx_cmd - 0.05, 0.0, None)
    pen = np.clip(residual / 0.3, 0.0, 2.0)
    weight = ctx.info.get("jump_curriculum", 1.0)
    return -pen * active * weight


def _reward_action_magnitude(ctx: RewardContext) -> np.ndarray:
    """动作幅度惩罚 (修复 config 中 action_magnitude 键无对应 fn 的静默失效)."""
    return np.sum(np.square(ctx.info["current_actions"]), axis=1)


def _reward_wheel_ground_matching(ctx: RewardContext) -> np.ndarray:
    """轮地速度匹配: 落地/蹬伸 (FSM 1/3/4) 轮速与地面速度一致, 飞行 (FSM 2) 轮不空转.

    修复 config 键 ``wheel_ground_matching`` 与 env fn 键 ``wheel_air_time``
    不一致导致奖励从未生效的 bug。轮速换算为线速度 (rad/s × WHEEL_R → m/s)
    并截断, 避免原始角速度平方 (58 rad/s → 3364) 造成的惩罚爆炸 (基线空中
    轮速峰值 58 rad/s, 若不截断 scale 20 下每步 -6.7k, 会压垮训练)。
    """
    fsm = ctx.info.get("fsm_state", -np.ones(ctx.num_envs, dtype=np.float64))
    wheel_vel = ctx.info.get("wheel_vel", np.zeros((ctx.num_envs, 2)))
    lin_x = ctx.linvel[:, 0:1]
    wheel_lin = wheel_vel * WHEEL_R
    contact_error = np.clip(np.sum(np.square(wheel_lin - lin_x), axis=1), 0.0, 4.0)
    contact_active = np.isin(fsm, [1, 3, 4]).astype(np.float64)
    spin = np.clip(np.sum(np.square(wheel_lin), axis=1), 0.0, 2.0)
    flight_active = (fsm == 2).astype(np.float64)
    weight = ctx.info.get("jump_curriculum", 1.0)
    return (-contact_error * contact_active - spin * flight_active) * weight


@registry.envcfg("XqRobotWLJumpSRLFlat")
@dataclass
class XqRobotWLJumpSRLFlatCfg(XqRobotWLWalkFlatCfg):
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


@registry.env("XqRobotWLJumpSRLFlat", sim_backend="mujoco")
class XqRobotWLJumpSRLFlatEnv(XqRobotWLWalkFlatEnv):
    """PPO-SRL 融合跳跃 — PPO骨架 + SLIP前馈 + FSM观测"""

    _cfg: XqRobotWLJumpSRLFlatCfg
    _jump_cfg: XqRobotWLJumpRewardConfig  # type: ignore[assignment]  # 收窄基类奖励配置类型

    def __init__(self, cfg: XqRobotWLJumpSRLFlatCfg, num_envs=1, backend_type="mujoco"):
        if cfg.reward_config is None:
            raise ValueError("reward_config must be provided via Hydra configuration")
        self._jump_cfg = cfg.reward_config
        self._total_env_steps = 0
        self._jump_curriculum_start = getattr(cfg.reward_config, "jump_curriculum_start", 0)
        self._jump_curriculum_end = getattr(cfg.reward_config, "jump_curriculum_end", 100_000)
        range_span = self._jump_curriculum_end - self._jump_curriculum_start
        self._jump_curriculum_step = float(range_span) if range_span > 0 else 1.0
        super().__init__(cfg, num_envs=num_envs, backend_type=backend_type)
        self._dr_manager._provider = XqRobotWLJumpDRProvider()  # type: ignore[union-attr]
        self._fsm_state = -np.ones(num_envs, dtype=np.int32)
        self._fsm_timer = np.zeros(num_envs, dtype=np.float64)
        self._episode_max_height = np.zeros(num_envs, dtype=np.float64)
        # 落地恢复追踪 (真实腾空→落地, 用于 landing_soft/landing_recovery 门控)
        self._grounded_steps = np.zeros(num_envs, dtype=np.float64)
        self._had_air = np.zeros(num_envs, dtype=np.float64)
        self._landing_timer = np.zeros(num_envs, dtype=np.float64)
        self._prev_airborne = np.zeros(num_envs, dtype=np.float64)
        self._obs_frame_dim = 33
        self._critic_frame_dim = 36
        self._obs_history = np.zeros(
            (num_envs, self._hist_len, self._obs_frame_dim), dtype=self._np_dtype
        )
        self._critic_history = np.zeros(
            (num_envs, self._hist_len, self._critic_frame_dim), dtype=self._np_dtype
        )
        self._critic_history = np.zeros(
            (num_envs, self._hist_len, self._critic_frame_dim), dtype=self._np_dtype
        )

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {"obs": 315, "critic": 342}  # 297+18=315, 324+18=342

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
            "wheel_ground_matching": self._reward_wheel_ground_matching,
            "landing_recovery": self._reward_landing_recovery,
            "anti_drift": self._reward_anti_drift,
            "action_magnitude": self._reward_action_magnitude,
            "vertical_thrust": self._reward_vertical_thrust,
            "crouch_depth": self._reward_crouch_depth,
            "lean_forward": self._reward_lean_forward,
            "height_progress": self._reward_height_progress,
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

    def _reward_lean_forward(self, ctx: RewardContext) -> np.ndarray:
        return _reward_lean_forward(ctx)

    def _reward_height_progress(self, ctx: RewardContext) -> np.ndarray:
        return _reward_height_progress(ctx)

    def _reward_landing_recovery(self, ctx: RewardContext) -> np.ndarray:
        return _reward_landing_recovery(ctx)

    def _reward_anti_drift(self, ctx: RewardContext) -> np.ndarray:
        return _reward_anti_drift(ctx)

    def _reward_action_magnitude(self, ctx: RewardContext) -> np.ndarray:
        return _reward_action_magnitude(ctx)

    def _reward_wheel_ground_matching(self, ctx: RewardContext) -> np.ndarray:
        return _reward_wheel_ground_matching(ctx)

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

    def step(self, actions):
        dof_pos = self.get_dof_pos()
        ff = _compute_feedforward(self._fsm_state, self.default_angles, dof_pos)
        gain = getattr(self._jump_cfg, "feedback_gain", SLIP_FF_GAIN)
        return super().step(ff * gain + actions)

    def update_state(self, state: NpEnvState) -> NpEnvState:
        self._update_commands(state.info)
        # Jump curriculum
        self._total_env_steps += self._num_envs
        progress = np.clip(
            (self._total_env_steps - self._jump_curriculum_start) / self._jump_curriculum_step,
            0.0,
            1.0,
        )
        state.info["commands"][:, 4] *= np.float64(progress)
        state.info["jump_curriculum"] = np.float64(progress)
        # Reset episode_max_height for freshly reset envs
        fresh = state.info["steps"] <= 1
        self._episode_max_height[fresh] = 0.0
        # Jump phase
        trigger_active = state.info["commands"][:, 4] > 0.5
        prev_phase = state.info.get("jump_phase", np.zeros(self._num_envs, dtype=np.float64))
        new_phase = np.where(trigger_active, prev_phase + 1, np.float64(0))
        state.info["jump_phase"] = new_phase
        linvel = self.get_local_linvel()
        gyro = self.get_gyro()
        gravity = self._backend.get_sensor_data(self._cfg.sensor.upvector)
        dof_pos = self.get_dof_pos()
        dof_vel = self.get_dof_vel()
        # ★ SLIP FSM
        base_z = np.asarray(self._backend.get_base_pos(), dtype=get_global_dtype())[:, 2]
        jt = state.info["commands"][:, 4]
        self._fsm_state, self._fsm_timer = _update_fsm_state(
            self._fsm_state,
            self._fsm_timer,
            base_z,
            linvel,
            dof_pos,
            jt,
            self._jump_cfg.base_height_target,
            self._cfg.ctrl_dt,
        )
        # Track episode max height for progress reward
        self._episode_max_height = np.maximum(self._episode_max_height, base_z)
        state.info["episode_max_height"] = self._episode_max_height.copy()
        # 几何接触检测 (对空中扇腿免疫) — 与 jump.py/jump_vmc 同款修复
        self._update_wheel_contact_geom(state.info)
        # 供奖励使用: FSM 状态 + 轮子角速度 (wheel_ground_matching)
        state.info["fsm_state"] = self._fsm_state.astype(np.float64).copy()
        state.info["wheel_vel"] = dof_vel[:, NUM_LEG_ACTIONS:].copy()
        # 落地恢复追踪: 真实腾空→落地后开窗 (landing_soft/landing_recovery 门控)
        self._update_jump_air_progress(state.info, base_z)
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
        """几何接触检测 (轮心世界 z < 0.13): 对空中扇腿免疫。

        与 jump.py 同款修复 — force 阈值法 (norm>10) 在 SRL 空中若腿部动作会误判
        着地, 使 air 门控奖励 (jump_height/wheel_air_time/landing_soft) 失真。
        SRL 的 FSM 参考让空中收腿 (轮力~0), 所以接触大多正确; 但几何法更稳。
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
        """追踪真实腾空→落地, 使 landing_soft/landing_recovery 只在真跳落地后开窗.

        - ``had_air``: 触发窗口内双轮真正离地 (需腾空前已稳定接地
          ``min_grounded_steps`` 步, 排除 reset 初始自由落体)
        - ``landing_timer``: 腾空→落地转换后从 ``landing_window`` 倒数, 空中/窗外归零
        """
        phase = info.get("jump_phase", np.zeros(self._num_envs, dtype=np.float64))
        wheel_contact = info.get("wheel_contact", np.ones((self._num_envs, 2)))
        air = 1.0 - np.mean(wheel_contact, axis=1)
        airborne = (air > 0.5).astype(np.float64)
        window_active = (phase >= 1.0).astype(np.float64)
        idle = (window_active <= 0.0).astype(np.float64)
        fresh = np.asarray(info.get("steps", np.zeros(self._num_envs)))[: self._num_envs] <= 1
        if fresh.any():
            self._grounded_steps[fresh] = 0.0
            self._had_air[fresh] = 0.0
            self._landing_timer[fresh] = 0.0
            self._prev_airborne[fresh] = 0.0
        prev_grounded = self._grounded_steps
        self._grounded_steps = np.where(airborne > 0, np.float64(0), prev_grounded + 1.0)
        min_grounded = int(getattr(self._jump_cfg, "min_grounded_steps", 5))
        real_air = np.where(
            (window_active > 0) & (prev_grounded >= min_grounded),
            airborne,
            np.float64(0),
        )
        self._had_air = np.where(idle > 0, np.float64(0), np.maximum(self._had_air, real_air))
        just_landed = ((self._prev_airborne > 0) & (airborne <= 0) & (self._had_air > 0)).astype(
            np.float64
        )
        landing_window = int(getattr(self._jump_cfg, "landing_window", 30))
        self._landing_timer = np.where(
            just_landed > 0, np.float64(landing_window), self._landing_timer
        )
        self._landing_timer = np.maximum(self._landing_timer - 1.0, 0.0)
        self._landing_timer = np.where(airborne > 0, np.float64(0), self._landing_timer)
        self._landing_timer = np.where(idle > 0, np.float64(0), self._landing_timer)
        self._prev_airborne = airborne
        info["had_air"] = self._had_air.copy()
        info["landing_timer"] = self._landing_timer.copy()

    def _compute_obs(self, info, linvel, gyro, gravity, dof_pos, dof_vel):
        base = super()._compute_obs(info, linvel, gyro, gravity, dof_pos, dof_vel)
        batch = linvel.shape[0]
        fsm_feat = self._fsm_state.astype(np.float64).reshape(-1, 1)[:batch] / 5.0
        timer_feat = np.clip(self._fsm_timer.reshape(-1, 1)[:batch] / 0.8, 0, 1)
        extra = np.tile(
            np.concatenate([fsm_feat, timer_feat], axis=1, dtype=get_global_dtype())[:, None, :],
            (1, 9, 1),
        ).reshape(batch, -1)
        base["obs"] = np.concatenate([base["obs"], extra], axis=1)
        base["critic"] = np.concatenate([base["critic"], extra], axis=1)
        return base

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
