"""xqrobotwl 改进结构单轮平衡 RL env — 站立按键切换横躺独轮车式.

改进结构 (devlog 20, 物理可行):
  - 机身横躺 up=[0,-1,0], **左腿近直支撑** L=(1.49,-0.1,-0.2), CoM 偏移 0.5cm
  - **右腿伸直配重** R=(-1.50,0,0), CoM 权限 ±5cm
  - 执行器 (env init 运行时设置): 腿 kp=300 kv=10 (防膝塌陷), 轮=扭矩源
  - 控制通道: 左轮扭矩控 pitch+移动, 右髋 roll/pitch 配重控 roll, 支撑腿钉住

经典控制验证 (devlog 20): PD+滤波 0.86s 才倒 (原结构 0.2s), 剩余=配重 roll
耦合 pitch 的 3D 控制 → 交给 RL 学.

关键观测: 横躺姿态下 pitch (绕轮轴=世界y) 在 up 向量不可见 (up∥轮轴),
必须加 base 长轴向量 (basexvector sensor) 到 obs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from unilab.base import registry
from unilab.base.np_env import NpEnvState
from unilab.dr import ResetPlan
from unilab.dr.dr_utils import build_common_reset_randomization, zero_actions
from unilab.dtype_config import get_global_dtype
from unilab.envs.locomotion.common import rewards
from unilab.envs.locomotion.common.commands import Commands
from unilab.envs.locomotion.common.rewards import RewardContext
from unilab.envs.locomotion.xqrobotwl.base import NUM_LEG_ACTIONS
from unilab.envs.locomotion.xqrobotwl.joystick import (
    XqRobotWLDRProvider,
    XqRobotWLRewardConfig,
    XqRobotWLWalkFlatCfg,
    XqRobotWLWalkFlatEnv,
)

# ── 改进结构参数 (devlog 20 结构搜索) ──
_SUPPORT_POSE = np.array([1.49, -0.1, -0.2], dtype=np.float64)  # 支撑腿(左) L_roll/L_pitch/L_knee
_CW_ROLL0 = -1.50
_CW_PITCH0 = -0.61
_CW_KNEE0 = -0.87
# 横躺参考 (up=base+z 指向 -y, base+x 指向 +x)
_UP_REF = np.array([0.0, -1.0, 0.0], dtype=np.float64)
_XVEC_REF = np.array([1.0, 0.0, 0.0], dtype=np.float64)
# 动作映射
_WHEEL_TORQUE_SCALE = 8.0  # RL 动作 a6 ∈ ~[-1,1] → 轮扭矩 Nm
_WHEEL_TORQUE_LIMIT = 20.0
_LEG_TORQUE_LIMIT = 30.0
_LEG_DAMPING = 10.0
_RROLL_SCALE = 0.5  # RL 动作 → R_hip_roll 增量 (rad)
_RPITCH_SCALE = 0.5
# reset 几何 (FK 标定): 横躺位姿下左轮贴地的 base 高度
_BASE_Z = 0.6776
_WHEEL_R = 0.11
_WALK_STAND_HEIGHT_COMMAND = 0.518
_NUM_CMD_DIM = 5
# obs 帧: gyro(3)+grav(3)+basex(3)+leg_diff(6)+leg_vel(6)+wheel_vel(2)+act(8)
#           + motion_features(5)
# motion_features=[vx_cmd, vx, vy, vx_cmd-vx, height]. 维度保持 36，允许从静态
# 平衡 checkpoint 热启动，同时补上经典控制已证明必需的 vx 反馈。
_OBS_FRAME_DIM = 36
_CRITIC_FRAME_DIM = 39
# 位姿默认角 (leg_diff 参考) — 支撑腿钉在 pose, 配重基准
_POSE_DEFAULT = np.array([1.49, -0.1, -0.2, -1.50, -0.61, -0.87], dtype=np.float64)
_STAND_POSE = np.array([0.10, 0.15, 0.15, -0.10, -0.15, -0.15], dtype=np.float64)
_RIGHT_TUCK_POSE = np.array([-1.50, -0.61, -0.87], dtype=np.float64)
# 终止阈值
_WHEEL_LIFT_THRESHOLD = 0.02  # 左轮 z 超过贴地高度 2cm = 离地
_ALIGN_BAD = 0.80  # up 或 basex 对齐度低于此 = 大幅倾覆
_MIN_BASE_Z = 0.30


@dataclass
class XqRobotWLSingleLegUnicycleCommands(Commands):
    # 5D: [vx, vy, vyaw, tsk, unicycle_trigger]。
    vel_limit: list[list[float]] = field(
        default_factory=lambda: [
            [-0.30, 0.0, 0.0, 0.0, 0.0],
            [0.30, 0.0, 0.0, 0.0, 1.0],
        ]
    )
    resampling_time: float = 3.0


@dataclass
class XqRobotWLSingleLegUnicycleRewardConfig(XqRobotWLRewardConfig):
    scales: dict[str, float] = field(default_factory=dict)
    tracking_sigma: float = 0.3
    base_height_target: float = _BASE_Z
    only_positive_rewards: bool = False
    max_tilt_deg: float = 45.0
    min_base_height: float = _MIN_BASE_Z
    command_start_speed: float = 0.05
    command_curriculum_steps: int = 50_000_000
    # Optimizer checkpoints do not serialize environment counters. Resume runs
    # can restore the command curriculum explicitly with this offset.
    command_curriculum_start_steps: int = 0
    stand_command_probability: float = 0.20
    reset_ang_vel: float = 0.10
    wheel_command_kp: float = 2.0
    wheel_command_ff: float = 2.0
    start_in_unicycle: bool = False
    unicycle_trigger_probability: float = 0.70
    transition_time: float = 2.0
    transition_reset_probability: float = 0.0
    transition_reset_min_progress: float = 0.20
    stand_wheel_kv: float = 1.0


def _update_mode_fsm(
    state: np.ndarray,
    timer: np.ndarray,
    trigger: np.ndarray,
    dt: float,
    transition_time: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Latch the H-key mode and run reversible stand/unicycle transitions."""
    timer += dt
    start = (state == -1) & (trigger > 0.5)
    state[start], timer[start] = 0, 0.0
    reached = (state == 0) & (timer >= transition_time)
    state[reached], timer[reached] = 1, 0.0
    release = (state == 1) & (trigger < 0.5)
    state[release], timer[release] = 2, 0.0
    returned = (state == 2) & (timer >= transition_time)
    state[returned], timer[returned] = -1, 0.0
    return state, timer


def _mode_progress(state: np.ndarray, timer: np.ndarray, transition_time: float) -> np.ndarray:
    """Smooth 0=two-wheel stand to 1=unicycle interpolation coordinate."""
    duration = max(float(transition_time), 1.0e-6)
    phase = np.clip(timer / duration, 0.0, 1.0)
    smooth = phase * phase * (3.0 - 2.0 * phase)
    progress = np.zeros_like(timer, dtype=np.float64)
    progress[state == 0] = smooth[state == 0]
    progress[state == 1] = 1.0
    progress[state == 2] = 1.0 - smooth[state == 2]
    return progress


def _leg_transition_progress(progress: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Roll onto the left support before deploying the right counterweight."""
    support_phase = np.clip((progress - 0.15) / 0.55, 0.0, 1.0)
    counterweight_phase = np.clip((progress - 0.80) / 0.20, 0.0, 1.0)
    support = support_phase * support_phase * (3.0 - 2.0 * support_phase)
    counterweight = counterweight_phase * counterweight_phase * (3.0 - 2.0 * counterweight_phase)
    return support, counterweight


def _transition_leg_target(progress: np.ndarray) -> np.ndarray:
    """Three-stage path: tuck free wheel, roll on left wheel, deploy counterweight."""
    progress = np.asarray(progress, dtype=np.float64)
    tuck_phase = np.clip(progress / 0.70, 0.0, 1.0)
    tuck = tuck_phase * tuck_phase * (3.0 - 2.0 * tuck_phase)
    support, deploy = _leg_transition_progress(progress)
    target = np.tile(_STAND_POSE, (len(progress), 1))
    target[:, :3] += support[:, None] * (_POSE_DEFAULT[:3] - _STAND_POSE[:3])
    tucked_right = _STAND_POSE[3:] + tuck[:, None] * (_RIGHT_TUCK_POSE - _STAND_POSE[3:])
    target[:, 3:] = tucked_right + deploy[:, None] * (_POSE_DEFAULT[3:] - tucked_right)
    return target


@registry.envcfg("XqRobotWLSingleLegUnicycle")
@dataclass
class XqRobotWLSingleLegUnicycleCfg(XqRobotWLWalkFlatCfg):
    commands: XqRobotWLSingleLegUnicycleCommands = field(
        default_factory=XqRobotWLSingleLegUnicycleCommands
    )
    reward_config: XqRobotWLSingleLegUnicycleRewardConfig | None = None
    max_episode_seconds: float = 8.0


# ── 奖励 ──


def _reward_upright(ctx: RewardContext) -> np.ndarray:
    """Track the smooth standing-to-sideways orientation reference."""
    up = np.asarray(ctx.gravity)[: ctx.num_envs]
    xvec = np.asarray(ctx.info.get("basex", np.tile(_XVEC_REF, (ctx.num_envs, 1))))
    progress = np.asarray(ctx.info.get("unicycle_orientation_progress", np.ones(ctx.num_envs)))
    angle = progress * (np.pi / 2.0)
    up_ref = np.stack([np.zeros_like(angle), -np.sin(angle), np.cos(angle)], axis=1)
    roll_err = np.arccos(np.clip(np.sum(up * up_ref, axis=1), -1.0, 1.0))
    pitch_err = np.arccos(np.clip(xvec @ _XVEC_REF, -1.0, 1.0))
    fsm = np.asarray(ctx.info.get("unicycle_fsm_state", np.ones(ctx.num_envs)))
    transitioning = np.isin(fsm, [0, 2])
    roll_sigma = np.where(transitioning, 0.35, 0.10)
    pitch_sigma = np.where(transitioning, 0.45, 0.22)
    return np.exp(-np.square(roll_err / roll_sigma) - np.square(pitch_err / pitch_sigma))


def _reward_tracking_vx(ctx: RewardContext) -> np.ndarray:
    """单轮纵向速度跟踪，不把正常前进误当成漂移。"""
    error = ctx.info["commands"][:, 0] - ctx.linvel[:, 0]
    progress = np.asarray(ctx.info.get("unicycle_mode_progress", np.ones(ctx.num_envs)))
    return np.exp(-np.square(error) / max(ctx.tracking_sigma, 1.0e-6)) * progress


def _reward_directional_progress(ctx: RewardContext) -> np.ndarray:
    """奖励命令方向上的进展，零速命令不产生该项。"""
    cmd = ctx.info["commands"][:, 0]
    vx = ctx.linvel[:, 0]
    progress = np.asarray(ctx.info.get("unicycle_mode_progress", np.ones(ctx.num_envs)))
    return np.clip(cmd * vx / 0.09, -1.0, 1.0) * progress


def _reward_wheel_down(ctx: RewardContext) -> np.ndarray:
    """支撑轮(左)贴地: z 接近 WHEEL_R 给奖。"""
    lw_z = np.asarray(ctx.info.get("left_wheel_z", np.full(ctx.num_envs, _WHEEL_R)))
    err = np.abs(lw_z - _WHEEL_R)
    return np.clip((0.03 - err) / 0.03, 0.0, 1.0)


def _reward_counterweight_pose(ctx: RewardContext) -> np.ndarray:
    """配重姿态: R_roll≈-1.50, R_pitch≈0, R_knee≈0 (伸直)。"""
    dp = ctx.dof_pos[: ctx.num_envs, 3:6]
    r_roll = dp[:, 0] - _CW_ROLL0
    r_pitch = dp[:, 1] - _CW_PITCH0
    r_knee = dp[:, 2] - _CW_KNEE0
    progress = np.asarray(ctx.info.get("unicycle_mode_progress", np.ones(ctx.num_envs)))
    return np.sum(np.square(np.stack([r_roll, r_pitch, r_knee], axis=1)), axis=1) * progress


def _reward_damping(ctx: RewardContext) -> np.ndarray:
    """角速度阻尼: 罚 gyro 幅度 (保持静止平衡)。"""
    gyro = ctx.gyro[: ctx.num_envs]
    fsm = np.asarray(ctx.info.get("unicycle_fsm_state", np.ones(ctx.num_envs)))
    steady = np.isin(fsm, [-1, 1]).astype(np.float64)
    return np.sum(np.square(gyro), axis=1) * steady


def _reward_lateral_drift(ctx: RewardContext) -> np.ndarray:
    """只惩罚横向漂移；纵向位移是单轮行走的任务目标。"""
    xy = np.asarray(ctx.info.get("base_xy", np.zeros((ctx.num_envs, 2))))
    fsm = np.asarray(ctx.info.get("unicycle_fsm_state", np.ones(ctx.num_envs)))
    steady = np.isin(fsm, [-1, 1]).astype(np.float64)
    return np.square(xy[:, 1]) * steady


def _reward_stop_hold(ctx: RewardContext) -> np.ndarray:
    """零速命令时约束纵向位置与速度，保留原静态平衡能力。"""
    cmd = ctx.info["commands"][:, 0]
    fsm = np.asarray(ctx.info.get("unicycle_fsm_state", np.ones(ctx.num_envs)))
    stopped = ((np.abs(cmd) < 0.02) & np.isin(fsm, [-1, 1])).astype(np.float64)
    xy = np.asarray(ctx.info.get("base_xy", np.zeros((ctx.num_envs, 2))))
    return (np.square(xy[:, 0]) + 0.5 * np.square(ctx.linvel[:, 0])) * stopped


def _reward_free_wheel_clearance(ctx: RewardContext) -> np.ndarray:
    """自由轮必须明显离地，防止退化成双轮行走。"""
    z = np.asarray(ctx.info.get("right_wheel_z", np.full(ctx.num_envs, _WHEEL_R)))
    progress = np.asarray(ctx.info.get("unicycle_mode_progress", np.ones(ctx.num_envs)))
    return np.clip((z - (_WHEEL_R + 0.05)) / 0.15, 0.0, 1.0) * progress


def _reward_lateral_velocity(ctx: RewardContext) -> np.ndarray:
    fsm = np.asarray(ctx.info.get("unicycle_fsm_state", np.ones(ctx.num_envs)))
    steady = np.isin(fsm, [-1, 1]).astype(np.float64)
    return np.square(ctx.linvel[:, 1]) * steady


def _reward_mode_complete(ctx: RewardContext) -> np.ndarray:
    fsm = np.asarray(ctx.info.get("unicycle_fsm_state", np.zeros(ctx.num_envs)))
    timer = np.asarray(ctx.info.get("unicycle_fsm_timer", np.ones(ctx.num_envs)))
    up = np.asarray(ctx.gravity)
    basex = np.asarray(ctx.info.get("basex", np.tile(_XVEC_REF, (ctx.num_envs, 1))))
    left_z = np.asarray(ctx.info.get("left_wheel_z", np.full(ctx.num_envs, _WHEEL_R)))
    right_z = np.asarray(ctx.info.get("right_wheel_z", np.full(ctx.num_envs, _WHEEL_R)))
    physical_goal = (
        (up @ _UP_REF > 0.90)
        & (basex @ _XVEC_REF > 0.90)
        & (left_z < _WHEEL_R + 0.03)
        & (right_z > _WHEEL_R + 0.05)
    )
    return ((fsm == 1) & (timer <= 1.0e-6) & physical_goal).astype(np.float64)


def _reward_stand_height(ctx: RewardContext) -> np.ndarray:
    """Keep the initial two-wheel mode at a usable standing height."""
    fsm = np.asarray(ctx.info.get("unicycle_fsm_state", np.ones(ctx.num_envs)))
    standing = (fsm == -1).astype(np.float64)
    return np.exp(-np.square((ctx.base_height - 0.55) / 0.10)) * standing


def _reward_wheel_effort(ctx: RewardContext) -> np.ndarray:
    return np.square(ctx.info["current_actions"][:, 6])


def _reward_action_rate(ctx: RewardContext) -> np.ndarray:
    cur = ctx.info["current_actions"][:, :8]
    lst = ctx.info["last_actions"][:, :8]
    return np.sum(np.square(cur - lst), axis=1)


class XqRobotWLSingleLegUnicycleDRProvider(XqRobotWLDRProvider):
    """Reset either in normal standing or in the legacy unicycle pose."""

    def _sample_commands(self, env: object, num_reset: int) -> np.ndarray:
        return env._sample_motion_commands(num_reset)  # type: ignore[attr-defined,no-any-return]

    def build_reset_plan(self, env: Any, env_ids: np.ndarray) -> ResetPlan:
        if not getattr(env._reward_cfg, "start_in_unicycle", False):
            plan = super().build_reset_plan(env, env_ids)
            probability = float(getattr(env._reward_cfg, "transition_reset_probability", 0.0))
            env._reset_mode_timer_target[env_ids] = 0.0
            reset_state = np.full(len(env_ids), -1, dtype=np.int32)
            reset_timer = np.zeros(len(env_ids), dtype=get_global_dtype())
            reset_progress = np.zeros(len(env_ids), dtype=get_global_dtype())
            reset_orientation = np.zeros(len(env_ids), dtype=get_global_dtype())
            plan.info_updates["unicycle_fsm_state"] = reset_state
            plan.info_updates["unicycle_fsm_timer"] = reset_timer
            plan.info_updates["unicycle_mode_progress"] = reset_progress
            plan.info_updates["unicycle_orientation_progress"] = reset_orientation
            selected = np.random.random(len(env_ids)) < probability
            if not np.any(selected):
                return plan

            minimum = float(getattr(env._reward_cfg, "transition_reset_min_progress", 0.20))
            phase = np.random.uniform(minimum, 0.98, int(np.count_nonzero(selected)))
            smooth = phase * phase * (3.0 - 2.0 * phase)
            support, _ = _leg_transition_progress(smooth)
            target = _transition_leg_target(smooth)
            rows = np.flatnonzero(selected)
            angle = support * (np.pi / 2.0)
            plan.qpos[rows, 3] = np.cos(angle / 2.0)
            plan.qpos[rows, 4] = np.sin(angle / 2.0)
            plan.qpos[rows, 5:7] = 0.0
            plan.qpos[rows, 7:10] = target[:, :3]
            plan.qpos[rows, 10] = 0.0
            plan.qpos[rows, 11:14] = target[:, 3:]
            plan.qpos[rows, 14] = 0.0
            plan.qvel[rows] = 0.0

            # Place the support wheel exactly on the floor for every sampled
            # pose. This is a reset-only kinematic calculation, not a runtime
            # teleport in the deployed H-key transition.
            import mujoco

            model = env._backend._model  # type: ignore[attr-defined]
            data = mujoco.MjData(model)
            sensor_adr = int(np.asarray(model.sensor("left_wheel_world_pos").adr).reshape(-1)[0])
            for row in rows:
                data.qpos[:] = plan.qpos[row]
                data.qpos[2] = 0.0
                mujoco.mj_forward(model, data)
                plan.qpos[row, 2] = _WHEEL_R - float(data.sensordata[sensor_adr + 2])

            plan.info_updates["commands"][rows, 4] = 1.0
            duration = float(getattr(env._reward_cfg, "transition_time", 2.0))
            env._reset_mode_timer_target[env_ids[rows]] = phase * duration
            reset_state[rows] = 0
            reset_timer[rows] = phase * duration
            reset_progress[rows] = smooth
            reset_orientation[rows] = support
            return plan
        num_reset = len(env_ids)
        # 横躺 quat = R_x(90°) → up=[0,-1,0]
        qpos = np.tile(
            np.array(
                [
                    0.0,
                    0.0,
                    _BASE_Z,
                    0.70710678,
                    0.70710678,
                    0.0,
                    0.0,
                    _SUPPORT_POSE[0],
                    _SUPPORT_POSE[1],
                    _SUPPORT_POSE[2],
                    0.0,  # 左腿
                    _CW_ROLL0,
                    _CW_PITCH0,
                    _CW_KNEE0,
                    0.0,  # 右腿
                ],
                dtype=get_global_dtype(),
            ),
            (num_reset, 1),
        )
        qvel = np.zeros((num_reset, 14), dtype=get_global_dtype())
        ang_vel = float(getattr(env._cfg.reward_config, "reset_ang_vel", 0.0))
        qvel[:, 3:6] = np.random.uniform(-ang_vel, ang_vel, size=(num_reset, 3))
        randomization = build_common_reset_randomization(env, num_reset)
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
            randomization=randomization,
        )


@registry.env("XqRobotWLSingleLegUnicycle", sim_backend="mujoco")
class XqRobotWLSingleLegUnicycleEnv(XqRobotWLWalkFlatEnv):
    """改进结构单轮平衡 — 横躺独轮车式 (轮扭矩pitch + 配重roll)。"""

    _cfg: XqRobotWLSingleLegUnicycleCfg

    def __init__(self, cfg: XqRobotWLSingleLegUnicycleCfg, num_envs=1, backend_type="mujoco"):
        super().__init__(cfg, num_envs=num_envs, backend_type=backend_type)
        self._dr_manager._provider = XqRobotWLSingleLegUnicycleDRProvider()  # type: ignore[union-attr]
        # 重设 obs 帧维度 (加 basex 3D)
        self._obs_frame_dim = _OBS_FRAME_DIM
        self._critic_frame_dim = _CRITIC_FRAME_DIM
        self._obs_history = np.zeros(
            (num_envs, self._hist_len, self._obs_frame_dim), dtype=get_global_dtype()
        )
        self._critic_history = np.zeros(
            (num_envs, self._hist_len, self._critic_frame_dim), dtype=get_global_dtype()
        )
        self._command_env_steps = int(
            getattr(self._reward_cfg, "command_curriculum_start_steps", 0)
        )
        start_mode = 1 if getattr(self._reward_cfg, "start_in_unicycle", False) else -1
        self._mode_state = np.full(num_envs, start_mode, dtype=np.int32)
        self._mode_timer = np.zeros(num_envs, dtype=np.float64)
        self._mode_progress = np.full(num_envs, float(start_mode == 1), dtype=np.float64)
        self._orientation_progress = self._mode_progress.copy()
        self._reset_mode_timer_target = np.zeros(num_envs, dtype=np.float64)
        # Keep the XML actuator dynamics unchanged so the mature WalkFlat
        # policy remains bit-for-bit compatible before H is pressed.  The
        # stronger unicycle PD and wheel torque are converted to equivalent
        # position/velocity commands in ``apply_action`` instead.

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {
            "obs": self._obs_frame_dim * self._hist_len,
            "critic": self._critic_frame_dim * self._hist_len,
        }

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        """钉住支撑腿 + 配重基准, RL 控 [R_roll, R_pitch, L_wheel]。

        策略动作序 [L_roll,L_pitch,L_knee,R_roll,R_pitch,R_knee,L_wheel,R_wheel]:
          - 支撑腿(L_roll,L_pitch,L_knee) 钉在 _SUPPORT_POSE
          - R_roll/R_pitch 配重: 基准 + RL 增量
          - L_wheel: RL 扭矩 (clipped ±20 Nm)
          - R_knee 钉 0, R_wheel 钉 0
        ctrl (MuJoCo 序) = [L_roll,L_pitch,L_knee,L_wheel,R_roll,R_pitch,R_knee,R_wheel]
        """
        raw = np.asarray(actions, dtype=get_global_dtype())
        standing_mask = (self._mode_state[: raw.shape[0]] == -1)[:, None]
        stand_actions = np.clip(raw, -100.0, 100.0)
        mode_actions = np.clip(
            raw, -self._cfg.control_config.clip_actions, self._cfg.control_config.clip_actions
        )
        clipped = np.where(standing_mask, stand_actions, mode_actions).astype(
            get_global_dtype(), copy=False
        )
        state.info["last_actions"] = state.info.get("current_actions", np.zeros_like(clipped))
        state.info["current_actions"] = clipped
        num = clipped.shape[0]
        ctrl = np.zeros((num, 8), dtype=get_global_dtype())
        progress = self._mode_progress[:num]
        target = _transition_leg_target(progress)
        transitioning = np.isin(self._mode_state[:num], [0, 2]).astype(np.float64)[:, None]
        standing = (self._mode_state[:num] == -1).astype(np.float64)[:, None]
        leg_flip = np.array([1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
        transition_residual = 0.30 * transitioning * clipped[:, :6]
        # The free leg's tuck/deploy path is a deterministic clearance action.
        # Letting an old policy perturb it was observed to put the right wheel
        # back on the floor exactly as the FSM entered unicycle mode.
        transition_residual[:, 3:6] = 0.0
        stand_residual = 0.60 * standing * clipped[:, :6] * leg_flip
        policy_leg_residual = transition_residual + stand_residual
        leg_target = target + policy_leg_residual
        # The XML leg actuator is kp=60.  Preserve the direct WalkFlat target in
        # stand mode; elsewhere emulate kp=300/kd=10 with an equivalent target.
        dof_pos = self.get_dof_pos()[:num, :6]
        dof_vel = self.get_dof_vel()[:num, :6]
        desired_leg_torque = np.clip(
            300.0 * (leg_target - dof_pos) - _LEG_DAMPING * dof_vel,
            -_LEG_TORQUE_LIMIT,
            _LEG_TORQUE_LIMIT,
        )
        unicycle_equivalent = dof_pos + desired_leg_torque / 60.0
        leg_target = np.where(standing > 0.5, leg_target, unicycle_equivalent)
        ctrl[:, 0] = leg_target[:, 0]
        ctrl[:, 1] = leg_target[:, 1]
        ctrl[:, 2] = leg_target[:, 2]
        wheel_velocity = self.get_dof_vel()[:num, 6:8]
        left_stand_target = 10.0 * clipped[:, 6]
        left_unicycle_tau = clipped[:, 6] * _WHEEL_TORQUE_SCALE
        # XML wheel control is target velocity (tau=ctrl-v).  Adding measured
        # velocity converts the desired unicycle torque back into that command.
        left_unicycle_target = left_unicycle_tau + wheel_velocity[:, 0]
        left_target = (
            standing[:, 0] * left_stand_target + (1.0 - standing[:, 0]) * left_unicycle_target
        )
        ctrl[:, 3] = np.clip(left_target, -_WHEEL_TORQUE_LIMIT, _WHEEL_TORQUE_LIMIT)
        unicycle_residual = clipped[:, 3] * _RROLL_SCALE * progress
        ctrl[:, 4] = np.clip(leg_target[:, 3] + unicycle_residual, -3.1416, 0.0)
        ctrl[:, 5] = np.clip(
            leg_target[:, 4] + clipped[:, 4] * _RPITCH_SCALE * progress, -2.094, 1.047
        )
        ctrl[:, 6] = leg_target[:, 5]
        right_stand_target = -10.0 * clipped[:, 7]
        ctrl[:, 7] = np.clip(
            right_stand_target * standing[:, 0],
            -_WHEEL_TORQUE_LIMIT,
            _WHEEL_TORQUE_LIMIT,
        )
        return ctrl

    def step(self, actions):
        """Add a stabilising velocity reference; PPO supplies the residual.

        Identification on the balanced robot shows negative left-wheel torque
        produces positive body vx.  The proportional term is the same velocity
        feedback that made the project's classical controller stable, while the
        policy learns coupling and contact corrections around it.
        """
        actions = np.asarray(actions, dtype=get_global_dtype()).copy()
        if self._state is None:
            return super().step(actions)
        commands = self._state.info.get("commands", np.zeros((self._num_envs, _NUM_CMD_DIM)))
        vx = self.get_local_linvel()[:, 0]
        cmd_vx = np.asarray(commands)[:, 0] * self._mode_progress
        kp = float(getattr(self._reward_cfg, "wheel_command_kp", 2.0))
        ff = float(getattr(self._reward_cfg, "wheel_command_ff", 2.0))
        wheel_reference = kp * (vx - cmd_vx) - ff * cmd_vx
        # Preserve the mature two-wheel expert exactly before H is pressed.
        # This asymmetric single-wheel feedback is only meaningful as the
        # support transfers to the left wheel.
        actions[:, 6] += wheel_reference * self._mode_progress
        return super().step(actions)

    def _command_speed_limit(self) -> float:
        cfg = self._reward_cfg
        full = max(
            abs(float(self._cfg.commands.vel_limit[0][0])),
            abs(float(self._cfg.commands.vel_limit[1][0])),
        )
        progress = self._command_curriculum_progress()
        start = min(float(getattr(cfg, "command_start_speed", full)), full)
        return float(start + progress * (full - start))

    def _command_curriculum_progress(self) -> float:
        duration = max(int(getattr(self._reward_cfg, "command_curriculum_steps", 1)), 1)
        return float(np.clip(self._command_env_steps / duration, 0.0, 1.0))

    def _sample_motion_commands(self, num_samples: int) -> np.ndarray:
        """课程采样 vx，并保留固定比例零速样本防止遗忘静态平衡。"""
        speed = self._command_speed_limit()
        cmds = np.zeros((num_samples, _NUM_CMD_DIM), dtype=get_global_dtype())
        full_low = float(self._cfg.commands.vel_limit[0][0])
        full_high = float(self._cfg.commands.vel_limit[1][0])
        low = -speed if full_low < 0.0 else 0.0
        high = speed if full_high > 0.0 else 0.0
        cmds[:, 0] = np.random.uniform(low, high, num_samples)
        stand_prob = float(getattr(self._reward_cfg, "stand_command_probability", 0.2))
        cmds[np.random.random(num_samples) < stand_prob, 0] = 0.0
        if getattr(self._reward_cfg, "start_in_unicycle", False):
            cmds[:, 4] = 1.0
        else:
            probability = float(getattr(self._reward_cfg, "unicycle_trigger_probability", 0.7))
            cmds[:, 4] = (np.random.random(num_samples) < probability).astype(get_global_dtype())
        return cmds

    def _update_commands(self, info: dict) -> None:
        commands = info.get("commands")
        if commands is None:
            return
        commands_arr = np.asarray(commands, dtype=get_global_dtype())
        if self._cfg.commands.resampling_time <= 0.0:
            info["command_speed_limit"] = self._command_speed_limit()
            return
        interval = max(int(round(self._cfg.commands.resampling_time / self._cfg.ctrl_dt)), 1)
        steps = np.asarray(info.get("steps", np.zeros(self._num_envs, dtype=np.uint32)))
        mask = (steps > 0) & ((steps % interval) == 0)
        if np.any(mask):
            commands_arr[mask] = self._sample_motion_commands(int(np.count_nonzero(mask)))
        info["commands"] = commands_arr
        info["command_speed_limit"] = self._command_speed_limit()

    def update_state(self, state: NpEnvState) -> NpEnvState:
        self._command_env_steps += self._num_envs
        updated = super().update_state(state)
        trigger = np.asarray(updated.info["commands"])[:, 4]
        transition_time = float(getattr(self._reward_cfg, "transition_time", 2.0))
        self._mode_state, self._mode_timer = _update_mode_fsm(
            self._mode_state, self._mode_timer, trigger, self._cfg.ctrl_dt, transition_time
        )
        self._mode_progress = _mode_progress(self._mode_state, self._mode_timer, transition_time)
        self._orientation_progress, _ = _leg_transition_progress(self._mode_progress)
        updated.info["unicycle_fsm_state"] = self._mode_state.copy()
        updated.info["unicycle_fsm_timer"] = self._mode_timer.copy()
        updated.info["unicycle_mode_progress"] = self._mode_progress.copy()
        updated.info["unicycle_orientation_progress"] = self._orientation_progress.copy()
        return updated

    def reset(self, env_ids: np.ndarray | None = None):
        out = super().reset(env_ids)
        if env_ids is None:
            env_ids = np.arange(self._num_envs, dtype=np.int32)
        ids = np.asarray(env_ids, dtype=np.int32)
        self._apply_mode_reset(ids)
        return out

    def _reset_done_envs(self) -> None:
        assert self._state is not None
        ids = np.flatnonzero(self._state.terminated | self._state.truncated).astype(np.int32)
        super()._reset_done_envs()
        self._apply_mode_reset(ids)

    def _apply_mode_reset(self, ids: np.ndarray) -> None:
        if ids.size == 0:
            return
        if getattr(self._reward_cfg, "start_in_unicycle", False):
            self._mode_state[ids] = 1
            self._mode_timer[ids] = 0.0
        else:
            timer = self._reset_mode_timer_target[ids]
            in_transition = timer > 0.0
            self._mode_state[ids] = np.where(in_transition, 0, -1)
            self._mode_timer[ids] = timer
        transition_time = float(getattr(self._reward_cfg, "transition_time", 2.0))
        self._mode_progress[ids] = _mode_progress(
            self._mode_state[ids], self._mode_timer[ids], transition_time
        )
        self._orientation_progress[ids], _ = _leg_transition_progress(self._mode_progress[ids])

    def _compute_obs(
        self, info: dict, linvel, gyro, gravity, dof_pos, dof_vel
    ) -> dict[str, np.ndarray]:
        noise_cfg = self._cfg.noise_config
        n_batch = linvel.shape[0]
        # A reset observation may contain an arbitrary subset of environments.
        # The DR provider supplies subset-aligned FSM values for that case;
        # regular full-batch observations use the live environment arrays.
        if n_batch == self._num_envs or "unicycle_mode_progress" not in info:
            info["unicycle_fsm_state"] = self._mode_state[:n_batch].copy()
            info["unicycle_fsm_timer"] = self._mode_timer[:n_batch].copy()
            info["unicycle_mode_progress"] = self._mode_progress[:n_batch].copy()
            info["unicycle_orientation_progress"] = self._orientation_progress[:n_batch].copy()
        mode_progress = np.asarray(info["unicycle_mode_progress"], dtype=get_global_dtype())
        basex = np.asarray(self._backend.get_sensor_data("basexvector"), dtype=get_global_dtype())[
            :n_batch
        ]
        # Keep the first 33 actor features layout-compatible with WalkFlat so
        # its mature standing policy can initialize this multimode policy.
        leg_diff = dof_pos[:, :NUM_LEG_ACTIONS] - _STAND_POSE
        leg_vel = dof_vel[:, :NUM_LEG_ACTIONS]
        wheel_vel = dof_vel[:, NUM_LEG_ACTIONS:]
        noisy_gyro = self._obs_noise(gyro, noise_cfg.scale_gyro)
        noisy_gravity = self._obs_noise(gravity, noise_cfg.scale_gravity)
        noisy_basex = self._obs_noise(basex, noise_cfg.scale_gravity)
        noisy_leg_diff = self._obs_noise(leg_diff, noise_cfg.scale_joint_angle)
        noisy_leg_vel = self._obs_noise(leg_vel, noise_cfg.scale_joint_vel)
        noisy_wheel_vel = self._obs_noise(wheel_vel, noise_cfg.scale_wheel_vel)
        last_actions = info.get("current_actions", np.zeros((linvel.shape[0], 8)))
        commands = info["commands"]
        # Blend measured velocity in gradually.  At curriculum start the five
        # slots match the old static checkpoint closely; by the end they expose
        # the full velocity feedback required for moving balance.
        velocity_blend = self._command_curriculum_progress()
        motion_features = np.stack(
            [
                commands[:, 0],
                velocity_blend * linvel[:, 0],
                velocity_blend * linvel[:, 1],
                commands[:, 0] - velocity_blend * linvel[:, 0],
                _WALK_STAND_HEIGHT_COMMAND + (_BASE_Z - _WALK_STAND_HEIGHT_COMMAND) * mode_progress,
            ],
            axis=1,
        )

        obs_frame = np.concatenate(
            [
                noisy_gyro,
                -noisy_gravity,
                noisy_leg_diff,
                noisy_leg_vel,
                noisy_wheel_vel,
                last_actions,
                motion_features,
                noisy_basex,
            ],
            axis=1,
            dtype=get_global_dtype(),
        )
        critic_frame = np.concatenate(
            [
                gyro,
                -gravity,
                leg_diff,
                leg_vel,
                wheel_vel,
                last_actions,
                motion_features,
                linvel,
                basex,
            ],
            axis=1,
            dtype=get_global_dtype(),
        )
        batch_size = obs_frame.shape[0]
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
        return {
            "obs": self._obs_history[:batch_size].reshape(batch_size, -1),
            "critic": self._critic_history[:batch_size].reshape(batch_size, -1),
        }

    def _compute_terminated(self, gravity: np.ndarray, dof_pos: np.ndarray) -> np.ndarray:
        base_z = np.asarray(self._backend.get_base_pos(), dtype=get_global_dtype())[:, 2]
        lw_pos = np.asarray(
            self._backend.get_sensor_data("left_wheel_world_pos"), dtype=get_global_dtype()
        )
        lw_z = lw_pos[:, 2]
        in_unicycle = self._mode_state == 1
        wheel_lost = (lw_z > _WHEEL_R + _WHEEL_LIFT_THRESHOLD) & in_unicycle
        rw_pos = np.asarray(
            self._backend.get_sensor_data("right_wheel_world_pos"), dtype=get_global_dtype()
        )
        free_wheel_down = (rw_pos[:, 2] < _WHEEL_R + 0.03) & in_unicycle
        basex = self._backend.get_sensor_data("basexvector")
        roll_a = gravity[:, :3] @ _UP_REF
        pitch_a = basex @ _XVEC_REF
        unicycle_tilted = ((roll_a < _ALIGN_BAD) | (pitch_a < _ALIGN_BAD)) & in_unicycle
        standing = self._mode_state == -1
        stand_tilted = (gravity[:, 2] < np.cos(np.deg2rad(45.0))) & standing
        transitioning = np.isin(self._mode_state, [0, 2])
        min_height = np.where(in_unicycle, _MIN_BASE_Z, np.where(transitioning, 0.05, 0.20))
        height_bad = base_z < min_height
        return np.asarray(
            wheel_lost | free_wheel_down | unicycle_tilted | stand_tilted | height_bad,
            dtype=bool,
        )

    def _init_reward_functions(self) -> None:
        self._reward_fns: dict[str, Any] = {
            "tracking_vx": _reward_tracking_vx,
            "directional_progress": _reward_directional_progress,
            "upright": _reward_upright,
            "wheel_down": _reward_wheel_down,
            "free_wheel_clearance": _reward_free_wheel_clearance,
            "counterweight_pose": _reward_counterweight_pose,
            "damping": _reward_damping,
            "lateral_drift": _reward_lateral_drift,
            "lateral_velocity": _reward_lateral_velocity,
            "stop_hold": _reward_stop_hold,
            "wheel_effort": _reward_wheel_effort,
            "action_rate": _reward_action_rate,
            "mode_complete": _reward_mode_complete,
            "stand_height": _reward_stand_height,
            "alive": rewards.alive,
        }

    def _compute_reward(self, info: dict, linvel, gyro, gravity, dof_pos, dof_vel) -> np.ndarray:
        dtype = get_global_dtype()
        num_obs = linvel.shape[0]
        info["unicycle_fsm_state"] = self._mode_state[:num_obs].copy()
        info["unicycle_fsm_timer"] = self._mode_timer[:num_obs].copy()
        info["unicycle_mode_progress"] = self._mode_progress[:num_obs].copy()
        info["unicycle_orientation_progress"] = self._orientation_progress[:num_obs].copy()
        info["basex"] = np.asarray(self._backend.get_sensor_data("basexvector"), dtype=dtype)
        lw_pos = np.asarray(self._backend.get_sensor_data("left_wheel_world_pos"), dtype=dtype)
        info["left_wheel_z"] = lw_pos[:, 2]
        rw_pos = np.asarray(self._backend.get_sensor_data("right_wheel_world_pos"), dtype=dtype)
        info["right_wheel_z"] = rw_pos[:, 2]
        base_pos = np.asarray(self._backend.get_base_pos(), dtype=dtype)
        info["base_xy"] = base_pos[:, :2]
        ctx = RewardContext(
            info=info,
            linvel=linvel,
            gyro=gyro,
            dof_pos=dof_pos[:, :NUM_LEG_ACTIONS],
            dof_vel=dof_vel[:, :NUM_LEG_ACTIONS],
            num_envs=num_obs,
            default_angles=_POSE_DEFAULT,
            tracking_sigma=self._reward_cfg.tracking_sigma,
            base_height_target=self._reward_cfg.base_height_target,
            base_height=self._base_height_values(num_obs),
            gravity=gravity,
            joint_range=None,
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
