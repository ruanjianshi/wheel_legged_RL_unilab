"""xqrobotwl Wheeled-SRL 跳跃环境 -- SLIP模型前馈 + PPO反馈融合

Wheeled-SRL 核心改进 vs 纯PPO:
  1. SLIP模型: 六状态FSM生成前馈参考轨迹
  2. 前馈-反馈融合: action = feedforward + k_b * policy_output
  3. 轮地速度匹配奖励 + 着陆冲击惩罚
  4. 扩展观测含 fsm_state + jump_phase

跳跃流程:
  jump_trigger > 0.5 -> FSM进入跳跃循环
  FSM: -1(初始化)->0(地面压缩)->1(跳跃加速)->2(飞行轮速调制)->3(着陆缓冲)->4(恢复)

Joint order: [L_hip_roll, L_hip_pitch, L_knee, R_hip_roll, R_hip_pitch, R_knee, L_wheel, R_wheel]
Mirror symmetry: L(+) <-> R(-) for hip_roll/hip_pitch, L(+) <-> R(-) for knee (specified in base.py)

消融实验: ablation_mode 控制开关
  - "full": 完整 Wheeled-SRL
  - "no_fsm": FSM固定为state=-1, 不做前馈, action=纯PPO输出
  - "no_wheel_match": 移除 wheel_ground_matching 奖励
  - "no_flight_mod": 飞行阶段轮速不斜升 (置零)
  - "no_vel_track": 移除 tracking_lin_vel 奖励
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
    DEFAULT_ANGLES,
    DEFAULT_LEG_ANGLES,
    NUM_ACTIONS,
    NUM_LEG_ACTIONS,
    NUM_WHEEL_ACTIONS,
)
from unilab.envs.locomotion.xqrobotwl.joystick import (
    XqRobotWLCurriculumConfig,
    XqRobotWLDRProvider,
    XqRobotWLWalkFlatCfg,
    XqRobotWLWalkFlatEnv,
)

_NUM_JUMP_CMD_DIM = 5

# === Wheeled-SRL 参数 ===
WHEEL_R = 0.065  # xqrobotwl 车轮半径 (m)


# ========== SLIP前馈计算 ==========


def compute_slip_feedforward(
    fsm_state: np.ndarray,
    fsm_timer: np.ndarray,
    dof_pos: np.ndarray,
    base_linvel: np.ndarray,
    default_angles: np.ndarray,
    wheel_r: float,
    ablation_mode: str = "full",
) -> np.ndarray:
    """六状态SLIP前馈: 输出(envs, 8)

    消融控制:
      - "no_fsm": 全部置零, action=纯PPO
      - "no_flight_mod": 飞行阶段轮速置零
    """
    num_envs = fsm_state.shape[0]
    ff = np.zeros((num_envs, NUM_ACTIONS), dtype=np.float64)

    if ablation_mode == "no_fsm":
        return ff

    for s in range(-1, 5):
        mask = fsm_state == s
        if not mask.any():
            continue

        if s == -1:  # 站立--保持默认姿态
            ff[mask, :NUM_LEG_ACTIONS] = default_angles[:NUM_LEG_ACTIONS]
            ff[mask, NUM_LEG_ACTIONS:] = 0.0

        elif s == 0:  # 地面接触--浅蹲
            ff[mask, 0] = 0.1
            ff[mask, 1] = 0.1
            ff[mask, 2] = 0.2
            ff[mask, 3] = -0.1
            ff[mask, 4] = -0.1
            ff[mask, 5] = -0.2
            ff[mask, 6:8] = 0.0

        elif s == 1:  # 跳跃加速--伸直推地
            ff[mask, 0] = 0.1
            ff[mask, 1] = 0.0
            ff[mask, 2] = -0.3
            ff[mask, 3] = -0.1
            ff[mask, 4] = 0.0
            ff[mask, 5] = 0.3
            ff[mask, 6:8] = 0.0

        elif s == 2:  # 飞行--收腿+轮速匹配
            progress = np.clip(fsm_timer[mask] / 0.3, 0.0, 1.0)
            grasp = 0.2 * (1 - progress)
            ff[mask, 0] = 0.1
            ff[mask, 1] = 0.15 - grasp
            ff[mask, 2] = grasp
            ff[mask, 3] = -0.1
            ff[mask, 4] = -0.15 + grasp
            ff[mask, 5] = -grasp
            if ablation_mode != "no_flight_mod":
                ground_vel = np.abs(base_linvel[mask, 0])
                target_rps = ground_vel / wheel_r
                ff[mask, 6] = target_rps * 0.5
                ff[mask, 7] = target_rps * 0.5
            else:
                ff[mask, 6:8] = 0.0

        elif s == 3:  # 着陆--缓冲
            lp = np.clip(fsm_timer[mask] / 0.1, 0.0, 1.0)
            bend = 0.3 * lp
            ff[mask, 0] = 0.1
            ff[mask, 1] = 0.15 + bend * 0.5
            ff[mask, 2] = bend
            ff[mask, 3] = -0.1
            ff[mask, 4] = -0.15 - bend * 0.5
            ff[mask, 5] = -bend
            gv = base_linvel[mask, 0]
            sync = gv / wheel_r
            ff[mask, 6] = sync * 0.5
            ff[mask, 7] = sync * 0.5

        elif s == 4:  # 恢复--回到站姿
            r = np.clip(fsm_timer[mask] / 0.2, 0.0, 1.0)
            cur = dof_pos[mask, :NUM_LEG_ACTIONS]
            df = default_angles[:NUM_LEG_ACTIONS]
            ff[mask, :NUM_LEG_ACTIONS] = df + (cur - df) * (1 - r[:, None])
            ff[mask, NUM_LEG_ACTIONS:] = 0.0

    return ff


def update_fsm(
    fsm_state: np.ndarray,
    fsm_timer: np.ndarray,
    base_height: np.ndarray,
    base_linvel: np.ndarray,
    dof_pos: np.ndarray,
    jump_trigger: np.ndarray,
    default_height: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """更新六状态FSM -- xqrobotwl 专用阈值"""
    v_z = base_linvel[:, 2]
    ground = base_height < default_height + 0.02
    fsm_timer += dt

    for s in range(-1, 5):
        mask = fsm_state == s
        if not mask.any():
            continue

        if s == -1:
            trigger = (jump_trigger[mask] > 0.5) & (fsm_timer[mask] > 0.1)
            next_mask = np.zeros_like(mask, dtype=bool)
            next_mask[mask] = trigger
            fsm_state[next_mask] = 0
            fsm_timer[next_mask] = 0.0

        elif s == 0:
            deep = (dof_pos[mask, 2] < 0.0) & (dof_pos[mask, 5] > 0.0)
            to = fsm_timer[mask] > 0.1
            next_mask = np.zeros_like(mask, dtype=bool)
            next_mask[mask] = deep | to
            fsm_state[next_mask] = 1
            fsm_timer[next_mask] = 0.0

        elif s == 1:
            if not np.any(fsm_timer[mask] > 0.05):
                continue
            ext = (dof_pos[mask, 2] > 0.05) & (dof_pos[mask, 5] < -0.05)
            to = fsm_timer[mask] > 0.2
            next_mask = np.zeros_like(mask, dtype=bool)
            next_mask[mask] = ext | to
            fsm_state[next_mask] = 2
            fsm_timer[next_mask] = 0.0

        elif s == 2:
            if not np.any(fsm_timer[mask] > 0.1):
                continue
            descending = v_z[mask] < 0
            near = base_height[mask] < default_height + 0.2
            to = fsm_timer[mask] > 0.8
            next_mask = np.zeros_like(mask, dtype=bool)
            next_mask[mask] = descending & (near | to)
            fsm_state[next_mask] = 3
            fsm_timer[next_mask] = 0.0

        elif s == 3:
            landed = ground[mask] & (fsm_timer[mask] > 0.05)
            next_mask = np.zeros_like(mask, dtype=bool)
            next_mask[mask] = landed
            fsm_state[next_mask] = 4
            fsm_timer[next_mask] = 0.0

        elif s == 4:
            ok = fsm_timer[mask] > 0.2
            next_mask = np.zeros_like(mask, dtype=bool)
            next_mask[mask] = ok
            fsm_state[next_mask] = -1
            fsm_timer[next_mask] = 0.0

    return fsm_state, fsm_timer


# ========== 奖励函数 ==========


def _reward_jump_height(ctx: RewardContext, cfg) -> np.ndarray:
    base_z = ctx.base_height
    jt = ctx.info["commands"][:, 4]
    jumping = jt > 0.5
    clamped = np.clip(base_z / cfg.jump_height_target, 0.0, 1.0)
    return clamped * jumping.astype(np.float64) * 2.0


def _reward_crouch_prep(ctx: RewardContext, cfg) -> np.ndarray:
    base_z = ctx.base_height
    jt = ctx.info["commands"][:, 4]
    crouching = (jt > 0.5) & (base_z < cfg.base_height_target)
    target = cfg.crouch_height_target
    height_ok = (base_z > cfg.min_base_height) & (base_z < target + 0.1)
    return height_ok.astype(np.float64) * crouching.astype(np.float64) * 0.5


def _reward_landing_soft(ctx: RewardContext) -> np.ndarray:
    vz = np.abs(ctx.linvel[:, 2])
    return np.exp(-vz / 0.5) * 0.3


def _reward_vertical_thrust(ctx: RewardContext, cfg) -> np.ndarray:
    jt = ctx.info["commands"][:, 4]
    base_z = ctx.base_height
    vz = ctx.linvel[:, 2]
    active = (jt > 0.5) & (base_z < 0.55) & (vz > 0.0)
    return vz * active.astype(np.float64)


def _reward_crouch_depth(ctx: RewardContext, cfg) -> np.ndarray:
    base_z = ctx.base_height
    jt = ctx.info["commands"][:, 4]
    crouching = (jt > 0.5) & (base_z < cfg.base_height_target)
    depth = np.clip((cfg.base_height_target - base_z) / 0.3, 0.0, 1.0)
    return depth * crouching.astype(np.float64) * 0.5


def _reward_wheel_ground_matching(ctx: RewardContext) -> np.ndarray:
    """轮地速度匹配 (论文式14)"""
    wheel_vel = ctx.info.get("wheel_vel", np.zeros((ctx.num_envs, 2)))
    lin_x = ctx.linvel[:, 0:1]
    error = np.sum(np.square(wheel_vel * WHEEL_R - lin_x), axis=1)
    fsm = ctx.info.get("fsm_state", -np.ones(ctx.num_envs))
    timer = ctx.info.get("fsm_timer", np.ones(ctx.num_envs))
    landing = (fsm == 3) & (timer < 0.05)
    r = -error
    r[~landing] = 0.0
    return r


# ========== 配置类 ==========


@dataclass
class XqRobotWLJumpSRLCommands(Commands):
    """跳跃命令: [vx, vy, vyaw, tsk, jump_trigger]"""

    vel_limit: list[list[float]] = field(
        default_factory=lambda: [[-0.3, 0.0, -0.5, -0.1, 0], [0.3, 0.0, 0.5, 0.1, 1]]
    )
    resampling_time: float = 4.0


@dataclass
class XqRobotWLJumpSRLRewardConfig:
    """Wheeled-SRL 跳跃奖励配置"""

    scales: dict[str, float]
    tracking_sigma: float = 0.25
    base_height_target: float = 0.65
    only_positive_rewards: bool = False
    max_tilt_deg: float = 45.0
    min_base_height: float = 0.15
    jump_height_target: float = 1.0
    crouch_height_target: float = 0.40
    feedback_gain: float = 0.2
    wheel_matching_sigma: float = 0.3
    jump_warmup_steps: int = 3000


@dataclass
class XqRobotWLJumpSRLCurriculumConfig(XqRobotWLCurriculumConfig):
    enabled: bool = False


@registry.envcfg("XqRobotWLJumpSRLFlat")
@dataclass
class XqRobotWLJumpSRLFlatCfg(XqRobotWLWalkFlatCfg):
    commands: XqRobotWLJumpSRLCommands = field(default_factory=XqRobotWLJumpSRLCommands)
    reward_config: XqRobotWLJumpSRLRewardConfig | None = None
    curriculum: XqRobotWLJumpSRLCurriculumConfig = field(
        default_factory=XqRobotWLJumpSRLCurriculumConfig
    )
    max_episode_seconds: float = 10.0
    ablation_mode: str = "full"


class XqRobotWLJumpSRLDRProvider(XqRobotWLDRProvider):
    def _sample_commands(self, env, num_reset):
        low = np.asarray(env._cfg.commands.vel_limit[0], dtype=get_global_dtype())
        high = np.asarray(env._cfg.commands.vel_limit[1], dtype=get_global_dtype())
        cmds = np.asarray(
            np.random.uniform(low=low, high=high, size=(num_reset, low.shape[0])),
            dtype=get_global_dtype(),
        )
        safe_linv = np.maximum(np.abs(cmds[:, 0]), 1e-4)
        cmds[:, 2] = np.clip(cmds[:, 2], -2.0 / safe_linv, 2.0 / safe_linv)
        return cmds


@registry.env("XqRobotWLJumpSRLFlat", sim_backend="mujoco")
class XqRobotWLJumpSRLFlatEnv(XqRobotWLWalkFlatEnv):
    """Wheeled-SRL 跳跃环境 (xqrobotwl)

    在父类基础上增加:
    - SLIP六状态FSM前馈
    - 前馈-反馈融合: action = ff + k_b * policy_output
    - 轮地匹配奖励
    - 扩展观测含 fsm_state + jump_phase
    - 消融实验开关
    """

    _cfg: XqRobotWLJumpSRLFlatCfg

    def __init__(self, cfg, num_envs=1, backend_type="mujoco"):
        self._jump_cfg = cfg.reward_config
        super().__init__(cfg, num_envs=num_envs, backend_type=backend_type)
        self._init_reward_functions()
        self._dr_manager._provider = XqRobotWLJumpSRLDRProvider()

        # FSM state
        self._fsm_state = -np.ones(num_envs, dtype=np.int32)
        self._fsm_timer = np.zeros(num_envs, dtype=np.float64)
        self._jump_phase = np.zeros(num_envs, dtype=np.float64)
        self._peak_height = np.zeros(num_envs, dtype=np.float64)
        self._feedback_gain = np.full(num_envs, cfg.reward_config.feedback_gain, dtype=np.float64)

        # 消融模式
        self._ablation_mode = getattr(cfg, "ablation_mode", "full")

        # 分阶段训练预热
        warmup_iters = cfg.reward_config.jump_warmup_steps
        self._warmup_cutoff = 0 if num_envs == 1 else warmup_iters * 24
        self._obs_frame_dim = 35
        self._critic_frame_dim = 38
        self._obs_history = np.zeros(
            (num_envs, self._hist_len, self._obs_frame_dim), dtype=self._np_dtype
        )
        self._critic_history = np.zeros(
            (num_envs, self._hist_len, self._critic_frame_dim), dtype=self._np_dtype
        )
        self._total_env_steps = 0

    @property
    def obs_groups_spec(self):
        return {
            "obs": self._obs_frame_dim * self._hist_len,
            "critic": self._critic_frame_dim * self._hist_len,
        }

    def _init_reward_functions(self):
        """跳跃奖励表: 通用 + Wheeled-SRL专用"""
        self._reward_fns = {
            "tracking_lin_vel": rewards.tracking_lin_vel,
            "tracking_ang_vel": rewards.tracking_ang_vel,
            "lin_vel_z": rewards.lin_vel_z,
            "ang_vel_xy": rewards.ang_vel_xy,
            "base_height": rewards.base_height,
            "orientation": rewards.orientation,
            "joint_action_rate": rewards.action_rate,
            "wheel_action_rate": self._reward_wheel_action_rate,
            "leg_mirror": self._reward_leg_mirror,
            "tsk": self._reward_tsk,
            "alive": rewards.alive,
            "jump_height": self._reward_jump_height,
            "crouch_prep": self._reward_crouch_prep,
            "landing_soft": self._reward_landing_soft,
            "wheel_ground_matching": self._reward_wheel_ground_matching,
            "vertical_thrust": self._reward_vertical_thrust,
            "crouch_depth": self._reward_crouch_depth,
            "action_magnitude": self._reward_action_magnitude,
        }

    # 奖励委托
    def _reward_jump_height(self, ctx):
        return _reward_jump_height(ctx, self._jump_cfg)

    def _reward_crouch_prep(self, ctx):
        return _reward_crouch_prep(ctx, self._jump_cfg)

    def _reward_landing_soft(self, ctx):
        return _reward_landing_soft(ctx)

    def _reward_vertical_thrust(self, ctx):
        return _reward_vertical_thrust(ctx, self._jump_cfg)

    def _reward_crouch_depth(self, ctx):
        return _reward_crouch_depth(ctx, self._jump_cfg)

    def _reward_wheel_ground_matching(self, ctx):
        if self._ablation_mode == "no_wheel_match":
            return np.zeros(ctx.num_envs, dtype=np.float64)
        return _reward_wheel_ground_matching(ctx)

    def _reward_action_magnitude(self, ctx):
        return np.sum(np.square(ctx.info["current_actions"]), axis=1)

    def _reward_wheel_action_rate(self, ctx):
        cur = ctx.info["current_actions"][:, NUM_LEG_ACTIONS:]
        lst = ctx.info["last_actions"][:, NUM_LEG_ACTIONS:]
        return np.sum(np.square(cur - lst), axis=1)

    def _reward_leg_mirror(self, ctx):
        return np.square(ctx.dof_pos[:, 0] + ctx.dof_pos[:, 3]) + np.sum(
            np.square(ctx.dof_pos[:, 1:3] + ctx.dof_pos[:, 4:6]), axis=1
        )

    def _reward_tsk(self, ctx):
        return np.square((ctx.dof_pos[:, 0] + ctx.dof_pos[:, 3]) - ctx.info["commands"][:, 3])

    # ── 核心: step覆写 ──

    def step(self, actions):
        """前馈-反馈融合 step"""
        dof_pos = self.get_dof_pos()
        linvel = self.get_local_linvel()
        ff = compute_slip_feedforward(
            self._fsm_state,
            self._fsm_timer,
            dof_pos,
            linvel,
            self.default_angles,
            WHEEL_R,
            self._ablation_mode,
        )
        kb = self._feedback_gain.reshape(-1, 1)
        fused = ff + kb * actions
        return super().step(fused)

    # ── 状态更新 ──

    def update_state(self, state: NpEnvState) -> NpEnvState:
        self._update_commands(state.info)

        # 渐进式跳跃预热
        self._total_env_steps += self._num_envs
        warmup_end = self._warmup_cutoff
        if self._total_env_steps < warmup_end:
            state.info["commands"][:, 4] = 0.0
        elif self._total_env_steps < warmup_end * 2:
            alpha = (self._total_env_steps - warmup_end) / warmup_end
            state.info["commands"][:, 4] *= alpha

        linvel = self.get_local_linvel()
        gyro = self.get_gyro()
        gravity = self._backend.get_sensor_data(self._cfg.sensor.upvector)
        dof_pos = self.get_dof_pos()
        dof_vel = self.get_dof_vel()
        self._update_wheel_contact(state.info)

        base_height = self._base_height_values(linvel.shape[0])
        jt = state.info["commands"][:, 4]
        self._fsm_state, self._fsm_timer = update_fsm(
            self._fsm_state,
            self._fsm_timer,
            base_height,
            linvel,
            dof_pos,
            jt,
            self._cfg.reward_config.base_height_target,
            self._cfg.ctrl_dt,
        )
        self._jump_phase = np.clip(self._fsm_timer / 0.8, 0.0, 1.0)
        flying = self._fsm_state == 2
        self._peak_height[flying] = np.maximum(self._peak_height[flying], base_height[flying])

        state.info["fsm_state"] = self._fsm_state
        state.info["fsm_timer"] = self._fsm_timer
        state.info["jump_phase"] = self._jump_phase
        state.info["wheel_vel"] = dof_vel[:, NUM_LEG_ACTIONS:]

        terminated = self._compute_terminated(gravity, dof_pos)
        reward = self._compute_reward(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        obs = self._compute_obs(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        return state.replace(obs=obs, reward=reward, terminated=terminated)

    def _update_wheel_contact(self, info):
        try:
            left = self._backend.get_sensor_data("left_wheel_force")
            right = self._backend.get_sensor_data("right_wheel_force")
            lf = np.asarray(left, dtype=get_global_dtype()).reshape(-1, 3)[: self._num_envs]
            rf = np.asarray(right, dtype=get_global_dtype()).reshape(-1, 3)[: self._num_envs]
            info["wheel_contact"] = np.stack(
                [
                    (np.linalg.norm(lf, axis=1) > 10.0).astype(np.float64),
                    (np.linalg.norm(rf, axis=1) > 10.0).astype(np.float64),
                ],
                axis=1,
            )
        except (KeyError, AttributeError):
            info["wheel_contact"] = np.zeros((self._num_envs, 2), dtype=np.float64)

    def _compute_terminated(self, gravity, dof_pos):
        tilt = np.arccos(np.clip(gravity[:, 2], -1, 1))
        terminated = tilt > np.deg2rad(self._jump_cfg.max_tilt_deg)
        terminated |= (
            np.asarray(self._backend.get_base_pos(), dtype=get_global_dtype())[:, 2]
            < self._jump_cfg.min_base_height
        )
        terminated |= (dof_pos[:, 1] < -0.3) | (dof_pos[:, 4] > 0.3)
        terminated |= (np.abs(dof_pos[:, 2]) > 1.5) | (np.abs(dof_pos[:, 5]) > 1.5)
        # 轮滑移终止
        linvel = self.get_local_linvel()
        wv = self.get_dof_vel()[:, NUM_LEG_ACTIONS:]
        slip = np.any(np.abs(wv * WHEEL_R - linvel[:, 0:1]) > 1.5, axis=1)
        is_ground = (
            np.asarray(self._backend.get_base_pos(), dtype=get_global_dtype())[:, 2]
            < self._jump_cfg.base_height_target + 0.05
        )
        terminated |= slip & is_ground
        return terminated

    def _compute_obs(self, info, linvel, gyro, gravity, dof_pos, dof_vel):
        noise_cfg = self._cfg.noise_config
        leg_diff = dof_pos[:, :NUM_LEG_ACTIONS] - DEFAULT_LEG_ANGLES[:NUM_LEG_ACTIONS]
        leg_vel = dof_vel[:, :NUM_LEG_ACTIONS]
        wheel_vel = dof_vel[:, NUM_LEG_ACTIONS:]
        last_actions = info.get("current_actions", np.zeros((linvel.shape[0], NUM_ACTIONS)))

        fsm_feat = (
            self._fsm_state.astype(np.float64).reshape(-1, 1)[: linvel.shape[0]] / 5.0
        )
        phase_feat = self._jump_phase.reshape(-1, 1)[: linvel.shape[0]]

        obs_frame = np.concatenate(
            [
                self._obs_noise(gyro, noise_cfg.scale_gyro),
                -self._obs_noise(gravity, noise_cfg.scale_gravity),
                self._obs_noise(leg_diff, noise_cfg.scale_joint_angle),
                self._obs_noise(leg_vel, noise_cfg.scale_joint_vel),
                self._obs_noise(wheel_vel, noise_cfg.scale_wheel_vel),
                last_actions,
                info["commands"],
                fsm_feat,
                phase_feat,
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
                info["commands"],
                fsm_feat,
                phase_feat,
                linvel,
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

        obs = self._obs_history[:batch_size].reshape(batch_size, -1)
        critic = self._critic_history[:batch_size].reshape(batch_size, -1)
        return {"obs": obs, "critic": critic}

    def _compute_reward(self, info, linvel, gyro, gravity, dof_pos, dof_vel):
        dtype = get_global_dtype()
        ctx = RewardContext(
            info=info,
            linvel=linvel,
            gyro=gyro,
            dof_pos=dof_pos[:, :NUM_LEG_ACTIONS],
            dof_vel=dof_vel[:, :NUM_LEG_ACTIONS],
            num_envs=linvel.shape[0],
            default_angles=self.default_angles[:NUM_LEG_ACTIONS].astype(dtype),
            tracking_sigma=self._jump_cfg.tracking_sigma,
            base_height_target=self._jump_cfg.base_height_target,
            base_height=self._base_height_values(linvel.shape[0]),
            gravity=gravity,
            joint_range=None,
        )
        # 消融: 移除速度跟踪
        if self._ablation_mode == "no_vel_track":
            scales = {
                k: (0.0 if k == "tracking_lin_vel" else v) for k, v in self._jump_cfg.scales.items()
            }
        else:
            scales = self._jump_cfg.scales
        return rewards.run_reward_dispatch(
            scales=scales,
            fns=self._reward_fns,
            ctx=ctx,
            info=info,
            enable_log=self._enable_reward_log,
            ctrl_dt=self._cfg.ctrl_dt,
            only_positive=self._jump_cfg.only_positive_rewards,
        )

    def _base_height_values(self, num_obs):
        pos = np.asarray(self._backend.get_base_pos(), dtype=get_global_dtype())
        return pos[:, 2] if pos.shape[0] == num_obs else np.zeros(num_obs, dtype=get_global_dtype())
