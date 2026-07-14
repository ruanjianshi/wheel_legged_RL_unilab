"""xqrobotwl Wheeled-SRL 跳跃环境 — SLIP模型前馈 + PPO反馈融合

Wheeled-SRL 核心改进 vs 原始跳:
  1. SLIP模型: 六状态FSM生成前馈参考轨迹
  2. 前馈-反馈融合: action = feedforward + k_b * policy_output
  3. 轮地速度匹配奖励 + 着陆冲击惩罚
  4. 扩展观测含 fsm_state + jump_phase

跳跃流程:
  jump_trigger > 0.5 → FSM进入跳跃循环
  FSM: -1(初始化)→0(地面压缩)→1(跳跃加速)→2(飞行轮速调制)→3(着陆缓冲)→4(恢复)

Joint order: [L_hip_roll, L_hip_pitch, L_knee, R_hip_roll, R_hip_pitch, R_knee, L_wheel, R_wheel]
Mirror symmetry: L(+) ↔ R(-) for hip_roll/hip_pitch, L(+) ↔ R(-) for knee (specified in base.py)
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
    NUM_ACTIONS,
    DEFAULT_ANGLES,
)
from .joystick import (
    XqRobotWLCurriculumConfig,
    XqRobotWLDRProvider,
    XqRobotWLWalkFlatCfg,
    XqRobotWLWalkFlatEnv,
)

_NUM_JUMP_CMD_DIM = 5

# === Wheeled-SRL 参数 ===
SLIP_K = 5000.0
SLIP_C = 50.0
WHEEL_R = 0.065        # xqrobotwl 车轮半径
WHEEL_J = 0.005        # 轮转动惯量


@dataclass
class XqRobotWLJumpCommands(Commands):
    """跳跃命令: [vx, vy, vyaw, tsk, jump_trigger]"""
    vel_limit: list[list[float]] = field(
        default_factory=lambda: [[-0.3, 0.0, -0.5, -0.1, 0], [0.3, 0.0, 0.5, 0.1, 1]]
    )
    resampling_time: float = 4.0


@dataclass
class XqRobotWLJumpRewardConfig:
    """Wheeled-SRL 跳跃奖励配置"""
    scales: dict[str, float]
    tracking_sigma: float = 0.25
    base_height_target: float = 0.65
    only_positive_rewards: bool = False
    max_tilt_deg: float = 45.0
    min_base_height: float = 0.15
    jump_height_target: float = 1.0
    crouch_height_target: float = 0.40
    # Wheeled-SRL
    feedback_gain: float = 0.5
    wheel_matching_sigma: float = 0.3


@dataclass
class XqRobotWLJumpCurriculumConfig(XqRobotWLCurriculumConfig):
    enabled: bool = False


# ========== SLIP前馈计算 ==========

def compute_slip_feedforward(
    fsm_state, fsm_timer, dof_pos, base_linvel, default_angles,
    jump_height_target, wheel_r, dt,
) -> np.ndarray:
    """六状态SLIP前馈: 输出(envs, 8)"""
    num_envs = fsm_state.shape[0]
    ff = np.zeros((num_envs, NUM_ACTIONS), dtype=np.float64)

    for s in range(-1, 5):
        mask = fsm_state == s
        if not mask.any():
            continue
        if s == -1:  # 初始化—站立
            ff[mask, :NUM_LEG_ACTIONS] = default_angles[:NUM_LEG_ACTIONS]
            ff[mask, NUM_LEG_ACTIONS:] = 0.0
        elif s == 0:  # 地面接触—下蹲
            # hip_roll维持默认, hip_pitch前屈, knee深屈
            ff[mask, 0] = 0.1   # L_hip_roll 外展
            ff[mask, 1] = 0.05  # L_hip_pitch 微前
            ff[mask, 2] = -0.8  # L_knee 深屈 (默认+0.15)
            ff[mask, 3] = -0.1  # R_hip_roll 外展
            ff[mask, 4] = -0.05 # R_hip_pitch
            ff[mask, 5] = -0.8  # R_knee
            ff[mask, 6:8] = 0.0
        elif s == 1:  # 跳跃加速—爆发伸展
            ff[mask, 0] = 0.1
            ff[mask, 1] = 0.25  # hip_pitch前推
            ff[mask, 2] = 0.05  # knee伸展
            ff[mask, 3] = -0.1
            ff[mask, 4] = -0.25
            ff[mask, 5] = 0.05
            expected_fwd = 0.5
            ff[mask, 6] = expected_fwd / wheel_r * 0.5
            ff[mask, 7] = expected_fwd / wheel_r * 0.5
        elif s == 2:  # 飞行—轮速匹配
            progress = np.clip(fsm_timer[mask] / 0.3, 0.0, 1.0)
            grasp = 0.2 * (1 - progress)  # 收腿
            ff[mask, 0] = 0.1
            ff[mask, 1] = 0.15 - grasp
            ff[mask, 2] = 0.0 - grasp
            ff[mask, 3] = -0.1
            ff[mask, 4] = -0.15 + grasp
            ff[mask, 5] = 0.0 - grasp
            ground_vel = np.abs(base_linvel[mask, 0])
            target_rps = ground_vel / wheel_r
            ff[mask, 6] = target_rps * 0.5
            ff[mask, 7] = target_rps * 0.5
        elif s == 3:  # 着陆—缓冲
            lp = np.clip(fsm_timer[mask] / 0.1, 0.0, 1.0)
            bend = -0.15 * lp
            ff[mask, 0] = 0.1
            ff[mask, 1] = 0.15 + bend
            ff[mask, 2] = bend
            ff[mask, 3] = -0.1
            ff[mask, 4] = -0.15 - bend
            ff[mask, 5] = bend
            gv = base_linvel[mask, 0]
            ff[mask, 6] = gv / wheel_r * 0.5
            ff[mask, 7] = gv / wheel_r * 0.5
        elif s == 4:  # 恢复
            r = np.clip(fsm_timer[mask] / 0.2, 0.0, 1.0)
            cur = dof_pos[mask, :NUM_LEG_ACTIONS]
            df = default_angles[:NUM_LEG_ACTIONS]
            ff[mask, :NUM_LEG_ACTIONS] = df + (cur - df) * (1 - r[:, None])
            ff[mask, NUM_LEG_ACTIONS:] = 0.0
    return ff


def update_fsm(fsm_state, fsm_timer, base_height, base_linvel, dof_pos, jump_trigger, default_height, dt):
    """更新六状态FSM"""
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
            fsm_state[next_mask] = 0; fsm_timer[next_mask] = 0.0
        elif s == 0:
            deep = (dof_pos[mask, 2] < -0.5) & (dof_pos[mask, 5] < -0.5)
            to = fsm_timer[mask] > 0.3
            next_mask = np.zeros_like(mask, dtype=bool)
            next_mask[mask] = deep | to
            fsm_state[next_mask] = 1; fsm_timer[next_mask] = 0.0
        elif s == 1:
            ext = (dof_pos[mask, 2] > -0.1) & (dof_pos[mask, 5] > -0.1)
            to = fsm_timer[mask] > 0.2
            next_mask = np.zeros_like(mask, dtype=bool)
            next_mask[mask] = ext | to
            fsm_state[next_mask] = 2; fsm_timer[next_mask] = 0.0
        elif s == 2:
            descending = v_z[mask] < 0
            near = base_height[mask] < default_height + 0.15
            to = fsm_timer[mask] > 0.5
            next_mask = np.zeros_like(mask, dtype=bool)
            next_mask[mask] = descending & (near | to)
            fsm_state[next_mask] = 3; fsm_timer[next_mask] = 0.0
        elif s == 3:
            landed = ground[mask] & (fsm_timer[mask] > 0.05)
            next_mask = np.zeros_like(mask, dtype=bool)
            next_mask[mask] = landed
            fsm_state[next_mask] = 4; fsm_timer[next_mask] = 0.0
        elif s == 4:
            ok = fsm_timer[mask] > 0.2
            next_mask = np.zeros_like(mask, dtype=bool)
            next_mask[mask] = ok
            fsm_state[next_mask] = -1; fsm_timer[next_mask] = 0.0
    return fsm_state, fsm_timer


# ========== 奖励函数 ==========

def _reward_jump_height(ctx, cfg):
    base_z, jt = ctx.base_height, ctx.info["commands"][:, 4]
    jumping = jt > 0.5
    clamped = np.clip(base_z / cfg.jump_height_target, 0.0, 1.0)
    return clamped * jumping.astype(np.float64) * 2.0

def _reward_crouch_prep(ctx, cfg):
    base_z, jt = ctx.base_height, ctx.info["commands"][:, 4]
    crouching = (jt > 0.5) & (base_z < cfg.base_height_target)
    target = cfg.crouch_height_target
    height_ok = (base_z > cfg.min_base_height) & (base_z < target + 0.1)
    return height_ok.astype(np.float64) * crouching.astype(np.float64) * 0.5

def _reward_landing_soft(ctx):
    vz = np.abs(ctx.linvel[:, 2])
    return np.exp(-vz / 0.5) * 0.3

def _reward_vertical_thrust(ctx, cfg):
    jt, base_z, vz = ctx.info["commands"][:, 4], ctx.base_height, ctx.linvel[:, 2]
    active = (jt > 0.5) & (base_z < 0.55) & (vz > 0.0)
    return vz * active.astype(np.float64)

def _reward_crouch_depth(ctx, cfg):
    base_z, jt = ctx.base_height, ctx.info["commands"][:, 4]
    crouching = (jt > 0.5) & (base_z < cfg.base_height_target)
    depth = np.clip((cfg.base_height_target - base_z) / 0.3, 0.0, 1.0)
    return depth * crouching.astype(np.float64) * 0.5

def _reward_wheel_ground_matching(ctx):
    """★ 轮地速度匹配 (式14)"""
    wheel_vel = ctx.info.get("wheel_vel", np.zeros((ctx.num_envs, 2)))
    lin_x = ctx.linvel[:, 0:1]
    error = np.sum(np.square(wheel_vel * WHEEL_R - lin_x), axis=1)
    fsm = ctx.info.get("fsm_state", -np.ones(ctx.num_envs))
    timer = ctx.info.get("fsm_timer", np.ones(ctx.num_envs))
    landing = (fsm == 3) & (timer < 0.05)
    r = -error
    r[~landing] = 0.0
    return r


@registry.envcfg("XqRobotWLJumpFlat")
@dataclass
class XqRobotWLJumpFlatCfg(XqRobotWLWalkFlatCfg):
    commands: XqRobotWLJumpCommands = field(default_factory=XqRobotWLJumpCommands)
    reward_config: XqRobotWLJumpRewardConfig | None = None
    curriculum: XqRobotWLJumpCurriculumConfig = field(default_factory=XqRobotWLJumpCurriculumConfig)
    max_episode_seconds: float = 10.0


class XqRobotWLJumpDRProvider(XqRobotWLDRProvider):
    def _sample_commands(self, env, num_reset):
        low = np.asarray(env._cfg.commands.vel_limit[0], dtype=get_global_dtype())
        high = np.asarray(env._cfg.commands.vel_limit[1], dtype=get_global_dtype())
        cmds = np.asarray(np.random.uniform(low=low, high=high, size=(num_reset, low.shape[0])), dtype=get_global_dtype())
        safe_linv = np.maximum(np.abs(cmds[:, 0]), 1e-4)
        cmds[:, 2] = np.clip(cmds[:, 2], -2.0 / safe_linv, 2.0 / safe_linv)
        return cmds


@registry.env("XqRobotWLJumpFlat", sim_backend="mujoco")
class XqRobotWLJumpFlatEnv(XqRobotWLWalkFlatEnv):
    """Wheeled-SRL 跳跃环境 (xqrobotwl)

    在原有跳跃环境上增加:
    - SLIP六状态FSM前馈
    - 前馈-反馈融合
    - 轮地匹配奖励
    - 扩展观测
    """
    _cfg: XqRobotWLJumpFlatCfg

    def __init__(self, cfg, num_envs=1, backend_type="mujoco"):
        self._jump_cfg = cfg.reward_config
        super().__init__(cfg, num_envs=num_envs, backend_type=backend_type)
        self._dr_manager._provider = XqRobotWLJumpDRProvider()
        # FSM
        self._fsm_state = -np.ones(num_envs, dtype=np.int32)
        self._fsm_timer = np.zeros(num_envs, dtype=np.float64)
        self._jump_phase = np.zeros(num_envs, dtype=np.float64)
        self._peak_height = np.zeros(num_envs, dtype=np.float64)
        self._feedback_gain = np.full(num_envs, cfg.reward_config.feedback_gain, dtype=np.float64)
        # 观测: 33 + fsm(1) + phase(1) = 35 / 38
        self._obs_frame_dim = 35
        self._critic_frame_dim = 38
        self._obs_history = np.zeros((num_envs, self._hist_len, self._obs_frame_dim), dtype=self._np_dtype)
        self._critic_history = np.zeros((num_envs, self._hist_len, self._critic_frame_dim), dtype=self._np_dtype)

    @property
    def obs_groups_spec(self):
        return {"obs": self._obs_frame_dim * self._hist_len, "critic": self._critic_frame_dim * self._hist_len}

    def _init_reward_functions(self):
        """跳跃奖励表: 通用 + Wheeled-SRL专用"""
        self._reward_fns = {
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
            "wheel_ground_matching": self._reward_wheel_ground_matching,  # ★ 替代 wheel_air_time
            "vertical_thrust": self._reward_vertical_thrust,
            "crouch_depth": self._reward_crouch_depth,
        }

    # 奖励委托
    def _reward_jump_height(self, ctx): return _reward_jump_height(ctx, self._jump_cfg)
    def _reward_crouch_prep(self, ctx): return _reward_crouch_prep(ctx, self._jump_cfg)
    def _reward_landing_soft(self, ctx): return _reward_landing_soft(ctx)
    def _reward_vertical_thrust(self, ctx): return _reward_vertical_thrust(ctx, self._jump_cfg)
    def _reward_crouch_depth(self, ctx): return _reward_crouch_depth(ctx, self._jump_cfg)
    def _reward_wheel_ground_matching(self, ctx): return _reward_wheel_ground_matching(ctx)

    def _reward_joint_action_rate(self, ctx):
        cur = ctx.info["current_actions"][:, :NUM_LEG_ACTIONS]
        lst = ctx.info["last_actions"][:, :NUM_LEG_ACTIONS]
        return np.sum(np.square(cur - lst), axis=1)

    def _reward_wheel_action_rate(self, ctx):
        cur = ctx.info["current_actions"][:, NUM_LEG_ACTIONS:]
        lst = ctx.info["last_actions"][:, NUM_LEG_ACTIONS:]
        return np.sum(np.square(cur - lst), axis=1)

    def _reward_leg_mirror(self, ctx):
        return np.square(ctx.dof_pos[:, 0] + ctx.dof_pos[:, 3]) + np.sum(np.square(ctx.dof_pos[:, 1:3] + ctx.dof_pos[:, 4:6]), axis=1)

    def _reward_tsk(self, ctx):
        return np.square((ctx.dof_pos[:, 0] + ctx.dof_pos[:, 3]) - ctx.info["commands"][:, 3])

    # ── 核心: step覆写 ──

    def step(self, actions):
        """前馈-反馈融合 step"""
        dof_pos = self.get_dof_pos()
        linvel = self.get_local_linvel()
        ff = compute_slip_feedforward(
            self._fsm_state, self._fsm_timer, dof_pos, linvel,
            self.default_angles, self._jump_cfg.jump_height_target, WHEEL_R, self._cfg.ctrl_dt,
        )
        kb = self._feedback_gain.reshape(-1, 1)
        fused = ff + kb * actions
        return super().step(fused)

    # ── 状态更新 ──

    def update_state(self, state):
        self._update_commands(state.info)
        linvel = self.get_local_linvel()
        gyro = self.get_gyro()
        gravity = self._backend.get_sensor_data(self._cfg.sensor.upvector)
        dof_pos = self.get_dof_pos()
        dof_vel = self.get_dof_vel()
        self._update_wheel_contact(state.info)

        base_height = self._base_height_values(linvel.shape[0])
        jt = state.info["commands"][:, 4]
        self._fsm_state, self._fsm_timer = update_fsm(
            self._fsm_state, self._fsm_timer, base_height, linvel, dof_pos, jt,
            self._cfg.reward_config.base_height_target, self._cfg.ctrl_dt,
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
            lf = np.asarray(left, dtype=get_global_dtype()).reshape(-1, 3)[:self._num_envs]
            rf = np.asarray(right, dtype=get_global_dtype()).reshape(-1, 3)[:self._num_envs]
            info["wheel_contact"] = np.stack([
                (np.linalg.norm(lf, axis=1) > 10.0).astype(np.float64),
                (np.linalg.norm(rf, axis=1) > 10.0).astype(np.float64),
            ], axis=1)
        except (KeyError, AttributeError):
            info["wheel_contact"] = np.zeros((self._num_envs, 2), dtype=np.float64)

    def _compute_terminated(self, gravity, dof_pos):
        tilt = np.arccos(np.clip(gravity[:, 2], -1, 1))
        terminated = tilt > np.deg2rad(self._jump_cfg.max_tilt_deg)
        terminated |= np.asarray(self._backend.get_base_pos(), dtype=get_global_dtype())[:, 2] < self._jump_cfg.min_base_height
        terminated |= (dof_pos[:, 1] < 0.02) | (dof_pos[:, 4] > -0.02)
        terminated |= (np.abs(dof_pos[:, 2]) > 1.2) | (np.abs(dof_pos[:, 5]) > 1.2)
        # 轮滑移终止
        linvel = self.get_local_linvel()
        wv = self.get_dof_vel()[:, NUM_LEG_ACTIONS:]
        slip = np.any(np.abs(wv * WHEEL_R - linvel[:, 0:1]) > 1.5, axis=1)
        is_ground = np.asarray(self._backend.get_base_pos(), dtype=get_global_dtype())[:, 2] < self._jump_cfg.base_height_target + 0.05
        terminated |= slip & is_ground
        return terminated



    def _compute_obs(self, info, linvel, gyro, gravity, dof_pos, dof_vel):
        batch_sz = linvel.shape[0]
        noise_cfg = self._cfg.noise_config
        leg_diff = dof_pos[:, :NUM_LEG_ACTIONS] - DEFAULT_ANGLES[:NUM_LEG_ACTIONS]
        leg_vel = dof_vel[:, :NUM_LEG_ACTIONS]
        wheel_vel = dof_vel[:, NUM_LEG_ACTIONS:]

        cmds_sliced = info["commands"][:batch_sz]
        reset_ids = info.get("_reset_ids", None)
        if reset_ids is not None:
            idx = reset_ids[:batch_sz]
            fsm_feat = self._fsm_state.astype(np.float64)[idx, None] / 5.0
            phase_feat = self._jump_phase[idx, None]
        else:
            fsm_feat = self._fsm_state.astype(np.float64).reshape(-1, 1)[:batch_sz] / 5.0
            phase_feat = self._jump_phase.reshape(-1, 1)[:batch_sz]

        def noise(x, scale):
            return x + np.random.randn(*x.shape).astype(x.dtype) * scale

        obs_frame = np.concatenate([
            noise(gyro, noise_cfg.scale_gyro), -noise(gravity, noise_cfg.scale_gravity),
            noise(leg_diff, noise_cfg.scale_joint_angle), noise(leg_vel, noise_cfg.scale_joint_vel),
            noise(wheel_vel, noise_cfg.scale_wheel_vel),
            info.get("current_actions", np.zeros((batch_sz, NUM_ACTIONS))),
            cmds_sliced,
            fsm_feat,
            phase_feat,
        ], axis=1, dtype=get_global_dtype())

        critic_frame = np.concatenate([
            gyro, -gravity,
            leg_diff, leg_vel, wheel_vel,
            info.get("current_actions", np.zeros((batch_sz, NUM_ACTIONS))),
            cmds_sliced,
            fsm_feat,
            phase_feat,
            linvel,
        ], axis=1, dtype=get_global_dtype())

        steps_arr = info.get("steps", np.zeros((batch_sz,), dtype=np.int32))
        if reset_ids is not None:
            # Subset reset: only update the resetting envs in history
            ids = reset_ids[:batch_sz]
            mask = (steps_arr < self._hist_len).flatten()
            local_ids = ids[mask]
            if local_ids.size > 0:
                self._obs_history[local_ids] = obs_frame[mask, None, :]
                self._critic_history[local_ids] = critic_frame[mask, None, :]
            self._obs_history[ids, :-1, :] = self._obs_history[ids, 1:, :]
            self._obs_history[ids, -1, :] = obs_frame
            self._critic_history[ids, :-1, :] = self._critic_history[ids, 1:, :]
            self._critic_history[ids, -1, :] = critic_frame
        else:
            if np.any(steps_arr < self._hist_len):
                mask = (steps_arr < self._hist_len).flatten()
                self._obs_history[mask] = obs_frame[mask, None, :]
                self._critic_history[mask] = critic_frame[mask, None, :]
            self._obs_history[:, :-1, :] = self._obs_history[:, 1:, :]
            self._obs_history[:, -1, :] = obs_frame
            self._critic_history[:, :-1, :] = self._critic_history[:, 1:, :]
            self._critic_history[:, -1, :] = critic_frame

        if reset_ids is not None:
            ids = reset_ids[:batch_sz]
            obs = self._obs_history[ids].reshape(batch_sz, -1)
            critic = self._critic_history[ids].reshape(batch_sz, -1)
        else:
            obs = self._obs_history[:batch_sz].reshape(batch_sz, -1)
            critic = self._critic_history[:batch_sz].reshape(batch_sz, -1)

        return {"obs": obs, "critic": critic}

    def _compute_reward(self, info, linvel, gyro, gravity, dof_pos, dof_vel):
        dtype = get_global_dtype()
        ctx = RewardContext(
            info=info, linvel=linvel, gyro=gyro,
            dof_pos=dof_pos[:, :NUM_LEG_ACTIONS],
            dof_vel=dof_vel[:, :NUM_LEG_ACTIONS],
            num_envs=linvel.shape[0],
            default_angles=self.default_angles[:NUM_LEG_ACTIONS].astype(dtype),
            tracking_sigma=self._jump_cfg.tracking_sigma,
            base_height_target=self._jump_cfg.base_height_target,
            base_height=self._base_height_values(linvel.shape[0]),
            gravity=gravity, joint_range=None,
        )
        return rewards.run_reward_dispatch(
            scales=self._jump_cfg.scales, fns=self._reward_fns, ctx=ctx, info=info,
            enable_log=self._enable_reward_log, ctrl_dt=self._cfg.ctrl_dt,
            only_positive=self._jump_cfg.only_positive_rewards,
        )

    def _base_height_values(self, num_obs):
        pos = np.asarray(self._backend.get_base_pos(), dtype=get_global_dtype())
        return pos[:, 2] if pos.shape[0] == num_obs else np.zeros(num_obs, dtype=get_global_dtype())
