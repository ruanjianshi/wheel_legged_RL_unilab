"""xqrobotV2 Wheeled-SRL 跳跃环境 — SLIP模型前馈 + PPO反馈融合

Wheeled-SRL 核心改进 vs 原始跳跃环境:
  1. SLIP模型: 六状态FSM生成前馈参考轨迹 (腿关节+轮速)
  2. 前馈-反馈融合: action = feedforward + k_b * policy_output
  3. 轮地速度匹配奖励: 替代 wheel_air_time, 惩罚落地时轮地速度差
  4. 扩展观测: 额外包含 fsm_state 和 jump_phase

跳跃流程:
  1. jump_trigger > 0.5 → FSM进入跳跃循环
  2. FSM状态: -1(初始化) → 0(地面压缩) → 1(跳跃加速) → 2(飞行轮速调制) → 3(着陆缓冲) → 4(恢复)
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
from unilab.envs.locomotion.xqrobotV2.base import (
    NUM_LEG_ACTIONS,
    NUM_WHEEL_ACTIONS,
    NUM_ACTIONS,
    DEFAULT_ANGLES,
)

from .joystick import (
    XqRobotCurriculumConfig,
    XqRobotDRProvider,
    XqRobotV2WalkFlatCfg,
    XqRobotV2WalkFlatEnv,
)

_NUM_JUMP_CMD_DIM = 5  # [vx, vy, vyaw, tsk, jump_trigger]

# ==================== SLIP模型参数 ====================
SLIP_K = 5000.0           # 弹簧刚度
SLIP_C = 50.0             # 阻尼系数
WHEEL_R = 0.065            # 车轮半径 (m) — 按 xqrobotV2 实际轮径
WHEEL_J = 0.005            # 车轮转动惯量
WHEEL_B = 0.01             # 车轮粘性摩擦


@dataclass
class XqRobotJumpCommands(Commands):
    vel_limit: list[list[float]] = field(
        default_factory=lambda: [[-0.3, -0.1, -0.5, -0.1, 0], [0.3, 0.1, 0.5, 0.1, 1]]
    )
    resampling_time: float = 4.0


@dataclass
class XqRobotJumpRewardConfig:
    """Wheeled-SRL 跳跃奖励配置"""
    scales: dict[str, float]
    tracking_sigma: float = 0.25
    base_height_target: float = 0.65
    only_positive_rewards: bool = False
    max_tilt_deg: float = 45.0
    min_base_height: float = 0.20
    jump_height_target: float = 1.0
    crouch_height_target: float = 0.40
    # Wheeled-SRL 特有参数
    feedback_gain: float = 0.5           # k_b: 反馈增益
    wheel_matching_sigma: float = 0.3    # 轮地匹配 sigma


@dataclass
class XqRobotJumpCurriculumConfig(XqRobotCurriculumConfig):
    enabled: bool = False


# ==================== Wheeled-SRL SLIP 前馈计算 ====================

def compute_slip_feedforward(
    fsm_state: np.ndarray,
    fsm_timer: np.ndarray,
    dof_pos: np.ndarray,
    base_lin_vel: np.ndarray,
    default_angles: np.ndarray,
    jump_height_target: float,
    wheel_r: float,
    dt: float,
) -> np.ndarray:
    """根据FSM状态六阶段SLIP模型生成前馈控制信号
    
    输出: (num_envs, 8) 前馈动作 (6腿 + 2轮)
    """
    num_envs = fsm_state.shape[0]
    ff = np.zeros((num_envs, NUM_ACTIONS), dtype=np.float64)

    for s in range(-1, 5):
        mask = fsm_state == s
        if not mask.any():
            continue

        if s == -1:  # 初始化 — 回到默认站姿
            ff[mask, :NUM_LEG_ACTIONS] = default_angles[:NUM_LEG_ACTIONS]  # 站立
            ff[mask, NUM_LEG_ACTIONS:] = 0.0  # 轮子停

        elif s == 0:  # 地面接触 — 下蹲蓄能
            # 大腿前屈+小腿深屈
            ff[mask, 0] = 0.0      # left_hip
            ff[mask, 1] = -0.4     # left_thigh (前屈)
            ff[mask, 2] = -1.2     # left_calf (深屈蓄能)
            ff[mask, 3] = 0.0      # right_hip
            ff[mask, 4] = -0.4     # right_thigh
            ff[mask, 5] = -1.2     # right_calf
            ff[mask, 6] = 0.0      # 车轮制动
            ff[mask, 7] = 0.0

        elif s == 1:  # 跳跃加速 — 爆发伸展
            # 腿部爆发伸展 (抵消默认角度的偏移)
            ff[mask, 0] = 0.0
            ff[mask, 1] = 0.3      # thigh 前推
            ff[mask, 2] = 0.2      # calf 伸展
            ff[mask, 3] = 0.0
            ff[mask, 4] = 0.3
            ff[mask, 5] = 0.2
            # 轮子主动前驱提供额外冲量
            expected_forward = 0.5  # 预期前进速度 (m/s)
            wheel_rps = expected_forward / wheel_r
            ff[mask, 6] = wheel_rps * 0.5  # 经 action_scale 换算
            ff[mask, 7] = wheel_rps * 0.5

        elif s == 2:  # 飞行阶段 — 调整姿态 + 轮速匹配
            time_scale = np.clip(fsm_timer[mask] / 0.3, 0.0, 1.0)  # 飞行~300ms
            # 腿部回收准备着陆
            retract = 0.2 * (1.0 - time_scale)
            ff[mask, 0] = 0.0
            ff[mask, 1] = -retract
            ff[mask, 2] = -retract
            ff[mask, 3] = 0.0
            ff[mask, 4] = -retract
            ff[mask, 5] = -retract
            # ★ 轮速斜坡匹配预期落地速度
            ground_vel = np.abs(base_lin_vel[mask, 0])
            target_wheel_rps = ground_vel / wheel_r
            ff[mask, 6] = target_wheel_rps * 0.5  # 经 action_scale 换算
            ff[mask, 7] = target_wheel_rps * 0.5

        elif s == 3:  # 着陆准备 — 缓冲
            landing_progress = np.clip(fsm_timer[mask] / 0.1, 0.0, 1.0)
            bend = -0.2 * landing_progress  # 微屈缓冲
            ff[mask, 0] = 0.0
            ff[mask, 1] = bend
            ff[mask, 2] = bend
            ff[mask, 3] = 0.0
            ff[mask, 4] = bend
            ff[mask, 5] = bend
            # 轮子同步地面速度
            ground_vel = base_lin_vel[mask, 0]
            sync_rps = ground_vel / wheel_r
            ff[mask, 6] = sync_rps * 0.5
            ff[mask, 7] = sync_rps * 0.5

        elif s == 4:  # 恢复 — 回到默认站姿
            recovery = np.clip(fsm_timer[mask] / 0.2, 0.0, 1.0)
            current_leg = dof_pos[mask, :NUM_LEG_ACTIONS]
            default_leg = default_angles[:NUM_LEG_ACTIONS]
            ff[mask, :NUM_LEG_ACTIONS] = default_leg + (current_leg - default_leg) * (1 - recovery[:, None])
            ff[mask, NUM_LEG_ACTIONS:] = 0.0

    return ff


def update_fsm(
    fsm_state: np.ndarray,
    fsm_timer: np.ndarray,
    base_height: np.ndarray,
    base_lin_vel: np.ndarray,
    dof_pos: np.ndarray,
    jump_trigger: np.ndarray,
    default_height: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """更新六状态FSM
    
    Returns:
        fsm_state, fsm_timer (已更新)
    """
    vertical_vel = base_lin_vel[:, 2]
    ground_contact = base_height < default_height + 0.02  # ~轮子触地
    fsm_timer += dt

    for s in range(-1, 5):
        mask = fsm_state == s
        if not mask.any():
            continue

        if s == -1:  # 初始化 → 地面接触
            # jump_trigger > 0.5 且已完成准备 (timer>0.1)
            trigger_active = jump_trigger[mask] > 0.5
            ready = fsm_timer[mask] > 0.1
            next_mask = np.zeros_like(mask, dtype=bool)
            next_mask[mask] = trigger_active & ready
            fsm_state[next_mask] = 0
            fsm_timer[next_mask] = 0.0

        elif s == 0:  # 地面接触 → 跳跃加速
            # 下蹲到位 (calf 够深) 或时间够 (300ms)
            deep_enough = (dof_pos[mask, 2] < -0.8) & (dof_pos[mask, 5] < -0.8)
            timed_out = fsm_timer[mask] > 0.3
            next_mask = np.zeros_like(mask, dtype=bool)
            next_mask[mask] = deep_enough | timed_out
            fsm_state[next_mask] = 1
            fsm_timer[next_mask] = 0.0

        elif s == 1:  # 跳跃加速 → 飞行
            # 腿伸展到接近伸直
            extended = (dof_pos[mask, 2] > -0.3) & (dof_pos[mask, 5] > -0.3)
            timed_out = fsm_timer[mask] > 0.2
            next_mask = np.zeros_like(mask, dtype=bool)
            next_mask[mask] = extended | timed_out
            fsm_state[next_mask] = 2
            fsm_timer[next_mask] = 0.0

        elif s == 2:  # 飞行 → 着陆准备
            descending = vertical_vel[mask] < 0
            near_ground = base_height[mask] < default_height + 0.15
            timed_out = fsm_timer[mask] > 0.5
            next_mask = np.zeros_like(mask, dtype=bool)
            next_mask[mask] = descending & (near_ground | timed_out)
            fsm_state[next_mask] = 3
            fsm_timer[next_mask] = 0.0

        elif s == 3:  # 着陆准备 → 恢复
            landed = ground_contact[mask] & (fsm_timer[mask] > 0.05)
            next_mask = np.zeros_like(mask, dtype=bool)
            next_mask[mask] = landed
            fsm_state[next_mask] = 4
            fsm_timer[next_mask] = 0.0

        elif s == 4:  # 恢复 → 初始化 (等待下次跳跃)
            recovered = fsm_timer[mask] > 0.2
            next_mask = np.zeros_like(mask, dtype=bool)
            next_mask[mask] = recovered
            fsm_state[next_mask] = -1
            fsm_timer[next_mask] = 0.0

    return fsm_state, fsm_timer


# ==================== 奖励函数 ====================

def _reward_jump_height(ctx: RewardContext, jump_cfg: XqRobotJumpRewardConfig) -> np.ndarray:
    base_z = ctx.base_height
    jump_trigger = ctx.info["commands"][:, 4]
    jumping = jump_trigger > 0.5
    target = jump_cfg.jump_height_target
    clamped = np.clip(base_z / target, 0.0, 1.0)
    return clamped * jumping.astype(np.float64) * 2.0


def _reward_crouch_prep(ctx: RewardContext, jump_cfg: XqRobotJumpRewardConfig) -> np.ndarray:
    base_z = ctx.base_height
    jump_trigger = ctx.info["commands"][:, 4]
    crouching = (jump_trigger > 0.5) & (base_z < jump_cfg.base_height_target)
    target = jump_cfg.crouch_height_target
    height_ok = (base_z > jump_cfg.min_base_height) & (base_z < target + 0.1)
    return height_ok.astype(np.float64) * crouching.astype(np.float64) * 0.5


def _reward_landing_soft(ctx: RewardContext) -> np.ndarray:
    base_linvel_z = ctx.linvel[:, 2]
    vz_mag = np.abs(base_linvel_z)
    return np.exp(-vz_mag / 0.5) * 0.3


def _reward_wheel_ground_matching(ctx: RewardContext) -> np.ndarray:
    """★ Wheeled-SRL 核心: 轮地速度匹配奖励 (式14)
    
    惩罚着陆瞬间轮子线速度与机身前进速度的差
    """
    # 读取info中的轮速信息
    wheel_vel = ctx.info.get("wheel_vel", np.zeros((ctx.num_envs, 2)))
    linvel_x = ctx.linvel[:, 0]
    wheel_lin = wheel_vel * 0.065  # wheel_r
    error = np.sum(np.square(wheel_lin - linvel_x.reshape(-1, 1)), axis=1)
    # 只在着陆瞬间惩罚
    fsm_state = ctx.info.get("fsm_state", -np.ones(ctx.num_envs))
    is_landing = (fsm_state == 3) & (ctx.info.get("fsm_timer", np.ones(ctx.num_envs)) < 0.05)
    reward = -error
    reward[~is_landing] = 0.0
    return reward


@registry.envcfg("XqRobotV2JumpFlat")
@dataclass
class XqRobotV2JumpFlatCfg(XqRobotV2WalkFlatCfg):
    commands: XqRobotJumpCommands = field(default_factory=XqRobotJumpCommands)
    reward_config: XqRobotJumpRewardConfig | None = None
    curriculum: XqRobotJumpCurriculumConfig = field(default_factory=XqRobotJumpCurriculumConfig)
    max_episode_seconds: float = 10.0


class XqRobotJumpDRProvider(XqRobotDRProvider):
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


@registry.env("XqRobotV2JumpFlat", sim_backend="mujoco")
class XqRobotV2JumpFlatEnv(XqRobotV2WalkFlatEnv):
    """Wheeled-SRL 跳跃环境
    
    在原有跳跃环境基础上增加:
    1. SLIP模型六状态FSM + 前馈计算
    2. 前馈-反馈融合动作
    3. 轮地速度匹配奖励
    4. 扩展观测 (fsm_state + jump_phase)
    """
    _cfg: XqRobotV2JumpFlatCfg

    def __init__(self, cfg: XqRobotV2JumpFlatCfg, num_envs=1, backend_type="mujoco"):
        self._jump_cfg = cfg.reward_config
        super().__init__(cfg, num_envs=num_envs, backend_type=backend_type)
        self._dr_manager._provider = XqRobotJumpDRProvider()

        # ★ Wheeled-SRL: FSM 状态
        self._fsm_state = -np.ones(num_envs, dtype=np.int32)
        self._fsm_timer = np.zeros(num_envs, dtype=np.float64)
        self._jump_phase = np.zeros(num_envs, dtype=np.float64)
        self._peak_height = np.zeros(num_envs, dtype=np.float64)
        self._feedback_gain = np.full(num_envs, cfg.reward_config.feedback_gain, dtype=np.float64)

        # ★ 观测维度: 原始33 + fsm_state(1) + jump_phase(1) = 35
        #  critic: 原始36 + 2 = 38
        self._obs_frame_dim = 35
        self._critic_frame_dim = 38
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
        """Wheeled-SRL 奖励表 — 含轮地匹配奖励"""
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
            "wheel_ground_matching": self._reward_wheel_ground_matching,  # ★ 替换 wheel_air_time
        }

    # ── 奖励委托 ──

    def _reward_jump_height(self, ctx: RewardContext) -> np.ndarray:
        return _reward_jump_height(ctx, self._jump_cfg)

    def _reward_crouch_prep(self, ctx: RewardContext) -> np.ndarray:
        return _reward_crouch_prep(ctx, self._jump_cfg)

    def _reward_landing_soft(self, ctx: RewardContext) -> np.ndarray:
        return _reward_landing_soft(ctx)

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
        hip_error = ctx.dof_pos[:, 0] + ctx.dof_pos[:, 3]
        pitch_error = ctx.dof_pos[:, 1:3] - ctx.dof_pos[:, 4:6]
        return np.square(hip_error) + np.sum(np.square(pitch_error), axis=1)

    def _reward_tsk(self, ctx: RewardContext) -> np.ndarray:
        tsk_cmd = ctx.info["commands"][:, 3]
        hip_diff = ctx.dof_pos[:, 0] + ctx.dof_pos[:, 3]
        return np.square(hip_diff - tsk_cmd)

    # ── Step 覆写: 注入 SLIP 前馈 ──

    def step(self, actions: np.ndarray) -> NpEnvState:
        """覆写 step: 前馈-反馈融合后再执行
        
        流程:
        1. 读取当前FSM状态
        2. 计算SLIP前馈动作
        3. 融合: fused = feedforward + k_b * policy_action
        4. 执行 fused 动作
        """
        # 计算SLIP前馈
        dof_pos = self.get_dof_pos()
        base_linvel = self.get_local_linvel()
        ff_action = compute_slip_feedforward(
            self._fsm_state, self._fsm_timer,
            dof_pos, base_linvel,
            self.default_angles,
            self._jump_cfg.jump_height_target,
            WHEEL_R, self._cfg.ctrl_dt,
        )

        # 前馈-反馈融合: fused = ff + k_b * policy_output
        k_b = self._feedback_gain.reshape(-1, 1)
        fused_actions = ff_action + k_b * actions

        # 传给父类的原始 step
        state = super().step(fused_actions)
        return state

    # ── 状态更新 ──

    def update_state(self, state: NpEnvState) -> NpEnvState:
        """更新FSM状态 + 原有更新"""
        # 先读传感器
        self._update_commands(state.info)
        linvel = self.get_local_linvel()
        gyro = self.get_gyro()
        gravity = self._backend.get_sensor_data(self._cfg.sensor.upvector)
        dof_pos = self.get_dof_pos()
        dof_vel = self.get_dof_vel()
        self._update_wheel_contact(state.info)

        # ★ 更新FSM
        base_height = self._base_height_values(linvel.shape[0])
        jump_trigger = state.info["commands"][:, 4]
        self._fsm_state, self._fsm_timer = update_fsm(
            self._fsm_state, self._fsm_timer,
            base_height, linvel, dof_pos,
            jump_trigger,
            self._cfg.reward_config.base_height_target,
            self._cfg.ctrl_dt,
        )
        # 更新跳跃相位
        total_cycle = 0.8  # ~800ms 完整跳跃周期
        self._jump_phase = np.clip(self._fsm_timer / total_cycle, 0.0, 1.0)
        # 追踪峰值高度
        is_flying = self._fsm_state == 2
        self._peak_height[is_flying] = np.maximum(self._peak_height[is_flying], base_height[is_flying])

        # 将FSM信息加入info dict供奖励函数使用
        state.info["fsm_state"] = self._fsm_state
        state.info["fsm_timer"] = self._fsm_timer
        state.info["jump_phase"] = self._jump_phase
        state.info["wheel_vel"] = dof_vel[:, NUM_LEG_ACTIONS:]  # 供 wheel_ground_matching 使用
        state.info["peak_height"] = self._peak_height

        terminated = self._compute_terminated(gravity, dof_pos)
        reward = self._compute_reward(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        obs = self._compute_obs(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        return state.replace(obs=obs, reward=reward, terminated=terminated)

    # ── 观测构建: 含 FSM 状态 ──

    def _compute_obs(self, info: dict, linvel, gyro, gravity, dof_pos, dof_vel) -> dict[str, np.ndarray]:
        """Wheeled-SRL 观测: 原始33维 + fsm_state(1) + jump_phase(1) = 35/38维"""
        batch_sz = linvel.shape[0]
        noise_cfg = self._cfg.noise_config
        leg_diff = dof_pos[:, :NUM_LEG_ACTIONS] - DEFAULT_ANGLES[:NUM_LEG_ACTIONS]
        leg_vel = dof_vel[:, :NUM_LEG_ACTIONS]
        wheel_vel = dof_vel[:, NUM_LEG_ACTIONS:]

        noisy_gyro = self._obs_noise(gyro, noise_cfg.scale_gyro)
        noisy_gravity = self._obs_noise(gravity, noise_cfg.scale_gravity)
        noisy_leg_diff = self._obs_noise(leg_diff, noise_cfg.scale_joint_angle)
        noisy_leg_vel = self._obs_noise(leg_vel, noise_cfg.scale_joint_vel)
        noisy_wheel_vel = self._obs_noise(wheel_vel, noise_cfg.scale_wheel_vel)
        last_actions = info.get("current_actions", np.zeros((batch_sz, NUM_ACTIONS)))

        # FSM特征 — 子集 reset 时按 env_ids 索引, 否则取前 batch_sz
        reset_ids = info.get("_reset_ids", None)
        if reset_ids is not None:
            idx = reset_ids[:batch_sz]
            fsm_feat = self._fsm_state.astype(np.float64)[idx, None] / 5.0
            phase_feat = self._jump_phase[idx, None]
        else:
            fsm_feat = self._fsm_state.astype(np.float64).reshape(-1, 1)[:batch_sz] / 5.0
            phase_feat = self._jump_phase.reshape(-1, 1)[:batch_sz]

        # Actor 单帧 (35维): 33 + fsm + phase
        obs_frame = np.concatenate([
            noisy_gyro, -noisy_gravity,
            noisy_leg_diff, noisy_leg_vel, noisy_wheel_vel,
            last_actions, info["commands"][:batch_sz],
            fsm_feat, phase_feat,
        ], axis=1, dtype=get_global_dtype())

        # Critic 单帧 (38维): 35 + linvel
        critic_frame = np.concatenate([
            gyro, -gravity,
            leg_diff, leg_vel, wheel_vel,
            last_actions, info["commands"][:batch_sz],
            fsm_feat, phase_feat, linvel,
        ], axis=1, dtype=get_global_dtype())

        # 历史堆叠
        steps_arr = np.asarray(info.get("steps", np.zeros((batch_sz,), dtype=np.int32)))
        if reset_ids is not None:
            # Subset reset: only update the resetting envs
            ids = reset_ids[:batch_sz]
            reset_mask = (steps_arr < self._hist_len).flatten()
            local_ids = ids[reset_mask]
            if local_ids.size > 0:
                self._obs_history[local_ids] = obs_frame[reset_mask, None, :]
                self._critic_history[local_ids] = critic_frame[reset_mask, None, :]
            self._obs_history[ids, :-1, :] = self._obs_history[ids, 1:, :]
            self._obs_history[ids, -1, :] = obs_frame
            self._critic_history[ids, :-1, :] = self._critic_history[ids, 1:, :]
            self._critic_history[ids, -1, :] = critic_frame
        else:
            if np.any(steps_arr < self._hist_len):
                reset_mask = (steps_arr < self._hist_len).flatten()
                self._obs_history[reset_mask] = obs_frame[reset_mask, None, :]
                self._critic_history[reset_mask] = critic_frame[reset_mask, None, :]
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

    # ── 终止条件: 增加轮滑移终止 ──

    def _compute_terminated(self, gravity: np.ndarray, dof_pos: np.ndarray) -> np.ndarray:
        base_height = np.asarray(self._backend.get_base_pos(), dtype=get_global_dtype())[:, 2]
        tilt = np.arccos(np.clip(gravity[:, 2], -1, 1))
        max_tilt = np.deg2rad(self._jump_cfg.max_tilt_deg)
        terminated = tilt > max_tilt
        terminated |= base_height < self._jump_cfg.min_base_height
        thigh_collapsed = (dof_pos[:, 1] < 0.02) | (dof_pos[:, 4] < 0.02)
        calf_extreme = (np.abs(dof_pos[:, 2]) > 1.2) | (np.abs(dof_pos[:, 5]) > 1.2)
        terminated |= thigh_collapsed
        terminated |= calf_extreme
        # ★ Wheeled-SRL: 轮滑移终止 (着陆时轮地速度差过大)
        linvel = self.get_local_linvel()
        wheel_vel = self.get_dof_vel()[:, NUM_LEG_ACTIONS:]
        wheel_linvel = wheel_vel * WHEEL_R
        slip = np.any(np.abs(wheel_linvel - linvel[:, 0:1]) > 1.5, axis=1)
        is_ground = base_height < self._jump_cfg.base_height_target + 0.05
        terminated |= slip & is_ground
        return terminated

    # ── 轮子触地检测 ──

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

    def _compute_reward(self, info: dict, linvel, gyro, gravity, dof_pos, dof_vel) -> np.ndarray:
        """使用 _jump_cfg (纯RL) 或 _jump_cfg (Wheeled-SRL)"""
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
