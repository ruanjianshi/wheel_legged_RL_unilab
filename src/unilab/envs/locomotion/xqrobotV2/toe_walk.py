"""xqrobotV2 踮脚步态环境 — 正弦参考轨迹 + 策略修正输出

借鉴 HumanoidSW2 (livelybot_pi_rl_baseline) 的设计思路:
- 使用相位时钟 (phase clock) 生成周期性参考关节轨迹
- 策略输出为参考轨迹上的修正量 (而非绝对关节角度)
- 轮子仅用于平衡 (几乎不转), 步态由腿部摆动完成

步态周期 (0.5s):
- 0~0.09s: 右腿摆动 (左腿支撑, 重心偏左)
- 0.25~0.34s: 左腿摆动 (右腿支撑, 重心偏右)
- 其余时间: 双支撑或过渡

关键差异 vs 平地行走:
- 4D 命令 [vx, vy, vyaw, tsk] (无 height)
- obs 附加 sin/cos(phase): actor=34×9=306, critic=37×9=333
- 策略输出加在参考轨迹上, 而非 DEFAULT_ANGLES
- 3 个踮脚专用奖励: ref_tracking, swing_lift, wheel_balance
- 终止条件增加 leg contact force (腿碰地 → die)
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

from .base import DEFAULT_LEG_ANGLES, NUM_LEG_ACTIONS, NUM_WHEEL_ACTIONS
from .joystick import (
    XqRobotCurriculumConfig,
    XqRobotDRProvider,
    XqRobotV2WalkFlatCfg,
    XqRobotV2WalkFlatEnv,
    _reward_feet_distance,
    _reward_hip_roll,
    _reward_wheel_symmetry,
)

_HISTORY_LEN = 9


# ═══ 踮脚命令配置 ═══
# 4D: [vx, vy, vyaw, tsk], 速度极低 (0.05 m/s) — 原地踏步为主


@dataclass
class XqRobotToeWalkCommands(Commands):
    vel_limit: list[list[float]] = field(
        default_factory=lambda: [[-0.3, -0.1, -0.3, -0.1], [0.3, 0.1, 0.3, 0.1]]
    )
    resampling_time: float = 6.0           # 6 秒换一次命令


@dataclass
class XqRobotToeWalkRewardConfig:
    """踮脚奖励配置

    cycle_time(0.5s):   步态周期 — 左右腿各摆动一次的时间
    ref_scale(0.15):    参考轨迹振幅 — 控制摆动幅度
    max_tilt_deg(45°):  比行走 (60°) 更严格 — 踮脚姿态更难维持
    min_base_height(0.15m): 比行走 (0.20m) 更低 — 允许更深的蹲姿
    """
    scales: dict[str, float]
    tracking_sigma: float = 0.25
    base_height_target: float = 0.65
    only_positive_rewards: bool = False
    max_tilt_deg: float = 45.0
    min_base_height: float = 0.15
    cycle_time: float = 0.5                # 步态周期 (秒)
    ref_scale: float = 0.15               # 参考轨迹振幅 (rad)


# ═══ 踮脚专用奖励函数 (3 个) ═══


def _reward_ref_tracking(ctx: RewardContext) -> np.ndarray:
    """参考轨迹跟踪奖励 — 腿关节必须紧跟正弦参考轨迹

    - err = Σ(actual - ref)^2: 6 个腿关节的平方误差和
    - base = exp(-2*err)*1.2 - 0.2*err: exp 项高精度奖励, linear 项防止完全偏离
    - swing 放大 ×3:  摆动腿跟踪必须更精确 (抬腿时机和高度不能出错)
    - stance 正常 ×1: 支撑腿可以稍宽松
    """
    ref = ctx.info.get("ref_dof_pos")
    if ref is None:
        return np.zeros((ctx.num_envs,), dtype=np.float64)
    err = np.sum(np.square(ctx.dof_pos[:, :NUM_LEG_ACTIONS] - ref), axis=1)
    base = np.exp(-2.0 * err) * 1.2 - 0.2 * err
    swing = ctx.info.get("swing_mask", np.zeros((ctx.num_envs,)))
    return base * (1.0 + 2.0 * swing)  # stance×1, swing×3


def _reward_swing_lift(ctx: RewardContext) -> np.ndarray:
    """摆动腿离地奖励 — 摆动腿的轮子必须离开地面

    - 只在有 swing_mask > 0 的环境上计分 (避免全零梯度)
    - air_time = 1 - mean(contact): 轮子离地比例
    - 只在摆动腿的 env 上给分
    """
    swing_mask = ctx.info.get("swing_mask", np.zeros((ctx.num_envs,)))
    wheel_contact = ctx.info.get("wheel_contact", np.zeros((ctx.num_envs, 2)))
    if np.max(swing_mask) < 0.5:
        return np.zeros((ctx.num_envs,), dtype=np.float64)
    air_time = 1.0 - np.mean(wheel_contact, axis=1)
    return air_time * swing_mask


def _reward_wheel_balance(ctx: RewardContext) -> np.ndarray:
    """轮子平衡奖励 — 轮子用于维持平衡, 不能主动驱动前进

    - wheel_ok: 轮速越小越好 (通过 1/(1+speed*3) 得到)
    - upright:  机身越直越好 (gravity_xy → 0 通过 1/(1+gravity_xy*10))
    - 乘积: 两者需同时满足 (轮子转得快 + 姿态倾斜 = 双重扣分)
    """
    wheel_vel = ctx.dof_vel[:, -NUM_WHEEL_ACTIONS:]
    speed = np.sqrt(np.sum(np.square(wheel_vel), axis=1))
    wheel_ok = 1.0 / (1.0 + speed * 3.0)       # speed=0→1.0, speed=1→0.25
    gravity_xy = np.sum(np.square(ctx.gravity[:, :2]), axis=1)
    upright = 1.0 / (1.0 + gravity_xy * 10.0)   # tilt=0→1.0, tilt=10°→0.77
    return wheel_ok * upright


@registry.envcfg("XqRobotV2ToeWalkFlat")
@dataclass
class XqRobotV2ToeWalkFlatCfg(XqRobotV2WalkFlatCfg):
    commands: XqRobotToeWalkCommands = field(default_factory=XqRobotToeWalkCommands)
    reward_config: XqRobotToeWalkRewardConfig | None = None
    curriculum: XqRobotCurriculumConfig = field(
        default_factory=lambda: XqRobotCurriculumConfig(enabled=False)
    )
    max_episode_seconds: float = 12.0


# ═══ 踮脚环境类 ═══


class XqRobotToeWalkDRProvider(XqRobotDRProvider):
    """踮脚 DR — 与平地 DR 相同, 但不解耦命令"""
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


@registry.env("XqRobotV2ToeWalkFlat", sim_backend="mujoco")
class XqRobotV2ToeWalkFlatEnv(XqRobotV2WalkFlatEnv):
    """踮脚步态环境 — 策略输出 = 参考轨迹 + 修正量

    核心差异:
    - apply_action: leg_targets = ref_dof_pos + action_correction
                    (非 DEFAULT_ANGLES + action)
    - _compute_ref_dof_pos:  生成正弦参考轨迹 (髋/大腿/小腿的周期性目标)
    - update_state:          在 super() 前计算 ref + wheel_contact
    - _compute_obs:          附加 sin/cos(phase) 到观测
    - _compute_terminated:   增加 leg contact force 终止

    相位时钟:
    - 每个 env 有独立的相位偏移 (随机), 一半左腿先摆, 一半右腿先摆
    - 相位编码: sin(2π·phase) 和 cos(2π·phase) 作为观测输入
    """
    _cfg: XqRobotV2ToeWalkFlatCfg

    def __init__(self, cfg: XqRobotV2ToeWalkFlatCfg, num_envs=1, backend_type="mujoco"):
        self._toe_cfg = cfg.reward_config
        super().__init__(cfg, num_envs=num_envs, backend_type=backend_type)
        self._dr_manager._provider = XqRobotToeWalkDRProvider()
        # 相位偏移: 随机初始化, 让一半 env 左腿先摆, 一半右腿先摆
        self._phase_offset = np.random.uniform(0, 2 * np.pi, (num_envs,)).astype(np.float64)
        self._ref_dof_pos = np.zeros((num_envs, NUM_LEG_ACTIONS), dtype=np.float64)
        self._swing_mask = np.zeros((num_envs,), dtype=np.float64)
        # 观测维度: 4D cmd → base=32 actor / 35 critic, +2 (sin/cos phase)
        self._obs_frame_dim = 34     # 32 + 2 = 34
        self._critic_frame_dim = 37  # 35 + 2 = 37
        self._obs_history = np.zeros((num_envs, self._hist_len, self._obs_frame_dim), dtype=self._np_dtype)
        self._critic_history = np.zeros((num_envs, self._hist_len, self._critic_frame_dim), dtype=self._np_dtype)

    def _init_reward_functions(self) -> None:
        """踮脚奖励表 — 11 个通用 + 3 个踮脚专用 + 3 个复用自 joystick = 17 个

        注意: tracking_lin_vel/ang_vel 权重通常为 0 (YAML 注入),
              因为踮脚本就不需要速度跟踪, 核心是 ref_tracking + swing_lift
        """
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
            "ref_tracking": _reward_ref_tracking,       # ★ 参考轨迹跟踪
            "wheel_balance": _reward_wheel_balance,      # ★ 轮子平衡
            "swing_lift": _reward_swing_lift,            # ★ 摆动腿离地
            "feet_distance": _reward_feet_distance,       # 复用 joystick
            "wheel_symmetry": _reward_wheel_symmetry,     # 复用 joystick
            "hip_roll": _reward_hip_roll,                 # 复用 joystick
        }

    def _reward_joint_action_rate(self, ctx: RewardContext) -> np.ndarray:
        current = ctx.info["current_actions"][:, :NUM_LEG_ACTIONS]
        last = ctx.info["last_actions"][:, :NUM_LEG_ACTIONS]
        return np.sum(np.square(current - last), axis=1)

    def _reward_wheel_action_rate(self, ctx: RewardContext) -> np.ndarray:
        current = ctx.info["current_actions"][:, NUM_LEG_ACTIONS:]
        last = ctx.info["last_actions"][:, NUM_LEG_ACTIONS:]
        return np.sum(np.square(current - last), axis=1)

    def _reward_leg_mirror(self, ctx: RewardContext) -> np.ndarray:
        """简化对称 — 髋镜像 + 大腿/小腿平行"""
        hip = ctx.dof_pos[:, 0] + ctx.dof_pos[:, 3]
        thigh = ctx.dof_pos[:, 1] - ctx.dof_pos[:, 4]
        calf = ctx.dof_pos[:, 2] - ctx.dof_pos[:, 5]
        return np.square(hip) + np.square(thigh) + np.square(calf)

    def _reward_tsk(self, ctx: RewardContext) -> np.ndarray:
        tsk_cmd = ctx.info["commands"][:, 3]
        hip_diff = ctx.dof_pos[:, 0] - ctx.dof_pos[:, 3]
        return np.square(hip_diff - tsk_cmd)

    # ── 踮脚动作执行 ─────────────────────────────────────────────

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        """踮脚步态动作映射 — 策略输出 = 参考轨迹上的修正量

        腿: leg_targets = ref_dof_pos + correction (相对参考轨迹的增量)
        轮: wheel_targets = action × wheel_scale (速度控制, 仅用于平衡)
        """
        clipped = np.clip(actions, -self._cfg.control_config.clip_actions, self._cfg.control_config.clip_actions)
        state.info["last_actions"] = state.info.get("current_actions", np.zeros_like(clipped))
        state.info["current_actions"] = clipped
        exec_actions = state.info["last_actions"] if self._cfg.control_config.simulate_action_latency else clipped
        # 腿: 策略输出 + 参考轨迹 (不是 DEFAULT_ANGLES!)
        leg_corr = exec_actions[:, :NUM_LEG_ACTIONS] * self._cfg.control_config.action_scale
        leg_targets = self._ref_dof_pos[: leg_corr.shape[0]] + leg_corr
        # 轮: 速度控制 (仅用于平衡, 不驱动前进)
        wheel_targets = exec_actions[:, NUM_LEG_ACTIONS:] * self._cfg.control_config.wheel_action_scale
        half_legs = NUM_LEG_ACTIONS // 2
        return np.concatenate(
            [
                leg_targets[:, :half_legs],
                wheel_targets[:, :1],
                leg_targets[:, half_legs:],
                wheel_targets[:, 1:],
            ],
            axis=1,
            dtype=self._np_dtype,
        )

    # ── 参考轨迹生成 ─────────────────────────────────────────────

    def _compute_ref_dof_pos(self, info: dict) -> None:
        """基于相位时钟生成正弦参考轨迹

        步态周期分解 (0.5s = 50 步 @ 100Hz):
        - phase ∈ [0, 1] 在 cycle_time 内线性增长
        - sin/cos 编码当前相位 (sin 过零点 = 左腿着地 ↔ 右摆)
        - T=0.4: 摆动窗口阈值 → sin > 0.4 时进入摆动相 (~36% 周期)
        - swing: thigh 快速前摆 (髋抬腿)
        - lift:   calf 快速弯曲 (膝关节抬轮离地)
        - lean:   身体向支撑腿偏转 (COM 转移, 保持平衡)

        左腿时序 (从 sin=0 开始):
        0.0→0.09s: cos>0.2, 身体左偏 (准备右腿摆动)
        0.09→0.18s: sin>0.4, 右腿摆动 ← 提前 0.04s 重心转移
        0.25s:      sin=0, 左腿着地
        0.25→0.34s: cos<0.2, 身体右偏 (准备左腿摆动)
        0.34→0.43s: sin<-0.4, 左腿摆动
        0.50s:      周期结束, 回到 sin=0
        """
        cycle_time = self._toe_cfg.cycle_time
        steps = info.get("steps", np.zeros((self._num_envs,), dtype=np.float64))
        dt = self._cfg.ctrl_dt
        # phase: 经过时间 / 周期长度 + 随机偏移 (不同 env 起摆不同)
        phase = (steps * dt / cycle_time) + self._phase_offset / (2 * np.pi)
        sin_pos = np.sin(2 * np.pi * phase)[:, None]
        cos_pos = np.cos(2 * np.pi * phase)[:, None]
        scale = self._toe_cfg.ref_scale

        # Swing window: sin 信号超过 T=0.4 时进入摆动相 (~36% 的半周期)
        T = 0.4
        left_swing  = np.clip((-sin_pos - T) / (1.0 - T), 0.0, 1.0)    # sin<-0.4 → left swing
        right_swing = np.clip((sin_pos - T) / (1.0 - T), 0.0, 1.0)     # sin>0.4  → right swing
        left_lift   = np.clip((-cos_pos - T) / (1.0 - T), 0.0, 1.0)    # knee lift
        right_lift  = np.clip((cos_pos - T) / (1.0 - T), 0.0, 1.0)

        # Weight shift: cos 指示支撑腿方向
        # cos→+1 (phase≈0):   右腿准备摆动 → 身体左偏 (weight on left)
        # cos→-1 (phase≈0.5): 左腿准备摆动 → 身体右偏 (weight on right)
        lean_L = np.clip((cos_pos - 0.2) / 0.8, 0.0, 1.0)    # lean left (for right swing)
        lean_R = np.clip((-cos_pos - 0.2) / 0.8, 0.0, 1.0)   # lean right (for left swing)

        ref = np.zeros((self._num_envs, NUM_LEG_ACTIONS), dtype=np.float64)

        # Hip roll: shift COM by varying hip abduction
        # Lean left →  left hip more abducted (more negative), right hip less
        ref[:, 0] = DEFAULT_LEG_ANGLES[0] - lean_L[:,0] * scale + lean_R[:,0] * scale
        ref[:, 3] = DEFAULT_LEG_ANGLES[3] + lean_L[:,0] * scale - lean_R[:,0] * scale

        # Thigh: forward swing during swing phase
        ref[:, 1] = DEFAULT_LEG_ANGLES[1] + left_swing[:,0] * scale * 0.5
        ref[:, 4] = DEFAULT_LEG_ANGLES[4] + right_swing[:,0] * scale * 0.5

        # Calf: knee bend for lift (×5 stronger than thigh, critical for clearance)
        ref[:, 2] = DEFAULT_LEG_ANGLES[2] - left_lift[:,0] * scale * 5
        ref[:, 5] = DEFAULT_LEG_ANGLES[5] - right_lift[:,0] * scale * 5

        self._ref_dof_pos = ref
        # swing_mask: 1.0 when either leg is in swing, 0 otherwise
        self._swing_mask = np.clip(left_swing[:,0] + right_swing[:,0], 0.0, 1.0)

    # ── 状态更新 ──────────────────────────────────────────────────

    def update_state(self, state: NpEnvState) -> NpEnvState:
        """踮脚状态更新 — 先计算参考轨迹 + 轮子接触, 再调用父类"""
        self._compute_ref_dof_pos(state.info)
        state.info["ref_dof_pos"] = self._ref_dof_pos
        state.info["swing_mask"] = self._swing_mask
        self._update_wheel_contact(state.info)  # 需要 force sensor 判断轮子离地
        return super().update_state(state)

    def _update_wheel_contact(self, info: dict) -> None:
        """检测轮子触地 — 通过 force sensor (阈值 1.5N, 比跳跃的 0.1N 高)

        踮脚时轮子虽然不下压, 但仍有支撑力, 1.5N 可以区分"支撑"和"悬空"
        """
        try:
            lf = self._backend.get_sensor_data("left_wheel_site")
            rf = self._backend.get_sensor_data("right_wheel_site")
            lf_arr = np.asarray(lf, dtype=np.float64).reshape(self._num_envs, -1)
            rf_arr = np.asarray(rf, dtype=np.float64).reshape(self._num_envs, -1)
            l_contact = (np.linalg.norm(lf_arr, axis=1) > 1.5).astype(np.float64)
            r_contact = (np.linalg.norm(rf_arr, axis=1) > 1.5).astype(np.float64)
            info["wheel_contact"] = np.stack([l_contact, r_contact], axis=1)
        except Exception:
            info["wheel_contact"] = np.zeros((self._num_envs, 2), dtype=np.float64)

    # ── 观测 + 相位编码 ──────────────────────────────────────────

    def _compute_obs(self, info: dict, linvel, gyro, gravity, dof_pos, dof_vel) -> dict[str, np.ndarray]:
        """踮脚观测 — 标准帧 + sin/cos(phase) 编码

        obs 帧: 34 维 = 32(4D cmd base) + 2(phase)
        critic 帧: 37 维 = 35(4D cmd base + linvel) + 2(phase)
        """
        noise_cfg = self._cfg.noise_config
        leg_diff = dof_pos[:, :NUM_LEG_ACTIONS] - DEFAULT_LEG_ANGLES[:NUM_LEG_ACTIONS]
        leg_vel = dof_vel[:, :NUM_LEG_ACTIONS]
        wheel_vel = dof_vel[:, NUM_LEG_ACTIONS:]
        noisy_gyro = self._obs_noise(gyro, noise_cfg.scale_gyro)
        noisy_gravity = self._obs_noise(-gravity, noise_cfg.scale_gravity)
        noisy_leg_diff = self._obs_noise(leg_diff, noise_cfg.scale_joint_angle)
        noisy_leg_vel = self._obs_noise(leg_vel, noise_cfg.scale_joint_vel)
        noisy_wheel_vel = self._obs_noise(wheel_vel, noise_cfg.scale_wheel_vel)
        last_actions = info.get(
            "current_actions", np.zeros((linvel.shape[0], NUM_LEG_ACTIONS + NUM_WHEEL_ACTIONS))
        )

        obs_frame = np.concatenate(
            [
                noisy_gyro,
                noisy_gravity,
                noisy_leg_diff,
                noisy_leg_vel,
                noisy_wheel_vel,
                last_actions,
                info["commands"],
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
                linvel,
            ],
            axis=1,
            dtype=get_global_dtype(),
        )

        batch_size = obs_frame.shape[0]

        # 相位编码: sin/cos(2π·phase) → 2 维连续周期性信号
        steps_p = info.get("steps", np.zeros((batch_size,), dtype=np.float64))
        phase = (
            steps_p[:batch_size] * self._cfg.ctrl_dt / self._toe_cfg.cycle_time
        ) + self._phase_offset[:batch_size] / (2 * np.pi)
        sin_phase = np.sin(2 * np.pi * phase)[:, None]
        cos_phase = np.cos(2 * np.pi * phase)[:, None]

        obs_frame = np.concatenate(
            [obs_frame, sin_phase, cos_phase], axis=1, dtype=get_global_dtype()
        )
        critic_frame = np.concatenate(
            [critic_frame, sin_phase, cos_phase], axis=1, dtype=get_global_dtype()
        )

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

    # ── 终止条件 ──────────────────────────────────────────────────

    def _compute_terminated(self, gravity: np.ndarray, dof_pos: np.ndarray) -> np.ndarray:
        """踮脚终止 — 标准终止 + 腿碰地力检测

        腿碰地: 小腿或大腿接触地面 → 步态失败 → 直接终止
        阈值 0.1N (极低, 轻微蹭地即算失败)
        """
        tilt = np.arccos(np.clip(gravity[:, 2], -1, 1))
        max_tilt = np.deg2rad(self._toe_cfg.max_tilt_deg)
        terminated = tilt > max_tilt
        base_height = np.asarray(self._backend.get_base_pos(), dtype=get_global_dtype())[:, 2]
        terminated |= base_height < self._toe_cfg.min_base_height
        thigh_collapsed = (dof_pos[:, 1] < 0.02) | (dof_pos[:, 4] < 0.02)
        calf_extreme = (np.abs(dof_pos[:, 2]) > 0.85) | (np.abs(dof_pos[:, 5]) > 0.85)
        terminated |= thigh_collapsed
        terminated |= calf_extreme
        # ★ 腿碰地终止: 检查 contact_body_names 的力传感器
        for name in getattr(self._cfg, 'contact_body_names', []):
            try:
                cf = self._backend.get_sensor_data(name)
                if cf is not None:
                    c = np.asarray(cf, dtype=np.float64).reshape(self._num_envs, -1)
                    contact = np.any(np.abs(c) > 0.1, axis=1)
                    terminated |= contact
            except (KeyError, AttributeError):
                pass
        return terminated

    def _compute_reward(self, info: dict, linvel, gyro, gravity, dof_pos, dof_vel) -> np.ndarray:
        dtype = get_global_dtype()
        num_obs = linvel.shape[0]
        ctx = RewardContext(
            info=info,
            linvel=linvel,
            gyro=gyro,
            dof_pos=dof_pos[:, :NUM_LEG_ACTIONS],
            dof_vel=dof_vel,
            num_envs=num_obs,
            default_angles=DEFAULT_LEG_ANGLES[:NUM_LEG_ACTIONS].astype(dtype),
            tracking_sigma=self._toe_cfg.tracking_sigma,
            base_height_target=self._toe_cfg.base_height_target,
            base_height=self._base_height_values(num_obs),
            gravity=gravity,
            joint_range=None,
        )
        return rewards.run_reward_dispatch(
            scales=self._toe_cfg.scales,
            fns=self._reward_fns,
            ctx=ctx,
            info=info,
            enable_log=self._enable_reward_log,
            ctrl_dt=self._cfg.ctrl_dt,
            only_positive=self._toe_cfg.only_positive_rewards,
        )

    def _base_height_values(self, num_obs: int) -> np.ndarray:
        base_pos = np.asarray(self._backend.get_base_pos(), dtype=get_global_dtype())
        if base_pos.shape[0] != num_obs:
            return np.zeros((num_obs,), dtype=get_global_dtype())
        return np.asarray(base_pos[:, 2], dtype=get_global_dtype())
