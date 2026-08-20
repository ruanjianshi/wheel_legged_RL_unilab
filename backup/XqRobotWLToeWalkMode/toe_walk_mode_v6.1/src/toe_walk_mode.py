"""xqrobotwl toe-walk MODE env: dual-mode (stand ⇄ toe-lift) with command-channel mode switch.

需求 (老板 2026-08-18):
  默认站立姿态 (mode=0) → 键盘按键切换成点足抬腿模式 (mode=1) →
  指令追踪 (前进/后退/侧向/转向) → 按键切回站立 (mode=0)。

设计 (参考 docs/references/2026-08-18_mode_switching_multi_skill.md):
  - commands 4D → 5D: [vx, vy, vyaw, tsk, mode], mode ∈ {0,1} 作为命令通道进观测
  - 奖励按 mode 门控:
      mode=0 站立: 共项 (tracking/高度/直立/平滑) + stand_still 微动平衡, 关抬腿项
      mode=1 抬腿: 共项 + 相位门控抬腿套件 (沿用 07-28 配方)
      新增 lift_symmetry: 摆动相轮离地滑动均值 L/R 差惩罚 (防 8-18 验证发现的单侧塌缩)
  - 课程 (mode_curriculum_iters=[2000,5000]): Stage1 强制 mode=0 → Stage2 强制 mode=1 →
    Stage3 随机切换混合 (mode 每 resampling_time 重采样, 覆盖任意切换时刻)
  - 相位时钟保持连续不重置, 切换无硬跳变
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from unilab.base import registry
from unilab.dtype_config import get_global_dtype
from unilab.envs.locomotion.common import rewards
from unilab.envs.locomotion.common.rewards import RewardContext

from .base import DEFAULT_LEG_ANGLES, NUM_LEG_ACTIONS
from .joystick import XqRobotWLCurriculumConfig
from .toe_walk import (
    XqRobotToeWalkCommands,
    XqRobotToeWalkDRProvider,
    XqRobotToeWalkRewardConfig,
    XqRobotWLToeWalkFlatCfg,
    XqRobotWLToeWalkFlatEnv,
)

# 抬腿模式专用奖励键 (mode=0 时权重清零)
_LIFT_ONLY_REWARD_KEYS = (
    "phase_knee_lift",
    "phase_knee_stance",
    "phase_stance_penalty",
    "window_lift",
    "window_penalty",
)


@dataclass
class XqRobotToeWalkModeCommands(XqRobotToeWalkCommands):
    """5D commands: [vx, vy, vyaw, tsk, mode]  (mode ∈ {0站立, 1点足抬腿})."""

    vel_limit: list[list[float]] = field(
        default_factory=lambda: [[-0.3, -0.1, -0.3, -0.1, 0.0], [0.3, 0.1, 0.3, 0.1, 1.0]]
    )
    resampling_time: float = 6.0


@dataclass
class XqRobotToeWalkModeRewardConfig(XqRobotToeWalkRewardConfig):
    """双模式奖励配置: 在原抬腿配方上增加 站立项 / 对称项 / 模式课程."""

    mode_curriculum_iters: list[int] = field(default_factory=lambda: [2000, 5000])
    symmetry_decay: float = 0.98  # lift_symmetry 滑动均值衰减


def _reward_lift_symmetry(ctx: RewardContext) -> np.ndarray:
    """摆动相轮离地滑动均值 L/R 差惩罚 (防单侧塌缩, 周期级比较, 非逐时刻).

    逐时刻 |L_air - R_air| 在正常反相步态下恒=1 (同一时刻只有一侧抬),
    所以必须用滑动均值累计后比较 (EMA 由 env._compute_reward 维护).
    """
    ema = ctx.info.get("lift_sym_ema")
    if ema is None:
        return np.zeros((ctx.num_envs,), dtype=np.float64)
    return -np.abs(ema[:, 0] - ema[:, 1])


def _mode_column(ctx: RewardContext) -> np.ndarray:
    """从命令取 mode 列 (5D commands 末位 0/1; 兼容 <5D = 全 0)."""
    commands = np.asarray(ctx.info.get("commands", np.zeros((ctx.num_envs, 1))))
    if commands.shape[1] >= 5:
        return commands[:, 4].astype(get_global_dtype())
    return np.zeros((ctx.num_envs,), dtype=get_global_dtype())


def _reward_mode_window_lift(ctx: RewardContext) -> np.ndarray:
    """窗级结算奖励 (v5): 摆动窗结束时, 若该窗曾离地且上一窗也离地 (交替链), 给窗内离地占比×权重.

    读 env._update_lift_windows 结算出的 info["lift_win_award"] (结算步一次性脉冲).
    """
    return np.asarray(ctx.info.get("lift_win_award", np.zeros((ctx.num_envs,))), dtype=np.float64)


def _reward_mode_window_penalty(ctx: RewardContext) -> np.ndarray:
    """窗级结算惩罚 (v5): 摆动窗结束仍未离地 → -权重 (每次结算一次).

    读 info["lift_win_penalty"].
    """
    return -np.asarray(ctx.info.get("lift_win_penalty", np.zeros((ctx.num_envs,))), dtype=np.float64)


def _masked_by_mode(fn, *, invert: bool = False):
    """奖励函数包装: 输出按 env 的 mode 掩码 (0=站立, 1=抬腿)."""

    def wrapped(ctx: RewardContext) -> np.ndarray:
        out = np.asarray(fn(ctx), dtype=get_global_dtype())
        mask = _mode_column(ctx)
        if invert:
            mask = 1.0 - mask
        return out * mask

    return wrapped


@registry.envcfg("XqRobotWLToeWalkMode")
@dataclass
class XqRobotWLToeWalkModeCfg(XqRobotWLToeWalkFlatCfg):
    commands: XqRobotToeWalkModeCommands = field(default_factory=XqRobotToeWalkModeCommands)
    reward_config: XqRobotToeWalkModeRewardConfig | None = None  # type: ignore[assignment]
    curriculum: XqRobotWLCurriculumConfig = field(
        default_factory=lambda: XqRobotWLCurriculumConfig(enabled=False)
    )
    max_episode_seconds: float = 12.0


class XqRobotToeWalkModeDRProvider(XqRobotToeWalkDRProvider):
    """reset 时的命令采样: 5D 均匀采样前 4 维 + mode 离散 0/1 (课程阶段强制)."""

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
        cmds[:, 4] = env._sample_mode_values(num_reset)
        return cmds


@registry.env("XqRobotWLToeWalkMode", sim_backend="mujoco")
class XqRobotWLToeWalkModeEnv(XqRobotWLToeWalkFlatEnv):
    """双模式点足行走: 站立 ⇄ 点足抬腿, mode 命令通道切换."""

    _cfg: XqRobotWLToeWalkModeCfg
    _toe_cfg: XqRobotToeWalkModeRewardConfig  # type: ignore[assignment]

    def __init__(self, cfg: XqRobotWLToeWalkModeCfg, num_envs=1, backend_type="mujoco"):
        if cfg.reward_config is None:
            raise ValueError("reward_config must be provided via Hydra configuration")
        self._toe_cfg = cfg.reward_config
        # 基类 __init__ 会以 34/37 帧维分配历史缓冲 → 在其后用 35/38 重分配
        super().__init__(cfg, num_envs=num_envs, backend_type=backend_type)
        self._dr_manager._provider = XqRobotToeWalkModeDRProvider()  # type: ignore[union-attr]
        # obs 维度: 命令 4D→5D (mode 通道), 其余不变
        self._obs_frame_dim = 35  # 32(base 4D→5D) + 2 = 35
        self._critic_frame_dim = 38  # 35 + linvel(3) = 38
        self._obs_history = np.zeros(
            (num_envs, self._hist_len, self._obs_frame_dim), dtype=self._np_dtype
        )
        self._critic_history = np.zeros(
            (num_envs, self._hist_len, self._critic_frame_dim), dtype=self._np_dtype
        )
        # 抬腿对称性滑动均值 (num_envs, 2) [L, R]
        self._air_ema = np.zeros((num_envs, 2), dtype=get_global_dtype())
        # 终止: 异常帧延迟计数 (v3, 防单帧误杀)
        self._bad_frames = np.zeros((num_envs,), dtype=np.int64)
        # 窗级交替考核状态 (v5): 每 env 记录摆动窗状态
        self._win_cur = np.full((num_envs,), -1, dtype=np.int64)  # -1=none, 0=L摆动窗, 1=R摆动窗
        self._win_lifted = np.zeros((num_envs,), dtype=bool)  # 本窗是否曾离地
        self._win_air_steps = np.zeros((num_envs,), dtype=np.float64)  # 本窗离地步数
        self._win_steps = np.zeros((num_envs,), dtype=np.float64)  # 本窗步数
        self._win_last_ok = np.ones((num_envs,), dtype=bool)  # 上一结算窗是否离地 (初始视为交替链完整)

    # ── 模式课程阶段 ─────────────────────────────────────────

    def _mode_stage(self) -> int:
        """课程阶段: 0=纯站立, 1=纯抬腿, 2=自由切换. 评估 (num_envs<=1) 恒为 2."""
        if self._num_envs <= 1:
            return 2
        iters = [int(x) for x in getattr(self._toe_cfg, "mode_curriculum_iters", [2000, 5000])]
        steps_per_iter = 24.0 * float(self._num_envs)  # num_steps_per_env=24 (该任务固定)
        cur_iter = self._total_env_steps / steps_per_iter
        if cur_iter < iters[0]:
            return 0
        if cur_iter < iters[1]:
            return 1
        return 2

    def _sample_mode_values(self, num_samples: int) -> np.ndarray:
        """按课程阶段产出 mode 列 (0/1)."""
        stage = self._mode_stage()
        dtype = get_global_dtype()
        if stage == 0:
            return np.zeros((num_samples,), dtype=dtype)
        if stage == 1:
            return np.ones((num_samples,), dtype=dtype)
        return np.random.randint(0, 2, size=(num_samples,)).astype(dtype)

    # ── 命令重采样 (episode 中途, resampling_time) ──────────

    def _update_commands(self, info: dict) -> None:
        """5D 命令中途重采样: 前 4 维均匀 + mode 离散 (课程阶段决定)."""
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
                sampled = np.random.uniform(
                    low=low, high=high, size=(num_resample, low.shape[0])
                ).astype(get_global_dtype())
                safe_linv = np.maximum(np.abs(sampled[:, 0]), 1e-4)
                angv_limit = 2.0 / safe_linv
                sampled[:, 2] = np.clip(sampled[:, 2], -angv_limit, angv_limit)
                sampled[:, 4] = self._sample_mode_values(num_resample)
                commands_arr[resample_mask] = sampled
        info["commands"] = commands_arr

    # ── 奖励函数注册 ─────────────────────────────────────────

    def _init_reward_functions(self) -> None:
        super()._init_reward_functions()
        # 窗级交替考核奖励 (v5): 取代逐时 phase_swing_lift / swing_contact_penalty
        self._reward_fns["window_lift"] = _reward_mode_window_lift
        self._reward_fns["window_penalty"] = _reward_mode_window_penalty
        # mode 门控: 抬腿项 ×mode, 站立项 ×(1-mode), 逐 env 掩码 (scale 保持标量)
        for key in _LIFT_ONLY_REWARD_KEYS:
            if key in self._reward_fns:
                self._reward_fns[key] = _masked_by_mode(self._reward_fns[key], invert=False)
        self._reward_fns["stand_still"] = _masked_by_mode(rewards.stand_still, invert=True)
        self._reward_fns["lift_symmetry"] = _masked_by_mode(_reward_lift_symmetry, invert=False)
        # v5: phase_knee_lift 必须当前摆动窗已离地才计 (消灭"深弯膝刷分"捷径)
        base_knee = self._reward_fns.get("phase_knee_lift")
        if base_knee is not None:

            def _knee_gated(ctx: RewardContext) -> np.ndarray:
                out = np.asarray(base_knee(ctx), dtype=get_global_dtype())
                gate = np.asarray(
                    ctx.info.get("win_lifted_cur", np.zeros((ctx.num_envs,))),
                    dtype=get_global_dtype(),
                )
                return out * gate

            self._reward_fns["phase_knee_lift"] = _knee_gated

    # ── 窗级交替考核状态机 (v5) ────────────────────────────
    #   摆动窗 (相位 sin>0.2 → L, sin<-0.2 → R) 结束时结算:
    #     - 本窗曾离地 且 上一结算窗也离地 (交替链完整) → 奖 window_lift × 窗内离地占比
    #     - 本窗离地但链断 → 半奖 (恢复机制)
    #     - 本窗从未离地 → 罚 window_penalty (一次)
    #   效果: 双侧交替每周期 +60, 单侧抬每周期 30-150=-120 (亏损), 不抬 -300 → 双侧唯一有利解

    def _update_lift_windows(self, info: dict) -> None:
        n = self._num_envs
        award = np.zeros((n,), dtype=get_global_dtype())
        penalty = np.zeros((n,), dtype=get_global_dtype())
        phase = np.asarray(info.get("phase", np.zeros((n, 1), dtype=get_global_dtype())))[:, 0]
        sin_p = np.sin(2 * np.pi * phase)
        win = np.where(sin_p > 0.2, 0, np.where(sin_p < -0.2, 1, -1)).astype(np.int64)
        contact = info.get("wheel_contact", np.zeros((n, 2), dtype=get_global_dtype()))

        changed = win != self._win_cur
        # 1) 结算刚结束的窗口
        settle = changed & (self._win_cur >= 0)
        if np.any(settle):
            prev_lifted = self._win_lifted[settle]
            frac = np.where(
                self._win_steps[settle] > 0,
                np.clip(self._win_air_steps[settle] / self._win_steps[settle], 0.0, 1.0),
                0.0,
            )
            ok_prev = self._win_last_ok[settle]
            full = prev_lifted & ok_prev
            recover = prev_lifted & ~ok_prev
            a = np.zeros(np.count_nonzero(settle), dtype=get_global_dtype())
            a[full] = frac[full]
            a[recover] = frac[recover] * 0.5
            p = np.zeros(np.count_nonzero(settle), dtype=get_global_dtype())
            p[~prev_lifted] = 1.0
            award[settle] = a
            penalty[settle] = p
            self._win_last_ok[settle] = prev_lifted
        # 2) 新窗口初始化
        start = changed & (win >= 0)
        if np.any(start):
            self._win_lifted[start] = False
            self._win_air_steps[start] = 0.0
            self._win_steps[start] = 0.0
        self._win_cur[:] = win
        # 3) 窗内累计离地占比 + 当前窗 lifted 标志
        active = win >= 0
        if np.any(active):
            left_air = (1.0 - contact[:, 0]) * (win == 0)
            right_air = (1.0 - contact[:, 1]) * (win == 1)
            self._win_steps[active] += 1.0
            self._win_air_steps[active] += (left_air + right_air)[active]
            self._win_lifted[active] |= (left_air + right_air)[active] > 0.5
        info["lift_win_award"] = award
        info["lift_win_penalty"] = penalty
        info["win_lifted_cur"] = self._win_lifted.astype(get_global_dtype())

    # ── 奖励计算: mode 门控 + 对称 EMA + 窗级考核 + 课程 ────────

    def _compute_reward(self, info: dict, linvel, gyro, gravity, dof_pos, dof_vel) -> np.ndarray:
        dtype = get_global_dtype()
        num_obs = linvel.shape[0]

        if not self._toe_cfg.use_reference and getattr(
            self._toe_cfg, "mode_curriculum_iters", []
        ):
            self._total_env_steps += self._num_envs

        self._update_lift_windows(info)  # v5: 窗级结算状态机 (先于 mode 门控, award/penalty 供奖励函数)

        # 有效 mode (课程阶段强制时与命令采样一致, 供对称 EMA 门控)
        mode_eff = np.asarray(info["commands"][:, 4], dtype=dtype).copy()
        stage = self._mode_stage()
        if stage == 0:
            mode_eff[:] = 0.0
        elif stage == 1:
            mode_eff[:] = 1.0

        # 抬腿对称性 EMA (仅抬腿模式更新; 站立期冻结)
        decay = float(getattr(self._toe_cfg, "symmetry_decay", 0.98))
        if np.max(mode_eff) > 0.5:
            contact = info.get("wheel_contact", np.zeros((num_obs, 2), dtype=dtype))
            phase = info.get("phase", np.zeros((num_obs, 1), dtype=dtype))
            sin_p = np.sin(2 * np.pi * phase)[:, 0] if phase.ndim == 2 else np.zeros(num_obs)
            swL = (sin_p > 0.2).astype(dtype)
            swR = (sin_p < -0.2).astype(dtype)
            airL = (1.0 - contact[:, 0]) * swL * mode_eff
            airR = (1.0 - contact[:, 1]) * swR * mode_eff
            self._air_ema[:, 0] = decay * self._air_ema[:, 0] + (1.0 - decay) * airL
            self._air_ema[:, 1] = decay * self._air_ema[:, 1] + (1.0 - decay) * airR
        info["lift_sym_ema"] = self._air_ema

        # v3: Stage2 (纯抬腿) 前 800 iter 追踪斜坡 — 先学会"抬腿不倒地"再叠加追踪
        scales = dict(self._toe_cfg.scales)
        if stage == 1:
            steps_per_iter = 24.0 * float(self._num_envs)
            iters = [int(x) for x in getattr(self._toe_cfg, "mode_curriculum_iters", [2000, 5000])]
            ramp = 800
            alpha = max(0.0, min(1.0, (self._total_env_steps / steps_per_iter - iters[0]) / ramp))
            scales["tracking_lin_vel"] = scales.get("tracking_lin_vel", 0.0) * alpha
            scales["tracking_ang_vel"] = scales.get("tracking_ang_vel", 0.0) * alpha

        ctx = RewardContext(
            info=info,
            linvel=linvel,
            gyro=gyro,
            dof_pos=dof_pos[:, :6],
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
            scales=scales,
            fns=self._reward_fns,
            ctx=ctx,
            info=info,
            enable_log=self._enable_reward_log,
            ctrl_dt=self._cfg.ctrl_dt,
            only_positive=self._toe_cfg.only_positive_rewards,
        )

    # ── 模式切换状态连续性: 相位时钟不重置 (继承基类行为) ────
    #   toe_walk 基类 _compute_ref_dof_pos 按 steps 连续推进 phase,
    #   模式切换只经命令通道, 无硬跳变。对称 EMA 在站立期冻结由 lift_mask 天然实现。

    # ── 终止判定 (v4): 按 mode 区分严格度 ──────────────
    #   8-18 诊断 (model_4000, mode=1): 抬腿探索动作撞 thigh<-0.5 单帧被误杀。
    #   v3 全局延迟 5 帧修复了抬腿, 但让站立模式也"歪着不死" (gyro 2.86/高 0.61)。
    #   v4: 异常帧延迟按模式区分 — mode=0 站立立即终止 (苛刻), mode=1 抬腿延迟 5 帧 (宽容探索)
    _BAD_FRAMES = 5  # 抬腿模式异常持续帧数 (50ms)

    def _compute_terminated(self, gravity: np.ndarray, dof_pos: np.ndarray) -> np.ndarray:
        tilt = np.arccos(np.clip(gravity[:, 2], -1, 1))
        max_tilt = np.deg2rad(self._toe_cfg.max_tilt_deg)
        bad = tilt > max_tilt
        base_height = np.asarray(self._backend.get_base_pos(), dtype=get_global_dtype())[:, 2]
        bad |= base_height < self._toe_cfg.min_base_height
        thighL, kneeL, kneeR = dof_pos[:, 1], dof_pos[:, 2], dof_pos[:, 5]
        bad |= (np.abs(thighL) > 0.9) | (np.abs(dof_pos[:, 4]) > 0.9)  # ±0.9 (物理极限±1.0留缓冲)
        bad |= (np.abs(kneeL) > 2.0) | (np.abs(kneeR) > 2.0)
        for name in getattr(self._cfg, "contact_body_names", []):
            try:
                cf = self._backend.get_sensor_data(name)
                if cf is not None:
                    c = np.asarray(cf, dtype=get_global_dtype()).reshape(self._num_envs, -1)
                    bad |= np.any(np.abs(c) > 8.0, axis=1)
            except (KeyError, AttributeError):
                pass
        # 按 mode 决定延迟: 站立 0 帧立即终止, 抬腿 5 帧
        mode = np.asarray(self._state_info_commands_mode(), dtype=get_global_dtype())
        delay = np.where(mode > 0.5, self._BAD_FRAMES, 1)
        self._bad_frames = np.where(bad, self._bad_frames + 1, 0)
        return self._bad_frames >= delay

    def _state_info_commands_mode(self) -> np.ndarray:
        info = getattr(self, "_state", None)
        info = getattr(info, "info", None)
        if isinstance(info, dict) and info.get("commands") is not None:
            c = np.asarray(info["commands"], dtype=get_global_dtype())
            if c.shape[1] >= 5:
                return c[:, 4]
        return np.zeros((self._num_envs,), dtype=get_global_dtype())
