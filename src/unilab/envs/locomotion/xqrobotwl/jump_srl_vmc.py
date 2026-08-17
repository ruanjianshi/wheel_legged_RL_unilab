"""xqrobotwl SRL+VMC jump env -- 干净消融: 除输出层外与 SRL 完全一致.

v8 设计 (对比实验): SRL+VMC 应只在与 SRL 的**输出/控制层**不同, 奖励、
观测、FSM 参考结构全部与 SRL (jump_srl) 一致, 用于隔离"关节空间 PD vs
虚拟腿 VMC"控制层的影响。

  * **奖励**: 与 SRL 完全相同的奖励集 (jump_srl._reward_*), 去掉 v7 的
    VMC 专属塑形项 (jump_upright/lateral_posture/standing_still/下蹲门控)。
    若 VMC 在相同奖励下出现外展/站立振荡 → 归因于控制层, 而非奖励补偿。
  * **观测**: 与 SRL 相同的关节空间观测 (315D, joint obs + FSM features),
    不再含虚拟腿运动学 — 策略与 SRL 看到相同状态, 只输出不同 (虚拟腿参考)。
  * **输出/控制层 (唯一不同)**: 策略输出虚拟腿变量, VMC 映射为关节力矩;
    SLIP-FSM 参考以 L0 轨迹施加 (残差式 ``final_L0 = ref + fb*policy``),
    分阶段 VMC 增益 (thrust 高前馈/低阻尼, landing 高阻尼吸收)。

控制层实现 (继承自 jump_vmc): VMC 雅可比任务空间力→关节力矩 + 膝守卫 +
虚拟腿参考 (crouch 0.28 → thrust 0.50 → flight 收腿 → preland 0.42 →
landing 吸收)。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from unilab.base import registry
from unilab.dtype_config import get_global_dtype
from unilab.envs.locomotion.common import rewards

from . import jump_srl as _srl
from .base import NUM_LEG_ACTIONS
from .jump_vmc import XqRobotWLJumpVMCFlatCfg, XqRobotWLJumpVMCFlatEnv

WHEEL_R = 0.065  # wheel radius (m), matches jump_srl.WHEEL_R


@registry.envcfg("XqRobotWLJumpSRLVMC")
@dataclass
class XqRobotWLJumpSRLVMCFlatCfg(XqRobotWLJumpVMCFlatCfg):
    reward_config: _srl.XqRobotWLJumpRewardConfig | None = None  # type: ignore[assignment]
    # Recovery window (ctrl steps) after a real landing during which the
    # landing_recovery / landing_soft rewards are active.
    landing_window: int = 50


@registry.env("XqRobotWLJumpSRLVMC", sim_backend="mujoco")
class XqRobotWLJumpSRLVMCFlatEnv(XqRobotWLJumpVMCFlatEnv):
    """SRL + VMC jump -- SLIP-FSM reference applied as a policy residual."""

    _cfg: XqRobotWLJumpSRLVMCFlatCfg

    def __init__(self, cfg: XqRobotWLJumpSRLVMCFlatCfg, num_envs=1, backend_type="mujoco"):
        super().__init__(cfg, num_envs=num_envs, backend_type=backend_type)
        # Extend the real-landing recovery window so the post-impact recovery
        # rewards stay active long enough to actually stand back up.
        self._landing_window = int(getattr(cfg, "landing_window", 50))
        self._init_reward_functions()
        # v8: 观测对齐 SRL (关节空间 315D) — 只保留输出层(VMC)不同。
        # jump_vmc.__init__ 已把观测帧设为虚拟腿 41D, 这里改回 SRL 的关节 33D
        # 并重分配历史缓冲。
        self._obs_frame_dim = 33
        self._critic_frame_dim = 36
        self._obs_history = np.zeros(
            (num_envs, self._hist_len, self._obs_frame_dim), dtype=self._np_dtype
        )
        self._critic_history = np.zeros(
            (num_envs, self._hist_len, self._critic_frame_dim), dtype=self._np_dtype
        )
        # v8e2: "真跳事件"锁存 — 触发窗口内 base_z 首次超过阈值记一次真跳,
        # 触发 jump_event 一次性大奖励 (引导策略跨过小跳局部最优)。
        self._window_jumped = np.zeros(num_envs, dtype=np.float64)
        # 真跳阈值 (m): base_z 超过此值视为"真跳" (站立 ~0.55 + 0.17m)
        self._jump_event_threshold = float(getattr(cfg.reward_config, "jump_event_threshold", 0.72))

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        # 与 SRL 相同: 33×9 + 2×9 = 315 / 36×9 + 2×9 = 342
        fsm_extra = 2 * self._hist_len
        return {
            "obs": self._obs_frame_dim * self._hist_len + fsm_extra,
            "critic": self._critic_frame_dim * self._hist_len + fsm_extra,
        }

    # ------------------------------------------------------------------ #
    # SLIP-FSM leg-length reference (corrected phase mapping)             #
    # ------------------------------------------------------------------ #

    def _jump_leg_reference(self) -> np.ndarray:
        """Physical L0 reference per FSM phase (SRLVMC-corrected mapping).

        FSM states: -1 idle, 0 crouch, 1 thrust, 2 flight, 3 prelanding,
        4 landing-absorption.
        """
        cfg = self._vmc_cfg
        target = np.full(self._num_envs, cfg.l0_offset, dtype=self._np_dtype)
        phase = self._fsm_state
        linvel = self.get_local_linvel()
        vz = np.asarray(linvel[:, 2], dtype=self._np_dtype)

        crouch = phase == 0
        target[crouch] = cfg.crouch_length
        thrust = phase == 1
        target[thrust] = cfg.thrust_length
        flight = phase == 2
        if flight.any():
            retract = cfg.flight_retract_length
            denom = max(cfg.prelanding_start_vz - cfg.prelanding_full_vz, 1.0e-6)
            frac = np.clip((cfg.prelanding_start_vz - vz) / denom, 0.0, 1.0)
            target[flight] = retract + frac[flight] * (cfg.prelanding_length - retract)
        preland = phase == 3
        # Hold the leg extended and ready for contact (do NOT compress yet).
        target[preland] = cfg.prelanding_length
        landing = phase == 4
        if landing.any():
            # Actual touchdown/impact: compress prelanding -> absorption length
            # over landing_compression_time to absorb the kinetic energy.
            frac = np.clip(
                self._fsm_timer[landing] / max(cfg.landing_compression_time, 1.0e-6),
                0.0,
                1.0,
            )
            target[landing] = cfg.prelanding_length + frac * (
                cfg.landing_absorption_length - cfg.prelanding_length
            )
        return target

    def get_l0_control_parameters(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Start from the VMC layer's default gains, then scale by FSM phase.
        # Landing absorption gains (high damping + high feedforward) apply to
        # state 4 (actual impact), NOT to state 3 (prelanding).
        kp, kd, ff = self._vmc.get_l0_control_parameters()
        cfg = self._vmc_cfg
        phase = self._fsm_state
        crouch = (phase == 0)[:, None]
        thrust = (phase == 1)[:, None]
        flight = (phase == 2)[:, None]
        landing = (phase == 4)[:, None]
        preland = (phase == 3)[:, None]
        kp = np.where(thrust, kp * cfg.thrust_kp_scale, kp)
        # Thrust uses low leg damping (kd_l0 * thrust_kd_scale) so the extension
        # is explosive enough to break wheel contact -> real lift-off.
        kd = np.where(thrust, kd * cfg.thrust_kd_scale, kd)
        ff = np.where(thrust, ff * cfg.thrust_ff_scale, ff)
        ff = np.where(flight, ff * cfg.flight_ff_scale, ff)
        kd = np.where(landing, kd * cfg.landing_kd_scale, kd)
        ff = np.where(landing, ff * cfg.landing_ff_scale, ff)
        # Prelanding: mild extra damping so the extended leg does not flail in
        # free-fall while holding the prelanding length.
        kd = np.where(preland, kd * (cfg.landing_kd_scale * 0.6), kd)
        # v8e (缺陷1修复): 蹲下相 (phase 0) 支撑前馈归零 — 前馈恒 +110N 推腿伸,
        # 与压缩方向对抗, 使 VMC 蹲下费力。归零后腿自由压缩 (P 项 kp_l0 仍控制
        # L0 追踪), 更深下蹲 → 更大蹬伸行程 → 更高跳跃。
        ff = np.where(crouch, np.float64(0.0), ff)
        return kp, kd, ff

    # ------------------------------------------------------------------ #
    # Step / reward                                                        #
    # ------------------------------------------------------------------ #

    def step(self, actions):
        actions = np.asarray(actions, dtype=self._np_dtype).copy()
        target_action = self._jump_leg_reference_action()
        residual_scale = getattr(self._jump_cfg, "feedback_gain", 0.15)
        # Residual mode: the policy provides a small residual on top of the
        # SLIP-FSM reference (reference-project mode), so it cannot cancel the
        # proven crouch-thrust trajectory.
        actions[:, 2] = target_action + residual_scale * actions[:, 2]  # L0_L
        actions[:, 5] = target_action + residual_scale * actions[:, 5]  # L0_R
        return super().step(actions)

    def _compute_reward(self, info, linvel, gyro, gravity, dof_pos, dof_vel) -> np.ndarray:
        # Expose FSM state + wheel angular velocity to the reward functions.
        info["fsm_state"] = self._fsm_state.astype(np.float64)
        info["wheel_vel"] = np.asarray(dof_vel[:, NUM_LEG_ACTIONS:], dtype=np.float64)
        # v8e2: 真跳事件锁存 — base_z 触发窗口内首次超过阈值 → jump_event=1 (每窗口一次)。
        phase = info.get("jump_phase", np.zeros(self._num_envs, dtype=np.float64))
        window_active = (phase >= 1.0).astype(np.float64)
        idle = (window_active <= 0.0).astype(np.float64)
        base_z = np.asarray(self._backend.get_base_pos(), dtype=get_global_dtype())[:, 2][: self._num_envs]
        fresh = np.asarray(info.get("steps", np.zeros(self._num_envs)))[: self._num_envs] <= 1
        if fresh.any():
            self._window_jumped[fresh] = 0.0
        crossed = (base_z > self._jump_event_threshold).astype(np.float64)
        event = np.where(
            (crossed > 0) & (self._window_jumped <= 0) & (window_active > 0),
            np.float64(1.0),
            np.float64(0.0),
        )
        self._window_jumped = np.where(
            idle > 0, np.float64(0), np.maximum(self._window_jumped, crossed)
        )
        info["window_jumped"] = self._window_jumped.copy()
        info["jump_event"] = event
        return super()._compute_reward(info, linvel, gyro, gravity, dof_pos, dof_vel)

    def _compute_obs(self, info, linvel, gyro, gravity, dof_pos, dof_vel) -> dict[str, np.ndarray]:
        # v8: 与 SRL 相同的关节空间观测 (33D/frame) + FSM features → 315D。
        # 跳过 jump_vmc 的虚拟腿观测 (41D), 用 walk 基类的关节观测 — 策略看到
        # 与 SRL 相同的状态, 只有输出层 (虚拟腿参考) 不同。
        base = super(XqRobotWLJumpVMCFlatEnv, self)._compute_obs(
            info, linvel, gyro, gravity, dof_pos, dof_vel
        )
        batch = linvel.shape[0]
        fsm_feat = self._fsm_state.astype(np.float64).reshape(-1, 1)[:batch] / 5.0
        timer_feat = np.clip(self._fsm_timer.reshape(-1, 1)[:batch] / 0.8, 0, 1)
        extra = np.tile(
            np.concatenate([fsm_feat, timer_feat], axis=1, dtype=get_global_dtype())[
                :, None, :
            ],
            (1, self._hist_len, 1),
        ).reshape(batch, -1)
        base["obs"] = np.concatenate([base["obs"], extra], axis=1)
        base["critic"] = np.concatenate([base["critic"], extra], axis=1)
        return base

    def _init_reward_functions(self) -> None:
        # v8: 与 SRL 完全相同的奖励集 (干净消融 — 只输出层不同)。
        # 去掉 v7 的 VMC 专属塑形 (jump_upright/lateral_posture/standing_still/
        # 下蹲门控), 补齐 SRL 有的 anti_drift/action_magnitude。
        self._reward_fns: dict[str, object] = {
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
            "lateral_posture": self._reward_lateral_posture,
            "jump_event": self._reward_jump_event,
            "jump_upright": self._reward_jump_upright,
            "jump_symmetry": self._reward_jump_symmetry,
        }

    def _reward_jump_upright(self, ctx):
        """腾空期保持躯干直立: 奖 up_z→1.

        v8e3 (参考: wheel_legged_lab `orientation -4.0 全程` + "pelvis roll/pitch
        最小化持续到腾空/落地"): v8e2 空中 up_z=0.911 / base_pitch=-0.21 —
        跳起来了但躯干歪斜, 正是缺"空中直立"显式奖励。本项 **air-gated**
        (轮离地才有效): 只在腾空把躯干立直, 不干扰地面蹬伸期的下蹲/前倾
        (否则会压低跳高)。
        """
        assert ctx.gravity is not None
        tilt = np.arccos(np.clip(ctx.gravity[:, 2], -1, 1))
        # v8e4: σ 0.12→0.20 — 高斯更平缓, 只罚明显倾斜, 不把高跳的轻微前倾
        # 判成失分 (v8e3 的 0.12 让高跳空中倾斜 → upright 掉 5/步, 策略弃高跳)。
        upright = np.exp(-np.square(tilt) / 0.20)
        wheel_contact = ctx.info.get("wheel_contact", np.ones((ctx.num_envs, 2)))
        air = 1.0 - np.mean(wheel_contact, axis=1)
        active = (air > 0.3).astype(np.float64)
        phase = ctx.info.get("jump_phase", np.zeros(ctx.num_envs, dtype=np.float64))
        active *= (phase >= 1.0).astype(np.float64)
        weight = ctx.info.get("jump_curriculum", 1.0)
        return upright * active * weight

    def _reward_jump_symmetry(self, ctx):
        """跳跃窗口左右腿协调: 膝伸展量对称 + 髋角互补.

        v8e3 (参考: wheel_legged_lab `jump_symmetry +0.5 CROUCH-LANDING`):
        v8e2 空中膝 [-0.30, +0.36] 不对称 + 左右腿一前一后 — 蹬伸不同步导致
        姿态歪。用 |L_knee|−|R_knee| 差和 L/R 髋角和惩罚, 高斯奖对称。
        """
        knee_dev = np.abs(np.abs(ctx.dof_pos[:, 2]) - np.abs(ctx.dof_pos[:, 5]))
        pitch_sum = np.abs(ctx.dof_pos[:, 1] + ctx.dof_pos[:, 4])
        sym = np.exp(-(knee_dev + pitch_sum) / 0.3)
        phase = ctx.info.get("jump_phase", np.zeros(ctx.num_envs, dtype=np.float64))
        active = (phase >= 1.0).astype(np.float64)
        weight = ctx.info.get("jump_curriculum", 1.0)
        return sym * active * weight

    def _reward_jump_event(self, ctx):
        """真跳事件奖励: 触发窗口内 base_z 首次超过 0.72 → +1 (每窗口一次).

        v8e2: 一次性大奖励, 给策略一个明确的"做一次真跳"信号 — 引导它跨过
        "小跳刷分"局部最优 (v8e 的硬门槛让策略卡在 0.145m 学不过去, 没有梯度)。
        """
        event = ctx.info.get("jump_event", np.zeros(ctx.num_envs, dtype=np.float64))
        weight = ctx.info.get("jump_curriculum", 1.0)
        return event * weight

    def _reward_lateral_posture(self, ctx):
        """跳跃期髋不外展: 罚 |roll−default| + 左右不对称 + roll速度 (跳跃窗口全程).

        v8d 加回 (实用版): 干净消融 (v8b) 证实 VMC 力控固有叉腿倾向 (髋外展 0.627,
        靠外展借力起跳 — 类似摆臂助跳)。本项在整个跳跃窗口把外展轴拉回默认角
        (L +0.1 / R −0.1), 配合加强蹬伸让 VMC 靠干净 L0 力控起跳。
        v8e (缺陷3修复): 增加 roll 速度惩罚 — 按步罚的 dev/asym 压不住起跳瞬间
        1-2 步的瞬时外展脉冲, 加 roll 速度项直接压"起跳借力的外展动作"。
        """
        roll_L = ctx.dof_pos[:, 0]
        roll_R = ctx.dof_pos[:, 3]
        dev = np.abs(roll_L - 0.1) + np.abs(roll_R + 0.1)
        asym = np.square(roll_L + roll_R)
        rollv = np.abs(ctx.dof_vel[:, 0]) + np.abs(ctx.dof_vel[:, 3])
        phase = ctx.info.get("jump_phase", np.zeros(ctx.num_envs, dtype=np.float64))
        active = (phase >= 1.0).astype(np.float64)
        weight = ctx.info.get("jump_curriculum", 1.0)
        return -(dev + asym + rollv * 0.5) * active * weight

    # ── SRL reward wrappers (全部指向 jump_srl, 与 SRL 完全一致) ───────

    def _reward_jump_height(self, ctx):
        r = _srl._reward_jump_height(ctx, self._jump_cfg)
        # v8e (缺陷2修复) + v8e2 (门槛下调): 最低跳高门槛 base_z > 目标+0.10
        # (0.65, 即 ~0.10m 跳) — 灭掉 stutter (0.05m) 刷分, 同时留梯度让策略
        # 能爬升 (v8e 的 0.70 硬门槛让策略卡在 0.145m 学不过去)。真正的"跨过大跳"
        # 由 jump_event 一次性奖励引导。
        min_jump_z = ctx.base_height_target + 0.10
        real = (ctx.base_height > min_jump_z).astype(np.float64)
        return r * real

    def _reward_crouch_prep(self, ctx):
        return _srl._reward_crouch_prep(ctx, self._jump_cfg)

    def _reward_landing_soft(self, ctx):
        return _srl._reward_landing_soft(ctx)

    def _reward_wheel_air_time(self, ctx):
        return _srl._reward_wheel_air_time(ctx)

    def _reward_wheel_ground_matching(self, ctx):
        return _srl._reward_wheel_ground_matching(ctx)

    def _reward_landing_recovery(self, ctx):
        return _srl._reward_landing_recovery(ctx)

    def _reward_anti_drift(self, ctx):
        return _srl._reward_anti_drift(ctx)

    def _reward_action_magnitude(self, ctx):
        return _srl._reward_action_magnitude(ctx)

    def _reward_vertical_thrust(self, ctx):
        r = _srl._reward_vertical_thrust(ctx, self._jump_cfg)
        # v8e5: 只在 FSM 蹬伸相及之后 (state>=1) 奖励向上 vz — 消灭"下蹲相提前升起"。
        # 实测 final10000 轨迹: 下蹲相 z 0.33→0.62 (提前升) → 回落 0.52 → 才真正
        # 起跳 — 双重运动, 跳跃"不干净"。根因是 vertical_thrust (phase>=12 即奖)
        # 让策略在下蹲相 (FSM 仍 0, L0_ref=0.24 蹲姿) 顶腿白拿奖励。改为 FSM 蹬伸
        # 相才奖, 策略学会蹲稳再蹬 (单次干净运动)。
        fsm = ctx.info.get("fsm_state", -np.ones(ctx.num_envs, dtype=np.float64))
        thrusting = (fsm >= 1.0).astype(np.float64)
        return r * thrusting

    def _reward_crouch_depth(self, ctx):
        return _srl._reward_crouch_depth(ctx, self._jump_cfg)

    def _reward_lean_forward(self, ctx):
        return _srl._reward_lean_forward(ctx)

    def _reward_height_progress(self, ctx):
        r = _srl._reward_height_progress(ctx)
        # v8e5: 同 vertical_thrust — FSM 蹬伸相及之后才奖高度进展, 压下蹲相提前升。
        fsm = ctx.info.get("fsm_state", -np.ones(ctx.num_envs, dtype=np.float64))
        thrusting = (fsm >= 1.0).astype(np.float64)
        return r * thrusting
