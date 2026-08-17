"""xqrobotwl Virtual Model Control (VMC) layer.

The policy controls each leg through virtual-leg coordinates:
    theta0 : virtual leg angle from vertical (sagittal plane, positive = forward)
    L0     : virtual leg length (hip -> wheel centre)
plus a hip-roll position reference and a wheel velocity reference.  The VMC
layer converts these references into physical joint torques:

    torque_theta = kp_theta*(theta0_ref - theta0) - kd_theta*theta0_dot
    force_L0     = kp_l0*(L0_ref - L0) - kd_l0*L0_dot + feedforward
    torque_wheel = kp_wheel*(wheel_ref - wheel_vel) + integral     (PI)
    torque_roll  = kp_roll*(roll_ref - roll_pos) - kd_roll*roll_vel (PD)

then maps (force_L0, torque_theta) through the 2-link leg Jacobian to hip/knee
torques.

Kinematics convention (numerically calibrated -- see
``tools/xqrobotwl/calibrate_xqrobotwl_vmc.py``):

    theta1 = hip_sign * q_hip_pitch + c1
    theta2 = knee_sign * q_knee + c2
    end_x  = offset + l1*cos(theta1) + l2*cos(theta1 + theta2)
    end_y  = l1*sin(theta1) + l2*sin(theta1 + theta2)
    L0     = sqrt(end_x**2 + end_y**2)
    theta0 = atan2(end_y, end_x) - pi/2

xqrobotwl joint axes differ from the reference robots, so the per-leg sign
conventions (hip_sign/knee_sign) and phase offsets (c1/c2) are calibration
outputs, not assumed values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class XqRobotWLVMCConfig:
    """Virtual-leg kinematics + VMC control parameters (calibrated values)."""

    # ── Virtual-leg kinematics (calibrated, see tools/xqrobotwl/calibrate_xqrobotwl_vmc.py) ──
    l1: float = 0.30053  # thigh length (hip_pitch -> knee)
    l2: float = 0.30070  # calf length (knee -> wheel)
    offset: float = -0.00003  # hip horizontal offset
    theta0_offset: float = 0.0747  # virtual leg angle at the default posture
    l0_offset: float = 0.3669  # virtual leg length at the default posture
    l0_min: float = 0.2047  # achievable L0 lower bound (with margin)
    l0_max: float = 0.5231  # achievable L0 upper bound (with margin)
    singularity_epsilon: float = 0.05
    # per-leg sign/phase conventions (calibrated): index 0 = left, 1 = right
    hip_sign: list[float] = field(default_factory=lambda: [1.0, -1.0])
    knee_sign: list[float] = field(default_factory=lambda: [-1.0, 1.0])
    c1: float = 2.4103
    c2: float = -1.6790

    # ── VMC gains ──
    kp_theta: float = 60.0
    kd_theta: float = 3.0
    kp_l0: float = 800.0
    kd_l0: float = 5.0  # 低腿长阻尼: 爆发式蹬伸打破轮地接触 (原 20)
    feedforward_force: float = 110.0  # support force along L0 (per leg)
    kp_roll: float = 40.0
    kd_roll: float = 2.0
    kp_wheel: float = 2.0
    kp_wheel_integral: float = 1.0
    wheel_integral_limit_ratio: float = 0.5
    wheel_ctrlrange: float = 10.0  # used only to bound the wheel PI integral

    # ── Action scaling (policy output in [-1, 1]) ──
    action_scale_theta: float = 0.5
    action_scale_l0: float = 0.12
    action_scale_vel: float = 10.0
    action_scale_roll: float = 0.3
    roll_default: list[float] = field(default_factory=lambda: [0.1, -0.1])
    wheel_sign: list[float] = field(default_factory=lambda: [1.0, -1.0])
    clip_actions: float = 1.0

    # ── SLIP-FSM leg-length phase references (physical L0, metres) ──
    crouch_length: float = 0.28
    thrust_length: float = 0.50
    flight_retract_length: float = 0.26
    prelanding_length: float = 0.42
    landing_absorption_length: float = 0.30
    prelanding_start_vz: float = 0.0
    prelanding_full_vz: float = -0.6
    landing_compression_time: float = 0.18
    # ── FSM phase durations (s). VMC force control is slower than joint-space
    # position control, so it needs longer crouch/thrust windows to reach the
    # same crouch depth and push-off travel (else it enters thrust half-crouched
    # and barely leaves the ground). Longer than the joint-space SRL FSM.
    fsm_crouch_time: float = 0.35
    fsm_thrust_time: float = 0.30
    # ── Phase-dependent VMC gain scales (SLIP-FSM phases) ──
    thrust_kp_scale: float = 3.0  # 强蹬伸刚度
    thrust_ff_scale: float = 3.0  # 强蹬伸前馈
    thrust_kd_scale: float = 0.25  # 蹬伸低阻尼 (kd_l0×0.25=5): 爆发式伸展打破轮地接触
    flight_ff_scale: float = 0.40
    landing_kd_scale: float = 2.5
    landing_ff_scale: float = 1.20
    # ── 膝关节守卫参数 (vmc.py 方向性前馈削减 + 力矩硬守卫) ──
    # 蹬伸相前馈从膝伸展 >knee_guard_start 开始削减, >knee_guard_limit 归零。
    # 提高 start = 允许更完整的蹬伸力 (但膝过冲更多); 降低 = 更早削力护膝。
    knee_guard_start: float = 0.50
    knee_guard_limit: float = 0.85
    # ── 主动膝超伸刹车 (v8e3) ──
    # 爆炸式蹬伸把膝以 ~17 rad/s 甩向机械止位(±0.873), 动量冲过守卫 → 撞止位
    # (实测膝 -0.97)。电机力矩(±50N·m)无法在止位前刹停(需~160), 但从膝"过直"
    # (伸展量 ext>knee_brake_start) 就持续反制, 有 ~0.87 rad 行程, 可以刹住。
    # 只对"已过直且仍向极限伸展"的膝生效 — 不干扰蹬伸主行程 (屈曲→伸直那段)。
    knee_brake_start: float = 0.0  # 伸展量超过此值启动刹车 (0 = 过直即刹)
    knee_brake_kd: float = 0.0  # 刹车刚度 (v8e3 扫描验证: 无法刹住动量且毁跳高, 默认关闭)
    # ── PPO+VMC full-action reference blend ──
    # fb<1.0 keeps the SLIP-FSM reference dominant so the policy cannot cancel
    # the crouch->thrust timing (baseline fb=1.0 let it pre-extend during the
    # crouch phase and farm launch_rise with the wheels never leaving ground).
    feedback_gain: float = 1.0
    # Crouch-phase L0 upper bound (m): L0 above this during fsm_state==0 is
    # treated as premature extension and penalised by ``anti_early_extend``.
    crouch_upper_bound: float = 0.38


class VirtualLegVMC:
    """Batched numpy VMC control layer (hot path, no backend access)."""

    def __init__(self, cfg: XqRobotWLVMCConfig, num_envs: int, dtype: type = np.float64):
        self._cfg = cfg
        self._num_envs = num_envs
        self._dtype = dtype
        self._hip_sign = np.asarray(cfg.hip_sign, dtype=dtype)
        self._knee_sign = np.asarray(cfg.knee_sign, dtype=dtype)
        self._pi = math.pi
        # (num_envs, 2) [L, R] wheel-speed PI integral state
        self._wheel_integral = np.zeros((num_envs, 2), dtype=dtype)

    # ------------------------------------------------------------------ #
    # Kinematics                                                          #
    # ------------------------------------------------------------------ #

    def _theta_from_joints(self, dof_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map dof-order joint positions to the calibrated (theta1, theta2)."""
        q_hip = dof_pos[:, [1, 4]]  # [L_hip_pitch, R_hip_pitch]
        q_knee = dof_pos[:, [2, 5]]  # [L_knee, R_knee]
        theta1 = self._hip_sign * q_hip + self._cfg.c1
        theta2 = self._knee_sign * q_knee + self._cfg.c2
        return theta1, theta2

    def forward_kinematics(
        self, theta1: np.ndarray, theta2: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (L0, theta0) for the 2-link virtual leg."""
        cfg = self._cfg
        end_x = cfg.offset + cfg.l1 * np.cos(theta1) + cfg.l2 * np.cos(theta1 + theta2)
        end_y = cfg.l1 * np.sin(theta1) + cfg.l2 * np.sin(theta1 + theta2)
        L0 = np.sqrt(end_x**2 + end_y**2)
        theta0 = np.arctan2(end_y, end_x) - self._pi / 2.0
        return L0, theta0

    def compute_kinematics(
        self, dof_pos: np.ndarray, dof_vel: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (theta1, theta2, theta0, L0, theta0_dot, L0_dot).

        Derivatives use a forward-Euler finite difference at dt=1e-3, following
        the reference wheel-legged VMC implementation.
        """
        theta1, theta2 = self._theta_from_joints(dof_pos)
        L0, theta0 = self.forward_kinematics(theta1, theta2)
        q_hip_dot = dof_vel[:, [1, 4]]
        q_knee_dot = dof_vel[:, [2, 5]]
        theta1_dot = self._hip_sign * q_hip_dot
        theta2_dot = self._knee_sign * q_knee_dot
        dt = 1.0e-3
        L0_t, theta0_t = self.forward_kinematics(theta1 + theta1_dot * dt, theta2 + theta2_dot * dt)
        L0_dot = (L0_t - L0) / dt
        theta0_dot = (theta0_t - theta0) / dt
        return theta1, theta2, theta0, L0, theta0_dot, L0_dot

    def vmc_jacobian_mapping(
        self,
        theta1: np.ndarray,
        theta2: np.ndarray,
        L0: np.ndarray,
        theta0: np.ndarray,
        force: np.ndarray,
        torque: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map task-space (force_L0, torque_theta) to hip/knee torques (T1, T2)."""
        cfg = self._cfg
        theta0_vmc = theta0 + self._pi / 2.0
        safe_L0 = np.maximum(L0, cfg.singularity_epsilon)
        t11 = cfg.l1 * np.sin(theta0_vmc - theta1) - cfg.l2 * np.sin(theta1 + theta2 - theta0_vmc)
        t12 = (
            cfg.l1 * np.cos(theta0_vmc - theta1) + cfg.l2 * np.cos(theta1 + theta2 - theta0_vmc)
        ) / safe_L0
        t21 = -cfg.l2 * np.sin(theta1 + theta2 - theta0_vmc)
        t22 = cfg.l2 * np.cos(theta1 + theta2 - theta0_vmc) / safe_L0
        T1 = t11 * force + t12 * torque
        T2 = t21 * force + t22 * torque
        return T1, T2

    # ------------------------------------------------------------------ #
    # Control                                                             #
    # ------------------------------------------------------------------ #

    def get_l0_control_parameters(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the default per-env (kp_l0, kd_l0, feedforward_force) shaped (num_envs, 2)."""
        cfg = self._cfg
        kp = np.full((self._num_envs, 2), cfg.kp_l0, dtype=self._dtype)
        kd = np.full((self._num_envs, 2), cfg.kd_l0, dtype=self._dtype)
        ff = np.full((self._num_envs, 2), cfg.feedforward_force, dtype=self._dtype)
        return kp, kd, ff

    def reset_wheel_integral(self, env_ids: np.ndarray) -> None:
        if env_ids.size:
            self._wheel_integral[env_ids] = 0.0

    def compute_torques(
        self,
        policy_ctrl: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        sim_dt: float,
        l0_kp: np.ndarray | None = None,
        l0_kd: np.ndarray | None = None,
        feedforward: np.ndarray | None = None,
    ) -> np.ndarray:
        """Convert physical VMC references to 8 actuator-order joint torques.

        Args:
            policy_ctrl: (num_envs, 8) physical references in actuator order
                [roll_L, theta_L, L0_L, wheel_L, roll_R, theta_R, L0_R, wheel_R].
            dof_pos / dof_vel: (num_envs, 8) joint state in dof order
                [L_roll, L_pitch, L_knee, R_roll, R_pitch, R_knee, L_wheel, R_wheel].
            sim_dt: physics substep dt (wheel PI integrates at this rate).
            l0_kp / l0_kd / feedforward: optional per-env ``(num_envs, 2)``
                leg-length control parameters. ``None`` uses ``get_l0_control_parameters``.

        Returns:
            (num_envs, 8) joint torques in actuator order.
        """
        cfg = self._cfg
        theta1, theta2, theta0, L0, theta0_dot, L0_dot = self.compute_kinematics(dof_pos, dof_vel)

        theta0_ref = policy_ctrl[:, [1, 5]]  # [L, R]
        L0_ref = policy_ctrl[:, [2, 6]]
        wheel_ref = policy_ctrl[:, [3, 7]]
        roll_ref = policy_ctrl[:, [0, 4]]

        torque_theta = cfg.kp_theta * (theta0_ref - theta0) - cfg.kd_theta * theta0_dot
        if l0_kp is None or l0_kd is None or feedforward is None:
            kp_l0, kd_l0, ff = self.get_l0_control_parameters()
            l0_kp = kp_l0 if l0_kp is None else l0_kp
            l0_kd = kd_l0 if l0_kd is None else l0_kd
            feedforward = ff if feedforward is None else feedforward
        # ── 膝关节机械极限守卫: 削减蹬伸前馈 (reflex 式, 文献 MARCO Hopper II) ──
        # MuJoCo 关节限位准静态有效 (±0.873), 但高速蹬伸冲击会瞬态过冲至 ±1.0
        # (膝"砸"向机械止位, 实测 PPO+VMC/SRL/SRL+VMC)。根因是蹬伸相大前馈
        # (110N×thrust_ff_scale) 在膝接近伸直时仍持续推 L0。
        # 方向性判断: 只削"伸展方向"接近极限的前馈 (L 腿膝负向伸展, R 腿膝正向
        # 伸展)。深蹲屈膝 (|knee| 大但方向相反) **不削支撑前馈** — 否则深蹲
        # 时支撑前馈被削 → 蹲不住/蹬不起来 (v7 SRL+VMC 深蹲后不跳的根因之一)。
        # 用一步预测 (伸展位 + 伸展速×预测窗口) 提前判定, 避免位置守卫滞后。
        knee_pos = dof_pos[:, [2, 5]]
        knee_vel = dof_vel[:, [2, 5]]
        ext = np.stack([-knee_pos[:, 0], knee_pos[:, 1]], axis=1)  # L: 负向伸展, R: 正向
        ext_vel = np.stack([-knee_vel[:, 0], knee_vel[:, 1]], axis=1)
        ext_pred = ext + ext_vel * 0.02  # ~2 ctrl 步预测
        ff_guard_start = float(getattr(cfg, "knee_guard_start", 0.50))
        ff_guard_limit = float(getattr(cfg, "knee_guard_limit", 0.85))
        ff_scale = np.clip(
            (ff_guard_limit - ext_pred) / (ff_guard_limit - ff_guard_start), 0.0, 1.0
        )
        ff_scale = np.where(ext_pred >= ff_guard_limit, 0.0, ff_scale)
        feedforward = np.asarray(feedforward, dtype=self._dtype) * ff_scale

        force_L0 = l0_kp * (L0_ref - L0) - l0_kd * L0_dot + feedforward

        wheel_vel = dof_vel[:, [6, 7]]
        wheel_err = wheel_ref - wheel_vel
        self._wheel_integral += wheel_err * sim_dt * cfg.kp_wheel_integral
        integral_limit = cfg.wheel_ctrlrange * cfg.wheel_integral_limit_ratio
        np.clip(self._wheel_integral, -integral_limit, integral_limit, out=self._wheel_integral)
        torque_wheel = cfg.kp_wheel * wheel_err + self._wheel_integral

        roll_pos = dof_pos[:, [0, 3]]
        roll_vel = dof_vel[:, [0, 3]]
        torque_roll = cfg.kp_roll * (roll_ref - roll_pos) - cfg.kd_roll * roll_vel

        T1, T2 = self.vmc_jacobian_mapping(theta1, theta2, L0, theta0, force_L0, torque_theta)

        # Physical joint torques = dtheta/dq * virtual torques (per-leg sign).
        torques = np.empty_like(policy_ctrl)
        torques[:, 0] = torque_roll[:, 0]  # roll_L
        torques[:, 1] = self._hip_sign[0] * T1[:, 0]  # pitch_L
        torques[:, 2] = self._knee_sign[0] * T2[:, 0]  # knee_L
        torques[:, 3] = torque_wheel[:, 0]  # wheel_L
        torques[:, 4] = torque_roll[:, 1]  # roll_R
        torques[:, 5] = self._hip_sign[1] * T1[:, 1]  # pitch_R
        torques[:, 6] = self._knee_sign[1] * T2[:, 1]  # knee_R
        torques[:, 7] = torque_wheel[:, 1]  # wheel_R

        # ── 膝关节机械极限守卫 (|q_knee| ≤ 0.85, CLAUDE.md §1.3) ─────────────
        # 蹬伸相低阻尼 (thrust_kd_scale) + 大前馈会让膝力矩把膝关节推出极限
        # (实测 PPO+VMC/SRL/SRL+VMC 膝过伸至 ±1.0)。守卫在膝接近极限且力矩
        # 推向外时按接近程度削减, 越过极限硬置零。guard_start=0.55 需足够早:
        # 蹬伸是爆发式的 (一两个 ctrl 步内膝从 ~0 冲到 ~1.0), 太晚 (0.70) 时
        # 动量已把膝冲出极限。0.55 对应 L0≈0.48, 低于 thrust 目标 0.50 所需
        # 膝位 (~0.62), 所以只在蹬伸末段减速, 不破坏主要推力。
        knee_pos = dof_pos[:, [2, 5]]
        knee_tau = torques[:, [2, 6]]
        pushing_out = ((knee_pos > 0.0) & (knee_tau > 0.0)) | (
            (knee_pos < 0.0) & (knee_tau < 0.0)
        )
        guard_start, guard_limit = 0.55, 0.85
        abs_knee = np.abs(knee_pos)
        scale = np.clip((guard_limit - abs_knee) / (guard_limit - guard_start), 0.0, 1.0)
        scale = np.where(pushing_out, scale, 1.0)
        scale = np.where(abs_knee >= guard_limit, 0.0, scale)
        torques[:, 2] *= scale[:, 0]
        torques[:, 6] *= scale[:, 1]

        # ── 主动膝超伸刹车 (v8e3) ─────────────────────────────────────────
        # 前馈守卫只削"推"的力, 挡不住腿的伸展动量 (实测膝仍冲到 -0.97 撞止位)。
        # 这里在膝"过直"(伸展量>knee_brake_start) 且仍向极限伸展时, 施加反向
        # 力矩持续反制, 把膝刹在机械止位之前。刹车方向: L 腿伸展=负向膝速 →
        # 反向(+); R 腿伸展=正向膝速 → 反向(−)。
        brake_start = float(getattr(cfg, "knee_brake_start", 0.0))
        brake_kd = float(getattr(cfg, "knee_brake_kd", 8.0))
        ext = np.stack([-knee_pos[:, 0], knee_pos[:, 1]], axis=1)
        ext_vel = np.stack([-knee_vel[:, 0], knee_vel[:, 1]], axis=1)
        past = ext - brake_start
        ramp = np.clip(past / max(0.85 - brake_start, 1e-6), 0.0, 1.0)
        brake = brake_kd * np.maximum(ext_vel, 0.0) * np.where(past > 0, ramp, 0.0)
        torques[:, 2] += brake[:, 0]
        torques[:, 6] -= brake[:, 1]
        return torques
