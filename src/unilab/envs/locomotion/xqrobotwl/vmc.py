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
``scripts/calibrate_xqrobotwl_vmc.py``):

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

    # ── Virtual-leg kinematics (calibrated, see scripts/calibrate_xqrobotwl_vmc.py) ──
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
        return torques
